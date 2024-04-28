from ..core.phase import PhaseLike, PhaseResult
from ..core.run_stacks import form_run_string
from ..core.status_message import StatusMessage
from ..core.config import (
    FribParameters,
    GetParameters,
    DetectorParameters,
    PadParameters,
)
from ..correction import (
    generate_electron_correction,
    create_electron_corrector,
    ElectronCorrector,
)
from ..core.spy_log import spyral_warn, spyral_error, spyral_info
from ..core.pad_map import PadMap
from ..trace.get_legacy_event import GetLegacyEvent
from ..core.point_cloud import PointCloud
from .schema import TRACE_SCHEMA, POINTCLOUD_SCHEMA

import h5py as h5
import numpy as np

from pathlib import Path
from multiprocessing import SimpleQueue


def get_event_range(trace_file: h5.File) -> tuple[int, int]:
    """
    The merger doesn't use attributes for legacy reasons, so everything is stored in datasets. Use this to retrieve the min and max event numbers.

    Parameters
    ----------
    trace_file: h5py.File
        File handle to a hdf5 file with AT-TPC traces

    Returns
    -------
    tuple[int, int]
        A pair of integers (first event number, last event number)
    """
    meta_group = trace_file.get("meta")
    meta_data = meta_group.get("meta")  # type: ignore
    return (int(meta_data[0]), int(meta_data[2]))  # type: ignore


class PointcloudLegacyPhase(PhaseLike):

    def __init__(
        self,
        get_params: GetParameters,
        frib_params: FribParameters,
        detector_params: DetectorParameters,
        pad_params: PadParameters,
    ):
        super().__init__(
            "PointcloudLegacy",
            incoming_schema=TRACE_SCHEMA,
            outgoing_schema=POINTCLOUD_SCHEMA,
        )
        self.get_params = get_params
        self.frib_params = frib_params
        self.det_params = detector_params
        self.pad_map = PadMap(pad_params)

    def create_assets(self, workspace_path: Path) -> bool:
        asset_path = self.get_asset_storage_path(workspace_path)
        garf_path = Path(self.det_params.garfield_file_path)
        self.electron_correction_path = asset_path / f"{garf_path.stem}.npy"

        if (
            not self.electron_correction_path.exists()
            and self.det_params.do_garfield_correction
        ):
            generate_electron_correction(
                self.electron_correction_path,
                garf_path,
                self.det_params,
            )
        return True

    def run(
        self,
        payload: PhaseResult,
        workspace_path: Path,
        msg_queue: SimpleQueue,
        rng: np.random.Generator,
    ) -> PhaseResult:
        trace_path = payload.artifact_path
        if not trace_path.exists():
            spyral_warn(
                __name__,
                f"Run {payload.run_number} does not exist for phase 1, skipping.",
            )
            return PhaseResult(Path("null"), True, payload.run_number)

        # Open files
        point_path = (
            self.get_artifact_path(workspace_path)
            / f"{form_run_string(payload.run_number)}.h5"
        )
        trace_file = h5.File(trace_path, "r")
        point_file = h5.File(point_path, "w")

        min_event, max_event = get_event_range(trace_file)

        # Load electric field correction
        corrector: ElectronCorrector | None = None
        if self.det_params.do_garfield_correction:
            corrector = create_electron_corrector(self.electron_correction_path)

        # Some checks for existance
        event_group = trace_file["get"]
        if not isinstance(event_group, h5.Group):
            spyral_error(
                __name__,
                f"GET event group does not exist in run {payload.run_number}, phase 1 cannot be run!",
            )
            return PhaseResult(Path("null"), True, payload.run_number)

        cloud_group = point_file.create_group("cloud")
        cloud_group.attrs["min_event"] = min_event
        cloud_group.attrs["max_event"] = max_event

        nevents = max_event - min_event
        total: int
        flush_val: int
        if nevents < 100:
            total = nevents
            flush_val = 0
        else:
            flush_percent = 0.01
            flush_val = int(flush_percent * (max_event - min_event))
            total = 100

        count = 0

        msg = StatusMessage(self.name, 1, total, 1)  # We always increment by 1

        # Process the data
        for idx in range(min_event, max_event + 1):
            count += 1
            if count > flush_val:
                count = 0
                msg_queue.put(msg)

            event_data: h5.Dataset
            try:
                event_data = event_group[f"evt{idx}_data"]  # type: ignore
            except Exception:
                continue

            event = GetLegacyEvent(
                event_data, idx, self.get_params, self.frib_params, rng
            )

            pc = PointCloud()
            pc.load_cloud_from_get_event(event, self.pad_map)
            pc.calibrate_z_position(
                self.det_params.micromegas_time_bucket,
                self.det_params.window_time_bucket,
                self.det_params.detector_length,
                corrector,
            )

            pc_dataset = cloud_group.create_dataset(
                f"cloud_{pc.event_number}", shape=pc.cloud.shape, dtype=np.float64
            )

            # default IC settings
            pc_dataset.attrs["ic_amplitude"] = -1.0
            pc_dataset.attrs["ic_integral"] = -1.0
            pc_dataset.attrs["ic_centroid"] = -1.0
            pc_dataset.attrs["ic_multiplicity"] = -1.0

            # Set IC if present; take first non-garbage peak
            if event.ic_trace is not None:
                # No way to disentangle multiplicity
                for peak in event.ic_trace.get_peaks():
                    pc_dataset.attrs["ic_amplitude"] = peak.amplitude
                    pc_dataset.attrs["ic_integral"] = peak.integral
                    pc_dataset.attrs["ic_centroid"] = peak.centroid
                    pc_dataset.attrs["ic_multiplicity"] = (
                        event.ic_trace.get_number_of_peaks()
                    )
                    break

            pc_dataset[:] = pc.cloud

        spyral_info(__name__, "Phase 1 complete")
        return PhaseResult(point_path, True, payload.run_number)
