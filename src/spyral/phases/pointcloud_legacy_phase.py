from ..core.phase import PhaseLike, PhaseResult
from ..core.run_stacks import form_run_string
from ..core.status_message import StatusMessage
from ..core.config import (
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
    """The legacy point cloud phase, inheriting from PhaseLike

    The goal of the legacy point cloud phase is to convert legacy (pre-FRIBDAQ) AT-TPC
    trace data into point clouds. It uses a combination of Fourier transform baseline
    removal and scipy.signal.find_peaks to extract signals from the traces. PointcloudLegacyPhase
    is expected to be the first phase in the Pipeline.

    Parameters
    ----------
    get_params: GetParameters
        Parameters controlling the GET-DAQ signal analysis
    ic_params: GetParameters
        Parameters in legacy to analyze auxilary detectors (IC, Si, etc)
    detector_params: DetectorParameters
        Parameters describing the detector
    pad_params: PadParameters
        Parameters describing the pad plane mapping

    Attributes
    ----------
    get_params: GetParameters
        Parameters controlling the GET-DAQ signal analysis
    ic_params: GetParameters
        Parameters in legacy to analyze auxilary detectors (IC, Si, etc)
    det_params: DetectorParameters
        Parameters describing the detector
    pad_map: PadMap
        Map which converts trace ID to pad ID

    """

    def __init__(
        self,
        get_params: GetParameters,
        ic_params: GetParameters,
        detector_params: DetectorParameters,
        pad_params: PadParameters,
    ):
        super().__init__(
            "PointcloudLegacy",
            incoming_schema=TRACE_SCHEMA,
            outgoing_schema=POINTCLOUD_SCHEMA,
        )
        self.get_params = get_params
        self.ic_params = ic_params
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
                garf_path,
                self.electron_correction_path,
                self.det_params,
            )
        return True

    def construct_artifact(
        self, payload: PhaseResult, workspace_path: Path
    ) -> PhaseResult:
        result = PhaseResult(
            artifact_path=self.get_artifact_path(workspace_path)
            / f"{form_run_string(payload.run_number)}.h5",
            successful=True,
            run_number=payload.run_number,
        )
        return result

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
            return PhaseResult.invalid_result(payload.run_number)

        # Open files
        result = self.construct_artifact(payload, workspace_path)
        trace_file = h5.File(trace_path, "r")
        point_file = h5.File(result.artifact_path, "w")

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
            return PhaseResult.invalid_result(payload.run_number)

        cloud_group = point_file.create_group("cloud")
        cloud_group.attrs["min_event"] = min_event
        cloud_group.attrs["max_event"] = max_event

        nevents = max_event - min_event + 1
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
            event_name = f"evt{idx}_data"
            if event_name not in event_group:
                continue
            else:
                event_data = event_group[event_name]  # type: ignore

            event = GetLegacyEvent(
                event_data, idx, self.get_params, self.ic_params, rng
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
        return result
