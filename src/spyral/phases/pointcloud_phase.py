from ..core.phase import PhaseLike
from ..core.schema import ResultSchema, PhaseResult
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
from ..core.spy_log import spyral_warn, spyral_info
from ..trace.trace_reader import create_reader
from ..trace.frib_event import TriggerType
from ..trace.peak import Peak
from ..core.pad_map import PadMap
from ..core.point_cloud import (
    point_cloud_from_get,
    calibrate_point_cloud_z,
    sort_point_cloud_in_z,
)
from .schema import TRACE_SCHEMA, POINTCLOUD_SCHEMA

import numpy as np
import h5py as h5

from pathlib import Path
from multiprocessing import SimpleQueue


class PointcloudPhase(PhaseLike):
    """The point cloud phase, inheriting from PhaseLike

    The goal of the point cloud phase is to convert AT-TPC trace data
    into point clouds. It uses a combination of Fourier transform baseline removal
    and scipy.signal.find_peaks to extract signals from the traces. PointcloudPhase
    is expected to be the first phase in the Pipeline.

    Parameters
    ----------
    get_params: GetParameters
        Parameters controlling the GET-DAQ signal analysis
    frib_params: FribParameters
        Parameters controlling the FRIBDAQ signal analysis
    detector_params: DetectorParameters
        Parameters describing the detector
    pad_params: PadParameters
        Parameters describing the pad plane mapping

    Attributes
    ----------
    get_params: GetParameters
        Parameters controlling the GET-DAQ signal analysis
    frib_params: FribParameters
        Parameters controlling the FRIBDAQ signal analysis
    det_params: DetectorParameters
        Parameters describing the detector
    pad_map: PadMap
        Map which converts trace ID to pad ID

    """

    def __init__(
        self,
        get_params: GetParameters,
        frib_params: FribParameters,
        detector_params: DetectorParameters,
        pad_params: PadParameters,
    ):
        super().__init__(
            "Pointcloud",
            incoming_schema=ResultSchema(TRACE_SCHEMA),
            outgoing_schema=ResultSchema(POINTCLOUD_SCHEMA),
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
                garf_path,
                self.electron_correction_path,
                self.det_params,
            )
        return True

    def construct_artifact(
        self, payload: PhaseResult, workspace_path: Path
    ) -> PhaseResult:
        result = PhaseResult(
            artifacts={
                "pointcloud": self.get_artifact_path(workspace_path)
                / f"{form_run_string(payload.run_number)}.h5"
            },
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
        # Copy phase_pointcloud.py here
        # Check that the traces exist
        trace_path = payload.artifacts["trace"]
        if not trace_path.exists():
            spyral_warn(
                __name__,
                f"Run {payload.run_number} does not exist for PointcloudPhase, skipping.",
            )
            return PhaseResult.invalid_result(payload.run_number)

        # Open files
        result = self.construct_artifact(payload, workspace_path)
        trace_reader = create_reader(trace_path, payload.run_number)
        if trace_reader is None:
            return PhaseResult.invalid_result(payload.run_number)
        point_file = h5.File(result.artifacts["pointcloud"], "w")

        # Load electric field correction
        corrector: ElectronCorrector | None = None
        if self.det_params.do_garfield_correction:
            corrector = create_electron_corrector(self.electron_correction_path)

        cloud_group = point_file.create_group("cloud")
        cloud_group.attrs["min_event"] = trace_reader.first_event()
        cloud_group.attrs["max_event"] = trace_reader.last_event()

        nevents = len(trace_reader)
        total: int
        flush_val: int
        if nevents < 100:
            total = nevents
            flush_val = 0
        else:
            flush_percent = 0.01
            flush_val = int(flush_percent * (nevents - 1))
            total = 100

        count = 0

        msg = StatusMessage(
            self.name, 1, total, payload.run_number
        )  # We always increment by 1

        ic_within_mult_count = 0

        # Process the data
        for idx in trace_reader.event_range():
            count += 1
            if count > flush_val:
                count = 0
                msg_queue.put(msg)

            event = trace_reader.read_event(idx, self.get_params, self.frib_params, rng)
            ic_mult = -1.0
            ic_peak: None | Peak = None
            if (
                event.frib is not None
                and event.frib.trigger == TriggerType.IC_DOWNSCALE_TRIGGER
            ):
                ic_mult = event.frib.get_ic_multiplicity(self.frib_params)
                ic_peak = event.frib.get_triggering_ic_peak(self.frib_params)
                if (
                    ic_mult > 0
                    and ic_mult <= self.frib_params.ic_multiplicity
                    and ic_peak is not None
                ):  # ugh
                    ic_within_mult_count += 1
                continue
            if event.get is None:
                continue

            # Convert traces to pointcloud
            cloud = point_cloud_from_get(event.get, self.pad_map)
            # Calibrate the time to z-position
            calibrate_point_cloud_z(cloud, self.det_params, corrector)
            # Sort the cloud in z
            sort_point_cloud_in_z(cloud)

            pc_dataset = cloud_group.create_dataset(
                f"cloud_{cloud.event_number}", cloud.data
            )
            # Store original run and event info
            pc_dataset.attrs["orig_run"] = event.original_run
            pc_dataset.attrs["orig_event"] = event.original_event
            # default IC settings
            pc_dataset.attrs["ic_amplitude"] = -1.0
            pc_dataset.attrs["ic_integral"] = -1.0
            pc_dataset.attrs["ic_centroid"] = -1.0
            pc_dataset.attrs["ic_multiplicity"] = -1.0

            if event.frib is not None:
                ic_peak = event.frib.get_triggering_ic_peak(self.frib_params)
                ic_mult = event.frib.get_ic_multiplicity(self.frib_params)
                # Check multiplicity condition and existence of trigger
                if (
                    ic_mult > 0.0 and ic_mult <= self.frib_params.ic_multiplicity
                ) and ic_peak is not None:
                    pc_dataset.attrs["ic_amplitude"] = ic_peak.amplitude
                    pc_dataset.attrs["ic_integral"] = ic_peak.integral
                    pc_dataset.attrs["ic_centroid"] = ic_peak.centroid
                    pc_dataset.attrs["ic_multiplicity"] = ic_mult

        # End of event data

        # Process scaler data if it exists
        scalers = trace_reader.read_scalers()
        if scalers is not None:
            scalers.write_scalers(
                self.get_artifact_path(workspace_path)
                / f"{form_run_string(payload.run_number)}_scaler.parquet"
            )
        else:
            spyral_info(__name__, f"Run {payload.run_number} does not have scaler data")

        gated_ic_path = (
            self.get_artifact_path(workspace_path)
            / f"{form_run_string(payload.run_number)}_gated_ic_scaler.txt"
        )
        with open(gated_ic_path, "w") as gated_ic_file:
            gated_ic_file.write(
                f"IC counts with multiplicity <= {self.frib_params.ic_multiplicity}\n"
            )
            gated_ic_file.write(f"{ic_within_mult_count}")

        spyral_info(__name__, f"Phase Pointcloud complete for run {payload.run_number}")
        return result
