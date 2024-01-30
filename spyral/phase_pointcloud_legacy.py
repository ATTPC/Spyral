from .core.config import GetParameters, DetectorParameters, FribParameters
from .core.pad_map import PadMap
from .core.point_cloud import PointCloud
from .core.workspace import Workspace
from .trace.frib_event import FribEvent
from .trace.get_event import GetEvent
from .correction import create_electron_corrector, ElectronCorrector
from .parallel.status_message import StatusMessage, Phase
from .core.spy_log import spyral_info, spyral_error, spyral_warn

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
    meta_data = meta_group.get("meta")
    return (int(meta_data[0]), int(meta_data[2]))


def phase_pointcloud_legacy(
    run: int,
    ws: Workspace,
    pad_map: PadMap,
    get_params: GetParameters,
    detector_params: DetectorParameters,
    queue: SimpleQueue,
):
    """The core loop of the pointcloud phase

    Generate point clouds from merged AT-TPC traces. Read in traces from a hdf5 file
    generated by the AT-TPC merger and convert the traces into point cloud events. This is
    the first phase of Spyral analysis.

    Parameters
    ----------
    run: int
        The run number to be processed
    ws: Workspace
        The project workspace
    pad_map: PadMap
        A map of pad number to geometry/hardware/calibrations
    get_params: GetParameters
        Configuration parameters for GET data signal analysis (AT-TPC pads)
    detector_params: DetectorParameters
        Configuration parameters for physical detector properties
    queue: SimpleQueue
        Communication channel back to the parent process
    """

    # Check that the traces exist
    trace_path = ws.get_trace_file_path(run)
    if not trace_path.exists():
        spyral_warn(__name__, f"Run {run} does not exist for phase 1, skipping.")
        return

    # Open files
    point_path = ws.get_point_cloud_file_path(run)
    trace_file = h5.File(trace_path, "r")
    point_file = h5.File(point_path, "w")

    min_event, max_event = get_event_range(trace_file)

    # Load electric field correction
    corrector: ElectronCorrector | None = None
    if detector_params.do_garfield_correction:
        corr_path = ws.get_correction_file_path(
            Path(detector_params.garfield_file_path)
        )
        corrector = create_electron_corrector(corr_path)

    # Some checks for existance
    event_group = trace_file["get"]
    if not isinstance(event_group, h5.Group):
        spyral_error(
            __name__,
            f"GET event group does not exist in run {run}, phase 1 cannot be run!",
        )
        return

    cloud_group = point_file.create_group("cloud")
    cloud_group.attrs["min_event"] = min_event
    cloud_group.attrs["max_event"] = max_event

    flush_percent = 0.01
    flush_val = int(flush_percent * (max_event - min_event))
    count = 0

    # Process the data
    for idx in range(min_event, max_event + 1):
        if count > flush_val:
            count = 0
            queue.put(StatusMessage(run, Phase.CLOUD, 1))
        count += 1

        event_data: h5.Dataset
        try:
            event_data = event_group[f"evt{idx}_data"]
        except Exception:
            continue

        event = GetEvent(event_data, idx, get_params, is_legacy=True)

        pc = PointCloud()
        pc.load_cloud_from_get_event(event, pad_map, corrector)

        pc_dataset = cloud_group.create_dataset(
            f"cloud_{pc.event_number}", shape=pc.cloud.shape, dtype=np.float64
        )

        # default IC settings
        pc_dataset.attrs["ic_amplitude"] = -1.0
        pc_dataset.attrs["ic_integral"] = -1.0
        pc_dataset.attrs["ic_centroid"] = -1.0

        pc_dataset[:] = pc.cloud

    spyral_info(__name__, "Phase 1 complete")
