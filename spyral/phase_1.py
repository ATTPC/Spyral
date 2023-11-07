from .core.config import TraceParameters, CrossTalkParameters, DetectorParameters, FribParameters
from .core.pad_map import PadMap
from .core.point_cloud import PointCloud
from .core.workspace import Workspace
from .trace.frib_event import FribEvent
from .trace.get_event import GetEvent
from .correction import create_electron_corrector
from .parallel.status_message import StatusMessage, Phase

from h5py import File, Group, Dataset
import numpy as np
from multiprocessing import SimpleQueue

def get_event_range(trace_file: File) -> tuple[int, int]:
    '''
    The old merger didn't seem to use attributes, so everything was stored in datasets. Use this to retrieve the min and max event numbers.

    ## Parameters
    trace_file: h5py.File, file handle to a file with traces

    ## Returns
    tuple[int, int]: a pair of integers (min_event, max_event)
    '''
    meta_group = trace_file.get('meta')
    meta_data = meta_group.get('meta')
    return (int(meta_data[0]), int(meta_data[2]))

def phase_1(run: int, ws: Workspace, pad_map: PadMap, trace_params: TraceParameters, frib_params: FribParameters, cross_params: CrossTalkParameters, detector_params: DetectorParameters, queue: SimpleQueue):
    trace_path = ws.get_trace_file_path(run)
    if not trace_path.exists():
        return
    
    point_path = ws.get_point_cloud_file_path(run)
    trace_file = File(trace_path, 'r')
    point_file = File(point_path, 'w')

    min_event, max_event = get_event_range(trace_file)

    corr_path = ws.get_correction_file_path(detector_params.efield_correction_name)
    corrector = create_electron_corrector(corr_path)

    event_group: Group = trace_file.get('get')
    frib_group: Group = trace_file.get('frib')
    frib_evt_group: Group = frib_group.get('evt')
    cloud_group: Group = point_file.create_group('cloud')
    cloud_group.attrs['min_event'] = min_event
    cloud_group.attrs['max_event'] = max_event

    flush_percent = 0.01
    flush_val = int(flush_percent * (max_event - min_event))
    count = 0

    for idx in range(min_event, max_event+1):

        if count > flush_val:
            count = 0
            queue.put(StatusMessage(run, Phase.CLOUD, 1))
        count += 1

        event_data: Dataset
        try:
            event_data = event_group[f'evt{idx}_data']
        except Exception:
            continue

        event = GetEvent(event_data, idx, trace_params)
        
        pc = PointCloud()
        pc.load_cloud_from_get_event(event, pad_map, corrector)
        # pc.eliminate_cross_talk(pad_map, cross_params)
        
        pc_dataset = cloud_group.create_dataset(f'cloud_{pc.event_number}', shape=pc.cloud.shape, dtype=np.float64)

        #default IC settings
        pc_dataset.attrs['ic_amplitude'] = -1.0
        pc_dataset.attrs['ic_integral'] = -1.0
        pc_dataset.attrs['ic_centroid'] = -1.0

        # Now analyze FRIBDAQ data
        frib_data: Dataset
        try:
            frib_data = frib_evt_group[f'evt{idx}_1903']
        except Exception:
            pc.calibrate_z_position(detector_params.micromegas_time_bucket, detector_params.window_time_bucket, detector_params.detector_length)
            pc_dataset[:] = pc.cloud
            continue

        frib_event = FribEvent(frib_data, idx, frib_params)

        ic_peak = frib_event.get_good_ic_peak(frib_params)
        if ic_peak is None:
            pc.calibrate_z_position(detector_params.micromegas_time_bucket, detector_params.window_time_bucket, detector_params.detector_length)
            pc_dataset[:] = pc.cloud
            continue
        pc_dataset.attrs['ic_amplitude'] = ic_peak.amplitude
        pc_dataset.attrs['ic_integral'] = ic_peak.integral
        pc_dataset.attrs['ic_centroid'] = ic_peak.centroid

        if frib_params.correct_ic_time:
            ic_cor = frib_event.correct_ic_time(ic_peak, detector_params.get_frequency)
            pc.calibrate_z_position(detector_params.micromegas_time_bucket, detector_params.window_time_bucket, detector_params.detector_length, ic_cor)
        else:
            pc.calibrate_z_position(detector_params.micromegas_time_bucket, detector_params.window_time_bucket, detector_params.detector_length)

        pc_dataset[:] = pc.cloud