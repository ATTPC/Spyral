from .core.config import TraceParameters, CrossTalkParameters, DetectorParameters
from .core.get_event import GetEvent
from .core.pad_map import PadMap
from .core.point_cloud import PointCloud
from .core.workspace import Workspace
from h5py import File, Group, Dataset
from time import time

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

def phase_1(run: int, ws: Workspace, pad_map: PadMap, trace_params: TraceParameters, cross_params: CrossTalkParameters, detector_params: DetectorParameters):
    start = time()
    trace_path = ws.get_trace_file_path(run)
    point_path = ws.get_point_cloud_file_path(run)
    trace_file = File(trace_path, 'r')
    point_file = File(point_path, 'w')

    min_event, max_event = get_event_range(trace_file)

    print(f'Running phase 1 on file {trace_path} for events {min_event} to {max_event}')

    event_group: Group = trace_file.get('get')
    cloud_group: Group = point_file.create_group('cloud')
    cloud_group.attrs['min_event'] = min_event
    cloud_group.attrs['max_event'] = max_event

    flush_percent = 0.01
    flush_val = int(flush_percent * (max_event - min_event))
    flush_count = 0
    count = 0

    for idx in range(min_event, max_event+1):

        if count > flush_val:
            count = 0
            flush_count += 1
            print(f'\rPercent of data processed: {int(flush_count * flush_percent * 100)}%', end='')
        count += 1

        event_data: Dataset | None = None
        try:
            event_data = event_group[f'evt{idx}_data']
        except Exception:
            continue

        event = GetEvent(event_data, idx, trace_params)
        
        pc = PointCloud()
        pc.load_cloud_from_get_event(event, pad_map)
        pc.eliminate_cross_talk(pad_map, cross_params)
        pc.calibrate_z_position(detector_params.micromegas_time_bucket, detector_params.window_time_bucket, detector_params.detector_length)
        
        cloud_group.create_dataset(f'cloud_{pc.event_number}', data=pc.cloud)

    stop = time()

    print(f'\nEllapsed time {stop-start}s')