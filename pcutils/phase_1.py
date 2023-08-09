from .core.config import Config
from .core.get_event import read_get_event
from .core.pad_map import PadMap
from .core.point_cloud import PointCloud
from pathlib import Path
from h5py import File
from time import time

def get_event_range(filepath: Path) -> tuple[int, int]:
    with File(filepath, 'r') as datafile:
        meta_group = datafile.get('meta')
        meta_data = meta_group.get('meta')
        return (int(meta_data[0]), int(meta_data[2]))

def write_point_cloud(filepath: Path, cloud: PointCloud):
    with File(filepath, 'a') as cloudfile:
        cloud_group = cloudfile.get('cloud')
        if cloud_group is None:
            cloud_group = cloudfile.create_group('cloud')
        cloud_group.create_dataset(f'cloud_{cloud.event_number}', data=cloud.cloud)

def phase_1(config: Config):
    start = time()
    min_event, max_event = get_event_range(config.workspace.trace_data_path)

    pad_map = PadMap(config.workspace.pad_geometry_path, config.workspace.pad_gain_path, config.workspace.pad_time_path, config.workspace.pad_electronics_path)

    print(f'Running phase 1 on file {config.workspace.trace_data_path} for events {min_event} to {max_event}')

    flush_percent = 0.01
    flush_val = int(flush_percent * (max_event - min_event))
    flush_count = 0
    count = 0

    for idx in range(min_event, max_event+1, 1):

        if count > flush_val:
            count = 0
            flush_count += 1
            print(f'\rPercent of data processed: {int(flush_count * flush_percent * 100)}%', end='')

        event = read_get_event(config, idx)
        pc = PointCloud()
        pc.load_cloud_from_get_event(event, pad_map)
        pc.eliminate_cross_talk(pad_map, config.cross)
        pc.calibrate_z_position(config.detector.micromegas_time_bucket, config.detector.window_time_bucket, config.detector.detector_length)
        write_point_cloud(config.workspace.point_cloud_data_path, pc)
        count += 1

    stop = time()

    print(f'\nEllapsed time {stop-start}s')