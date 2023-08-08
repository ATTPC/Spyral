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


def phase_1(config: Config):
    start = time()
    min_event, max_event = get_event_range(config.trace_data_path)

    pad_map = PadMap(config.pad_geometry_path, config.pad_gain_path, config.pad_time_path)

    for idx in range(min_event, max_event+1, 1):
        print(f'\rProcessing event {idx}')
        event = read_get_event(config.trace_data_path, idx)
        pc = PointCloud()
        pc.load_cloud_from_get_event(event, pad_map, config.signal_peak_separation, config.singal_peak_threshold)
        pc.eliminate_cross_talk(config.cross_talk_saturation_threshold, config.cross_talk_peak_threshold, config.cross_talk_search_range, config.cross_talk_time_range)
        pc.calibrate_z_position(config.micromegas_time_bucket, config.window_time_bucket, config.detector_length)

    stop = time()

    print(f'\nEllapsed time {stop-start}s\n')