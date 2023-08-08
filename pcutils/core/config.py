from dataclasses import dataclass
from pathlib import Path
from json import load
from typing import Optional, Any

@dataclass
class Config:
    #General paths
    trace_data_path: str = ''
    point_cloud_data_path: str = ''
    ntuple_data_path: str = ''
    pad_geometry_path: str = ''
    pad_gain_path: str = ''
    pad_time_path: str = ''

    #Detector settings
    magnetic_field: float = 0.0 #Tesla
    electric_field: float = 0.0 #V/m
    tilt_angle: float = 0.0 #degrees
    detector_length: float = 0.0 #mm
    beam_region_radius: float = 0.0 #mm
    micromegas_time_bucket: float = 0.0
    window_time_bucket: float = 0.0

    #Gas settings
    density: float = 0.0 #g/cm^3
    energy_loss_path: str = ''

    #Trace evaluation settings
    singal_peak_threshold = 0.0 #amplitude
    signal_baseline_window_scale = 0.0
    signal_peak_separation = 0.0
    signal_smoothing_factor = 0.0 #scale

    #Cross talk settings
    cross_talk_saturation_threshold = 0.0
    cross_talk_peak_threshold = 0.0
    cross_talk_search_range = 0.0
    cross_talk_time_range =  0.0

    #Clustering settings
    cluster_smoothing_neighbor_distance = 0.0 #mm
    cluster_min_points = 0.0
    cluster_max_neighbor_distance_fractional = 0.0

def json_load_config_hook(json_data: dict[Any, Any]) -> Config:
    config = Config()
    config.trace_data_path = json_data['trace_data_path']
    config.point_cloud_data_path = json_data['point_cloud_data_path']
    config.ntuple_data_path = json_data['ntuple_data_path']
    config.pad_geometry_path = json_data['pad_geometry_path']
    config.pad_gain_path = json_data['pad_gain_path']
    config.pad_time_path = json_data['pad_time_path']

    config.magnetic_field = json_data['magnetic_field(T)']
    config.electric_field = json_data['electric_field(V/m)']
    config.tilt_angle = json_data['tilt_angle(degrees)']
    config.detector_length = json_data['detector_length(mm)']
    config.beam_region_radius = json_data['beam_region_radius(mm)']
    config.micromegas_time_bucket = json_data['micromegas_time_bucket']
    config.window_time_bucket = json_data['window_time_bucket']

    config.density = json_data['gas_density(g/cm^3)']
    config.energy_loss_path = json_data['gas_energy_loss_path']

    config.singal_peak_threshold = json_data['trace_peak_threshold']
    config.signal_baseline_window_scale = json_data['trace_baseline_window_scale']
    config.signal_peak_separation = json_data['trace_peak_separation']
    config.signal_smoothing_factor = json_data['trace_smoothing_factor']

    config.cross_talk_saturation_threshold = json_data['cross_talk_saturation_threshold']
    config.cross_talk_peak_threshold = json_data['cross_talk_peak_threshold']
    config.cross_talk_search_range = json_data['cross_talk_search_range(mm)']
    config.cross_talk_time_range = json_data['cross_talk_time_range(bucket)']

    config.cluster_smoothing_neighbor_distance = json_data['cluster_smoothing_neighbor_distance(mm)']
    config.cluster_min_points = json_data['cluster_minimum_points']
    config.cluster_max_neighbor_distance_fractional = json_data['cluster_max_neighbor_distance(fractional)']
    
    return config

def load_config(file_path: Path) -> Config:
    with open(file_path, 'r') as json_file:
        config = load(json_file, object_hook=json_load_config_hook)
        return config