from dataclasses import dataclass, field
from pathlib import Path
from json import load
from typing import Any

@dataclass
class WorkspaceParameters:
    trace_data_path: str = ''
    workspace_path: str = ''
    pad_geometry_path: str = ''
    pad_gain_path: str = ''
    pad_time_path: str = ''
    pad_electronics_path: str = ''

@dataclass
class RunParameters:
    run_min: int = -1
    run_max: int = -1
    do_phase1: bool = False
    do_phase2: bool = False
    do_phase3: bool = False
    do_phase4: bool = False
    do_phase5: bool = False

@dataclass
class DetectorParameters:
    magnetic_field: float = 0.0 #Tesla
    electric_field: float = 0.0 #V/m
    tilt_angle: float = 0.0 #degrees
    detector_length: float = 0.0 #mm
    beam_region_radius: float = 0.0 #mm
    micromegas_time_bucket: float = 0.0
    window_time_bucket: float = 0.0

@dataclass
class GasParameters:
    density: float = 0.0 #g/cm^3
    energy_loss_path: str = ''

@dataclass
class TraceParameters:
    baseline_window_scale: float = 20.0
    peak_separation: float = 50.0
    peak_prominence: float = 20.0
    peak_max_width: float = 100.0
    peak_threshold: float = 25.0

@dataclass
class CrossTalkParameters:
    saturation_threshold: float = 2000.0
    cross_talk_threshold: float = 1000.0
    neighborhood_threshold: float = 1500.0
    channel_range: int = 5
    distance_range: float = 5.0
    time_range: int =  10

@dataclass
class ClusterParameters:
    smoothing_neighbor_distance: float = 0.0 #mm
    min_points: int = 0
    min_size: int = 0
    max_center_distance: float = 0.0

@dataclass
class Config:
    #Workspace
    workspace: WorkspaceParameters = field(default_factory=WorkspaceParameters)

    #Run
    run: RunParameters = field(default_factory=RunParameters)

    #Detector
    detector: DetectorParameters = field(default_factory=DetectorParameters)

    #Gas
    gas: GasParameters = field(default_factory=GasParameters)

    #Traces
    trace: TraceParameters = field(default_factory=TraceParameters)

    #Cross talk
    cross: CrossTalkParameters = field(default_factory=CrossTalkParameters)

    #Clustering settings
    cluster: ClusterParameters = field(default_factory=ClusterParameters)

def json_load_config_hook(json_data: dict[Any, Any]) -> Config:
    config = Config()
    config.workspace.trace_data_path = json_data['trace_data_path']
    config.workspace.workspace_path = json_data['workspace_path']
    config.workspace.pad_geometry_path = json_data['pad_geometry_path']
    config.workspace.pad_gain_path = json_data['pad_gain_path']
    config.workspace.pad_time_path = json_data['pad_time_path']
    config.workspace.pad_electronics_path = json_data['pad_electronics_path']

    config.run.run_min = json_data['run_min']
    config.run.run_max = json_data['run_max']
    config.run.do_phase1 = json_data['phase1']
    config.run.do_phase2 = json_data['phase2']
    config.run.do_phase3 = json_data['phase3']
    config.run.do_phase4 = json_data['phase4']
    config.run.do_phase5 = json_data['phase5']

    config.detector.magnetic_field = json_data['magnetic_field(T)']
    config.detector.electric_field = json_data['electric_field(V/m)']
    config.detector.tilt_angle = json_data['tilt_angle(degrees)']
    config.detector.detector_length = json_data['detector_length(mm)']
    config.detector.beam_region_radius = json_data['beam_region_radius(mm)']
    config.detector.micromegas_time_bucket = json_data['micromegas_time_bucket']
    config.detector.window_time_bucket = json_data['window_time_bucket']

    config.gas.density = json_data['gas_density(g/cm^3)']
    config.gas.energy_loss_path = json_data['gas_energy_loss_path']

    config.trace.baseline_window_scale = json_data['trace_baseline_window_scale']
    config.trace.peak_separation = json_data['trace_peak_separation']
    config.trace.peak_prominence = json_data['trace_peak_prominence']
    config.trace.peak_max_width = json_data['trace_peak_max_width']
    config.trace.peak_threshold = json_data['trace_peak_threshold']

    config.cross.saturation_threshold = json_data['cross_talk_saturation_threshold']
    config.cross.cross_talk_threshold = json_data['cross_talk_threshold']
    config.cross.neighborhood_threshold = json_data['cross_talk_neighborhood_threshold']
    config.cross.channel_range = json_data['cross_talk_channel_range']
    config.cross.distance_range = json_data['cross_talk_distance_range(mm)']
    config.cross.time_range = json_data['cross_talk_time_range(bucket)']

    config.cluster.smoothing_neighbor_distance = json_data['cluster_smoothing_neighbor_distance(mm)']
    config.cluster.min_size = json_data['cluster_minimum_size']
    config.cluster.min_points = json_data['cluster_minimum_points']
    config.cluster.max_center_distance = json_data['cluster_max_center_distance(mm)']
    
    return config

def load_config(file_path: Path) -> Config:
    with open(file_path, 'r') as json_file:
        config = load(json_file, object_hook=json_load_config_hook)
        return config