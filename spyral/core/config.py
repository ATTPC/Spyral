from dataclasses import dataclass, field
from pathlib import Path
from json import load
from typing import Any

@dataclass
class WorkspaceParameters:
    '''
    Parameters describing paths to various resources used across the application
    '''
    trace_data_path: str = ''
    workspace_path: str = ''
    pad_geometry_path: str = ''
    pad_gain_path: str = ''
    pad_time_path: str = ''
    pad_electronics_path: str = ''
    nuclear_data_path: str = ''

@dataclass
class RunParameters:
    '''
    Parameters describing the set of operations to be peformed
    '''
    run_min: int = -1
    run_max: int = -1
    n_processes: int = -1
    do_phase1: bool = False
    do_phase2: bool = False
    do_phase3: bool = False
    do_phase4: bool = False

@dataclass
class DetectorParameters:
    '''
    Parameters describing the detector configuration
    '''
    magnetic_field: float = 0.0 #Tesla
    electric_field: float = 0.0 #V/m
    detector_length: float = 0.0 #mm
    beam_region_radius: float = 0.0 #mm
    micromegas_time_bucket: float = 0.0
    window_time_bucket: float = 0.0
    get_frequency: float = 0.0 #MHz
    garfield_file_path: str = ''
    efield_correction_name: str = ''

@dataclass
class TraceParameters:
    '''
    Parameters for trace baseline and peak finding
    '''
    baseline_window_scale: float = 20.0
    peak_separation: float = 50.0
    peak_prominence: float = 20.0
    peak_max_width: float = 100.0
    peak_threshold: float = 25.0

@dataclass
class FribParameters:
    '''
    Parameters for data taken using the FRIBDAQ (IC, Si, etc)
    '''
    baseline_window_scale: float = 20.0
    peak_separation: float = 50.0
    peak_prominence: float = 20.0
    peak_max_width: float = 100.0
    peak_threshold: float = 25.0
    ic_multiplicity: int = 1
    correct_ic_time: bool = True

@dataclass
class ClusterParameters:
    '''
    Parameters for clustering, cluster joining, and cluster cleaning
    '''
    smoothing_neighbor_distance: float = 0.0 #mm
    min_points: int = 0
    min_size: int = 0
    circle_overlap_ratio: float = 0.0
    fractional_charge_threshold: float = 0.0
    n_neighbors_outiler_test: int = 0

@dataclass
class EstimateParameters:
    '''
    Parameters for physics estimation
    '''
    min_total_trajectory_points: int = 0
    max_distance_from_beam_axis: float = 0.0 #mm

@dataclass
class SolverParameters:
    '''
    Parameters for physics solving
    '''
    gas_data_path: str = ''
    particle_id_filename: str = ''
    ic_min_val: float = 0.0
    ic_max_val: float = 0.0
    interp_file_name: str = ''
    interp_ke_min: float = 0.0
    interp_ke_max: float = 0.0
    interp_ke_bins: int = 0
    interp_polar_min: float = 0.0
    interp_polar_max: float = 0.0
    interp_polar_bins: int = 0

@dataclass
class Config:
    '''
    Container which holds all configuration parameters. Can be serialized/deserialized to json.
    '''
    #Workspace
    workspace: WorkspaceParameters = field(default_factory=WorkspaceParameters)

    #Run
    run: RunParameters = field(default_factory=RunParameters)

    #Detector
    detector: DetectorParameters = field(default_factory=DetectorParameters)

    #Traces
    trace: TraceParameters = field(default_factory=TraceParameters)

    #FRIB data
    frib: FribParameters = field(default_factory=FribParameters)

    #Clustering settings
    cluster: ClusterParameters = field(default_factory=ClusterParameters)

    #Physics Estimate settings
    estimate: EstimateParameters =  field(default_factory=EstimateParameters)

    #Physics Solver
    solver: SolverParameters = field(default_factory=SolverParameters)

def json_load_config_hook(json_data: dict[Any, Any]) -> Config:
    config = Config()
    config.workspace.trace_data_path = json_data['trace_data_path']
    config.workspace.workspace_path = json_data['workspace_path']
    config.workspace.pad_geometry_path = json_data['pad_geometry_path']
    config.workspace.pad_gain_path = json_data['pad_gain_path']
    config.workspace.pad_time_path = json_data['pad_time_path']
    config.workspace.pad_electronics_path = json_data['pad_electronics_path']
    config.workspace.nuclear_data_path = json_data['nuclear_data_path']

    config.run.run_min = json_data['run_min']
    config.run.run_max = json_data['run_max']
    config.run.n_processes = json_data['n_processes']
    config.run.do_phase1 = json_data['phase1']
    config.run.do_phase2 = json_data['phase2']
    config.run.do_phase3 = json_data['phase3']
    config.run.do_phase4 = json_data['phase4']

    config.detector.magnetic_field = json_data['magnetic_field(T)']
    config.detector.electric_field = json_data['electric_field(V/m)']
    config.detector.detector_length = json_data['detector_length(mm)']
    config.detector.beam_region_radius = json_data['beam_region_radius(mm)']
    config.detector.micromegas_time_bucket = json_data['micromegas_time_bucket']
    config.detector.window_time_bucket = json_data['window_time_bucket']
    config.detector.get_frequency = json_data['get_frequency(MHz)']
    config.detector.garfield_file_path = json_data['electric_field_garfield_path']
    config.detector.efield_correction_name = json_data['electric_field_correction_file_name']

    config.trace.baseline_window_scale = json_data['trace_baseline_window_scale']
    config.trace.peak_separation = json_data['trace_peak_separation']
    config.trace.peak_prominence = json_data['trace_peak_prominence']
    config.trace.peak_max_width = json_data['trace_peak_max_width']
    config.trace.peak_threshold = json_data['trace_peak_threshold']

    config.frib.baseline_window_scale = json_data['frib_trace_baseline_window_scale']
    config.frib.peak_separation = json_data['frib_trace_peak_separation']
    config.frib.peak_prominence = json_data['frib_trace_peak_prominence']
    config.frib.peak_max_width = json_data['frib_trace_peak_max_width']
    config.frib.peak_threshold = json_data['frib_trace_peak_threshold']
    config.frib.ic_multiplicity = json_data['frib_event_ic_multiplicity']
    config.frib.correct_ic_time = json_data['frib_event_correct_ic_time']

    config.cluster.smoothing_neighbor_distance = json_data['cluster_smoothing_neighbor_distance(mm)']
    config.cluster.min_size = json_data['cluster_minimum_size']
    config.cluster.min_points = json_data['cluster_minimum_points']
    config.cluster.circle_overlap_ratio = json_data['cluster_circle_overlap_ratio']
    config.cluster.fractional_charge_threshold = json_data['cluster_fractional_charge_threshold']
    config.cluster.n_neighbors_outiler_test = json_data['cluster_n_neighbors_outlier_test']

    config.estimate.min_total_trajectory_points = json_data['estimate_mininum_total_trajectory_points']
    config.estimate.max_distance_from_beam_axis = json_data['estimate_maximum_distance_from_beam_axis']

    config.solver.gas_data_path = json_data['solver_gas_data_path']
    config.solver.particle_id_filename = json_data['solver_particle_id_file']
    config.solver.ic_min_val = json_data['solver_ic_min']
    config.solver.ic_max_val = json_data['solver_ic_max']
    config.solver.interp_file_name = json_data['solver_interp_file_name']
    config.solver.interp_ke_min = json_data['solver_interp_ke_min(MeV)']
    config.solver.interp_ke_max = json_data['solver_interp_ke_max(MeV)']
    config.solver.interp_ke_bins = json_data['solver_interp_ke_bins']
    config.solver.interp_polar_min = json_data['solver_interp_polar_min(deg)']
    config.solver.interp_polar_max = json_data['solver_interp_polar_max(deg)']
    config.solver.interp_polar_bins = json_data['solver_interp_polar_bins']
    
    return config

def load_config(file_path: Path) -> Config:
    with open(file_path, 'r') as json_file:
        config = load(json_file, object_hook=json_load_config_hook)
        return config
