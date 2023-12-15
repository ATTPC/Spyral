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

@dataclass
class RunParameters:
    '''
    Parameters describing the set of operations to be peformed
    '''
    run_min: int = -1
    run_max: int = -1
    n_processes: int = -1
    do_pointcloud: bool = False
    do_cluster: bool = False
    do_estimate: bool = False
    do_solve: bool = False

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

@dataclass
class GetParameters:
    '''
    Parameters for GET traces
    '''
    baseline_window_scale: float = 20.0
    peak_separation: float = 50.0
    peak_prominence: float = 20.0
    peak_max_width: float = 100.0
    peak_threshold: float = 25.0

@dataclass
class FribParameters:
    '''
    Parameters for traces taken using the FRIBDAQ (IC, Si, etc)
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
    n_time_steps: int = 0
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
    get: GetParameters = field(default_factory=GetParameters)

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
    ws_params = json_data['Workspace']
    config.workspace.trace_data_path = ws_params['trace_data_path']
    config.workspace.workspace_path = ws_params['workspace_path']
    config.workspace.pad_geometry_path = ws_params['pad_geometry_path']
    config.workspace.pad_gain_path = ws_params['pad_gain_path']
    config.workspace.pad_time_path = ws_params['pad_time_path']
    config.workspace.pad_electronics_path = ws_params['pad_electronics_path']

    run_params = json_data['Run']
    config.run.run_min = run_params['run_min']
    config.run.run_max = run_params['run_max']
    config.run.n_processes = run_params['n_processes']
    config.run.do_pointcloud = run_params['phase_pointcloud']
    config.run.do_cluster = run_params['phase_cluster']
    config.run.do_estimate = run_params['phase_estimate']
    config.run.do_solve = run_params['phase_solve']

    det_params = json_data['Detector']
    config.detector.magnetic_field = det_params['magnetic_field(T)']
    config.detector.electric_field = det_params['electric_field(V/m)']
    config.detector.detector_length = det_params['detector_length(mm)']
    config.detector.beam_region_radius = det_params['beam_region_radius(mm)']
    config.detector.micromegas_time_bucket = det_params['micromegas_time_bucket']
    config.detector.window_time_bucket = det_params['window_time_bucket']
    config.detector.get_frequency = det_params['get_frequency(MHz)']
    config.detector.garfield_file_path = det_params['electric_field_garfield_path']

    get_params = json_data['GET']
    config.get.baseline_window_scale = get_params['baseline_window_scale']
    config.get.peak_separation = get_params['peak_separation']
    config.get.peak_prominence = get_params['peak_prominence']
    config.get.peak_max_width = get_params['peak_max_width']
    config.get.peak_threshold = get_params['peak_threshold']

    frib_params = json_data['FRIB']
    config.frib.baseline_window_scale = frib_params['baseline_window_scale']
    config.frib.peak_separation = frib_params['peak_separation']
    config.frib.peak_prominence = frib_params['peak_prominence']
    config.frib.peak_max_width = frib_params['peak_max_width']
    config.frib.peak_threshold = frib_params['peak_threshold']
    config.frib.ic_multiplicity = frib_params['event_ic_multiplicity']
    config.frib.correct_ic_time = frib_params['event_correct_ic_time']

    cluster_params = json_data['Cluster']
    config.cluster.smoothing_neighbor_distance = cluster_params['smoothing_neighbor_distance(mm)']
    config.cluster.min_size = cluster_params['minimum_size']
    config.cluster.min_points = cluster_params['minimum_points']
    config.cluster.circle_overlap_ratio = cluster_params['circle_overlap_ratio']
    config.cluster.fractional_charge_threshold = cluster_params['fractional_charge_threshold']
    config.cluster.n_neighbors_outiler_test = cluster_params['n_neighbors_outlier_test']

    est_params = json_data['Estimate']
    config.estimate.min_total_trajectory_points = est_params['mininum_total_trajectory_points']
    config.estimate.max_distance_from_beam_axis = est_params['maximum_distance_from_beam_axis']

    solver_params = json_data['Solver']
    config.solver.gas_data_path = solver_params['gas_data_path']
    config.solver.particle_id_filename = solver_params['particle_id_file']
    config.solver.ic_min_val = solver_params['ic_min']
    config.solver.ic_max_val = solver_params['ic_max']
    config.solver.n_time_steps = solver_params['ode_n_time_steps']
    config.solver.interp_ke_min = solver_params['interp_ke_min(MeV)']
    config.solver.interp_ke_max = solver_params['interp_ke_max(MeV)']
    config.solver.interp_ke_bins = solver_params['interp_ke_bins']
    config.solver.interp_polar_min = solver_params['interp_polar_min(deg)']
    config.solver.interp_polar_max = solver_params['interp_polar_max(deg)']
    config.solver.interp_polar_bins = solver_params['interp_polar_bins']
    
    return config

def load_config(file_path: Path) -> Config:
    with open(file_path, 'r') as json_file:
        config_data = load(json_file)
        return json_load_config_hook(config_data)
