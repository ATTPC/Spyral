from dataclasses import dataclass
from pathlib import Path

INVALID_PATH: Path = Path("IllegalPath")


@dataclass
class PadParameters:
    """Parameters describing the pad map paths

    Attributes
    ----------
    pad_geometry_path: Path
        Path to the csv file containing the pad geometry
    pad_gain_path: Path
        Path to the csv file containing the relative pad gains
    pad_time_path: Path
        Path to the csv file containing the pad time corrections
    pad_electronics_path: Path
        Path to the csv file containing the pad electronics ids
    """

    is_default: bool
    is_default_legacy: bool
    pad_geometry_path: Path
    pad_gain_path: Path
    pad_time_path: Path
    pad_electronics_path: Path
    pad_scale_path: Path


@dataclass
class RunParameters:
    """Parameters describing the set of operations to be peformed by Spyral

    Attributes
    ----------
    run_min: int
        The first run number, inclusive
    run_max: int
        The last run number, inclusive
    n_processes: int
        The number of child processes
    do_pointcloud: bool
        If true, data is run through the pointcloud phase
    do_cluster: bool
        If true, data is run through the cluster phase
    do_estimate: bool
        If true, data is run through the estimate phase
    do_solve: bool
        If true, data is run through the solve phase
    is_legacy: bool
        If true, data is considered AT-TPC legacy format,
        before the introduction of the FRIBDAQ.
    """

    run_min: int = -1
    run_max: int = -1
    n_processes: int = -1
    do_pointcloud: bool = False
    do_cluster: bool = False
    do_estimate: bool = False
    do_solve: bool = False
    is_legacy: bool = False


@dataclass
class DetectorParameters:
    """Parameters describing the detector configuration

    Attributes
    ----------
    magnetic_field: float
        The magnitude of the magnetic field in Tesla
    electric_field: float
        The magnitude of the electric field in V/m
    detector_length: float
        The detector length in mm
    beam_region_radius: float
        The beam region radius in mm
    micromegas_time_bucket: float
        The micromegas time reference in GET time buckets
    window_time_bucket: float
        The window time reference in GET time buckets
    get_frequency: float
        The GET DAQ sampling frequency in MHz. Typically 3.125 or 6.25
    garfield_file_path: str
        Path to a Garfield simulation file containing electron drift corrections

    """

    magnetic_field: float  # Tesla
    electric_field: float  # V/m
    detector_length: float  # mm
    beam_region_radius: float  # mm
    micromegas_time_bucket: float
    window_time_bucket: float
    get_frequency: float  # MHz
    garfield_file_path: Path
    do_garfield_correction: bool


@dataclass
class GetParameters:
    """Parameters for GET trace signal analysis

    Attributes
    ----------
    baseline_window_scale: float
        The scale factor for the basline correction algorithm
    peak_separation: float
        The peak separation parameter used in scipy.signal.find_peaks
    peak_prominence: float
        The peak prominence parameter used in scipy.signal.find_peaks
    peak_max_width: float
        The maximum peak width parameter used in scipy.signal.find_peaks
    peak_threshold: float
        The minimum amplitude of a valid peak
    """

    baseline_window_scale: float
    peak_separation: float
    peak_prominence: float
    peak_max_width: float
    peak_threshold: float


@dataclass
class FribParameters:
    """Parameters for FRIBDAQ (IC, Si, etc) trace signal analysis

    Attributes
    ----------
    baseline_window_scale: float
        The scale factor for the basline correction algorithm
    peak_separation: float
        The peak separation parameter used in scipy.signal.find_peaks
    peak_prominence: float
        The peak prominence parameter used in scipy.signal.find_peaks
    peak_max_width: float
        The maximum peak width parameter used in scipy.signal.find_peaks
    peak_threshold: float
        The minimum amplitude of a valid peak
    ic_delay_time_bucket: int
        The delay to the IC signal in FRIB TimeBuckets. All IC peaks
        before this TB are ignored when considering IC multiplicity/validity
    ic_multiplicity: int
        The maximum allowed ion chamber multiplicity
    correct_ic_time: bool
        If true, the ion chamber time correction is used to correct the GET trace time
    """

    baseline_window_scale: float
    peak_separation: float
    peak_prominence: float
    peak_max_width: float
    peak_threshold: float
    ic_delay_time_bucket: int
    ic_multiplicity: int
    correct_ic_time: bool


@dataclass
class ClusterParameters:
    """Parameters for clustering, cluster joining, and cluster cleaning

    Attributes
    ----------
    min_cloud_size: int
        The minimum size for a point cloud to be clustered
    min_points: int
        min_samples parameter in scikit-learns' HDBSCAN algorithm
    min_size_scale_factor: int
        Factor which is multiplied by the number of points in a point cloud to set
        the min_cluster_size parameter in scikit-learn's HDBSCAN algorithm
    min_size_lower_cutoff: int
        min_cluster_size parameter in scikit-learn's HDBSCAN algorithm for events where n_points * min_size_scale_factor
        are less than this value.
    cluster_selection_epsilon: float
        cluster_selection_epsilon parameter in scikit-learn's HDBSCAN algorithm. Clusters less than this distance apart
        are merged in the hierarchy
    circle_overlap_ratio: float
        minimum overlap ratio between two circles in the cluster joining algorithm
    fractional_charge_threshold: float
        The maximum allowed difference between two clusters mean charge (relative to the larger mean charge of the two)
        for them to be joined
    outlier_scale_factor: float
        Factor which is multiplied by the number of points in a trajectory to set the number of neighbors parameter
        for scikit-learns LocalOutlierFactor test
    """

    min_cloud_size: int
    min_points: int
    min_size_scale_factor: float
    min_size_lower_cutoff: int
    cluster_selection_epsilon: float
    circle_overlap_ratio: float
    fractional_charge_threshold: float
    outlier_scale_factor: float


@dataclass
class EstimateParameters:
    """Parameters for physics estimation

    Attributes
    ----------
    min_total_trajectory_points: int
        minimum number of points in a cluster for the cluster to be considered a particle trajectory
    smoothing_factor: float
        smoothing factor for creation of smoothing splines. See scipy's documentaion for make_smoothing_splines
        and in particular the lam parameter for more details.
    """

    min_total_trajectory_points: int
    smoothing_factor: float


@dataclass
class SolverParameters:
    """Parameters for physics solving

    Attributes
    ----------
    gas_data_path: str
        Path to a spyral-utils GasTarget file
    particle_id_filename: str
        Name of a particle ID cut file
    ic_min_val: float
        Low value the desired beam region of the ion chamber spectrum
    ic_max_value: float
        High value the desired beam region of the ion chamber spectrum
    n_time_steps: int
        The number of timesteps used in the ODE solver
    interp_ke_min: float
        The minimum value of kinetic energy used in the interpolation scheme in MeV
    interp_ke_max: float
        The maximum value of kinetic energy used in the interpolation scheme in MeV
    interp_ke_bins: int
        The number of kinetic energy bins used in the interpolation scheme
    interp_polar_min: float
        The minimum value of polar angle used in the interpolation scheme in degrees
    interp_polar_max: float
        The maximum value of polar angle used in the interpolation scheme in degrees
    interp_polar_bins: int
        The number of polar angle bins used in the interpolation scheme
    """

    gas_data_path: Path
    particle_id_filename: Path
    ic_min_val: float
    ic_max_val: float
    n_time_steps: int
    interp_ke_min: float
    interp_ke_max: float
    interp_ke_bins: int
    interp_polar_min: float
    interp_polar_max: float
    interp_polar_bins: int
