from dataclasses import dataclass
from pathlib import Path

DEFAULT_MAP: Path = Path("DefaultPath")


@dataclass
class PadParameters:
    """Parameters describing the pad map paths

    Attributes
    ----------
    pad_geometry_path: Path
        Path to the csv file containing the pad geometry. If set to DEFAULT_MAP
        uses the packaged maps.
    pad_gain_path: Path
        Path to the csv file containing the relative pad gains. If set to DEFAULT_MAP
        uses the packaged maps.
    pad_time_path: Path
        Path to the csv file containing the pad time corrections. If set to DEFAULT_MAP
        uses the packaged maps.
    """

    pad_geometry_path: Path
    pad_time_path: Path
    pad_scale_path: Path


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


def calculate_window_time(
    micromegas_time_bucket: float,
    drift_velocity: float,
    detector_length: float,
    get_frequency: float,
) -> float:
    """Calculate the window time from a drift velocity

    Given a known micromegas time, drift velocity, detector length, and GET sampling frequency
    calculate the expected window time.

    Parameters
    ----------
    micromegas_time_bucket: float
        The micromegas time in GET Time Buckets
    drift_velocity: float
        The electron drift velocity in mm/us
    detector_length: float
        The detector length in mm
    get_frequency: float
        The GET electronics sampling frequency

    Returns
    -------
    float
        The window time in GET time buckets
    """
    return (detector_length / drift_velocity) * get_frequency + micromegas_time_bucket


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
    """

    baseline_window_scale: float
    peak_separation: float
    peak_prominence: float
    peak_max_width: float
    peak_threshold: float
    ic_delay_time_bucket: int
    ic_multiplicity: int


@dataclass
class ContinuityJoinParameters:
    """Parameters for joining clusters based on continuity

    Attributes
    ----------
    join_radius_fraction: float
        The percent difference allowed for the cylindrical radius. Used as
        radius_threshold = join_radius_fraction * (max_radius - min_radius)
        where max_radius and min_radius are the maximum, minimum radius value over both
        clusters being compared
    join_z_fraction: float
        The percent difference allowed for the cylindrical z. Used as
        z_threshold = join_z_fraction * (max_z - min_z)
        where max_z and min_z are the maximum, minimum z value over both
        clusters being compared

    """

    join_radius_fraction: float
    join_z_fraction: float


@dataclass
class OverlapJoinParameters:
    """Parameters for joining clusters based on area of overlap

    Attributes
    ----------
    min_cluster_size_join: int
        The minimum size of a cluster for it to be included in the joining algorithm
    circle_overlap_ratio: float
        Minimum overlap ratio between two circles in the cluster joining algorithm. Used
        as area_overlap > circle_overlap_ratio * (min_area) where area_overlap is the
        area overlap and min_area is the minimum area of the two clusters being compared
    """

    circle_overlap_ratio: float
    min_cluster_size_join: int


@dataclass
class TripclustParameters:
    """Parameters for the tripclust (Dalitz) clustering algorithms

    Attributes
    ----------
    r: float
        maximum neighbour distance for smoothing (default 2)
    rdnn: boolean
        whether or not compute r with dnn (default true)
    k: int
        number of tested neighbours of triplet mid point (default 19)
    n: int
        max number of triplets to one mid point (default 2)
    a: float
        1 - cos alpha, where alpha is the angle between the two triplet branches (default 0.03)
    s: float
        distance scale factor in metric (default 0.3)
    sdnn: boolean
        whether or not compute s with dnn (default true)
    t: float
        threshold for cdist in clustering (default 0.0)
    tauto: boolean
        whether or not auto generate t (default true)
    dmax: float
        maximum gap width (default 0.0)
    dmax_dnn: boolean
        whether or not use dnn for dmax (default false)
    ordered: boolean
        whether or not points are in chronological order (default false)
    link: int
        linkage method for clustering (default 0=SINGLE)
    m: int
        min number of triplets per cluster (default 5)
    postprocess: boolean
        whether or not post processing should be enabled (default false)
    min_depth: int
        minimum number of points making a branch in curve in post processing (default 25)
    """

    r: float
    rdnn: bool
    k: int
    n: int
    a: float
    s: float
    sdnn: bool
    t: float
    tauto: bool
    dmax: float
    dmax_dnn: bool
    ordered: bool
    link: int
    m: int
    postprocess: bool
    min_depth: int


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
    min_cluster_size_join: int
        The minimum size of a cluster for it to be included in the joining algorithm
    circle_overlap_ratio: float
        minimum overlap ratio between two circles in the cluster joining algorithm
    outlier_scale_factor: float
        Factor which is multiplied by the number of points in a trajectory to set the number of neighbors parameter
        for scikit-learns LocalOutlierFactor test
    """

    min_cloud_size: int
    min_points: int
    min_size_scale_factor: float
    min_size_lower_cutoff: int
    cluster_selection_epsilon: float
    overlap_join: OverlapJoinParameters | None
    continuity_join: ContinuityJoinParameters | None
    outlier_scale_factor: float
    tc_params: TripclustParameters | None


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
        Path to a spyral-utils GasTarget or GasMixtureTarget file
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
    fit_vertex_rho: bool
        Control whether or not the vertex rho position is fitted (True=fitted, False=fixed)
    fit_vertex_phi: bool
        Control whether or not the vertex phi position is fitted (True=fitted, False=fixed)
    fit_azimuthal: bool
        Control whether or not the trajectory azimuthal angle is fitted (True=fitted, False=fixed)
    fit_method: str
        What type of fitting to use, options are "lbfgsb" or "leastsq"
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
    fit_vertex_rho: bool
    fit_vertex_phi: bool
    fit_azimuthal: bool
    fit_method: str
