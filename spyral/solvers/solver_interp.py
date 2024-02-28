from .guess import Guess
from ..core.cluster import Cluster
from ..interpolate.track_interpolator import TrackInterpolator
from ..core.constants import QBRHO_2_P
from ..core.config import DetectorParameters

from spyral_utils.nuclear import NucleusData

from lmfit import Parameters, minimize, fit_report
from lmfit.minimizer import MinimizerResult
import numpy as np
from numba import njit, prange


@njit(fastmath=True, error_model="numpy", inline="always")
def distances(track: np.ndarray, data: np.ndarray) -> float:
    """Calculate the average distance (error) of a track solution to the data

    Loop over the data and approximate the error as the closest distance of a point in the
    track solution to the data. JIT-ed for speed

    Parameters
    ----------
    track: ndarray
        The track solution
    data: ndarray
        The data to compare to

    Returns
    -------
    float
        The average distance (error)
    """
    assert track.shape[1] == 3
    assert data.shape[1] == 3

    dists = np.zeros((len(data), len(track)))
    errors = np.zeros(len(data))
    for i in prange(len(data)):
        for j in prange(len(track)):
            dists[i, j] = np.sqrt(
                (track[j, 0] - data[i, 0]) ** 2.0
                + (track[j, 1] - data[i, 1]) ** 2.0
                + (track[j, 2] - data[i, 2]) ** 2.0
            )
        errors[i] = np.min(dists[i])
    return np.average(errors)


@njit
def calc_azimuthal(
    vertex_x: float, vertex_y: float, center_x: float, center_y: float
) -> float:
    """Calculate the azimuthal angle of the trajectory from the vertex

    Uses the center of the circle from the estimation phase and the updated vertex position

    Parameters
    ----------
    vertex_x: float
        vertex x coordinate
    vertex_y: float
        vertex y coordinate
    center_x: float
        center x coordinate
    center_y: float
        center y coordinate

    Returns
    -------
    float
        The azimuthal angle of a trajectory from the vertex
    """
    azimuthal = np.arctan2(vertex_y - center_y, vertex_x - center_x)
    if azimuthal < 0.0:
        azimuthal += 2.0 * np.pi
    azimuthal += np.pi * 0.5
    if azimuthal > 2.0 * np.pi:
        azimuthal -= 2.0 * np.pi
    return azimuthal


def interpolate_trajectory(
    fit_params: Parameters, interpolator: TrackInterpolator, ejectile: NucleusData
) -> np.ndarray | None:
    """Use the interpolation scheme to generate a trajectory from the given fit parameters.

    Parameters
    ----------
    fit_params: Parameters
        the set of lmfit Parameters
    interpolator: TrackInterpolator
        the interpolation scheme
    ejectile: NucleusData
        data for the particle being tracked

    Returns
    -------
    ndarray | None
        Returns a array of interpolated ODE trajectory solutions. Upon failure (typically an out of bounds for the interpolation scheme) returns None.
    """
    vertex_x = fit_params["vertex_x"].value
    vertex_y = fit_params["vertex_y"].value
    vertex_z = fit_params["vertex_z"].value
    momentum = QBRHO_2_P * (fit_params["brho"].value * float(ejectile.Z))
    kinetic_energy = np.sqrt(momentum**2.0 + ejectile.mass**2.0) - ejectile.mass
    polar = fit_params["polar"].value
    azimuthal = calc_azimuthal(
        vertex_x, vertex_y, fit_params["center_x"].value, fit_params["center_y"].value
    )

    return interpolator.get_trajectory(
        vertex_x,
        vertex_y,
        vertex_z,
        polar,
        azimuthal,
        kinetic_energy,
    )


def objective_function(
    fit_params: Parameters,
    x: np.ndarray,
    interpolator: TrackInterpolator,
    ejectile: NucleusData,
) -> float:
    """Function to be minimized. Returns errors for data compared to estimated track.

    Parameters
    ----------
    fit_params: Parameters
        the set of lmfit Parameters
    x: ndarray
        the data to be fit (x,y,z) coordinates in meters
    interpolator: TrackInterpolator
        the interpolation scheme to be used
    ejectile: NucleusData
        the data for the particle being tracked

    Returns
    -------
    float
        the error between the estimate and the data
    """
    trajectory = interpolate_trajectory(fit_params, interpolator, ejectile)
    if trajectory is None:
        return 1.0e6
    return distances(trajectory, x)


def create_params(
    guess: Guess,
    ejectile: NucleusData,
    interpolator: TrackInterpolator,
    center_x: float,
    center_y: float,
    det_params: DetectorParameters,
) -> Parameters:
    """Create the lmfit parameters with appropriate bounds

    Convert all values to correct units (meters, radians, etc.) as well

    Parameters
    ----------
    guess: Guess
        the initial values of the parameters
    ejectile: NucleusData
        the data for the particle being tracked
    interpolator: TrackInterpolator
        the interpolation scheme to be used
    center_x: float
        The x-coordinate of the center of the circle fit from the estimation phase
        Units of mm
    center_y: float
        The y-coordinate of the center of the circle fit from the estimation phase
        Units of mm
    det_params: DetectorParameters
        Configuration parameters for detector characteristics

    Returns
    -------
    Parameters
        the lmfit parameters with bounds
    """
    interp_min_momentum = np.sqrt(
        interpolator.ke_min * (interpolator.ke_min + 2.0 * ejectile.mass)
    )
    interp_max_momentum = np.sqrt(
        interpolator.ke_max * (interpolator.ke_max + 2.0 * ejectile.mass)
    )
    interp_min_brho = (interp_min_momentum / QBRHO_2_P) / ejectile.Z
    interp_max_brho = (interp_max_momentum / QBRHO_2_P) / ejectile.Z
    interp_min_polar = interpolator.polar_min * np.pi / 180.0
    interp_max_polar = interpolator.polar_max * np.pi / 180.0

    uncertainty_position_z = 0.1
    uncertainty_brho = 1.0

    min_brho = guess.brho - uncertainty_brho * 2.0
    if min_brho < interp_min_brho:
        min_brho = interp_min_brho
    max_brho = guess.brho + uncertainty_brho * 2.0
    if max_brho > interp_max_brho:
        max_brho = interp_max_brho

    min_polar = interp_min_polar
    max_polar = interp_max_polar
    if guess.polar > np.pi * 0.5 and min_polar < np.pi * 0.5:
        min_polar = np.pi * 0.5
    elif guess.polar < np.pi * 0.5 and max_polar > np.pi * 0.5:
        max_polar = np.pi * 0.5

    min_z = guess.vertex_z * 0.001 - uncertainty_position_z * 2.0
    max_z = guess.vertex_z * 0.001 + uncertainty_position_z * 2.0
    if min_z < 0.0:
        min_z = 0.0
    if max_z > det_params.detector_length * 0.001:
        max_z = det_params.detector_length * 0.001

    vert_phi = np.arctan2(guess.vertex_y, guess.vertex_x)
    if vert_phi < 0.0:
        vert_phi += np.pi * 2.0
    vert_rho = np.sqrt(guess.vertex_x**2.0 + guess.vertex_y**2.0) * 0.001

    fit_params = Parameters()
    fit_params.add("center_x", value=center_x * 0.001, vary=False)
    fit_params.add("center_y", value=center_y * 0.001, vary=False)
    fit_params.add("brho", guess.brho, min=min_brho, max=max_brho)
    fit_params.add("polar", guess.polar, min=min_polar, max=max_polar)
    fit_params.add(
        "vertex_rho",
        value=vert_rho,
        min=0.0,
        max=det_params.beam_region_radius * 0.001,
        vary=True,
    )
    fit_params.add("vertex_phi", value=vert_phi, min=0.0, max=np.pi * 2.0, vary=True)
    fit_params.add("vertex_x", expr="vertex_rho * cos(vertex_phi)")
    fit_params.add("vertex_y", expr="vertex_rho * sin(vertex_phi)")
    fit_params.add("vertex_z", guess.vertex_z * 0.001, min=min_z, max=max_z, vary=True)
    return fit_params


# For testing, not for use in production
def fit_model_interp(
    cluster: Cluster,
    guess: Guess,
    ejectile: NucleusData,
    interpolator: TrackInterpolator,
    center_x: float,
    center_y: float,
    det_params: DetectorParameters,
) -> Parameters | None:
    """Used for jupyter notebooks examining the good-ness of the model

    Parameters
    ----------
    cluster: Cluster
        the data to be fit
    guess: Guess
        the initial values of the parameters
    ejectile: NucleusData
        the data for the particle being tracked
    interpolator: TrackInterpolator
        the interpolation scheme to be used
    center_x: float
        The x-coordinate of the center of the circle fit from the estimation phase
        Units of mm
    center_y: float
        The y-coordinate of the center of the circle fit from the estimation phase
        Units of mm
    det_params: DetectorParameters
        Configuration parameters for detector characteristics

    Returns
    -------
    Parameters | None
        Returns the best fit Parameters upon success, or None upon failure
    """
    traj_data = cluster.data[:, :3] * 0.001
    momentum = QBRHO_2_P * (guess.brho * float(ejectile.Z))
    kinetic_energy = np.sqrt(momentum**2.0 + ejectile.mass**2.0) - ejectile.mass
    if not interpolator.check_values_in_range(kinetic_energy, guess.polar):
        return None

    fit_params = create_params(
        guess, ejectile, interpolator, center_x, center_y, det_params
    )

    result: MinimizerResult = minimize(
        objective_function,
        fit_params,
        args=(traj_data, interpolator, ejectile),
        method="lbfgsb",
    )
    print(fit_report(result))

    return result.params


def solve_physics_interp(
    cluster_index: int,
    cluster: Cluster,
    guess: Guess,
    ejectile: NucleusData,
    interpolator: TrackInterpolator,
    center_x: float,
    center_y: float,
    det_params: DetectorParameters,
    results: dict[str, list],
):
    """High level function to be called from the application.

    Takes the Cluster and fits a trajectory to it using the initial Guess. It then writes the results to the dictionary.

    Parameters
    ----------
    cluster_index: int
        Index of the cluster in the hdf5 scheme. Used only for debugging
    cluster: Cluster
        the data to be fit
    guess: Guess
        the initial values of the parameters
    ejectile: NucleusData
        the data for the particle being tracked
    interpolator: TrackInterpolator
        the interpolation scheme to be used
    center_x: float
        The x-coordinate of the center of the circle fit from the estimation phase
        Units of mm
    center_y: float
        The y-coordinate of the center of the circle fit from the estimation phase
        Units of mm
    det_params: DetectorParameters
        Configuration parameters for detector characteristics
    results: dict[str, list]
        storage for results from the fitting, which will later be written as a dataframe.
    """
    traj_data = cluster.data[:, :3] * 0.001
    momentum = QBRHO_2_P * (guess.brho * float(ejectile.Z))
    kinetic_energy = np.sqrt(momentum**2.0 + ejectile.mass**2.0) - ejectile.mass
    if not interpolator.check_values_in_range(kinetic_energy, guess.polar):
        return

    fit_params = create_params(
        guess, ejectile, interpolator, center_x, center_y, det_params
    )

    best_fit: MinimizerResult = minimize(
        objective_function,
        fit_params,
        args=(traj_data, interpolator, ejectile),
        method="lbfgsb",
    )

    results["event"].append(cluster.event)
    results["cluster_index"].append(cluster_index)
    results["cluster_label"].append(cluster.label)
    # Best fit values and uncertainties
    results["vertex_x"].append(best_fit.params["vertex_x"].value)
    results["vertex_y"].append(best_fit.params["vertex_y"].value)
    results["vertex_z"].append(best_fit.params["vertex_z"].value)
    results["brho"].append(best_fit.params["brho"].value)
    results["polar"].append(best_fit.params["polar"].value)
    azim = calc_azimuthal(
        results["vertex_x"][-1], results["vertex_y"][-1], center_x, center_y
    )
    results["azimuthal"].append(azim)
    results["redchisq"].append(best_fit.redchi)

    # Right now we can't quantify uncertainties
    results["sigma_vx"].append(1.0e6)
    results["sigma_vy"].append(1.0e6)
    results["sigma_vz"].append(1.0e6)
    results["sigma_brho"].append(1.0e6)
    results["sigma_polar"].append(1.0e6)
    results["sigma_azimuthal"].append(1.0e6)
