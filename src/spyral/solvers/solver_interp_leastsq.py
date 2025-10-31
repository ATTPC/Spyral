from .guess import Guess, SolverResult, create_params
from ..core.cluster import Cluster
from ..interpolate.track_interpolator import TrackInterpolator
from ..core.constants import QBRHO_2_P, BIG_PAD_HEIGHT
from ..core.config import DetectorParameters, SolverParameters

from spyral_utils.nuclear import NucleusData

from lmfit import Parameters, minimize, fit_report
from lmfit.minimizer import MinimizerResult
import numpy as np
from numba import njit, prange


@njit(fastmath=True, error_model="numpy", inline="always")
def distances(track: np.ndarray, data: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Calculate the distance (error) of a track solution to the data

    Loop over the data and approximate the error as the closest distance of a point in the
    track solution to the data. JIT-ed for speed

    Parameters
    ----------
    track: numpy.ndarray
        The track solution
    data: numpy.ndarray
        The data to compare to
    weights: numpy.ndarray
        The weights due to data uncertainty (1/sigma)

    Returns
    -------
    np.ndarray
        The minimum error for each data point weighted by uncertainty
    """
    assert track.shape[1] == 3
    assert data.shape[1] == 3
    assert len(data) == len(weights)

    dists = np.zeros((len(data), len(track)))
    error = np.zeros(len(data))
    for i in prange(len(data)):
        for j in prange(len(track)):
            dists[i, j] = np.sqrt(
                (track[j, 0] - data[i, 0]) ** 2.0
                + (track[j, 1] - data[i, 1]) ** 2.0
                + (track[j, 2] - data[i, 2]) ** 2.0
            )
        error[i] = np.min(dists[i]) * weights[i]
    return error


def interpolate_trajectory(
    fit_params: Parameters, interpolator: TrackInterpolator, ejectile: NucleusData
) -> np.ndarray | None:
    """Use the interpolation scheme to generate a trajectory from the given fit parameters.

    Parameters
    ----------
    fit_params: lmfit.Parameters
        the set of lmfit Parameters
    interpolator: TrackInterpolator
        the interpolation scheme
    ejectile: NucleusData
        data for the particle being tracked

    Returns
    -------
    numpy.ndarray | None
        Returns a array of interpolated ODE trajectory data. Upon failure (typically an out of bounds for the interpolation scheme) returns None.
    """
    vertex_x = fit_params["vertex_x"].value
    vertex_y = fit_params["vertex_y"].value
    vertex_z = fit_params["vertex_z"].value
    momentum = QBRHO_2_P * (fit_params["brho"].value * float(ejectile.Z))
    kinetic_energy = np.sqrt(momentum**2.0 + ejectile.mass**2.0) - ejectile.mass
    polar = fit_params["polar"].value
    azimuthal = fit_params["azimuthal"].value

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
    weights: np.ndarray,
    interpolator: TrackInterpolator,
    ejectile: NucleusData,
) -> np.ndarray:
    """Function to be minimized. Returns errors for data compared to estimated track.

    Parameters
    ----------
    fit_params: lmfit.Parameters
        the set of lmfit Parameters
    x: numpy.ndarray
        the data to be fit (x,y,z) coordinates in meters
    weights: numpy.ndarray
        The assoicated weights due uncertainties of the data (1/sigma)
    interpolator: TrackInterpolator
        the interpolation scheme to be used
    ejectile: spyral_utils.nuclear.NucleusData
        the data for the particle being tracked

    Returns
    -------
    numpy.ndarray
        the residuals weighted by uncertainty
    """
    trajectory = interpolate_trajectory(fit_params, interpolator, ejectile)
    if trajectory is None:
        return np.full(len(x), 1.0e6)
    return distances(trajectory, x, weights)


# For testing, not for use in production
def fit_model_interp(
    cluster: Cluster,
    guess: Guess,
    ejectile: NucleusData,
    interpolator: TrackInterpolator,
    det_params: DetectorParameters,
    solver_params: SolverParameters,
) -> tuple[SolverResult, Parameters] | None:
    """Used for jupyter notebooks examining the good-ness of the model

    Parameters
    ----------
    cluster: Cluster
        the data to be fit
    guess: Guess
        the initial values of the parameters
    ejectile: spyral_utils.nuclear.NucleusData
        the data for the particle being tracked
    interpolator: TrackInterpolator
        the interpolation scheme to be used
    det_params: DetectorParameters
        Configuration parameters for detector characteristics
    solver_params: SolverParameters
        Configuration parameters for the solver

    Returns
    -------
    tuple[SolverResult, lmfit.Parameters] | None
        Returns the best fit Parameters upon success, or None upon failure
    """
    traj_data = cluster.data[:, :3] * 0.001
    momentum = QBRHO_2_P * (guess.brho * float(ejectile.Z))
    kinetic_energy = np.sqrt(momentum**2.0 + ejectile.mass**2.0) - ejectile.mass
    if not interpolator.check_values_in_range(kinetic_energy, guess.polar):
        return None

    # Uncertainty due to TB resolution in meters
    z_error = (
        det_params.detector_length
        / float(det_params.window_time_bucket - det_params.micromegas_time_bucket)
        * 0.001
    ) * 0.5
    # uncertainty due to pad size, treat as bounding rectangle
    # Note that doesn't matter which side is short/long as we just use
    # total error (distance)
    x_error = cluster.data[:, 4] * BIG_PAD_HEIGHT * 0.5
    y_error = cluster.data[:, 4] * BIG_PAD_HEIGHT / np.sqrt(3.0)
    # total positional variance per point
    total_error = np.sqrt(x_error**2.0 + y_error**2.0 + z_error**2.0)
    weights = 1.0 / total_error

    fit_params = create_params(guess, ejectile, det_params, solver_params)

    result: MinimizerResult = minimize(
        objective_function,
        fit_params,
        args=(traj_data, weights, interpolator, ejectile),
        method="leastsq",
    )
    print(fit_report(result))

    return result.params  # type: ignore


def solve_physics_interp_leastsq(
    cluster_index: int,
    orig_run: int,
    orig_event: int,
    cluster: Cluster,
    guess: Guess,
    ejectile: NucleusData,
    interpolator: TrackInterpolator,
    det_params: DetectorParameters,
    solver_params: SolverParameters,
) -> SolverResult | None:
    """High level function to be called from the application.

    Takes the Cluster and fits a trajectory to it using the initial Guess. It then writes the results to the dictionary.

    Parameters
    ----------
    cluster_index: int
        Index of the cluster in the hdf5 scheme. Used only for debugging
    orig_run: int
        The original run number
    orig_event: int
        The original event number
    cluster: Cluster
        the data to be fit
    guess: Guess
        the initial values of the parameters
    ejectile: spyral_utils.nuclear.NucleusData
        the data for the particle being tracked
    interpolator: TrackInterpolator
        the interpolation scheme to be used
    det_params: DetectorParameters
        Configuration parameters for detector characteristics
    solver_params: SolverParameters
        Configuration parameters for the solver

    Returns
    -------
    SolverResult | None
        The best-fit result of the solver, or None if it failed
    """
    traj_data = cluster.data[:, :3] * 0.001
    momentum = QBRHO_2_P * (guess.brho * float(ejectile.Z))
    kinetic_energy = np.sqrt(momentum**2.0 + ejectile.mass**2.0) - ejectile.mass
    if not interpolator.check_values_in_range(kinetic_energy, guess.polar):
        return None

    # Uncertainty due to TB resolution in meters
    z_error = (
        det_params.detector_length
        / float(det_params.window_time_bucket - det_params.micromegas_time_bucket)
        * 0.001
    ) * 0.5
    # uncertainty due to pad size, treat as bounding rectangle
    # Note that doesn't matter which side is short/long as we just use
    # total error (distance)
    x_error = cluster.data[:, 4] * BIG_PAD_HEIGHT * 0.5
    y_error = cluster.data[:, 4] * BIG_PAD_HEIGHT / np.sqrt(3.0)
    # total positional variance per point
    total_error = np.sqrt(x_error**2.0 + y_error**2.0 + z_error**2.0)
    weights = 1.0 / total_error

    fit_params = create_params(guess, ejectile, det_params, solver_params)

    best_fit: MinimizerResult = minimize(
        objective_function,
        fit_params,
        args=(traj_data, weights, interpolator, ejectile),
        method="leastsq",
    )

    scale_factor = QBRHO_2_P * float(ejectile.Z)
    brho: float = best_fit.params["brho"].value  # type: ignore
    p = brho * scale_factor  # type: ignore
    ke = np.sqrt(p**2.0 + ejectile.mass**2.0) - ejectile.mass
    results = SolverResult(
        event=cluster.event,
        cluster_index=cluster_index,
        cluster_label=cluster.label,
        orig_run=orig_run,
        orig_event=orig_event,
        vertex_x=best_fit.params["vertex_x"].value,  # type: ignore
        sigma_vx=1.0e6,
        vertex_y=best_fit.params["vertex_y"].value,  # type: ignore
        sigma_vy=1.0e6,
        vertex_z=best_fit.params["vertex_z"].value,  # type: ignore
        sigma_vz=1.0e6,
        brho=best_fit.params["brho"].value,  # type: ignore
        sigma_brho=1.0e6,
        ke=ke,
        sigma_ke=1.0e6,
        polar=best_fit.params["polar"].value,  # type: ignore
        sigma_polar=1.0e6,
        azimuthal=best_fit.params["azimuthal"].value,  # type: ignore
        sigma_azimuthal=1.0e6,
        redchisq=best_fit.redchi,  # type: ignore
    )

    if hasattr(best_fit, "uvars"):
        results.sigma_vx = best_fit.uvars["vertex_x"].std_dev  # type: ignore
        results.sigma_vy = best_fit.uvars["vertex_y"].std_dev  # type: ignore
        results.sigma_vz = best_fit.uvars["vertex_z"].std_dev  # type: ignore
        results.sigma_brho = best_fit.uvars["brho"].std_dev  # type: ignore

        # sigma_f = sqrt((df/dx)^2*sigma_x^2 + ...)
        ke_std_dev = np.fabs(
            scale_factor**2.0
            * brho
            / np.sqrt((brho * scale_factor) ** 2.0 + ejectile.mass**2.0)
            * best_fit.uvars["brho"].std_dev  # type: ignore
        )
        results.sigma_ke = ke_std_dev

        results.sigma_polar = best_fit.uvars["polar"].std_dev  # type: ignore
        results.sigma_azimuthal = best_fit.uvars["azimuthal"].std_dev  # type: ignore

    return results
