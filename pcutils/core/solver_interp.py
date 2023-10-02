from .nuclear_data import NucleusData
from .cluster import Cluster
from .track_generator import TrackInterpolator, InitialState, QBRHO_2_P
from .config import DetectorParameters

from lmfit import Parameters, minimize, fit_report
from lmfit.minimizer import MinimizerResult
import numpy as np
import math
from scipy import constants, interpolate
from scipy.interpolate import CubicSpline
from dataclasses import dataclass

@dataclass
class Guess:
    brho: float
    polar: float
    azimuthal: float
    vertex_x: float
    vertex_y: float
    vertex_z: float

def generate_trajectory(fit_params: Parameters, interpolator: TrackInterpolator, ejectile: NucleusData) -> CubicSpline:
    state = InitialState()
    state.vertex_x = fit_params['vertex_x'].value
    state.vertex_y = fit_params['vertex_y'].value
    state.vertex_z = fit_params['vertex_z'].value
    momentum = QBRHO_2_P * (fit_params['brho'].value * 10.0 * 100.0 * float(ejectile.Z))
    state.kinetic_energy = math.sqrt(momentum**2.0 + ejectile.mass**2.0) - ejectile.mass
    state.polar = fit_params['polar'].value
    state.azimuthal = fit_params['azimuthal'].value

    return interpolator.get_interpolated_trajectory(state)


def objective_function(fit_params: Parameters, x: np.ndarray, interpolator: TrackInterpolator, ejectile: NucleusData) -> np.ndarray:
    trajectory = generate_trajectory(fit_params, interpolator, ejectile)
    errors = np.full(len(x), 1.0e6)
    xy = trajectory(x[:, 2])
    valid_trajectory = xy[~np.isnan(xy[:, 0])]
    limit = len(valid_trajectory)
    errors[:limit] = np.linalg.norm(x[:limit, :2] - valid_trajectory, axis=1)
    return errors[:]

def create_params(guess: Guess, ejectile: NucleusData, interpolator: TrackInterpolator) -> Parameters:
    interp_min_momentum = math.sqrt(interpolator.ke_min * (interpolator.ke_min + 2.0 * ejectile.mass))
    interp_max_momentum = math.sqrt(interpolator.ke_max * (interpolator.ke_max + 2.0 * ejectile.mass))
    interp_min_brho =  interp_min_momentum / QBRHO_2_P / (ejectile.Z * 10.0 * 100.0)
    interp_max_brho =  interp_max_momentum / QBRHO_2_P / (ejectile.Z * 10.0 * 100.0)

    interp_min_polar = interpolator.polar_min * np.pi / 180.0
    interp_max_polar = interpolator.polar_max * np.pi / 180.0

    uncertainty_position = 0.05
    uncertainty_brho = 1.0

    min_brho = guess.brho - guess.brho * uncertainty_brho * 2.0
    if min_brho < interp_min_brho:
        min_brho = interp_min_brho
    max_brho = guess.brho + guess.brho * uncertainty_brho * 2.0
    if max_brho > interp_max_brho:
        max_brho = interp_max_brho

    min_polar = interp_min_polar
    max_polar = interp_max_polar
    if guess.polar > np.pi * 0.5:
        min_polar += np.pi * 0.5
        max_polar += np.pi * 0.5

    min_azimuthal = guess.azimuthal - np.pi*0.25
    max_azimuthal = guess.azimuthal + np.pi*0.25

    min_x = guess.vertex_x * 0.001 - uncertainty_position * 2.0
    max_x = guess.vertex_x * 0.001 + uncertainty_position * 2.0
    min_y = guess.vertex_y * 0.001 - uncertainty_position * 2.0
    max_y = guess.vertex_y * 0.001 + uncertainty_position * 2.0
    min_z = guess.vertex_z * 0.001 - uncertainty_position * 2.0
    max_z = guess.vertex_z * 0.001 + uncertainty_position * 2.0

    fit_params = Parameters()
    fit_params.add('brho', guess.brho, min=min_brho, max=max_brho)
    fit_params.add('polar', guess.polar, min=min_polar, max=max_polar)
    fit_params.add('azimuthal', guess.azimuthal, min=min_azimuthal, max=max_azimuthal)
    fit_params.add('vertex_x', guess.vertex_x * 0.001, min=min_x, max=max_x)
    fit_params.add('vertex_y', guess.vertex_y * 0.001, min=min_y, max=max_y)
    fit_params.add('vertex_z', guess.vertex_z * 0.001, min=min_z, max=max_z)
    return fit_params


#For testing, not for use in production
def fit_model(cluster: Cluster, guess: Guess, interpolator: TrackInterpolator, ejectile: NucleusData) -> Parameters | None:
    traj_data = cluster.data[:, :3] * 0.001
    momentum = QBRHO_2_P * (guess.brho * 10.0 * 100.0 * float(ejectile.Z))
    kinetic_energy = math.sqrt(momentum**2.0 + ejectile.mass**2.0) - ejectile.mass
    if not interpolator.check_values_in_range(kinetic_energy, guess.polar):
        return None
    
    fit_params = create_params(guess, ejectile, interpolator)

    result: MinimizerResult = minimize(objective_function, fit_params, args = (traj_data, interpolator, ejectile))
    print(fit_report(result))

    return result.params
        

def solve_physics_interp(cluster_index: int, cluster: Cluster, guess: Guess, interpolator: TrackInterpolator, ejectile: NucleusData, results: dict[str, list]):
    traj_data = cluster.data[:, :3] * 0.001
    momentum = QBRHO_2_P * (guess.brho * 10.0 * 100.0 * float(ejectile.Z))
    kinetic_energy = math.sqrt(momentum**2.0 + ejectile.mass**2.0) - ejectile.mass
    if not interpolator.check_values_in_range(kinetic_energy, guess.polar):
        return None
    
    fit_params = create_params(guess, ejectile, interpolator)

    best_fit: MinimizerResult = minimize(objective_function, fit_params, args = (traj_data, interpolator, ejectile))

    results['event'].append(cluster.event)
    results['cluster_index'].append(cluster_index)
    results['cluster_label'].append(cluster.label)
    #Best fit values and uncertainties
    results['vertex_x'].append(best_fit.params['vertex_x'].value)
    results['vertex_y'].append(best_fit.params['vertex_y'].value)
    results['vertex_z'].append(best_fit.params['vertex_z'].value)
    results['brho'].append(best_fit.params['brho'].value)
    results['polar'].append(best_fit.params['polar'].value)
    results['azimuthal'].append(best_fit.params['azimuthal'].value)
    results['redchisq'].append(best_fit.redchi)

    #Sometimes fit is so bad uncertainties cannot be estimated
    if (hasattr(best_fit, 'uvars')):
        results['sigma_vx'].append(best_fit.uvars['vertex_x'].s)
        results['sigma_vy'].append(best_fit.uvars['vertex_y'].s)
        results['sigma_vz'].append(best_fit.uvars['vertex_z'].s)
        results['sigma_brho'].append(best_fit.uvars['brho'].s)
        results['sigma_polar'].append(best_fit.uvars['polar'].s)
        results['sigma_azimuthal'].append(best_fit.uvars['azimuthal'].s)
    else:
        results['sigma_vx'].append(1.0e6)
        results['sigma_vy'].append(1.0e6)
        results['sigma_vz'].append(1.0e6)
        results['sigma_brho'].append(1.0e6)
        results['sigma_polar'].append(1.0e6)
        results['sigma_azimuthal'].append(1.0e6)