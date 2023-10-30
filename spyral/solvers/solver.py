from .guess import Guess
from ..core.config import DetectorParameters
from ..core.target import Target
from ..core.nuclear_data import NucleusData
from ..core.cluster import Cluster
from ..core.estimator import Direction
from ..core.constants import MEV_2_JOULE, MEV_2_KG, QBRHO_2_P, C, E_CHARGE

from scipy import integrate
import numpy as np
import math
from dataclasses import dataclass
from lmfit import Parameters, minimize, fit_report
from lmfit.minimizer import MinimizerResult

TIME_WINDOW: float = 0.1e-6 #1us window
INTERVAL: float = 0.5e-9 #0.1 ns sample
SAMPLING_PERIOD: float = 1.0e-6/512 # seconds, converts time bucket interval to time
SAMPLING_RANGE: np.ndarray = np.linspace(0., TIME_WINDOW, int(TIME_WINDOW/INTERVAL))
PRECISION: float = 2.0e-6

#State = [x, y, z, vx, vy, vz]
#Derivative = [vx, vy, vz, ax, ay, az] (returns)
def equation_of_motion(t: float, state: np.ndarray, Bfield: float, Efield: float, target: Target, ejectile: NucleusData) -> np.ndarray:

    speed = math.sqrt(state[3]**2.0 + state[4]**2.0 + state[5]**2.0)
    unit_vector = state[3:] / speed # direction
    kinetic_energy = ejectile.mass * (1.0/math.sqrt(1.0 - (speed / C)**2.0) - 1.0) #MeV
    if kinetic_energy < PRECISION:
        return np.zeros(6)
    mass_kg = ejectile.mass * MEV_2_KG
    charge_c = ejectile.Z * E_CHARGE
    qm = charge_c/mass_kg

    deceleration = (target.get_dedx(ejectile, kinetic_energy) * MEV_2_JOULE * target.density * 100.0) / mass_kg
    results = np.zeros(6)
    results[0] = state[3]
    results[1] = state[4]
    results[2] = state[5]
    results[3] = qm * state[4] * Bfield - deceleration * unit_vector[0]
    results[4] = qm * (-1.0 * state[3] * Bfield) - deceleration * unit_vector[1]
    results[5] = qm * Efield - deceleration * unit_vector[2]

    return results

def jacobian(t, state: np.ndarray, Bfield: float, Efield: float, target: Target, ejectile: NucleusData) -> np.ndarray:
    jac = np.zeros((len(state), len(state)))
    mass_kg = ejectile.mass * MEV_2_KG
    charge_c = ejectile.Z * E_CHARGE
    qm = charge_c/mass_kg
    jac[0, 3] = 1.0
    jac[1, 4] = 1.0
    jac[2, 5] = 1.0
    jac[3, 4] = qm * Bfield
    jac[4, 3] = -1.0 * qm * Bfield

    return jac

def get_sampling_steps(traj_data: np.ndarray, vertexX: float, vertexY: float, vertexZ: float) -> np.ndarray:
    steps = np.zeros(len(traj_data))
    for idx, point in enumerate(traj_data):
        if idx == 0:
            steps[idx] = math.sqrt((point[0] - vertexX)**2.0 + (point[1] - vertexY)**2.0 + (point[2] - vertexZ)**2.0)
        else:
            steps[idx] = steps[idx-1] + np.linalg.norm(point - traj_data[idx-1])
    return steps

def generate_trajectory(fit_params: Parameters, Bfield: float, Efield: float, target: Target, ejectile: NucleusData) -> np.ndarray:
    #Convert guessed parameters into initial values for ODE x, v
    initial_value = np.zeros(6)
    initial_value[0] = fit_params['vertex_x'].value
    initial_value[1] = fit_params['vertex_y'].value
    initial_value[2] = fit_params['vertex_z'].value
    momentum = QBRHO_2_P * (fit_params['brho'].value * float(ejectile.Z))
    speed = momentum / ejectile.mass * C
    initial_value[3] = speed * math.sin(fit_params['polar'].value) * math.cos(fit_params['azimuthal'].value)
    initial_value[4] = speed * math.sin(fit_params['polar'].value) * math.sin(fit_params['azimuthal'].value)
    initial_value[5] = speed * math.cos(fit_params['polar'].value)

    #print(f'guess: {initial_value}')
    result = integrate.solve_ivp(equation_of_motion, (0.0, TIME_WINDOW), initial_value, method='BDF', args=(Bfield, Efield, target, ejectile), t_eval=SAMPLING_RANGE, jac=jacobian)
    positions: np.ndarray = result.y.T[:, :3]
    return positions


def objective_function(fit_params: Parameters, x: np.ndarray, Bfield: float, Efield: float, target: Target, ejectile: NucleusData, direction: Direction) -> np.ndarray:
    trajectory = generate_trajectory(fit_params, Bfield, Efield, target, ejectile)
    errors = np.zeros(len(x))

    valid_traj: np.ndarray | None = None
    if direction is Direction.FORWARD:
        max_z = np.max(x[:, 2])
        valid_traj = trajectory[trajectory[:, 2] < max_z]
    else:
        min_z = np.min(x[:, 1])
        valid_traj = trajectory[trajectory[:, 2] > min_z]

    if len(valid_traj) == 0:
        errors += 1.0e6 #This was a super bad guess resulting in a trajectory going the wrong way, happens with gross bad events sometimes
        return errors
    
    for idx, point in enumerate(x):
        # errors[idx] = np.linalg.norm(valid_traj[np.argmin(valid_traj[:, 2] - point[2]), :2] - point[:2])
        errors[idx] = np.min(np.linalg.norm(valid_traj - point, axis=1))
    return errors

def create_params(initial_value: Guess, ejectile: NucleusData, data: np.ndarray) -> Parameters:
    momentum = 1.0 * ejectile.mass
    phys_max_brho = 0.5 * momentum / QBRHO_2_P / ejectile.Z # set upper limit at v=0.5c

    uncertainty_position = 0.25
    uncertainty_brho = 1.0

    min_brho = initial_value.brho - initial_value.brho * uncertainty_brho * 2.0
    if min_brho < 0.0:
        min_brho = 0.0
    max_brho = initial_value.brho + initial_value.brho * uncertainty_brho * 2.0
    if max_brho > phys_max_brho:
        max_brho = phys_max_brho

    max_polar = np.pi
    if initial_value.direction is Direction.FORWARD:
        max_polar = np.pi * 0.5
    min_polar = 0.0
    if initial_value.direction is Direction.BACKWARD:
        min_polar = np.pi * 0.5

    min_x = initial_value.vertex_x * 0.001 - initial_value.vertex_x * 0.001 * uncertainty_position * 2.0
    max_x = initial_value.vertex_x * 0.001 + initial_value.vertex_x * 0.001 * uncertainty_position * 2.0
    min_y = initial_value.vertex_y * 0.001 - initial_value.vertex_y * 0.001 * uncertainty_position * 2.0
    max_y = initial_value.vertex_y * 0.001 + initial_value.vertex_y * 0.001 * uncertainty_position * 2.0
    min_z = initial_value.vertex_z * 0.001 - initial_value.vertex_z * 0.001 * uncertainty_position * 2.0
    max_z = initial_value.vertex_z * 0.001 + initial_value.vertex_z * 0.001 * uncertainty_position * 2.0

    if initial_value.direction is Direction.FORWARD and max_z > np.max(data[:, 2]):
        max_z = np.max(data[:, 2])
    elif initial_value.direction is Direction.BACKWARD and min_z < np.min(data[:, 2]):
        min_z = np.min(data[:, 2])

    fit_params = Parameters()
    fit_params.add('brho', initial_value.brho, min=min_brho, max=max_brho)
    fit_params.add('polar', initial_value.polar, min=min_polar, max=max_polar)
    fit_params.add('azimuthal', initial_value.azimuthal, min=0.0, max=2.0*np.pi)
    fit_params.add('vertex_x', initial_value.vertex_x * 0.001, min=min_x, max=max_x, vary=True)
    fit_params.add('vertex_y', initial_value.vertex_y * 0.001, min=min_y, max=max_y, vary=True)
    fit_params.add('vertex_z', initial_value.vertex_z * 0.001, min=min_z, max=max_z, vary=True)
    return fit_params


#For testing, not for use in production
def fit_model(cluster: Cluster, initial_value: Guess, detector_params: DetectorParameters, target: Target, ejectile: NucleusData) -> Parameters:
    traj_data = cluster.data[:, :3] * 0.001
    fit_params = create_params(initial_value, ejectile, traj_data)

    bfield = -1.0 * detector_params.magnetic_field
    efield = -1.0 * detector_params.electric_field

    result: MinimizerResult = minimize(objective_function, fit_params, args = (traj_data, bfield, efield, target, ejectile, initial_value.direction))
    print(fit_report(result))

    return result.params
        

def solve_physics(cluster_index: int, cluster: Cluster, initial_value: Guess, detector_params: DetectorParameters, target: Target, ejectile: NucleusData, results: dict[str, list]):
    traj_data = cluster.data[:, :3] * 0.001
    fit_params = create_params(initial_value, ejectile, traj_data)

    bfield = -1.0 * detector_params.magnetic_field
    efield = -1.0 * detector_params.electric_field

    best_fit: MinimizerResult = minimize(objective_function, fit_params, args = (traj_data, bfield, efield, target, ejectile, initial_value.direction))

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