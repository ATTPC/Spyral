from .config import DetectorParameters
from .target import Target
from .nuclear_data import NucleusData
from .clusterize import ClusteredCloud
from .estimator import Direction
from .constants import MEV_2_JOULE, MEV_2_KG
from scipy import optimize, integrate, constants
import numpy as np
import math
from dataclasses import dataclass
from lmfit import Parameters, minimize
from lmfit.minimizer import MinimizerResult

@dataclass
class InitialValue:
    polar: float = 0.0 #radians
    azimuthal: float = 0.0 #radians
    brho: float = 0.0 #Tm
    vertex_x: float = 0.0 #mm
    vertex_y: float = 0.0 #mm
    vertex_z: float = 0.0 #mm
    direction: Direction = Direction.NONE 

    def convert_to_array(self) -> np.ndarray:
        return np.array([self.polar, self.azimuthal, self.brho, self.vertex_x, self.vertex_y, self.vertex_z])

TIME_WINDOW: float = 0.1e-6 #1us window
INTERVAL: float = 0.5e-9 #0.1 ns sample
QBRHO_2_P: float = 1.0e-9 * constants.speed_of_light #kG * cm -> MeV
SAMPLING_PERIOD: float = 1.0e-6/512 # seconds, converts time bucket interval to time
SAMPLING_RANGE: np.ndarray = np.linspace(0., TIME_WINDOW, int(TIME_WINDOW/INTERVAL))
PRECISION: float = 2.0e-6

#State = [x, y, z, vx, vy, vz]
#Derivative = [vx, vy, vz, ax, ay, az] (returns)
def equation_of_motion(t: float, state: np.ndarray, Bfield: float, Efield: float, target: Target, ejectile: NucleusData) -> np.ndarray:

    speed = math.sqrt(state[3]**2.0 + state[4]**2.0 + state[5]**2.0)
    unit_vector = state[3:] / speed # direction
    kinetic_energy = ejectile.mass * (1.0/math.sqrt(1.0 - (speed / constants.speed_of_light)**2.0) - 1.0) #MeV
    if kinetic_energy < PRECISION:
        return np.zeros(6)
    mass_kg = ejectile.mass * MEV_2_KG
    charge_c = ejectile.Z * constants.elementary_charge
    qm = charge_c/mass_kg

    deceleration = (target.get_dedx(ejectile, kinetic_energy) * MEV_2_JOULE * target.data.density * 100.0) / mass_kg
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
    charge_c = ejectile.Z * constants.elementary_charge
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

def generate_trajectory(fit_params: np.ndarray, Bfield: float, Efield: float, target: Target, ejectile: NucleusData) -> np.ndarray:
    #Convert guessed parameters into initial values for ODE x, v
    initial_value = np.zeros(6)
    initial_value[:3] = fit_params[3:]
    momentum = QBRHO_2_P * (fit_params[2] * 10.0 * 100.0 * float(ejectile.Z))
    speed = momentum / ejectile.mass * constants.speed_of_light
    initial_value[3] = speed * math.sin(fit_params[0]) * math.cos(fit_params[1])
    initial_value[4] = speed * math.sin(fit_params[0]) * math.sin(fit_params[1])
    initial_value[5] = speed * math.cos(fit_params[0])

    result = integrate.solve_ivp(equation_of_motion, (0.0, TIME_WINDOW), initial_value, method='BDF', args=(Bfield, Efield, target, ejectile), t_eval=SAMPLING_RANGE, jac=jacobian)
    positions: np.ndarray = result.y.T[:, :3]
    return positions

def objective_function(fit_params: np.ndarray, data: np.ndarray, Bfield: float, Efield: float, target: Target, ejectile: NucleusData) -> float:
    trajectory = generate_trajectory(fit_params, Bfield, Efield, target, ejectile)
    error = 0.0
    for point in data:
        error += np.min(np.linalg.norm(trajectory - point, axis=1))
    return error / len(data)

def generate_trajectory_lm(fit_params: Parameters, Bfield: float, Efield: float, target: Target, ejectile: NucleusData) -> np.ndarray:
    #Convert guessed parameters into initial values for ODE x, v
    initial_value = np.zeros(6)
    initial_value[0] = fit_params['vertex_x'].value
    initial_value[1] = fit_params['vertex_y'].value
    initial_value[2] = fit_params['vertex_z'].value
    momentum = QBRHO_2_P * (fit_params['brho'].value * 10.0 * 100.0 * float(ejectile.Z))
    speed = momentum / ejectile.mass * constants.speed_of_light
    initial_value[3] = speed * math.sin(fit_params['polar'].value) * math.cos(fit_params['azimuthal'].value)
    initial_value[4] = speed * math.sin(fit_params['polar'].value) * math.sin(fit_params['azimuthal'].value)
    initial_value[5] = speed * math.cos(fit_params['polar'].value)

    #print(f'guess: {initial_value}')
    result = integrate.solve_ivp(equation_of_motion, (0.0, TIME_WINDOW), initial_value, method='BDF', args=(Bfield, Efield, target, ejectile), t_eval=SAMPLING_RANGE, jac=jacobian)
    positions: np.ndarray = result.y.T[:, :3]
    return positions


def objective_function_lm(fit_params: Parameters, x: np.ndarray, Bfield: float, Efield: float, target: Target, ejectile: NucleusData) -> np.ndarray:
    trajectory = generate_trajectory_lm(fit_params, Bfield, Efield, target, ejectile)
    errors = np.zeros(len(x))
    max_z = np.max(x[:, 2])
    valid_traj = trajectory[trajectory[:, 2] < max_z]
    for idx, point in enumerate(x):
        errors[idx] = np.min(np.linalg.norm(valid_traj - point, axis=1))
    return errors

def create_params(initial_value: InitialValue, ejectile: NucleusData, data: np.ndarray) -> Parameters:
    momentum = 1.0 * ejectile.mass
    phys_max_brho = 0.5 * momentum / QBRHO_2_P / (ejectile.Z * 10.0 * 100.0) # set upper limit at v=0.5c

    uncertainty_position = 0.1
    uncertainty_brho = 0.25

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

    if max_z > np.max(data[:, 2]):
        max_z = np.max(data[:, 2])

    fit_params = Parameters()
    fit_params.add('brho', initial_value.brho, min=min_brho, max=max_brho)
    fit_params.add('polar', initial_value.polar, min=min_polar, max=max_polar)
    fit_params.add('azimuthal', initial_value.azimuthal, min=0.0, max=2.0*np.pi)
    fit_params.add('vertex_x', initial_value.vertex_x * 0.001, min=min_x, max=max_x, vary=True)
    fit_params.add('vertex_y', initial_value.vertex_y * 0.001, min=min_y, max=max_y, vary=True)
    fit_params.add('vertex_z', initial_value.vertex_z * 0.001, min=min_z, max=max_z, vary=True)
    return fit_params


def run_lm(cluster: ClusteredCloud, initial_value: InitialValue, detector_params: DetectorParameters, target: Target, ejectile: NucleusData) -> dict[str, float]:
    traj_data = cluster.point_cloud.cloud[:, :3] * 0.001
    fit_params = create_params(initial_value, ejectile, traj_data)

    bfield = -1.0 * detector_params.magnetic_field
    efield = -1.0 * detector_params.electric_field

    result: MinimizerResult = minimize(objective_function_lm, fit_params, args = (traj_data, bfield, efield, target, ejectile))

    return result.params.valuesdict()
        

def solve_physics(cluster_index: int, cluster: ClusteredCloud, initial_value: InitialValue, detector_params: DetectorParameters, target: Target, ejectile: NucleusData, results: dict[str, list]):
    Efield = -1.0 * detector_params.electric_field
    Bfield = -1.0 * detector_params.magnetic_field
    
    best_fit = optimize.minimize(objective_function, initial_value.convert_to_array(), args=(cluster.point_cloud.cloud[:, :3], Bfield, Efield, target, ejectile))

    results['event'].append(cluster.point_cloud.event_number)
    results['cluster_index'].append(cluster_index)
    results['cluster_label'].append(cluster.label)
    results['vertex_x'].append(best_fit.x[3])
    results['vertex_y'].append(best_fit.x[4])
    results['vertex_z'].append(best_fit.x[5])
    results['brho'].append(best_fit.x[2])
    results['polar'].append(best_fit.x[0])
    results['azimuthal'].append(best_fit.x[1])