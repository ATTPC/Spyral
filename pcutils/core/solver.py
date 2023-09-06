from .config import DetectorParameters
from .target import Target
from .nuclear_data import NucleusData
from .clusterize import ClusteredCloud
from .estimator import Direction
from .constants import MEV_2_JOULE, MEV_2_KG
from scipy import optimize, integrate, constants, linalg
import numpy as np
import math
from dataclasses import dataclass
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
from functools import partial

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

TIME_WINDOW: float = 1.0e-6 #1us window
INTERVAL: float = 1.0e-10 #0.1 ns sample
QBRHO_2_P: float = 1.0e-9 * constants.speed_of_light #kG * cm -> MeV
SAMPLING_PERIOD: float = 1.0e-6/512 # seconds, converts time bucket interval to time
SAMPLING_RANGE: np.ndarray = np.linspace(0., TIME_WINDOW, int(TIME_WINDOW/INTERVAL))

def calculate_decel(speed: float, target: Target, ejectile: NucleusData) -> tuple[float, float]:
    kinetic_energy = ejectile.mass * (1.0/math.sqrt(1.0 - (speed / constants.speed_of_light)**2.0) - 1.0) #MeV
    charge = ejectile.Z * constants.elementary_charge #Coulombs
    mass_kg = ejectile.mass * MEV_2_KG #kg

    dEdx = target.get_dedx(ejectile, kinetic_energy) #MeV/g/cm^2
    dEdx *= MEV_2_JOULE #J/g/cm^2
    force = dEdx * target.data.density * 100.0 # J/cm * cm/m = J/m = kg m/s^2
    deceleration = force / (mass_kg)
    qm = charge / mass_kg
    return (deceleration, qm)

def hx(x: np.ndarray) -> np.ndarray:
    return np.array([x[0], x[1], x[2]])

def apply_kalman_filter(data: np.ndarray, fit_params: np.ndarray, detector_params: DetectorParameters, target: Target, ejectile: NucleusData) -> np.ndarray:
    Efield = -1.0 * detector_params.electric_field
    Bfield = -1.0 * detector_params.magnetic_field
    initial_state = np.zeros(6)
    momentum = QBRHO_2_P * (fit_params[2] * 10.0 * 100.0 * float(ejectile.Z))
    speed = momentum / ejectile.mass * constants.speed_of_light

    initial_state[:3] = fit_params[3:]
    initial_state[3] = speed * math.sin(fit_params[0]) * math.cos(fit_params[1])
    initial_state[4] = speed * math.sin(fit_params[0]) * math.sin(fit_params[1])
    initial_state[5] = speed * math.cos(fit_params[0])

    def fx(x: np.ndarray, ds: float) -> np.ndarray:
        speed = math.sqrt(x[3]**2.0 + x[4]**2.0 + x[5]**2.0)
        dt = ds/speed
        unit_vector = x[3:] / (speed)# direction
        deceleration, qm = calculate_decel(speed, target, ejectile)
        accel = np.array([qm * (x[4] * Bfield) - deceleration * unit_vector[0],
                          qm * ( -1.0 * x[3] * Bfield) - deceleration * unit_vector[1],
                          qm * Efield - deceleration * unit_vector[2]])

        transition = np.array( [[1., 0., 0., dt, 0., 0.],
                                [0., 1., 0., 0., dt, 0.],
                                [0., 0., 1., 0., 0., dt],
                                [0., 0., 0., 1., 0., 0.],
                                [0., 0., 0., 0., 1., 0.],
                                [0., 0., 0., 0., 0., 1.],
                               ])
        offset = np.array([accel[0]*0.5*dt**2.0, accel[1]*0.5*dt**2.0, accel[2]*0.5*dt**2.0, accel[0]*dt, accel[1]*dt, accel[2]*dt])
        return np.dot(transition, x) + offset


    first_ds = np.linalg.norm(initial_state[:3] - data[0, :])
    ds_set = [np.linalg.norm(point[:3] - data[idx-1, :]) for idx, point in enumerate(data[1:])]
    ds_set.insert(0, first_ds)
    points = MerweScaledSigmaPoints(6, alpha=.1, beta=2., kappa=-3)
    k_filter = UnscentedKalmanFilter(6, 3, 0.0005, hx, fx, points)
    sigma_speed = 0.1 * speed
    sigma_pos = 0.0005
    k_filter.P = np.diag([(10.0*sigma_pos)**2.0, (10.0*sigma_pos)**2.0, (10.0*sigma_pos)**2.0, sigma_speed**2.0, sigma_speed**2.0, sigma_speed**2.0])
    k_filter.x = initial_state
    k_filter.R = np.diag([sigma_pos**2.0, sigma_pos**2.0, sigma_pos**2.0])
    k_filter.Q = Q_discrete_white_noise(dim=2, dt=0.0005, var=(sigma_speed)**2.0, block_size=3, order_by_dim=False)
    (means, covs) = k_filter.batch_filter(data, dts=ds_set)
    trajectory, traj_cov, _ = k_filter.rts_smoother(means, covs, dts=ds_set)
    return trajectory

def optimize_kalman_filter(fit_params: np.ndarray, data: np.ndarray, detector_params: DetectorParameters, target: Target, ejectile: NucleusData) -> float:
    trajectory = apply_kalman_filter(data, fit_params, detector_params, target, ejectile)
    return np.average(np.linalg.norm(data[:, :] - trajectory[:, :3], axis=1))

def solve_physics_kalman(cluster_index: int, cluster: ClusteredCloud, initial_value: InitialValue, detector_params: DetectorParameters, target: Target, ejectile: NucleusData, results: dict[str, list]):
    cluster.point_cloud.drop_isolated_points()
    cluster.point_cloud.smooth_cloud()
    cluster.point_cloud.sort_in_z()
    #We need to order in time of the ejectile
    if initial_value.direction == Direction.BACKWARD:
        cluster.point_cloud.cloud = np.flip(cluster.point_cloud.cloud, axis=0)

    #Convert everyting into meters
    cluster.point_cloud.cloud[:, :3] *= 0.001
    initial_value.vertex_x *= 0.001
    initial_value.vertex_y *= 0.001
    initial_value.vertex_z *= 0.001

    best_fit = optimize.minimize(optimize_kalman_filter, initial_value.convert_to_array(), args=(cluster.point_cloud.cloud[:, :3], detector_params, target, ejectile))

#State = [x, y, z, vx, vy, vz]
#Derivative = [vx, vy, vz, ax, ay, az] (returns)
def equation_of_motion(t: float, state: np.ndarray, Bfield: float, Efield: float, target: Target, ejectile: NucleusData) -> np.ndarray:

    speed = math.sqrt(state[3]**2.0 + state[4]**2.0 + state[5]**2.0)
    unit_vector = state[3:] / speed # direction
    kinetic_energy = ejectile.mass * (1.0/math.sqrt(1.0 - (speed / constants.speed_of_light)**2.0) - 1.0) #MeV
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

def get_sampling_steps(cluster: ClusteredCloud, vertexX: float, vertexY: float, vertexZ: float) -> np.ndarray:
    steps = np.zeros(len(cluster.point_cloud.cloud))
    for idx, point in enumerate(cluster.point_cloud.cloud):
        if idx == 0:
            steps[idx] = math.sqrt((point[0] - vertexX)**2.0 + (point[1] - vertexY)**2.0 + (point[2] - vertexZ)**2.0)
        else:
            steps[idx] = steps[idx-1] + np.linalg.norm(point[:3] - cluster.point_cloud.cloud[idx-1, :3])
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