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
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise

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

def calculate_decel(speed: float, target: Target, ejectile: NucleusData) -> tuple[float, float]:
    print(f'speed: {speed/constants.speed_of_light}')
    kinetic_energy = ejectile.mass * (1.0/math.sqrt(1.0 - (speed / constants.speed_of_light)**2.0) - 1.0) #MeV
    charge = ejectile.Z * constants.elementary_charge #Coulombs
    mass_kg = ejectile.mass * MEV_2_KG #kg

    dEdx = target.get_dedx(ejectile, kinetic_energy) #MeV/g/cm^2
    dEdx *= MEV_2_JOULE #J/g/cm^2
    force = dEdx * target.data.density * 100.0 # J/cm * cm/m = J/m = kg m/s^2
    deceleration = force / (mass_kg)
    qm = charge / mass_kg
    return (deceleration, qm)

# def fx(x: np.ndarray, dt: float, Bfield: np.ndarray, Efield: np.ndarray, target: Target, ejectile: NucleusData) -> np.ndarray:
#     speed = np.linalg.norm(x[3:6])
#     unit_vector = x[3:] / (speed)# direction
#     deceleration, qm = calculate_decel(speed, target, ejectile)
#     transition = np.array( [[1., 0., 0., dt, 0., 0., 0.5*dt**2.0, 0., 0.],
#                             [0., 1., 0., 0., dt, 0., 0., 0.5*dt**2.0, 0., 0.],
#                             [0., 0., 1., 0., 0., dt, 0., 0., 0.5*dt**2.0],
#                             [0., 0., 0., 1., 0., 0., dt, 0., 0.],
#                             [0., 0., 0., 0., 1., 0., 0., dt, 0.],
#                             [0., 0., 0., 0., 0., 1., 0., 0., dt],
#                             [0., 0., 0., 0., qm * Bfield[2],  -1.0 * qm * Bfield[1], 0., 0., 0.],
#                             [0., 0., 0., -1.0 * qm * Bfield[2], 0., 1.0, qm *Bfield[0], 0., 0., 0.],
#                             [0., 0., 0., qm * Bfield[1], -1.0 * qm * Bfield[0], 0., 0., 0., 0.],
#                            ])
#     constant = np.array([0., 0., 0., 0., 0., 0., qm * Efield[0] - deceleration * unit_vector[0], qm * Efield[1] - deceleration * unit_vector[1], qm * Efield[2] - deceleration * unit_vector[2]])
#     return np.dot(transition, x) + constant

def hx(x: np.ndarray) -> np.ndarray:
    return np.array([x[0], x[1], x[2]])

def apply_kalman_filter(cluster: ClusteredCloud, initial_value: InitialValue, detector_params: DetectorParameters, target: Target, ejectile: NucleusData) -> np.ndarray:
    cluster.point_cloud.drop_isolated_points()
    cluster.point_cloud.smooth_cloud()
    cluster.point_cloud.sort_in_z()
    #We need to order in time of the ejectile
    if initial_value.direction == Direction.BACKWARD:
        cluster.point_cloud.cloud = np.flip(cluster.point_cloud.cloud, axis=0)

    E_vec = np.array([0.0, 0.0, -1.0 * detector_params.electric_field])
    B_vec = np.array([0.0, -1.0 * detector_params.magnetic_field * math.sin(detector_params.tilt_angle), -1.0 * detector_params.magnetic_field * math.cos(detector_params.tilt_angle)])
    fit_params = initial_value.convert_to_array()
    initial_state = np.zeros(9)
    initial_state[:3] = fit_params[3:] * 0.001 # vertex, convert to m
    momentum = QBRHO_2_P * (fit_params[2] * 10.0 * 100.0 * float(ejectile.Z))
    speed = momentum / ejectile.mass * constants.speed_of_light
    decel, qm = calculate_decel(speed, target, ejectile)

    initial_state[3] = speed * math.sin(fit_params[0]) * math.cos(fit_params[1])
    initial_state[4] = speed * math.sin(fit_params[0]) * math.sin(fit_params[1])
    initial_state[5] = speed * math.cos(fit_params[0])
    initial_state[6] = qm * (E_vec[0] + initial_state[4] * B_vec[2] - initial_state[5] * B_vec[1]) - decel * math.sin(fit_params[0]) * math.cos(fit_params[1])
    initial_state[7] = qm * (E_vec[1] - initial_state[3] * B_vec[2] + initial_state[5] * B_vec[0]) - decel * math.sin(fit_params[0]) * math.sin(fit_params[1])
    initial_state[8] = qm * (E_vec[2] + initial_state[3] * B_vec[1] - initial_state[4] * B_vec[0]) - decel * math.cos(fit_params[0])

    def fx(x: np.ndarray, dt: float) -> np.ndarray:
        print(f'state at call: {x}')
        speed = np.linalg.norm(x[3:6])
        unit_vector = x[3:6] / (speed)# direction
        deceleration, qm = calculate_decel(speed, target, ejectile)
        transition = np.array( [[1., 0., 0., dt, 0., 0., 0.5*dt**2.0, 0., 0.],
                                [0., 1., 0., 0., dt, 0., 0., 0.5*dt**2.0, 0.],
                                [0., 0., 1., 0., 0., dt, 0., 0., 0.5*dt**2.0],
                                [0., 0., 0., 1., 0., 0., dt, 0., 0.],
                                [0., 0., 0., 0., 1., 0., 0., dt, 0.],
                                [0., 0., 0., 0., 0., 1., 0., 0., dt],
                                [0., 0., 0., 0., qm * B_vec[2],  -1.0 * qm * B_vec[1], 0., 0., 0.],
                                [0., 0., 0., -1.0 * qm * B_vec[2], 0., qm *B_vec[0], 0., 0., 0.],
                                [0., 0., 0., qm * B_vec[1], -1.0 * qm * B_vec[0], 0., 0., 0., 0.],
                               ])
        constant = np.array([0., 0., 0., 0., 0., 0., qm * E_vec[0] - deceleration * unit_vector[0], qm * E_vec[1] - deceleration * unit_vector[1], qm * E_vec[2] - deceleration * unit_vector[2]])
        return np.dot(transition, x) + constant


    vertex_bucket = detector_params.window_time_bucket - initial_value.vertex_z / detector_params.detector_length * (detector_params.window_time_bucket - detector_params.micromegas_time_bucket)
    first_dt = np.abs(vertex_bucket - cluster.point_cloud.cloud[0, 6]) * SAMPLING_PERIOD * 0.001
    dts = [np.abs(cluster.point_cloud.cloud[idx-1, 6] - point[6]) * SAMPLING_PERIOD * 0.001 for idx, point in enumerate(cluster.point_cloud.cloud[1:])]
    dts.insert(0, first_dt)
    points = MerweScaledSigmaPoints(9, alpha=.1, beta=2., kappa=-6)
    k_filter = UnscentedKalmanFilter(9, 3, SAMPLING_PERIOD, hx, fx, points)
    k_filter.P = np.diag([0.5**2.0, 0.5**2.0, 0.5**2.0, speed**2.0, speed**2.0, speed**2.0, speed**2.0, speed**2.0, speed**2.0])
    k_filter.x = initial_state
    k_filter.R = np.diag([0.5**2.0, 0.5**2.0, 0.5**2.0])
    k_filter.Q = Q_discrete_white_noise(dim=3, dt=SAMPLING_PERIOD * 0.001, var=0.25, block_size=3, order_by_dim=False)
    print(k_filter)
    #(means, covs) = k_filter.batch_filter(cluster.point_cloud.cloud[:, :3], dts=dts)
    for index, dt in enumerate(dts):
        k_filter.predict(dt)
        print(f'state after predict: {k_filter.x}')
        print(f'covariance after predict: {k_filter.P}')
        k_filter.update(cluster.point_cloud.cloud[index, :3])
    trajectory, traj_cov = k_filter.rts_smoother(means, covs)
    return trajectory * 1000.0 #convert back to mm

#State = [x, y, z, vx, vy, vz]
#Derivative = [vx, vy, vz, ax, ay, az] (returns)
def equation_of_motion(t: float, state: np.ndarray, Bfield: np.ndarray, Efield: np.ndarray, target: Target, ejectile: NucleusData) -> np.ndarray:

    speed = np.linalg.norm(state[3:])
    unit_vector = state[3:] / (speed)# direction
    kinetic_energy = ejectile.mass * (1.0/math.sqrt(1.0 - (speed / constants.speed_of_light)**2.0) - 1.0) #MeV
    charge = ejectile.Z * constants.elementary_charge #Coulombs
    mass_kg = ejectile.mass * MEV_2_KG #kg

    dEdx = target.get_dedx(ejectile, kinetic_energy) #MeV/g/cm^2
    dEdx *= MEV_2_JOULE #J/g/cm^2
    force = dEdx * target.data.density * 100.0 # J/cm * cm/m = J/m = kg m/s^2
    deceleration = force / (mass_kg)
    result = np.zeros(len(state))
    qm = charge / mass_kg

    #v = v
    result[0] = state[3]
    result[1] = state[4]
    result[2] = state[5]
    # a = q/m * (E + v x B) - stopping
    result[3] = qm * (Efield[0] + state[4] * Bfield[2] - state[5] * Bfield[1]) - deceleration * unit_vector[0]
    result[4] = qm * (Efield[1] - state[3] * Bfield[2] + state[5] * Bfield[0]) - deceleration * unit_vector[1]
    result[5] = qm * (Efield[2] + state[3] * Bfield[1] - state[4] * Bfield[0]) - deceleration * unit_vector[2]
    #result[3:] = (charge/mass_kg * (Efield + np.cross(state[3:], Bfield)) - deceleration * unit_vector)

    return result

def generate_trajectory(fit_params: np.ndarray, Bfield: np.ndarray, Efield: np.ndarray, target: Target, ejectile: NucleusData) -> np.ndarray:
    #Convert guessed parameters into initial values for ODE x, v
    initial_value = np.zeros(6)
    initial_value[:3] = fit_params[3:] * 0.001 # vertex, convert to m
    momentum = QBRHO_2_P * (fit_params[2] * 10.0 * 100.0 * float(ejectile.Z))
    speed = momentum / ejectile.mass * constants.speed_of_light
    initial_value[3] = speed * math.sin(fit_params[0]) * math.cos(fit_params[1])
    initial_value[4] = speed * math.sin(fit_params[0]) * math.sin(fit_params[1])
    initial_value[5] = speed * math.cos(fit_params[0])

    result = integrate.solve_ivp(equation_of_motion, (0.0, 1.0e-6), initial_value, args=(Bfield, Efield, target, ejectile), t_eval=np.linspace(0.,TIME_WINDOW, int(TIME_WINDOW/INTERVAL)), max_step=0.1)
    positions: np.ndarray = result.y.T[:, :3] * 1000.0 # convert back to mm to match data

    return positions

def objective_function(fit_params: np.ndarray, data: np.ndarray, Bfield: np.ndarray, Efield: np.ndarray, target: Target, ejectile: NucleusData) -> float:
    trajectory = generate_trajectory(fit_params, Bfield, Efield, target, ejectile)
    error = 0.0
    for point in data:
        error += np.min(np.linalg.norm(trajectory - point, axis=1))
    return error / len(data)
        

def solve_physics(cluster_index: int, cluster: ClusteredCloud, initial_value: InitialValue, detector_params: DetectorParameters, target: Target, ejectile: NucleusData, results: dict[str, list]):
    E_vec = np.array([0.0, 0.0, -1.0 * detector_params.electric_field])
    B_vec = np.array([0.0, -1.0 * detector_params.magnetic_field * math.sin(detector_params.tilt_angle), -1.0 * detector_params.magnetic_field * math.cos(detector_params.tilt_angle)])

    best_fit = optimize.minimize(objective_function, initial_value.convert_to_array(), args=(cluster.point_cloud.cloud[:, :3], B_vec, E_vec, target, ejectile))

    results['event'].append(cluster.point_cloud.event_number)
    results['cluster_index'].append(cluster_index)
    results['cluster_label'].append(cluster.label)
    results['vertex_x'].append(best_fit.x[3])
    results['vertex_y'].append(best_fit.x[4])
    results['vertex_z'].append(best_fit.x[5])
    results['brho'].append(best_fit.x[2])
    results['polar'].append(best_fit.x[0])
    results['azimuthal'].append(best_fit.x[1])