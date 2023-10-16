from .target import Target
from .nuclear_data import NucleusData
from .kalman_args import get_kalman_args, set_kalman_args
from .constants import MEV_2_KG, MEV_2_JOULE
from .config import DetectorParameters
from .cluster import Cluster
from .estimator import Direction

from scipy import constants
import numpy as np
import math
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
from dataclasses import dataclass, field

QBRHO_2_P: float = 1.0e-9 * constants.speed_of_light * 100.0 * 10.0 #T * m -> MeV/c

@dataclass
class Guess:
    vertex_x: float = 0.0
    vertex_y: float = 0.0
    vertex_z: float = 0.0
    brho: float = 0.0
    polar: float = 0.0
    azimuthal: float = 0.0
    direction: Direction = field(default=-1)

def fx(x: np.ndarray, dz: float) -> np.ndarray:
        args = get_kalman_args()
        speed = math.sqrt(x[3]**2.0 + x[4]**2.0 + x[5]**2.0)
        dt = dz/x[5]
        unit_vector = x[3:] / (speed)# direction
        momentum = speed/constants.speed_of_light * args.ejectile.mass
        kinetic_energy = math.sqrt(momentum ** 2.0 + args.ejectile.mass ** 2.0) - args.ejectile.mass #MeV
        if kinetic_energy < 0.001:
             return x
        charge = args.ejectile.Z * constants.elementary_charge #Coulombs
        mass_kg = args.ejectile.mass * MEV_2_KG #kg
        deceleration = args.target.get_dedx(args.ejectile, kinetic_energy) * MEV_2_JOULE \
                        * args.target.data.density() * 100.0 / mass_kg # m/s^2
        qm = charge / mass_kg
        accel = np.array([- deceleration * unit_vector[0],
                          - deceleration * unit_vector[1],
                          qm * args.efield - deceleration * unit_vector[2]])

        transition = np.array( [[1., 0., 0., dt, 0., 0.],
                                [0., 1., 0., 0., dt, 0.],
                                [0., 0., 1., 0., 0., dt],
                                [0., 0., 0., 1., qm*args.bfield*dt, 0.],
                                [0., 0., 0., -1.0*qm*args.bfield*dt, 1., 0.],
                                [0., 0., 0., 0., 0., 1.],
                               ])
        offset = np.array([0.0, 0.0, 0.0, accel[0]*dt, accel[1]*dt, accel[2]*dt])
        return np.dot(transition, x) + offset

def hx(x: np.ndarray) -> np.ndarray:
    return np.array([x[0], x[1], x[2]])

def apply_kalman_filter(data: np.ndarray, dz: float, initial_guess: Guess) -> tuple[np.ndarray, np.ndarray]:
    initial_state = np.zeros(6)
    args = get_kalman_args()
    momentum = QBRHO_2_P * (initial_guess.brho * float(args.ejectile.Z))
    speed = momentum / args.ejectile.mass * constants.speed_of_light

    initial_state[0] = initial_guess.vertex_x
    initial_state[1] = initial_guess.vertex_y
    initial_state[2] = initial_guess.vertex_z
    initial_state[3] = speed * math.sin(initial_guess.polar) * math.cos(initial_guess.azimuthal)
    initial_state[4] = speed * math.sin(initial_guess.polar) * math.sin(initial_guess.azimuthal)
    initial_state[5] = speed * math.cos(initial_guess.polar)

    first_dz = np.abs(initial_state[2] - data[0, 2])
    dz_set = [np.abs(point[2] - data[idx-1, 2]) for idx, point in enumerate(data) if idx != 0]
    mean_dz = np.average(dz_set)
    sigma_dz = np.std(dz_set)
    dz_set.insert(0, first_dz)
    cutoff = len(dz_set)
    for idx, ds in enumerate(dz_set[1:]):
         if ds > (mean_dz + 2.0 * sigma_dz):
              cutoff = idx
              break

    points = MerweScaledSigmaPoints(6, alpha=.001, beta=2., kappa=0)
    k_filter = UnscentedKalmanFilter(6, 3, dz, hx, fx, points)
    sigma_speed = 0.01 * speed
    sigma_pos = 0.001
    k_filter.P = np.diag([(sigma_pos)**2.0, (sigma_pos)**2.0, (dz)**2.0, sigma_speed**2.0, sigma_speed**2.0, sigma_speed**2.0])
    k_filter.x = initial_state
    k_filter.R = np.diag([sigma_pos**2.0, sigma_pos**2.0, sigma_pos**2.0])
    k_filter.Q = Q_discrete_white_noise(dim=2, dt=dz, var=(sigma_speed)**2.0, block_size=3, order_by_dim=False)

    (means, covs) = k_filter.batch_filter(data[:cutoff], dts=dz_set[:cutoff])

    dz_set.insert(0, 0.0)
    all_means = np.insert(means, 0, initial_state, axis=0)
    all_cov = np.insert(covs, 0, np.diag([(sigma_pos)**2.0, (sigma_pos)**2.0, (sigma_pos)**2.0, sigma_speed**2.0, sigma_speed**2.0, sigma_speed**2.0]), axis=0)
    trajectory, traj_cov, _ = k_filter.rts_smoother(all_means, all_cov, dts=dz_set[:(cutoff+1)])
    return trajectory, traj_cov

def solve_physics_kalman(cluster_index: int, cluster: Cluster, initial_guess: Guess, det_params: DetectorParameters, target: Target, ejectile: NucleusData, results: dict[str, list[float]]):
    bfield = -1.0 * det_params.magnetic_field
    efield = -1.0 * det_params.electric_field

    set_kalman_args(target, ejectile, bfield, efield)

    #convert everyone to meters from mm
    data = cluster.data[:, :3]
    data *= 0.001
    initial_guess.vertex_x *= 0.001
    initial_guess.vertex_y *= 0.001
    initial_guess.vertex_z *= 0.001

    if initial_guess.direction is Direction.BACKWARD:
        np.flip(data, axis=0)

    try:
        trajectory, covariance = apply_kalman_filter(data, cluster.z_bin_width * 0.001, initial_guess)
    except Exception:
         return
    momentum = np.linalg.norm(trajectory[0, 3:]) * ejectile.mass / constants.speed_of_light
    polar = np.arctan2(np.linalg.norm(trajectory[0, 3:5]), trajectory[0, 5])
    azimuthal = np.arctan2(trajectory[0, 4], trajectory[0, 3])
    if azimuthal < 0.0:
         azimuthal += 2.0 * np.pi
    sigma_velox = math.sqrt(np.abs(covariance[0,3,3]))
    sigma_veloy = math.sqrt(np.abs(covariance[0,4,4]))
    sigma_veloz = math.sqrt(np.abs(covariance[0,5,5]))
    sigma_velo = np.linalg.norm(trajectory[0, 3:]) * math.sqrt((sigma_velox/trajectory[0,3])**2.0 + (sigma_veloy/trajectory[0,4])**2.0 + (sigma_veloz/trajectory[0,5])**2.0)
    sigma_veloxy = np.linalg.norm(trajectory[0, 3:]) * math.sqrt((sigma_velox/trajectory[0,3])**2.0 + (sigma_veloy/trajectory[0,4])**2.0)
    sigma_momentum = sigma_velo * ejectile.mass/constants.speed_of_light
    sigma_polar = abs(polar - np.arctan2(np.linalg.norm(trajectory[0, 3:5]) + sigma_veloxy, trajectory[0,5] + sigma_veloz))
    sigma_azimuth = abs(azimuthal - np.arctan2(trajectory[0,4] + sigma_veloy, trajectory[0,3] + sigma_velox))

    results['cluster_index'].append(cluster_index)
    results['cluster_label'].append(cluster.label)
    results['event'].append(cluster.event)
    results['vertex_x'].append(trajectory[0,0] * 1000.0)
    results['sigma_vx'].append(math.sqrt(covariance[0,0,0]))
    results['vertex_y'].append(trajectory[0,1] * 1000.0)
    results['sigma_vy'].append(math.sqrt(covariance[0,1,1]))
    results['vertex_z'].append(trajectory[0,2] * 1000.0)
    results['sigma_vz'].append(math.sqrt(covariance[0,2,2]))
    results['brho'].append(momentum / (QBRHO_2_P * ejectile.Z))
    results['sigma_brho'].append(sigma_momentum / (QBRHO_2_P * ejectile.Z))
    results['polar'].append(polar)
    results['sigma_polar'].append(sigma_polar)
    results['azimuthal'].append(azimuthal)
    results['sigma_azimuthal'].append(sigma_azimuth)
    results['redchisq'].append(0.0)