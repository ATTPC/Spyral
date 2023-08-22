from .config import DetectorParameters
from .target import Target
from .nuclear_data import NucleusData
from .clusterize import ClusteredCloud
from .constants import MEV_2_JOULE, MEV_2_KG
from scipy import optimize, integrate, interpolate, constants
import numpy as np
import math
from dataclasses import dataclass

@dataclass
class InitialValue:
    polar: float = 0.0 #radians
    azimuthal: float = 0.0 #radians
    brho: float = 0.0 #Tm
    vertex_x: float = 0.0 #mm
    vertex_y: float = 0.0 #mm
    vertex_z: float = 0.0 #mm

    def convert_to_array(self) -> np.ndarray:
        return np.array([self.polar, self.azimuthal, self.brho, self.vertex_x, self.vertex_y, self.vertex_z])

TIME_WINDOW: float = 1.0e-6 #1us window
INTERVAL: float = 1.0e-10 #0.1 ns sample
QBRHO_2_P: float = 1.0e-9 * constants.speed_of_light #kG * cm -> MeV

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

    print(Bfield, Efield)

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