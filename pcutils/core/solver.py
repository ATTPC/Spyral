from .config import DetectorParameters
from .target import Target
from .nuclear_data import NucleusData
from .clusterize import ClusteredCloud
from constants import MEV_2_JOULE, MEV_2_KG
from scipy import optimize, integrate, interpolate, constants
import numpy as np
import math

#State = [x, y, z, vx, vy, vz]
#Derivative = [vx, vy, vz, ax, ay, az] (returns)
def equation_of_motion(t: float, state: np.ndarray, Bfield: np.ndarray, Efield: np.ndarray, target: Target, ejectile: NucleusData) -> np.ndarray:

    speed = np.linalg.norm(state[3:])
    unit_vector = state[3:] / speed # direction
    kinetic_energy = ejectile.mass * (1.0/math.sqrt(1.0 - (speed**2.0 / constants.speed_of_light**2.0)) - 1.0) #MeV
    charge = ejectile.Z * constants.elementary_charge #Coulombs
    mass_kg = ejectile.mass * MEV_2_KG #kg

    dEdx = target.get_dedx(ejectile, kinetic_energy) #MeV/g/cm^2
    dEdx *= MEV_2_JOULE #J/g/cm^2
    force = dEdx * target.data.density * 100.0 # J/cm * cm/m = J/m = kg m/s^2
    deceleration = force / (mass_kg)

    result = np.zeros(len(state))

    result[:3] = state[3:]
    # a = q/m * (E + v x B) - stopping
    result[3:] = charge/mass_kg * (Efield + np.cross(state[3:], Bfield)) - deceleration * unit_vector

    return result

def generate_trajectory_function(polar: float, azimuth: float, brho: float, vertex: np.ndarray, Bfield: np.ndarray, Efield: np.ndarray, target: Target, Z: int, mass: float) -> interpolate.RBFInterpolator:


def objective_function(guess: np.ndarray, data: np.ndarray) -> float:

def solve_physics(cluster_index: int, cluster: ClusteredCloud, detector_params: DetectorParameters, ejectile: NucleusData, target: Target, results: dict[str, list]):
