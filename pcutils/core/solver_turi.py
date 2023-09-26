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
C: float = 3.0e8 # speed of light in m/s

def motionIVP(invec: list, Efield: list, Bfield: list, dens: float, dEdx_interp: callable, q: int, m: int) -> list:
    x, y, z, vx, vy, vz = invec
    rr = np.sqrt(vx**2 + vy**2 + vz**2)
    azi = np.arctan2(vy, vx)
    pol = np.arccos(vz / rr)
    
    vv = np.sqrt(vx**2 + vy**2 + vz**2)
    E = amuev * (1/np.sqrt(1 - (vv / C)**2) - 1)
    if E < 0.001 or E > 50:
        return 1
    
    st = dEdx_interp(E) * 1000 # In MeV/(g/cm^2)
    st *= 1.6021773349e-13 # Converts to J/(g/cm^2)
    st *= dens*100 # Converts to kg*m/s^2
    st /= m # Converts to m/s^2
    dvecdt = [vx,
              vy,
              vz,
              (q/m)*(Efield[0] + vy*Bfield[2] - vz*Bfield[1]) - st*np.sin(pol)*np.cos(azi),
              (q/m)*(Efield[1] + vz*Bfield[0] - vx*Bfield[2]) - st*np.sin(pol)*np.sin(azi),
              (q/m)*(Efield[2] + vx*Bfield[1] - vy*Bfield[0]) - st*np.cos(pol)]
    return dvecdt
