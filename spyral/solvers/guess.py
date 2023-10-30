from ..core.estimator import Direction
import numpy as np
from dataclasses import dataclass

@dataclass
class Guess:
    '''
    Dataclass which is a simple container to hold initial guess info for solvers. Can be converted into ndarray (excluding direction)
    '''
    polar: float = 0.0 #radians
    azimuthal: float = 0.0 #radians
    brho: float = 0.0 #Tm
    vertex_x: float = 0.0 #mm
    vertex_y: float = 0.0 #mm
    vertex_z: float = 0.0 #mm
    direction: Direction = Direction.NONE 

    def convert_to_array(self) -> np.ndarray:
        return np.array([self.polar, self.azimuthal, self.brho, self.vertex_x, self.vertex_y, self.vertex_z])