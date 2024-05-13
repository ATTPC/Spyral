from ..core.estimator import Direction
import numpy as np
from dataclasses import dataclass


@dataclass
class Guess:
    """Dataclass which is a simple container to hold initial guess info for solvers. Can be converted into ndarray (excluding direction)

    Attributes
    ----------
    polar: float
        The polar angle in radians
    azimuthal: float
        The azimuthal angle in radians
    brho: float
        The magnetic rigidity in Tm
    vertex_x: float
        The vertex x-coordinate in mm
    vertex_y: float
        The vertex y-coordinate in mm
    vertex_z: float
        The vertex z-coordinate in mm
    direction: Direction
        The Direction of the trajectory

    Methods
    -------
    convert_to_array() -> numpy.ndarray
        Converts the Guess to an ndarray, excluding the direction attribute
    """

    polar: float = 0.0  # radians
    azimuthal: float = 0.0  # radians
    brho: float = 0.0  # Tm
    vertex_x: float = 0.0  # mm
    vertex_y: float = 0.0  # mm
    vertex_z: float = 0.0  # mm
    direction: Direction = Direction.NONE

    def convert_to_array(self) -> np.ndarray:
        """Converts the Guess to an ndarray, excluding the direction attribute

        Returns
        -------
        numpy.ndarray:
            Format of [polar, azimuthal, brho, vertex_x, vertex_y, vertex_z]
        """
        return np.array(
            [
                self.polar,
                self.azimuthal,
                self.brho,
                self.vertex_x,
                self.vertex_y,
                self.vertex_z,
            ]
        )
