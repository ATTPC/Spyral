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


@dataclass
class SolverResult:
    """Container for the results of solver algorithm with metadata

    Attributes
    ----------
    event: int
        The event number
    cluster_index: int
        The cluster index
    cluster_label: int
        The label from the clustering algorithm
    orig_run: int
        The original run number
    orig_event: int
        The original event number
    vertex_x: float
        The vertex x position (m)
    sigma_vx: float
        The uncertainty of the vertex x position (m)
    vertex_y: float
        The vertex y position (m)
    sigma_vy: float
        The uncertainty of the vertex y position (m)
    vertex_z: float
        The vertex z position (m)
    sigma_vz: float
        The uncertainty of the vertex z position (m)
    brho: float
        The magnetic rigidity (Tm)
    sigma_brho: float
        The uncertainty of the magnetic rigidity (Tm)
    ke: float
        The kinetic energy (MeV)
    sigma_ke: float
        The uncertainty of the kinetic energy (MeV)
    polar: float
        The polar angle (radians)
    sigma_polar: float
        The uncertianty of the polar angle (radians)
    azimuthal: float
        The azimuthal angle (radians)
    sigma_azimuthal: float
        The uncertianty of the azimuthal angle (radians)
    redchisq: float
        The best-fit value of the objective function (in the case of least squares,
        the reduced chi-square)
    """

    event: int
    cluster_index: int
    cluster_label: int
    orig_run: int
    orig_event: int
    vertex_x: float
    sigma_vx: float
    vertex_y: float
    sigma_vy: float
    vertex_z: float
    sigma_vz: float
    brho: float
    sigma_brho: float
    ke: float
    sigma_ke: float
    polar: float
    sigma_polar: float
    azimuthal: float
    sigma_azimuthal: float
    redchisq: float
