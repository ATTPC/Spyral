from ..core.estimator import Direction
from ..core.config import DetectorParameters, SolverParameters
from ..core.constants import QBRHO_2_P
import numpy as np
from dataclasses import dataclass
from spyral_utils.nuclear import NucleusData
from lmfit import Parameters


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


def create_params(
    guess: Guess,
    ejectile: NucleusData,
    det_params: DetectorParameters,
    solver_params: SolverParameters,
) -> Parameters:
    """Create lmfit parameters with appropriate bounds

    Convert all values to correct units (meters, radians, etc.) as well

    Parameters
    ----------
    guess: Guess
        the initial values of the parameters
    ejectile: spyral_utils.nuclear.NucleusData
        the data for the particle being tracked
    det_params: DetectorParameters
        Configuration parameters for detector characteristics
    solver_params: SolverParameters
        Configuration parameters for the solver

    Returns
    -------
    lmfit.Parameters
        the lmfit parameters with bounds
    """
    interp_min_momentum = np.sqrt(
        solver_params.interp_ke_min
        * (solver_params.interp_ke_min + 2.0 * ejectile.mass)
    )
    interp_max_momentum = np.sqrt(
        solver_params.interp_ke_max
        * (solver_params.interp_ke_max + 2.0 * ejectile.mass)
    )
    interp_min_brho = (interp_min_momentum / QBRHO_2_P) / ejectile.Z
    interp_max_brho = (interp_max_momentum / QBRHO_2_P) / ejectile.Z
    interp_min_polar = solver_params.interp_polar_min * np.pi / 180.0
    interp_max_polar = solver_params.interp_polar_max * np.pi / 180.0

    uncertainty_position_z = 0.1
    uncertainty_brho = 1.0

    min_brho = guess.brho - uncertainty_brho * 2.0
    if min_brho < interp_min_brho:
        min_brho = interp_min_brho
    max_brho = guess.brho + uncertainty_brho * 2.0
    if max_brho > interp_max_brho:
        max_brho = interp_max_brho

    min_polar = interp_min_polar
    max_polar = interp_max_polar
    if guess.polar > np.pi * 0.5 and min_polar < np.pi * 0.5:
        min_polar = np.pi * 0.5
    elif guess.polar < np.pi * 0.5 and max_polar > np.pi * 0.5:
        max_polar = np.pi * 0.5

    min_z = guess.vertex_z * 0.001 - uncertainty_position_z * 2.0
    max_z = guess.vertex_z * 0.001 + uncertainty_position_z * 2.0
    if min_z < 0.0:
        min_z = 0.0
    if max_z > det_params.detector_length * 0.001:
        max_z = det_params.detector_length * 0.001

    vert_phi = np.arctan2(guess.vertex_y, guess.vertex_x)
    if vert_phi < 0.0:
        vert_phi += np.pi * 2.0
    vert_rho = np.sqrt(guess.vertex_x**2.0 + guess.vertex_y**2.0) * 0.001

    fit_params = Parameters()
    fit_params.add(
        "brho",
        guess.brho,
        min=min_brho,
        max=max_brho,
        vary=True,
    )
    fit_params.add("polar", guess.polar, min=min_polar, max=max_polar, vary=True)
    fit_params.add(
        "vertex_rho",
        value=vert_rho,
        min=0.0,
        max=det_params.beam_region_radius * 0.001,
        vary=solver_params.fit_vertex_rho,
    )
    fit_params.add(
        "vertex_phi",
        value=vert_phi,
        min=0.0,
        max=np.pi * 2.0,
        vary=solver_params.fit_vertex_phi,
    )
    fit_params.add(
        "azimuthal",
        value=guess.azimuthal,
        min=0.0,
        max=2.0 * np.pi,
        vary=solver_params.fit_azimuthal,
    )
    fit_params.add("vertex_x", expr="vertex_rho * cos(vertex_phi)")
    fit_params.add("vertex_y", expr="vertex_rho * sin(vertex_phi)")
    fit_params.add("vertex_z", guess.vertex_z * 0.001, min=min_z, max=max_z, vary=True)

    return fit_params
