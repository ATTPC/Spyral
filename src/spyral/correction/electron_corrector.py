from ..interpolate import BilinearInterpolator, clamp
from pathlib import Path
import numpy as np


class ElectronCorrector:
    """Class which uses a BilinearInterpolator to correct the drift time of electrons

    AT-TPC electric field correction for electron drift time

    Parameters
    ----------
    interp: BilinearInterpolator
        The interpolator object

    Attributes
    ----------
    correction: BilinearInterpolator
        The interpolator object

    Methods
    -------
    ElectronCorrector(interp: BilinearInterpolator)
        Construct the corrector
    correct_point(point: ndarray) -> ndarray
        Apply the correction to a point in a point cloud

    """

    def __init__(self, interp: BilinearInterpolator):
        self.correction: BilinearInterpolator = interp

    def correct_point(self, point: np.ndarray) -> np.ndarray:
        """Apply the correction to a point in a point cloud

        Parameters
        ----------
        point: ndarray
            The point to be corrected

        Returns
        -------
        ndarray
            The corrected point
        """
        # Correction returns [rho_cor, trans_cor, z_cor]
        corrected_point = np.zeros(len(point))
        corrected_point[3:] = point[3:]

        radius: float = np.linalg.norm(point[:2])  # type: ignore
        azimuthal: float = np.arctan2(point[1], point[0])

        correction = self.correction.interpolate(radius, point[2])

        z_correction = clamp(correction[2], 0.0, 1000.0)

        corrected_radius = np.sqrt(
            (radius + correction[0]) ** 2.0 + correction[1] ** 2.0
        )
        corrected_azim = azimuthal + np.arctan2(correction[1], (radius + correction[0]))

        corrected_point[0] = corrected_radius * np.cos(corrected_azim)
        corrected_point[1] = corrected_radius * np.sin(corrected_azim)
        corrected_point[2] = point[2] - z_correction

        return corrected_point


def create_electron_corrector(ecorr_path: Path) -> ElectronCorrector:
    """Create an ElectronCorrector

    Parameters
    ----------
    ecorr_path: pathlib.Path
        The path to the correction grid data

    Returns
    -------
    ElectronCorrector
        The corrector object

    """
    rho_bin_min = 0.0
    rho_bin_max = 275.0
    rho_bins = 276

    z_bin_min = 0.0
    z_bin_max = 1000.0
    z_bins = 1001

    grid: np.ndarray = np.load(ecorr_path)

    interpolator = BilinearInterpolator(
        rho_bin_min, rho_bin_max, rho_bins, z_bin_min, z_bin_max, z_bins, grid
    )
    return ElectronCorrector(interpolator)
