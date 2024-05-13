from ..core.constants import INVALID_PEAK_CENTROID
from dataclasses import dataclass


@dataclass
class Peak:
    """Dataclass representing a singal peak in a trace

    Attributes
    ----------
    centroid: float
        The peak location
    amplitude: float
        The amplitude of the peak
    positive_inflection: float
        The location of the left edge of the peak for integration
        TODO: This name is bad, change it
    negative_inflection: float
        The location of the right edge of the peak for integration
        TODO: This name is bad, change it
    integral: float
        The integral of the peak between postive_inflection and negative_inflection
    """

    centroid: float = INVALID_PEAK_CENTROID
    positive_inflection: float = 0.0
    negative_inflection: float = 0.0
    amplitude: float = 0.0
    uncorrected_amplitude: float = 0.0
    integral: float = 0.0
