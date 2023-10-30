from ..core.constants import INVALID_PEAK_CENTROID
from dataclasses import dataclass

@dataclass
class Peak:
    '''
    Dataclass representing a singal peak in a trace

    ## Fields
    centroid: float - the peak location
    amplitude: float - the amplitude of the peak
    positive_inflection: float - the location of the rising edge of the peak
    negative_inflection: float - the location of the falling edge of the peak
    integral: float - the integral of the peak between postive_inflection and negative_inflection
    '''
    centroid: float = INVALID_PEAK_CENTROID
    positive_inflection: float = 0.0
    negative_inflection: float = 0.0
    amplitude: float = 0.0
    uncorrected_amplitude: float = 0.0
    integral: float = 0.0