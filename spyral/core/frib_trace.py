from .peak import Peak
from .config import FribParameters
from .constants import FRIB_TRACE_LENGTH

import numpy as np
from scipy import signal


class FribTrace:
    '''
    # FribTrace
    Similar to GetTrace, FribTrace represents a raw signal from the SIS3300 module which is managed through the FRIBDAQ.
    Typically contains signals for the ion chamber (IC), auxillary silicon detectors (Si), and the mesh signal.
    '''

    def __init__(self,  data: np.ndarray, params: FribParameters):
        self.trace: np.ndarray = np.empty(0, dtype=np.int32)
        self.peaks: list[Peak] = []
        self.set_trace_data(data, params)

    def set_trace_data(self, data: np.ndarray, params: FribParameters):
        if len(data) != FRIB_TRACE_LENGTH:
            print(f'Error at ICTrace, trace was given with length {len(data)}, expected {FRIB_TRACE_LENGTH}')

        self.trace = data.astype(np.int32)
        self.find_peaks(params)

    def is_valid(self) -> bool:
        return len(self.trace) != 0
        

    def find_peaks(self, params: FribParameters):
        '''
        Peak finding algorithm, similar to used in GetTrace

        ## Parameters
        params: FribParameters, config class containing the variables which control the peak finding algorithm
        '''

        if not self.is_valid():
            return

        self.peaks.clear()
        pks, props = signal.find_peaks(self.trace, distance=params.peak_separation, prominence=params.peak_prominence, width=(0, params.peak_max_width), rel_height=0.85)
        for idx, p in enumerate(pks):
            peak = Peak()
            peak.centroid = p
            peak.amplitude = float(self.trace[p])
            peak.positive_inflection = int(np.floor(props['left_ips'][idx]))
            peak.negative_inflection = int(np.ceil(props['right_ips'][idx]))
            peak.integral = np.sum(np.abs(self.trace[peak.positive_inflection:peak.negative_inflection]))
            if peak.amplitude > params.peak_threshold:
                self.peaks.append(peak)

    def get_number_of_peaks(self) -> int:
        return len(self.peaks)

    def get_peaks(self) -> list[Peak]:
        return self.peaks
    
