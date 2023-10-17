import numpy as np
import h5py as h5
from scipy import signal
from .get_trace import Peak

FRIB_TRACE_LENGTH: int = 2048

class FribTrace:

    def __init__(self,  data: np.ndarray):
        self.trace: np.ndarray = np.empty(0, dtype=np.int32)
        self.peaks: list[Peak] = []
        self.set_trace_data(data)

    def set_trace_data(self, data: np.ndarray):
        if len(data) != FRIB_TRACE_LENGTH:
            print(f'Error at ICTrace, trace was given with length {len(data)}, expected {FRIB_TRACE_LENGTH}')

        self.trace = data.astype(np.int32)
        self.find_peaks()

    def is_valid(self) -> bool:
        return len(self.trace) != 0
        

    def find_peaks(self):

        if not self.is_valid():
            return

        self.peaks.clear()
        pks, props = signal.find_peaks(self.trace, distance=50.0, prominence=20.0, width=(0, 500.0), rel_height=0.85)
        for idx, p in enumerate(pks):
            peak = Peak()
            peak.centroid = p
            peak.amplitude = float(self.trace[p])
            peak.positive_inflection = int(np.floor(props['left_ips'][idx]))
            peak.negative_inflection = int(np.ceil(props['right_ips'][idx]))
            peak.integral = np.sum(np.abs(self.trace[peak.positive_inflection:peak.negative_inflection]))
            if peak.amplitude > 100.0:
                self.peaks.append(peak)

    def get_number_of_peaks(self) -> int:
        return len(self.peaks)


    def get_peaks(self) -> list[Peak]:
        return self.peaks
    
