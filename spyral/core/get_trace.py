from .constants import INVALID_PAD_ID, NUMBER_OF_TIME_BUCKETS
from .peak import Peak
from .hardware_id import HardwareID
from .config import TraceParameters

from scipy import signal
import numpy as np

class GetTrace:
    def __init__(self, data: np.ndarray, id: HardwareID, params: TraceParameters):
        self.raw_data: np.ndarray = np.empty(0, dtype=np.int32)
        self.peaks: list[Peak] = []
        self.hw_id: HardwareID = HardwareID()
        if  isinstance(data, np.ndarray) and id.pad_id != INVALID_PAD_ID:
            self.set_trace_data(data, id, params)

    def set_trace_data(self, data: np.ndarray, id: HardwareID, params: TraceParameters):
        data_shape = np.shape(data)
        if data_shape[0] != NUMBER_OF_TIME_BUCKETS:
            print(f"GetTrace was given data that did not have the correct shape! Expected 512 time buckets, instead got {data_shape[0]}")
            return
        
        self.raw_data = data.astype(np.int32) #Widen the type and sign it
        #Edges can be strange, so smooth them a bit
        self.raw_data[0] = self.raw_data[1]
        self.raw_data[511] = self.raw_data[510]
        #self.corrected_data = (self.raw_data - self.evaluate_baseline(baseline_window_scale)).clip(min = 0) #remove the baseline
        self.hw_id = id
        self.find_peaks(params.peak_separation, params.peak_prominence, params.peak_max_width, params.peak_threshold)

    def is_valid(self) -> bool:
        return self.hw_id.pad_id != INVALID_PAD_ID and isinstance(self.raw_data, np.ndarray)

    def get_pad_id(self) -> int:
        return self.hw_id.pad_id
    
    def find_peaks(self, separation: float, prominence: float, max_width: float, threshold: float):
        '''
            Find the signals in a Trace. 
            The goal is to determine the centroid location of a signal peak within a given pad trace. Use the find_peaks
            function of scipy.signal to determine peaks. We then use this info to extract peak amplitudes, and integrated charge.
        '''

        if self.is_valid() == False:
            return
        
        self.peaks.clear()

        ## GWM 08/08/23 -- This method is way faster cause we don't need to make splines in set_trace_data, but is way more impacted
        ## by user input parameters
        pks, props = signal.find_peaks(self.raw_data, distance=separation, prominence=prominence, width=(0, max_width), rel_height=0.85)
        for idx, p in enumerate(pks):
            peak = Peak()
            peak.centroid = p
            peak.amplitude = float(self.raw_data[p])
            peak.positive_inflection = int(np.floor(props['left_ips'][idx]))
            peak.negative_inflection = int(np.ceil(props['right_ips'][idx]))
            peak.integral = np.sum(np.abs(self.raw_data[peak.positive_inflection:peak.negative_inflection]))
            if peak.amplitude > threshold:
                self.peaks.append(peak)

    def get_number_of_peaks(self) -> int:
        return len(self.peaks)
    
    def get_peaks(self) -> list[Peak]:
        return self.peaks
