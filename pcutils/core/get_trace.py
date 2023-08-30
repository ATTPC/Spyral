import numpy as np
from .constants import INVALID_PAD_ID, NUMBER_OF_TIME_BUCKETS, INVALID_PEAK_CENTROID
from .hardware_id import HardwareID
from .config import TraceParameters
from typing import Optional
from scipy import signal
from dataclasses import dataclass

@dataclass
class Peak:
    '''
    Dataclass representing a singal peak in a raw pad trace

    ## Fields
    centroid: float - the peak location in time buckets
    amplitude: float - the basline corrected amplitude of the peak
    integral: float - the basline corrected integral of the peak from the postive inflection point to negative inflection point (where possible)
    '''
    centroid: float = INVALID_PEAK_CENTROID
    positive_inflection: float = 0.0
    negative_inflection: float = 0.0
    amplitude: float = 0.0
    uncorrected_amplitude: float = 0.0
    integral: float = 0.0


class GetTrace:
    def __init__(self, data: Optional[np.ndarray] = None, id: HardwareID = HardwareID(), params: TraceParameters = TraceParameters()):
        self.raw_data: Optional[np.ndarray] = None
        self.peaks: list[Peak] = []
        self.hw_id: HardwareID = HardwareID()
        if  isinstance(data, np.ndarray) and id.pad_id != INVALID_PAD_ID:
            self.set_trace_data(data, id, params)

    def invalidate(self):
        self.raw_data = None
        self.smoothing_spline = None
        self.smoothed_output = None
        self.pad_id: int = INVALID_PAD_ID

    def set_trace_data(self, data: np.ndarray, id: HardwareID, params: TraceParameters):
        data_shape = np.shape(data)
        if data_shape[0] != NUMBER_OF_TIME_BUCKETS:
            print(data_shape[0])
            self.invalidate()
            return
        
        self.raw_data = data.astype(np.int32) #Widen the type and sign it
<<<<<<< HEAD
        #Edges can be strange, so smooth them a bit
        self.raw_data[0] = self.raw_data[1]
        self.raw_data[511] = self.raw_data[510]
        #self.corrected_data = (self.raw_data - self.evaluate_baseline(baseline_window_scale)).clip(min = 0) #remove the baseline
=======
>>>>>>> 577a179e20daa618f00875e03c7f66a197dbeacc
        self.hw_id = id
        self.find_peaks(params.peak_separation, params.peak_prominence, params.peak_max_width, params.peak_threshold)

    def is_valid(self) -> bool:
        return self.hw_id.pad_id != INVALID_PAD_ID and isinstance(self.raw_data, np.ndarray)

    def get_pad_id(self) -> int:
        return self.hw_id.pad_id
    
    def find_peaks(self, separation: float, prominence: float, max_width: float, threshold: float) -> bool:
        '''
            Find the signals in a Trace. 
            The goal is to determine the centroid location of a signal peak within a given pad trace. Use the find_peaks
            function of scipy.signal to determine peaks. We then use this info to extract peak amplitudes, and integrated charge.

            ## Returns
            bool: Returns False if no peak is found, or True if any peak is found. Use get_peaks() to retrieve the peak data.
        '''

        if self.is_valid() == False:
            return None
        
        self.peaks.clear()

        ## GWM 08/08/23 -- This method is way faster cause we don't need to make splines in set_trace_data, but is way more impacted
        ## by user input parameters
        pks, props = signal.find_peaks(self.raw_data, distance=separation, prominence=prominence, width=(0, max_width), rel_height=0.85)
        for idx, p in enumerate(pks):
            peak = Peak()
            peak.centroid = p
            peak.amplitude = self.raw_data[p]
            peak.positive_inflection = int(np.floor(props['left_ips'][idx]))
            peak.negative_inflection = int(np.ceil(props['right_ips'][idx]))
            peak.integral = np.sum(self.raw_data[peak.positive_inflection:peak.negative_inflection])
            if peak.amplitude > threshold:
                self.peaks.append(peak)

        #Get rid of peaks from saturated trace (NMT)
        #temp_arr = np.array([[Peak.positive_inflection, Peak.negative_inflection, Peak.uncorrected_amplitude] for Peak in self.peaks])
        #unique, counts = np.unique(temp_arr, axis = 0, return_counts = True)
        #duplicates = unique[counts > 1].tolist()
        #self.peaks = [Peak for Peak in self.peaks if np.logical_and(~(np.isin([Peak.positive_inflection, Peak.negative_inflection, Peak.uncorrected_amplitude], duplicates).all()), Peak.uncorrected_amplitude < 4095)]

        if len(self.peaks) > 0:
            return True
        else:
            return False
        
    def get_number_of_peaks(self) -> int:
        if self.peaks is None:
            return 0
        else:
            return len(self.peaks)
    
    def get_peaks(self) -> list[Peak]:
        return self.peaks
