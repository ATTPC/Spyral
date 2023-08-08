import numpy as np
from .constants import INVALID_PAD_ID, NUMBER_OF_TIME_BUCKETS, INVALID_PEAK_CENTROID
from .hardware_id import HardwareID
from typing import Optional
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
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
    integral: float = 0.0


class GetTrace:
    def __init__(self, data: Optional[np.ndarray], id = HardwareID):
        self.raw_data: Optional[np.ndarray] = None
        self.smoothing_spline: Optional[UnivariateSpline] = None
        self.smoothed_output: Optional[np.ndarray] = None
        self.peaks: list[Peak] = []
        self.hw_id: HardwareID = HardwareID()
        if  isinstance(data, np.ndarray) and id.pad_id != INVALID_PAD_ID:
            self.set_trace_data(data, id)

    def invalidate(self):
        self.raw_data = None
        self.smoothing_spline = None
        self.smoothed_output = None
        self.pad_id: int = INVALID_PAD_ID

    def set_trace_data(self, data: np.ndarray, id: HardwareID, smoothing: float = 7.0):
        data_shape = np.shape(data)
        if data_shape[0] != NUMBER_OF_TIME_BUCKETS:
            print(data_shape[0])
            self.invalidate()
            return
        
        self.raw_data = data.astype(np.int32) #Widen the type and sign it
        self.hw_id = id
        smoothness = len(self.raw_data) * smoothing
        self.smoothing_spline = UnivariateSpline(np.arange(0, NUMBER_OF_TIME_BUCKETS, 1), self.raw_data, k=4, s=smoothness) #Need min order 4 spline; deriv -> 3 order, min needed for roots
        self.smoothed_output = self.smoothing_spline(np.arange(0, NUMBER_OF_TIME_BUCKETS, 1))
        

    def is_valid(self) -> bool:
        return self.hw_id.pad_id != INVALID_PAD_ID and isinstance(self.raw_data, np.ndarray)

    def get_pad_id(self) -> int:
        return self.hw_id.pad_id
    
    def find_peaks(self, separation: float = 50.0, threshold = 250.0) -> bool:
        '''
            Find the signals in a Trace. 
            The goal is to determine the centroid location of a signal peak within a given pad trace. This is accomplished by
            taking a smoothed raw trace, differentiating it and finding the inflection points of the trace as well as the roots of the differentiated trace.
            The peak is then taken as the root which lies between a positive and negative inflection point (in that order). The height of the peak is then taken as
            the height of the timebucket which the identified peak lies in for the raw signal minus the height of the positive inflection point in the raw signal. This method
            is roughly equivalent to the methods used in the IgorPro AT-TPC analysis.

            ## Notes
            This method identifies any peaks within the signal.

            ## Returns
            bool: Returns False if no peak is found, or True if any peak is found. Use get_peaks() to retrieve the peak data.
        '''

        if self.is_valid() == False:
            return None
        
        self.peaks.clear()

        ## GWM 08/08/23 -- This method is way faster cause we don't need to make splines in set_trace_data, but is way more impacted
        ## by user input parameters
        ## Tested with fake data, this seems to work well. Needs validation with real data
        # pks, props = find_peaks(self.raw_data, distance=separation, prominence=10, width=0, rel_height=0.1)
        # for idx, p in enumerate(pks):
        #     peak = Peak()
        #     peak.centroid = p
        #     peak.amplitude = self.raw_data[p]
        #     peak.positive_inflection = int(props['left_ips'][idx])
        #     peak.negative_inflection = int(props['right_ips'][idx])
        #     peak.integral = np.sum(self.raw_data[peak.positive_inflection:peak.negative_inflection])
        #     if peak.integral > threshold:
        #         self.peaks.append(peak)

        ## This method is too slow; or rather, the requirement of having created a smoothing spline is too slow (factor of 4 at worst, factor of 6-7 at best)
        deriv = self.smoothing_spline.derivative()
        deriv_array = deriv(np.arange(0, NUMBER_OF_TIME_BUCKETS, 1))
        smoothed_roots = self.smoothing_spline.derivative().roots()
        positive_inflection, _ = find_peaks(deriv_array, distance=separation)
        negative_inflection, _ = find_peaks(-1.0 * deriv_array, distance=separation)
        positive_inflection.sort()
        negative_inflection.sort()
        searches: list[tuple[float, float]] = []
        #Find valid search regions, which are bounded by a positive inflection on the low end and a negative inflection on the high end
        for pos in positive_inflection:
            for neg in negative_inflection:
                if pos < neg:
                   searches.append((pos, neg))
                   break

        if len(searches) == 0:
            #print("Oh no can't find inflections!")
            return False

        #Look for a root of the derivative which lies between a positive and negative inflection point
        for root in smoothed_roots:
            for region in searches:
                if root > region[0] and root < region[1]:
                    peak = Peak()
                    peak.centroid = root
                    peak.positive_inflection = region[0]
                    peak.negative_inflection = region[1]
                    peak_bucket = int(peak.centroid)
                    pi_bucket = int(region[0])
                    ni_bucket = int(region[1])
                    peak.amplitude = self.raw_data[peak_bucket]
                    peak.integral = np.sum(self.raw_data[pi_bucket:(ni_bucket+1)], dtype=np.float64)
                    if (peak.integral > threshold):
                        self.peaks.append(peak)
                    break

        if len(self.peaks) > 0:
            return True
        else:
            #self.peaks.clear()
            return False
        
    def get_number_of_peaks(self) -> int:
        if self.peaks is None:
            return 0
        else:
            return len(self.peaks)
    
    def get_peaks(self) -> list[Peak]:
        return self.peaks