import numpy as np
from constants import INVALID_PAD_ID, NUMBER_OF_TIME_BUCKETS, INVALID_PEAK
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
    centroid: float = INVALID_PEAK
    amplitude: float = 0.0
    integral: float = 0.0


class Trace:
    def __init__(self, data: Optional[np.ndarray], pad_id: int = INVALID_PAD_ID):
        self.raw_data: Optional[np.ndarray] = None
        self.smoothing_spline: Optional[UnivariateSpline] = None
        self.smoothed_output: Optional[np.ndarray] = None
        self.pad_id: int = INVALID_PAD_ID
        if (data is not None) and (pad_id != INVALID_PAD_ID):
            self.set_trace_data(data, pad_id)

    def invalidate(self):
        self.raw_data = None
        self.smoothing_spline = None
        self.smoothed_output = None
        self.pad_id: int = INVALID_PAD_ID

    def set_trace_data(self, data: np.ndarray, pad_id: int):
        data_shape = np.shape(data)
        if data_shape[0] != NUMBER_OF_TIME_BUCKETS:
            self.invalidate()
            return
        
        self.raw_data = data
        self.pad_id = pad_id
        x_coords = np.arange(1, len(self.raw_data)-1, 1)
        smoothness = len(self.raw_data) * 11.0 #ICK bad, tested on GWM data but should be input parameter
        self.smoothing_spline = UnivariateSpline(self.raw_data[1:len(self.raw_data)], x_coords, k=4, s=smoothness) #Need min order 4 spline; deriv -> 3 order, min needed for roots
        self.smoothed_output = self.smoothing_spline(x_coords)
        

    def is_valid(self) -> bool:
        return self.pad_id != INVALID_PAD_ID and self.raw_data is not None

    def get_pad_id(self) -> int:
        return self.pad_id
    
    def find_peak(self) -> Optional[Peak]:
        '''
            Find the broad signal in a Trace. 
            The goal is to determine the centroid location of a signal peak within a given pad trace. This is accomplished by
            taking a smoothed raw trace, differentiating it and finding the inflection points of the trace as well as the roots of the differentiated trace.
            The peak is then taken as the root which lies between a positive and negative inflection point (in that order). The height of the peak is then taken as
            the height of the timebucket which the identified peak lies in for the raw signal minus the height of the positive inflection point in the raw signal. This method
            is roughly equivalent to the methods used in the IgorPro AT-TPC analysis.

            ## Notes
            This method identifies *only* one peak. Signals can contain more than one postive or negative inflection point. Currently this algorithim only keeps one of each 
            (the inflection points which correspond to the greatest change in slope).

            ## ToDo
            GWM: Handle multiple peaks? Use inflection points which correspond to largest raw signal instead of largest derivative?
            GWM: Baseline correction - probably better to do linear interpolation of positive and negative inflection over range of peak rather than single value. Looking at the traces
            the baseline restoration seems slow on the period of the peaks.

            ## Returns
            Optional[Peak]: Returns None if no peak is found, or a Peak type if a peak is found. The Peak dataclass contains three fields, centroid (peak location), amplitude, and integral
        '''

        peak = Peak()
        if self.is_valid() == False:
            return None
        
        deriv = self.smoothing_spline.derivative()
        deriv_array = deriv(np.arange(1, len(self.raw_data)-1, 1))
        positive_inflection, _ = find_peaks(deriv_array, distance=len(deriv_array)) # only the largest is kept
        negative_inflection, _ = find_peaks(-1.0 * deriv_array, distance=len(deriv_array)) # only the largest is kept
        if negative_inflection is None or positive_inflection is None:
            print(f"Oh no, cant find inflection points!")
            return False
        
        smoothed_roots = deriv.roots()

        #Edge case: peak is positioned such that only one inflection point is present in the signal
        #This will cause another negative inflection point to be found, and the order will be wrong. 
        #In this case, we check to see if there is a root past the positive inflection point
        if negative_inflection < positive_inflection:
            for root in smoothed_roots:
                if root > positive_inflection:
                    peak.centroid = root
                    peak_bucket = int(self.peak)
                    pi_bucket = int(positive_inflection[0])
                    peak.amplitude = self.raw_data[peak_bucket] - self.raw_data[pi_bucket]
                    peak.integral = np.sum(self.raw_data[pi_bucket:] - self.raw_data[pi_bucket])
        else:
            # Normal case: look for a root of the derivative which lies between a positive and negative inflection point
            for root in smoothed_roots:
                if root > positive_inflection and root < negative_inflection:
                    peak.centroid = root
                    peak_bucket = int(self.peak)
                    pi_bucket = int(positive_inflection[0])
                    ni_bucket = int(negative_inflection[0])
                    peak.amplitude = self.raw_data[peak_bucket] - self.raw_data[pi_bucket]
                    peak.integral = np.sum(self.raw_data[pi_bucket:(ni_bucket+1)] - self.raw_data[pi_bucket])
                    break

        if peak.centroid != INVALID_PEAK:
            return peak
        else:
            return None