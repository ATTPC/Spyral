import numpy as np
from constants import INVALID_PAD_ID, NUMBER_OF_TIME_BUCKETS, INVALID_PEAK
from typing import Optional
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline


class Trace:
    def __init__(self, data: Optional[np.ndarray], pad_id: int = INVALID_PAD_ID):
        self.raw_data: Optional[np.ndarray] = None
        self.smoothing_spline: Optional[UnivariateSpline] = None
        self.smoothed_output: Optional[np.ndarray] = None
        self.pad_id: int = INVALID_PAD_ID
        self.peak: float = INVALID_PEAK
        self.peak_height: float = 0.0 #Amplitude
        self.peak_energy: float = 0.0 #Integral
        if (data is not None) and (pad_id != INVALID_PAD_ID):
            self.set_trace_data(data, pad_id)

    def invalidate(self):
        self.raw_data = None
        self.smoothing_spline = None
        self.smoothed_output = None
        self.pad_id: int = INVALID_PAD_ID
        self.peaks = INVALID_PEAK
        self.peak_height = 0.0
        self.peak_energy = 0.0

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
    
    def find_peak(self) -> bool:
        if self.is_valid() == False:
            return
        
        deriv = self.smoothing_spline.derivative()
        deriv_array = deriv(np.arange(1, len(self.raw_data)-1, 1))
        positive_inflection, _ = find_peaks(deriv_array, distance=len(deriv_array)) # only the largest is kept
        negative_inflection, _ = find_peaks(-1.0 * deriv_array, distance=len(deriv_array)) # only the largest is kept
        if negative_inflection is None or positive_inflection is None:
            print(f"Oh no, cant find inflection points!")
            return False
        
        smoothed_roots = deriv.roots()

        #Edge case: peak is positioned such that only one inflection point is present in the signal
        #Take whatever root lies past one of the inflection points (in the case of multples, take the one that occurs latest in the signal)
        if negative_inflection < positive_inflection:
            for root in smoothed_roots:
                if root > positive_inflection:
                    self.peak = root
                    peak_bucket = int(self.peak)
                    self.peak_height = self.raw_data[peak_bucket]
                    self.peak_energy = np.sum(self.raw_data[int(positive_inflection):])
                elif root < negative_inflection:
                    self.peak = root
                    peak_bucket = int(self.peak)
                    self.peak_height = self.raw_data[peak_bucket]
                    self.peak_energy = np.sum(self.raw_data[:int(negative_inflection)])
        else:
            # Normal case: look for a root of the derivative which lies between a positive and negative inflection point
            for root in smoothed_roots:
                if root > positive_inflection and root < negative_inflection:
                    self.peak = root
                    peak_bucket = int(self.peak)
                    self.peak_height = self.raw_data[peak_bucket]
                    self.peak_energy = self.raw_data[int(positive_inflection):int(negative_inflection)]
                    break

        if self.peak != INVALID_PEAK:
            return True
        else:
            return False