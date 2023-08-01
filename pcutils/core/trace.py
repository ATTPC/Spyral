import numpy as np
from constants import INVALID_PAD_ID, NUMBER_OF_TIME_BUCKETS
from typing import Optional
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
from math import sqrt

class Trace:
    def __init__(self, data: Optional[np.ndarray], pad_id: int = INVALID_PAD_ID):
        self.raw_data: Optional[np.ndarray] = None
        self.smoothing_spline: Optional[UnivariateSpline] = None
        self.smoothed_output: Optional[np.ndarray] = None
        self.pad_id: int = INVALID_PAD_ID
        self.peaks: Optional[np.ndarray] = None
        self.peak_heights: Optional[np.ndarray] = None #Amplitude
        self.peak_energies: Optional[np.ndarray] = None #Integral
        if (data is not None) and (pad_id != INVALID_PAD_ID):
            self.set_trace_data(data, pad_id)

    def invalidate(self):
        self.raw_data = None
        self.smoothing_spline = None
        self.smoothed_output = None
        self.pad_id: int = INVALID_PAD_ID
        self.peaks = None
        self.peak_heights = None
        self.peak_energies = None

    def set_trace_data(self, data: np.ndarray, pad_id: int):
        data_shape = np.shape(data)
        if data_shape[0] != NUMBER_OF_TIME_BUCKETS:
            self.invalidate()
            return
        
        self.raw_data = data
        self.pad_id = pad_id
        x_coords = np.arange(0, len(self.raw_data), 1)
        smoothness = len(self.raw_data) - sqrt(2.0 * len(self.raw_data))
        self.smoothing_spline = UnivariateSpline(self.raw_data, x_coords, s=smoothness)
        self.smoothed_output = self.smoothing_spline(x_coords)

    def is_valid(self) -> bool:
        return self.pad_id != INVALID_PAD_ID and self.raw_data is not None

    def get_pad_id(self) -> int:
        return self.pad_id
    
    def find_peaks(self):
        if self.is_valid == False:
            return

        min_peak_width = 24 #Assume here that the full width is 2 * 3 * sigma, and sigma = 4
        min_peak_height = 70
        self.peaks, props = find_peaks(self.smoothed_output, width=min_peak_width, height=min_peak_height)
        self.peak_heights = np.zeros(len(self.peaks))
        self.peak_energies = np.zeros(len(self.peaks))
        low_edge = 0
        hi_edge = 0
        for idx, peak_loc in enumerate(self.peaks):
            self.peak_heights[idx] = props["heights"][peak_loc]
            low_edge = props["left_ips"]
            hi_edge = props["right_ips"]
            if ( hi_edge > len(self.smoothed_output)):
                hi_edge = len(self.smoothed_output)
            if ( low_edge < 0 ):
                low_edge = 0
            self.peak_energies[idx] = np.sum(self.smoothed_output[low_edge:hi_edge])

        sorted_indicies = np.argsort(self.peak_heights) #sort by peak height
        self.peak_energies = self.peak_energies[sorted_indicies]
        self.peak_heights = self.peak_heights[sorted_indicies]
        self.peaks = self.peaks[sorted_indicies]
        