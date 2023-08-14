import numpy as np
from .constants import INVALID_PAD_ID, NUMBER_OF_TIME_BUCKETS, INVALID_PEAK_CENTROID
from .hardware_id import HardwareID
from typing import Optional
from scipy.signal import find_peaks, filtfilt
from scipy.interpolate import UnivariateSpline
from scipy.fft import fft, ifft, ifftshift
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
    def __init__(self, data: Optional[np.ndarray], id = HardwareID):
        self.raw_data: Optional[np.ndarray] = None
        self.corrected_data: Optional[np.ndarray] = None
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

    def set_trace_data(self, data: np.ndarray, id: HardwareID, smoothing: float = 3.0, baseline_window_scale: float = 20.0):
        data_shape = np.shape(data)
        if data_shape[0] != NUMBER_OF_TIME_BUCKETS:
            print(data_shape[0])
            self.invalidate()
            return
        
        self.raw_data = data.astype(np.int32) #Widen the type and sign it
        #Edges can be strange, so smooth them a bit
        self.raw_data[0] = self.raw_data[1]
        self.raw_data[511] = self.raw_data[510]
        self.corrected_data = (self.raw_data - self.evaluate_baseline(baseline_window_scale)).clip(min = 0) #remove the baseline
        self.hw_id = id
        smoothness = len(self.corrected_data) * smoothing #ICK bad, tested on GWM data but should be input parameter
        self.smoothing_spline = UnivariateSpline(np.arange(0, NUMBER_OF_TIME_BUCKETS, 1), self.corrected_data, k=4, s=smoothness) #Need min order 4 spline; deriv -> 3 order, min needed for roots
        self.smoothed_output = self.smoothing_spline(np.arange(0, NUMBER_OF_TIME_BUCKETS, 1))
        

    def is_valid(self) -> bool:
        return self.hw_id.pad_id != INVALID_PAD_ID and isinstance(self.raw_data, np.ndarray)

    def get_pad_id(self) -> int:
        return self.hw_id.pad_id
    
    def evaluate_baseline(self, window_scale: float) -> np.ndarray:
        '''
            Calculate the baseline of a trace using Fast Fourier Transforms
            Create a moving average of the baseline, after removing peaks from the signal.
            A peak here is defined as any region which is 1.5 std. deviations above the mean of the signal.
            Algorithm taken from pytpc by Josh Bradt, et al.

            ## Parameters
            window_scale: float, sets the scale of the averaging window. Larger values correspond to smaller windows. Default is 20
            ## Returns
            ndarray: the baseline array
        '''
        base = self.raw_data.copy()
        sigma = base.std()
        mask = base - np.mean(base) > sigma * 1.5
        base[mask] = base[~mask].mean()

        window_range = np.arange(-256, 256, 1)
        filter = ifftshift(np.sinc(window_range/window_scale))
        transformed = fft(base)
        self.untransformed = ifft(transformed * filter)
        return self.untransformed.real
    
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
                    peak.amplitude = self.corrected_data[peak_bucket]
                    peak.uncorrected_amplitude = self.raw_data[peak_bucket]
                    peak.integral = np.sum(self.corrected_data[pi_bucket:(ni_bucket+1)], dtype=np.float64)
                    if (peak.amplitude > threshold):
                        self.peaks.append(peak)
                    break

        #Get rid of peaks from saturated trace (NMT)
        #temp_arr = np.array([[Peak.positive_inflection, Peak.negative_inflection, Peak.uncorrected_amplitude] for Peak in self.peaks])
        #unique, counts = np.unique(temp_arr, axis = 0, return_counts = True)
        #duplicates = unique[counts > 1].tolist()
        #self.peaks = [Peak for Peak in self.peaks if np.logical_and(~(np.isin([Peak.positive_inflection, Peak.negative_inflection, Peak.uncorrected_amplitude], duplicates).all()), Peak.uncorrected_amplitude < 4095)]

        if len(self.peaks) > 0:
            return True
        else:
            self.peaks.clear()
            return False
        
    def get_number_of_peaks(self) -> int:
        if self.peaks is None:
            return 0
        else:
            return len(self.peaks)
    
    def get_peaks(self) -> list[Peak]:
        return self.peaks
