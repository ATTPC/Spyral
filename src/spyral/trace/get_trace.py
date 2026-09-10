from .peak import Peak
from ..core.constants import INVALID_PAD_ID, NUMBER_OF_TIME_BUCKETS
from ..core.config import GetParameters
from ..core.hardware_id import HardwareID

from scipy import signal
import numpy as np
from numba import njit

class GetTraceError(Exception):
    pass

WINDOW = np.linspace(0,NUMBER_OF_TIME_BUCKETS,NUMBER_OF_TIME_BUCKETS)
@njit
def personalized_find_peaks(y_axis : np.ndarray , npeaks: int , peak_separation: float, peak_threshold: float)->tuple[np.ndarray, np.ndarray,np.ndarray]:
    """ 
    Log: Created in 08-21-2026     				 
    Author: Daniela Ramirez (FRIB)				 
    ramirezc@frib.msu.edu    
    
    Preparation for GetTrace.find_spyral that hopefully will replace signal.find_peaks. Please improve this~ 
    The function is jitted and it identifies 10 peaks with the highest amplitude. Without an overlapping 
    peak separation and peak treshhold  

    Parameters
    ----------
    y_axis : np.ndarray
        Array with the amplitude of the peak. 
    npeaks : float 
        Restriction with the amount of peaks accepted per trace. 
    peak_separation : float 
        Minimum separation between peaks, value from GetParameters. 
        Note: For jitted functions the value must be input manually.
    peak_threshold : float 
        Minimum peak amplitude that is accepted, value from GetParameters.

    """
    # Organize counts by counts descending order
    idx_amplitude = np.argsort(y_axis)[::-1]

    # Find peaks that satisfy a separation distance 
    # Two peaks cannot be within the same separation 
    peaks_x = np.empty(npeaks,dtype=np.int64)
    peaks_y = np.empty(npeaks,dtype=np.int32)
    store_idx = np.empty(npeaks,dtype = np.int64)

    n = 0 # maximum Npeaks
    for i in idx_amplitude:
        # Last peak that satisties the minimum condition
        if y_axis[i] <= peak_threshold: 
            break 
        # Filter peaks at the edges of the detector (5 timebuckets)
        position_new_peak = WINDOW[i]
        if not (position_new_peak > 5 and position_new_peak<NUMBER_OF_TIME_BUCKETS-5+1): continue
        valid_peak = True
        # Check among existing peaks 
        for j in range(n):
            if abs(position_new_peak - peaks_x[j]) < peak_separation: 
                valid_peak = False
                break 
        if valid_peak: 
            peaks_x[n] = position_new_peak
            peaks_y[n] = y_axis[i]
            store_idx[n] = i
            n+=1
            if n==npeaks: 
                break
    # Extract the information 
    peaks_x = peaks_x[:n]
    peaks_y = peaks_y[:n]
    store_idx = store_idx[:n]
    
    # Sort ascending order (time order)
    idx_sort = np.argsort(peaks_x)
    return  peaks_x[idx_sort],peaks_y[idx_sort],store_idx[idx_sort]

@njit 
def find_inflection_points(trace: np.ndarray, idx_peaks: np.ndarray, inflections_thresholds: int = 10)->tuple[np.ndarray, np.ndarray]:
    """ 
    Log: Created in 08-21-2026     				 
    Author: Daniela Ramirez (FRIB)				 
    ramirezc@frib.msu.edu    
    
    Preparation for GetTrace.find_spyral that hopefully will replace signal.find_peaks. It locates inflection points and expands the 
    inflections. Also, it doesn't merge the peaks if there is an overlap. That is inside GetTrace.find_peaks_spyral()

    Parameters
    ----------
    y_axis : np.ndarray
        Array with the amplitude of the peak. 
    npeaks : float 
        Restriction with the amount of peaks accepted per trace. 
    params : GetParameters 
        Get Electronics parameters 

    """
    npeaks = idx_peaks.size
    idx_positive_inflection = np.empty(npeaks, dtype=np.int64)
    idx_negative_inflection = np.empty(npeaks, dtype=np.int64)

    for i, p in enumerate(idx_peaks):
        # Positive inflection (left)
        # iterate in the left side of the peak until find a negative slope that is close to the baseline
        left_idx = 0 
        for step in range(p,0,-1):
            moving_left = trace[step] - trace[step-1]
            if moving_left <0.0 or trace[step-1] <= inflections_thresholds: 
                left_idx = step-1 
                break 
        idx_positive_inflection[i] = left_idx

        
        # Negative inflection (right)
        # iterate in the left side of the peak until find a positive slope that is close to the baseline
        right_idx = NUMBER_OF_TIME_BUCKETS - 1
        for step in range(p,NUMBER_OF_TIME_BUCKETS-1):
            moving_right = trace[step] - trace[step+1]
            if moving_right > 0.0 or trace[step+1] <= inflections_thresholds: 
                right_idx = step+1 
                break 
        idx_negative_inflection[i] = right_idx
        
        # Expand width of peak if it isn't close to the baseline
        # Check left side of peak is close to baseline 
        if trace[idx_positive_inflection[i]] > inflections_thresholds: 
            for j in range(idx_positive_inflection[i],-1,-1):
                if trace[j] <= inflections_thresholds:
                    # move to the left in window until it satisties
                    idx_positive_inflection[i] = j -1 
                    break 

        if trace[idx_negative_inflection[i]] > inflections_thresholds: 
            for j in range(idx_negative_inflection[i], 512):
                # move to the left in window until it satisties
                if trace[j]<=inflections_thresholds:
                    idx_negative_inflection[i] = j 
                    break      

    return idx_positive_inflection, idx_negative_inflection  


class GetTrace:
    """A single trace from the GET DAQ data

    Represents a raw signal from the AT-TPC pad plane through the GET data acquisition.

    Parameters
    ----------
    data: ndarray
        The trace data
    id: HardwareID
        The HardwareID for the pad this trace came from
    params: GetParameters
        Configuration parameters controlling the GET signal analysis
    rng: numpy.random.Generator
        A random number generator for use in the signal analysis

    Attributes
    ----------
    trace: ndarray
        The trace data
    peaks: list[Peak]
        The peaks found in the trace
    hw_id: HardwareID
        The hardware ID for the pad this trace came from

    Methods
    -------
    GetTrace(data: ndarray, id: HardwareID, params: GetParameters, rng: numpy.random.Generator)
        Construct the GetTrace and find peaks
    set_trace_data(data: ndarray, id: HardwareID, params: GetParameters, rng: numpy.random.Generator)
        Set the trace data and find peaks
    is_valid() -> bool:
        Check if the trace is valid
    get_pad_id() -> int
        Get the pad id for this trace
    find_peaks(params: GetParameters, rng: numpy.random.Generator, rel_height: float)
        Find the peaks in the trace
    get_number_of_peaks() -> int
        Get the number of peaks found in the trace
    get_peaks(params: GetParameters) -> list[Peak]
        Get the peaks found in the trace
    """

    def __init__(
        self,
        data: np.ndarray,
        id: HardwareID,
        params: GetParameters,
        rng: np.random.Generator,
    ):
        self.trace: np.ndarray = np.empty(0, dtype=np.int32)
        self.peaks: list[Peak] = []
        self.hw_id: HardwareID = HardwareID()
        if isinstance(data, np.ndarray) and id.pad_id != INVALID_PAD_ID:
            self.set_trace_data(data, id, params, rng)

    def set_trace_data(
        self,
        data: np.ndarray,
        id: HardwareID,
        params: GetParameters,
        rng: np.random.Generator,
    ):
        """Set trace data and find peaks

        Parameters
        ----------
        data: ndarray
            The trace data
        id: HardwareID
            The HardwareID for the pad this trace came from
        params: GetParameters
            Configuration parameters controlling the GET signal analysis
        rng: numpy.random.Generator
            A random number generator for use in the signal analysis
        """
        data_shape = np.shape(data)
        if data_shape[0] != NUMBER_OF_TIME_BUCKETS:
            raise GetTraceError(
                f"GetTrace was given data that did not have the correct shape! Expected 512 time buckets, instead got {data_shape[0]}"
            )

        self.trace = data.astype(np.int32)  # Widen the type and sign it
        self.hw_id = id
        if params.find_peaks_method=="scipy":
            self.find_peaks(params, rng)
        elif params.find_peaks_method=="spyral":
            self.find_peaks_spyral(params,rng)
        else: 
            raise Exception(f"Provide a method for finding peaks. Available options: scipy or spyral.")
    def is_valid(self) -> bool:
        """Check if the trace is valid

        Returns
        -------
        bool
            If True the trace is valid
        """
        return self.hw_id.pad_id != INVALID_PAD_ID and isinstance(
            self.trace, np.ndarray
        )

    def get_pad_id(self) -> int:
        """Get the pad id for this trace

        Returns
        -------
        int
            The ID number for the pad this trace came from
        """
        return self.hw_id.pad_id

    def find_peaks(
        self, params: GetParameters, rng: np.random.Generator, rel_height: float = 0.95
    ):
        """Find the peaks in the trace data

        The goal is to determine the centroid location of a signal peak within a given pad trace. Use the find_peaks
        function of scipy.signal to determine peaks. We then use this info to extract peak amplitudes, and integrated charge.

        Note: A random number generator is used to smear the centroids by within their identified time bucket. A time bucket
        is essentially a bin in time over which the signal is sampled. As such, the peak is identified to be on the interval
        [centroid, centroid+1). We sample over this interval to make the data represent this uncertainty.

        Parameters
        ----------
        params: GetParameters
            Configuration paramters controlling the GET signal analysis
        rng: numpy.random.Generator
            A random number generator for use in the signal analysis
        rel_height: float
            The relative height at which the left and right ips points are evaluated. Typically this is
            not needed to be modified, but for some legacy data is necessary
        """

        if not self.is_valid():
            return

        self.peaks.clear()

        pks, props = signal.find_peaks(
            self.trace,
            distance=params.peak_separation,
            prominence=params.peak_prominence,
            width=(1.0, params.peak_max_width),
            rel_height=rel_height,
        )
        for idx, p in enumerate(pks):
            peak = Peak()
            peak.centroid = float(p) + rng.random()/1000.0
            peak.amplitude = float(self.trace[p])
            peak.positive_inflection = int(np.floor(props["left_ips"][idx]))
            peak.negative_inflection = int(np.ceil(props["right_ips"][idx]))
            peak.integral = np.sum(
                np.abs(self.trace[peak.positive_inflection : peak.negative_inflection])
            )
            if peak.amplitude > params.peak_threshold:
                self.peaks.append(peak)

    def find_peaks_spyral(
            self, params: GetParameters, rng: np.random.Generator, rel_height: float = 0.95
        ):
            """Find the peaks in the trace data

            The goal is to determine the centroid location of a signal peak within a given pad trace. Use the find_peaks
            function of scipy.signal to determine peaks. We then use this info to extract peak amplitudes, and integrated charge.

            Note: A random number generator is used to smear the centroids by within their identified time bucket. A time bucket
            is essentially a bin in time over which the signal is sampled. As such, the peak is identified to be on the interval
            [centroid, centroid+1). We sample over this interval to make the data represent this uncertainty.

            Parameters
            ----------
            params: GetParameters
                Configuration paramters controlling the GET signal analysis
            rng: numpy.random.Generator
                A random number generator for use in the signal analysis
            rel_height: float
                The relative height at which the left and right ips points are evaluated. Typically this is
                not needed to be modified, but for some legacy data is necessary
            """

            
            if not self.is_valid():
                return

            self.peaks.clear()

            peaks_x, peaks_y, idx  = personalized_find_peaks(self.trace,10,params.peak_separation, params.peak_threshold)
            if len(peaks_x) ==0: 
                return 
            # Find inflections and stretch 
            inflections_thresholds = 10
            idx_positive_inflection, idx_negative_inflection = find_inflection_points(self.trace, idx, inflections_thresholds)
            
            # Check if there are two overlaps or the peak is at the edge 
            remove_peaks_idx = []
            found_peaks = len(peaks_x)
            for p in range(1, found_peaks):
                # notes: p - post-peak, p-1 - pre-peak
                # Check if the first peak positive (left) and negative (right) inflection encapsulates
                # 1. Peak centroid of another peak (take highest peak)
                # print(peaks_x[p])
                if (WINDOW[idx_negative_inflection[p-1]] >= peaks_x[p]): 
                    if peaks_y[p]>peaks_y[p-1]: 
                        remove_peaks_idx.append(p-1)
                        # idx_negative_inflection[p]= idx_negative_inflection[p-1]# replace right side of peak with the big one
                    elif peaks_y[p]<=peaks_y[p-1]: 
                        remove_peaks_idx.append(p)

            if peaks_x[len(peaks_x)-1] > 510: 
                remove_peaks_idx.append(len(peaks_x)-1)
    

            # new peaks with deleted value 
            peaks_x = np.delete(peaks_x,remove_peaks_idx)
            peaks_y = np.delete(peaks_y,remove_peaks_idx)
            idx_negative_inflection = np.delete(idx_negative_inflection,remove_peaks_idx)
            idx_positive_inflection = np.delete(idx_positive_inflection,remove_peaks_idx)
            
            

            # store information  
            for p in range(len(peaks_x)):     
                peak = Peak()
                peak.centroid = peaks_x[p] + rng.random()/1000.0
                peak.amplitude = float(peaks_y[p])
                peak.positive_inflection = WINDOW[idx_positive_inflection[p]]
                peak.negative_inflection = WINDOW[idx_negative_inflection[p]]
                peak.integral = np.sum(
                    np.abs(self.trace[idx_positive_inflection[p] : idx_negative_inflection[p]])
                )
                self.peaks.append(peak)

    def get_number_of_peaks(self) -> int:
        """Get the number of peaks found in the trace

        Returns
        -------
        int
            Number of found peaks
        """
        return len(self.peaks)

    def get_peaks(self) -> list[Peak]:
        """Get the peaks found in the trace

        Returns
        -------
        list[Peak]
            The peaks found in the trace
        """
        return self.peaks
