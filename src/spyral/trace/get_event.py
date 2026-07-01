from .get_trace import GetTrace
from ..core.config import GetParameters, AVAILABLE_MAPS
from ..core.constants import INVALID_EVENT_NUMBER
from ..core.hardware_id import hardware_id_from_array, validation_merged_padmap,fix_array_with_padmap

import numpy as np
from numba import njit, objmode
from scipy.ndimage import median_filter

GET_DATA_TRACE_START: int = 5
GET_DATA_TRACE_STOP: int = 512 + 5


class GetEvent:
    """Class representing an event in the GET DAQ

    Contains traces (GetTraces) from the AT-TPC pad plane.

    Parameters
    ----------
    raw_data: h5py.Dataset
        The hdf5 Dataset that contains trace data
    event_number: int
        The event number
    params: GetParameters
        Configuration parameters controlling the GET signal analysis
    rng: numpy.random.Generator
        A random number generator for use with the signal analysis


    Attributes
    ----------
    traces: list[GetTrace]
        The pad plane traces from the event
    number:
        The event number
    padmap_validated: 
        Checks if the padmap matches with the merged file. True h5file and map matches. 

    Methods
    -------
    GetEvent(raw_data: h5py.Dataset, event_number: int, params: GetParameters, rng: numpy.random.Generator)
        Construct the event and process traces
    is_valid() -> bool
        Check if the event is valid
    is_merged_map_correct() -> bool
        Checks if the data was merged with the padmap sequence as the predefault Spyral. True - correct
    """

    def __init__(
        self,
        raw_data: np.ndarray,
        event_number: int,
        params: GetParameters,
        rng: np.random.Generator,
    ):
        self.traces: list[GetTrace] = []
        self.number = event_number
        # Baseline correction
        if params.trace_version == 'v0' or params.trace_version == 'default':
            trace_matrix = preprocess_traces(
                raw_data[:, GET_DATA_TRACE_START:GET_DATA_TRACE_STOP].copy(),
                params.baseline_window_scale,
            )
        elif params.trace_version == 'v1': 
            trace_matrix = preprocess_traces_v1(
                raw_data[:, GET_DATA_TRACE_START:GET_DATA_TRACE_STOP].copy(),
                params.baseline_window_scale,
                params.peak_threshold,
            )
        else:
            raise Exception(f"Trace version {params.trace_version} is not valid! Use v0 or v1.")
        
        
        # Fix traces order using the PADMAP from the experiment.
        path_in_available_maps = False
        for nmap in AVAILABLE_MAPS:
            if nmap == params.padmap:
                self.padmap_validated = validation_merged_padmap(raw_data[:,:5], nmap)
                raw_data = fix_array_with_padmap(raw_data,nmap) 
                path_in_available_maps = True
        # Create an option in case the user has their own map, and it isn't available in the current Spyral version
        if not path_in_available_maps: 
            self.padmap_validated = validation_merged_padmap(raw_data[:,:5], params.padmap)
            raw_data = fix_array_with_padmap(raw_data, params.padmap)

        self.traces = [
            GetTrace(trace_matrix[idx], hardware_id_from_array(row[0:5]), params, rng)
            for idx, row in enumerate(raw_data)
        ]

    def is_valid(self) -> bool:
        return self.number != INVALID_EVENT_NUMBER
    
    def is_merged_map_correct(self) ->bool:
        return self.padmap_validated

@njit
def preprocess_traces(traces: np.ndarray, baseline_window_scale: float) -> np.ndarray:
    """JIT-ed Method for pre-cleaning the trace data in bulk before doing trace analysis

    These methods are more suited to operating on the entire dataset rather than on a trace by trace basis
    It includes

    - Removal of edge effects in traces (first and last time buckets can be noisy)
    - Baseline removal via fourier transform method (see J. Bradt thesis, pytpc library)

    Parameters
    ----------
    traces: ndarray
        A (n, 512) matrix where n is the number of traces and each row corresponds to a trace. This should be a copied
        array, not a reference to an array in an hdf file
    baseline_window_scale: float
        The scale of the baseline filter used to perform a moving average over the basline

    Returns
    -------
    ndarray
        A new (n, 512) matrix which contains the traces with their baselines removed and edges smoothed
    """
    # Smooth out the edges of the traces
    traces[:, 0] = traces[:, 1]
    traces[:, -1] = traces[:, -2]

    # Remove peaks from baselines and replace with average
    bases: np.ndarray = traces.copy()
    for row in bases:
        mean = np.mean(row)
        sigma = np.std(row)
        mask = row - mean > sigma * 1.5
        row[mask] = np.mean(row[~mask])

    # Create the filter
    window = np.arange(-256.0, 256.0, 1.0)
    fil = np.fft.ifftshift(np.sinc(window / baseline_window_scale))
    transformed = np.fft.fft2(bases, axes=(1,))
    result = np.real(
        np.fft.ifft2(transformed * fil, axes=(1,))
    )  # Apply the filter -> multiply in Fourier = convolve in normal

    return traces - result




@njit
def preprocess_traces_v1(traces: np.ndarray, baseline_window_scale: float, peak_threshold: float, edge: int = 5) -> np.ndarray:
    """JIT-ed Method Version 2.
    - Artifact 1 edge effect: Removal of edge effects in traces by a controled by a broaded edge time bucket range 
    - Artifact 2 phase effect: Estimation of the baseline mean with a robust method: median_filter (not JIT-ed)
        - Now the estiamted baseline define the peak cutoff from the input peak_threshold. 
        - Baseline uses the same FFT method as the preprocess_traces(). 
    - Artifact 3 memory error: Signal goes to zero in ADC (amplitude)

    Parameters
    ----------
    traces: ndarray
        A (n, 512) matrix where n is the number of traces and each row corresponds to a trace. This should be a copied
        array, not a reference to an array in an hdf file
    baseline_window_scale: float
        The scale of the baseline filter used to perform a moving average over the basline
    peak_treshold: float 
        The peak treshold added for the peak finding is now used to estimate the peaks above the baseline. 
    edge: int 
        Accounts N time buckets of the edges to tackle edge effect. 


    Returns
    -------
    ndarray
        A new (n, 512) matrix which contains the traces with their baselines removed and edges smoothed
    """

    # Remove peaks from baselines and replace with average
    bases: np.ndarray = traces.copy()
    row_number = 0
    for row in bases:
        # Get mean amplitude and remove memory error pads
        mean = np.zeros_like(row, dtype =np.float64)
        mask = row!=0 # false-> 0 amplitude peaks
        mean_no_0= np.zeros_like(row, dtype =np.float64)
        with objmode(mean_no_0='float64[:]'):
            # Get mean of the trace without 0 peaks for a better mean 
                f0= median_filter(row[mask],size=512) 
                mean_no_0 = np.ascontiguousarray(f0,dtype=np.float64)
        row[~mask] = np.mean(mean_no_0)

        with objmode(mean='float64[:]'):
            # use median filter as a method to remove phase offset artifact from GET electronics
            f = median_filter(row, size=512)
            mean = np.ascontiguousarray(f,dtype=np.float64)

        # Get mean amplitude (excluding real signals)
        mask = row - mean > peak_threshold
        # include peaks from the edges
        mask[:edge] = False
        mask[-edge:] = False
        # remove draft real signals from the mean
        row[mask] = np.mean(row[~mask])
        row_number +=1


    # Create the filter
    window = np.arange(-256.0, 256.0, 1.0)
    fil = np.fft.ifftshift(np.sinc(window / baseline_window_scale))
    transformed = np.fft.fft2(bases, axes=(1,))
    baseline = np.real(
        np.fft.ifft2(transformed * fil, axes=(1,))
    )  # Apply the filter -> multiply in Fourier = convolve in normal

    # Include the "real" trace from the edges into the baseline 
    baseline[:,:edge] = traces[:,:edge]
    baseline[:,-edge:] = traces[:,-edge:]


    # include the memory error signals to the rightful timebucket 
    for i in range(baseline.shape[0]):
        for j in range(baseline.shape[1]):
            if traces[i, j] == 0:
                baseline[i, j] = 0 

    return traces - baseline
