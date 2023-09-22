from .get_trace import GetTrace
from .config import TraceParameters
from .constants import INVALID_EVENT_NAME, INVALID_EVENT_NUMBER
from .hardware_id import hardware_id_from_array
import numpy as np
import h5py

GET_DATA_TRACE_START: int = 5
GET_DATA_TRACE_STOP: int = 512+5

class GetEvent:

    def __init__(self, raw_data: h5py.Dataset, event_number: int, trace_params: TraceParameters):
        self.traces: list[GetTrace] = []
        self.name: str = INVALID_EVENT_NAME
        self.number: int = INVALID_EVENT_NUMBER
        self.load_traces(raw_data, event_number, trace_params)

    def load_traces(self, raw_data: h5py.Dataset, event_number: int, trace_params: TraceParameters):
        self.name = str(raw_data.name)
        self.number = event_number
        trace_matrix = preprocess_traces(raw_data[:, GET_DATA_TRACE_START:GET_DATA_TRACE_STOP].copy(), trace_params.baseline_window_scale)
        self.traces = [GetTrace(trace_matrix[idx], hardware_id_from_array(row[0:5]), trace_params) for idx, row in enumerate(raw_data)]

    def is_valid(self) -> bool:
        return self.name != INVALID_EVENT_NAME and self.number != INVALID_EVENT_NUMBER

def preprocess_traces(traces: np.ndarray, baseline_window_scale: float) -> np.ndarray:
    '''
        Method for pre-cleaning the trace data in bulk before doing trace analysis
        
        These methods are more suited to operating on the entire dataset rather than on a trace by trace basis
        It includes
        
        - Removal of edge effects in traces (first and last time buckets can be noisy)
        - Baseline removal via fourier transform method (see J. Bradt thesis, pytpc library)

        ## Parameters
        traces: np.ndarray, a (n, 512) matrix where n is the number of traces and each row corresponds to a trace. This should be a copied
        array, not a reference to an array in an hdf file
        baseline_window_scale: float, the scale of the baseline filter used to perform a moving average over the basline

        ## Returns
        np.ndarray: a new (n, 512) matrix which contains the traces with their baselines removed and edges smoothed
    '''
    #Smooth out the edges of the traces
    traces[:, 0] = traces[:, 1]
    traces[:, -1] = traces[:, -2]

    #Remove peaks from baselines and replace with average
    bases: np.ndarray = traces.copy()
    for row in bases:
        mean = np.mean(row)
        sigma = np.std(row)
        mask = row - mean > sigma * 1.5
        row[mask] = np.mean(row[~mask])


    #Create the filter
    window = np.arange(-256.0, 256.0, 1.0)
    fil = np.fft.ifftshift(np.sinc(window/baseline_window_scale))
    transformed = np.fft.fft2(bases, axes=(1,))
    result = np.real(np.fft.ifft2(transformed * fil, axes=(1,))) #Apply the filter -> multiply in Fourier = convolve in normal
        
    return traces - result
