from .frib_trace import FribTrace
from .get_trace import Peak

import h5py as h5
import numpy as np

IC_COLUMN: int = 0
SI_COLUMN: int = 2
MESH_COLUMN: int = 1

class FribEvent:
    def __init__(self, raw_data: h5.Dataset, event_number: int):
        self.event_number = event_number
        self.event_name = str(raw_data.name)
        trace_data = preprocess_frib_traces(raw_data[:].copy(), 100.0)
        self.traces = [FribTrace(column) for column in trace_data.T]

    def get_ic_trace(self) -> FribTrace:
        return self.traces[IC_COLUMN]
    
    def get_si_trace(self) -> FribTrace:
        return self.traces[SI_COLUMN]
    
    def get_mesh_trace(self) -> FribTrace:
        return self.traces[MESH_COLUMN]
    
    def get_good_ic_peak(self) -> Peak | None:

        ic_peaks = self.get_ic_trace().get_peaks()
        si_peaks = self.get_si_trace().get_peaks()

        if len(ic_peaks) == 0:
            return None
        elif len(si_peaks) == 0:
            if len(ic_peaks) == 1:
                return ic_peaks[0]
            else:
                return None

        good_ic_count = 0
        good_ic_index = -1
        for idx, ic in enumerate(ic_peaks):
            for si in si_peaks:
                if abs(ic.centroid - si.centroid) < 50.0:
                    continue
                else:
                    good_ic_count += 1
                    good_ic_index = idx
        
        if good_ic_count > 1 or good_ic_count == 0:
            return None
        else:
            return ic_peaks[good_ic_index]
    


def preprocess_frib_traces(traces: np.ndarray, baseline_window_scale: float) -> np.ndarray:
    '''
        Method for pre-cleaning the trace data in bulk before doing trace analysis
        
        These methods are more suited to operating on the entire dataset rather than on a trace by trace basis
        It includes
        
        - Removal of edge effects in traces (first and last time buckets can be noisy)
        - Baseline removal via fourier transform method (see J. Bradt thesis, pytpc library)

        ## Parameters
        traces: np.ndarray, a (2048, n) matrix where n is the number of traces and each column corresponds to a trace. This should be a copied
        array, not a reference to an array in an hdf file
        baseline_window_scale: float, the scale of the baseline filter used to perform a moving average over the basline

        ## Returns
        np.ndarray: a new (2048, n) matrix which contains the traces with their baselines removed and edges smoothed
    '''
    #transpose cause its easier to process
    traces = traces.T

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
    window = np.arange(-1024.0, 1024.0, 1.0)
    fil = np.fft.ifftshift(np.sinc(window/baseline_window_scale))
    transformed = np.fft.fft2(bases, axes=(1,))
    result = np.real(np.fft.ifft2(transformed * fil, axes=(1,))) #Apply the filter -> multiply in Fourier = convolve in normal
    
    #Make sure to transpose back into og format
    return (traces - result).T