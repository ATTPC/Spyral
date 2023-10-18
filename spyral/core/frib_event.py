from .frib_trace import FribTrace
from .get_trace import Peak
from .config import FribParameters

import h5py as h5
import numpy as np

IC_COLUMN: int = 0
SI_COLUMN: int = 2
MESH_COLUMN: int = 1

SAMPLING_FREQUENCY: float = 12.5 #MHz

class FribEvent:
    '''
    # FribEvent
    Represents an event from the FRIBDAQ system. Contains traces from the SIS3300 module, typically encompassing the ion chamber, 
    silicon detector, and mesh signals.
    '''
    def __init__(self, raw_data: h5.Dataset, event_number: int, params: FribParameters):
        self.event_number = event_number
        self.event_name = str(raw_data.name)
        trace_data = preprocess_frib_traces(raw_data[:].copy(), params.baseline_window_scale)
        self.traces = [FribTrace(column, params) for column in trace_data.T]

    def get_ic_trace(self) -> FribTrace:
        return self.traces[IC_COLUMN]
    
    def get_si_trace(self) -> FribTrace:
        return self.traces[SI_COLUMN]
    
    def get_mesh_trace(self) -> FribTrace:
        return self.traces[MESH_COLUMN]
    
    def get_good_ic_peak(self, params: FribParameters) -> Peak | None:
        '''
        Attempts to retrieve the "good" ion chamber signal. 

        In many cases the ion chamber has multiple peaks due to bunches of beam being delivered to the AT-TPC.
        This poses an issue for the analysis; we must now deterimine which ion chamber signal to use, if any, for
        this event. The algorithm below uses the silicon signal to veto ion chamber signals which correspond to un-reacted beam
        particles and effectively reduce the multiplicity. The user then has control over how many ion chamber signals are
        allowed for a good event. In general, AT-TPC would only accept multiplicty one type events.

        ## Parameters
        params: FribParameters, configuration class that contains the variables controlling the algorithm

        ## Returns
        Peak | None: returns None on failure to find any good ion chamber signal; otherwise returns the Peak associated with a good signal
        '''
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
        si_coinc = False
        for idx, ic in enumerate(ic_peaks):
            si_coinc = False
            for si in si_peaks:
                if abs(ic.centroid - si.centroid) < 50.0:
                    si_coinc = True
                    break
            if not si_coinc:
                good_ic_count += 1
                good_ic_index = idx
        
        if good_ic_count > params.ic_multiplicity or good_ic_count == 0:
            return None
        else:
            return ic_peaks[good_ic_index]
        
    def correct_ic_time(self, good_peak: Peak, get_frequency: float) -> float:
        '''
        Calculate the correction to the GET time buckets using the ion chamber signal

        In a case where the ion chamber fired multiple times, somtimes the wrong
        beam particle starts the event. That is: we trigger on an unreacted beam particle,
        but still capture a good event within the trigger window. After the good ion
        chamber peak is identified using the get_good_ic_peak method, a correction
        for this time walk effect can be calculated by taking the difference between the
        centroids in the good peak and the earliest peak in the ion chamber trace. 
        The difference can then be converted to GET time buckets by multiplying by the
        ratio of the sampling frequency of the GET system and the SIS3300 module.

        ## Parameters:
        good_peak: Peak, the good peak in the ion chamber trace, typically found using the get_good_ic_peak method
        get_frequency: float, the sampling frequency of the GET electronics

        ## Returns:
        float: the correction to the GET time in units of GET time buckets.
        '''

        earliest_ic: Peak = Peak()
        for ic in self.get_ic_trace().get_peaks():
            if ic.centroid < earliest_ic.centroid or earliest_ic.centroid < 0.0:
                earliest_ic = ic
        return (good_peak.centroid - earliest_ic.centroid) * get_frequency/SAMPLING_FREQUENCY
    


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