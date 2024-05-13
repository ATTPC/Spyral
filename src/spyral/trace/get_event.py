from .get_trace import GetTrace
from ..core.config import GetParameters
from ..core.constants import INVALID_EVENT_NAME, INVALID_EVENT_NUMBER
from ..core.hardware_id import hardware_id_from_array

import numpy as np
import h5py as h5
from numba import njit

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
    name: str
        The event name
    number:
        The event number

    Methods
    -------
    GetEvent(raw_data: h5py.Dataset, event_number: int, params: GetParameters, rng: numpy.random.Generator)
        Construct the event and process traces
    load_traces(raw_data: h5py.Dataset, event_number: int, params: GetParameters, rng: numpy.random.Generator)
        Process traces
    is_valid() -> bool
        Check if the event is valid
    """

    def __init__(
        self,
        raw_data: h5.Dataset,
        event_number: int,
        params: GetParameters,
        rng: np.random.Generator,
    ):
        self.traces: list[GetTrace] = []
        self.name: str = INVALID_EVENT_NAME
        self.number: int = INVALID_EVENT_NUMBER
        self.load_traces(raw_data, event_number, params, rng)

    def load_traces(
        self,
        raw_data: h5.Dataset,
        event_number: int,
        params: GetParameters,
        rng: np.random.Generator,
    ):
        """Process the traces

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

        """
        self.name = str(raw_data.name)
        self.number = event_number
        trace_matrix = preprocess_traces(
            raw_data[:, GET_DATA_TRACE_START:GET_DATA_TRACE_STOP].copy(),
            params.baseline_window_scale,
        )
        self.traces = [
            GetTrace(trace_matrix[idx], hardware_id_from_array(row[0:5]), params, rng)
            for idx, row in enumerate(raw_data)
        ]

    def is_valid(self) -> bool:
        return self.name != INVALID_EVENT_NAME and self.number != INVALID_EVENT_NUMBER


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
