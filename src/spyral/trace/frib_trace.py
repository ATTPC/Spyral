from .peak import Peak
from ..core.config import FribParameters
from ..core.constants import FRIB_TRACE_LENGTH

import numpy as np
from scipy import signal


class FribTrace:
    """A single trace from the FRIBDAQ data

    Similar to GetTrace, FribTrace represents a raw signal from the SIS3300 module which is managed through the FRIBDAQ.
    Typically contains signals for the ion chamber (IC), auxillary silicon detectors (Si), and the mesh signal.

    Parameters
    ----------
    data: ndarray
        The raw trace data
    params: FribParameters
        Configuration parameters controlling the FRIBDAQ signal analysis.

    Attributes
    ----------
    trace: ndarray
        The trace data
    peaks: list[Peak]
        The peaks identified in the trace

    Methods
    -------
    FribTrace(data: ndarray, params: FribParameters)
        Construct the FribTrace and find peaks
    set_trace_data(data: ndarray, params: FribParameters)
        Set the raw trace data and find peaks
    is_valid() -> bool
        Check if this trace has valid data
    find_peaks(params: FribParameters)
        Find peaks in the trace and store the results
    get_number_of_peaks() -> int
        Get the number of peaks found in the trace
    get_peaks() -> list[Peaks]
        Get the found peaks
    """

    def __init__(self, data: np.ndarray, params: FribParameters):
        self.trace: np.ndarray = np.empty(0, dtype=np.int32)
        self.peaks: list[Peak] = []
        self.set_trace_data(data, params)

    def set_trace_data(self, data: np.ndarray, params: FribParameters):
        """Sets the raw trace data and performs peak finding

        Parameters
        ----------
        data: ndarray
            The raw trace data
        params: FribParameters
            Configuration parameters controlling the FRIBDAQ signal analysis.
        """
        if len(data) != FRIB_TRACE_LENGTH:
            print(
                f"Error at ICTrace, trace was given with length {len(data)}, expected {FRIB_TRACE_LENGTH}"
            )
            return

        self.trace = data.astype(np.int32)
        self.find_peaks(params)

    def is_valid(self) -> bool:
        """Check if this trace has valid data

        Returns
        -------
        bool
            If true, the trace is valid. If false, it isn't
        """
        return len(self.trace) != 0

    def find_peaks(self, params: FribParameters):
        """Find peaks in the trace and store the results

        Peak finding algorithm, similar to used in GetTrace. Uses scipy.signal.find_peaks

        Parameters
        ----------
        params: FribParameters
            Configuration parameters which control the peak finding algorithm
        """

        if not self.is_valid():
            return

        self.peaks.clear()
        pks, props = signal.find_peaks(
            self.trace,
            distance=params.peak_separation,
            prominence=params.peak_prominence,
            width=(0, params.peak_max_width),
            rel_height=0.85,
        )
        for idx, p in enumerate(pks):
            peak = Peak()
            peak.centroid = p
            peak.amplitude = float(self.trace[p])
            peak.positive_inflection = int(np.floor(props["left_ips"][idx]))
            peak.negative_inflection = int(np.ceil(props["right_ips"][idx]))
            peak.integral = np.sum(
                np.abs(self.trace[peak.positive_inflection : peak.negative_inflection])
            )
            if peak.amplitude > params.peak_threshold:
                self.peaks.append(peak)

    def get_number_of_peaks(self) -> int:
        """Get the number of peaks found in the trace

        Returns
        -------
        int
            The number of found peaks
        """
        return len(self.peaks)

    def get_peaks(self) -> list[Peak]:
        """Get the found peaks

        Returns
        -------
        list[Peak]
            The found peaks
        """
        return self.peaks
