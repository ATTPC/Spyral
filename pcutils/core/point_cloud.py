from .get_event import GetEvent
from .get_trace import Peak
from .pad_map import PadMap
from .constants import INVALID_EVENT_NUMBER, INVALID_PEAK_CENTROID
import numpy as np
from typing import Optional

class PointCloud:

    def __init__(self):
        self.event_number: int = INVALID_EVENT_NUMBER
        self.cloud: Optional[np.ndarray] = None

    def load_cloud_from_get_event(self, event: GetEvent, pad_geometry: PadMap):
        self.cloud = np.empty((0,5)) # point elements are x, y, z, height, integral
        count = 0
        for trace in event.traces:
            if trace.find_peaks():
                count += trace.get_number_of_peaks()
        self.cloud = np.zeros((count, 5))
        idx = 0
        for trace in event.traces:
            if trace.get_number_of_peaks() == 0:
                continue
            pad = pad_geometry.get_pad_data(trace.pad_id)
            for peak in trace.get_peaks():
                self.cloud[idx, 0] = pad.x # X-coordinate, geometry
                self.cloud[idx, 1] = pad.y # Y-coordinate, geometry
                self.cloud[idx, 2] = peak.centroid # Z-coordinate, time
                self.cloud[idx, 3] = peak.amplitude
                self.cloud[idx, 4] = peak.integral * pad.gain
                idx += 1

    def load_cloud_from_hdf5_data(self, data: np.ndarray, event_number: int):
        self.event_number: int = event_number
        self.cloud = data

    def is_valid(self) -> bool:
        return self.event_number != INVALID_EVENT_NUMBER

    def retrieve_spatial_coordinates(self) -> np.ndarray:
        return self.cloud[:, 0:3]