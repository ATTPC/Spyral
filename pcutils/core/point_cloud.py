from .get_event import GetEvent
from .get_trace import Peak
from .pad_map import PadMap, PadData
from .constants import INVALID_EVENT_NUMBER, INVALID_PEAK_CENTROID
import numpy as np
from typing import Optional

class PointCloud:

    def __init__(self):
        self.event_number: int = INVALID_EVENT_NUMBER
        self.cloud: Optional[np.ndarray] = None

    def load_cloud_from_get_event(self, event: GetEvent, pad_geometry: PadMap, peak_separation: float, peak_threshold: float):
        #self.cloud = np.empty((0,5)) # point elements are x, y, z, height, integral
        count = 0
        for trace in event.traces:
            if trace.find_peaks(peak_separation, peak_threshold):
                count += trace.get_number_of_peaks()
        self.cloud = np.zeros((count, 6))
        idx = 0
        for trace in event.traces:
            if trace.get_number_of_peaks() == 0 or trace.hw_id.cobo_id == 10:
                continue
            pad = pad_geometry.get_pad_data(trace.hw_id.pad_id)
            for peak in trace.get_peaks():
                self.cloud[idx, 0] = pad.x # X-coordinate, geometry
                self.cloud[idx, 1] = pad.y # Y-coordinate, geometry
                self.cloud[idx, 2] = peak.centroid + pad.time_offset # Z-coordinate, time with correction
                self.cloud[idx, 3] = peak.amplitude
                self.cloud[idx, 4] = peak.integral * pad.gain
                self.cloud[idx, 5] = trace.hw_id.pad_id
                idx += 1

    def load_cloud_from_hdf5_data(self, data: np.ndarray, event_number: int):
        self.event_number: int = event_number
        self.cloud = data

    def is_valid(self) -> bool:
        return self.event_number != INVALID_EVENT_NUMBER

    def retrieve_spatial_coordinates(self) -> np.ndarray:
        return self.cloud[:, 0:3]
    
    def eliminate_cross_talk(self, saturation_threshold: float = 2000.0, cross_talk_threshold: float = 1000.0, physical_range: int = 5, bucket_range: int = 10):
        points_to_keep: np.ndarray = np.full(len(self.cloud), fill_value = True, dtype = bool)
        average_neighbor_amplitude = 0.0
        n_neighbors = 1
        for idx, point in enumerate(self.cloud):
            average_neighbor_amplitude = 0.0
            n_neighbors = 0.0
            if point[3] < saturation_threshold:
                continue

            for comp_idx, comp_point in enumerate(self.cloud):
                if comp_idx == idx: #dont include self
                    continue
                elif (comp_point[0] < point[0] - physical_range) or (comp_point[0] > point[0] + physical_range): #xbounds
                    continue
                elif (comp_point[1] < point[1] - physical_range) or (comp_point[1] > point[1] + physical_range): #ybounds
                    continue
                elif (comp_point[2] < point[2] or comp_point[2] > point[2] + bucket_range): #zbounds
                    continue

                average_neighbor_amplitude += comp_point[3]
                n_neighbors += 1
            if n_neighbors > 0:
                average_neighbor_amplitude  /= float(n_neighbors)
                if (average_neighbor_amplitude < cross_talk_threshold):
                    points_to_keep[idx] = False
            else:
                points_to_keep[idx] = False

        self.cloud = self.cloud[points_to_keep]

    def calibrate_z_position(self, micromegas_tb: float, window_tb: float, detector_length: float):
        #for idx, point in enumerate(self.cloud):
            #self.cloud[idx][2] = (window_tb - point[2]) / (window_tb - micromegas_tb) * detector_length
        self.cloud[:,2] = (window_tb - self.cloud[:,2]) / (window_tb - micromegas_tb) * detector_length

    def smooth_cloud(self, max_distance:float = 10.0):
        smoothed_pc = []
        for i in range(len(self.cloud)):
            # Create mask that determines all points within max_distance of the ith point in the cloud
            mask = np.sqrt((self.cloud[:,0]-self.cloud[i,0])**2 + (self.cloud[:,1]-self.cloud[i,1])**2 + (self.cloud[:,2]-self.cloud[i,2])**2) <= max_distance
            neighbors = self.cloud[mask]
            # Weight points
            xs = sum(neighbors[:,0] * neighbors[:,4])
            ys = sum(neighbors[:,1] * neighbors[:,4])
            zs = sum(neighbors[:,2] * neighbors[:,4])
            cs = sum(neighbors[:,3])
            ics = sum(neighbors[:,4])

            if len(neighbors) > 0 and ics != 0:
                smoothed_pc.append(np.array([xs/ics, ys/ics, zs/ics, cs/len(neighbors), ics/len(neighbors)]))

        smoothed_pc = np.vstack(smoothed_pc)
        # Removes duplicate points
        smoothed_pc = smoothed_pc[sorted(np.unique(smoothed_pc.round(decimals = 8), axis = 0, return_index = True)[1])]
        # Removes NaNs
        smoothed_pc = smoothed_pc[~np.isnan(smoothed_pc).any(axis = 1)]
        self.cloud = smoothed_pc
