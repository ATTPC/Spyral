from .get_event import GetEvent
from .pad_map import PadMap
from .constants import INVALID_EVENT_NUMBER
from .config import CrossTalkParameters
import numpy as np
from typing import Optional

class PointCloud:

    def __init__(self):
        self.event_number: int = INVALID_EVENT_NUMBER
        self.cloud: Optional[np.ndarray] = None

    def load_cloud_from_get_event(self, event: GetEvent, pmap: PadMap):
        self.event_number = event.number
        self.cloud = np.empty((0,5)) # point elements are x, y, z, height, integral
        count = 0
        for trace in event.traces:
            if trace.hw_id.cobo_id != 10:
                count += trace.get_number_of_peaks()
        self.cloud = np.zeros((count, 6))
        idx = 0
        for trace in event.traces:
            if trace.get_number_of_peaks() == 0 or trace.hw_id.cobo_id == 10:
                continue
            pad = pmap.get_pad_data(trace.hw_id.pad_id)
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
        self.cloud = data.copy()

    def is_valid(self) -> bool:
        return self.event_number != INVALID_EVENT_NUMBER

    def retrieve_spatial_coordinates(self) -> np.ndarray:
        return self.cloud[:, 0:3]
    
    def eliminate_cross_talk(self, pmap: PadMap, params: CrossTalkParameters):
        points_to_keep: np.ndarray = np.full(len(self.cloud), fill_value=True, dtype=bool)
        average_neighbor_amplitude = 0.0
        n_neighbors = 0
        #First find a saturated pad
        for idx, point in enumerate(self.cloud):
            
            if point[3] < params.saturation_threshold:
                continue

            point_hardware = pmap.get_pad_data(point[5]).hardware
            channel_max = point_hardware.aget_channel + params.channel_range
            if channel_max > 67:
                channel_max = 67
            channel_min = point_hardware.aget_channel - params.channel_range
            if channel_min < 0:
                channel_min = 0

            saturator_time_bucket = point[2]

            #Now look for pads that are in the electronic neighborhood
            for channel in range(channel_min, channel_max+1, 1):
                point_hardware.aget_channel = channel
                suspect_pad = pmap.get_pad_from_hardware(point_hardware)
                if suspect_pad is None:
                    continue
                suspect_point_indicies = np.where(self.cloud[:, 5] == suspect_pad)[0]
                #For each pad in the electronic neighborhood, see if they are cross-talk like
                for suspect_index in suspect_point_indicies:
                    suspect_point = self.cloud[suspect_index]
                    if suspect_point[3] > params.cross_talk_threshold or suspect_point[2] > (saturator_time_bucket + params.time_range) or suspect_point[2] < (saturator_time_bucket - params.time_range):
                        continue
                    average_neighbor_amplitude = 0.0
                    n_neighbors = 0
                    #To ensure cross-talk behavior, check the physical neighborhood. If the neighbors see some signficant signal, probably not cross-talk
                    for comp_idx, comp_point in enumerate(self.cloud):
                        if comp_idx == idx: #dont include self
                            continue
                        elif (comp_point[0] < suspect_point[0] - params.distance_range) or (comp_point[0] > suspect_point[0] + params.distance_range): #xbounds
                            continue
                        elif (comp_point[1] < suspect_point[1] - params.distance_range) or (comp_point[1] > suspect_point[1] + params.distance_range): #ybounds
                            continue
                        elif (comp_point[2] < suspect_point[2] or comp_point[2] > suspect_point[2] + params.time_range): #zbounds
                            continue

                        average_neighbor_amplitude += comp_point[3]
                        n_neighbors += 1
                    if n_neighbors > 0:
                        average_neighbor_amplitude  /= float(n_neighbors)
                        if average_neighbor_amplitude < params.neighborhood_threshold:
                            points_to_keep[suspect_index] = False
                    else:
                        points_to_keep[suspect_index] = False

        self.cloud = self.cloud[points_to_keep]

    def calibrate_z_position(self, micromegas_tb: float, window_tb: float, detector_length: float):
        for idx, point in enumerate(self.cloud):
            self.cloud[idx][2] = (window_tb - point[2]) / (window_tb - micromegas_tb) * detector_length


    def smooth_cloud(self, max_distance: float = 10.0):
        smoothed_cloud = np.zeros(self.cloud.shape)
        for idx, point in enumerate(self.cloud):
            mask = np.sqrt((self.cloud[:,0]-point[0])**2.0+(self.cloud[:,1]-point[1])**2.0+(self.cloud[:,2]-point[2])**2.0) <= max_distance
            neighbors = self.cloud[mask]
            if len(neighbors) == 0:
                continue
            # Weight points
            xs = sum(neighbors[:,0] * neighbors[:,4])
            ys = sum(neighbors[:,1] * neighbors[:,4])
            zs = sum(neighbors[:,2] * neighbors[:,4])
            cs = sum(neighbors[:,3])
            ics = sum(neighbors[:,4])
            #smoothed_pc.append(np.average(neighbors, axis = 0))
            smoothed_cloud[idx] = np.array([xs/ics, ys/ics, zs/ics, cs/len(neighbors), ics/len(neighbors), point[5]])
        # Removes duplicate points
        smoothed_cloud = smoothed_cloud[smoothed_cloud[:, 3] != 0.0]
        self.cloud = np.unique(smoothed_cloud, axis = 0)
