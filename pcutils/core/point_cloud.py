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
        count = 0
        for trace in event.traces:
            if trace.hw_id.cobo_id != 10:
                count += trace.get_number_of_peaks()
        self.cloud = np.zeros((count, 7))
        idx = 0
        for trace in event.traces:
            if trace.get_number_of_peaks() == 0 or trace.hw_id.cobo_id == 10:
                continue
            pad = pmap.get_pad_data(trace.hw_id.pad_id)
            for peak in trace.get_peaks():
                self.cloud[idx, 0] = pad.x # X-coordinate, geometry
                self.cloud[idx, 1] = pad.y # Y-coordinate, geometry
                self.cloud[idx, 2] = peak.centroid + pad.time_offset # Z-coordinate, time with correction until calibrated with calibrate_z_position()
                self.cloud[idx, 3] = peak.amplitude
                self.cloud[idx, 4] = peak.integral * pad.gain
                self.cloud[idx, 5] = trace.hw_id.pad_id
                self.cloud[idx, 6] = peak.centroid + pad.time_offset # Time bucket, stored for later use
                idx += 1

    def load_cloud_from_hdf5_data(self, data: np.ndarray, event_number: int):
        self.event_number: int = event_number
        self.cloud = data

    def is_valid(self) -> bool:
        return self.event_number != INVALID_EVENT_NUMBER

    def retrieve_spatial_coordinates(self) -> np.ndarray:
        return self.cloud[:, 0:3]
    
    def eliminate_cross_talk(self, pmap: PadMap, params: CrossTalkParameters):
        '''
        Routine to attempt to eliminate cross talk. Adapted from the IgorPro analysis routine written by Z. Serikow.
        First, find traces which were estimated to saturate. Then check the neighboring channels in the electronics to see if they
        had a signal which occured at the same sample time. If the neighbor has such a signal, it is said to be a cross talk suspect. Check the pad neighborhood of the
        suspect to see if these other proximal pads saw a signal as well. If they did, the suspect is not cross talk. If they did not, the suspect is considered
        cross talk and rejected.

        ## Parameters
        pmap: PadMap, the pad information
        params: CrossTalkParameters, the parameters for the cross talk algorithm
        '''
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
        '''
        Calibrate the point cloud z-poisition using a known time calibration for the window and micromegas
        '''
        for idx, point in enumerate(self.cloud):
            self.cloud[idx][2] = (window_tb - point[2]) / (window_tb - micromegas_tb) * detector_length

    def smooth_cloud(self, max_distance: float = 10.0):
        '''
        Smooth the point cloud by averaging over nearest neighbors, weighted by the integrated charge.

        ## Parameters
        max_distance: float, the maximum distance between two neighboring points
        '''
        smoothed_cloud = np.zeros(self.cloud.shape)
        for idx, point in enumerate(self.cloud):
            mask = np.linalg.norm((self.cloud[:, :3] - point[:3]), axis=1) < max_distance
            neighbors = self.cloud[mask]
            if len(neighbors) == 0:
                continue
            # Weight points
            xs = np.sum(neighbors[:,0] * neighbors[:,4])
            ys = np.sum(neighbors[:,1] * neighbors[:,4])
            zs = np.sum(neighbors[:,2] * neighbors[:,4])
            cs = np.sum(neighbors[:,3])
            ics = np.sum(neighbors[:,4])
            if np.isclose(ics, 0.0):
                continue
            #smoothed_pc.append(np.average(neighbors, axis = 0))
            smoothed_cloud[idx] = np.array([xs/ics, ys/ics, zs/ics, cs/len(neighbors), ics/len(neighbors), point[5], point[6]])
        # Removes duplicate points
        smoothed_cloud = smoothed_cloud[smoothed_cloud[:, 3] != 0.0]
        _, indicies = np.unique(np.round(smoothed_cloud[:, :3], decimals=2), axis=0, return_index=True)
        self.cloud = smoothed_cloud[indicies]

    def sort_in_z(self):
        indicies = np.argsort(self.cloud[:, 2])
        self.cloud = self.cloud[indicies]

    def drop_isolated_points(self, neighborhood_radius: float = 15.0, min_neighbors: int = 5):
        mask = np.full(shape=(len(self.cloud)), fill_value=False)
        for idx, point in enumerate(self.cloud):
            neighbors = np.linalg.norm((self.cloud[:, :3] - point[:3]), axis=1) < neighborhood_radius
            mask[idx] = len(self.cloud[neighbors]) >= min_neighbors
        self.cloud = self.cloud[mask]

    def bin_cloud_z(self, fractional_bin_size = 0.05):
        sigma_z = np.std(self.cloud[:, 2])
        bin_width = sigma_z * fractional_bin_size

        bin_mins = np.arange(np.min(self.cloud[:, 2]), np.max(self.cloud[:, 2]), step=bin_width)
        binned_cloud = np.full((len(bin_mins), 7), np.nan, dtype=np.float64)
        for idx, z_low_edge in enumerate(bin_mins):
            z_diffs = z_low_edge - self.cloud[:, 2]
            which_points_in_bin = np.logical_and(z_diffs < bin_width, z_diffs >= 0.0)
            points_in_bin = self.cloud[which_points_in_bin]
            if len(points_in_bin) == 0:
                continue

            r = np.linalg.norm(points_in_bin[:, :2], axis=1)
            mean_r = np.mean(r)
            std_r  = np.std(r)
            if std_r != 0.0:
                points_in_bin = points_in_bin[np.abs(r - mean_r) < std_r]
            if len(points_in_bin) == 0:
                continue

            binned_cloud[idx, 0] = np.mean(points_in_bin[:, 0])
            binned_cloud[idx, 1] = np.mean(points_in_bin[:, 1])
            binned_cloud[idx, 2] = z_low_edge + 0.5 * bin_width
            binned_cloud[idx, 3] = np.max(points_in_bin[:, 3])
            binned_cloud[idx, 4] = np.max(points_in_bin[:, 4])
            binned_cloud[idx, 5] = 0.0 #this has no meaning any more
            binned_cloud[idx, 6] = np.mean(points_in_bin[:, 6])

        self.cloud = binned_cloud[~np.isnan(binned_cloud[:, 0])]