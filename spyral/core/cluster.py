from .point_cloud import PointCloud
from .config import ClusterParameters
import numpy as np
from dataclasses import dataclass, field
from sklearn.neighbors import LocalOutlierFactor

@dataclass
class LabeledCloud:
    '''
    # LabeledCloud
    Utility class just for temporary holding in the clustering algorithims
    '''
    label: int = -1 #default is noise label
    point_cloud: PointCloud = field(default_factory=PointCloud)

class Cluster:
    '''
    # Cluster
    Representation of cluster data.
    '''
    def __init__(self, event: int = -1, label: int = -1, data: np.ndarray = np.empty(0, dtype=np.float64)):
        self.event = event
        self.label = label
        self.data = data

        self.z_bin_width = 0.0
        self.z_bin_low_edge = 0.0
        self.z_bin_hi_edge = 0.0
        self.n_z_bins = 0

    def from_labeled_cloud(self, cloud: LabeledCloud, params: ClusterParameters):
        '''
        Convert a LabeledCloud to a Cluster, applying z-binning to the data
        '''
        self.event = cloud.point_cloud.event_number
        self.label = cloud.label
        self.copy_cloud(cloud.point_cloud, params)
        self.drop_outliers()
        # Z-binning is bad! destroys particle ID at higher energies. Maybe needs some tweaking??
        # self.bin_cloud_z(cloud.point_cloud, params)

    def copy_cloud(self, cloud: PointCloud, params: ClusterParameters):
        '''
        Copy point cloud data to the cluster
        '''
        cloud.sort_in_z()
        self.data = np.zeros((len(cloud.cloud), 4))
        self.data[:, :3] = cloud.cloud[:, :3]
        self.data[:, 3] = cloud.cloud[:, 4]

    def bin_cloud_z(self, cloud: PointCloud, params: ClusterParameters):
        '''
        Bin the point cloud data in z and apply some noise removal by rejecting outliers in radial distance within
        the z-bin. This method is adaptive. The number of z-bins is determined by the standard deviation of the data in z
        and a user paramter (z_bin_fractional_size).
        '''
        cloud.sort_in_z()
        sigma_z = np.std(cloud.cloud[:, 2])
        bin_width = sigma_z * params.z_bin_fractional_size

        bin_mins = np.arange(np.min(cloud.cloud[:, 2]), np.max(cloud.cloud[:, 2]), step=bin_width)
        binned_cloud = np.full((len(bin_mins), 7), np.nan, dtype=np.float64)
        for idx, z_low_edge in enumerate(bin_mins):
            z_diffs = cloud.cloud[:, 2] - z_low_edge
            which_points_in_bin = np.logical_and(z_diffs < bin_width, z_diffs >= 0.0)
            points_in_bin = cloud.cloud[which_points_in_bin]
            if len(points_in_bin) == 0:
                continue
            
            if len(points_in_bin) > 2:
                r = np.linalg.norm(points_in_bin[:, :2], axis=1)
                mean_r = np.mean(r)
                std_r  = np.std(r)
                if std_r != 0:
                    points_in_bin = points_in_bin[np.abs(r - mean_r)/std_r < params.z_bin_outlier_cutoff]
            if len(points_in_bin) == 0:
                continue

            binned_cloud[idx, 0] = np.mean(points_in_bin[:, 0])
            binned_cloud[idx, 1] = np.mean(points_in_bin[:, 1])
            binned_cloud[idx, 2] = z_low_edge + 0.5 * bin_width
            binned_cloud[idx, 3] = np.max(points_in_bin[:, 4])

        self.data = binned_cloud[~np.isnan(binned_cloud[:, 0])]
        self.n_z_bins = len(bin_mins)
        self.z_bin_width = bin_width
        self.z_bin_low_edge = bin_mins[0]
        self.z_bin_hi_edge = bin_mins[-1] + bin_width
    
    def drop_outliers(self, neighbors=2):
        '''
        Use scikit-learn LocalOutlierFactor to test the cluster for spatial outliers.
        This helps reduce noise when fitting the data.

        ## Parameters
        neighbors: int, the number of neighbors to compare to for the outlier test
        '''
        test_data = self.data[:, :3].copy()
        neigh = LocalOutlierFactor(n_neighbors=neighbors)
        result = neigh.fit_predict(test_data)
        self.data = self.data[result > 0]

def convert_labeled_to_cluster(cloud: LabeledCloud, params: ClusterParameters) -> Cluster:
    '''
    Function which takes in a LabeledCloud and ClusterParamters and returns a Cluster
    '''
    cluster = Cluster()
    cluster.from_labeled_cloud(cloud, params)
    return cluster