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

    def from_labeled_cloud(self, cloud: LabeledCloud, params: ClusterParameters):
        '''
        Convert a LabeledCloud to a Cluster, applying z-binning to the data
        '''
        self.event = cloud.point_cloud.event_number
        self.label = cloud.label
        self.copy_cloud(cloud.point_cloud)
        self.drop_outliers(params.n_neighbors_outiler_test)

    def copy_cloud(self, cloud: PointCloud):
        '''
        Copy point cloud data to the cluster
        '''
        cloud.sort_in_z()
        self.data = np.zeros((len(cloud.cloud), 5))
        self.data[:, :3] = cloud.cloud[:, :3] # position
        self.data[:, 3] = cloud.cloud[:, 4] #integrated charge
        self.data[:, 4] = cloud.cloud[:, 7] #scale

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
        self.data = self.data[result > 0] #label=-1 is an outlier

def convert_labeled_to_cluster(cloud: LabeledCloud, params: ClusterParameters) -> Cluster:
    '''
    Function which takes in a LabeledCloud and ClusterParamters and returns a Cluster
    '''
    cluster = Cluster()
    cluster.from_labeled_cloud(cloud, params)
    return cluster
