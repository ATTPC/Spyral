from .point_cloud import PointCloud, sort_point_cloud_in_z
from .config import ClusterParameters
import numpy as np
from dataclasses import dataclass
from sklearn.neighbors import LocalOutlierFactor
from scipy.interpolate import BSpline, make_smoothing_spline
from enum import Enum

EMPTY_DATA = np.empty(0, dtype=float)

class Direction(Enum):
    """Enum for the direction of a trajectory

    Attributes
    ----------
    NONE: int
        Invalid value (-1)
    FORWARD: int
        Trajectory traveling in the positive z-direction (0)
    BACKWARD: int
        Trajectory traveling in the negative z-direction (1)
    """

    NONE = -1  # type: ignore
    FORWARD = 0  # type: ignore
    BACKWARD = 1  # type: ignore


@dataclass
class LabeledCloud:
    """Utility dataclass just for temporary holding in the clustering algorithims

    Attributes
    ----------
    label: int
        The label from the clustering algorithm
    point_cloud:
        The cluster data in original point cloud coordinates
    parent_indicies:
        The incidies of this cluster's data in the original parent point cloud
    """

    label: int  # default is noise label
    direction: Direction # default is None
    point_cloud: PointCloud
    parent_indicies: np.ndarray


class Cluster:
    """Representation of trajectory cluster data.

    Parameters
    ----------
    event: int
        The event number (default = -1)
    label: int
        The label from the clustering algorithm (default = -1)
    data: ndarray
        The PointCloud data for the Cluster (default = empty array)

    Attributes
    ----------
    event: int
        The event number
    label: int
        The cluster label from the algorithm
    direction: Direction
        The direction of the cluster
    data: ndarray
        The point cloud data (trimmed down). Contains position, integrated charge
    x_spline: BSpline | None
        An optional spline on z-x to smooth the cluster.
    y_spline: BSpline | None
        An optional spline on z-y to smooth the cluster.
    c_spline: BSpline | None
        An optional spline on z-charge to smooth the cluster.

    Methods
    -------
    apply_smoothing_splines(smoothing=1.0)
        Apply smoothing to the underlying cluster data with smoothing splines
    create_splines(smoothing=1.0)
        Create smoothing splines for the x,y,charge dimensions
    drop_outliers(scale=0.05)
        Use the scikit-learn LocalOutlierFactor to identify and remove outliers in the Cluster
    """

    def __init__(
        self,
        event: int = -1,
        label: int = -1,
        direction: Direction = Direction.NONE,
        data: np.ndarray = EMPTY_DATA,
    ):
        self.event = event
        self.label = label
        self.direction = direction
        self.data = data
        self.x_spline: BSpline | None = None
        self.y_spline: BSpline | None = None
        self.c_spline: BSpline | None = None

    def drop_outliers(self, scale: float = 0.05) -> np.ndarray:
        """Use scikit-learn LocalOutlierFactor to test the cluster for spatial outliers.

        This helps reduce noise when fitting the data.

        Parameters
        ----------
        scale: float
            Scale factor to be multiplied by the length of the trajectory to get
            the number of neighbors over which to test

        Returns
        -------
        numpy.ndarray
            The indicies of points labeled as outliers
        """
        neighbors = int(scale * len(self.data))  # 0.05 default
        if neighbors < 2:
            neighbors = 2
        test_data = self.data[:, :3].copy()
        neigh = LocalOutlierFactor(n_neighbors=neighbors)
        result = neigh.fit_predict(test_data)
        mask = result > 0
        self.data = self.data[mask]  # label=-1 is an outlier
        return np.flatnonzero(~mask)  # Invert the mask to get outliers

    def create_splines(self, smoothing: float = 1.0) -> None:
        """Create smoothing splines for the x,y,charge dimensions

        Create smoothing splines along the z-coordinate for x, y, and charge.
        The degree of smoothing is controlled by the smoothing parameter. smoothing = 0.0 is
        no smoothing (pure interpolation) and higher values gives a higher degree of smoothing.

        Parameters
        ----------
        smoothing: float
            The smoothing factor (lambda in the scipy notation). Must be a positive float or zero.
        """

        self.x_spline = make_smoothing_spline(
            self.data[:, 2], self.data[:, 0], lam=smoothing
        )
        self.y_spline = make_smoothing_spline(
            self.data[:, 2], self.data[:, 1], lam=smoothing
        )
        self.c_spline = make_smoothing_spline(
            self.data[:, 2], self.data[:, 3], lam=smoothing
        )

    def apply_smoothing_splines(self, smoothing: float = 1.0) -> None:
        """Apply smoothing to the underlying cluster data with smoothing splines

        Apply smoothing splines to the x, y, and charge dimensions as a function of
        z. The degree of smoothing is controlled by the smoothing parameter. If the splines
        are not already created using the create_splines function, they will be created here.

        Note: This function modifies the underlying data in the cluster. This is not a reversible operation.

        Parameters
        ----------
        smoothing: float
            The smoothing factor (lambda in the scipy notation). Must be a positive float or zero.

        """

        if self.x_spline is None or self.y_spline is None or self.c_spline is None:
            self.create_splines(smoothing)

        self.data[:, 0] = self.x_spline(self.data[:, 2])  # type: ignore
        self.data[:, 1] = self.y_spline(self.data[:, 2])  # type: ignore
        self.data[:, 3] = self.c_spline(self.data[:, 2])  # type: ignore


def convert_labeled_to_cluster(
    cloud: LabeledCloud, params: ClusterParameters
) -> tuple[Cluster, np.ndarray]:
    """Function which takes in a LabeledCloud and ClusterParamters and returns a Cluster

    Parameters
    ----------
    cloud: LabeledCloud
        The LabeledCloud to convert
    params: ClusterParameters
        Configuration parameters for the cluster

    Returns
    -------
    tuple[Cluster, np.ndarray]
        A two element tuple containing first the Cluster,
        and second a list of indicies in the preciding
        cloud that were labeled as noise.
    """
    # Joining can make point cloud unsorted
    sort_point_cloud_in_z(cloud.point_cloud)
    data = np.zeros((len(cloud.point_cloud), 5))
    data[:, :3] = cloud.point_cloud.data[:, :3]  # position
    data[:, 3] = cloud.point_cloud.data[:, 4]  # peak integral
    data[:, 4] = cloud.point_cloud.data[:, 7]  # scale (big or small)
    cluster = Cluster(cloud.point_cloud.event_number, cloud.label, cloud.direction, data)
    outliers = cluster.drop_outliers(params.outlier_scale_factor)
    return (cluster, outliers)
