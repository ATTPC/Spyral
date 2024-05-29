from .point_cloud import PointCloud
from .config import ClusterParameters
import numpy as np
from dataclasses import dataclass, field
from sklearn.neighbors import LocalOutlierFactor
from scipy.interpolate import BSpline, make_smoothing_spline


@dataclass
class LabeledCloud:
    """Utility dataclass just for temporary holding in the clustering algorithims

    Attributes
    ----------
    label: int
        The label from the clustering algorithm
    point_cloud:
        The cluster data in original point cloud coordinates
    clustered_data:
        The acutal data clustering was perfomed on, in transformed coordinates
    parent_indicies:
        The incidies of this cluster's data in the original parent point cloud
    """

    label: int = -1  # default is noise label
    point_cloud: PointCloud = field(default_factory=PointCloud)
    clustered_data: np.ndarray = field(default_factory=lambda: np.zeros(0))
    parent_indicies: np.ndarray = field(default_factory=lambda: np.zeros(0))


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
    Cluster(event: int=-1, label: int=-1, data: ndarray=np.empty(0, type=numpy.float64))
        Construct a Cluster
    from_labeled_cloud(cloud: LabeledCloud, params: ClusterParams)
        load the data from a LabeledCloud
    copy_cloud(cloud: PointCloud)
        copy the data from a PointCloud to the Cluster
    drop_outliers()
        Use the scikit-learn LocalOutlierFactor to identify and remove outliers in the Cluster
    """

    def __init__(
        self,
        event: int = -1,
        label: int = -1,
        data: np.ndarray = np.empty(0, dtype=np.float64),
    ):
        self.event = event
        self.label = label
        self.data = data
        self.x_spline: BSpline | None = None
        self.y_spline: BSpline | None = None
        self.c_spline: BSpline | None = None

    def from_labeled_cloud(
        self, cloud: LabeledCloud, params: ClusterParameters
    ) -> np.ndarray:
        """Convert a LabeledCloud to a Cluster, dropping any outliers

        Parameters
        ----------
        cloud: LabeledCloud
            The LabeledCloud to convert
        params: ClusterParameters
            Configuration parameters for the cluster

        Returns
        -------
        numpy.ndarray
            The indicies of points labeled outliers
        """
        self.event = cloud.point_cloud.event_number
        self.label = cloud.label
        self.copy_cloud(cloud.point_cloud)
        return self.drop_outliers(params.outlier_scale_factor)

    def copy_cloud(self, cloud: PointCloud):
        """Copy PointCloud data to the cluster

        Copy a subset of the point cloud into the Cluster
        Only keep position, integrated charge

        Parameters
        ----------
        cloud: PointCloud
            The PointCloud to copy from
        """
        cloud.sort_in_z()
        self.data = np.zeros((len(cloud.cloud), 5))
        self.data[:, :3] = cloud.cloud[:, :3]  # position
        self.data[:, 3] = cloud.cloud[:, 4]  # peak integral
        self.data[:, 4] = cloud.cloud[:, 7]  # scale (big or small)

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
    cluster = Cluster()
    outliers = cluster.from_labeled_cloud(cloud, params)
    return (cluster, outliers)
