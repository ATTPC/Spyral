from .point_cloud import PointCloud
from .config import ClusterParameters
import sklearn.cluster as skcluster
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, field
import numpy as np

@dataclass
class ClusteredCloud:
    label: int = -1 #default is noise label
    point_cloud: PointCloud = field(default_factory=PointCloud)

def least_squares_circle(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    '''
    Implementation of analytic least squares circle fit. Taken from the scipy cookbooks.

    ## Parameters
    x: ndarray, list of all x position coordinates to be fit
    y: ndarray, list of all y position coordinates to be fit

    ## Returns
    tuple[float, float, float, float]: A four member tuple containing the center x-coordinate, the center y-coordinate, the radius, and the RMSE (in that order)
    These are NaN if the matrix is singular
    '''
    mean_x = x.mean()
    mean_y = y.mean()

    #Reduced coordinates
    u = x - mean_x
    v = y - mean_y

    #linear system defining the center (uc, vc) in reduced coordinates
    # Suu * uc + Suv * vc = (Suuu + Suvv)/2
    # Suv * uc + Svv * vc = (Suuv + Svvv)/2
    # => A * c = B
    Suv = np.sum(u*v)
    Suu = np.sum(u**2.0)
    Svv = np.sum(v**2.0)
    Suuv = np.sum(u**2.0 * v)
    Suvv = np.sum(u * v**2.0)
    Suuu = np.sum(u**3.0)
    Svvv = np.sum(v**3.0)

    matrix_a = np.array([[Suu, Suv], [Suv, Svv]])
    matrix_b = np.array([(Suuu + Suvv)*0.5, (Suuv + Svvv)*0.5])
    c = None
    try:
        c = np.linalg.solve(matrix_a, matrix_b)
    except:
        return (np.nan, np.nan, np.nan, np.nan)

    xc = c[0] + mean_x
    yc = c[1] + mean_y
    radii = np.sqrt((x - xc)**2.0 + (y - yc)**2.0)
    mean_radius = np.mean(radii)
    residual = np.sum((radii - mean_radius)**2.0)
    return (xc, yc, mean_radius, residual)


def join_clusters(clusters: list[ClusteredCloud], params: ClusterParameters) -> list[ClusteredCloud]:
    '''
    Combine clusters based on the center around which they orbit. This is necessary because often times tracks are
    fractured or contain regions of varying density which causes clustering algorithms to separate them.

    ## Paramters
    clusters: list[ClusteredCloud], the set of clusters to examine
    params: ClusterParameters, contains the parameters controlling the joining algorithm (max_center_distance)

    ## Returns
    list[ClusteredCloud]: the set of joined clusters
    '''
    #Can't join 1 or 0 clusters
    if len(clusters) < 2:
        return clusters
    
    event_number = clusters[0].point_cloud.event_number

    #Fit the clusters with circles
    centers = np.zeros((len(clusters), 3))
    for idx, cluster in enumerate(clusters):
        centers[idx, 0], centers[idx, 1], centers[idx, 2], _ = least_squares_circle(cluster.point_cloud.cloud[:, 0], cluster.point_cloud.cloud[:, 1])

    #Make a dictionary of center groups
    #First everyone is in their own group
    groups: dict[int, list[int]] = {}
    for idx, cluster in enumerate(clusters):
        groups[cluster.label] = [idx]

    #Now regroup, searching for clusters which match centers
    for idx, center in enumerate(centers):
        cluster = clusters[idx]
        #Reject noise
        if cluster.label == -1 or np.isnan(center[0]) or center[2] < 10.0:
            continue

        for cidx, comp_cluster in enumerate(clusters):
            comp_center = centers[cidx]
            if comp_cluster.label == -1 or np.isnan(comp_center[0]) or center[2] < 10.0:
                continue
            center_distance = np.sqrt((center[0] - comp_center[0])**2.0 + (center[1] - comp_center[1])**2.0)
            #If we find matching centers (that havent already been matched) take all of the clouds in both groups and merge them
            comp_mean_charge = np.mean(comp_cluster.point_cloud.cloud[:, 4], axis=0)
            mean_charge = np.mean(cluster.point_cloud.cloud[:, 4], axis=0)
            charge_diff = np.abs(mean_charge - comp_mean_charge)
            threshold = params.fractional_charge_threshold * np.max([comp_mean_charge, mean_charge])
            if (center_distance < params.max_center_distance) and (cidx not in groups[cluster.label]) and (charge_diff < threshold):
                comp_group = groups.pop(comp_cluster.label)
                for subs in comp_group:
                    clusters[subs].label = cluster.label
                groups[cluster.label].extend(comp_group)
    
    #Now reform the clouds such that there is one cloud per group
    new_clusters: list[ClusteredCloud] = []
    for g in groups.keys():
        if g == -1:
            continue

        new_cluster = ClusteredCloud(g, PointCloud())
        new_cluster.point_cloud.event_number = event_number
        new_cluster.point_cloud.cloud = np.zeros((0,7))
        for idx in groups[g]:
            new_cluster.point_cloud.cloud = np.concatenate((new_cluster.point_cloud.cloud, clusters[idx].point_cloud.cloud), axis=0)
        new_clusters.append(new_cluster)
    
    return new_clusters
    
def clusterize(pc: PointCloud, cluster_params: ClusterParameters) -> list[ClusteredCloud]:
    '''
    Analyze a point cloud, and group the points into clusters which in principle should correspond to particle trajectories. This analysis contains several steps,
    and revolves around the HDBSCAN clustering algorithm implemented in scikit-learn (see [their description](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html) for details)
    First the point cloud is smoothed by averaging over nearest neighbors (defined by the smoothing_neighbor_distance parameter) to remove small deviations. 
    The data is then scaled where each coordinate (x,y,z,int) is centered to its mean and then scaled to its std deviation using the scikit-learn StandardScaler. This data is then
    clustered by HDBSCAN and the clusters are returned.

    ## Parameters
    pc: PointCloud, the cloud to be clustered
    cluster_params: ClusterParameters, parameters controlling the clustering algorithms

    ## Returns
    list[ClusteredCloud]: list of clusters found by the algorithm
    '''
    clusterizer = skcluster.HDBSCAN(min_cluster_size=cluster_params.min_size, min_samples=cluster_params.min_points, cluster_selection_epsilon=cluster_params.fractional_distance_min)

    #Smooth out the point cloud by averaging over neighboring points within a distance, droping any duplicate points
    pc.smooth_cloud(cluster_params.smoothing_neighbor_distance)

    #Use spatial dimensions and integrated charge
    cluster_data = np.empty(shape=(len(pc.cloud), 4))
    cluster_data[:, :3] = pc.cloud[:, :3]
    cluster_data[:, 3] = pc.cloud[:, 4]
    
    #Unfiy feature ranges to their means and std deviations. StandardScaler calculates mean, and std for each feature 
    cluster_data = StandardScaler().fit(cluster_data).transform(cluster_data)

    fitted_clusters = clusterizer.fit(cluster_data)
    labels = np.unique(fitted_clusters.labels_)

    #Select out data into clusters
    clusters: list[ClusteredCloud] = []
    for idx, label in enumerate(labels):
        clusters.append(ClusteredCloud(label, PointCloud()))
        mask = fitted_clusters.labels_ == label
        clusters[idx].point_cloud.cloud = pc.cloud[mask]
        clusters[idx].point_cloud.event_number = pc.event_number

    return clusters