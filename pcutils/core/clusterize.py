from .point_cloud import PointCloud
from .config import ClusterParameters, DetectorParameters
import sklearn.cluster as skcluster
from circle_fit import least_squares_circle
from dataclasses import dataclass, field
import numpy as np

@dataclass
class ClusteredCloud:
    label: int = -1 #default is noise label
    point_cloud: PointCloud = field(default_factory=PointCloud)

def least_squares_circle(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
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
    if len(clusters) < 2:
        return clusters
    
    event_number = clusters[0].point_cloud.event_number

    centers = np.zeros((len(clusters), 2))
    for idx, cluster in enumerate(clusters):
        centers[idx, 0], centers[idx, 1], _, _ = least_squares_circle(cluster.point_cloud.cloud[:, 0], cluster.point_cloud.cloud[:, 1])

    #Make a dictionary of center groups
    #First everyone is in their own group
    groups: dict[int, list[int]] = {}
    for idx, cluster in enumerate(clusters):
        groups[cluster.label] = [idx]

    #Now regroup, searching for clusters which match centers
    for idx, center in enumerate(centers):
        cluster = clusters[idx]
        if cluster.label == -1 or np.isnan(center[0]):
            continue

        for cidx, comp_cluster in enumerate(clusters):
            comp_center = centers[cidx]
            if comp_cluster.label == -1 or np.isnan(comp_center[0]):
                continue
            center_distance = np.sqrt((center[0] - comp_center[0])**2.0 + (center[1] - comp_center[1])**2.0)
            #If we find matching centers (that havent already been matched) take all of the clouds in both groups and merge them
            if center_distance < params.max_center_distance and cidx not in groups[cluster.label]:
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
        new_cluster.point_cloud.cloud = np.zeros((0,6))
        for idx in groups[g]:
            new_cluster.point_cloud.cloud = np.concatenate((new_cluster.point_cloud.cloud, clusters[idx].point_cloud.cloud), axis=0)
        new_clusters.append(new_cluster)
    
    return new_clusters
    
def clusterize(pc: PointCloud, cluster_params: ClusterParameters, detector_params: DetectorParameters) -> list[ClusteredCloud]:
    clusterizer = skcluster.HDBSCAN(min_cluster_size=cluster_params.min_size, min_samples=cluster_params.min_points)

    #Smooth out the point cloud by averaging over neighboring points within a distance
    pc.smooth_cloud(cluster_params.smoothing_neighbor_distance)

    #Unfiy spatial ranges to be over same scale. Otherwise, scales with greater total range can have differing impact on the clustering algorithm
    max_amplitude = pc.cloud[:, 3].max()
    max_charge = pc.cloud[:, 4].max()

    pc.cloud[:, 0] /= 292.0
    pc.cloud[:, 1] /= 292.0
    pc.cloud[:, 2] /= detector_params.detector_length
    pc.cloud[:, 3] /= max_amplitude
    pc.cloud[:, 4] /= max_charge

    fitted_clusters = clusterizer.fit(pc.cloud[:, :3])
    labels = np.unique(fitted_clusters.labels_)

    #Re-scale
    pc.cloud[:, 0] *= 292.0
    pc.cloud[:, 1] *= 292.0
    pc.cloud[:, 2] *= detector_params.detector_length
    pc.cloud[:, 3] *= max_amplitude
    pc.cloud[:, 3] *= max_charge

    #Select out data into clusters
    clusters: list[ClusteredCloud] = []
    for idx, label in enumerate(labels):
        clusters.append(ClusteredCloud(label, PointCloud()))
        mask = fitted_clusters.labels_ == label
        clusters[idx].point_cloud.cloud = pc.cloud[mask]
        clusters[idx].point_cloud.event_number = pc.event_number

    return clusters