from .point_cloud import PointCloud
from .cluster import LabeledCloud, Cluster, convert_labeled_to_cluster
from .config import ClusterParameters
from ..utils.circle import least_squares_circle

import sklearn.cluster as skcluster
from sklearn.preprocessing import StandardScaler
import numpy as np

def join_clusters_depth(clusters: list[LabeledCloud], params: ClusterParameters) -> list[LabeledCloud]:
    '''
    Join clusters until either only one cluster is left or no clusters meet the criteria to be joined together.

    ## Parameters
    clusters: list[LabeledCloud], the set of clusters to examine
    params: ClusterParameters, contains parameters controlling the joining algorithm

    ## Returns
    list[LabeledCloud]: the set of joined clusters
    '''
    jclusters = clusters.copy()
    before = len(jclusters)
    after = 0
    while before != after and len(jclusters) > 1:
        before = len(jclusters)
        jclusters = join_clusters(jclusters, params)
        after = len(jclusters)
    return jclusters



def join_clusters(clusters: list[LabeledCloud], params: ClusterParameters) -> list[LabeledCloud]:
    '''
    Combine clusters based on the center around which they orbit. This is necessary because often times tracks are
    fractured or contain regions of varying density which causes clustering algorithms to separate them.

    ## Paramters
    clusters: list[LabeledCloud], the set of clusters to examine
    params: ClusterParameters, contains the parameters controlling the joining algorithm (max_center_distance)

    ## Returns
    list[LabeledCloud]: the set of joined clusters
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

    #Now regroup, searching for clusters whose circles mostly overlap
    for idx, center in enumerate(centers):
        cluster = clusters[idx]
        #Reject noise
        if cluster.label == -1 or np.isnan(center[0]) or center[2] < 10.0:
            continue
        radius: float = np.linalg.norm(center[:2])
        area = np.pi * radius**2.0

        for cidx, comp_cluster in enumerate(clusters):
            comp_center = centers[cidx]
            comp_radius = np.linalg.norm(comp_center[:2])
            comp_area = np.pi * comp_radius**2.0
            if comp_cluster.label == -1 or np.isnan(comp_center[0]) or center[2] < 10.0 or cidx == idx:
                continue
            
            #Calculate area of overlap between the two circles
            #See Wolfram MathWorld https://mathworld.wolfram.com/Circle-CircleIntersection.html
            center_distance = np.sqrt((center[0] - comp_center[0])**2.0 + (center[1] - comp_center[1])**2.0)
            term1 = (center_distance**2.0 + radius**2.0 - comp_radius**2.0)/(2.0 * center_distance * radius)
            term2 = (center_distance**2.0 + comp_radius**2.0 - radius**2.0)/(2.0 * center_distance * comp_radius)
            term3 = (-center_distance + radius + comp_radius)*(center_distance + radius - comp_radius)*(center_distance - radius + comp_radius)*(center_distance + radius + comp_radius)
            if term3 < 0.0: #term3 cant be negative, inside sqrt
                continue
            term1 = min(1.0, max(-1.0, term1)) #clamp to arccos range to avoid silly floating point precision errors
            term2 = min(1.0, max(-1.0, term2))
            area_overlap = radius**2.0 * np.arccos(term1) + comp_radius**2.0 * np.arccos(term2) - 0.5 * np.sqrt(term3)

            smaller_area = min(area, comp_area)
            comp_mean_charge = np.mean(comp_cluster.point_cloud.cloud[:, 4], axis=0)
            mean_charge = np.mean(cluster.point_cloud.cloud[:, 4], axis=0)
            charge_diff = np.abs(mean_charge - comp_mean_charge)
            threshold = params.fractional_charge_threshold * np.max([comp_mean_charge, mean_charge])
            if (area_overlap > params.circle_overlap_ratio * smaller_area) and (cidx not in groups[cluster.label]) and (charge_diff < threshold):
                comp_group = groups.pop(comp_cluster.label)
                for subs in comp_group:
                    clusters[subs].label = cluster.label
                groups[cluster.label].extend(comp_group)
    
    #Now reform the clouds such that there is one cloud per group
    new_clusters: list[LabeledCloud] = []
    for g in groups.keys():
        if g == -1:
            continue

        new_cluster = LabeledCloud(g, PointCloud())
        new_cluster.point_cloud.event_number = event_number
        new_cluster.point_cloud.cloud = np.zeros((0,8))
        for idx in groups[g]:
            new_cluster.point_cloud.cloud = np.concatenate((new_cluster.point_cloud.cloud, clusters[idx].point_cloud.cloud), axis=0)
        new_clusters.append(new_cluster)

    return new_clusters

def cleanup_clusters(clusters: list[LabeledCloud], params: ClusterParameters) -> list[Cluster]:
    '''
    Converts the LabeledClouds to Clusters and bins the data in z
    '''
    return [convert_labeled_to_cluster(cluster, params) for cluster in clusters if cluster.label != -1]
    
def form_clusters(pc: PointCloud, params: ClusterParameters) -> list[LabeledCloud]:
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
    list[LabeledCloud]: list of clusters found by the algorithm
    '''
    clusterizer = skcluster.HDBSCAN(min_cluster_size=params.min_size, min_samples=params.min_points)

    #Smooth out the point cloud by averaging over neighboring points within a distance, droping any duplicate points
    # pc.drop_outliers(25)
    if len(pc.cloud) < 50:
        return []
    pc.smooth_cloud_neighborship(25)
    # pc.smooth_cloud(params.smoothing_neighbor_distance)


    if len(pc.cloud) < params.min_size:
        return []

    #Use spatial dimensions and integrated charge
    cluster_data = np.empty(shape=(len(pc.cloud), 4))
    cluster_data[:, :3] = pc.cloud[:, :3]
    cluster_data[:, 3] = pc.cloud[:, 4]
    
    #Unfiy feature ranges to their means and std deviations. StandardScaler calculates mean, and std for each feature 
    cluster_data = StandardScaler().fit(cluster_data).transform(cluster_data)

    fitted_clusters = clusterizer.fit(cluster_data)
    labels = np.unique(fitted_clusters.labels_)

    #Select out data into clusters
    clusters: list[LabeledCloud] = []
    for idx, label in enumerate(labels):
        clusters.append(LabeledCloud(label, PointCloud()))
        mask = fitted_clusters.labels_ == label
        clusters[idx].point_cloud.cloud = pc.cloud[mask]
        clusters[idx].point_cloud.event_number = pc.event_number
        
    return clusters