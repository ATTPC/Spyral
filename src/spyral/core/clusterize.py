from .point_cloud import PointCloud
from .cluster import LabeledCloud, Cluster, convert_labeled_to_cluster, Direction
from .config import ClusterParameters
from ..geometry.circle import least_squares_circle

import sklearn.cluster as skcluster
import numpy as np
import tripclust
from enum import Enum
from scipy.interpolate import BSpline, make_smoothing_spline
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest


NOISE_LABEL: int = -1

class ClusterError(Exception):
    pass


# Credit Daniela Ramirez-Chavez 2025 ######
def get_direction(cluster: LabeledCloud, params: ClusterParameters):
    """Determine the direction of a cluster based on the 
    differences in angle between adjacent points in z.
    Smoothing is applied to iron out fluctuations and
    a statistical determination is based on the fraction
    of angles greater than a chosen threshold

    Parameters
    ----------
    cluster: LabeledCloud
        the cluster in consideration
    params: ClusterParameters
        contains the parameters related to clustering
    """
    cluster_data = cluster.point_cloud.data.copy()
    if len(cluster_data) > 5:
        x_spline = make_smoothing_spline(cluster_data[:, 2], cluster_data[:, 0], lam=10.0)
        y_spline = make_smoothing_spline(cluster_data[:, 2], cluster_data[:, 1], lam=10.0)
        cluster_data[:, 0] = x_spline(cluster_data[:, 2])  # type: ignore
        cluster_data[:, 1] = y_spline(cluster_data[:, 2])  # type: ignore
    cx, cy, _, _ = least_squares_circle(cluster_data[:,0], cluster_data[:,1])

    angle_right = np.arctan2(cluster_data[:,1] - cy, cluster_data[:,0] - cx)

    for i in range(0,len(angle_right)):
        if angle_right[i]<0:
            angle_right[i]= 2*np.pi + angle_right[i]  

    da_right = np.diff(np.rad2deg(angle_right))

    positive_fraction = sum(1 for num in da_right if num > 0) / len(da_right)
    negative_fraction = 1 - positive_fraction
    if positive_fraction > params.direction_threshold:
        cluster.direction = Direction.FORWARD
    elif negative_fraction > params.direction_threshold:
        cluster.direction = Direction.BACKWARD
    else:
        cluster.direction = Direction.NONE    

def join_clusters(
    clusters: list[LabeledCloud], params: ClusterParameters, labels: np.ndarray
) -> tuple[list[LabeledCloud], np.ndarray]:
    # First determine direction for each cluster
    for cluster in clusters:
        get_direction(cluster, params)
    """Detect which joining algorithm to use and dispatch"""
    if params.continuity_join is not None:
        return join_clusters_continuity(clusters, params, labels)
    elif params.overlap_join is not None:
        return join_clusters_overlap(clusters, params, labels)
    else:
        raise ClusterError(
            "No joining parameters were given! Clustering cannot proceed."
        )


def join_clusters_overlap(
    clusters: list[LabeledCloud], params: ClusterParameters, labels: np.ndarray
) -> tuple[list[LabeledCloud], np.ndarray]:
    """Join clusters until either only one cluster is left or no clusters meet the criteria to be joined together.

    Parameters
    ----------
    clusters: list[LabeledCloud]
        the set of clusters to examine
    params: ClusterParameters
        contains parameters controlling the joining algorithm
    labels: numpy.ndarray
        The cluster label for each point in the original point cloud

    Returns
    -------
    tuple[list[LabeledCloud], numpy.ndarray]
        A two element tuple, the first the list of joined clusters,
        the second being an updated list of labels for each point in
        the point cloud
    """
    jclusters = clusters.copy()
    before = len(jclusters)
    after = 0
    while before != after and len(jclusters) > 1:
        before = len(jclusters)
        jclusters, labels = join_clusters_overlap_step(jclusters, params, labels)
        after = len(jclusters)
    return (jclusters, labels)


def join_clusters_overlap_step(
    clusters: list[LabeledCloud], params: ClusterParameters, labels: np.ndarray
) -> tuple[list[LabeledCloud], np.ndarray]:
    """A single step of joining clusters

    Combine clusters based on the amount of overlap between circles fit to their x-y (pad plane) projection.
    This is necessary because often times tracks are fractured or contain regions of varying density
    which causes clustering algorithms to separate them.

    Parameters
    ----------
    clusters: list[LabeledCloud]
        the set of clusters to examine
    params: ClusterParameters
        contains the parameters controlling the joining algorithm (max_center_distance)
    labels: numpy.ndarray
        The cluster label for each point in the original point cloud

    Returns
    -------
    tuple[list[LabeledCloud], numpy.ndarray]
        A two element tuple, the first the list of joined clusters,
        the second being an updated list of labels for each point in
        the point cloud
    """
    # Can't join 1 or 0 clusters
    if len(clusters) < 2:
        return (clusters, labels)

    event_number = clusters[0].point_cloud.event_number

    # Fit the clusters with circles
    centers = np.zeros((len(clusters), 3))
    for idx, cluster in enumerate(clusters):
        centers[idx, 0], centers[idx, 1], centers[idx, 2], _ = least_squares_circle(
            cluster.point_cloud.data[:, 0], cluster.point_cloud.data[:, 1]
        )

    # Make a dictionary of center groups, label groups
    # First everyone is in their own group
    groups_index: dict[int, list[int]] = {}  # Regroup the actual cluster data
    groups_label: dict[int, list[int]] = {}  # Regroup the label array
    for idx, cluster in enumerate(clusters):
        groups_index[cluster.label] = [idx]  # Use indicies for data
        groups_label[cluster.label] = [cluster.label]  # Use labels for ... labels

    # Now regroup, searching for clusters whose circles mostly overlap
    for idx, center in enumerate(centers):
        cluster = clusters[idx]
        # Reject noise
        if (
            cluster.label == NOISE_LABEL
            or np.isnan(center[0])
            or center[2] < 10.0
            or len(cluster.point_cloud) < params.overlap_join.min_cluster_size_join  # type: ignore
        ):
            continue
        radius = center[2]
        area = np.pi * radius**2.0

        for cidx, comp_cluster in enumerate(clusters):
            comp_center = centers[cidx]
            comp_radius = comp_center[2]
            comp_area = np.pi * comp_radius**2.0
            if (
                comp_cluster.label == NOISE_LABEL
                or np.isnan(comp_center[0])
                or center[2] < 10.0
                or cidx == idx
                or (
                    len(comp_cluster.point_cloud)
                    < params.overlap_join.min_cluster_size_join  # type: ignore
                )
            ):
                continue

            # Calculate area of overlap between the two circles
            # See Wolfram MathWorld https://mathworld.wolfram.com/Circle-CircleIntersection.html
            center_distance = np.sqrt(
                (center[0] - comp_center[0]) ** 2.0
                + (center[1] - comp_center[1]) ** 2.0
            )
            term1 = (center_distance**2.0 + radius**2.0 - comp_radius**2.0) / (
                2.0 * center_distance * radius
            )
            term2 = (center_distance**2.0 + comp_radius**2.0 - radius**2.0) / (
                2.0 * center_distance * comp_radius
            )
            c1 = -center_distance + radius + comp_radius
            c2 = center_distance + radius - comp_radius
            c3 = center_distance - radius + comp_radius
            c4 = center_distance + radius + comp_radius
            term3 = c1 * c2 * c3 * c4
            area_overlap = 0.0
            # term3 cant be negative, inside sqrt.
            if term3 < 0.0 and c1 < 0.0:
                # Cannot possibly overlap, too far apart
                continue
            elif term3 < 0.0 and (c2 < 0.0 or c3 < 0.0):
                # Entirely overlap one circle inside of the other
                area_overlap = min(area, comp_area)
            else:
                # Circles only somewhat overlap
                term1 = min(
                    1.0, max(-1.0, term1)
                )  # clamp to arccos range to avoid silly floating point precision errors
                term2 = min(1.0, max(-1.0, term2))
                area_overlap = (
                    radius**2.0 * np.arccos(term1)
                    + comp_radius**2.0 * np.arccos(term2)
                    - 0.5 * np.sqrt(term3)
                )

            # Apply the condition
            smaller_area = min(area, comp_area)
            if (
                area_overlap > params.overlap_join.circle_overlap_ratio * smaller_area  # type: ignore
            ) and (cidx not in groups_index[cluster.label]) and (cluster.direction == comp_cluster.direction):
                comp_indicies = groups_index.pop(comp_cluster.label)
                comp_labels = groups_label.pop(comp_cluster.label)
                for subs in comp_indicies:
                    clusters[subs].label = cluster.label
                groups_index[cluster.label].extend(comp_indicies)
                groups_label[cluster.label].extend(comp_labels)

    # Now reform the clouds such that there is one cloud per group
    new_clusters: list[LabeledCloud] = []
    for g in groups_index.keys():
        if g == NOISE_LABEL:
            continue

        new_cluster = LabeledCloud(
            g, PointCloud(event_number, np.zeros((0, 8))), np.empty(0)
        )
        for idx in groups_index[g]:
            new_cluster.point_cloud.data = np.concatenate(
                (new_cluster.point_cloud.data, clusters[idx].point_cloud.data), axis=0
            )
            # Merge the indicies
            new_cluster.parent_indicies = np.concatenate(
                (new_cluster.parent_indicies, clusters[idx].parent_indicies)
            )
        new_clusters.append(new_cluster)

    # Now replace the labels in the label array with the joined
    # values
    new_labels = labels
    for g in groups_label.keys():
        if g == NOISE_LABEL:
            continue
        for label in groups_label[g]:
            new_labels[labels == label] = g

    return (new_clusters, new_labels)


def join_clusters_continuity(
    clusters: list[LabeledCloud], params: ClusterParameters, labels: np.ndarray
) -> tuple[list[LabeledCloud], np.ndarray]:
    """Join clusters until either only one cluster is left or no clusters meet the criteria to be joined together.

    Parameters
    ----------
    clusters: list[LabeledCloud]
        the set of clusters to examine
    params: ClusterParameters
        contains parameters controlling the joining algorithm
    labels: numpy.ndarray
        The cluster label for each point in the original point cloud

    Returns
    -------
    tuple[list[LabeledCloud], numpy.ndarray]
        A two element tuple, the first the list of joined clusters,
        the second being an updated list of labels for each point in
        the point cloud
    """
    jclusters = [cluster for cluster in clusters if cluster.label != NOISE_LABEL]
    before = len(jclusters)
    after = 0
    while before != after and len(jclusters) > 1:
        before = len(jclusters)
        # Order clusters by start time
        jclusters = sorted(
            jclusters,
            key=lambda x: x.point_cloud.data[0, 2],
            # * np.mean(x.point_cloud.data[:, 3])
            # * np.mean(x.point_cloud.data[:, 7]),
        )
        jclusters, labels = join_clusters_continuity_step(jclusters, params, labels)
        after = len(jclusters)
    return (jclusters, labels)


def join_clusters_continuity_step(
    clusters: list[LabeledCloud], params: ClusterParameters, labels: np.ndarray
) -> tuple[list[LabeledCloud], np.ndarray]:
    """A single step of joining clusters

    Combine clusters based on the amount of overlap between circles fit to their x-y (pad plane) projection.
    This is necessary because often times tracks are fractured or contain regions of varying density
    which causes clustering algorithms to separate them.

    Parameters
    ----------
    clusters: list[LabeledCloud]
        the set of clusters to examine
    params: ClusterParameters
        contains the parameters controlling the joining algorithm (max_center_distance)
    labels: numpy.ndarray
        The cluster label for each point in the original point cloud

    Returns
    -------
    tuple[list[LabeledCloud], numpy.ndarray]
        A two element tuple, the first the list of joined clusters,
        the second being an updated list of labels for each point in
        the point cloud
    """
    # Can't join 1 or 0 clusters
    if len(clusters) < 2:
        return (clusters, labels)

    event_number = clusters[0].point_cloud.event_number

    # Make a dictionary of center groups, label groups
    # First everyone is in their own group
    used_clusters = {idx: False for idx in range(len(clusters))}
    groups_index: dict[int, list[int]] = {}  # Regroup the actual cluster data
    groups_label: dict[int, list[int]] = {}  # Regroup the label array
    for idx, cluster in enumerate(clusters):
        groups_index[cluster.label] = [idx]  # Use indicies for data
        groups_label[cluster.label] = [cluster.label]  # Use labels for ... labels

    # Now regroup, searching for clusters whose circles mostly overlap
    for idx, cluster in enumerate(clusters):
        # Reject noise
        if cluster.label == NOISE_LABEL or used_clusters[idx]:
            continue
        for cidx, comp_cluster in enumerate(clusters):
            # if clusters have different directions or already used, try the next
            if comp_cluster.label == NOISE_LABEL or cidx == idx or used_clusters[cidx] or comp_cluster.direction != cluster.direction :
                continue

            # downstream
            # determine center of cluster from circle fit and take positions in that circle ref
            centered = cluster.point_cloud.data.copy()
            cx, cy, radius, _ = least_squares_circle(cluster.point_cloud.data[:,0], cluster.point_cloud.data[:,1])
            centered[:, 0] -= cx
            centered[:, 1] -= cy
            avg_pos = np.median(centered[-3:, :3], axis=0)  # downstream
            min_z = np.min(cluster.point_cloud.data[:, 2])
            max_z = np.max(cluster.point_cloud.data[:, 2])
            avg_rho = np.linalg.norm(avg_pos[:2])

            # upstream
            # determine center of cluster from circle fit and take positions in that circle ref
            centered = comp_cluster.point_cloud.data.copy()
            cx, cy, comp_radius, _ = least_squares_circle(comp_cluster.point_cloud.data[:,0], comp_cluster.point_cloud.data[:,1])
            centered[:, 0] -= cx
            centered[:, 1] -= cy
            avg_pos_comp = np.median(centered[:3, :3], axis=0)  # upstream
            min_z_comp = np.min(comp_cluster.point_cloud.data[:, 2])
            max_z_comp = np.max(comp_cluster.point_cloud.data[:, 2])
            avg_rho_comp = np.linalg.norm(avg_pos_comp[:2])

           # compare z and rho differences to thresholds
            z_thresh = (max(max_z, max_z_comp) - min(min_z, min_z_comp)) * params.continuity_join.join_z_fraction  # type: ignore
            rho_thresh = (radius + comp_radius) / 2.0 * params.continuity_join.join_radius_fraction
            z_diff = np.fabs(avg_pos[2] - avg_pos_comp[2])
            rho_diff = np.fabs(avg_rho - avg_rho_comp)
            if (rho_diff < rho_thresh and z_diff < z_thresh ):
                comp_indicies = groups_index.pop(comp_cluster.label)
                comp_labels = groups_label.pop(comp_cluster.label)
                for subs in comp_indicies:
                    clusters[subs].label = cluster.label
                groups_index[cluster.label].extend(comp_indicies)
                groups_label[cluster.label].extend(comp_labels)
                used_clusters[idx] = True
                used_clusters[cidx] = True
                break
        # GWM: TODO make this less gross, optimize lower part
        if used_clusters[idx]:
            break

    # Now reform the clouds such that there is one cloud per group
    new_clusters: list[LabeledCloud] = []
    for g in groups_index.keys():
        if g == NOISE_LABEL:
            continue

        new_cluster = LabeledCloud(
            g, Direction.NONE, PointCloud(event_number, np.zeros((0, 8))), np.empty(0)
        )
        for idx in groups_index[g]:
            new_cluster.direction = clusters[idx].direction
            new_cluster.point_cloud.data = np.concatenate(
                (new_cluster.point_cloud.data, clusters[idx].point_cloud.data), axis=0
            )
            indicies = np.argsort(new_cluster.point_cloud.data[:, 2])
            new_cluster.point_cloud.data = new_cluster.point_cloud.data[indicies]
            # Merge the indicies
            new_cluster.parent_indicies = np.concatenate(
                (new_cluster.parent_indicies, clusters[idx].parent_indicies)
            )
            new_cluster.parent_indicies = new_cluster.parent_indicies[indicies]
        new_clusters.append(new_cluster)

    # Now replace the labels in the label array with the joined
    # values
    new_labels = labels
    for g in groups_label.keys():
        if g == NOISE_LABEL:
            continue
        for label in groups_label[g]:
            new_labels[labels == label] = g

    return (new_clusters, new_labels)


def cleanup_clusters(
    clusters: list[LabeledCloud], params: ClusterParameters, labels: np.ndarray
) -> tuple[list[Cluster], np.ndarray]:
    """Converts the LabeledClouds to Clusters

    In this conversion, the LocalOutlierFactor algorithm is applied to the data
    to remove any spurious points.

    Parameters
    ----------
    clusters: list[LabeledCloud]
        clusters to clean
    params: ClusterParameters
        Configuration parameters controlling the clustering algorithms
    labels: numpy.ndarray
        The cluster label for each point in the original point cloud

    Returns
    -------
    tuple[list[Cluster], numpy.ndarray]
        A two element tuple, the first the list of cleaned clusters,
        the second being an updated list of labels for each point in
        the point cloud
    """
    cleaned = []
    for cluster in clusters:
        # Cluster must have more than two points to have outlier test applied
        if cluster.label == NOISE_LABEL or len(cluster.point_cloud) < 2:
            continue
        cleaned_cluster, outliers = convert_labeled_to_cluster(cluster, params)
        cleaned.append(cleaned_cluster)
        if len(outliers) == 0:
            continue
        orig_indicies = (cluster.parent_indicies[outliers]).astype(dtype=np.int32)
        labels[orig_indicies] = NOISE_LABEL
    return (cleaned, labels)


def clean_clusters(clusters: list[LabeledCloud], params: ClusterParameters, labels: np.ndarray
                   ) -> tuple[list[LabeledCloud], np.ndarray]:
    """Attempt to clean the clusters using the LocalOutlierFactor algorithm
    
    Parameters
    ----------
    clusters: list[LabeledCloud]
        clusters to clean
    params: ClusterParameters
        Configuration parameters controlling the clustering algorithms
    labels: numpy.ndarray
        The cluster label for each point in the original point cloud

    Returns
    -------
    tuple[list[LabeledCloud], numpy.ndarray]
        A two element tuple, the first the list of cleaned clusters,
        the second being an updated list of labels for each point in
        the point cloud
    
    """
    cleaned = []
    for cluster in clusters:
        if cluster.label == NOISE_LABEL or len(cluster.point_cloud) < 2:
            continue
        cleaned_cluster, outliers = clean_cluster(cluster, params.outlier_scale_factor)
        cleaned.append(cleaned_cluster)
        if len(outliers) == 0:
            continue
        orig_indicies = (cluster.parent_indicies[outliers]).astype(dtype=np.int32)
        labels[orig_indicies] = NOISE_LABEL
    return (cleaned, labels)

def clean_cluster(cluster: LabeledCloud, scale) -> tuple[np.ndarray, np.ndarray]:
    """Attempt to clean an individual cluster using the LocalOutlierFactor algorithm
    
    Parameters
    ----------
    cluster: LabeledCloud
        cluster to clean
    scale: Float
        Scale factor applied to the algorithm

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        A two element tuple, the first the cleaned cluster,
        the second being an array of the outliers
    
    """
    neighbors = int(scale * len(cluster.point_cloud.data))  # 0.05 default
    if neighbors < 2:
        neighbors = 2
    test_data = cluster.point_cloud.data[:, :3].copy()
    # clf = IsolationForest(contamination=0.1, random_state=42)
    # clf.fit(test_data)
    # result = clf.predict(test_data)
    neigh = LocalOutlierFactor(n_neighbors=neighbors)
    result = neigh.fit_predict(test_data)
    mask = result > 0
    cleaned_cluster = LabeledCloud(cluster.label, cluster.direction, cluster.point_cloud, cluster.parent_indicies)
    cleaned_cluster.point_cloud.data = cleaned_cluster.point_cloud.data[mask]  # label=-1 is an outlier
    cleaned_cluster.parent_indicies = cleaned_cluster.parent_indicies[mask]
    outliers = np.flatnonzero(~mask)
    return (cleaned_cluster, outliers)

def form_clusters(
    pc: PointCloud, params: ClusterParameters
) -> tuple[list[LabeledCloud], np.ndarray]:
    """Apply clustering algorithm based on the values of the parameters

    Parameters
    ----------
    pc: PointCloud
        The point cloud to be clustered
    params: ClusterParameters
        Configuration parameters controlling the clustering algorithms

    Returns
    --------
    tuple[list[LabeledCloud], numpy.ndarray]
        Two element tuple, the first being a ist of clusters found by the algorithm with labels
        the second being an array of length of the point cloud with each element containing
        that point's label.
    """

    if len(pc) < params.min_cloud_size:
        return ([], np.empty(0))

    """Detect which clustering algorithm to use and dispatch"""
    if params.hdbscan_parameters is not None:
        return hdbscan_clusters(pc, params)
    elif params.tripclust_parameters is not None:
        return tripclust_clusters(pc, params)
    else:
        raise ClusterError(
            "No clustering parameters were given! Clustering cannot proceed."
        )

def hdbscan_clusters(
    pc: PointCloud, params: ClusterParameters
) -> tuple[list[LabeledCloud], np.ndarray]:
    """Apply the HDBSCAN clustering algorithm to a PointCloud

    Analyze a point cloud, and group the points into clusters which in principle should correspond to particle trajectories.
    This analysis revolves around the HDBSCAN clustering algorithm implemented in scikit-learn
    See [their description](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html) for details. We trim
    illegal (out-of-bounds) points from the point cloud, and cluster on spatial dimensions (x,y,z). Z is rescaled to match the X-Y
    scale to avoid over-emphasizing separation in Z in the clustering algorithm. The minimum size of a cluster is scaled off of the
    size of the original point cloud.


    Parameters
    ----------
    pc: PointCloud
        The point cloud to be clustered
    params: ClusterParameters
        Configuration parameters controlling the clustering algorithms

    Returns
    --------
    tuple[list[LabeledCloud], numpy.ndarray]
        Two element tuple, the first being a ist of clusters found by the algorithm with labels
        the second being an array of length of the point cloud with each element containing
        that point's label.
    """

    n_points = len(pc)
    min_size = int(params.hdbscan_parameters.min_size_scale_factor * n_points)
    if min_size < params.hdbscan_parameters.min_size_lower_cutoff:
        min_size = params.hdbscan_parameters.min_size_lower_cutoff

    # Use spatial dimensions
    cluster_data = np.empty(shape=(len(pc), 3))
    cluster_data[:, :] = pc.data[:, :3]

    # Rescale z to have same dims as x,y. Otherwise clusters too likely to break on z
    cluster_data[:, 2] *= 584.0 / 1000.0

    clusterizer = skcluster.HDBSCAN(  # type: ignore
        min_cluster_size=min_size,
        min_samples=params.hdbscan_parameters.min_points,
        cluster_selection_epsilon=params.hdbscan_parameters.cluster_selection_epsilon,
    )

    fitted_clusters = clusterizer.fit(cluster_data)
    labels = np.unique(fitted_clusters.labels_)

    # Select out data into clusters
    clusters: list[LabeledCloud] = []
    for label in labels:
        mask = fitted_clusters.labels_ == label
        clusters.append(
            LabeledCloud(
                label,
                Direction.NONE,
                PointCloud(pc.event_number, pc.data[mask]),
                np.flatnonzero(mask),
            )
        )
    return (clusters, fitted_clusters.labels_)


def tripclust_clusters(
    pc: PointCloud, params: ClusterParameters
) -> tuple[list[LabeledCloud], np.ndarray]:
    """Apply the tripclust clustering algorithm to a PointCloud

    Parameters
    ----------
    pc: PointCloud
        The point cloud to be clustered
    params: ClusterParameters
        Configuration parameters controlling the clustering algorithms

    Analyze a point cloud, and group the points into clusters which in principle should correspond to particle trajectories.
    This function uses the tripclust (Dalitz) clustering algorithm, which groups points in triplets and does
    a hierarchical clustering on them to find trajectories that are collinear.
    For details see the publication from Dalitz et al. (Dalitz et al., Comp. Phys. Comm. 235 (2019), 159,
    DOI: https://doi.org/10.1016/j.cpc.2018.09.010)

    Returns
    --------
    tuple[list[LabeledCloud], numpy.ndarray]
        Two element tuple, the first being a ist of clusters found by the algorithm with labels
        the second being an array of length of the point cloud with each element conatining
        that point's label.
    """

    if len(pc) < params.min_cloud_size:
        return ([], np.empty(0))

    # Create class to pass parameters and calls to C++ code
    tc = tripclust.tripclust()
    tc.set_r(params.tripclust_parameters.r)
    tc.set_rdnn(params.tripclust_parameters.rdnn)
    tc.set_k(params.tripclust_parameters.k)
    tc.set_n(params.tripclust_parameters.n)
    tc.set_a(params.tripclust_parameters.a)
    tc.set_s(params.tripclust_parameters.s)
    tc.set_sdnn(params.tripclust_parameters.sdnn)
    tc.set_t(params.tripclust_parameters.t)
    tc.set_tauto(params.tripclust_parameters.tauto)
    tc.set_dmax(params.tripclust_parameters.dmax)
    tc.set_dmax_dnn(params.tripclust_parameters.dmax_dnn)
    tc.set_ordered(params.tripclust_parameters.ordered)
    # tc.set_link(params.tripclust_parameters.link)
    tc.set_m(params.tripclust_parameters.m)
    tc.set_postprocess(params.tripclust_parameters.postprocess)
    tc.set_min_depth(params.tripclust_parameters.min_depth)

    # Perform tripclust (Dalitz) clustering
    tc.fill_pointcloud(pc.data)
    tc.perform_clustering()
    labels = tc.get_labels()

    # Below is based on a copy from form_clusters to get same output
    # Select out data into clusters
    clusters: list[LabeledCloud] = []
    ulabels = np.unique(labels)
    for label in ulabels:
        mask = labels == label
        clusters.append(
            LabeledCloud(
                label,
                Direction.NONE,
                PointCloud(pc.event_number, pc.data[mask]),
                np.flatnonzero(mask),
            )
        )
    return (clusters, labels)
