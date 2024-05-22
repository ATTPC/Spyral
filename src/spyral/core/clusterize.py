from .point_cloud import PointCloud
from .cluster import LabeledCloud, Cluster, convert_labeled_to_cluster
from .config import ClusterParameters
from ..geometry.circle import least_squares_circle

import sklearn.cluster as skcluster
import numpy as np

NOISE_LABEL: int = -1


def join_clusters(
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
        jclusters, labels = join_clusters_step(jclusters, params, labels)
        after = len(jclusters)
    return (jclusters, labels)


def join_clusters_step(
    clusters: list[LabeledCloud], params: ClusterParameters, labels: np.ndarray
) -> tuple[list[LabeledCloud], np.ndarray]:
    """A single step of joining clusters

    Combine clusters based on the center around which they orbit. This is necessary because often times tracks are
    fractured or contain regions of varying density which causes clustering algorithms to separate them.

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
            cluster.point_cloud.cloud[:, 0], cluster.point_cloud.cloud[:, 1]
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
        if cluster.label == NOISE_LABEL or np.isnan(center[0]) or center[2] < 10.0:
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
            if (area_overlap > params.circle_overlap_ratio * smaller_area) and (
                cidx not in groups_index[cluster.label]
            ):
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

        new_cluster = LabeledCloud(g, PointCloud())
        new_cluster.point_cloud.event_number = event_number
        new_cluster.point_cloud.cloud = np.zeros((0, 8))
        for idx in groups_index[g]:
            new_cluster.point_cloud.cloud = np.concatenate(
                (new_cluster.point_cloud.cloud, clusters[idx].point_cloud.cloud), axis=0
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


def cleanup_clusters(
    clusters: list[LabeledCloud], params: ClusterParameters, labels: np.ndarray
) -> tuple[list[Cluster], np.ndarray]:
    """Converts the LabeledClouds to Clusters

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
        if cluster.label == NOISE_LABEL or len(cluster.point_cloud.cloud) < 2:
            continue
        cleaned_cluster, outliers = convert_labeled_to_cluster(cluster, params)
        cleaned.append(cleaned_cluster)
        if len(outliers) == 0:
            continue
        orig_indicies = (cluster.parent_indicies[outliers]).astype(dtype=np.int32)
        labels[orig_indicies] = NOISE_LABEL
    return (cleaned, labels)


def form_clusters(
    pc: PointCloud, params: ClusterParameters
) -> tuple[list[LabeledCloud], np.ndarray]:
    """Apply the HDBSCAN clustering algorithm to a PointCloud

    Analyze a point cloud, and group the points into clusters which in principle should correspond to particle trajectories. This analysis contains several steps,
    and revolves around the HDBSCAN clustering algorithm implemented in scikit-learn (see [their description](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html) for details)
    First the point cloud is smoothed by averaging over nearest neighbors (defined by the smoothing_neighbor_distance parameter) to remove small deviations.
    The data is then scaled where each coordinate (x,y,z,int) is centered to its mean and then scaled to its std deviation using the scikit-learn StandardScaler. This data is then
    clustered by HDBSCAN and the clusters are returned.

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
        the second being an array of length of the point cloud with each element conatining
        that point's label.
    """

    # Smooth out the point cloud by averaging over neighboring points within a distance, droping any duplicate points
    pc.remove_illegal_points()
    if len(pc.cloud) < params.min_cloud_size:
        return ([], np.empty(0))

    n_points = len(pc.cloud)
    min_size = int(params.min_size_scale_factor * n_points)
    if min_size < params.min_size_lower_cutoff:
        min_size = params.min_size_lower_cutoff

    # Use spatial dimensions and integrated charge
    cluster_data = np.empty(shape=(len(pc.cloud), 3))
    cluster_data[:, :] = pc.cloud[:, :3]

    # Unfiy feature ranges to their means and have standard variance (1)
    cluster_data[:, 2] *= 584.0 / 1000.0

    clusterizer = skcluster.HDBSCAN(  # type: ignore
        min_cluster_size=min_size,
        min_samples=params.min_points,
        allow_single_cluster=True,
        cluster_selection_epsilon=params.cluster_selection_epsilon,
    )

    fitted_clusters = clusterizer.fit(cluster_data)
    labels = np.unique(fitted_clusters.labels_)

    # Select out data into clusters
    clusters: list[LabeledCloud] = []
    for idx, label in enumerate(labels):
        clusters.append(LabeledCloud(label, PointCloud(), np.empty(0)))
        mask = fitted_clusters.labels_ == label
        clusters[idx].point_cloud.cloud = pc.cloud[mask]
        clusters[idx].point_cloud.event_number = pc.event_number
        clusters[idx].clustered_data = cluster_data[mask]
        clusters[idx].parent_indicies = np.flatnonzero(mask)
    return (clusters, fitted_clusters.labels_)
