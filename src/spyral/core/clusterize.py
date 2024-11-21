from .point_cloud import PointCloud
from .cluster import LabeledCloud, Cluster, convert_labeled_to_cluster
from .config import ClusterParameters

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
    jclusters = [cluster for cluster in clusters if cluster.label != NOISE_LABEL]
    before = len(jclusters)
    after = 0
    while before != after and len(jclusters) > 1:
        before = len(jclusters)
        # Order clusters by start time
        jclusters = sorted(
            jclusters, key=lambda x: x.point_cloud.data[0, 2] / len(x.point_cloud)
        )
        jclusters, labels = join_clusters_step(jclusters, params, labels)
        after = len(jclusters)
    return (jclusters, labels)


def join_clusters_step(
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
            if comp_cluster.label == NOISE_LABEL or cidx == idx or used_clusters[cidx]:
                continue

            avg_pos = np.median(cluster.point_cloud.data[-3:, :3], axis=0)  # downstream
            avg_rho = np.linalg.norm(avg_pos[:2])
            avg_phi = np.arctan2(avg_pos[1], avg_pos[0])
            if avg_phi < 0.0:
                avg_phi += 2.0 * np.pi
            avg_pos_comp = np.median(
                comp_cluster.point_cloud.data[:3, :3], axis=0
            )  # upstream
            avg_rho_comp = np.linalg.norm(avg_pos_comp[:2])
            avg_phi_comp = np.arctan2(avg_pos_comp[1], avg_pos_comp[0])
            if avg_phi_comp < 0.0:
                avg_phi_comp += 2.0 * np.pi
            is_near_beam_region = (
                np.fabs(avg_rho - 25.0) < 7.0 and np.fabs(avg_rho_comp - 25.0) < 7.0
            )
            z_diff = np.fabs(avg_pos[2] - avg_pos_comp[2])
            rho_diff = np.fabs(avg_rho - avg_rho_comp)
            phi_diff = np.fabs(avg_phi - avg_phi_comp)
            if phi_diff > np.pi:
                phi_diff = 2.0 * np.pi - phi_diff

            if (
                (is_near_beam_region and z_diff < 300.0)
                or (
                    rho_diff < params.join_radius_threshold
                    and z_diff < params.join_z_threshold
                    and phi_diff < np.pi * 0.25
                )
            ) and (avg_pos[2] < avg_pos_comp[2] or z_diff < 10.0):
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
            g, PointCloud(event_number, np.zeros((0, 8))), np.empty(0)
        )
        for idx in groups_index[g]:
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


def form_clusters(
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
        the second being an array of length of the point cloud with each element conatining
        that point's label.
    """

    if len(pc) < params.min_cloud_size:
        return ([], np.empty(0))

    n_points = len(pc)
    min_size = int(params.min_size_scale_factor * n_points)
    if min_size < params.min_size_lower_cutoff:
        min_size = params.min_size_lower_cutoff

    # Use spatial dimensions
    cluster_data = np.empty(shape=(len(pc), 3))
    cluster_data[:, :] = pc.data[:, :3]

    # Rescale z to have same dims as x,y. Otherwise clusters too likely to break on z
    cluster_data[:, 2] *= 584.0 / 1000.0

    clusterizer = skcluster.HDBSCAN(  # type: ignore
        min_cluster_size=min_size,
        min_samples=params.min_points,
        cluster_selection_epsilon=params.cluster_selection_epsilon,
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
                PointCloud(pc.event_number, pc.data[mask]),
                np.flatnonzero(mask),
            )
        )
    return (clusters, fitted_clusters.labels_)
