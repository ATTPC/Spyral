from .core.config import ClusterParameters
from .core.point_cloud import PointCloud
from .core.clusterize import form_clusters, join_clusters_depth, cleanup_clusters
from .core.workspace import Workspace
import h5py as h5
from time import time

def phase_2(run: int, ws: Workspace, cluster_params: ClusterParameters):

    start = time()
    point_path = ws.get_point_cloud_file_path(run)
    if not point_path.exists():
        return
    
    cluster_path = ws.get_cluster_file_path(run)

    point_file = h5.File(point_path, 'r')
    cluster_file = h5.File(cluster_path, 'w')

    cloud_group: h5.Group = point_file.get('cloud')
    min_event: int = cloud_group.attrs['min_event']
    max_event: int = cloud_group.attrs['max_event']
    cluster_group: h5.Group = cluster_file.create_group('cluster')
    cluster_group.attrs['min_event'] = min_event
    cluster_group.attrs['max_event'] = max_event

    print(f'Clustering point clouds in file {point_path} over events {min_event} to {max_event}')

    flush_percent = 0.01
    flush_val = int(flush_percent * (max_event - min_event))
    flush_count = 0
    count = 0

    for idx in range(min_event, max_event+1):

        if count > flush_val:
            count = 0
            flush_count += 1
            print(f'\rPercent of data processed: {int(flush_count * flush_percent * 100)}%', end='')
        count += 1

        cloud_data: h5.Dataset | None = None
        try:
            cloud_data = cloud_group.get(f'cloud_{idx}')
        except Exception:
            continue

        if cloud_data is None:
            continue

        cloud = PointCloud()
        cloud.load_cloud_from_hdf5_data(cloud_data[:].copy(), idx)

        clusters = form_clusters(cloud, cluster_params)
        joined = join_clusters_depth(clusters, cluster_params)
        cleaned = cleanup_clusters(joined, cluster_params)

        cluster_event_group = cluster_group.create_group(f'event_{idx}')
        cluster_event_group.attrs['nclusters'] = len(cleaned)
        cluster_event_group.attrs['ic_amplitude'] = cloud_data.attrs['ic_amplitude']
        cluster_event_group.attrs['ic_centroid'] = cloud_data.attrs['ic_centroid']
        cluster_event_group.attrs['ic_integral'] = cloud_data.attrs['ic_integral']
        for cidx, cluster in enumerate(cleaned):
            local_group = cluster_event_group.create_group(f'cluster_{cidx}')
            local_group.attrs['label'] = cluster.label
            local_group.attrs['z_bin_width'] = cluster.z_bin_width
            local_group.attrs['z_bin_low_edge'] = cluster.z_bin_low_edge
            local_group.attrs['z_bin_hi_edge'] = cluster.z_bin_hi_edge
            local_group.attrs['n_z_bins'] = cluster.n_z_bins
            local_group.create_dataset('cloud', data=cluster.data)


    stop = time()
    print(f'\nProcessing complete. Duration: {stop - start}s')



