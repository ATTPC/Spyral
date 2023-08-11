from .core.config import ClusterParameters, DetectorParameters
from .core.point_cloud import PointCloud
from .core.clusterize import ClusteredCloud, clusterize
import h5py as h5
from pathlib import Path
from time import time

def get_cloud_event_range(point_file: h5.File) -> tuple[int, int]:
    meta_group: h5.Group = point_file.get('meta')
    min_event: int = meta_group['min_event'][()]
    max_event: int = meta_group['max_event'][()]
    return (min_event, max_event)

def write_cluster_metadata(cluster_file: h5.File, event_range: tuple[int, int]):
    meta_group: h5.Group = cluster_file.create_group('meta')
    meta_group.create_dataset('min_event', data=event_range[0])
    meta_group.create_dataset('max_event', data=event_range[1])

def phase_2(point_path: Path, cluster_path: Path, cluster_params: ClusterParameters, detector_params: DetectorParameters):

    start = time()

    point_file = h5.File(point_path, 'r')
    cluster_file = h5.File(cluster_path, 'w')

    min_event, max_event = get_cloud_event_range(point_file)
    write_cluster_metadata(cluster_file, (min_event, max_event))

    cloud_group: h5.Group = point_file.get('cloud')
    cluster_group: h5.Group = cluster_file.create_dataset('cluster')

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

        cloud_data: h5.Dataset | None = None
        try:
            cloud_data = cloud_group.get(f'cloud_{idx}')
        except:
            continue

        cloud = PointCloud()
        cloud.load_cloud_from_hdf5_data(cloud_data, idx)

        clusters = clusterize(cloud, cluster_params, detector_params)

        #Write the clusters
        cluster_event_group = cluster_group.create_group(f'event_{idx}')
        cluster_event_group.create_dataset('nclusters', data=len(clusters))
        for cidx, cluster in enumerate(clusters.values()):
            local_group = cluster_event_group.create_group(f'cluster_{cidx}')
            local_group.create_dataset('label', data=cluster.label)
            local_group.create_dataset('cloud', data=cluster.point_cloud.cloud)



