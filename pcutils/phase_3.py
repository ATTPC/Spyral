from .core.clusterize import ClusteredCloud
from .core.config import DetectorParameters
from pathlib import Path
from time import time
import h5py as h5

def phase_4(cluster_path: Path, parquet_path: Path, params: DetectorParameters):
    start = time()

    cluster_file = h5.File(cluster_path, 'r')
    cluster_group: h5.Group = cluster_file.get('cluster')

    min_event = cluster_group.attrs['min_event']
    max_event = cluster_group.attrs['max_event']

    print(f'Running initial analysis on clusters in {cluster_path} over events {min_event} to {max_event}')

    flush_percent = 0.01
    flush_val = int(flush_percent * (max_event - min_event))
    flush_count = 0
    count = 0

    data: dict[str, list] = {'event': [], 'cluster': [], 'vertex_x': [], 'vertex_y': [], 'vertex_z': [], 'polar': [], 'azimuthal': [], 'brho': [], 'dEdx': [], 'pathlength': []}

    for idx in range(min_event, max_event+1):
        if count > flush_val:
            count = 0
            flush_count += 1
            print(f'\rPercent of data processed: {int(flush_count * flush_percent * 100)}%', end='')
        count += 1

        event: h5.Group | None = None
        try:
            event = cluster_group.get(f'event_{idx}')
        except:
            continue

        nclusters = event.attrs['nclusters']
        for cidx in range(0, nclusters):
            cluster_group: h5.Group | None = None
            try:
                cluster_group = event.get[f'cluster_{cidx}']
            except:
                continue

            cluster = ClusteredCloud()
            cluster.label = cluster_group.attrs['label']
            cluster_data = cluster['cloud']
            cluster.point_cloud.load_cloud_from_hdf5_data(cluster_data[:].copy(), idx)

            #Cluster is loaded do some analysis

    stop = time()
    print(f'\nEllapsed time: {stop-start}s')