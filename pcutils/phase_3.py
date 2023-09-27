from .core.cluster import Cluster
from .core.config import DetectorParameters, EstimateParameters
from .core.estimator import estimate_physics
from .core.workspace import Workspace
from polars import DataFrame
from time import time
import h5py as h5

def phase_3(run: int, ws: Workspace, estimate_params: EstimateParameters, detector_params: DetectorParameters):
    start = time()

    cluster_path = ws.get_cluster_file_path(run)
    if not cluster_path.exists():
        return
    
    estimate_path = ws.get_estimate_file_path_parquet(run)

    cluster_file = h5.File(cluster_path, 'r')
    cluster_group: h5.Group = cluster_file.get('cluster')

    min_event = cluster_group.attrs['min_event']
    max_event = cluster_group.attrs['max_event']

    print(f'Running physics estimation on clusters in {cluster_path} over events {min_event} to {max_event}')

    flush_percent = 0.01
    flush_val = int(flush_percent * (max_event - min_event))
    flush_count = 0
    count = 0

    data: dict[str, list] = {
        'event': [], 
        'cluster_index': [], 
        'cluster_label': [], 
        'vertex_x': [], 
        'vertex_y': [], 
        'vertex_z': [],
        'center_x': [], 
        'center_y': [], 
        'center_z': [], 
        'polar': [], 
        'azimuthal': [],
        'brho': [], 
        'dEdx': [], 
        'dE': [], 
        'arclength': [], 
        'direction': []
    }

    for idx in range(min_event, max_event+1):
        if count > flush_val:
            count = 0
            flush_count += 1
            print(f'\rPercent of data processed: {int(flush_count * flush_percent * 100)}%', end='')
        count += 1

        event: h5.Group | None = None
        try:
            event = cluster_group[f'event_{idx}']
        except Exception:
            continue

        nclusters = event.attrs['nclusters']
        for cidx in range(0, nclusters):
            local_cluster: h5.Group | None = None
            try:
                local_cluster = event[f'cluster_{cidx}']
            except Exception:
                continue

            cluster = Cluster(idx, local_cluster.attrs['label'], local_cluster['cloud'][:].copy())
            
            #Cluster is loaded do some analysis
            estimate_physics(cidx, cluster, estimate_params, detector_params, data)

    df = DataFrame(data)
    df.write_parquet(estimate_path)


    stop = time()
    print(f'\nEllapsed time: {stop-start}s')