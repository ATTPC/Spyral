from .core.cluster import Cluster
from .core.config import DetectorParameters, EstimateParameters
from .core.estimator import estimate_physics
from .core.workspace import Workspace
from .parallel.status_message import StatusMessage, Phase

from polars import DataFrame
import h5py as h5
from multiprocessing.queues import SimpleQueue


def phase_3(run: int, ws: Workspace, estimate_params: EstimateParameters, detector_params: DetectorParameters, queue: SimpleQueue):

    cluster_path = ws.get_cluster_file_path(run)
    if not cluster_path.exists():
        return
    
    estimate_path = ws.get_estimate_file_path_parquet(run)

    cluster_file = h5.File(cluster_path, 'r')
    cluster_group: h5.Group = cluster_file.get('cluster')

    min_event: int = cluster_group.attrs['min_event']
    max_event: int = cluster_group.attrs['max_event']

    flush_percent = 0.01
    flush_val = int(flush_percent * (max_event - min_event))
    count = 0

    data: dict[str, list] = {
        'event': [], 
        'cluster_index': [], 
        'cluster_label': [],
        'ic_amplitude': [],
        'ic_centroid': [],
        'ic_integral': [], 
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
            queue.put(StatusMessage(run, Phase.ESTIMATE, 1))
        count += 1

        event: h5.Group | None = None
        try:
            event = cluster_group[f'event_{idx}']
        except Exception:
            continue

        nclusters = event.attrs['nclusters']
        ic_amp: float = event.attrs['ic_amplitude']
        ic_cent: float = event.attrs['ic_centroid']
        ic_int: float = event.attrs['ic_integral']
        for cidx in range(0, nclusters):
            local_cluster: h5.Group | None = None
            try:
                local_cluster = event[f'cluster_{cidx}']
            except Exception:
                continue

            cluster = Cluster(idx, local_cluster.attrs['label'], local_cluster['cloud'][:].copy())
            
            #Cluster is loaded do some analysis
            estimate_physics(cidx, cluster, ic_amp, ic_cent, ic_int, estimate_params, detector_params, data)

    df = DataFrame(data)
    df.write_parquet(estimate_path)