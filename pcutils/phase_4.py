from .core.config import DetectorParameters, SolverParameters
from .core.clusterize import ClusteredCloud
from .core.nuclear_data import NuclearDataMap
from .core.particle_id import ParticleID, load_particle_id
from .core.target import Target
import h5py as h5
import polars as pl
from pathlib import Path
from time import time

def phase_4(cluster_path: Path, estimate_path: Path, result_path: Path, detector_params: DetectorParameters, solver_params: SolverParameters):
    start = time()

    nuclear_data = NuclearDataMap()

    pid: ParticleID = load_particle_id(solver_params.particle_id_path, nuclear_data)
    if pid is None:
        print('Particle ID error at phase 4!')
        return
    
    target: Target = Target(solver_params.gas_data_path, nuclear_data)

    cluster_file = h5.File(cluster_path, 'r')
    estimate_df = pl.scan_parquet(estimate_path)

    cluster_group: h5.Group = cluster_file.get('cluster')

    print(f'Running physics solver on clusters in {cluster_path} using initial guesses from {estimate_path}')
    print(f'Selecting data which corresponds to particle group from {solver_params.particle_id_path}')

    #Select the particle group data, convert to dictionary for row-wise operations
    estimates_gated = estimate_df.filter(pl.struct(['dEdx', 'brho']).map(pid.cut.is_cols_inside)).collect().to_dict()


    flush_percent = 0.01
    flush_val = int(flush_percent * (len(estimates_gated['event'])))
    flush_count = 0
    count = 0

    results: dict[str, list] = { 'event': [], 'cluster_index': [], 'cluster_label': [], 'vertex_x': [], 'vertex_y': [], 'vertex_z': [], \
                                 'E': [], 'polar': [], 'azimuthal': [], 'P': []}
    
    for row, event in enumerate(estimates_gated['event']):
        if count > flush_val:
            count = 0
            flush_count += 1
            print(f'\rPercent of data processed: {int(flush_count * flush_percent * 100)}%', end='')
        count += 1

        event_group = cluster_group[f'event_{event}']
        local_cluster: h5.Dataset = event_group[f'cluster_{estimates_gated["cluster_index"][row]}']
        cluster = ClusteredCloud()
        cluster.label = local_cluster.attrs['label']
        cluster.point_cloud.load_cloud_from_hdf5_data(local_cluster[:].copy(), event)

        #Do the solver
        #solve_physics(cluster_index, cluster, detector_params, pid.nucleus, target, results)
        #or something

    stop = time()
    print(f'\nEllapsed time: {stop-start}s')
    
    
