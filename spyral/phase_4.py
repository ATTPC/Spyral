from .core.config import DetectorParameters, SolverParameters
from .core.workspace import Workspace
from .core.cluster import Cluster
from .core.nuclear_data import NuclearDataMap
from .core.particle_id import ParticleID, load_particle_id
from .core.target import Target
from .core.estimator import Direction
from .solvers.solver import solve_physics, Guess

import h5py as h5
import polars as pl
from time import time

def phase_4(run: int, ws: Workspace, detector_params: DetectorParameters, solver_params: SolverParameters, nuclear_data: NuclearDataMap):
    start = time()

    pid: ParticleID = load_particle_id(ws.get_gate_file_path(solver_params.particle_id_filename), nuclear_data)
    if pid is None:
        print('Particle ID error at phase 4!')
        return
    
    target: Target = Target(solver_params.gas_data_path, nuclear_data)

    cluster_path = ws.get_cluster_file_path(run)
    estimate_path = ws.get_estimate_file_path_parquet(run)
    if not cluster_path.exists() or not estimate_path.exists():
        return
    
    result_path = ws.get_physics_file_path_parquet(run, pid.nucleus)
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

    results: dict[str, list] = { 'event': [], 'cluster_index': [], 'cluster_label': [], 'vertex_x': [], 'sigma_vx': [], 'vertex_y': [], 'sigma_vy': [], 'vertex_z': [], 'sigma_vz': [], \
                                 'brho': [], 'sigma_brho': [], 'polar': [], 'sigma_polar': [], 'azimuthal': [], 'sigma_azimuthal': [], 'redchisq': []}
    print('Starting solver...')
    for row, event in enumerate(estimates_gated['event']):
        if count > flush_val:
            count = 0
            flush_count += 1
            print(f'\rPercent of data processed: {int(flush_count * flush_percent * 100)}%', end='')
        count += 1

        event_group = cluster_group[f'event_{event}']
        cidx = estimates_gated['cluster_index'][row]
        local_cluster: h5.Dataset = event_group[f'cluster_{cidx}']
        cluster = Cluster(event, local_cluster.attrs['label'], local_cluster['cloud'][:].copy())

        #Do the solver
        iv = Guess(polar=estimates_gated['polar'][row], azimuthal=estimates_gated['azimuthal'][row], brho=estimates_gated['brho'][row],
                          vertex_x=estimates_gated['vertex_x'][row] * 0.001, vertex_y=estimates_gated['vertex_y'][row] * 0.001, vertex_z=estimates_gated['vertex_z'][row] * 0.001, direction=Direction(estimates_gated['direction'][row]))
        solve_physics(cidx, cluster, iv, detector_params, target, pid.nucleus, results)

    physics_df = pl.DataFrame(results)
    physics_df.write_parquet(result_path)

    stop = time()
    print(f'\nEllapsed time: {stop-start}s')
    
    
