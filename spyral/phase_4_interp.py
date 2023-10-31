from .core.config import SolverParameters, DetectorParameters
from .core.track_generator import GeneratorParams, generate_tracks, TrackInterpolator, create_interpolator
from .core.workspace import Workspace
from .core.nuclear_data import NuclearDataMap
from .core.particle_id import load_particle_id, ParticleID
from .core.target import Target
from .core.cluster import Cluster
from .solvers.solver_interp import solve_physics_interp, Guess
from .parallel.status_message import StatusMessage, Phase

import h5py as h5
import polars as pl
from time import time
from pathlib import Path
from multiprocessing import SimpleQueue

def phase_4_interp(run: int, ws: Workspace, solver_params: SolverParameters, det_params: DetectorParameters, nuclear_map: NuclearDataMap, queue: SimpleQueue):
    # start = time()

    pid: ParticleID | None = load_particle_id(ws.get_gate_file_path(solver_params.particle_id_filename), nuclear_map)
    if pid is None:
        print('Particle ID error at phase 4!')
        return
    
    target: Target = Target(Path(solver_params.gas_data_path), nuclear_map)

    cluster_path = ws.get_cluster_file_path(run)
    estimate_path = ws.get_estimate_file_path_parquet(run)
    if not cluster_path.exists() or not estimate_path.exists():
        return
    
    result_path = ws.get_physics_file_path_parquet(run, pid.nucleus)
    cluster_file = h5.File(cluster_path, 'r')
    estimate_df = pl.scan_parquet(estimate_path)

    cluster_group: h5.Group = cluster_file.get('cluster')

    # print(f'Running physics solver on clusters in {cluster_path} using initial guesses from {estimate_path}')
    # print(f'Selecting data which corresponds to particle group from {solver_params.particle_id_filename}')

    #Select the particle group data, convert to dictionary for row-wise operations
    estimates_gated = estimate_df.filter(
            pl.struct(['dEdx', 'brho']).map(pid.cut.is_cols_inside) & 
            (pl.col('ic_amplitude') > solver_params.ic_min_val) & 
            (pl.col('ic_amplitude') < solver_params.ic_max_val)
        ).collect().to_dict()


    flush_percent = 0.01
    flush_val = int(flush_percent * (len(estimates_gated['event'])))
    flush_count = 0
    count = 0

    results: dict[str, list] = { 
        'event': [], 
        'cluster_index': [], 
        'cluster_label': [], 
        'vertex_x': [], 
        'sigma_vx': [], 
        'vertex_y': [], 
        'sigma_vy': [], 
        'vertex_z': [], 
        'sigma_vz': [],
        'brho': [], 
        'sigma_brho': [], 
        'polar': [], 
        'sigma_polar': [], 
        'azimuthal': [], 
        'sigma_azimuthal': [], 
        'redchisq': []
    }

    interp_path = ws.get_track_file_path(solver_params.interp_file_name)
    interpolator = create_interpolator(interp_path)

    # print(f'Looking for track interpolation data file {solver_params.interp_file_name}...')
    # interp_path = ws.get_track_file_path(solver_params.interp_file_name)
    # gen_params = GeneratorParams(
    #                 target, pid.nucleus, 
    #                 det_params.magnetic_field, 
    #                 det_params.electric_field, 
    #                 solver_params.interp_ke_min, 
    #                 solver_params.interp_ke_max, 
    #                 solver_params.interp_ke_bins, 
    #                 solver_params.interp_polar_min, 
    #                 solver_params.interp_polar_max, 
    #                 solver_params.interp_polar_bins
    #             )
    # print(f'Using interpolation with energy range {gen_params.ke_min} to {gen_params.ke_max} MeV with {gen_params.ke_bins} bins and')
    # print(f'angle range {gen_params.polar_min} to {gen_params.polar_max} degrees with {gen_params.polar_bins} bins')
    # interpolator: TrackInterpolator
    # if not interp_path.exists():
    #     print(f'Interpolation data does not exist, generating... This may take some time...')
    #     generate_tracks(gen_params, interp_path)
    #     print(f'Interpolation data generated. Loading interpolation scheme...')
    #     interpolator = create_interpolator(interp_path)
    #     print(f'Interpolation loaded.')
    # else:
    #     print(f'Found interpolation data. Checking to see if it matches expected configuration...')
    #     interpolator = create_interpolator(interp_path)
    #     if interpolator.check_interpolator(gen_params.particle.isotopic_symbol, gen_params.bfield, gen_params.efield, gen_params.target.pretty_string, gen_params.ke_min, gen_params.ke_max, gen_params.ke_bins, gen_params.polar_min, gen_params.polar_max, gen_params.polar_bins):
    #         print('Interpolator configuration matches. Interpolation loaded.')
    #     else:
    #         print('Interpolator configuration does not match! If you want to regenerate this data, please delete the extant file.')
    #         print('Exiting.')
    #         return
        
    # print('Starting solver...')
    for row, event in enumerate(estimates_gated['event']):
        if count > flush_val:
            count = 0
            # flush_count += 1
            # print(f'\rPercent of data processed: {int(flush_count * flush_percent * 100)}%', end='')
            queue.put(StatusMessage(run, Phase.SOLVE, 1))
        count += 1

        event_group = cluster_group[f'event_{event}']
        cidx = estimates_gated['cluster_index'][row]
        local_cluster: h5.Dataset = event_group[f'cluster_{cidx}']
        cluster = Cluster(event, local_cluster.attrs['label'], local_cluster['cloud'][:].copy())
        cluster.z_bin_width = local_cluster.attrs['z_bin_width']
        cluster.z_bin_low_edge = local_cluster.attrs['z_bin_low_edge']
        cluster.z_bin_hi_edge = local_cluster.attrs['z_bin_hi_edge']
        cluster.n_z_bins = local_cluster.attrs['n_z_bins']

        #Do the solver
        guess = Guess(
                    estimates_gated['polar'][row],
                    estimates_gated['azimuthal'][row],
                    estimates_gated['brho'][row],
                    estimates_gated['vertex_x'][row],
                    estimates_gated['vertex_y'][row],
                    estimates_gated['vertex_z'][row]
                )
        solve_physics_interp(cidx, cluster, guess, interpolator, pid.nucleus, results)

    physics_df = pl.DataFrame(results)
    physics_df.write_parquet(result_path)

    # stop = time()
    # print(f'\nEllapsed time: {stop-start}s')

