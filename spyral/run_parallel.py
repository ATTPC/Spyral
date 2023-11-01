from .core.config import Config
from .core.workspace import Workspace
from .core.track_generator import generate_tracks, GeneratorParams
from .core.target import Target
from .core.particle_id import load_particle_id
from .correction import generate_electron_correction
from .parallel.status_message import StatusMessage, Phase
from .parallel.run_stack import create_run_stacks
from .run import run_spyral

from multiprocessing import Process, SimpleQueue
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path

def generate_shared_resources(config: Config):
    print('Checking to see if any shared resources need to be created...')
    ws = Workspace(config.workspace)
    nuc_map = ws.get_nuclear_map()
    track_path = ws.get_track_file_path(config.solver.interp_file_name)
    ecorr_path = ws.get_correction_file_path(config.detector.efield_correction_name)

    if not ecorr_path.exists() and config.run.do_phase1:
        print('Creating the electric field correction data... This may take some time...')
        generate_electron_correction(Path(config.detector.garfield_file_path), ecorr_path, config.detector)
        print('Done.')

    if not track_path.exists() and config.run.do_phase4:
        print('Creating the interpolation scheme... This may take some time...')
        target = Target(Path(config.solver.gas_data_path), nuc_map)
        pid = load_particle_id(ws.get_gate_file_path(config.solver.particle_id_filename), nuc_map)
        if pid is None:
            print('Could not create interpolation scheme, particle ID does not have the correct format!')
            print('Particle ID is required for running the solver stage (phase 4).')
            raise Exception
        gen_params = GeneratorParams(
                    target, pid.nucleus, 
                    config.detector.magnetic_field, 
                    config.detector.electric_field, 
                    config.solver.interp_ke_min, 
                    config.solver.interp_ke_max, 
                    config.solver.interp_ke_bins, 
                    config.solver.interp_polar_min, 
                    config.solver.interp_polar_max, 
                    config.solver.interp_polar_bins
                )
        generate_tracks(gen_params, track_path)
        print('Done.')
    print('Shared resources are ready.')

def run_spyral_parallel(config: Config):
    #Some data must be made and shared across processes through files.
    generate_shared_resources(config)

    n_processes = config.run.n_processes

    queues: list[SimpleQueue] = []
    processes: list[Process] = []
    pbars: list[tqdm] = []
    stats: list[Phase] = []

    print('Optimally generating run lists for each process...')
    stacks = create_run_stacks(config, n_processes)
    print('Done.')

    for s in range(0, len(stacks)):
        local_config = deepcopy(config)
        queues.append(SimpleQueue())
        processes.append(Process(target=run_spyral, args=(local_config, stacks[s], queues[-1]), daemon=False))
        pbars.append(tqdm(total=100))
        stats.append(Phase.WAIT)
        pbars[-1].set_description(f'| Process {s} | { str(stats[-1]) } |')

    for process in processes:
        process.start()

    anyone_alive: bool
    # main loop
    while True:
        anyone_alive = False
        # check processes still going
        for process in processes:
            if process.is_alive():
                anyone_alive = True
                break
        
        if not anyone_alive:
            break

        for idx, q in enumerate(queues):
            if q.empty():
                continue

            msg: StatusMessage = q.get()
            if msg.phase != stats[idx]:
                pbars[idx].reset()
                pbars[idx].set_description(f'| Process {idx} | {msg.task_str()}')
                stats[idx] = msg.phase
            pbars[idx].update(msg.progress)
    
    # People get nervous if the bars don't reach 100%
    for bar in pbars:
        remain = bar.total - bar.last_print_n
        if remain > 0:
            bar.update(remain)

    for bar in pbars:
        bar.close()

    for q in queues:
        q.close()

    for process in processes:
        process.join()