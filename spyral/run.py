from .core.config import Config
from .core.workspace import Workspace
from .phase_1 import phase_1
from .phase_2 import phase_2
from .phase_3 import phase_3
from .phase_4_interp import phase_4_interp
    
from multiprocessing import SimpleQueue

def run_spyral(config: Config, run_list: list[int], queue: SimpleQueue):
    '''
    This is the function to run a single processor of Spyral.
    Typically called by run_spyral_parallel and spawned to a child process.

    ## Parameters
    config: Config, the project configuration
    run_list: list[int], the set of runs for this process
    queue: SimpleQueue, a communication channel back to the parent process for monitoring progress
    '''
    ws = Workspace(config.workspace)
    pad_map = ws.get_pad_map()
    nuclear_map = ws.get_nuclear_map()

    for idx in run_list:

        if config.run.do_phase1:
            phase_1(idx, ws, pad_map, config.trace, config.frib, config.cross, config.detector, queue)

        if config.run.do_phase2:
            phase_2(idx, ws, config.cluster, queue)

        if config.run.do_phase3:
            phase_3(idx, ws, config.estimate, config.detector, queue)

        if config.run.do_phase4:
            phase_4_interp(idx, ws, config.solver, config.detector, nuclear_map)
