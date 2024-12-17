from .pipeline import Pipeline, _run_pipeline, SPLASH
from .spy_log import init_spyral_logger_parent, spyral_warn, spyral_info
from .run_stacks import create_run_stacks
from .status_message import StatusMessage

from dragon.native.machine import System, cpu_count
import multiprocessing as mp
from numpy.random import SeedSequence
import time


def _run_shared_memory_manager(
    pipeline: Pipeline,
    ready: mp.Event,  # type: ignore
    shutdown: mp.Event,  # type: ignore
) -> None:
    """Simple manager process for shared memory on each node

    Parameters
    ----------
    pipeline: Pipeline
        The pipeline definition
    shutdown: multiprocessing.Event
        A multiprocessing event which will fire when the work is done
        and indicate that all shared memory should be cleaned up.
    """
    handles = {}
    try:
        pipeline.create_shared_data(handles)
    except Exception:
        # Hack to make sure we don't hang when creation fails
        ready.set()
        return
    ready.set()
    shutdown.wait()
    for handle in handles.values():
        handle.close()
        handle.unlink()


def start_pipeline_dragon(
    pipeline: Pipeline,
    run_min: int,
    run_max: int,
    runs_to_skip: list[int] | None = None,
) -> None:
    system = System()
    n_nodes = system.nnodes
    total_cpus = cpu_count()  # logical cpus, use PBS to restrict
    if runs_to_skip is None:
        runs_to_skip = []

    # One cpu allocated for this process?
    # One cpu per node allocated to manage shared memory
    worker_cpus = total_cpus - n_nodes - 1

    print(SPLASH)
    print(f"Total cpus available: {total_cpus}")
    print(f"Number of nodes: {n_nodes}")
    print(f"Calculated worker cpus: {worker_cpus}")
    print(f"Creating workspace: {pipeline.workspace} ...", end=" ")
    pipeline.create_workspace()
    print("Done.")
    print("Initializing logs...", end=" ")
    init_spyral_logger_parent(pipeline.workspace)
    print("Done.")
    print("Creating any phase assets...", end=" ")
    pipeline.create_assets()
    print("Done.")
    print("Validating Pipeline...", end=" ")
    result = pipeline.validate()
    if False in result.values():
        print("")
        print("Pipeline validation failed!")
        print(f"Status: {result}")
        return
    print("Pipeline successfully validated.")

    # Dragon allocates processes in a round-robin style
    # Once per node we allocate a manager
    print("Setting up shared memory managers...")
    shutdown_event = mp.Event()
    managers = []
    # Barrier has limited multinode support so fake it with some Events
    readies = []
    for _ in range(n_nodes):
        readies.append(mp.Event())
        managers.append(
            mp.Process(
                target=_run_shared_memory_manager,
                args=(pipeline, readies[-1], shutdown_event),
            )
        )

        managers[-1].start()

    print("Waiting for shared memory...")
    # Wait for all of the memory to be ready
    for ready in readies:
        ready.wait()
    print("Shared memory is ready.")

    stacks, stack_load = create_run_stacks(
        pipeline.traces, run_min, run_max, worker_cpus, runs_to_skip
    )
    stack_count = 0
    for stack in stacks:
        stack_count += len(stack)
    if len(stack) == 0:
        spyral_warn(
            __name__,
            f"No runs were found in trace path: {pipeline.traces}. Traces must exist to create workload!",
        )
    spyral_info(__name__, f"Run stacks: {stacks}")
    for idx, load in enumerate(stack_load):
        if load != 0.0:
            spyral_info(__name__, f"Stack {idx} load: {load:.2f}%")

    seq = SeedSequence()
    queues: list[mp.SimpleQueue] = []
    processes: list[mp.Process] = []
    active_phases: list[str] = []
    active_runs: list[int] = []
    seeds = seq.spawn(len(stacks))

    print("Running Spyral...")
    start = time.time()
    # Create the child processes
    for s in range(0, len(stacks)):
        queues.append(mp.SimpleQueue())
        processes.append(
            mp.Process(
                target=_run_pipeline,
                args=(
                    pipeline,
                    stacks[s],
                    queues[-1],
                    seeds[s],
                    s,
                ),  # All of this gets pickled/unpickled
                daemon=False,
            )
        )
        active_phases.append("Waiting")  # put something here
        active_runs.append(-1)

    for process in processes:
        process.start()

    anyone_alive: bool
    # main loop
    while True:
        anyone_alive = False
        # check processes still going, or if queues have data to be read out
        for idx, process in enumerate(processes):
            if process.is_alive() or (not queues[idx].empty()):
                anyone_alive = True
                break

        if not anyone_alive:
            break

        # Read events out of the queues
        for idx, q in enumerate(queues):
            if q.empty():
                continue

            msg: StatusMessage = q.get()
            if msg.phase != active_phases[idx] or msg.run != active_runs[idx]:
                active_phases[idx] = msg.phase
                active_runs[idx] = msg.run

    # We're all done! Shut it down
    shutdown_event.set()

    for q in queues:
        q.close()

    for process in processes:
        process.join()

    for manager in managers:
        manager.join()

    # Time info for housekeeping
    stop = time.time()
    duration = stop - start
    hours, sec = divmod(duration, 3600)
    minutes, sec = divmod(sec, 60)
    spyral_info(
        __name__, f"Total ellapsed time: {int(hours)} hrs {int(minutes)} min {sec:.4} s"
    )
