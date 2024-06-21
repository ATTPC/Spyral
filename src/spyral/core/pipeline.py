from .run_stacks import form_run_string, create_run_stacks
from .status_message import StatusMessage
from .phase import PhaseLike, PhaseResult
from .spy_log import (
    init_spyral_logger_parent,
    init_spyral_logger_child,
    spyral_info,
    spyral_warn,
    spyral_except,
)

from tqdm import tqdm
from pathlib import Path
from multiprocessing import SimpleQueue, Process
from multiprocessing.managers import SharedMemoryManager
from numpy.random import SeedSequence, default_rng
import time

# Generated using https://www.asciiart.eu
SPLASH: str = r"""
-------------------------------
 ____                        _ 
/ ___| _ __  _   _ _ __ __ _| |
\___ \|  _ \| | | |  __/ _  | |
 ___| | |_| | |_| | | | |_| | |
|____/|  __/ \__  |_|  \__ _|_|
      |_|    |___/             
-------------------------------
"""


class Pipeline:
    """A representation of an analysis pipeline in Spyral

    The Pipeline controls the analysis workflow. It is given a list
    of PhaseLike objects and paths to workspace and trace data and runs the
    data through the pipeline. It also requires a list of booleans of the same
    length as the list of PhaseLikes. Each bool in the list is a switch which
    turns on/off that Phase. If the Phase is off (False), it is assumed that
    any artifacts expected to be produced by that Phase have been made if
    requested.

    The first Phase in the Pipeline should always expect to recieve AT-TPC
    trace data.

    Parameters
    ----------
    phases: list[PhaseLike]
        The Phases of the analysis pipeline
    active: list[bool]
        A list of boolean switches of the same length as phases. Each switch
        controls the corresponding phase in the pipeline.
    workspace_path: pathlib.Path
        The path to the workspace (the place where Spyral will write data)
    trace_path: pathlib.Path
        The path to the AT-TPC trace data

    Attributes
    ----------
    phases: list[PhaseLike]
        The Phases of the analysis pipeline
    active: list[bool]
        A list of boolean switches of the same length as phases. Each switch
        controls the corresponding phase in the pipeline.
    workspace: pathlib.Path
        The path to the workspace (the place where Spyral will write data)
    traces: pathlib.Path
        The path to the AT-TPC trace data

    Methods
    -------
    create_workspace()
        Create the workspace and subdirectories
    create_assets()
        Have each phase create any necessary assets.
    validate()
        Validate the pipeline by comparing the schema of the phases.
    run(run_list, msg_queue, seed)
        Run the pipeline for a set of runs

    """

    def __init__(
        self,
        phases: list[PhaseLike],
        active: list[bool],
        workspace_path: Path,
        trace_path: Path,
    ):
        self.phases = phases
        self.active = active
        self.workspace = workspace_path
        self.traces = trace_path

    def create_workspace(self) -> None:
        """Create the workspace and all child directories.

        This should be called before running the pipeline.

        """

        if not self.workspace.exists():
            self.workspace.mkdir()

        for phase in self.phases:
            asset_dir = phase.get_asset_storage_path(self.workspace)
            artifact_dir = phase.get_artifact_path(self.workspace)
            if not asset_dir.exists():
                asset_dir.mkdir()
            if not artifact_dir.exists():
                artifact_dir.mkdir()

    def create_assets(self) -> bool:
        """Have each phase create any necessary assets.

        Call each PhaseLike's create_assets function.
        This should be called before running the pipeline.

        Returns
        -------
        bool
            True if all phases were successful, False otherwise
        """
        for idx, phase in enumerate(self.phases):
            # Skip inactive phases
            if not self.active[idx]:
                continue
            if not phase.create_assets(self.workspace):
                return False
        return True

    def create_shared_data(self, manager: SharedMemoryManager) -> None:
        """Have each phase create any shared memory

        Call each PhaseLike's create_shared_data function.
        This should be called before running the pipeline with a
        valid, started SharedMemoryManager.

        Parameters
        ----------
        manager: multiprocessing.manager.SharedMemoryManager
            The manager of the program's shared memory

        """
        for idx, phase in enumerate(self.phases):
            # Skip inactive phases
            if not self.active[idx]:
                continue
            else:
                phase.create_shared_data(self.workspace, manager)

    def validate(self) -> dict[str, bool]:
        """Validate the pipeline by comparing the schema of the phases.

        Compare the expected incoming schema of a phase to the expected outgoing schema
        of the previous phase. The only excption is the first phase, which should only
        ever expect to recieve AT-TPC trace data.

        Any Phase which has it's schema set to None automatically passes validation.

        Returns
        -------
        dict[str, bool]
            A dictionary mapping phase name to validation success.
        """
        # First phase can't be validated, only user can control initial incoming format
        schema = self.phases[0].outgoing_schema
        success: dict[str, bool] = {}
        for idx, phase in enumerate(self.phases[1:]):
            test, schema = phase.validate(schema)
            success[f"{self.phases[idx]}->{phase}"] = test
        return success

    def run(
        self, run_list: list[int], msg_queue: SimpleQueue, seed: SeedSequence
    ) -> None:
        """Run the pipeline for a set of runs

        Each Phase is only run if it is active. Any artifact requested
        from an inactive Phase is expected to have already been created.

        Parameters
        ----------
        run_list: list[int]
            List of run numbers to be processed
        msg_queue: multiprocessing.SimpleQueue
            A queue to transmit progress messages through
        seed: numpy.random.SeedSequence
            A seed to initialize the pipeline random number generator

        """

        rng = default_rng(seed=seed)
        try:
            for run in run_list:
                result = PhaseResult(
                    Path(self.traces / f"{form_run_string(run)}.h5"), True, run
                )
                if not result.artifact_path.exists():
                    continue
                for idx, phase in enumerate(self.phases):
                    if self.active[idx]:
                        result = phase.run(result, self.workspace, msg_queue, rng)
                    else:
                        result = phase.construct_artifact(result, self.workspace)
        except Exception as e:
            spyral_except(__name__, e)
        msg_queue.put(StatusMessage("Complete", 0, 0, -1))


def _run_pipeline(
    pipeline: Pipeline,
    run_list: list[int],
    msg_queue: SimpleQueue,
    seed: SeedSequence,
    process_id: int,
) -> None:
    """A wrapper for multiprocessing. Do not call explicitly!"""
    init_spyral_logger_child(pipeline.workspace, process_id)
    pipeline.run(run_list, msg_queue, seed)


def start_pipeline(
    pipeline: Pipeline,
    run_min: int,
    run_max: int,
    n_procs: int = 1,
    disable_display: bool = False,
) -> None:
    """Function from which a Pipeline should be run

    Use this function to start running the Pipeline system. It
    will create a multiprocessed version of the pipeline and
    run a balanced load across the processes.

    Parameters
    ----------
    pipeline: Pipeline
        The pipeline to be run
    run_min: int
        The minimum run number (inclusive)
    run_max: int
        The maximum run number (inclusive)
    n_procs: int
        The number of parallel processes
    disable_display: bool, default=False
        Option to turn off terminal display. Default is False,
        i.e. terminal interface will be displayed.

    """
    # Setup
    # Note the manager exists outside the pipeline
    shared_manager = SharedMemoryManager(("", 50000))
    shared_manager.start()

    print(SPLASH)
    print(f"Creating workspace: {pipeline.workspace} ...", end=" ")
    pipeline.create_workspace()
    print("Done.")
    print("Initializing logs...", end=" ")
    init_spyral_logger_parent(pipeline.workspace)
    print("Done.")
    print("Creating any phase assets...", end=" ")
    pipeline.create_assets()
    print("Done.")
    print("Initializing shared memory...", end=" ")
    pipeline.create_shared_data(shared_manager)
    print("Done.")
    print("Validating Pipeline...", end=" ")
    result = pipeline.validate()
    if False in result.values():
        print("")
        print("Pipeline validation failed!")
        print(f"Status: {result}")
        return
    print("Pipeline successfully validated.")

    stacks = create_run_stacks(pipeline.traces, run_min, run_max, n_procs)
    stack_count = 0
    for stack in stacks:
        stack_count += len(stack)
    if len(stack) == 0:
        spyral_warn(
            __name__,
            f"No runs were found in trace path: {pipeline.traces}. Traces must exist to create workload!",
        )
    spyral_info(__name__, f"Run stacks: {stacks}")

    seq = SeedSequence()

    queues: list[SimpleQueue] = []
    processes: list[Process] = []
    pbars: list[tqdm] = []
    active_phases: list[str] = []
    seeds = seq.spawn(len(stacks))

    print("Running Spyral...")
    start = time.time()
    # Create the child processes
    for s in range(0, len(stacks)):
        queues.append(SimpleQueue())
        processes.append(
            Process(
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
        pbars.append(
            tqdm(total=0, disable=disable_display, miniters=1, mininterval=0.001)
        )
        active_phases.append("Waiting")  # put something here
        pbars[-1].set_description(f"| Process {s} | Waiting |")

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
            if msg.phase != active_phases[idx]:
                pbars[idx].reset(total=msg.total)
                pbars[idx].set_description(f"| Process {idx} | {msg}")
                active_phases[idx] = msg.phase
            pbars[idx].update(msg.increment)

    # Shutdown

    for bar in pbars:
        bar.close()

    for q in queues:
        q.close()

    for process in processes:
        process.join()

    shared_manager.shutdown()

    stop = time.time()
    duration = stop - start
    hours, sec = divmod(duration, 3600)
    minutes, sec = divmod(sec, 60)
    spyral_info(
        __name__, f"Total ellapsed time: {int(hours)} hrs {int(minutes)} min {sec:.4} s"
    )
