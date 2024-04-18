from tqdm import tqdm
from typing import Protocol
from dataclasses import dataclass
from pathlib import Path
from multiprocessing import SimpleQueue, Process
from copy import deepcopy


def form_run_string(run_number: int) -> str:
    """Make the run_* string

    Parameters
    ----------
    run_number: int
        The run number

    Returns
    -------
    str
        The run string
    """
    return f"run_{run_number:04d}"


def get_size_path(path: Path) -> int:
    """
    Get the size of a path item.

    Parameters
    ----------
    path: Path
        the path item to be inspected

    Returns
    -------
    int
        the size of the item at the given path, or 0 if no item exists.
    """
    if not path.exists():
        return 0
    else:
        return path.stat().st_size


def collect_runs(trace_path: Path, run_min: int, run_max: int) -> dict[int, int]:
    """Make dict of runs with the size of the raw data file

    Using the Workspace and the Config run_min and run_max, get a dict of run numbers to be processed
    and sort the list based on the size of the raw trace files. The dict key is the run number, and the
    associated data is the size of the raw trace file. Runs that do not exist or have no data are ommitted
    from the list.

    Parameters
    ----------
    ws: Workspace
        the project Workspace
    run_min: int
        the first run
    run_max: int
        the last run, inclusive

    Returns
    -------
    dict[int, int]
        a dictionary where the keys are run numbers and the values are the size of the associated raw trace files. The
        dict is sorted descending on the size of the raw trace files.
    """
    run_dict = {
        run: get_size_path(trace_path / f"{form_run_string(run)}.h5")
        for run in range(run_min, run_max + 1)
        if get_size_path(trace_path / f"{form_run_string(run)}.h5") != 0
    }
    run_dict = dict(sorted(run_dict.items(), key=lambda item: item[1], reverse=True))
    return run_dict


def create_run_stacks(
    trace_path: Path, run_min: int, run_max: int, n_stacks: int
) -> list[list[int]]:
    """Create a set of runs to be processed for each stack in n_stacks.

    Each stack is intended to be handed off to a single processor. As such,
    the goal is to balance the load of work across all of the stacks. collect_runs
    is used to retrieve the list of runs sorted on their sizes. The list of stacks
    is then snaked, depositing one run in each stack in each iteration.
    This seems to provide a fairly balanced load without too much effort.

    Parameters
    ----------
    config: Config
        the project configuration
    n_stacks: int
        the number of stacks, should be equal to number of processors

    Returns
    -------
    list[list[int]]
    The stacks. Each stack is a list of ints, where each value is a run number for that stack to process.
    """

    # create an empty list for each stack
    stacks = [[] for _ in range(0, n_stacks)]
    total_load = 0
    load_per_stack = [0 for _ in range(0, n_stacks)]
    sorted_runs = collect_runs(trace_path, run_min, run_max)
    if len(sorted_runs) == 0:
        return stacks

    # Snake through the stacks, putting the next run in the next stack
    # This should put the most equal data load across the stacks
    stack_index = 0
    reverse = False
    for run, size in sorted_runs.items():
        if stack_index != -1 and stack_index != n_stacks:
            stacks[stack_index].append(run)
            load_per_stack[stack_index] += size
        elif stack_index == -1:
            stack_index += 1
            stacks[stack_index].append(run)
            load_per_stack[stack_index] += size
            reverse = False
        elif stack_index == n_stacks:
            stack_index -= 1
            stacks[stack_index].append(run)
            load_per_stack[stack_index] += size
            reverse = True
        total_load += size

        if reverse:
            stack_index -= 1
        else:
            stack_index += 1

    # Remove any unused processors
    stacks = [s for s in stacks if len(s) != 0]

    print("Approximate data load per process:")
    for idx, load in enumerate(load_per_stack):
        print(f"Process {idx}: {float(load/total_load) * 100:.2f}%")

    return stacks


@dataclass
class StatusMessage:
    phase: str
    increment: int
    total: int
    run: int

    def __str__(self) -> str:
        f"Run {self.run} | Task: {self.phase} |"


@dataclass
class PhaseResult:
    artifact_path: Path
    successful: bool
    run_number: int


class PhaseLike(Protocol):
    name: str

    def run(
        self, payload: PhaseResult, workspace_path: Path, msg_queue: SimpleQueue
    ) -> PhaseResult:
        raise NotImplementedError

    def get_artifact_path(self, workspace_path: Path) -> Path:
        return workspace_path / f"{self.name}"

    def get_asset_storage_path(self, workspace_path: Path) -> Path:
        return workspace_path / f"{self.name}_assets"

    def create_assets(self, workspace_path: Path) -> bool:
        raise NotImplementedError


class Pipeline:

    def __init__(self, phases: list[PhaseLike], workspace_path: Path, trace_path: Path):
        self.phases = phases
        self.workspace = workspace_path
        self.traces = trace_path

    def create_assets(self) -> bool:
        for phase in self.phases:
            if not phase.create_assets(self.workspace):
                return False
        return True

    def run(self, run_list: list[int], msg_queue: SimpleQueue) -> None:

        for run in run_list:
            result = PhaseResult(
                Path(self.traces / f"{form_run_string(run)}.h5"), True, run
            )
            if not result.artifact_path.exists():
                continue
            for phase in self.phases:
                result = phase.run(result, self.workspace, msg_queue)


def _run_pipeline(
    pipeline: Pipeline, run_list: list[int], msg_queue: SimpleQueue
) -> None:
    pipeline.run(run_list, msg_queue)


def start_pipeline(
    pipeline: Pipeline,
    run_min: int,
    run_max: int,
    n_procs: int = 1,
    display: bool = True,
) -> None:

    pipeline.create_assets()

    stacks = create_run_stacks(pipeline.traces, run_min, run_max, n_procs)

    queues: list[SimpleQueue] = []
    processes: list[Process] = []
    pbars: list[tqdm] = []
    active_phases: list[str] = []

    # Create the child processes
    for s in range(0, len(stacks)):
        local_pipeline = deepcopy(pipeline)
        queues.append(SimpleQueue())
        processes.append(
            Process(
                target=_run_pipeline,
                args=(local_pipeline, stacks[s], queues[-1], s),
                daemon=False,
            )
        )
        pbars.append(tqdm(total=0, disable=display, miniters=1, mininterval=0.001))
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
