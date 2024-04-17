from ..parallel.run_stack import create_run_stacks

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


@dataclass
class PhaseResult:
    artifact_path: Path
    successful: bool


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
            result = PhaseResult(Path(self.traces / f"{form_run_string(run)}.h5"), True)
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

    stacks = create_run_stacks(run_min, run_max, n_procs)  # needs reimpl

    queues: list[SimpleQueue] = []
    processes: list[Process] = []
    pbars: list[tqdm] = []
    status: list[StatusMessage] = []  # needs reimpl

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
        status.append(SomethingHere())  # put something here
        pbars[-1].set_description(f"| Process {s} | { str(status[-1]) } |")
