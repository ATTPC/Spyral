from .run_stacks import form_run_string, create_run_stacks
from .status_message import StatusMessage
from .phase import PhaseLike, PhaseResult
from .spy_log import init_spyral_logger_parent, init_spyral_logger_child

from tqdm import tqdm
from pathlib import Path
from multiprocessing import SimpleQueue, Process
from copy import deepcopy


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
    pipeline: Pipeline, run_list: list[int], msg_queue: SimpleQueue, process_id: int
) -> None:
    init_spyral_logger_child(pipeline.workspace, process_id)
    pipeline.run(run_list, msg_queue)


def start_pipeline(
    pipeline: Pipeline,
    run_min: int,
    run_max: int,
    n_procs: int = 1,
    display: bool = True,
) -> None:

    init_spyral_logger_parent(pipeline.workspace)

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
