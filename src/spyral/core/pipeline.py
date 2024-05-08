from .run_stacks import form_run_string, create_run_stacks
from .status_message import StatusMessage
from .phase import PhaseLike, PhaseResult, ArtifactSchema
from .spy_log import init_spyral_logger_parent, init_spyral_logger_child

from tqdm import tqdm
from pathlib import Path
from multiprocessing import SimpleQueue, Process
from copy import deepcopy
from numpy.random import SeedSequence, default_rng


class Pipeline:
    """A representation of an analysis pipeline in Spyral

    The Pipeline controls the analysis workflow. It is given a list
    of PhaseLike objects and paths to workspace and trace data and runs the
    data through the pipeline.

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
        if not self.workspace.exists():
            self.workspace.mkdir()
        self.traces = trace_path

    def create_assets(self) -> bool:
        for phase in self.phases:
            if not phase.create_assets(self.workspace):
                return False
        return True

    def validate(self) -> dict[str, bool]:
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

        rng = default_rng(seed=seed)
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


def _run_pipeline(
    pipeline: Pipeline,
    run_list: list[int],
    msg_queue: SimpleQueue,
    seed: SeedSequence,
    process_id: int,
) -> None:
    init_spyral_logger_child(pipeline.workspace, process_id)
    pipeline.run(run_list, msg_queue, seed)


def start_pipeline(
    pipeline: Pipeline,
    run_min: int,
    run_max: int,
    n_procs: int = 1,
    disable_display: bool = False,
) -> None:

    init_spyral_logger_parent(pipeline.workspace)

    pipeline.create_assets()
    result = pipeline.validate()
    if False in result.values():
        print("Pipeline validation failed!")
        print(f"Status: {result}")
        return
    print("Pipeline successfully validated.")

    stacks = create_run_stacks(pipeline.traces, run_min, run_max, n_procs)

    seq = SeedSequence()

    queues: list[SimpleQueue] = []
    processes: list[Process] = []
    pbars: list[tqdm] = []
    active_phases: list[str] = []
    seeds = seq.spawn(len(stacks))

    # Create the child processes
    for s in range(0, len(stacks)):
        local_pipeline = deepcopy(pipeline)
        queues.append(SimpleQueue())
        processes.append(
            Process(
                target=_run_pipeline,
                args=(local_pipeline, stacks[s], queues[-1], seeds[s], s),
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
