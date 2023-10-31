from enum import Enum

class Phase(Enum):
    CLOUD: int = 0
    CLUSTER: int = 1
    ESTIMATE: int = 2
    SOLVE: int = 3
    WAIT: int = 4

    def __str__(self) -> str:
        match self:
            case Phase.CLOUD:
                return 'Forming clouds'
            case Phase.CLUSTER:
                return 'Forming clusters'
            case Phase.ESTIMATE:
                return 'Estimating'
            case Phase.SOLVE:
                return 'Solving'
            case _:
                return 'Waiting...'

class StatusMessage:

    def __init__(self, run: int, phase: Phase, progress: float):
        self.run = run
        self.phase = phase
        self.progress = progress

    def task_str(self) -> str:
        return f'Run {self.run} | Task: {self.phase} |'