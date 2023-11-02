from enum import Enum

class Phase(Enum):
    '''
    Enum used to indicate to the parent process what stage
    of the analysis the child process is on. All cases should
    be castable into a string description
    '''
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
    '''
    # Status Message
    Message type for transmitting data from the child process to the parent process
    For now just contains the run number being processed, a Phase enum, and the progress increment
    '''

    def __init__(self, run: int, phase: Phase, progress: float):
        self.run = run
        self.phase = phase
        self.progress = progress

    def task_str(self) -> str:
        '''
        Construct a string describing the current task being computed on the child process
        '''
        return f'Run {self.run} | Task: {self.phase} |'