from enum import Enum


class Phase(Enum):
    """Enum used to indicate to the parent process what stageof the analysis the child process is on.

    All cases should be castable into a string description

    Attributes
    ----------
    CLOUD: int
        Point cloud phase (0)
    CLUSTER: int
        Clustering phase (1)
    ESTIMATE: int
        Estimating phase (2)
    SOLVE: int
        Solver phase (3)
    WAIT: int
        Waiting (4)

    Methods
    -------
    __str__() -> str
        Convert the Enum to a string message
    """

    CLOUD: int = 0
    CLUSTER: int = 1
    ESTIMATE: int = 2
    SOLVE: int = 3
    WAIT: int = 4

    def __str__(self) -> str:
        """Convert the Enum to a string message

        Returns
        -------
        str
            The message
        """
        match self:
            case Phase.CLOUD:
                return "Forming clouds"
            case Phase.CLUSTER:
                return "Forming clusters"
            case Phase.ESTIMATE:
                return "Estimating"
            case Phase.SOLVE:
                return "Solving"
            case _:
                return "Waiting..."


class StatusMessage:
    """Message type for transmitting data from the child process to the parent process

    For now just contains the run number being processed, a Phase enum, and the progress increment

    Attributes
    ----------
    run: int
        Run number being processed
    phase: Phase
        Which phase is being run
    progress: float
        How much progress has been made

    Methods
    -------
    StatusMessage(run: int, phase: Phase, progress: float)
        Construct the message
    __str__() -> str:
        Construct a string describing the current task being computed on the child process
    """

    def __init__(self, run: int, phase: Phase, progress: float):
        self.run = run
        self.phase = phase
        self.progress = progress

    def __str__(self) -> str:
        """Construct a string describing the current task being computed on the child process

        Returns
        -------
        str
            The message
        """
        return f"Run {self.run} | Task: {self.phase} |"
