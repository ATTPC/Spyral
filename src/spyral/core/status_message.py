from dataclasses import dataclass


@dataclass
class StatusMessage:
    """A status message to be emitted by a Phase

    This is how we give progress updates.

    Attributes
    ----------
    phase: str
        The phase name
    increment: int
        The amount of progress we made
    total: int
        The total progress size
    run: int
        The current run number

    """

    phase: str
    increment: int
    total: int
    run: int

    def __str__(self) -> str:
        return f"Run {self.run} | Task: {self.phase} |"
