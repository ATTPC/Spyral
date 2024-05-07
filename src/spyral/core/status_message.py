from dataclasses import dataclass


@dataclass
class StatusMessage:
    phase: str
    increment: int
    total: int
    run: int

    def __str__(self) -> str:
        return f"Run {self.run} | Task: {self.phase} |"
