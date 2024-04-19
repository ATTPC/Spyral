from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from multiprocessing import SimpleQueue


@dataclass
class PhaseResult:
    artifact_path: Path
    successful: bool
    run_number: int
    metadata: dict = field(default_factory=dict)


class PhaseLike(ABC):

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(
        self, payload: PhaseResult, workspace_path: Path, msg_queue: SimpleQueue
    ) -> PhaseResult:
        raise NotImplementedError

    def get_artifact_path(self, workspace_path: Path) -> Path:
        path = workspace_path / f"{self.name}"
        if not path.exists():
            path.mkdir()
        return path

    def get_asset_storage_path(self, workspace_path: Path) -> Path:
        path = workspace_path / f"{self.name}_assets"
        if not path.exists():
            path.mkdir()
        return path

    @abstractmethod
    def create_assets(self, workspace_path: Path) -> bool:
        raise NotImplementedError
