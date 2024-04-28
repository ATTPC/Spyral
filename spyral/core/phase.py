from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from multiprocessing import SimpleQueue
from numpy.random import Generator
from typing import Any, Self


@dataclass
class ArtifactSchema:
    extension: str
    structure: list[str] | dict[str, Any] | None

    def __eq__(self, value: Self) -> bool:
        return self.extension == value.extension and self.structure == value.structure


@dataclass
class PhaseResult:
    artifact_path: Path
    successful: bool
    run_number: int
    metadata: dict = field(default_factory=dict)


class PhaseLike(ABC):

    def __init__(
        self,
        name: str,
        incoming_schema: ArtifactSchema | None = None,
        outgoing_schema: ArtifactSchema | None = None,
    ):
        self.name = name
        self.incoming_schema = incoming_schema
        self.outgoing_schema = outgoing_schema

    def __str__(self) -> str:
        return f"{self.name}Phase"

    @abstractmethod
    def run(
        self,
        payload: PhaseResult,
        workspace_path: Path,
        msg_queue: SimpleQueue,
        rng: Generator,
    ) -> PhaseResult:
        raise NotImplementedError

    @abstractmethod
    def create_assets(self, workspace_path: Path) -> bool:
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

    def validate(
        self, incoming: ArtifactSchema | None
    ) -> tuple[bool, ArtifactSchema | None]:
        success = (
            incoming is None
            or self.incoming_schema is None
            or self.incoming_schema == incoming
        )
        return (success, self.outgoing_schema)
