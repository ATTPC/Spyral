import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from multiprocessing import SimpleQueue
from numpy.random import Generator
from collections import Counter
from multiprocessing.managers import SharedMemoryManager

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self
from typing import Any


@dataclass
class ArtifactSchema:
    """Dataclass representing a phase artifact schema

    Used to validate a pipeline

    Attributes
    ----------
    extension: str
        The artifact file extension (i.e. ".h5" or ".parquet")
    structure: list[str] | dict[str, Any] | None
        The artifact data structure. Can be a list of strings (dataframe columns),
        a dictionary (HDF5 groups) or None (not specified)

    """

    extension: str
    structure: list[str] | dict[str, Any] | None

    def __eq__(self, value: Self) -> bool:
        """Allow comparison of schemas for equivalency

        Parameters
        ----------
        value: ArtifactSchema
            The schema to compare to

        Returns
        -------
        bool
            True if the schemas are equivalent or False if they are different

        """
        return self.extension == value.extension and (
            Counter(self.structure) == Counter(value.structure)
        )


@dataclass
class PhaseResult:
    """Dataclass representing the result of a Phase

    Attributes
    ----------
    artifact_path: Path
        Path to the artifact (data) created by this Phase
    successful: bool
        True if the Phase was successful or False if it failed
    run_number: int
        The run number that was analyzed
    metadata: dict
        A dict containing any extra data needed for the next Phase

    Methods
    -------
    invalid_result(run_number)
        Create an invalid PhaseResult
    """

    artifact_path: Path
    successful: bool
    run_number: int
    metadata: dict = field(default_factory=dict)

    @staticmethod
    def invalid_result(run_number: int):
        return PhaseResult(Path("Invalid"), False, run_number)


class PhaseLike(ABC):
    """Abstract Base Class all Phases inherit from

    Parameters
    ----------
    name: str
        The name of the Phase (Pointcloud, Cluster, Estimation, etc.)
    incoming_schema: ArtifactSchema | None
        Optional schema describing the expected incoming artifact (payload).
        Default is None.
    outgoing_schema: ArtifactSchema | None
        Optional schema describing the expected outgoing artifact (result).
        Default is None.

    Attributes
    ----------
    name: str
        The name of the Phase (Pointcloud, Cluster, Estimation, etc.)
    incoming_schema: ArtifactSchema | None
        Schema describing the expected incoming artifact (payload).
    outgoing_schema: ArtifactSchema | None
        Schema describing the expected outgoing artifact (result).

    Methods
    -------
    run(payload, workspace_path, msg_queue, rng)
        Run the phase. This is an abstract method.
    create_assets(workspace_path)
        Create any necessary assets to run. This is an abstract method.
    get_artifact_path(workspace_path)
        Get the path to the phase artifacts.
    get_asset_storage_path(workspace_path)
        Get the path to the phase assets.
    validate(incoming)
        Validate the phase by comparing the given incoming
        schema to the expected incoming schema.
    """

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
        """Run the phase. This is an abstract method.

        It must be overriden by any child class.

        Parameters
        ----------
        payload: PhaseResult
            The result from the previous Phase
        workspace_path: pathlib.Path
            The path to the workspace
        msg_queue: multiprocessing.SimpleQueue
            The queue for submitting progress messages
        rng: numpy.random.Generator
            A random number generator

        Returns
        -------
        PhaseResult
            The result of this phase containing the artifact information
        """
        raise NotImplementedError

    @abstractmethod
    def create_assets(self, workspace_path: Path) -> bool:
        """Create any necessary assets to run. This is an abstract method.

        It must be overriden by any child class.

        Parameters
        ----------
        workspace_path: pathlib.Path
            The path to the workspace

        Returns
        -------
        bool
            True if artifacts are successfully created, False if unsuccessful
        """
        raise NotImplementedError

    def create_shared_data(
        self, workspace_path: Path, manager: SharedMemoryManager
    ) -> None:
        """Create shared-memory data for use across all processes

        This should be used sparingly, in cases where it would be beneficial to share large memory
        footprints across processes in a read-only way. In general, most phases should not override
        and re-implement this method.

        The obvious case for this is in the default InterpSolverPhase where we want to share
        the interpolation mesh across processes.

        Parameters
        ----------
        workspace_path: Path
            The path tot he workspace
        manager: SharedMemoryManager
            A resource manager for parallel processes
        """
        return

    @abstractmethod
    def construct_artifact(
        self, payload: PhaseResult, workspace_path: Path
    ) -> PhaseResult:
        """Construct a new artifact

        The artifact_path should be initialized to aa good path,
        success True, and run_number preserved.

        Parameters
        ----------
        payload: PhaseResult
            The result of the previous Phase
        workspace_path: pathlib.Path
            The path to the workspace

        Returns
        -------
        PhaseResult
            A new artifact with the path initialized
        """
        raise NotImplementedError

    def get_artifact_path(self, workspace_path: Path) -> Path:
        """Get the path to the phase artifacts.

        The data artifact (Phase result) is stored at a
        specific path in the workspace based on the name of the
        Phase.

        Parameters
        ----------
        workspace_path: pathlib.Path
            The path to the workspace

        Returns
        -------
        pathlib.Path
            The artifact path

        """
        path = workspace_path / f"{self.name}"
        return path

    def get_asset_storage_path(self, workspace_path: Path) -> Path:
        """Get the path to the phase assets.

        All Phase assets are stored at a specific path
        in the workspace based on the name of the Phase.

        Parameters
        ----------
        workspace_path: pathlib.Path
            The path to the workspace

        Returns
        -------
        pathlib.Path
            The asset path

        """
        path = workspace_path / f"{self.name}_assets"
        return path

    def validate(
        self, incoming: ArtifactSchema | None
    ) -> tuple[bool, ArtifactSchema | None]:
        """Validate the phase by comparing the given incoming schema to the expected incoming schema.

        Parameters
        ----------
        incoming: ArtifactSchema | None
            The schema of the artifact of the previous Phase

        Returns
        -------
        tuple[bool, ArtifactSchema | None]
            A tuple of the result of the result of the comparison and
            the outgoing (result) schema of this Phase
        """
        success = (
            incoming is None
            or self.incoming_schema is None
            or self.incoming_schema == incoming
        )
        return (success, self.outgoing_schema)
