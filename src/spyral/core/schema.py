from pathlib import Path
import sys

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self
from typing import Any
from dataclasses import dataclass
from json import loads, load


@dataclass
class PhaseResult:
    artifacts: dict[str, Path]
    successful: bool
    run_number: int

    @classmethod
    def invalid_result(cls, run_number: int) -> Self:
        return cls({}, False, run_number)


class ArtifactSchema:
    def __init__(self, extension: str | None = None, data: dict | None = None):
        self.extension = extension
        self.data = data

    def __eq__(self, other: Self) -> bool:
        if self.extension != other.extension or self.data != other.data:
            return False
        return True


class ResultSchema:
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

    def __init__(self, json_string: str | None = None, json_file: Path | None = None):
        self.artifacts: None | dict[str, ArtifactSchema] = None
        self.load_schema(json_string, json_file)

    def load_schema(
        self, json_string: str | None = None, json_file: Path | None = None
    ):
        json_data: dict[str, Any]
        if json_string is not None:
            json_data = loads(json_string)
            self.artifacts = {}
        elif json_file is not None:
            with open(json_file) as json_handle:
                json_data = load(json_handle)
            self.artifacts = {}
        else:
            return

        for key, artifact in json_data.items():
            self.artifacts[key] = ArtifactSchema(
                artifact["extension"], artifact["data"]
            )

    def __eq__(self, other: Self) -> bool:
        """Allow comparison of schemas for equivalency

        Parameters
        ----------
        other: ArtifactSchema
            The schema to compare to

        Returns
        -------
        bool
            True if the schemas are equivalent or False if they are different

        """
        return self.artifacts == other.artifacts
