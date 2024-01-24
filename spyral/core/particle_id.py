from spyral_utils.nuclear import NuclearDataMap, NucleusData
from spyral_utils.plot import Cut2D

from dataclasses import dataclass
from pathlib import Path
from json import load


@dataclass
class ParticleID:
    """Thin wrapper over spyral-utils Cut2D that attaches a NucleusData

    Used to gate on particle groups in Brho and dEdx

    Attributes
    ----------
    cut: spyral_utils.plot.Cut2D
        A spyral-utils Cut2D on brho and dEdx estimated parameters
    nucleus: NucleusData
        The nucleus species associated with this ID

    """

    cut: Cut2D
    nucleus: NucleusData


def load_particle_id(cut_path: Path, nuclear_map: NuclearDataMap) -> ParticleID | None:
    """Load a ParticleID from a JSON file

    Parameters
    ----------
    cut_path: Path
        The path to a JSON file containing a ParticleID
    nuclear_map: NuclearDataMap
        An instance of a spyral_utils.nuclear.NuclearDataMap

    Returns
    -------
    ParticleID | None
        The deserialized ParticleID or None on failure
    """
    with open(cut_path, "r") as cut_file:
        json_data = load(cut_file)
        if (
            "name" not in json_data
            or "vertices" not in json_data
            or "Z" not in json_data
            or "A" not in json_data
        ):
            print(
                f"ParticleID could not load cut in {cut_path}, the requested data is not present."
            )
            return None

        pid = ParticleID(
            Cut2D(json_data["name"], json_data["vertices"]),
            nuclear_map.get_data(json_data["Z"], json_data["A"]),
        )

        if pid.nucleus.A == 0:
            print(
                f'Nucleus Z: {json_data["Z"]} A: {json_data["A"]} requested by ParticleID does not exist.'
            )
            return None

        return pid
