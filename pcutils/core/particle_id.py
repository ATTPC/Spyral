from ..plot.cut import Cut2D
from .nuclear_data import NuclearDataMap, NucleusData
from dataclasses import dataclass, field
from pathlib import Path
from json import load
from typing import Optional

@dataclass
class ParticleID:
    cut: Cut2D
    nucleus: NucleusData

def load_particle_id(cut_path: Path, nuclear_map: NuclearDataMap) -> Optional[ParticleID]:
    with open(cut_path, 'r') as cut_file:
        json_data = load(cut_file)
        if 'name' not in json_data or 'vertices' not in json_data or 'Z' not in json_data or 'A' not in json_data:
            print(f'ParticleID could not load cut in {cut_path}, the requested data is not present.')
            return None

        pid = ParticleID(Cut2D(json_data['name'], json_data['vertices']), nuclear_map.get_data(json_data['Z'], json_data['A']))

        if pid.nucleus.A == 0:
            print(f'Nucleus Z: {json_data["Z"]} A: {json_data["A"]} requested by ParticleID does not exist.')
            return None

        return pid
