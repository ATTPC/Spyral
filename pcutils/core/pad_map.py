from .constants import INVALID_PAD_ID
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class PadData:
    x: float = 0.0
    y: float = 0.0
    gain: float = 1.0


class PadMap:
    def __init__(self, geometry_path: Path, gain_path: Path):
        self.map: dict[int, PadData] = {}
        self.load(geometry_path, gain_path)

    def load(self, geometry_path: Path, gain_path: Path):
        with open(geometry_path, "r") as geofile:
            geofile.readline() # Remove header
            lines = geofile.readlines()
            for pad_number, line in enumerate(lines):
                entries = line.split(",")
                self.map[pad_number] = PadData(float(entries[0]), float(entries[1]), 1.0)
        with open(gain_path, 'r') as gainfile:
            gainfile.readline()
            lines = gainfile.readlines()
            for pad_number, line in enumerate(lines):
                entries = line.split(',')
                self.map[pad_number].gain = float(entries[0])
        

    def get_pad_data(self, pad_number: int) -> Optional[PadData]:
        if (pad_number == INVALID_PAD_ID) or not (pad_number in self.map.keys()):
            return None
        
        return self.map[pad_number]