from constants import INVALID_PAD_ID
from pathlib import Path

class PadMap:
    def __init__(self, file_path: Path):
        self.map: dict[int, tuple[float, float]] = {}
        self.load(file_path)

    def load(self, file_path: Path):
        with open(file_path, "r") as mapfile:
            mapfile.readline() # Remove header
            lines = mapfile.readlines()
            position = (0.0, 0.0)
            for pad_number, line in enumerate(lines):
                entries = line.split(",")
                position[0] = entries[0]
                position[1] = entries[1]
                self.map[pad_number] = position

    def get_pad_position(self, pad_number: int) -> tuple[float, float]:
        if (pad_number == INVALID_PAD_ID) or not (pad_number in self.map.keys()):
            return (0.0, 0.0)
        
        return self.map[pad_number]