from .constants import INVALID_PAD_ID
from .hardware_id import HardwareID, generate_electronics_id
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class PadData:
    x: float = 0.0
    y: float = 0.0
    gain: float = 1.0
    time_offset: float = 0.0
    scale: float = 0.0
    hardware: HardwareID = field(default_factory=HardwareID)

class PadMap:
    def __init__(self, geometry_path: Path, gain_path: Path, time_correction_path: Path, electronics_path: Path, scale_path: Path):
        self.map: dict[int, PadData] = {}
        self.elec_map: dict[int, int] = {}
        self.load(geometry_path, gain_path, time_correction_path, electronics_path, scale_path)

    def load(self, geometry_path: Path, gain_path: Path, time_correction_path: Path, electronics_path: Path, scale_path: Path):
        with open(geometry_path, "r") as geofile:
            geofile.readline() # Remove header
            lines = geofile.readlines()
            for pad_number, line in enumerate(lines):
                entries = line.split(",")
                self.map[pad_number] = PadData(x=float(entries[0]), y=float(entries[1]))
        with open(gain_path, 'r') as gainfile:
            gainfile.readline()
            lines = gainfile.readlines()
            for pad_number, line in enumerate(lines):
                entries = line.split(',')
                self.map[pad_number].gain = float(entries[0])
        with open(time_correction_path, 'r') as timefile:
            timefile.readline()
            lines = timefile.readlines()
            for pad_number, line in enumerate(lines):
                entries = line.split(",")
                self.map[pad_number].time_offset = float(entries[0])

        with open(electronics_path, 'r') as elecfile:
            elecfile.readline()
            lines = elecfile.readlines()
            for line in lines:
                entries = line.split(',')
                hardware = HardwareID(int(entries[4]), int(entries[0]), int(entries[1]), int(entries[2]), int(entries[3]))
                self.map[hardware.pad_id].hardware = hardware
                self.elec_map[generate_electronics_id(hardware)] = hardware.pad_id
        
        with open(scale_path, "r") as scalefile:
            scalefile.readline()
            lines = scalefile.readlines()
            for pad_number, line in enumerate(lines):
                entries = line.split(",")
                self.map[pad_number].scale = float(entries[0])

        

    def get_pad_data(self, pad_number: int) -> PadData | None:
        if (pad_number == INVALID_PAD_ID) or not (pad_number in self.map.keys()):
            return None
        
        return self.map[pad_number]
    
    def get_pad_from_hardware(self, hardware: HardwareID) -> int | None:
        key = generate_electronics_id(hardware)
        if key in self.elec_map.keys():
            return self.elec_map[generate_electronics_id(hardware)]
        
        return None