from .constants import INVALID_PAD_ID
from .config import PadParameters, DEFAULT_MAP
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
import numpy as np
from .legacy_beam_pads import LEGACY_BEAM_PADS, DISABLE_PADS

@dataclass
class MergerPadData: 
    """Dataclass for storing the hardware information and match with the right padmap.

    Attributes
    ----------
    params : Path 
        Gives the pad of the experiment padmap. If DEFAULT_MAP is given, it won't open a file.
    """
    params : PadParameters
    def load_merger_padmap(self)->np.ndarray:
        geopath = self.params.merger_padmap_path
        pad_map = np.full(
            (11, 4, 4, 68),
            INVALID_PAD_ID,
            dtype=np.int32,
        )
        with open(geopath, "r") as geofile: 
            next(geofile) # skip headers 
            for line in geofile:
                entries = line.strip().split(",")
                #assign right pad number: Cobo, AsAd, AGet Channel 
                pad_map[int(entries[0]), int(entries[1]), int(entries[2]), int(entries[3])] = int(entries[4])
        return pad_map
    def is_merger_padmap_loaded(self)->bool:
        if self.params.merger_padmap_path == DEFAULT_MAP: 
            return False
        else: 
            return True

@dataclass
class PadData:
    """Dataclass for storing AT-TPC pad information

    Attributes
    ----------
    x: float
        The pad x-coordinates
    y: float
        The pad y-coordinates
    gain: float
        The relative pad gain
    time_offset: float
        The pad time offset due to GET electronics
    scale: float
        The pad scale (big pad or small pad)
    """

    x: float = 0.0
    y: float = 0.0
    gain: float = 1.0
    time_offset: float = 0.0
    scale: float = 0.0


class PadMap:
    """A map of pad number to PadData

    Parameters
    ----------
    params: PadParameters
        Pad map configuration parameters

    Attributes
    ----------
    map: dict[int, PadData]
        The forward map (pad number -> PadData)

    Methods
    -------
    PadMap(params)
        Construct the PadMap
    get_pad_data(pad_number)
        Get the PadData for a given pad. Returns None if the pad does not exist
    is_beam_pad(pad_number)
        Check if a pad is a beam pad (returns True if beam, False otherwise)
    is_pad_disable(padnumber)
        Check if a pad is disable by user (returns True if disable, False otherwise)
    """

    def __init__(self, params: PadParameters):
        self.map: dict[int, PadData] = {}
        self.is_valid = False
        self.load(params)
        self.disable_pads(params)

    def load(self, params: PadParameters):
        """Load the map data

        Parameters
        ----------
        params: PadParameters
            Paths to map files
        """
        # Defaults are here
        directory = resources.files("spyral.data")

        # Geometry
        if params.pad_geometry_path == DEFAULT_MAP:
            geom_handle = directory.joinpath("padxy.csv")
            with resources.as_file(geom_handle) as geopath:
                geofile = open(geopath, "r")
                geofile.readline()  # Remove header
                lines = geofile.readlines()
                for pad_number, line in enumerate(lines):
                    entries = line.split(",")
                    self.map[pad_number] = PadData(
                        x=float(entries[0]), y=float(entries[1])
                    )
                geofile.close()
        else:
            with open(params.pad_geometry_path, "r") as geofile:
                geofile.readline()  # Remove header
                lines = geofile.readlines()
                for pad_number, line in enumerate(lines):
                    entries = line.split(",")
                    self.map[pad_number] = PadData(
                        x=float(entries[0]), y=float(entries[1])
                    )

        # Time
        if params.pad_time_path == DEFAULT_MAP:
            time_handle = directory.joinpath("pad_time_correction.csv")
            with resources.as_file(time_handle) as timepath:
                timefile = open(timepath, "r")
                timefile.readline()
                lines = timefile.readlines()
                for pad_number, line in enumerate(lines):
                    entries = line.split(",")
                    self.map[pad_number].time_offset = float(entries[0])
                timefile.close()
        else:
            with open(params.pad_time_path, "r") as timefile:
                timefile.readline()
                lines = timefile.readlines()
                for pad_number, line in enumerate(lines):
                    entries = line.split(",")
                    self.map[pad_number].time_offset = float(entries[0])

        # Scale
        if params.pad_scale_path == DEFAULT_MAP:
            scale_handle = directory.joinpath("pad_scale.csv")
            with resources.as_file(scale_handle) as scalepath:
                scalefile = open(scalepath, "r")
                scalefile.readline()
                lines = scalefile.readlines()
                for pad_number, line in enumerate(lines):
                    entries = line.split(",")
                    self.map[pad_number].scale = float(entries[0])
                scalefile.close()
        else:
            with open(params.pad_scale_path, "r") as scalefile:
                scalefile.readline()
                lines = scalefile.readlines()
                for pad_number, line in enumerate(lines):
                    entries = line.split(",")
                    self.map[pad_number].scale = float(entries[0])

        self.is_valid = True

    def disable_pads(self, params: PadParameters):
        geom_path = params.disable_pads
        if geom_path != DEFAULT_MAP: 
            with open(geom_path, "r") as geofile: 
                next(geofile)
                for line in geofile: 
                    pad=line.strip()
                    DISABLE_PADS.append(int(pad))

    def get_pad_data(self, pad_number: int) -> PadData | None:
        """Get the PadData associated with a pad number

        Returns None if the pad number is invalid

        Parameters
        ----------
        pad_number: int
            A pad number

        Returns
        -------
        PadData | None
            The associated PadData, or None if the pad number is invalid

        """
        if (pad_number == INVALID_PAD_ID) or pad_number not in self.map.keys():
            return None

        return self.map[pad_number]

    def is_beam_pad(self, pad_id: int) -> bool:
        """Check if a pad is a Beam Pad (TM)

        Parameters
        ----------
        pad_id: int
            The pad number to check

        Returns
        -------
        bool
            True if Beam Pad, False otherwise
        """
        return pad_id in LEGACY_BEAM_PADS
    
    def is_pad_disable(self, pad_id:int)->bool:
        """
            Disable pads if allowed by user. 
        """
        return pad_id in DISABLE_PADS
