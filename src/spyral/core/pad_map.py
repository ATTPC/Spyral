from .constants import INVALID_PAD_ID
from .hardware_id import HardwareID, generate_electronics_id
from .config import PadParameters
from dataclasses import dataclass, field
from importlib import resources

from .legacy_beam_pads import LEGACY_BEAM_PADS


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
    hardware: HardwareID
        The pad HardwareID
    """

    x: float = 0.0
    y: float = 0.0
    gain: float = 1.0
    time_offset: float = 0.0
    scale: float = 0.0
    hardware: HardwareID = field(default_factory=HardwareID)


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
    elec_map: dict[int -> int]
        Essentially a reverse map of HardwareID -> pad number

    Methods
    -------
    PadMap(geometry_path: Path, gain_path: Path, time_correction_path: Path, electronics_path: Path, scale_path: Path)
        Construct the PadMap
    load(geometry_path: Path, gain_path: Path, time_correction_path: Path, electronics_path: Path, scale_path: Path)
        load the map data
    get_pad_data(pad_number: int) -> PadData | None
        Get the PadData for a given pad. Returns None if the pad does not exist
    get_pad_from_hardware(hardware: HardwareID) -> int | None
        Get the pad number for a given HardwareID. Returns None if the HardwareID is invalid

    """

    def __init__(self, params: PadParameters):
        self.map: dict[int, PadData] = {}
        self.elec_map: dict[int, int] = {}
        self.is_valid = False
        if params.is_default or params.is_default_legacy:
            self.load_default(params.is_default_legacy)
        else:
            self.load(params)

    def load_default(self, is_legacy: bool = False):
        """Load maps with defaults shipped in the package

        Spyral ships with default maps that should be maintained
        for compatibility with the current detector configuration.

        Parameters
        ----------
        is_legacy: bool, default=False
            Optional flag indicating if the legacy maps should
            be loaded
        """

        suffix = ""
        if is_legacy:
            suffix = "_legacy"

        directory = resources.files("spyral.data")

        geom_handle = directory.joinpath(f"padxy.csv")
        with resources.as_file(geom_handle) as geopath:
            geofile = open(geopath, "r")
            geofile.readline()  # Remove header
            lines = geofile.readlines()
            for pad_number, line in enumerate(lines):
                entries = line.split(",")
                self.map[pad_number] = PadData(x=float(entries[0]), y=float(entries[1]))
            geofile.close()
        #
        # NOTE This is left in as a comment to show how a gain map could
        # be implemented. This is not used in current Spyral as it has no
        # impact on the analysis
        #
        # gain_handle = directory.joinpath("pad_gain_map.csv")
        # with resources.as_file(gain_handle) as gainpath:
        #     gainfile = open(gainpath, "r")
        #     gainfile.readline()
        #     lines = gainfile.readlines()
        #     for pad_number, line in enumerate(lines):
        #         entries = line.split(",")
        #         self.map[pad_number].gain = float(entries[0])
        #     gainfile.close()
        #
        time_handle = directory.joinpath("pad_time_correction.csv")
        with resources.as_file(time_handle) as timepath:
            timefile = open(timepath, "r")
            timefile.readline()
            lines = timefile.readlines()
            for pad_number, line in enumerate(lines):
                entries = line.split(",")
                self.map[pad_number].time_offset = float(entries[0])
            timefile.close()
        elec_handle = directory.joinpath(f"pad_electronics{suffix}.csv")
        with resources.as_file(elec_handle) as elecpath:
            elecfile = open(elecpath, "r")
            elecfile.readline()
            lines = elecfile.readlines()
            for line in lines:
                entries = line.split(",")
                hardware = HardwareID(
                    int(entries[4]),
                    int(entries[0]),
                    int(entries[1]),
                    int(entries[2]),
                    int(entries[3]),
                )
                self.map[hardware.pad_id].hardware = hardware
                self.elec_map[generate_electronics_id(hardware)] = hardware.pad_id
            elecfile.close()
        scale_handle = directory.joinpath("pad_scale.csv")
        with resources.as_file(scale_handle) as scalepath:
            scalefile = open(scalepath, "r")
            scalefile.readline()
            lines = scalefile.readlines()
            for pad_number, line in enumerate(lines):
                entries = line.split(",")
                self.map[pad_number].scale = float(entries[0])
            scalefile.close()
        self.is_valid = True

    def load(self, params: PadParameters):
        """Load the map data

        Parameters
        ----------
        params: PadParameters
            Paths to map files
        """
        with open(params.pad_geometry_path, "r") as geofile:
            geofile.readline()  # Remove header
            lines = geofile.readlines()
            for pad_number, line in enumerate(lines):
                entries = line.split(",")
                self.map[pad_number] = PadData(x=float(entries[0]), y=float(entries[1]))
        #
        # NOTE This is left in as a comment to show how a gain map could
        # be implemented. This is not used in current Spyral as it has no
        # impact on the analysis
        #
        # with open(params.pad_gain_path, "r") as gainfile:
        #     gainfile.readline()
        #     lines = gainfile.readlines()
        #     for pad_number, line in enumerate(lines):
        #         entries = line.split(",")
        #         self.map[pad_number].gain = float(entries[0])
        #
        with open(params.pad_time_path, "r") as timefile:
            timefile.readline()
            lines = timefile.readlines()
            for pad_number, line in enumerate(lines):
                entries = line.split(",")
                self.map[pad_number].time_offset = float(entries[0])

        with open(params.pad_electronics_path, "r") as elecfile:
            elecfile.readline()
            lines = elecfile.readlines()
            for line in lines:
                entries = line.split(",")
                hardware = HardwareID(
                    int(entries[4]),
                    int(entries[0]),
                    int(entries[1]),
                    int(entries[2]),
                    int(entries[3]),
                )
                self.map[hardware.pad_id].hardware = hardware
                self.elec_map[generate_electronics_id(hardware)] = hardware.pad_id

        with open(params.pad_scale_path, "r") as scalefile:
            scalefile.readline()
            lines = scalefile.readlines()
            for pad_number, line in enumerate(lines):
                entries = line.split(",")
                self.map[pad_number].scale = float(entries[0])
        self.is_valid = True

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
        if (pad_number == INVALID_PAD_ID) or not (pad_number in self.map.keys()):
            return None

        return self.map[pad_number]

    def get_pad_from_hardware(self, hardware: HardwareID) -> int | None:
        """Get the pad number associated with a HardwareID

        Returns None if the HardwareID is invalid

        Parameters
        ----------
        hardware: HardwareID
            A HardwareID

        Returns
        -------
        int | None
            The associated pad number, or None if the HardwareID is invalid

        """
        key = generate_electronics_id(hardware)
        if key in self.elec_map.keys():
            return self.elec_map[generate_electronics_id(hardware)]

        return None

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
