from dataclasses import dataclass
from .constants import INVALID_PAD_ID
import polars as pl 
from pathlib import Path
import numpy as np
from .pad_map import MergerPadData
from .constants import RIGHT_PAD_ID

GET_DATA_COBO_INDEX: int = 0
GET_DATA_ASAD_INDEX: int = 1
GET_DATA_AGET_INDEX: int = 2
GET_DATA_CHANNEL_INDEX: int = 3
GET_DATA_PAD_INDEX: int = 4

# Load a single time the Hardware padmap, in case the merger used the wrong padmap. 
# MergerPadMap = MergerPadData()
# if MergerPadData.is_merger_padmap_loaded():
#     right_pad_id = MergerPadData.load_merger_padmap()
# else: right_pad_id = None

@dataclass
class HardwareID:
    """Dataclass for AT-TPC pad hardware information

    Attributes
    ----------
    pad_id: int
        The pad id number
    cobo_id: int
        The CoBo id number
    asad_id: int
        The AsAd id number
    aget_id: int
        The AGET id number
    aget_channel: int
        The AGET channel number

    Methods
    -------
    __str__() -> str
        Convert the HardwareID to a string

    """

    pad_id: int = INVALID_PAD_ID
    cobo_id: int = INVALID_PAD_ID
    asad_id: int = INVALID_PAD_ID
    aget_id: int = INVALID_PAD_ID
    aget_channel: int = INVALID_PAD_ID

    def __str__(self) -> str:
        """Convert the HardwareID to a string

        Returns
        -------
        str
            The HardwareID string
        """
        return f"HardwareID -> pad: {self.pad_id} cobo: {self.cobo_id} asad: {self.asad_id} aget: {self.aget_id} channel: {self.aget_channel}"


def hardware_id_from_array(array: np.ndarray) -> HardwareID:
    """Convert an array of id numbers to a HardwareID

    Typically used with the raw hdf5 data from the AT-TPC merger.

    Parameters
    ----------
    array: ndarray
        An array of hardware id's in the appropriate order

    Returns
    -------
    HardwareID
        The HardwareID object
    """
    hw_id = HardwareID()
    hw_id.cobo_id = int(array[GET_DATA_COBO_INDEX])
    hw_id.asad_id = int(array[GET_DATA_ASAD_INDEX])
    hw_id.aget_id = int(array[GET_DATA_AGET_INDEX])
    hw_id.aget_channel = int(array[GET_DATA_CHANNEL_INDEX])
    if RIGHT_PAD_ID is not None: 
        hw_id.pad_id = RIGHT_PAD_ID[hw_id.cobo_id,hw_id.asad_id,hw_id.aget_id,hw_id.aget_channel]
    else: 
        hw_id.pad_id = int((array[GET_DATA_PAD_INDEX]))
    return hw_id


def generate_electronics_id(hardware: HardwareID) -> int:
    """Get a UUID for a given HardwareID

    Parameters
    ----------
    hardware: HardwareID

    Returns
    -------
    int
        a single value UUID

    """
    return (
        hardware.aget_channel
        + hardware.aget_id * 100
        + hardware.asad_id * 10000
        + hardware.cobo_id * 1000000
    )
