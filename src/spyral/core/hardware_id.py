from dataclasses import dataclass
from .constants import INVALID_PAD_ID
import polars as pl 
from pathlib import Path
import numpy as np

GET_DATA_COBO_INDEX: int = 0
GET_DATA_ASAD_INDEX: int = 1
GET_DATA_AGET_INDEX: int = 2
GET_DATA_CHANNEL_INDEX: int = 3
GET_DATA_PAD_INDEX: int = 4


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
    hw_id.pad_id = int((array[GET_DATA_PAD_INDEX]))
    hw_id.cobo_id = int(array[GET_DATA_COBO_INDEX])
    hw_id.asad_id = int(array[GET_DATA_ASAD_INDEX])
    hw_id.aget_id = int(array[GET_DATA_AGET_INDEX])
    hw_id.aget_channel = int(array[GET_DATA_CHANNEL_INDEX])
    return hw_id

def validation_merged_padmap(array: np.ndarray, padmap_path: Path) -> bool:
    """ Validates the padmap used in the merging corresponds to the experiment. 
    This function aids the user to identify if their h5file matches with their experiment map.
    
    Parameters 
    ----------
    array: ndarray
        An array of hardware id's in the appropriate order with dimensions (Ntraces,5)
    padmap_path: Path 
        Padmap address with the information of the PAD properties (Cobo, Asad, Aget, Channel IDs)

    Returns
    -------
    bool
        True is the h5file was merged with the right path.

    """
    # Create a dataframe with the harward matrix 
    # │ CoboID ┆ AsadID ┆ AgetID ┆ ChannelID ┆ PadID 
    hw_id_frame = {"CoboID": array[:,0], "AsadID": array[:,1], "AgetID":array[:,2], "ChannelID":array[:,3], "PadID": array[:,4]}
    
    # Data Frame of the H5file (ordered by pad number)
    padconstructed = pl.DataFrame(hw_id_frame)
    padconstructed = padconstructed.sort("PadID")
    
    # Date Frame created with the input file  (ordered by pad number)
    padmap = pl.read_csv(padmap_path)
    padmap = padmap.sort("PadID")
    
    is_equal = padmap.equals(padconstructed) 
    
    return is_equal

def fix_array_with_padmap(array: np.ndarray, padmap_path: Path) -> np.ndarray:
    """
    Replaces the pad number with the HardwareID from PADMAP. 

    Parameters 
    ----------
    array: ndarray
        An array of hardware id's in the appropriate order with dimensions (Ntraces,517)
    padmap_path: Path 
        Padmap address with the information of the PAD properties (Cobo, Asad, Aget, Channel IDs)

    Returns
    -------
    new_array: ndarray
        Hardware id's array fixed with the appropriate pad id (Ntraces, 517).
    """
    # Date Frame created with the input file  (ordered by pad number)
    padmap = pl.read_csv(padmap_path)
    # Fix PadID from the merger 
    new_array = array.copy()
    for i in range(len(array)):
        map_copied = padmap.filter((pl.col("CoboID")==array[i,0]) & (pl.col("AsadID")==array[i,1]) & (pl.col("AgetID")==array[i,2]) & (pl.col("ChannelID")==array[i,3]))
        if len(map_copied) > 0:
            new_array[i,4] = int(map_copied["PadID"].item(0))
    return new_array


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
