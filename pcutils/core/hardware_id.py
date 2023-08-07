from dataclasses import dataclass
from .constants import INVALID_PAD_ID
import numpy as np


GET_DATA_COBO_INDEX: int = 0
GET_DATA_ASAD_INDEX: int = 1
GET_DATA_AGET_INDEX: int = 2
GET_DATA_CHANNEL_INDEX: int = 3
GET_DATA_PAD_INDEX: int = 4

@dataclass
class HardwareID:
    pad_id: int = INVALID_PAD_ID
    cobo_id: int = INVALID_PAD_ID
    asad_id: int = INVALID_PAD_ID
    aget_id: int = INVALID_PAD_ID
    aget_channel: int = INVALID_PAD_ID


def hardware_id_from_array(array: np.ndarray) -> HardwareID:
    hw_id = HardwareID()
    hw_id.pad_id = array[GET_DATA_PAD_INDEX]
    hw_id.cobo_id = array[GET_DATA_COBO_INDEX]
    hw_id.asad_id = array[GET_DATA_ASAD_INDEX]
    hw_id.aget_id = array[GET_DATA_AGET_INDEX]
    hw_id.aget_channel = array[GET_DATA_CHANNEL_INDEX]
    return hw_id