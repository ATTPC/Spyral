from .get_trace import GetTrace
import h5py
from pathlib import Path
from .constants import INVALID_EVENT_NAME, INVALID_EVENT_NUMBER
from .hardware_id import hardware_id_from_array


GET_DATA_TRACE_START: int = 5
GET_DATA_TRACE_STOP: int = 512+5

class GetEvent:

    def __init__(self, raw_data: h5py.Dataset, event_number: int):
        self.traces: list[GetTrace] = []
        self.name: str = INVALID_EVENT_NAME
        self.number: int = INVALID_EVENT_NUMBER
        self.load_traces(raw_data, event_number)

    def load_traces(self, raw_data: h5py.Dataset, event_number: int):
        self.name = raw_data.name
        self.number = event_number
        self.traces = convert_hdf5_get_event_to_traces(raw_data)

    def is_valid(self) -> bool:
        return self.name != INVALID_EVENT_NAME and self.number != INVALID_EVENT_NUMBER
    
def convert_hdf5_get_event_to_traces(get_event: h5py.Dataset) -> list[GetTrace]:
    return [GetTrace(row[GET_DATA_TRACE_START:GET_DATA_TRACE_STOP], hardware_id_from_array(row[0:5])) for row in get_event]

#GWM: test speed up between reading chunks of events vs. single event at a time

def read_get_event_chunk(filepath: Path, start_event: int, stop_event: int) -> list[GetEvent]:
    try:
        with h5py.File(filepath, 'r') as hfile:
            get_group = hfile['get']
            event_list: list[GetEvent] = []
            for evt in range(start_event, stop_event+1):
                evt_name = f'evt{evt}_data'
                event_list.append(GetEvent(get_group[evt_name], evt))
            return event_list
    except Exception as e:
        print(f'While reading file {filepath} for events {start_event} to {stop_event} recieved the following exception:')
        print(f'\t{type(e)}: {e}')
        print(f'No events will have been read.')
        return list()
    
def read_get_event(filepath: Path, event: int) -> GetEvent:
    try:
        with h5py.File(filepath, 'r') as hfile:
            get_group = hfile['get']
            event_name = f'evt{event}_data'
            return GetEvent(get_group[event_name], event)
    except Exception as e:
        print(f'While reading file {filepath} for event {event} recieved the following exception:')
        print(f'\t{type(e)}: {e}')
        print(f'No event will have been read.')
        return list()
