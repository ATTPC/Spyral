from ..core.config import GetParameters, FribParameters
from .get_event import GetEvent
from .frib_event import FribEvent
from .frib_scalers import FribScalers

import h5py as h5
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from numpy.random import Generator


class TraceReaderError(Exception):
    pass


class TraceVersion(Enum):
    MERGER_010 = 1
    MERGER_CURRENT = 2
    INVALID = -1


@dataclass
class Event:
    id: int
    get: GetEvent | None
    frib: FribEvent | None


class TraceReader:
    def __init__(self, path: Path):
        self.file_path = path
        self.file = h5.File(self.file_path, "r")
        self.version = TraceVersion.INVALID
        self.min_event: int = -1
        self.max_event: int = -1
        self.detect_version_and_init()

    def detect_version_and_init(self):
        if "meta" in self.file.keys() and "frib" in self.file.keys():
            self.version = TraceVersion.MERGER_010
            self.init_merger_010()
        elif "events" in self.file.keys():
            self.version = TraceVersion.MERGER_CURRENT
            self.init_merger_current()
        else:
            self.version = TraceVersion.INVALID

    def init_merger_010(self):
        meta_group = self.file["meta"]
        meta_data = meta_group["meta"]  # type: ignore
        self.min_event = int(meta_data[0])  # type: ignore
        self.max_event = int(meta_data[2])  # type: ignore

    def init_merger_current(self):
        self.min_event = int(self.file["events"].attrs["min_event"])  # type: ignore
        self.max_event = int(self.file["events"].attrs["max_event"])  # type: ignore

    def event_range(self) -> range:
        return range(self.min_event, self.max_event + 1)

    def read_event(
        self,
        event_id: int,
        get_params: GetParameters,
        frib_params: FribParameters,
        rng: Generator,
    ) -> Event:
        match self.version:
            case TraceVersion.MERGER_010:
                return self.read_event_merger_010(
                    event_id, get_params, frib_params, rng
                )
            case TraceVersion.MERGER_CURRENT:
                return self.read_event_merger_current(
                    event_id, get_params, frib_params, rng
                )
            case _:
                raise TraceReaderError(
                    f"Cannot read event {event_id} from trace file {self.file_path}, merger version not supported!"
                )

    def read_event_merger_010(
        self,
        event_id: int,
        get_params: GetParameters,
        frib_params: FribParameters,
        rng: Generator,
    ) -> Event:
        get_group: h5.Group = self.file["get"]  # type: ignore
        frib_evt_group: h5.Group = self.file["frib"]["evt"]  # type: ignore
        event_name = f"evt{event_id}_data"
        frib_event_name = f"evt{event_id}_1903"

        event = Event(event_id, None, None)
        if event_name in get_group:
            get_data: h5.Dataset = get_group[event_name]  # type: ignore
            event.get = GetEvent(get_data[:], event_id, get_params, rng)
        if frib_event_name in frib_evt_group:
            frib_data: h5.Dataset = frib_evt_group[frib_event_name]  # type: ignore
            event.frib = FribEvent(frib_data[:], event_id, frib_params)
        return event

    def read_event_merger_current(
        self,
        event_id: int,
        get_params: GetParameters,
        frib_params: FribParameters,
        rng: Generator,
    ) -> Event:
        events_group: h5.Group = self.file["events"]  # type: ignore
        event_name = f"event_{event_id}"

        event = Event(event_id, None, None)
        if event_id in events_group:
            event_data: h5.Group = events_group[event_name]  # type: ignore
            get_data: h5.Dataset = event_data["event"]  # type: ignore
            event.get = GetEvent(get_data[:], event_id, get_params, rng)
            if "frib_physics" in event_data:
                frib_1903_data: h5.Dataset = events_group["frib_physics"]["1903"]  # type: ignore
                event.frib = FribEvent(frib_1903_data[:], event_id, frib_params)
        return event

    def read_scalers(self) -> FribScalers | None:
        match self.version:
            case TraceVersion.MERGER_010:
                return self.read_scalers_merger_010()
            case TraceVersion.MERGER_CURRENT:
                return self.read_scalers_merger_current()
            case _:
                raise TraceReaderError(
                    f"Cannot read scalers from trace file {self.file_path}, merger version not supported!"
                )

    def read_scalers_merger_current(self) -> FribScalers | None:
        if "scalers" not in self.file:
            return None
        scaler_group: h5.Group = self.file["scalers"]  # type: ignore
        scaler_min = int(scaler_group.attrs["min_event"])  # type: ignore
        scaler_max = int(scaler_group.attrs["min_event"])  # type: ignore
        scalers = FribScalers()

        for event in range(scaler_min, scaler_max + 1):
            event_name = f"event_{event}"
            if event_name in scaler_group:
                event_data: h5.Dataset = scaler_group[event_name]  # type: ignore
                scalers.load_scalers(event, event_data)
        return scalers

    def read_scalers_merger_010(self) -> FribScalers | None:
        frib_group: h5.Group = self.file["frib"]  # type: ignore
        scalers = FribScalers()
        if "scalers" not in frib_group:
            return None
        scaler_group: h5.Group = frib_group["scalers"]  # type: ignore

        # We don't store any metadata about the number of scaler events,
        # so we have to scan to failure. This is annoying and slow, and
        # should be patched in the next update of the merger
        event = 0
        while True:
            scaler_data: h5.Dataset
            # Attempt to extract the next scaler
            try:
                scaler_data = scaler_group[f"scaler{event}_data"]  # type: ignore
            except Exception:
                break

            scalers.load_scalers(event, scaler_data)
            event += 1

        return scalers
