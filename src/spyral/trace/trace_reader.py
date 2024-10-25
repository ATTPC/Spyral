from ..core.config import GetParameters, FribParameters
from .get_event import GetEvent
from .frib_event import FribEvent
from .frib_scalers import FribScalers
from ..core.spy_log import spyral_error

import h5py as h5
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from numpy.random import Generator
from typing import Protocol, Iterable
import numpy as np


class TraceVersion(Enum):
    """Enum indicating what version the traces are formatted to

    This is used to handle format versions from different
    Trace producing tools (attpc_merger, harmonizer)
    Much easier to programmatically handle version enums
    over version strings. Probably a little over-optimizing
    at the moment, but if more versions get made, this should
    allow us to be more flexible.

    Attributes
    ----------
    MERGER_1_0: int
        The 1.0 output libattpc_merger format
    HARMONIZER_0_1: int
        The 0.1.0 harmonizer output
    INVALID: int
        An illegal format
    """

    MERGER_1_0 = 2
    HARMONIZER_0_1 = 3
    INVALID = -1


def version_string_to_enum(version: str) -> TraceVersion:
    """Converts a version string to a TraceVersion

    Much easier to programmatically handle version enums
    over version strings. Probably a little over-optimizing
    at the moment, but if more versions get made, this should
    allow us to be more flexible.

    Parameters
    ----------
    version: str
        The version string

    Returns
    -------
    TraceVersion:
        The version enum
    """

    if version == "libattpc_merger:1.0":
        return TraceVersion.MERGER_1_0
    elif version == "harmonizer:0.1.0":
        return TraceVersion.HARMONIZER_0_1
    else:
        return TraceVersion.INVALID


@dataclass
class Event:
    """Simple dataclass wrapping the event data

    Attributes
    ----------
    id: int
        The event id number
    get: GetEvent | None
        Optional GET trace data
    frib: FribEvent | None
        Optional FRIBDAQ trace data
    original_run: int
        The original run number. In the case of merger data
        this is the same as the current run. In the case of harmonizer
        data this is the run number before harmonization.
    original_event: int
        The original event number. In the case of merger data
        this is the same as the current event. In the case of harmonizer
        data this is the event number before harmonization.
    """

    id: int
    get: GetEvent | None
    frib: FribEvent | None
    original_run: int
    original_event: int


class TraceReader(Protocol):
    """Protocol defining a reader for trace data

    As the attpc_merger and its friends have evolved, the format of
    trace files has evolved as well. We define this Protocol, which will
    be implemented for each supported trace format.

    As this is a protocol, do not instantiate it directly.

    Parameters
    ----------
    file: h5py.File
        The hdf5 file to read traces from
    run_number: int
        The trace run number

    Methods
    -------
    event_range()
        Get the event range as an iterator for use in a loop
    __len()___
        The length of the file in events
    read_event(event_id, get_params, frib_params, rng)
        Read a specific event
    read_raw_get_event(event_id)
        Read a specific GET event and return the raw data array
    read_raw_frib_event(event_id)
        Read a specific FRIB event and return the raw data array
    read_scalers()
        Read the scalers, if they exist
    """

    def __init__(self, file: h5.File, run_number: int): ...

    def event_range(self) -> Iterable[int]:
        """Get the event range as an iterator for use in a loop

        Returns
        -------
        Iterable[int]
            A range of event numbers to iterate over
        """
        ...

    def __len__(self) -> int:
        """The length of the file in events

        Returns
        -------
        int
            The number of events in the file
        """
        ...

    def first_event(self) -> int:
        """The first event number

        Returns
        -------
        int
            The first event number
        """
        ...

    def last_event(self) -> int:
        """The last event number

        Returns
        -------
        int
            The last event number
        """
        ...

    def read_event(
        self,
        event_id: int,
        get_params: GetParameters,
        frib_params: FribParameters,
        rng: Generator,
    ) -> Event:
        """Read a specific event

        Read all of the data for a specific AT-TPC event and collect it into
        a unified structure (Event).

        Parameters
        ----------
        event_id: int
            The event to read
        get_params: GetParameters
            The GET electronics signal analysis parameters
        frib_params: FribParameters
            The FRIBDAQ signal analysis parameters
        rng: numpy.random.Generator
            A random number generator

        Returns
        -------
        Event
            A unified AT-TPC event structure
        """
        ...

    def read_raw_get_event(self, event_id: int) -> np.ndarray | None:
        """Read a specific GET event and return the raw data array

        Read the raw data from a specific GET event and returns the underlying
        data. In general, the data matrix format is preserved between different
        overarching file formats, but this cannot be guaranteed, so use with
        caution.

        Main intent is for use with exploratory notebooks.

        Parameters
        ----------
        event_id: int
            The event to read

        Returns
        -------
        numpy.ndarray | None
            The raw GET data matrix, or None if it does not exist
        """
        ...

    def read_raw_frib_event(self, event_id: int) -> np.ndarray | None:
        """Read a specific FRIB event and return the raw data array

        Read the raw data from a specific FRIBDAQ event and returns the underlying
        data. In general, the data matrix format is preserved between different
        overarching file formats, but this cannot be guaranteed, so use with
        caution.

        Main intent is for use with exploratory notebooks.

        Parameters
        ----------
        event_id: int
            The event to read

        Returns
        -------
        numpy.ndarray | None
            The raw FRIB data matrix, or None if that FRIB event does not exist
        """
        ...

    def read_scalers(self) -> FribScalers | None:
        """Read the scalers, if they exist

        Read the scaler data into an FribScaler container.
        Some formats do not contain scalers at all (i.e. harmonizer),
        so this may return None.

        Returns
        -------
        FribScalers | None
            The scaler container or None if scalers do not exist
        """
        ...


class MergerLegacyReader:
    """A TraceReader for legacy (un-versioned) attpc_merger data

    Attributes
    ----------
    file: h5py.File
        The hdf5 file
    run_number: int
        The run_number
    min_event: int
        The first event number
    max_event: int
        The last event number
    """

    def __init__(self, file: h5.File, run_number: int):
        self.file = file
        self.run_number = run_number
        meta_group = self.file["meta"]
        meta_data = meta_group["meta"]  # type: ignore
        self.min_event = int(meta_data[0])  # type: ignore
        self.max_event = int(meta_data[2])  # type: ignore

    def event_range(self) -> range:
        return range(self.min_event, self.max_event + 1)

    def __len__(self) -> int:
        return self.max_event - self.min_event + 1

    def first_event(self) -> int:
        return self.min_event

    def last_event(self) -> int:
        return self.max_event

    def read_event(
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
        frib_coinc_name = f"evt{event_id}_977"

        event = Event(event_id, None, None, self.run_number, event_id)
        if event_name in get_group:
            get_data: h5.Dataset = get_group[event_name]  # type: ignore
            event.get = GetEvent(get_data[:], event_id, get_params, rng)
        if frib_event_name in frib_evt_group and frib_coinc_name in frib_evt_group:
            frib_data: h5.Dataset = frib_evt_group[frib_event_name]  # type: ignore
            frib_coinc: h5.Dataset = frib_evt_group[frib_coinc_name]  # type: ignore
            event.frib = FribEvent(frib_data[:], frib_coinc[:], event_id, frib_params)
        return event

    def read_raw_get_event(self, event_id: int) -> np.ndarray | None:
        get_group: h5.Group = self.file["get"]  # type: ignore
        event_name = f"evt{event_id}_data"
        if event_name in get_group:
            return get_group[event_name][:].copy()  # type: ignore

    def read_raw_frib_event(self, event_id: int) -> np.ndarray | None:
        frib_evt_group: h5.Group = self.file["frib"]["evt"]  # type: ignore
        frib_event_name = f"evt{event_id}_1903"
        if frib_event_name in frib_evt_group:
            return frib_evt_group[frib_event_name][:].copy()  # type: ignore

    def read_scalers(self) -> FribScalers | None:
        frib_group: h5.Group = self.file["frib"]  # type: ignore
        scalers = FribScalers()
        if "scaler" not in frib_group:
            return None
        scaler_group: h5.Group = frib_group["scaler"]  # type: ignore

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


class MergerCurrentReader:
    """A TraceReader for the current (1.0) libattpc_merger data

    Attributes
    ----------
    file: h5py.File
        The hdf5 file
    run_number: int
        The run_number
    version: str
        The version string
    min_event: int
        The first event number
    max_event: int
        The last event number
    """

    def __init__(self, file: h5.File, run_number: int):
        self.file = file
        self.run_number: int = run_number
        self.version: str = self.file["events"].attrs["version"]  # type: ignore
        self.min_event = int(self.file["events"].attrs["min_event"])  # type: ignore
        self.max_event = int(self.file["events"].attrs["max_event"])  # type: ignore

    def event_range(self) -> range:
        return range(self.min_event, self.max_event)

    def __len__(self) -> int:
        return self.max_event - self.min_event + 1

    def first_event(self) -> int:
        return self.min_event

    def last_event(self) -> int:
        return self.max_event

    def read_event(
        self,
        event_id: int,
        get_params: GetParameters,
        frib_params: FribParameters,
        rng: Generator,
    ) -> Event:
        events_group: h5.Group = self.file["events"]  # type: ignore
        event_name = f"event_{event_id}"

        event = Event(event_id, None, None, self.run_number, event_id)
        if event_name in events_group:
            event_data: h5.Group = events_group[event_name]  # type: ignore
            get_data: h5.Dataset = event_data["get_traces"]  # type: ignore
            event.get = GetEvent(get_data[:], event_id, get_params, rng)
            if "frib_physics" in event_data:
                frib_1903_data: h5.Dataset = events_group["frib_physics"]["1903"]  # type: ignore
                frib_977_data: h5.Dataset = events_group["frib_physics"]["977"]  # type: ignore
                event.frib = FribEvent(
                    frib_1903_data[:], frib_977_data[:], event_id, frib_params
                )
        return event

    def read_raw_get_event(self, event_id: int) -> np.ndarray | None:
        events_group: h5.Group = self.file["events"]  # type: ignore
        event_name = f"event_{event_id}"
        if event_name in events_group:
            event_data: h5.Group = events_group[event_name]  # type: ignore
            return event_data["get_traces"][:].copy()  # type: ignore

    def read_raw_frib_event(self, event_id: int) -> np.ndarray | None:
        events_group: h5.Group = self.file["events"]  # type: ignore
        event_name = f"event_{event_id}"
        if event_name in events_group:
            event_data: h5.Group = events_group[event_name]  # type: ignore
            if "frib_physics" in event_data:
                return events_group["frib_physics"]["1903"][:].copy()  # type: ignore

    def read_scalers(self) -> FribScalers | None:
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


class HarmonizerReader:
    """A TraceReader for the current (0.1) harmonizer data

    Attributes
    ----------
    file: h5py.File
        The hdf5 file
    run_number: int
        The run_number
    version: str
        The version string
    min_event: int
        The first event number
    max_event: int
        The last event number
    """

    def __init__(self, file: h5.File, run_number: int):
        self.file = file
        self.run_number: int = run_number
        self.version: str = self.file["events"].attrs["version"]  # type: ignore
        self.min_event = int(self.file["events"].attrs["min_event"])  # type: ignore
        self.max_event = int(self.file["events"].attrs["max_event"])  # type: ignore

    def event_range(self) -> range:
        return range(self.min_event, self.max_event)

    def __len__(self) -> int:
        return self.max_event - self.min_event + 1

    def first_event(self) -> int:
        return self.min_event

    def last_event(self) -> int:
        return self.max_event

    def read_event(
        self,
        event_id: int,
        get_params: GetParameters,
        frib_params: FribParameters,
        rng: Generator,
    ) -> Event:
        events_group: h5.Group = self.file["events"]  # type: ignore
        event_name = f"event_{event_id}"

        event = Event(event_id, None, None, -1, -1)
        if event_name in events_group:
            event_data: h5.Group = events_group[event_name]  # type: ignore
            event.original_event = event_data.attrs["orig_event"]  # type: ignore
            event.original_run = event_data.attrs["orig_run"]  # type: ignore
            get_data: h5.Dataset = event_data["get_traces"]  # type: ignore
            event.get = GetEvent(get_data[:], event_id, get_params, rng)
            if "frib_physics" in event_data:
                frib_1903_data: h5.Dataset = events_group["frib_physics"]["1903"]  # type: ignore
                frib_977_data: h5.Dataset = events_group["frib_physics"]["977"]  # type: ignore
                event.frib = FribEvent(
                    frib_1903_data[:], frib_977_data[:], event_id, frib_params
                )
        return event

    def read_raw_get_event(self, event_id: int) -> np.ndarray | None:
        events_group: h5.Group = self.file["events"]  # type: ignore
        event_name = f"event_{event_id}"
        if event_name in events_group:
            event_data: h5.Group = events_group[event_name]  # type: ignore
            return event_data["get_traces"][:].copy()  # type: ignore

    def read_raw_frib_event(self, event_id: int) -> np.ndarray | None:
        events_group: h5.Group = self.file["events"]  # type: ignore
        event_name = f"event_{event_id}"
        if event_name in events_group:
            event_data: h5.Group = events_group[event_name]  # type: ignore
            if "frib_physics" in event_data:
                return events_group["frib_physics"]["1903"][:].copy()  # type: ignore

    def read_scalers(self) -> FribScalers | None:
        return None


def create_reader(path: Path, run_number: int) -> TraceReader | None:
    """Create a TraceReader

    This function detects the appropriate implementation and instantiates it.

    Parameters
    ----------
    path: Path
        Path to the trace file
    run_number: int
        The run number of the trace file

    Returns
    -------
    TraceReader | None
        A specific implementation of the TraceReader protocol, or None
        if no appropriate implementation was found.
    """
    if not path.exists():
        spyral_error(__name__, f"The trace file {path} does not exist")
        return None

    file = h5.File(path, "r")
    if "meta" in file:
        return MergerLegacyReader(file, run_number)
    elif "events" in file:
        version_string: str = file["events"].attrs["version"]  # type: ignore
        version = version_string_to_enum(version_string)
        match version:
            case TraceVersion.MERGER_1_0:
                return MergerCurrentReader(file, run_number)
            case TraceVersion.HARMONIZER_0_1:
                return HarmonizerReader(file, run_number)
            case TraceVersion.INVALID:
                spyral_error(
                    __name__,
                    f"Traces at {path} have an unrecognized version string: {version_string}",
                )
                return None
    else:
        spyral_error(__name__, f"Traces at {path} do not match any known format")
        return None
