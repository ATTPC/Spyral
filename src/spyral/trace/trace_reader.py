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
    """An Error produced by the TraceReader"""

    pass


class TraceVersion(Enum):
    """Enum indicating what version the traces are formatted to

    Attributes
    ----------
    MERGER_LEGACY: int
        The original merger format contaning a meta dataset
    MERGER_1_0: int
        The 1.0 output libattpc_merger format
    HARMONIZER_0_1: int
        The 0.1.0 harmonizer output
    INVALID: int
        An illegal format
    """

    MERGER_LEGACY = 1
    MERGER_1_0 = 2
    HARMONIZER_0_1 = 3
    INVALID = -1


def version_string_to_enum(version: str) -> TraceVersion:
    """Converts a version string to a TraceVersion

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


class TraceReader:
    """Opens a trace file and reads the traces out

    As the attpc_merger has evolved, the format of the merged
    file has evolved as well. To handle these multiple cases, rather
    than making a specialized phase for each, we make a reader
    which will detect the version and invoke the correct method for reading.

    If more changes to formats are made this may be changed to a Protocol,
    with specific implementations handling the details. For now this is fine,
    but it could quickly get too cluttered.

    Attributes
    ----------
    file_path: Path
        Path to the trace file
    file: h5py.File
        The HDF5 file handle
    version: TraceVersion
        The trace format version
    run_number: int
        The current run number
    min_event: int
        The first event number
    max_event: int
        The last event number

    Parameters
    ----------
    path: Path
        The path to the trace file
    run_number: int
        The trace run number

    Methods
    -------
    event_range()
        Get the event range as an iterator for use in a loop
    read_event(event_id, get_params, frib_params, rng)
        Read a specific event
    read_scalers()
        Read the scalers
    should_have_scalers()
        See if the trace file should have scalers
    """

    def __init__(self, path: Path, run_number: int):
        self.file_path = path
        self.file = h5.File(self.file_path, "r")
        self.version = TraceVersion.INVALID
        self.run_number: int = run_number
        self.min_event: int = -1
        self.max_event: int = -1
        self.detect_version_and_init()

    def detect_version_and_init(self):
        """Detect what trace version this is

        Also setup the event range

        """
        if "meta" in self.file.keys() and "frib" in self.file.keys():
            self.version = TraceVersion.MERGER_LEGACY
            self.init_merger_legacy()
        elif "events" in self.file.keys():
            self.version = version_string_to_enum(self.file["events"].attrs["version"])  # type: ignore
            self.init_merger_modern()
        else:
            self.version = TraceVersion.INVALID

    def init_merger_legacy(self):
        """Init the reader for legacy merger format data"""
        meta_group = self.file["meta"]
        meta_data = meta_group["meta"]  # type: ignore
        self.min_event = int(meta_data[0])  # type: ignore
        self.max_event = int(meta_data[2])  # type: ignore

    def init_merger_modern(self):
        """Init the reader for modern merger format data"""
        self.min_event = int(self.file["events"].attrs["min_event"])  # type: ignore
        self.max_event = int(self.file["events"].attrs["max_event"])  # type: ignore

    def event_range(self) -> range:
        """Get the event range as an iterator for use in a loop

        Returns
        -------
        range
            The event range as a Python range (inclusive)
        """
        return range(self.min_event, self.max_event + 1)

    def read_event(
        self,
        event_id: int,
        get_params: GetParameters,
        frib_params: FribParameters,
        rng: Generator,
    ) -> Event:
        """Read a specific event

        Parameters
        ----------
        event_id: int
            The event to be read
        get_params: GetParameters
            Parameters controlling the GET trace analysis
        frib_params: FribParameters
            Parameters controlling the FRIBDAQ trace analysis
        rng: numpy.random.Generator
            numpy random number generator

        Returns
        -------
        Event
            The event data
        """
        match self.version:
            case TraceVersion.MERGER_LEGACY:
                return self.read_event_merger_legacy(
                    event_id, get_params, frib_params, rng
                )
            case TraceVersion.MERGER_1_0:
                return self.read_event_merger_1_0(
                    event_id, get_params, frib_params, rng
                )
            case TraceVersion.HARMONIZER_0_1:
                return self.read_event_harmonizer_0_1(
                    event_id, get_params, frib_params, rng
                )
            case _:
                raise TraceReaderError(
                    f"Cannot read event {event_id} from trace file {self.file_path}, merger version not supported!"
                )

    def read_event_merger_legacy(
        self,
        event_id: int,
        get_params: GetParameters,
        frib_params: FribParameters,
        rng: Generator,
    ) -> Event:
        """Read a specific event for trace version MERGER_LEGACY

        Parameters
        ----------
        event_id: int
            The event to be read
        get_params: GetParameters
            Parameters controlling the GET trace analysis
        frib_params: FribParameters
            Parameters controlling the FRIBDAQ trace analysis
        rng: numpy.random.Generator
            numpy random number generator

        Returns
        -------
        Event
            The event data
        """
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

    def read_event_merger_1_0(
        self,
        event_id: int,
        get_params: GetParameters,
        frib_params: FribParameters,
        rng: Generator,
    ) -> Event:
        """Read a specific event for trace version MERGER_1_0

        Parameters
        ----------
        event_id: int
            The event to be read
        get_params: GetParameters
            Parameters controlling the GET trace analysis
        frib_params: FribParameters
            Parameters controlling the FRIBDAQ trace analysis
        rng: numpy.random.Generator
            numpy random number generator

        Returns
        -------
        Event
            The event data
        """
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

    def read_event_harmonizer_0_1(
        self,
        event_id: int,
        get_params: GetParameters,
        frib_params: FribParameters,
        rng: Generator,
    ) -> Event:
        """Read a specific event for trace version HARMONIZER_0_1

        Parameters
        ----------
        event_id: int
            The event to be read
        get_params: GetParameters
            Parameters controlling the GET trace analysis
        frib_params: FribParameters
            Parameters controlling the FRIBDAQ trace analysis
        rng: numpy.random.Generator
            numpy random number generator

        Returns
        -------
        Event
            The event data
        """
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

    def read_scalers(self) -> FribScalers | None:
        """Read the scalers

        Returns
        -------
        FribScalers | None
            If no scalers are present, returns None. Otherwise returns FribScalers
        """
        match self.version:
            case TraceVersion.MERGER_LEGACY:
                return self.read_scalers_merger_legacy()
            case TraceVersion.MERGER_1_0:
                return self.read_scalers_merger_1_0()
            case TraceVersion.HARMONIZER_0_1:
                return None
            case _:
                raise TraceReaderError(
                    f"Cannot read scalers from trace file {self.file_path}, merger version not supported!"
                )

    def read_scalers_merger_1_0(self) -> FribScalers | None:
        """Read the scalers for trace version MERGER_1_0

        Returns
        -------
        FribScalers | None
            If no scalers are present, returns None. Otherwise returns FribScalers
        """
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

    def read_scalers_merger_legacy(self) -> FribScalers | None:
        """Read the scalers for trace version MERGER_LEGACY

        Returns
        -------
        FribScalers | None
            If no scalers are present, returns None. Otherwise returns FribScalers
        """
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

    def should_have_scalers(self) -> bool:
        """Returns True if the file should have scalers

        Returns
        -------
        bool
            True if the file should have scalers (i.e. not harmonizer)
        """

        return self.version != TraceVersion.HARMONIZER_0_1
