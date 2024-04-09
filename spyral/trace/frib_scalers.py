import h5py as h5
import polars as pl
from pathlib import Path

CLOCK_FREE_INDEX: int = 0
CLOCK_LIVE_INDEX: int = 1
TRIGGER_FREE_INDEX: int = 2
TRIGGER_LIVE_INDEX: int = 3
IC_SCA_INDEX: int = 4
MESH_SCA_INDEX: int = 5
SI1_CFD_INDEX: int = 6
SI2_CFD_INDEX: int = 7
SIPM_INDEX: int = 8
IC_DS_INDEX: int = 9
IC_CFD_INDEX: int = 10


class FribScalers:
    """Dataclass representing FRIBDAQ scalers

    Scalers are incremental counters typically used to monitor rates in the dataset. The default available
    scalers are:

    clock_free: The time ellapsed while running the data acqusition
    clock_live: The amount of time for which the acquisition is "live" (able to accept triggers)
    trigger_free: The total number of trigger signals recieved by the acquisition
    trigger_live: The total number of triggers which acutally cause events in the acquisition
    ic_sca: The total number of ion chamber signals recieved by the acquisition
    mesh_sca: The total number of mesh signals recieved by the acquisition
    si1_cfd: The total number of Si detector 1 signals recieved by the acquisition
    si2_cfd: The total number of Si detector 2 signals recieved by the acquisition
    sipm: Unclear
    ic_ds: The downscaled rate into the ion chamber
    ic_cfd: Unclear

    AT-TPC scalers are set as incremental. That is, one must sum the values over all scaler events to get the total
    scaler value for a given run.

    Attributes
    ----------
    scalers: dict[str, list[int]]
        The collection of scaler values in a dictionary. Each scaler is keyed by it's name (matching the
        FRIBDAQ name), and corresponds to a list of integer values. An additional column of event numbers is
        stored for cross-referencing. Note that the scaler event number is *not* the same as the GET/FRIB event number.

    scaler_map: dict[str, int]
        A dictionary mapping the scaler name to the index where it is stored in a dataset

    Methods
    -------
    load_scalers(event_number: int, dataset: h5py.Dataset) -> None:
        Load the scalers for an event in the dataset
    write_scalers(scaler_path: Path) -> None:
        Write the scalers to disk
    """

    def __init__(self):
        self.scalers: dict[str, list[int]] = {
            "event": [],
            "clock_free": [],
            "clock_live": [],
            "trigger_free": [],
            "trigger_live": [],
            "ic_sca": [],
            "mesh_sca": [],
            "si1_cfd": [],
            "si2_cfd": [],
            "sipm": [],
            "ic_ds": [],
            "ic_cfd": [],
        }

        self.scaler_map: dict[str, int] = {
            "clock_free": CLOCK_FREE_INDEX,
            "clock_live": CLOCK_LIVE_INDEX,
            "trigger_free": TRIGGER_FREE_INDEX,
            "trigger_live": TRIGGER_LIVE_INDEX,
            "ic_sca": IC_SCA_INDEX,
            "mesh_sca": MESH_SCA_INDEX,
            "si1_cfd": SI1_CFD_INDEX,
            "si2_cfd": SI2_CFD_INDEX,
            "sipm": SIPM_INDEX,
            "ic_ds": IC_DS_INDEX,
            "ic_cfd": IC_CFD_INDEX,
        }

    def load_scalers(self, event_number: int, dataset: h5.Dataset) -> None:
        """Load the scalers for an event in the HDF5 dataset

        Parameters
        ----------
        event_number: int
            The event number
        dataset: h5py.Dataset
            The raw scaler data
        """
        self.scalers["event"].append(event_number)
        for key in self.scaler_map.keys():
            # Explicitly cast to int, just in case
            self.scalers[key].append(int(dataset[self.scaler_map[key]]))

    def write_scalers(self, scaler_path: Path) -> None:
        """Write the scalers to a dataframe file

        Parameters
        ----------
        scaler_path: Path
            Path to which the scalers should be written

        """
        df = pl.DataFrame(self.scalers)
        df.write_parquet(scaler_path)


def process_scalers(scaler_group: h5.Group, scaler_path: Path) -> None:
    """Extract the scaler data for a run and write it to disk

    Parameters
    ----------
    scaler_group: h5py.Group
        The group containing scaler data
    scaler_path: Path
        The path to which scaler data should be written

    """

    scalers = FribScalers()

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

    scalers.write_scalers(scaler_path)
