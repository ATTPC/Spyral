from .pad_map import PadMap
from .constants import INVALID_EVENT_NUMBER
from ..correction import ElectronCorrector
from ..trace.get_event import GetEvent
from ..trace.get_legacy_event import GetLegacyEvent
from .spy_log import spyral_warn

import numpy as np


class PointCloud:
    """Representation of a AT-TPC event

    A PointCloud is a geometric representation of an event in the AT-TPC
    The GET traces are converted into points in space within the AT-TPC

    Attributes
    ----------
    event_number: int
        The event number
    cloud: ndarray
        The Nx8 array of points in AT-TPC space
        Each row is [x,y,z,amplitude,integral,pad id,time,scale]

    Methods
    -------
    PointCloud()
        Create an empty point cloud
    load_cloud_from_get_event(event: GetEvent, pmap: PadMap, corrector: ElectronCorrector)
        Load a point cloud from a GetEvent
    load_cloud_from_hdf5_data(data: ndarray, event_number: int)
        Load a point cloud from an hdf5 file dataset
    is_valid() -> bool
        Check if the point cloud is valid
    retrieve_spatial_coordinates() -> ndarray
        Get the positional data from the point cloud
    calibrate_z_position(micromegas_tb: float, window_tb: float, detector_length: float, ic_correction: float = 0.0)
        Calibrate the cloud z-position from the micromegas and window time references
    remove_illegal_points(detector_length: float)
        Remove any points which lie outside the legal detector bounds in z
    sort_in_z()
        Sort the internal point cloud array by z-position
    """

    def __init__(self):
        self.event_number: int = INVALID_EVENT_NUMBER
        self.cloud: np.ndarray = np.empty(0, dtype=np.float64)

    def load_cloud_from_get_event(
        self,
        event: GetEvent | GetLegacyEvent,
        pmap: PadMap,
    ):
        """Load a point cloud from a GetEvent

        Loads the points from the signals in the traces and applies
        the pad relative gain correction and the pad time correction

        Parameters
        ----------
        event: GetEvent
            The GetEvent whose data should be loaded
        pmap: PadMap
            The PadMap used to get pad correction values
        """
        self.event_number = event.number
        count = 0
        for trace in event.traces:
            count += trace.get_number_of_peaks()
        self.cloud = np.zeros((count, 8))
        idx = 0
        for trace in event.traces:
            if trace.get_number_of_peaks() == 0 or trace.get_number_of_peaks() > 5:
                continue

            pid = trace.hw_id.pad_id
            check = pmap.get_pad_from_hardware(trace.hw_id)
            if check is None:
                spyral_warn(
                    __name__,
                    f"When checking pad number of hardware: {trace.hw_id}, recieved None!",
                )
                continue
            if (
                check != pid
            ):  # This is dangerous! We trust the pad map over the merged data!
                pid = check

            pad = pmap.get_pad_data(check)
            if pad is None or pmap.is_beam_pad(check):
                continue
            for peak in trace.get_peaks():
                self.cloud[idx, 0] = pad.x  # X-coordinate, geometry
                self.cloud[idx, 1] = pad.y  # Y-coordinate, geometry
                self.cloud[idx, 2] = (
                    peak.centroid + pad.time_offset
                )  # Z-coordinate, time with correction until calibrated with calibrate_z_position()
                self.cloud[idx, 3] = peak.amplitude
                self.cloud[idx, 4] = peak.integral
                self.cloud[idx, 5] = trace.hw_id.pad_id
                self.cloud[idx, 6] = (
                    peak.centroid + pad.time_offset
                )  # Time bucket with correction
                self.cloud[idx, 7] = pad.scale
                idx += 1
        self.cloud = self.cloud[self.cloud[:, 3] != 0.0]

    def load_cloud_from_hdf5_data(self, data: np.ndarray, event_number: int):
        """Load a point cloud from an hdf5 file dataset

        Parameters
        ----------
        data: ndarray
            This should be a copy of the point cloud data from the hdf5 file
        event_number: int
            The event number
        """
        self.event_number: int = event_number
        self.cloud = data

    def is_valid(self) -> bool:
        """Check if the PointCloud is valid

        Returns
        -------
        bool
            True if the PointCloud is valid
        """
        return self.event_number != INVALID_EVENT_NUMBER

    def retrieve_spatial_coordinates(self) -> np.ndarray:
        """Get only the spatial data from the point cloud


        Returns
        -------
        ndarray
            An Nx3 array of the spatial data of the PointCloud
        """
        return self.cloud[:, 0:3]

    def calibrate_z_position(
        self,
        micromegas_tb: float,
        window_tb: float,
        detector_length: float,
        efield_correction: ElectronCorrector | None = None,
        ic_correction: float = 0.0,
    ):
        """Calibrate the cloud z-position from the micromegas and window time references

        Also applies the ion chamber time correction and electric field correction if given
        Trims any points beyond the bounds of the detector (0 to detector length)

        Parameters
        ----------
        micromegas_tb: float
            The micromegas time reference in GET Time Buckets
        window_tb: float
            The window time reference in GET Time Buckets
        detector_length: float
            The detector length in mm
        efield_correction: ElectronCorrector | None
            The optional Garfield electric field correction to the electron drift
        ic_correction: float
            The ion chamber time correction in GET Time Buckets
        """
        # Maybe use mm as the reference because it is more stable?
        for idx, point in enumerate(self.cloud):
            self.cloud[idx][2] = (
                (window_tb - (point[6] - ic_correction))
                / (window_tb - micromegas_tb)
                * detector_length
            )
            if efield_correction is not None:
                self.cloud[idx] = efield_correction.correct_point(self.cloud[idx])

    def remove_illegal_points(self, detector_length: float = 1000.0):
        """Remove any points which lie outside the legal detector bounds in z

        Parameters
        ----------
        detector_length: float
            The length of the detector in the same units as the point cloud data
            (typically mm)

        """
        mask = np.logical_and(
            self.cloud[:, 2] < detector_length, self.cloud[:, 2] > 0.0
        )
        self.cloud = self.cloud[mask]

    def sort_in_z(self):
        """Sort the internal point cloud array by the z-coordinate"""
        indicies = np.argsort(self.cloud[:, 2])
        self.cloud = self.cloud[indicies]
