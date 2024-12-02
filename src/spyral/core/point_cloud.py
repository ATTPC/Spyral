from .pad_map import PadMap
from .config import DetectorParameters
from ..correction import ElectronCorrector
from ..trace.get_event import GetEvent
from .spy_log import spyral_warn
from dataclasses import dataclass
import numpy as np


@dataclass
class PointCloud:
    """Geometric representation of an AT-TPC event

    A PointCloud is a geometric representation of an event in the AT-TPC
    The GET traces are converted into points in space within the AT-TPC

    Overloads the length (len()) operator with the number of points in the cloud
    (number of rows in the data matrix)

    Attributes
    ----------
    event_number: int
        The event number
    cloud: numpy.ndarray
        The Nx8 array of points in AT-TPC space
        Each row is [x,y,z,amplitude,integral,pad id,time,scale]
    """

    event_number: int
    data: np.ndarray

    def __len__(self) -> int:
        """Number of points in the cloud

        AKA number of rows in the data matrix

        Returns
        -------
        int
            The number of points in the cloud
        """
        return len(self.data)


def point_cloud_from_get(event: GetEvent, pad_map: PadMap) -> PointCloud:
    """Load a point cloud from a GetEvent

    Loads the points from the signals in the traces and applies
    the pad relative gain correction and the pad time correction

    Parameters
    ----------
    event: GetEvent
        The GetEvent whose data should be loaded
    pad_map: PadMap
        The PadMap used to get pad correction values
    """
    count = 0
    for trace in event.traces:
        count += trace.get_number_of_peaks()
    cloud_matrix = np.zeros((count, 8))
    idx = 0
    for trace in event.traces:
        if trace.get_number_of_peaks() == 0 or trace.get_number_of_peaks() > 5:
            continue

        pid = trace.hw_id.pad_id
        check = pad_map.get_pad_from_hardware(trace.hw_id)
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

        pad = pad_map.get_pad_data(check)
        if pad is None or pad_map.is_beam_pad(check):
            continue
        for peak in trace.get_peaks():
            cloud_matrix[idx, 0] = pad.x  # X-coordinate, geometry
            cloud_matrix[idx, 1] = pad.y  # Y-coordinate, geometry
            cloud_matrix[idx, 2] = (
                peak.centroid + pad.time_offset
            )  # Z-coordinate, time with correction until calibrated with calibrate_z_position()
            cloud_matrix[idx, 3] = peak.amplitude
            cloud_matrix[idx, 4] = peak.integral
            cloud_matrix[idx, 5] = trace.hw_id.pad_id
            cloud_matrix[idx, 6] = (
                peak.centroid + pad.time_offset
            )  # Time bucket with correction
            cloud_matrix[idx, 7] = pad.scale
            idx += 1
    cloud_matrix = cloud_matrix[cloud_matrix[:, 3] != 0.0]
    return PointCloud(event.number, cloud_matrix)


def calibrate_point_cloud_z(
    cloud: PointCloud,
    detector_params: DetectorParameters,
    efield_correction: ElectronCorrector | None = None,
):
    """Calibrate the cloud z-position from the micromegas and window time references

    Also applies the ion chamber time correction and electric field correction if given
    Any points which were invalidated (NaN'ed) by this operation are removed.

    WARNING: This modifies the point cloud data, including removing points from the
    point cloud which were invalidated (NaN'ed).

    Parameters
    ----------
    cloud: PointCloud
        The point cloud to calibrate
    detector_params: DetectorParameters
        The detector parameters
    efield_correction: ElectronCorrector | None
        The optional Garfield electric field correction to the electron drift
    """
    # Maybe use mm as the reference because it is more stable?
    for idx, point in enumerate(cloud.data):
        cloud.data[idx][2] = (
            (detector_params.window_time_bucket - point[6])
            / (
                detector_params.window_time_bucket
                - detector_params.micromegas_time_bucket
            )
            * detector_params.detector_length
        )
        if efield_correction is not None:
            cloud.data[idx] = efield_correction.correct_point(cloud.data[idx])
    # Remove any invalid data after the calibration
    mask = np.any(np.isnan(cloud.data), axis=1)
    cloud.data = cloud.data[~mask]  # Invert the mask to reject the NaNs


def sort_point_cloud_in_z(cloud: PointCloud):
    """Sort the point cloud array by the z-coordinate

    Note: This modifies the underlying data in the point cloud

    Parameters
    ----------
    cloud: PointCloud
        The point cloud to sort
    """
    indicies = np.argsort(cloud.data[:, 2])
    cloud.data = cloud.data[indicies]
