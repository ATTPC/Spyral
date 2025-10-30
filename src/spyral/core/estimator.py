from .cluster import Cluster, Direction
from .config import DetectorParameters, EstimateParameters
from ..geometry.circle import generate_circle_points, least_squares_circle
from .spy_log import spyral_warn

import numpy as np
import math
from scipy.stats import linregress
from dataclasses import dataclass


@dataclass
class EstimateResult:
    """Container for results of estimation algorithm with metadata

    Attributes
    ----------
    event: int
        The event number
    cluster_index: int
        The cluster index
    cluster_label: int
        The label from the clustering algorithm
    orig_run: int
        The original run number
    orig_event: int
        The original event number
    ic_amplitude: float
        The ion chamber signal amplitude
    ic_centroid: float
        The ion chamber signal centroid
    ic_integral: float
        The ion chamber signal integral
    ic_multiplicity: float
        The ion chamber signal multiplicity
    vertex_x: float
        The vertex x position (mm)
    vertex_y: float
        The vertex y position (mm)
    vertex_z: float
        The vertex z position (mm)
    center_x: float
        The center x position (mm)
    center_y: float
        The center y position (mm)
    center_z: float
        The center z position (mm)
    polar: float
        The trajectory polar angle (radians)
    azimuthal: float
        The trajectory azimuthal angle (radians)
    brho: float
        The magnetic rigidity (Tm)
    dEdx: float
        The relative energy loss per mm (arb)
    sqrt_dEdx: float
        The square root of the relative energy loss per mm (arb)
    dE: float
        The relative energy loss
    arclength: float
        The pathlength of the first arc in mm
    direction: int
        The direction value
    """

    event: int
    cluster_index: int
    cluster_label: int
    orig_run: int
    orig_event: int
    ic_amplitude: float
    ic_centroid: float
    ic_integral: float
    ic_multiplicity: float
    vertex_x: float
    vertex_y: float
    vertex_z: float
    center_x: float
    center_y: float
    center_z: float
    polar: float
    azimuthal: float
    brho: float
    dEdx: float
    sqrt_dEdx: float
    dE: float
    arclength: float
    direction: int


def estimate_physics(
    cluster_index: int,
    cluster: Cluster,
    ic_amplitude: float,
    ic_centroid: float,
    ic_integral: float,
    ic_multiplicity: float,
    orig_run: int,
    orig_event: int,
    estimate_params: EstimateParameters,
    detector_params: DetectorParameters,
) -> EstimateResult | None:
    """Entry point for estimation

    This is the parent function for estimation. It handles checking that the data
    meets the conditions to be estimated, applying splines to data, and
    esuring that the estimation results pass a sanity check.

    Parameters
    ----------
    cluster_index: int
        The cluster index in the HDF5 file.
    cluster: Cluster
        The cluster to estimate
    ic_amplitude:
        The ion chamber amplitude for this cluster
    ic_centroid:
        The ion chamber centroid for this cluster
    ic_integral:
        The ion chamber integral for this cluster
    orig_run:
        The original event run number
    orig_event:
        The original event number
    detector_params:
        Configuration parameters for the physical detector properties

    Returns
    -------
    EstimationResult | None
        The resulting estimated data, or None if the algorithm failed
    """
    # Check if we have enough points to estimate
    if len(cluster.data) < estimate_params.min_total_trajectory_points:
        return None
    # Generate smoothing splines, these will give us better distance measures
    try:
        cluster.apply_smoothing_splines(estimate_params.smoothing_factor)
    except Exception as e:
        # Spline creation can fail for two main reasons:
        # - Not enough points in the data (need at least 5)
        # - The data is inherentily multivalued in z (a spark event where the pad plane lights up at one instance in time)
        # We do not analyze un-splineable events. But we do report a warning in the log file that these events failed
        spyral_warn(
            __name__,
            f"Spline creation failed for event {cluster.event} with error: {e}",
        )
        return None

    # Run estimation where we attempt to guess the right direction
    result = estimate_physics_pass(
        cluster_index,
        cluster,
        ic_amplitude,
        ic_centroid,
        ic_integral,
        ic_multiplicity,
        orig_run,
        orig_event,
        detector_params,
    )
    return result


def estimate_physics_pass(
    cluster_index: int,
    cluster: Cluster,
    ic_amplitude: float,
    ic_centroid: float,
    ic_integral: float,
    ic_multiplicity: float,
    orig_run: int,
    orig_event: int,
    detector_params: DetectorParameters,
) -> EstimateResult:
    """Estimate the physics parameters for a cluster which could represent a particle trajectory

    Estimation is an imprecise process (by definition), and as such this algorithm requires a lot of
    manipulation of the data.

    Parameters
    ----------
    cluster_index: int
        The cluster index in the HDF5 file.
    cluster: Cluster
        The cluster to estimate
    ic_amplitude: float
        The ion chamber amplitude for this cluster
    ic_centroid: float
        The ion chamber centroid for this cluster
    ic_integral: float
        The ion chamber integral for this cluster
    ic_multiplicity: float
        The ion chamber multiplicity for this cluster
    orig_run: int
        The original run number
    orig_event: int
        The original event number
    detector_params: DetectorParameters
        Configuration parameters for the physical detector properties
    chosen_direction: Direction, default=Direction.NONE
        Optional direction for the trajectory. Default
        estimates the direction.

    Returns
    -------
    EstimateResult | Direction
        Returns a the estimated parameters, or Direction if the algorithm failed.
        The Direction indicates which direction the algorithm attempted to use for
        estimation

    """

    direction = cluster.direction # We already know the direction from the clustering phase
    if direction == Direction.NONE: # We could not determine the direction of the track
        return None
    vertex = np.array([0.0, 0.0, 0.0])  # reaction vertex
    center = np.array([0.0, 0.0, 0.0])  # spiral center
    # copy the data so we can modify it without worrying about side-effects
    cluster_data = cluster.data.copy()

    # If chosen direction is set to NONE, we want to have the algorithm
    # try to decide which direction the trajectory is going
    # if direction == Direction.NONE:
    #     direction = choose_direction(cluster_data)

    if direction == Direction.BACKWARD:
        cluster_data = np.flip(cluster_data, axis=0)

    # Guess that the vertex is the first point; make sure to copy! not reference
    vertex[:] = cluster_data[0, :3]

    # Find the first point that is furthest from the vertex in rho (maximum) to get the first arc of the trajectory
    rho_to_vertex = np.linalg.norm(cluster_data[1:, :2] - vertex[:2], axis=1)
    maximum = np.argmax(rho_to_vertex)
    first_arc = cluster_data[: (maximum + 1)]

    # Fit a circle to the first arc and extract some physics
    center[0], center[1], radius, _ = least_squares_circle(
        first_arc[:, 0], first_arc[:, 1]
    )
    # radius = np.linalg.norm(cluster_data[0, :2] - center[:2])
    circle = generate_circle_points(center[0], center[1], radius)
    # Re-estimate vertex using the fitted circle. Extrapolate back to point closest to beam axis
    vertex_estimate_index = np.argsort(np.linalg.norm(circle, axis=1))[0]
    vertex[:2] = circle[vertex_estimate_index]
    # Re-calculate distance to vertex, maximum, first arc
    rho_to_vertex = np.linalg.norm((cluster_data[:, :2] - vertex[:2]), axis=1)
    maximum = np.argmax(rho_to_vertex)
    first_arc = cluster_data[: (maximum + 1)]

    # Do a linear fit to small segment of trajectory to extract rho vs. z and extrapolate vertex z
    test_index = max(10, int(maximum * 0.5))
    # test_index = 10
    fit = linregress(cluster_data[:test_index, 2], rho_to_vertex[:test_index])
    vertex_rho = np.linalg.norm(vertex[:2])
    # Since we fit to rho_to_vertex, just find intercept point
    # Check to see if slope is zero, as this can lead to NaN's
    if fit.slope == 0.0:  # type: ignore
        return None
    vertex[2] = -1.0 * fit.intercept / fit.slope  # type: ignore
    center[2] = vertex[2]

    # Toss tracks whose verticies are not close to the origin in x,y
    if vertex_rho > detector_params.beam_region_radius:
        return None

    polar = math.atan(fit.slope)  # type: ignore
    # We have a self consistency case here. Polar should match chosen Direction
    if (polar > 0.0 and direction == Direction.BACKWARD) or (polar < 0.0 and direction == Direction.FORWARD):
        return None # Our direction guess was bad, we need to try again with the other direction
    elif direction is Direction.BACKWARD:
        polar += math.pi

    # From the trigonometry of the system to the center
    azimuthal = math.atan2(vertex[1] - center[1], vertex[0] - center[0])
    if azimuthal < 0:
        azimuthal += 2.0 * math.pi
    azimuthal += math.pi * 0.5
    if azimuthal > math.pi * 2.0:
        azimuthal -= 2.0 * math.pi

    brho = (
        detector_params.magnetic_field * radius * 0.001 / np.abs(math.sin(polar))
    )  # Sometimes our angle is in the wrong quadrant
    if np.isnan(brho):
        brho = 0.0

    # arclength = 0.0
    charge_deposited = first_arc[0, 3]
    small_pad_cutoff = -1  # Where do we cross from big pads to small pads
    for idx in range(len(first_arc) - 1):
        # Stop integrating if we leave the small pad region
        if np.linalg.norm(first_arc[idx + 1, :2]) > 152.0:
            small_pad_cutoff = idx + 1
            break
        # arclength += np.linalg.norm(first_arc[idx + 1, :3] - first_arc[idx, :3])
        charge_deposited += first_arc[idx + 1, 3]
    if charge_deposited == first_arc[0, 3]:
        return None

    # Use the splines to do a fine-grained line integral to calculate the distance
    points = np.empty((1000, 3))
    points[:, 2] = np.linspace(first_arc[0, 2], first_arc[small_pad_cutoff, 2], 1000)
    points[:, 0] = cluster.x_spline(points[:, 2])  # type: ignore
    points[:, 1] = cluster.y_spline(points[:, 2])  # type: ignore
    arclength = np.sqrt((np.diff(points, axis=0) ** 2.0).sum(axis=1)).sum()  # integrate

    dEdx = charge_deposited / arclength

    # fill in our map
    return EstimateResult(
            event=cluster.event,
            cluster_index=cluster_index,
            cluster_label=cluster.label,
            orig_run=orig_run,
            orig_event=orig_event,
            ic_amplitude=ic_amplitude,
            ic_centroid=ic_centroid,
            ic_integral=ic_integral,
            ic_multiplicity=ic_multiplicity,
            vertex_x=vertex[0],
            vertex_y=vertex[1],
            vertex_z=vertex[2],
            center_x=center[0],
            center_y=center[1],
            center_z=center[2],
            polar=polar,
            azimuthal=azimuthal,
            brho=brho,
            dEdx=dEdx,
            sqrt_dEdx=np.sqrt(np.fabs(dEdx)),
            dE=charge_deposited,
            arclength=arclength,
            direction=direction.value,
        )
