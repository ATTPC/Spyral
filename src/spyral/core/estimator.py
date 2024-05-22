from .cluster import Cluster
from .config import DetectorParameters, EstimateParameters
from ..geometry.circle import generate_circle_points, least_squares_circle
from .spy_log import spyral_warn

import numpy as np
import math
from scipy.stats import linregress
from enum import Enum


class Direction(Enum):
    """Enum for the direction of a trajectory

    Attributes
    ----------
    NONE: int
        Invalid value (-1)
    FORWARD: int
        Trajectory traveling in the positive z-direction (0)
    BACKWARD: int
        Trajectory traveling in the negative z-direction (1)
    """

    NONE: int = -1  # type: ignore
    FORWARD: int = 0  # type: ignore
    BACKWARD: int = 1  # type: ignore


def estimate_physics(
    cluster_index: int,
    cluster: Cluster,
    ic_amplitude: float,
    ic_centroid: float,
    ic_integral: float,
    ic_multiplicity: float,
    estimate_params: EstimateParameters,
    detector_params: DetectorParameters,
    results: dict[str, list],
):
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
    detector_params:
        Configuration parameters for the physical detector properties
    results: dict[str, int]
        Dictionary to store estimation results in

    """
    # Check if we have enough points to estimate
    if len(cluster.data) < estimate_params.min_total_trajectory_points:
        return
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
        return

    # Run estimation where we attempt to guess the right direction
    is_good, direction = estimate_physics_pass(
        cluster_index,
        cluster,
        ic_amplitude,
        ic_centroid,
        ic_integral,
        ic_multiplicity,
        detector_params,
        results,
    )

    # If estimation was consistent or didn't meet valid criteria we're done
    if is_good or (not is_good and direction == Direction.NONE):
        return
    # If we made a bad guess, try the other direction
    elif direction == Direction.FORWARD:
        estimate_physics_pass(
            cluster_index,
            cluster,
            ic_amplitude,
            ic_centroid,
            ic_integral,
            ic_multiplicity,
            detector_params,
            results,
            Direction.BACKWARD,
        )
    else:
        estimate_physics_pass(
            cluster_index,
            cluster,
            ic_amplitude,
            ic_centroid,
            ic_integral,
            ic_multiplicity,
            detector_params,
            results,
            Direction.FORWARD,
        )


def choose_direction(cluster_data: np.ndarray) -> Direction:
    """Choose a direction for the trajectory based on which end of the data is in-spiraling

    Parameters
    ----------
    cluster_data: np.ndarray
        T

    """
    rhos = np.linalg.norm(cluster_data[:, :2], axis=1)  # cylindrical coordinates rho
    direction = Direction.NONE

    # See if in-spiraling to the window or microgmegas, sets the direction and guess of z-vertex
    if rhos[0] < rhos[-1]:
        direction = Direction.FORWARD
    else:
        direction = Direction.BACKWARD

    return direction


def estimate_physics_pass(
    cluster_index: int,
    cluster: Cluster,
    ic_amplitude: float,
    ic_centroid: float,
    ic_integral: float,
    ic_multiplicity: float,
    detector_params: DetectorParameters,
    results: dict[str, list],
    chosen_direction: Direction = Direction.NONE,
) -> tuple[bool, Direction]:
    """Estimate the physics parameters for a cluster which could represent a particle trajectory

    Estimation is an imprecise process (by definition), and as such this algorithm requires a lot of
    manipulation of the data.

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
    detector_params:
        Configuration parameters for the physical detector properties
    results: dict[str, int]
        Dictionary to store estimation results in
    chosen_direction: Direction, default=Direction.NONE
        Optional direction for the trajectory. Default
        estimates the direction.

    """

    direction = chosen_direction
    vertex = np.array([0.0, 0.0, 0.0])  # reaction vertex
    center = np.array([0.0, 0.0, 0.0])  # spiral center
    # copy the data so we can modify it without worrying about side-effects
    cluster_data = cluster.data.copy()

    # If chosen direction is set to NONE, we want to have the algorithm
    # try to decide which direction the trajectory is going
    if direction == Direction.NONE:
        direction = choose_direction(cluster_data)

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
    vertex[2] = (vertex_rho - fit.intercept) / fit.slope  # type: ignore
    center[2] = vertex[2]

    # Toss tracks whose verticies are not close to the origin in x,y
    if vertex_rho > detector_params.beam_region_radius:
        return (False, Direction.NONE)

    polar = math.atan(fit.slope)  # type: ignore
    # We have a self consistency case here. Polar should match chosen Direction
    if (polar > 0.0 and direction == Direction.BACKWARD) or (
        polar < 0.0 and direction == Direction.FORWARD
    ):
        return (
            False,
            direction,
        )  # Our direction guess was bad, we need to try again with the other direction
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
        return (False, Direction.NONE)

    # Use the splines to do a fine-grained line integral to calculate the distance
    points = np.empty((1000, 3))
    points[:, 2] = np.linspace(first_arc[0, 2], first_arc[small_pad_cutoff, 2], 1000)
    points[:, 0] = cluster.x_spline(points[:, 2])  # type: ignore
    points[:, 1] = cluster.y_spline(points[:, 2])  # type: ignore
    arclength = np.sqrt((np.diff(points, axis=0) ** 2.0).sum(axis=1)).sum()  # integrate

    dEdx = charge_deposited / arclength

    # fill in our map
    results["event"].append(cluster.event)
    results["cluster_index"].append(cluster_index)
    results["cluster_label"].append(cluster.label)
    results["ic_amplitude"].append(ic_amplitude)
    results["ic_centroid"].append(ic_centroid)
    results["ic_integral"].append(ic_integral)
    results["ic_multiplicity"].append(ic_multiplicity)
    results["vertex_x"].append(vertex[0])
    results["vertex_y"].append(vertex[1])
    results["vertex_z"].append(vertex[2])
    results["center_x"].append(center[0])
    results["center_y"].append(center[1])
    results["center_z"].append(center[2])
    results["polar"].append(polar)
    results["azimuthal"].append(azimuthal)
    results["brho"].append(brho)
    results["dEdx"].append(dEdx)
    results["dE"].append(charge_deposited)
    results["arclength"].append(arclength)
    results["direction"].append(direction.value)
    return (True, direction)
