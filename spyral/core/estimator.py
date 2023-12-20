from .cluster import Cluster
from .config import DetectorParameters, EstimateParameters
from ..utils.circle import generate_circle_points, least_squares_circle

import numpy as np
import math
from scipy.signal import argrelmax
from scipy.stats import linregress
from enum import Enum

class Direction(Enum):
    NONE: int = -1
    FORWARD: int = 0
    BACKWARD: int = 1

def estimate_physics(cluster_index: int, cluster: Cluster, ic_amplitude: float, ic_centroid: float, ic_integral: float, estimate_params: EstimateParameters, detector_params: DetectorParameters, results: dict[str, list]):

    # Do some cleanup, reject clusters which have too few points
    if len(cluster.data) < estimate_params.min_total_trajectory_points:
        return
    beam_region_fraction = float(len(cluster.data[np.linalg.norm(cluster.data[:, :2], axis=1) < detector_params.beam_region_radius])) / float(len(cluster.data))
    if beam_region_fraction > 0.9:
        return
    
    rhos = np.linalg.norm(cluster.data[:, :2], axis=1) #cylindrical coordinates rho
    direction = Direction.NONE

    vertex = np.array([0., 0., 0.]) #reaction vertex
    center = np.array([0., 0., 0.]) #spiral center

    halfway = int(len(cluster.data) * 0.5)
    _, _, begin_radius, _ = least_squares_circle(cluster.data[:halfway, 0], cluster.data[:halfway, 1])
    _, _, end_radius, _ = least_squares_circle(cluster.data[halfway:, 0], cluster.data[halfway:, 1])

    #See if in-spiraling to the window or microgmegas, sets the direction and guess of z-vertex
    #If backward, flip the ordering of the cloud to simplify algorithm
    if begin_radius < end_radius:
        direction = Direction.BACKWARD
        rhos = np.flip(rhos, axis=0)
        cluster.data = np.flip(cluster.data, axis=0)
    else:
        direction = Direction.FORWARD

    #Guess that the vertex is the first point; make sure to copy! not reference
    vertex[:] = cluster.data[0, :3]
    
    #Find the first point that is furthest from the vertex in rho (local maximum) to get the first arc of the trajectory
    rho_to_vertex = np.linalg.norm(cluster.data[1:, :2] - vertex[:2], axis=1)
    maxima = argrelmax(rho_to_vertex, order=10)[0]
    maximum = 0
    if len(maxima) == 0:
        maximum = len(cluster.data)
    else:
        maximum = maxima[0]
    first_arc = cluster.data[:(maximum+1)]

    #Fit a circle to the first arc and extract some physics
    center[0], center[1], radius, _ = least_squares_circle(first_arc[:, 0], first_arc[:, 1])
    circle = generate_circle_points(center[0], center[1], radius)
    #Re-estimate vertex using the fitted circle. Extrapolate back to point closest to beam axis
    vertex_estimate_index = np.argsort(np.linalg.norm(circle, axis=1))[0]
    vertex[:2] = circle[vertex_estimate_index]
    #Re-calculate distance to vertex
    rho_to_vertex = np.linalg.norm((cluster.data[:, :2] - vertex[:2]), axis=1)

    #Do a linear fit to small segment of trajectory to extract rho vs. z and extrapolate vertex z
    test_index = max(10, int(maximum * 0.5))
    fit = linregress(cluster.data[:test_index, 2], rho_to_vertex[:test_index])
    vertex_rho = np.linalg.norm(vertex[:2])
    vertex[2] = (vertex_rho - fit.intercept) / fit.slope
    center[2] = vertex[2]

    #Toss tracks whose verticies are not close to the origin in x,y
    if vertex_rho > estimate_params.max_distance_from_beam_axis:
        return

    polar = math.atan(fit.slope)
    if direction is Direction.BACKWARD:
        polar += math.pi

    #From the trigonometry of the system to the center
    azimuthal = math.atan2(vertex[1] - center[1], vertex[0] - center[0])
    if azimuthal < 0:
        azimuthal += 2.0 * math.pi
    azimuthal -= math.pi * 1.5
    if azimuthal < 0:
        azimuthal += 2.0 * math.pi

    brho = detector_params.magnetic_field * radius * 0.001 / (math.sin(polar))
    if np.isnan(brho):
        brho = 0.0
    
    # big_pads = np.where(first_arc[:, 4] >= 1.0)
    first_big_pad_index = len(first_arc)
    # # if len(big_pads) > 1:
    # #     first_big_pad_index = big_pads[0][0]
    # # if first_big_pad_index == 0:
    # #     print("Oops all big pads")
    # #     return
    arclength = 0.0
    for idx in range(first_big_pad_index - 1):
        arclength += np.linalg.norm(first_arc[idx+1, :3] - first_arc[idx, :3])
    charge_deposited = np.sum(first_arc[:, 3])
    dEdx = charge_deposited/arclength

    #fill in our map
    results['event'].append(cluster.event)
    results['cluster_index'].append(cluster_index)
    results['cluster_label'].append(cluster.label)
    results['ic_amplitude'].append(ic_amplitude)
    results['ic_centroid'].append(ic_centroid)
    results['ic_integral'].append(ic_integral)
    results['vertex_x'].append(vertex[0])
    results['vertex_y'].append(vertex[1])
    results['vertex_z'].append(vertex[2])
    results['center_x'].append(center[0])
    results['center_y'].append(center[1])
    results['center_z'].append(center[2])
    results['polar'].append(polar)
    results['azimuthal'].append(azimuthal)
    results['brho'].append(brho)
    results['dEdx'].append(dEdx)
    results['dE'].append(charge_deposited)
    results['arclength'].append(arclength)
    results['direction'].append(direction.value)