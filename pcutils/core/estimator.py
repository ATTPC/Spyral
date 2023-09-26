from .clusterize import least_squares_circle, ClusteredCloud
from .config import DetectorParameters, EstimateParameters
import numpy as np
import math
from scipy.signal import argrelmax
from scipy.stats import linregress
from enum import Enum

class Direction(Enum):
    NONE: int = -1
    FORWARD: int = 0
    BACKWARD: int = 1

def generate_circle_points(center_x: float, center_y: float, radius: float) -> np.ndarray:
    theta = np.linspace(0., 2.0 * np.pi, 100000)
    array = np.zeros(shape=(len(theta), 2))
    array[:, 0] = center_x + np.cos(theta) * radius
    array[:, 1] = center_y + np.sin(theta) * radius
    return array

def estimate_physics(cluster_index: int, cluster: ClusteredCloud, estimate_params: EstimateParameters, detector_params: DetectorParameters, results: dict[str, list]):
    #Reject any clusters that were labeled as noise by the clustering algorithm
    #if cluster.label == -1:
    #    return

    if len(cluster.point_cloud.cloud) < estimate_params.min_total_trajectory_points:
        return
    
    rhos = np.linalg.norm(cluster.point_cloud.cloud[:, :2], axis=1) #cylindrical coordinates rho
    direction = Direction.NONE

    #Sample the beginning and the end of the trajectory radii
    average_window_rho = np.mean(rhos[:5])
    average_micromegas_rho = np.mean(rhos[-5:-1])

    vertex = np.array([0., 0., 0.]) #reaction vertex
    center = np.array([0., 0., 0.]) #spiral center

    #See if in-spiraling to the window or microgmegas, sets the direction and guess of z-vertex
    #If backward, flip the ordering of the cloud to simplify algorithm
    if average_window_rho > average_micromegas_rho:
        direction = Direction.BACKWARD
        rhos = np.flip(rhos, axis=0)
        cluster.point_cloud.cloud = np.flip(cluster.point_cloud.cloud, axis=0)
    else:
        direction = Direction.FORWARD

    average_window_rho = np.mean(rhos[:5])
    average_micromegas_rho = np.mean(rhos[-5:-1])

    #Guess that the vertex is the first point; make sure to copy! not reference
    vertex[:] = cluster.point_cloud.cloud[0, :3]
    
    #Find the first point that is furthest from the vertex in rho (local maximum) to get the first arc of the trajectory
    rho_to_vertex = np.linalg.norm(cluster.point_cloud.cloud[1:, :2] - vertex[:2], axis=1)
    maxima = argrelmax(rho_to_vertex, order=2)[0]
    maximum = 0
    if len(maxima) == 0:
        maximum = len(cluster.point_cloud.cloud)
    else:
        maximum = maxima[0]
    first_arc = cluster.point_cloud.cloud[:(maximum+1)]

    #Fit a circle to the first arc and extract some physics
    center[0], center[1], radius, _ = least_squares_circle(first_arc[:, 0], first_arc[:, 1])
    circle = generate_circle_points(center[0], center[1], radius)
    #Re-estimate vertex using the fitted circle. Extrapolate back to point closest to beam axis
    vertex_estimate_index = np.argsort(np.linalg.norm(circle, axis=1))[0]
    vertex[:2] = circle[vertex_estimate_index]
    #Re-calculate distance to vertex
    rho_to_vertex = np.linalg.norm((cluster.point_cloud.cloud[:, :2] - vertex[:2]), axis=1)

    #Do a linear fit to small segment of trajectory to extract rho vs. z and extrapolate vertex z
    test_index = max(5, int(maximum * 0.5))
    fit = linregress(cluster.point_cloud.cloud[:test_index, 2], rho_to_vertex[:test_index])
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
    arclength = 0.0
    for idx in range(len(first_arc)-1):
        arclength += np.linalg.norm(first_arc[idx+1, :3] - first_arc[idx, :3])
    charge_deposited = np.sum(first_arc[:, 4])
    dEdx = charge_deposited/arclength

    #fill in our map
    results['event'].append(cluster.point_cloud.event_number)
    results['cluster_index'].append(cluster_index)
    results['cluster_label'].append(cluster.label)
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


    

