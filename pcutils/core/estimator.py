from .clusterize import least_squares_circle, ClusteredCloud
from .config import DetectorParameters
import numpy as np
import math
from scipy.signal import argrelmax
from scipy.stats import linregress
from enum import Enum

class Direction(Enum):
    NONE: int = -1
    FORWARD: int = 0
    BACKWARD: int = 1

def estimate_physics(cluster: ClusteredCloud, params: DetectorParameters, results: dict[str, list]):

    #Reject beam like trajectories
    n_out_of_beam = len(np.sqrt(cluster.point_cloud.cloud[:, 0]**2.0 + cluster.point_cloud.cloud[:, 1]**2.0) < params.beam_region_radius)
    if n_out_of_beam < 0.1 * len(cluster.point_cloud.cloud):
        return
    
    #Drop any points which do not have a minimum number of neighbors
    #cluster.point_cloud.drop_isolated_points()
    cluster.point_cloud.smooth_cloud()
    if len(cluster.point_cloud.cloud) < 10:
        return
    #Sort our cloud to be ordered in z
    cluster.point_cloud.sort_in_z()
    #Smooth the cloud out again; this is safe as we operate on a single cluster
    
    rhos = np.linalg.norm(cluster.point_cloud.cloud[:, :2], axis=1) #cylindrical coordinates rho
    direction = Direction.NONE

    #Sample the beginning and the end of the trajectory radii
    average_window_rho = np.mean(rhos[:5])
    average_micromegas_rho = np.mean(rhos[-5:-1])

    vertex = np.array([0., 0., 0.]) #reaction vertex
    center = np.array([0., 0., 0.]) #spiral center

    #See if in-spiraling to the window or microgmegas, sets the direction and guess of z-vertex
    if average_window_rho > average_micromegas_rho:
        direction = Direction.BACKWARD
        rhos = np.flip(rhos, axis=0)
        cluster.point_cloud.cloud = np.flip(cluster.point_cloud.cloud, axis=0)
    else:
        direction = Direction.FORWARD

    average_window_rho = np.mean(rhos[:5])
    average_micromegas_rho = np.mean(rhos[-5:-1])

    #Guess that the vertex is the first point
    vertex = cluster.point_cloud.cloud[0, :3]
    
    #Find the first point that is furthest from the vertex in rho (local maximum) to get the first arc of the trajectory
    rho_to_vertex = np.linalg.norm(cluster.point_cloud.cloud[1:, :2] - vertex[:2], axis=1)
    maxima = argrelmax(rho_to_vertex, order=10)[0]
    maximum = 0
    if len(maxima) == 0:
        maximum = len(cluster.point_cloud.cloud)
    else:
        maximum = maxima[0]
    first_arc = cluster.point_cloud.cloud[:(maximum+1)]

    #Fit a circle to the first arc and extract some physics
    center[0], center[1], radius, _ = least_squares_circle(first_arc[:, 0], first_arc[:, 1])
    center[2] = vertex[2]
    test_index = max(10, int(maximum * 0.5))
    fit = linregress(cluster.point_cloud.cloud[:test_index, 2], rho_to_vertex[:test_index])
    polar = math.atan(fit.slope)
    if direction is Direction.BACKWARD:
        polar += math.pi

    azimuthal = math.atan2(vertex[1] - center[1], vertex[0] - center[0])
    if azimuthal < 0:
        azimuthal += 2.0 * math.pi

    brho = params.magnetic_field * radius * 0.001 / (math.sin(polar))
    arclength = 0.0
    for idx in range(len(first_arc)-1):
        arclength += np.linalg.norm(first_arc[idx+1, :3] - first_arc[idx, :3])
    charge_deposited = np.sum(first_arc[:, 4])
    dEdx = charge_deposited/arclength

    #fill in our map
    results['event'].append(cluster.point_cloud.event_number)
    results['cluster'].append(cluster.label)
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



    

