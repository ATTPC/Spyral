import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import sys
sys.path.append('..')
from TPCH5_utils import get_first_last_event_num, HDF5_LoadClouds
import h5py
import circle_fit
import multiprocessing
import multiprocessing.pool
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

# PHASE 3 (simple track analysis)

def make_circle(xc, yc, R):
    theta = np.linspace(0, 2*np.pi, 1000)
    rx = R * np.cos(theta) + xc
    ry = R * np.sin(theta) + yc
    return rx, ry

def dist_func(t, r0, m):
    return r0 + m*t

def SimpleAnalysis(data, track_id):
    subset = data[data[:,5] == track_id]
    if len(subset) <= 100:
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    dists = np.sqrt((subset[:,0])**2 + (subset[:,1])**2)
    farthest_pt = np.argsort(dists)[::-1][0]

    # Fits a circle on the points from the start of the track to the farthest point (should be a half circle)
    if np.abs(len(subset) - farthest_pt+1) <= 2: # If the farthest point is in the last two points (i.e. track goes forward but not a full circle)
        farthest_pt -= 2
        #direction = 1
    elif farthest_pt <= 2: # If the farthest point is in the first two points (i.e. track goes backward but not a full circle)
        farthest_pt += 2
        #direction = -1
    
    if (farthest_pt / len(subset)) < 0.5: # If the farthest point is towards the end of track (i.e. track goes backward)
        direction = 1
    elif (farthest_pt / len(subset)) >= 0.5: # If the farthest point is towards start of track (i.e. track goes forwards)
        direction = -1
    
    #if (farthest_pt / len(subset)) >= 0.5: # If the farthest point is towards the end of track (i.e. track goes backward)
    if direction == -1:
        #direction = -1
        xc, yc, R, _ = circle_fit.least_squares_circle(subset[farthest_pt:,:2])
        rx, ry = make_circle(xc, yc, R)
        cdist = np.sqrt(rx**2 + ry**2)
        xvert = rx[np.argsort(cdist)[0]]
        yvert = ry[np.argsort(cdist)[0]]

        distpopt, distpcov = curve_fit(dist_func, 
                                       subset[farthest_pt:, 2],
                                       (np.sqrt((subset[:,0] - xvert)**2 + (subset[:,1] - yvert)**2))[farthest_pt:])

        zvert = (np.sqrt(xvert**2 + yvert**2) - distpopt[0]) / distpopt[1]
        polar = np.arctan(distpopt[1]) * 180 / np.pi + 180
        azimuth = np.arctan2(subset[-1, 1], subset[-1, 0]) * 180 / np.pi
        if azimuth < 0:
            azimuth += 360
        brho = Bmag * R / 1000 / np.sin(polar * np.pi / 180) # In T*m
        #betagamma = brho / 3.107 * ch / ma
        #energy = amuev * (np.sqrt(betagamma**2 + 1) - 1) # In MeV/u

    #elif (farthest_pt / len(subset)) < 0.5: # If the farthest point is towards start of track (i.e. track goes forwards)
    elif direction == 1:
        #direction = 1
        xc, yc, R, _ = circle_fit.least_squares_circle(subset[:farthest_pt,:2])
        rx, ry = make_circle(xc, yc, R)
        cdist = np.sqrt(rx**2 + ry**2)
        xvert = rx[np.argsort(cdist)[0]]
        yvert = ry[np.argsort(cdist)[0]]

        distpopt, distpcov = curve_fit(dist_func, 
                                       subset[:farthest_pt, 2],
                                       (np.sqrt((subset[:,0] - xvert)**2 + (subset[:,1] - yvert)**2))[:farthest_pt])

        zvert = (np.sqrt(xvert**2 + yvert**2) - distpopt[0]) / distpopt[1]
        polar = np.arctan(distpopt[1]) * 180 / np.pi
        azimuth = np.arctan2(subset[0, 1], subset[0, 0]) * 180 / np.pi
        if azimuth < 0:
            azimuth += 360
        brho = Bmag * R / 1000 / np.sin(polar * np.pi / 180) # In T*m
        #betagamma = brho / 3.107 * ch / ma
        #energy = amuev * (np.sqrt(betagamma**2 + 1) - 1) # In MeV/u

    results = np.array([polar, azimuth, brho, xvert, yvert, zvert, direction])

    return results

if __name__ == '__main__':
    Bmag = 2.991

    PATH = '/mnt/analysis/e20009/e20009_Turi/run_0348.h5'

    # First event num: 146686, Last event num: 228636

    for event_num in tqdm(range(146686+3784, 146686+3785)): 
        data = HDF5_LoadClouds(PATH, event_num)
        for track_id in np.unique(data[:,5]):
            subset = data[data[:,5] == track_id]
            dists = np.sqrt((subset[:,0])**2 + (subset[:,1])**2)
            farthest_pt = np.argsort(dists)[::-1][0]
            #print(np.abs(len(subset) - farthest_pt+1))
            print(farthest_pt, len(subset))
            results = SimpleAnalysis(data, track_id)
            print(farthest_pt, len(subset))
