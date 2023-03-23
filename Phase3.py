import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from statsmodels.nonparametric.smoothers_lowess import lowess
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
    '''
    Parameters:
        xc : x-coordinate of the center of the circle to be drawn.
        yc : y-coordinate of the center of the circle to be drawn.
        R  : Radius of the circle to be drawn.

    Returns:
        rx : Array of the x-coordinates of the circle.
        ry : Array of the y-coordinates of the circle.
    '''

    theta = np.linspace(0, 2*np.pi, 1000)
    rx = R * np.cos(theta) + xc
    ry = R * np.sin(theta) + yc
    return rx, ry

def dist_func(t, r0, m):
    '''
    Parameters:
        t  : Arbitrary independent variable.
        r0 : y-intercept to be fit.
        m  : Slope to be fit.
    Returns:
    '''
    return r0 + m*t

def SimpleAnalysis(data, track_id):
    '''
    Parameters:
        data     : Point cloud to be analyzed.
        track_id : Label of the track in the point cloud that you want to analyze.   

    Returns:
        results  : Array of results from the simple analysis (polar angle, azimuthal angle, brho, x-vertex, y-vertex, z-vertex, direction).
    '''
    subset = data[data[:,5] == track_id]
    if len(subset) < 50:
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

    dists = np.sqrt((subset[:,0])**2 + (subset[:,1])**2)
    smoothed = lowess(endog = dists, exog = np.arange(len(dists)), frac = 0.05)
    peaks = find_peaks(smoothed[:,1], distance = 50, prominence = 7)[0]
    valleys = find_peaks(-1*smoothed[:,1], distance = 50, prominence = 7)[0]
    n_peaks = len(peaks)
    n_valleys = len(valleys)
    
    farthest_pt = np.argsort(dists)[::-1][0]

    if ((farthest_pt <= 2) or (farthest_pt >= (len(subset)-2))):
        farthest_pt = np.argsort(dists)[::-1][1]
    
    # Fits a circle on the points from the start of the track to the farthest point (should be a half circle)
        
    # Incomplete rotation (hits the wall)
    if (((farthest_pt / len(subset)) < 0.5) and (n_peaks <= 1) and (n_valleys == 0)):
        direction = -1
    elif (((farthest_pt / len(subset)) >= 0.5) and (n_peaks <= 1) and (n_valleys == 0)):
        direction = 1
    # Weird case. 1 valley, no peak
    elif (((farthest_pt / len(subset)) < 0.5) and (n_peaks == 0) and (n_valleys > 0)):
        direction = -1
    elif (((farthest_pt / len(subset)) >= 0.5) and (n_peaks == 0) and (n_valleys > 0)):
        direction = 1
    # Exactly one rotation
    elif (((farthest_pt / len(subset)) < 0.5) and (n_peaks == 1) and (n_valleys == 1)):
        direction = 1
    elif (((farthest_pt / len(subset)) >= 0.5) and (n_peaks == 1) and (n_valleys == 1)):
        direction = -1
    # Multi-rotation
    elif (((farthest_pt / len(subset)) < 0.5) and (n_peaks >= 1) and (n_valleys >= 1)):
        direction = 1
    elif (((farthest_pt / len(subset)) >= 0.5) and (n_peaks >= 1) and (n_valleys >= 1)):
        direction = -1

    if direction == -1:
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

    elif direction == 1:
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

def Phase3(evt_num_array):
    '''
    Parameters:
        evt_num_array   : An array of event numbers of which you want to analyze.

    Returns:
        all_results_seg : 2D list of the compiled results from the simple analysis (event number, track id, xyz-vertices, polar angle, azimuth angle, brho, direction.

    '''
    all_results_seg = []
    
    for event_num_i in tqdm(range(len(evt_num_array))):
        event_num = evt_num_array[event_num_i]
        
        try:
            data = HDF5_LoadClouds(PATH, event_num)
        except TypeError:
            continue

        # If the point cloud has fewer than 100 points, skip it.
        if len(data) < 100:
            continue
      
        # Loops through the different tracks, runs the simple analysis, and throws out any entries with all NaNs.
        for track_id in np.unique(data[:,5]):
            results = SimpleAnalysis(data, track_id)
            if ~np.all(np.isnan(results)):
                all_results_seg.append([event_num, track_id, *results[3:6], *results[:3], results[6]])

    return all_results_seg

if __name__ == '__main__':
    # Constants and conversions
    C = 2.99792E8 # Speed of light in m/s
    amuev = 931.494028 # Conversion from amu to eV

    # Experiment set-up specific info
    Bmag = 2.991 # B field in T

    #all_cores = cpu_count()
    all_cores = 1

    #PATH = '/mnt/analysis/e20009/e20009_Turi/run_0348.h5'
    PATH = '/mnt/analysis/e20009/e20009_Turi/Be10dp178.h5'
    first_event_num, last_event_num = get_first_last_event_num(PATH)

    evt_parts = np.array_split(np.arange(first_event_num, last_event_num+1), all_cores)

    with Pool(all_cores) as evt_p:
        run_parts = evt_p.map(Phase3, evt_parts)

    all_results = np.vstack(run_parts)

    ntuple_additions = pd.DataFrame(all_results, columns = ['evt', 'track_id', 'gxvert', 'gyvert', 'gzvert', 'gpolar', 'gazimuth', 'gbrho', 'direction'])

    try:
        old_ntuple = pd.read_csv('all_ntuple_Be10dp178_Turi.txt', delimiter = ',')
        new_ntuple = pd.concat([old_ntuple, ntuple_additions])
        new_ntuple.reset_index(inplace = True, drop = True)
        new_ntuple.to_csv('all_ntuple_Be10dp178_Turi.txt', ',', index = False)
    except FileNotFoundError:
        ntuple_additions.to_csv('all_ntuple_Be10dp178_Turi.txt', ',', index = False)

    print('Phase 3 finished successfully')
