import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
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
        L  : Line according to the given parameters.
    '''
    return r0 + m*t

def SimpleAnalysis(data, track_id):
    '''
    Parameters:
        data     :
        track_id : 
    Returns:
        results  : 
    '''
    subset = data[data[:,6] == track_id]
    
    if len(subset) <= 100:
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    
    dists = np.sqrt((subset[:,0])**2 + (subset[:,1])**2)
    
    if dists[0] < dists[-1]:
        zvert = subset[0,2]
    else:
        zvert = subset[-1,2]
    
    if np.abs(min(subset[:,2])-zvert) > np.abs(max(subset[:,2])-zvert):
        direction = -1
    else:
        direction = 1
        
    if direction == -1:
        subset = subset[::-1]
        dists = dists[::-1]

    smoothed = lowess(endog = dists, exog = np.arange(len(dists)), frac = 0.05)
    valleys = find_peaks(-1*smoothed[:,1], distance = 50, prominence = 7)[0]
    valleys = valleys[np.logical_and((valleys < 4), (valleys > (len(subset)-4)))]

    def lin_fit(x, m, b):
        return m * x + b

    if len(valleys) > 0:
        popt, pcov = curve_fit(lin_fit, subset[:valleys[0],2], np.gradient(dists)[:valleys[0]])
        farthest_pt = np.abs(-popt[1]/popt[0] - subset[:,2]).argmin()
    else:
        farthest_pt = len(subset)-1
    
    xc, yc, R, _ = circle_fit.hyperLSQ(subset[:round((1/2)*farthest_pt),:2])

    rx, ry = make_circle(xc, yc, R)
    cdist = np.sqrt(rx**2 + ry**2)
    xvert = rx[np.argsort(cdist)[0]]
    yvert = ry[np.argsort(cdist)[0]]

    #fz = interp1d(dists[:20], subset[:20,2], fill_value = 'extrapolate')
    #zvert = fz(0)

    distpopt, _ = curve_fit(dist_func, 
                            subset[:round((1/4)*farthest_pt), 2],
                            (np.sqrt((subset[:,0]-xvert)**2+(subset[:,1]-yvert)**2))[:round((1/4)*farthest_pt)])

    zvert = (np.sqrt(xvert**2 + yvert**2) - distpopt[0]) / distpopt[1]

    polar1 = np.arctan(distpopt[1]) * 180 / np.pi
    if direction == -1:
        polar1 += 180
    polar = polar1

    azimuth = np.arctan2(-(xc-xvert), (yc-yvert)) * 180 / np.pi
    if azimuth < 0:
        azimuth += 360
        
    brho = Bmag * R / 1000 / np.sin(polar * np.pi / 180) # In T*m

    subsubset = subset[:round((9/10)*len(subset))]

    track_len = sum(np.array([np.sqrt((subsubset[i+1,0]-subsubset[i,0])**2+(subsubset[i+1,1]-subsubset[i,1])**2+(subsubset[i+1,2]-subsubset[i,2])**2) for i in range(len(subsubset)-1)]))    
    dEdx = sum(subsubset[:,4]) / track_len
    deavg = sum(subsubset[:,4]) / len(subsubset[:,4])

    results = np.array([polar, azimuth, brho, xvert, yvert, zvert, direction, dEdx, deavg])

    return results

def Phase3(evt_num_array):
    '''
    Parameters:
        evt_num_array   : An array of event numbers of which you want to analyze.

    Returns:
        all_results_seg : 2D list of the compiled results from the simple analysis (event number, track id, xyz-vertices, polar angle, azimuth angle, brho, direction, deavg, dEdx.

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
        for track_id in np.unique(data[:,6]):
            #try:
                #results = SimpleAnalysis(data, track_id)
            #except TypeError:
                #print('Error with event: ', event_num, '\nTrack ID: ', track_id)
            results = SimpleAnalysis(data, track_id)
            if ~np.all(np.isnan(results)):

                all_results_seg.append([event_num, track_id, *results[3:6], *results[:3], results[6], results[7], results[8]])

    return all_results_seg

if __name__ == '__main__':
    # Constants and conversions
    C = 2.99792E8 # Speed of light in m/s
    amuev = 931.494028 # Conversion from amu to eV

    # Experiment set-up specific info
    Bmag = 2.991 # B field in T

    #all_cores = cpu_count()
    all_cores = 20

    params = np.loadtxt('params.txt', dtype = str, delimiter = ':')
    PATH = params[0, 1]
    ntuple_PATH = params[1, 1]

    first_event_num, last_event_num = get_first_last_event_num(PATH)
    print('First event number: ', first_event_num, '\nLast event num: ', last_event_num)

    evt_parts = np.array_split(np.arange(first_event_num, last_event_num+1), all_cores)

    with Pool(all_cores) as evt_p:
        run_parts = evt_p.map(Phase3, evt_parts)

    all_results = np.vstack(run_parts)

    ntuple_additions = pd.DataFrame(all_results, columns = ['evt', 'track_id', 'gxvert', 'gyvert', 'gzvert', 'gpolar', 'gazimuth', 'gbrho', 'direction', 'dEdx',  'deavg'])

    try:
        old_ntuple = pd.read_csv(ntuple_PATH, delimiter = ',')
        new_ntuple = pd.concat([old_ntuple, ntuple_additions])
        new_ntuple.reset_index(inplace = True, drop = True)
        new_ntuple.to_csv(ntuple_PATH, ',', index = False)
    except FileNotFoundError:
        ntuple_additions.to_csv(ntuple_PATH, ',', index = False)

    print('Phase 3 finished successfully')
