import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import splrep, splev
from iteration_utilities import duplicates
from ..hdf.TPCH5_utils import get_first_last_event_num, HDF5_LoadClouds
from multiprocessing import Pool
from tqdm import tqdm

# PHASE 3 (simple track analysis)

def FindNN(subset, r = 15):
    '''
    Parameters:
        subset : The subset of the point cloud corresponding to the chosen track.
        r      : Radius of the nearest neighbor (NN) search given in mm.
    
    Returns:
        NNs    : Boolean array that contains if each point in the subset has at least 5 NNs. Should be used as a mask.
    '''
    dists = np.sqrt(subset[:,0]**2 + subset[:,1]**2) # Calculates the distance of each point to the z-axis.
    NNs = np.array([sum(np.sqrt((subset[i,2]-subset[:,2])**2+(dists[i]-dists)**2) < r) for i in range(len(dists))]) # Calculates the number of NNs for each point.
                   
    return np.array(NNs) > 5

def JiggleDupes(subset):
    '''
    Parameters:
        subset     : The subset of points.
        
    Returns:
        new_subset : New subset where any duplicates in the z-coordinate are interpolated between the neighbors.
    '''
    a = np.unique(np.array(list(duplicates(subset[:,2]))))

    for i in range(len(a)):

        inds = np.where(subset[:,2] == a[i])[0][[0,-1]]
        if inds[0] > 0:
            inds[0] -= 1
        if inds[1] < (len(subset)-1):
            inds[1] += 1

        subset[inds[0]:inds[1]+1, 2] = np.linspace(subset[inds[0], 2], subset[inds[1], 2], len(subset[inds[0]:inds[1]+1, 2]))
    return subset

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

def CircleBoi(data):
    '''
    Parameters:
        data    : 2D array meant to be analyzed (First column must be x-data, second column must be y-data).

    Returns:
        results : Array of results (First element is x-center, second element is y-center, third element is radius).
    '''
    results = np.array([])
    
    xs = data[:,0]
    ys = data[:,1]
    
    xb = sum(xs) / len(xs)
    yb = sum(ys) / len(ys)
    
    upix = xs - xb
    vpix = ys - yb
    
    suu = sum(upix**2)
    suv = sum(upix * vpix)
    svv = sum(vpix**2)
    suuu = sum(upix**3)
    suuv = sum(upix**2 * vpix)
    suvv = sum(upix * vpix**2)
    svvv = sum(vpix**3)
    
    vc = (svvv + suuv - suv/suu*(suuu + suvv)) / 2 / (svv - suv*suv/suu)
    uc = (suuu/2 + suvv/2 - vc*suv) / suu
    
    results = np.append(results, uc + xb)
    results = np.append(results, vc + yb)
    
    al = uc**2 + vc**2 + (suu + svv) / len(xs)
    
    results = np.append(results, np.sqrt(al))
    
    return results

def SimpleAnalysis(data, track_id, Bmag):
    '''
    Parameters:
        data     : Total point cloud
        track_id : Track id number that you want to analyze.
        Bmag: Magnetic field in Tesla

    Returns:
        results  : Array of simple analysis results (polar angle, azimuth angle, brho, xyz-vertices direction, dEdx, deavg)
    '''
    subset = data[data[:,6] == track_id]
    subset = subset[FindNN(subset, r = 15)]
    subset = JiggleDupes(subset)
    
    # Track is too sparse -> Return NaNs
    if len(subset) <= 100:
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    
    # Track is confined to beam region (likely a beam event) -> Return NaNs
    if len(subset[np.sqrt(subset[:,0]**2 + subset[:,1]**2) > 25]) / len(subset) < 0.1:
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
        
    subset0 = min(subset[:,2])
    subsetscale = max(subset[:,2]) - min(subset[:,2])
    #subset[:,2] = (subset[:,2] - subset0) / subsetscale
    new_subset = (subset - subset0) / subsetscale

    dists0 = min(dists)
    distsscale = max(dists) - min(dists)
    #dists = (dists - dists0) / distsscale
    new_dists = (dists - dists0) / distsscale
    
    xnew = np.arange(min(subset[:,2]), max(subset[:,2]), 0.1)
        
    if direction == -1:
        tck = splrep(new_subset[::-1,2], new_dists[::-1], s = 1, task = 0)
        spline = splev(new_subset[::-1,2], tck, der = 0)[::-1]
        dspline = splev(new_subset[::-1,2], tck, der = 1)[::-1]
        ddspline = splev(new_subset[::-1,2], tck, der = 2)[::-1]
    else:
        tck = splrep(new_subset[:,2], new_dists, s = 1, task = 0)
        spline = splev(new_subset[:,2], tck, der = 0)
        dspline = splev(new_subset[:,2], tck, der = 1)
        ddspline = splev(new_subset[:,2], tck, der = 2)       

    #subset[:,2] = subset[:,2] * subsetscale + subset0
    #dists = dists * distsscale + dists0
    xnew = xnew * subsetscale + subset0
    spline = spline * distsscale + dists0
    dspline = dspline * distsscale + dists0
    ddspline = ddspline * distsscale + dists0
    
    try:
        farthest_pts = np.where(np.logical_and(np.diff(np.sign(dspline)), ddspline[:-1] < 0))[0]
        farthest_pts = farthest_pts[farthest_pts > 10]
        farthest_pt = farthest_pts[0]
    except IndexError:
        farthest_pt = len(subset)-1        

    farthest_pt = round(farthest_pt / 2)

    #xc, yc, R, _ = circle_fit.hyperLSQ(subset[:round((1/2)*farthest_pt),:2])
    xc, yc, R = CircleBoi(subset[:farthest_pt,:2])

    rx, ry = make_circle(xc, yc, R)
    cdist = np.sqrt(rx**2 + ry**2)
    xvert = rx[np.argsort(cdist)[0]]
    yvert = ry[np.argsort(cdist)[0]]

    try:
        distpopt, _ = curve_fit(dist_func, 
                                subset[:max(10, round((1/2)*farthest_pt)), 2],
                                np.sqrt((subset[:,0]-xvert)**2+(subset[:,1]-yvert)**2)[:max(10, round((1/2)*farthest_pt))])
    except ValueError:
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

    zvert = (np.sqrt(xvert**2 + yvert**2) - distpopt[0]) / distpopt[1]

    polar = np.arctan(distpopt[1]) * 180 / np.pi
    if direction == -1:
        polar += 180

    azimuth = np.arctan2(-(xc-xvert), (yc-yvert)) * 180 / np.pi
    if azimuth < 0:
        azimuth += 360
        
    brho = Bmag * R / 1000 / np.sin(polar * np.pi / 180) # In T*m

    total_track_len = sum(np.array([np.sqrt((subset[i+1,0]-subset[i,0])**2+(subset[i+1,1]-subset[i,1])**2+(subset[i+1,2]-subset[i,2])**2) for i in range(len(subset)-1)]))

    subsubset = subset[:farthest_pt]
    #subsubset = subset[:round((1/4)*len(subset))]

    track_len = sum(np.array([np.sqrt((subsubset[i+1,0]-subsubset[i,0])**2+(subsubset[i+1,1]-subsubset[i,1])**2+(subsubset[i+1,2]-subsubset[i,2])**2) for i in range(len(subsubset)-1)]))    
    dEdx = sum(subsubset[:,4]) / track_len
    deavg = sum(subsubset[:,4]) / len(subsubset[:,4])

    results = np.array([polar, azimuth, brho, xvert, yvert, zvert, direction, dEdx, deavg, total_track_len])

    return results

def Phase3(evt_num_array, hdf5_path, Bmag):
    '''
    Parameters:
        evt_num_array   : An array of event numbers of which you want to analyze.
        hdf5_path: Path to a file containing point cloud data
        Bmag: Magnetic field in Tesla

    Returns:
        all_results_seg : 2D list of the compiled results from the simple analysis (event number, track id, xyz-vertices, polar angle, azimuth angle, brho, direction, deavg, dEdx.

    '''
    all_results_seg = []
    
    for event_num_i in tqdm(range(len(evt_num_array))):
        event_num = evt_num_array[event_num_i]
        
        try:
            data = HDF5_LoadClouds(hdf5_path, event_num)
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
            results = SimpleAnalysis(data, track_id, Bmag)
            if ~np.all(np.isnan(results)):

                all_results_seg.append([event_num, track_id, *results[3:6], *results[:3], results[6], results[7], results[8], results[9]])

    return all_results_seg

def main():
    # Experiment set-up specific info
    Bmag = 2.991 # B field in T

    #all_cores = cpu_count()
    all_cores = 20

    params = np.loadtxt('params.txt', dtype = str, delimiter = ':')
    hdf5_path = params[0, 1]
    ntuple_PATH = params[1, 1]

    first_event_num, last_event_num = get_first_last_event_num(hdf5_path)
    print('First event number: ', first_event_num, '\nLast event num: ', last_event_num)

    evt_parts = np.array_split(np.arange(first_event_num, last_event_num+1), all_cores)

    with Pool(all_cores) as evt_p:
        run_parts = evt_p.map(Phase3, evt_parts, hdf5_path, Bmag)

    all_results = np.vstack(run_parts)

    ntuple_additions = pd.DataFrame(all_results, columns = ['evt', 'track_id', 'gxvert', 'gyvert', 'gzvert', 'gpolar', 'gazimuth', 'gbrho', 'direction', 'dEdx',  'deavg', 'track_len'])

    try:
        old_ntuple = pd.read_csv(ntuple_PATH, delimiter = ',')
        new_ntuple = pd.concat([old_ntuple, ntuple_additions])
        new_ntuple.reset_index(inplace = True, drop = True)
        new_ntuple.to_csv(ntuple_PATH, ',', index = False)
    except FileNotFoundError:
        ntuple_additions.to_csv(ntuple_PATH, ',', index = False)

    print('Phase 3 finished successfully')

if __name__ == "__main__":
    main()
