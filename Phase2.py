import numpy as np
import pandas as pd
from TPCH5_utils import HDF5_LoadClouds, get_first_last_event_num
from sklearn.cluster import DBSCAN
import h5py
import circle_fit # circle_fit found from: https://www.sciencedirect.com/science/article/pii/S0167947310004809?via%3Dihub
import multiprocessing
import multiprocessing.pool
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

# PHASE 2 (Noise removal, clustering, and recombining broken tracks)

def RemoveNoiseAndCluster(data, eps1 = 0.25, eps2 = 0.125):
    # Scale data down

    data[:,0] /= 292
    data[:,1] /= 292
    data[:,2] /= 512
    
    # Noise removal clustering

    clustering1 = DBSCAN(eps = eps1, min_samples = 5).fit(data[:,:3])
    data_no_noise = data[clustering1.labels_ != -1]
    
    if len(data_no_noise) == 0:
        data_no_noise[:,0] *= 292
        data_no_noise[:,1] *= 292
        data_no_noise[:,2] *= 512
        return data_no_noise

    clustering2 = DBSCAN(eps = eps2, min_samples = 5).fit(data_no_noise[:,:3])
    data_no_noise2 = data_no_noise[clustering2.labels_ != -1]
    
    # Add the clustering labels as a column in the dataset
    
    data_no_noise2 = np.hstack((data_no_noise2, np.array([clustering2.labels_[clustering2.labels_ != -1]]).T))
    
    # Gets rid of clusters where the number of points is below a threshold
    
    min_pts = 20
    
    for i in np.unique(data_no_noise2[:,5]):
        if len(data_no_noise2[data_no_noise2[:,5] == i]) < min_pts:
            data_no_noise2 = data_no_noise2[data_no_noise2[:,5] != i]
            
    # Renormalizes the cluster labels to increment from 0 to number_of_clusters-1 by 1
    
    for i in range(len(np.unique(data_no_noise2[:,5]))):
        data_no_noise2[data_no_noise2[:,5] == np.unique(data_no_noise2[:,5])[i], 5] = i
            
    
    # Rescale data up
    
    data_no_noise2[:,0] *= 292
    data_no_noise2[:,1] *= 292
    data_no_noise2[:,2] *= 512
    
    return data_no_noise2

def make_circle(xc, yc, R):
    theta = np.linspace(0, 2*np.pi, 1000)
    rx = R * np.cos(theta) + xc
    ry = R * np.sin(theta) + yc
    return rx, ry

def RecombineTracks(data):
    data_recom = np.copy(data)

    track_centers = []

    for label in np.unique(data_recom[:,5]):
        xc, yc, R, _ = circle_fit.least_squares_circle(data_recom[data_recom[:,5] == label, :2])
        track_centers.append([xc, yc])

    if len(track_centers) == 1:
        return data_recom

    track_centers = np.vstack(track_centers)

    # Clusters track centers that are within 25 mm of each other in the x-y plane
    tc_clusters = DBSCAN(eps = 25, min_samples = 1).fit(track_centers)
    # Loops through the different clusters recommended for track recombination
    for i in np.unique(tc_clusters.labels_):
        # Grabs the cluster id of the tracks that are going to be recombined
        cluster_labels = np.where(tc_clusters.labels_ == i)[0]    
        data_recom[np.isin(data_recom[:,5], cluster_labels), 5] = i
    
    data_recom = data_recom[np.lexsort((data_recom[:,2], data_recom[:,5]))]
    
    return data_recom

def Phase2(evt_num_array):
    all_clouds_seg = []
    for event_num_i in tqdm(range(len(evt_num_array))):
        event_ind = int(evt_num_array[event_num_i])

        data = HDF5_LoadClouds(PATH, event_ind)
        data = RemoveNoiseAndCluster(data, eps1 = 0.1, eps2 = 0.05)
        if len(data) < 50:
            data[:,2] = (data[:,2] - window) / (micromegas - window) * length
            all_clouds_seg.append([event_ind, data])
            continue
        data = RecombineTracks(data)
        data[:,2] = (data[:,2] - window) / (micromegas - window) * length
        data = data[np.lexsort((data[:,2], data[:,5]))]
        all_clouds_seg.append([event_ind, data])       
    return all_clouds_seg

if __name__ == '__main__':
    start = time.time()
    # Constants for converting from timebucket (tb) to position (mm)
    micromegas = 66.0045 # Time bucket of the micromega edge
    window = 399.455 # Time bucket of the window edge
    length = 1000 # Length of the detector in mm

    #all_cores = int(cpu_count() / 4)
    all_cores = 4

    PATH = '/mnt/analysis/e20009/e20009_Turi/run_0348.h5'
    first_event_num, last_event_num = get_first_last_event_num(PATH)
    #evt_ind = 147472
    
    evt_parts = np.array_split(np.arange(first_event_num, last_event_num+1), all_cores)

    with Pool(all_cores) as evt_p:
        run_parts = evt_p.map(Phase2, evt_parts)

    print('It takes: ', time.time()-start, ' seconds to process ', last_event_num-first_event_num+1, ' events.')

    f = h5py.File(PATH, 'r+')
    clouds = f['clouds']
    for part in run_parts:
        for evt in part:
            try:
                clouds.create_dataset('evt'+str(int(evt[0]))+'_cloud', data = evt[1])            
            except OSError:
                del clouds['evt'+str(int(evt[0]))+'_cloud']
                clouds.create_dataset('evt'+str(int(evt[0]))+'_cloud', data = evt[1])           
    
    f.close()


