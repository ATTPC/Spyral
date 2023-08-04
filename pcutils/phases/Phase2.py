import numpy as np
from ..hdf.TPCH5_utils import HDF5_LoadClouds, get_first_last_event_num
from sklearn.cluster import DBSCAN
import h5py
import circle_fit # circle_fit found from: https://www.sciencedirect.com/science/article/pii/S0167947310004809?via%3Dihub
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

# PHASE 2 (Noise removal, clustering, and recombining broken tracks)

def RemoveNoiseAndCluster(data, eps1 = 0.1, eps2 = 0.05):
    '''
    Parameters:
        data           : A point cloud where the first 3 columns are the x, y, and z positions (x and y are in mm, z is in timebuckets).
        eps1           : The maximum distance between two samples for one to be considered as in the neighborhood of the other in first clustering attempt.
        eps2           : The maximum distance between two samples for one to be considered as in the neighborhood of the other in second clustering attempt.

    Returns:
        data_no_noise2 : The point cloud after two rounds of noise removal.
    '''
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
    #data_no_noise2 = np.hstack((data_no_noise2, np.array([np.ones(len(data_no_noise2))]).T))    

    # Gets rid of clusters where the number of points is below a threshold
    
    min_pts = 20
    
    for i in np.unique(data_no_noise2[:,6]):
        if len(data_no_noise2[data_no_noise2[:,6] == i]) < min_pts:
            data_no_noise2 = data_no_noise2[data_no_noise2[:,6] != i]
            
    # Renormalizes the cluster labels to increment from 0 to number_of_clusters-1 by 1
    
    for i in range(len(np.unique(data_no_noise2[:,6]))):
        data_no_noise2[data_no_noise2[:,6] == np.unique(data_no_noise2[:,6])[i], 6] = i
            
    
    # Rescale data up
    
    data_no_noise2[:,0] *= 292
    data_no_noise2[:,1] *= 292
    data_no_noise2[:,2] *= 512
    
    return data_no_noise2

def Cluster(data, eps1 = 0.25, eps2 = 0.125):
    # Scale data down
    
    data_no_noise2 = data
    
    # Add the clustering labels as a column in the dataset
    
    data_no_noise2 = np.hstack((data_no_noise2, np.zeros((len(data_no_noise2), 1))))
    
    return data_no_noise2

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

def RecombineTracks(data):
    '''
    Parameters:
        data       : Point cloud where you want broken tracks to be recombined.

    Returns:
        data_recom : Point cloud after broken tracks are recombined.
    '''
    data_recom = np.copy(data)

    track_centers = []

    # Fits a circle to each of the clustered tracks and records their respective centers.
    for label in np.unique(data_recom[:,6]):
        xc, yc, R, _ = circle_fit.least_squares_circle(data_recom[data_recom[:,6] == label, :2])
        track_centers.append([xc, yc])

    # If there is only one track, exit.
    if len(track_centers) == 1:
        return data_recom

    track_centers = np.vstack(track_centers)

    # Clusters track centers that are within 25 mm of each other in the x-y plane
    tc_clusters = DBSCAN(eps = 25, min_samples = 1).fit(track_centers)
    # Loops through the different clusters recommended for track recombination
    for i in np.unique(tc_clusters.labels_):
        # Grabs the cluster id of the tracks that are going to be recombined
        cluster_labels = np.where(tc_clusters.labels_ == i)[0]    
        data_recom[np.isin(data_recom[:,6], cluster_labels), 6] = i
    
    # Sorts the array by track_id and then by z-coordinate.
    data_recom = data_recom[np.lexsort((data_recom[:,2], data_recom[:,6]))]
    
    return data_recom

def SmoothPC(pc, r = 10):
    '''
    Parameters:
        pc          : Point cloud.
        r           : Maximum distance between points to classify them as neighbors.

    Returns:
        smoothed_pc : Smoothed point cloud where a weighted average was taken.
    ''' 
    smoothed_pc = []
    for i in range(len(pc)):
        neighbors = pc[np.sqrt((pc[:,0]-pc[i,0])**2+(pc[:,1]-pc[i,1])**2+(pc[:,2]-pc[i,2])**2) <= r]
        # Weight points
        xs = sum(neighbors[:,0] * neighbors[:,4])
        ys = sum(neighbors[:,1] * neighbors[:,4])
        zs = sum(neighbors[:,2] * neighbors[:,4])
        cs = sum(neighbors[:,3])
        ics = sum(neighbors[:,4])
        #smoothed_pc.append(np.average(neighbors, axis = 0))
        smoothed_pc.append(np.array([xs/ics, ys/ics, zs/ics, cs/len(neighbors), ics/len(neighbors), pc[i,5]]))
    smoothed_pc = np.vstack(smoothed_pc)
    # Removes duplicate points
    smoothed_pc = smoothed_pc[sorted(np.unique(smoothed_pc, axis = 0, return_index = True)[1])]
    # Removes NaNs
    smoothed_pc = smoothed_pc[~np.isnan(smoothed_pc).any(axis = 1)]
    return smoothed_pc

def Phase2(evt_num_array, micromegas, window, length, hdf5_path):
    '''
    Parameters:
        evt_num_array  : Array of event numbers of which you want to analyze.
        micromegas: Timebucket of the micromegas edge
        window: Timebucket of the window edge
        length: Length of the detector in mm
        hdf5_path: Path to hdf5 file containing point cloud data

    Returns:
        all_clouds_seg : 2D list where, for each entry, the first element is the event number, and the second element is the updated point cloud.
    '''
    all_clouds_seg = []
    for event_num_i in tqdm(range(len(evt_num_array))):
        event_ind = int(evt_num_array[event_num_i])

        try:
            data = HDF5_LoadClouds(hdf5_path, event_ind)
        except TypeError:
            continue

        # If point cloud has less than 50 points, do no data removal and give all points the same track_id.
        if len(data) < 50:
            data[:,2] = (data[:,2] - window) / (micromegas - window) * length # Converts third column from timebuckets to z-coordinate.
            data = np.hstack((data, np.zeros((len(data), 1))))
            all_clouds_seg.append([event_ind, data])
            continue

        data = SmoothPC(data, r = 10)

        # Noise removal and clustering.
        data = RemoveNoiseAndCluster(data, eps1 = 0.1, eps2 = 0.05)
        #data = Cluster(data, eps1 = 0.1, eps2 = 0.05)

        # If point cloud has less than 50 points after clustering once, do no more data removal and give all points the same track_id.
        if len(data) < 50:
            data[:,2] = (data[:,2] - window) / (micromegas - window) * length # Converts third column from timebuckets to z-coordinate.
            data = np.hstack((data, np.zeros((len(data), 1))))
            all_clouds_seg.append([event_ind, data])
            continue

        data = RecombineTracks(data)
        data[:,2] = (data[:,2] - window) / (micromegas - window) * length # Converts third column from timebuckets to z-coordinate.
        data = data[np.lexsort((data[:,2], data[:,6]))] # Sorts point cloud by track_id and then by z-coordinate.
        all_clouds_seg.append([event_ind, data])
    return all_clouds_seg

#GWM -- Best practice to have a main function; probably rename this later
def main():
    start = time.time()
    # Constants for converting from timebucket (tb) to position (mm)
    micromegas = 66.0045 # Timebucket of the micromega edge for Be10
    window = 399.455 # Timebucket of the window edge for Be10
    length = 1000 # Length of the detector in mm
    #micromegas = 15 # Timebucket of the the micromega edge for C14
    #window = 500 # Timebucket of the window edge for C14

    #all_cores = int(cpu_count() / 4)
    all_cores = 2

    params = np.loadtxt('params.txt', dtype = str, delimiter = ':')
    hdf5_path = params[0, 1]

    #PATH = '/mnt/analysis/e20009/a1954_Turi/run_0055.h5'
    #PATH = '/mnt/analysis/e20009/e20009_Turi/run_0347.h5'
    #PATH = '/mnt/analysis/e20009/e20009_Turi/Be10dp178.h5'
    first_event_num, last_event_num = get_first_last_event_num(hdf5_path)
    print('First event number: ', first_event_num, '\nLast event num: ', last_event_num)
    
    evt_parts = np.array_split(np.arange(first_event_num+1, last_event_num+1), all_cores)

    with Pool(all_cores) as evt_p:
        run_parts = evt_p.map(Phase2, evt_parts, micromegas, window, length, hdf5_path)

    print('It takes: ', time.time()-start, ' seconds to process ', last_event_num-first_event_num+1, ' events.')

    f = h5py.File(hdf5_path, 'r+')
    clouds = f['clouds']
    for part in run_parts:
        for evt in part:
            try:
                clouds.create_dataset('evt'+str(int(evt[0]))+'_cloud', data = evt[1])            
            except OSError:
                del clouds['evt'+str(int(evt[0]))+'_cloud']
                clouds.create_dataset('evt'+str(int(evt[0]))+'_cloud', data = evt[1])           
    
    f.close()

    print('Phase 2 finished successfully')

if __name__ == "__main__":
    main()
