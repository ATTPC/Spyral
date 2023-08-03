import numpy as np
from scipy.signal import find_peaks
from scipy.integrate import trapezoid
import time
from multiprocessing import Pool, cpu_count
import multiprocessing
import multiprocessing.pool
from tqdm import tqdm
import sys
sys.path.insert(0, 'TPC-utils')
from tpc_utils import search_high_res
from hdf.TPCH5_utils import get_first_last_event_num, load_trace
import h5py
import os

# PHASE 1 (Constructing point clouds from trace data)

# Daemon workaround found from: https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic

class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, val):
        pass


class NoDaemonProcessPool(multiprocessing.pool.Pool):

    def Process(self, *args, **kwds):
        proc = super(NoDaemonProcessPool, self).Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess

        return proc

def deconv(traces):
    '''
    Parameters:
        traces : A 2D array with N entries of traces, each with a width of 512 entries for each time bucket.

    Returns:
        out    : A deconvolution of each of the traces that were fed in. Has the same Nx512 dimensions as the input.
    '''
    return search_high_res(traces, sigma = 4, threshold = 60, remove_bkg = True, number_it = 200, markov = True, aver_window = 5)[0]

def Phase1(event_num_array, deconv_cores, padxy):
    '''
    Parameters:
    	event_num_array : Array of event numbers of which you want to make point clouds.
        deconv_cores: Number of processes which will be allocated for deconvolution
        padxy: Map of pad number to x-y geometry

    Returns:
    	all_clouds_seg  : 2D list where, for each entry, the first element is the event number, and the second element is the constructed point cloud.
    '''
    process_id = os.getpid()
    ftmp = h5py.File(parent_PATH+'tmp/test_'+str(process_id)+'.h5', 'w')

    #all_clouds_seg = []
    for event_num_i in tqdm(range(len(event_num_array))):
        event_ind = event_num_array[event_num_i]

        # Picks out the traces from the first event.
        meta, all_traces = load_trace(PATH, event_ind)
        if len(all_traces) < 10:
            continue
        all_traces = all_traces.astype(np.float64)

        # Replaces the first and last values in each trace.
        all_traces[:, 0] = all_traces[:, 1]
        all_traces[:, -1] = all_traces[:, -2]

        all_peaks = np.array([])
        all_energies = np.array([])
        all_energies_int = np.array([])
        all_x = np.array([])
        all_y = np.array([])
        all_pad_nums = np.array([])

        all_traces_parts = np.array_split(all_traces, deconv_cores, axis = 0)

        with Pool(deconv_cores) as deconv_p:
            deconv_parts = deconv_p.map(deconv, all_traces_parts)
            response = np.vstack(deconv_parts)

        for trace_num in range(len(all_traces)):

                sig = 4

                peaks, _ = find_peaks(response[trace_num], height = 70)
                peaks = peaks[np.argsort(response[trace_num][peaks])[::-1]][:]

                if len(peaks) != 0:
                    num_pts = int(np.round(sig))

                    energies = np.array([])
                    energies_int = np.array([])

                    # Loop calculates the integrated charge for each peak
                    for peak in peaks:
                            if (((peak+num_pts) < len(response[0])) and ((peak-num_pts) > 0)):
                                extra_pts = np.arange(peak-num_pts, peak+num_pts, dtype = int)

                            energies = np.append(energies, response[trace_num][peak])
                            energies_int = np.append(energies_int, trapezoid(response[trace_num][extra_pts], extra_pts))
                            all_x = np.append(all_x, padxy[:,0][meta[:,4][trace_num]])
                            all_y = np.append(all_y, padxy[:,1][meta[:,4][trace_num]])
                            all_pad_nums = np.append(all_pad_nums, meta[:,4][trace_num])

                    all_peaks = np.append(all_peaks, peaks)
                    all_energies = np.append(all_energies, energies)
                    all_energies_int = np.append(all_energies_int, energies_int)

        all_z = all_peaks

        pc = np.stack((all_x, all_y, all_z, all_energies, all_energies_int, all_pad_nums)).T
        
        # Drops pads where there are more than 4 points. Consider them "noisy"
        pads_to_drop = np.unique(pc[:,5])[np.where(np.array([list(pc[:,5]).count(i) for i in np.unique(pc[:,5])]) >= 10)]
        pc = pc[~np.isin(pc[:,5], pads_to_drop)]

        ftmp.create_dataset('evt'+str(event_ind)+'_cloud', data = pc)
        #all_clouds_seg.append([event_ind, pc])

    ftmp.close()

    #return all_clouds_seg

#GWM -- Best practice to have a main function; probably rename this later
def main():
    start = time.time()    

    all_cores = cpu_count()
    deconv_cores = 10
    evt_cores = 2

    params = np.loadtxt('params.txt', dtype = str, delimiter = ':')
    global PATH
    PATH = params[0, 1]
    global parent_PATH
    parent_PATH = params[2, 1]

    padxy = np.loadtxt('etc/padxy.csv', delimiter = ',', skiprows = 1)

    try:
        os.mkdir(parent_PATH+'tmp')
    except FileExistsError:
        pass

    first_event_num, last_event_num = get_first_last_event_num(PATH)
    print('First event number: ', first_event_num, '\nLast event number: ', last_event_num)

    evt_parts = np.array_split(np.arange(first_event_num+1, last_event_num+1), evt_cores)

    with NoDaemonProcessPool(evt_cores) as evt_p:
        evt_p.map(Phase1, evt_parts, deconv_cores, padxy)
        #run_parts = evt_p.map(Phase1, evt_parts)
    
    print('It takes', time.time()-start, 'seconds to process all', last_event_num-first_event_num, 'events.')

    files = os.listdir(parent_PATH+'tmp/')

    fout = h5py.File(PATH, 'a')

    try:
        clouds = fout.create_group('clouds')
    except ValueError:
        print('Cloud group already exists')
        clouds = fout['clouds']

    for file_i in files:
        fsrc = h5py.File(parent_PATH+'tmp/'+str(file_i), 'r')

        for dataset_i in list(fsrc.keys()):
            try:
                clouds.create_dataset(str(dataset_i), data = fsrc.get(str(dataset_i)))
            except OSError:
                del clouds[str(dataset_i)]
                clouds.create_dataset(str(dataset_i), data = fsrc.get(str(dataset_i)))

        fsrc.close()

    fout.close()

    for file_i in files:
        os.remove(parent_PATH+'tmp/'+str(file_i))
            
    '''
    f = h5py.File(PATH, 'r+')
    try:
        clouds = f.create_group('clouds')
    except ValueError:
        print('Cloud group already exists')
        clouds = f['clouds']

    for part in run_parts:
        for evt in part:
            try:            
                clouds.create_dataset('evt'+str(evt[0])+'_cloud', data = evt[1])
            except OSError:
                del clouds['evt'+str(evt[0])+'_cloud']
                clouds.create_dataset('evt'+str(evt[0])+'_cloud', data = evt[1])

    f.close()
    '''

    print('Phase 1 finished successfully')

if __name__ == "__main__":
    main()