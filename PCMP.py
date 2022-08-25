import numpy as np
from scipy.signal import find_peaks
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import sys
sys.path.insert(0, 'TPC-utils')
from tpc_utils import background, search_high_res
from TPCH5_utils import get_first_last_event_num, load_trace

def deconv(traces):
	return search_high_res(traces, sigma = 4, threshold = 60, remove_bkg = True, number_it = 200, markov = True, aver_window = 5)[0]


if __name__ == '__main__':
	start = time.time()

	PATH = '/mnt/research/attpc/e20009/h5/run_0231.h5'

	first_event_num, last_event_num = get_first_last_event_num(PATH)

	zap_pads = np.loadtxt('Zap_pads.csv', skiprows = 2, delimiter = ',')

	padxy = np.loadtxt('padxy.csv', delimiter = ',', skiprows = 1)

	pad_loc = pd.read_csv('flutsize.csv')
	pad_big = pad_loc[pad_loc['size'] == 1]
	pad_small = pad_loc[pad_loc['size'] == 0]

	#event_ind = first_event_num+2
	for i in tqdm(range(first_event_num, last_event_num+1)):

		event_ind = i

		# Picks out the traces from the first event.
		meta, all_traces = load_trace(PATH, event_ind)
		all_traces = all_traces.astype(np.float64)

		# Replaces the first and last values in each trace.
		all_traces[:, 0] = all_traces[:, 1]
		all_traces[:, -1] = all_traces[:, -2]

		trim_ct = False

		if trim_ct == True:
    			all_traces, meta = remove_ct(all_traces, meta, threshold1 = 2000, threshold2 = 1600, channel_width = 3, tb_range = 7)

		all_peaks = np.array([])
		all_energies = np.array([])
		all_x = np.array([])
		all_y = np.array([])
		all_pad_nums = np.array([])

		#cores = cpu_count()
		cores = 10
		all_traces_parts = np.array_split(all_traces, cores, axis = 0)

		with Pool(cores) as p:
			parts = p.map(deconv, all_traces_parts)
			response = np.vstack(parts)

			#response, _ = search_high_res(all_traces, 
							#sigma = sig, 
							#threshold = 60, 
							#remove_bkg = True, 
							#number_it = 200, 
							#markov = True, 
							#aver_window = 5)

		for trace_num in range(len(all_traces)):

    			sig = 4

    			peaks, _ = find_peaks(response[trace_num], height = 70)
    			peaks = peaks[np.argsort(response[trace_num][peaks])[::-1]][:]

    			if len(peaks) != 0:
        			num_pts = int(np.round(sig))
        
        			energies = np.array([])
        
        			# Loop calculates the energy for each peak
        			for peak in peaks:
            				if (((peak+num_pts) < len(response[0])) and ((peak-num_pts) > 0)):
                				extra_pts = np.arange(peak-num_pts, peak+num_pts, dtype = int)
                
            				energies = np.append(energies, trapezoid(response[trace_num][extra_pts], extra_pts))
            				#all_x = np.append(all_x, padxy['x'][meta[:,4][trace_num]])
            				all_x = np.append(all_x, padxy[:,0][meta[:,4][trace_num]])
            				#all_y = np.append(all_y, padxy['y'][meta[:,4][trace_num]])
            				all_y = np.append(all_y, padxy[:,1][meta[:,4][trace_num]])
            				all_pad_nums = np.append(all_pad_nums, meta[:,4][trace_num])
        
        			all_peaks = np.append(all_peaks, peaks)
        			all_energies = np.append(all_energies, energies)
        
		drift_vel = 333.076 # In time bucket difference
		#all_z = all_peaks / np.shape(all_traces)[1] * drift_vel
		all_z = all_peaks / np.shape(all_traces)[1] * 1000

		pc = np.stack((all_x, all_y, all_z, all_energies, all_pad_nums)).T

		# Beam region
		energy_thresh_br = 500

		#pc_br = pc[np.isin(all_pad_nums, zap_pads['pad num'])]
		pc_br = pc[np.isin(all_pad_nums, zap_pads[:,4])]
		pc_br = pc_br[np.where(pc_br[:,3] >= energy_thresh_br)]

		# Not beam region
		#energy_thresh_nbr = 1000
		energy_thresh_nbr_big = 4000
		energy_thresh_nbr_small = 4000

		#pc_nbr = pc[~np.isin(all_pad_nums, zap_pads['pad num'])]
		pc_nbr = pc[~np.isin(all_pad_nums, zap_pads[:,4])]
		#pc_nbr = pc_nbr[np.where(pc_nbr[:,3] >= energy_thresh_nbr)]
		pc_nbr_big = pc_nbr[np.where(np.isin(pc_nbr[:,4], pad_big))]
		pc_nbr_big = pc_nbr_big[np.where(pc_nbr_big[:,3] >= energy_thresh_nbr_big)]     
		pc_nbr_small = pc_nbr[np.where(np.isin(pc_nbr[:,4], pad_small))]
		pc_nbr_small = pc_nbr_small[np.where(pc_nbr_small[:,3] >= energy_thresh_nbr_small)]
		pc_nbr = np.vstack((pc_nbr_big, pc_nbr_small))
		# Drops pads where there are more than 4 points. Consider them "noisy"
		pads_to_drop = np.unique(pc_nbr[:,4])[np.where(np.array([list(pc_nbr[:,4]).count(i) for i in np.unique(pc_nbr[:,4])]) >= 5)]
		pc_nbr = pc_nbr[~np.isin(pc_nbr[:,4], pads_to_drop)]

		# Recombine point cloud
		pc = np.vstack((pc_br, pc_nbr))

	print('It takes', time.time()-start, 'seconds to process', last_event_num-first_event_num+1, 'events while using', cores, 'cores.')
