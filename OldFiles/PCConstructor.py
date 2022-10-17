import numpy as np
from scipy.signal import find_peaks
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
import plotly.graph_objects as go
from IPython.display import clear_output, display
import sys
sys.path.insert(0, 'TPC-utils')
from tpc_utils import search_high_res
from TPCH5_utils import get_first_last_event_num, load_trace

class PointCloud:   
    def __init__(self, PATH, event_ind, deconv_it = 200, remove_ct = False, thresholds = [2000, 500, 1000]):
        self.PATH = PATH
        first_event_num, last_event_num = get_first_last_event_num(self.PATH)
        if ((event_ind >= first_event_num) and (event_ind <= last_event_num)):
            self.event_ind = event_ind
        else:
            raise Exception("Event number must be between "+str(first_event_num)+" and "+str(last_event_num))

        global zap_pads, padxy

        zap_pads = np.loadtxt('Zap_pads.csv', skiprows = 2, delimiter = ',')

        padxy = np.loadtxt('padxy.csv', delimiter = ',', skiprows = 1)

        # Picks out the traces from the first event.
        meta, all_traces = load_trace(PATH, event_ind)
        all_traces = all_traces.astype(np.float64)

        # Replaces the first and last values in each trace.
        all_traces[:, 0] = all_traces[:, 1]
        all_traces[:, -1] = all_traces[:, -2]
        
        if remove_ct == True:
            flags = flag_ct(all_traces, meta, threshold1 = thresholds[0], threshold2 = thresholds[1], threshold3 = thresholds[2])
            all_traces = all_traces[~np.isin(meta, flags).all(axis = 1)]
            meta = meta[~np.isin(meta, flags).all(axis = 1)]

        all_peaks = np.array([])
        all_energies = np.array([])
        all_x = np.array([])
        all_y = np.array([])
        all_pad_nums = np.array([])
        
        sig = 4
        
        response, _ = search_high_res(all_traces, 
                                              sigma = sig, 
                                              threshold = 60, 
                                              remove_bkg = True, 
                                              number_it = deconv_it, 
                                              markov = True, 
                                              aver_window = 5)
        
        for trace_num in range(len(all_traces)):
            peaks, _ = find_peaks(response[trace_num], height = 70)
            peaks = peaks[np.argsort(response[trace_num][peaks])[::-1]][:]

            if len(peaks) != 0:
                num_pts = int(np.round(sig))

                energies = np.array([])

                # Loop calculates the energy for each peak
                for peak in peaks:
                    if (((peak+num_pts) < len(response[trace_num])) and ((peak-num_pts) > 0)):
                        extra_pts = np.arange(peak-num_pts, peak+num_pts, dtype = int)

                    energies = np.append(energies, trapezoid(response[trace_num][extra_pts], extra_pts))
                    all_x = np.append(all_x, padxy[:, 0][meta[:,4][trace_num]])
                    all_y = np.append(all_y, padxy[:, 1][meta[:,4][trace_num]])
                    all_pad_nums = np.append(all_pad_nums, meta[:,4][trace_num])

                all_peaks = np.append(all_peaks, peaks)
                all_energies = np.append(all_energies, energies)

        drift_vel = 333.076 # In time bucket difference
        #all_z = all_peaks / np.shape(all_traces)[1] * drift_vel
        all_z = all_peaks / np.shape(all_traces)[1] * 1000

        self.pc = np.stack((all_x, all_y, all_z, all_energies, all_pad_nums)).T
      
    def trim(self, energy_thresh_br = 500, energy_thresh_nbr_big = 5000, energy_thresh_nbr_small = 4000):
        # Reads in the pad sizes
        pad_loc = pd.read_csv('flutsize.csv')
        pad_big = pad_loc[pad_loc['size'] == 1]
        pad_small = pad_loc[pad_loc['size'] == 0]
        
        #pc_br = self.pc[np.isin(self.pc[:,4], zap_pads['pad num'])]
        pc_br = self.pc[np.isin(self.pc[:,4], zap_pads[:,4])]
        pc_br = pc_br[np.where(pc_br[:,3] >= energy_thresh_br)]

        #pc_nbr = self.pc[~np.isin(self.pc[:,4], zap_pads['pad num'])] # ~ operator inverts booleans
        pc_nbr = self.pc[~np.isin(self.pc[:,4], zap_pads[:,4])]
    
        pc_nbr_big = pc_nbr[np.where(np.isin(pc_nbr[:,4], pad_big))]
        pc_nbr_big = pc_nbr_big[np.where(pc_nbr_big[:,3] >= energy_thresh_nbr_big)]     
        pc_nbr_small = pc_nbr[np.where(np.isin(pc_nbr[:,4], pad_small))]
        pc_nbr_small = pc_nbr_small[np.where(pc_nbr_small[:,3] >= energy_thresh_nbr_small)]
        pc_nbr = np.vstack((pc_nbr_big, pc_nbr_small))
        
        pads_to_drop = np.unique(pc_nbr[:,4])[np.where(np.array([list(pc_nbr[:,4]).count(i) for i in np.unique(pc_nbr[:,4])]) >= 5)]
        pc_nbr = pc_nbr[~np.isin(pc_nbr[:,4], pads_to_drop)]
        
        self.pc = np.vstack((pc_br, pc_nbr))
     
    def plot(self, view = '2D'):
        #clear_output()
        if (view == '3D'):
            fig = go.Figure(data=[go.Scatter3d(x=self.pc[:,0], y=self.pc[:,1], z=self.pc[:,2],
                                               mode='markers', 
                                               marker = dict(size = 2, color = self.pc[:,3], colorscale = 'viridis'))])
            # Forces all plots to have the same limits
            fig.update_layout(scene = dict(xaxis = dict(range = [-292, 292]),
                                           yaxis = dict(range = [-292, 292]),
                                           zaxis = dict(range = [0, 1000])))
            # Forces all axes to have the same scaling in the plots
            fig.update_layout(scene_aspectmode='manual', scene_aspectratio=dict(x=1, y=1, z=1000/292))
            fig.show()
        elif (view == '2D'):
            plt.figure(figsize = (9, 8))
            plt.scatter(self.pc[:,0], self.pc[:,1], c = self.pc[:,3], cmap = 'seismic', s = 2.5, alpha = 1)
            plt.xlim(-292, 292)
            plt.ylim(-292, 292)
            plt.colorbar(pad = 0)
            plt.grid()
            