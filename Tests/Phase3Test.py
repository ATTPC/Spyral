import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import sys
sys.path.append('..')
from TPCH5_utils import get_first_last_event_num, HDF5_LoadClouds
from Phase3 import make_circle, dist_func, SimpleAnalysis
import h5py
import circle_fit
from scipy.signal import find_peaks
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import interp1d
import multiprocessing
import multiprocessing.pool
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

import matplotlib.pyplot as plt

# PHASE 3 (simple track analysis)

if __name__ == '__main__':
    '''
    global Bmag
    Bmag = 2.991

    PATH = '/mnt/analysis/e20009/e20009_Turi/run_0348.h5'

    # First event num: 146686, Last event num: 228636

    first, last = get_first_last_event_num(PATH)

    for event_num in (range(first+1, last)): 
        data = HDF5_LoadClouds(PATH, event_num)
        if type(data) == int:
            continue
        #print(np.unique(data[:,5]))
        for track_id in np.unique(data[:,6]):
            subset = data[data[:,6] == track_id]
            #results = SimpleAnalysis(data, track_id)
            print(event_num, track_id)
    '''
    params = np.loadtxt('../params.txt', dtype = str, delimiter = ':')
    PATH = params[0, 1]
    ntuple_PATH = params[1, 1]

    ntuple = pd.read_csv(ntuple_PATH, delimiter = ',')

    plt.figure(figsize = (12, 8))
    #plt.scatter(ntuple['dEdx'], ntuple['gbrho'], s = 0.25)
    plt.scatter(ntuple['dEdx'], ntuple['gbrho'], s = 0.25)
    plt.grid()
    plt.xlim(0, 5000)
    plt.ylim(0, 2)
    plt.show()

    print('Simple analysis successful')
