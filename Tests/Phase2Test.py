import matplotlib.pyplot as plt
import numpy as np
import h5py
import sys
sys.path.append('..')
from TPCH5_utils import HDF5_LoadClouds
from Phase2 import SmoothPC

if __name__ == '__main__':

    PATH = '/mnt/analysis/e20009/e20009_Turi/run_0348.h5'

    evt_ind = 147472

    smoothed = HDF5_LoadClouds(PATH, evt_ind)

    for i in np.unique(smoothed[:,6]):
        print(len(smoothed[smoothed[:,6] == i]))

    #for evt_ind in range(93947, 93948):

        #smoothed = HDF5_LoadClouds(PATH, evt_ind)

        #smoothed = SmoothPC(pc)
        #print(smoothed[np.isnan(smoothed).any(axis = 1)])
        #print(evt_ind, np.isnan(smoothed).any(axis = 0))

    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(smoothed[:,0], smoothed[:,1], smoothed[:,2], c = smoothed[:,6], s = 10)

    ax.set_xlim([-292, 292])
    ax.set_ylim([-292, 292])
    ax.set_zlim([0, 1000])

    plt.show()
