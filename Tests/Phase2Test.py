import matplotlib.pyplot as plt
import numpy as np
import h5py
import sys
sys.path.append('..')
from TPCH5_utils import HDF5_LoadClouds
from Phase2 import SmoothPC

if __name__ == '__main__':

    PATH = '/mnt/analysis/e20009/e20009_Turi/run_0348.h5'

    evt_ind = 147050

    smoothed = HDF5_LoadClouds(PATH, evt_ind)
    print(np.shape(smoothed))

    for i in np.unique(smoothed[:,6]):
        print(len(smoothed[smoothed[:,6] == i]))

    #for evt_ind in range(93947, 93948):

        #smoothed = HDF5_LoadClouds(PATH, evt_ind)

        #smoothed = SmoothPC(pc)
        #print(smoothed[np.isnan(smoothed).any(axis = 1)])
        #print(evt_ind, np.isnan(smoothed).any(axis = 0))

    fig = plt.figure(figsize = (12, 6))
    ax = fig.add_subplot(projection='3d')
    
    ax.scatter(smoothed[:,2], smoothed[:,1], smoothed[:,0], c = smoothed[:,6], s = 10)

    ax.set_box_aspect((1000/584, 1, 1))
    ax.set_xlim([0, 1000])
    ax.set_ylim([-292, 292])
    ax.set_zlim([-292, 292])
    ax.set_xlabel('Z')
    ax.set_ylabel('Y')
    ax.set_zlabel('X')
    ax.legend()

    plt.show()
