import numpy as np
import h5py

if __name__ == '__main__':
    PATH = '/mnt/analysis/e20009/e20009_Turi/Be10dp178.h5'

    f = h5py.File(PATH, 'r+')
    #clouds = f['clouds']
    clouds = f['clouds']
    labels = list(clouds.keys())
    print(labels)
    f.close()
