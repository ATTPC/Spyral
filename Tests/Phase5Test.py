import numpy as np
import pandas as pd
import sys
sys.path.append('..')
from Phase5 import *

if __name__ == '__main__':

    C = 2.99792E8
    amuev = 931.494028

    tilt = 0
    Emag = 60000
    Efield = [0, 0, -Emag]
    Bmag = 2.991
    Bfield = [0, -Bmag*np.sin(tilt*np.pi/180), -Bmag*np.sin(tilt*np.pi/180)]
    dens = 0.00013136

    micromegas = 66.0045
    window = 399.455
    length = 1000

    global PATH, ntuple_PATH

    params = np.loadtxt('../params.txt', dtype = str, delimiter = ':')
    PATH = params[0, 1]
    ntuple_PATH = params[1, 1]

    ntuple = pd.read_csv(ntuple_PATH, delimiter = ',')
    evts = np.unique(ntuple['evt'])[:4]
    results = Phase5(evts)

    print('Phase 5 test finished successfully!')
