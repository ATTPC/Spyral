import numpy as np
import pandas as pd
import sys
sys.path.append('..')
from Phase5 import *

def Phase5(evt_num_array):

    all_results_seg = []
    ntuple = pd.read_csv(ntuple_PATH, delimiter = ',')

    for event_num_i in tqdm(range(len(evt_num_array))):
        data = HDF5_LoadClouds(PATH, evt_num_array[event_num_i])
        ntuple_i = ntuple[ntuple['evt'] == evt_num_array[event_num_i]]
        results = np.hstack(np.array([ntuple_i['gpolar'],
                                      ntuple_i['gazimuth'],
                                      ntuple_i['gbrho'],
                                      ntuple_i['gxvert'],
                                      ntuple_i['gyvert'],
                                      ntuple_i['gzvert'],
                                      ntuple_i['direction']]))

        ch = int(ntuple_i['charge'])
        ma = int(ntuple_i['mass'])
        q = ch * 1.6021773349e-19
        m = ma * 1.660538782e-27

        subset = data[data[:,6] == int(ntuple_i['track_id'])]

        global t
        t = np.arange(0, 1e-6, 1e-10)[:len(subset)]

        res = minimize(FunkyKongODE,
                       x0 = results[:6],
                       method = 'Nelder-Mead',
                       args = (subset, t),
                       bounds = ((0, 180),
                                 (0, 360),
                                 (0, 5),
                                 (results[3], results[3]),
                                 (results[4], results[4]),
                                 (results[5], results[5])),
                       options = {'maxiter':2000},
                       tol = 1e-3)

        all_results_seg.append(res.x)

    return all_results_seg

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
