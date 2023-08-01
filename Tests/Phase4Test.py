import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, rgb2hex
import sys
sys.path.append('..')
from pcutils.phases.Phase4 import *
from pcutils.hdf.TPCH5_utils import load_trace

# PHASE 4 (PID)

cmap = plt.cm.get_cmap("jet")

white_jet = LinearSegmentedColormap.from_list('white_jet', [
    (0, '#ffffff'),
    (1e-20, rgb2hex(cmap(1e-20))),
    (0.2, rgb2hex(cmap(0.2))),
    (0.4, rgb2hex(cmap(0.4))),
    (0.6, rgb2hex(cmap(0.6))),
    (0.8, rgb2hex(cmap(0.8))),
    (1, rgb2hex(cmap(0.9))),  
], N = 256)

if __name__ == '__main__':
    params = np.loadtxt('../params.txt', dtype = str, delimiter = ':')
    PATH = params[0, 1]
    ntuple_PATH = params[1, 1]

    #ntuple = pd.read_csv(ntuple_PATH, delimiter = ',')

    #evt_i = np.unique(ntuple['evt'])[0]

    #print(evt_i)

    meta, traces = load_trace(PATH, 147050)

    trace = traces[np.logical_and(np.logical_and(meta[:,0] == 10, meta[:,2] == 3), meta[:,3] == 34)][0]

    plt.plot(np.gradient(trace[1:-1]))
    plt.show()
