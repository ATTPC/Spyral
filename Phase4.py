import pandas as pd
import numpy as np
from TPCH5_utils import load_trace, get_first_last_event_num
import time
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
from tqdm import tqdm
from scipy.signal import find_peaks

def FindBeamEvent(PATH, evt_ind):
    meta, traces = load_trace(PATH, evt_ind)
    # Checks CoBo 10, AGET 3, channel 34 for a peak. If there is a peak, it is a beam event.
    trace = traces[np.logical_and(np.logical_and(meta[:,0] == 10, meta[:,2] == 3), meta[:,3] == 34)][0]
    if len(find_peaks(np.gradient(trace[1:-1]), height = 50)[0]) != 0:
        return True # Bem event
    else:
        return False

def compile_ntuple(all_ntuples):
    trimmed_ntuple = pd.DataFrame(columns = ['evt', 'track_id', 'gxvert', 'gyvert', 'gzvert', 'gpolar', 'gazimuth', 'gbrho', 'direction', 'dEdx', 'deavg'])
    
    for ntuple_i in all_ntuples:
        for evt_i in tqdm(np.unique(ntuple_i['evt'])):
            if FindBeamEvent(PATH, int(evt_i)):
                continue
            sub = ntuple_i[ntuple_i['evt'] == evt_i]
            trimmed_ntuple = trimmed_ntuple.append(sub.loc[abs(90-sub['gbrho']) == min(abs(90-sub['gbrho']))])
    trimmed_ntuple.reset_index(inplace = True, drop = True)
    #trimmed_ntuple = trimmed_ntuple[np.logical_and(trimmed_ntuple['gpolar'] >= 5, trimmed_ntuple['gpolar'] <= 175)]
    trimmed_ntuple.reset_index(inplace = True, drop = True)
    return trimmed_ntuple

def DrawGate(xdata, ydata, xlim = [0, 3000], ylim = [0, 3], gates = None):
    if gates != None:
        if type(gates) != list:
            raise TypeError('Gates must be passed as a list!')
        for gate in gates:
            ax.plot(gate[:,0], gate[:,1], 'k-')
    
    fig, ax = plt.subplots(figsize = (12, 10))
    pts = ax.scatter(xdata, ydata, s = 0.25)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.grid()

    class DrawPolygon:
        def __init__(self, ax, collection):
            self.canvas = ax.figure.canvas
            self.collection = collection
            self.poly = PolygonSelector(ax, self.onselect)

            self.verts = np.array([])

        def onselect(self, verts):
            self.verts = np.array([list(i) for i in verts])
            self.verts = np.vstack([self.verts, self.verts[0]])

    poly = DrawPolygon(ax, pts)

    print("Click on the figure to create a polygon.")
    print("Press the 'esc' key to start a new polygon.")
    print("Try holding the 'shift' key to move all of the vertices.")
    print("Try holding the 'ctrl' key to move a single vertex.")
    return poly

def PID(ntuple):
    m1 = 1
    m2 = 2
    m3 = 3
    m4 = 4

    vert1 = np.array([[3000, 0.1], [0, 0.1], [5 , 2], [100, 1.5], [250, 1], [500, 0.6], [1000, 0.45], [1500, 0.35], [3000, 0.1]])

    poly1 = Path(vert1)
    yesno1 = poly1.contains_points(np.array(trimmed_ntuple[['dEdx', 'gbrho']])).astype(int) * m1

    vert2 = np.array([[2500, 0.5], [1000, 0.55], [500, 0.7], [300, 1], [200, 1.2], [100, 1.7], [50, 2], [5, 4], [300, 2.5], [500, 1.75], [1000, 1.2], [1500, 1], [2500, 0.6]])

    poly2 = Path(vert2)
    yesno2 = poly2.contains_points(np.array(trimmed_ntuple[['dEdx', 'gbrho']])).astype(int) * m2

    yesno = yesno1 + yesno2

    if len(yesno) == len(ntuple):
        ntuple['mass'] = yesno
    else:
        raise Exception('Error occurred with PID')

if __name__ == '__main__':
    global PATH, ntuple_PATH

    params = np.loadtxt('params.txt', dtype = str, delimiter = ':')
    PATH = params[0, 1]
    ntuple_PATH = params[1, 1]

    ntuple = pd.read_csv(ntuple_PATH, delimiter = ',')

    print(len(ntuple))

    all_ntuples = [ntuple]
    trimmed_ntuple = compile_ntuple(all_ntuples)
    #PID(trimmed_ntuple)

    print(len(trimmed_ntuple))
 
    trimmed_ntuple.to_csv(ntuple_PATH, sep = ',', index = False)

    print('Phase 4 finished successfully')
