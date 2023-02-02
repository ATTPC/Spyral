import matplotlib.pyplot as plt
import numpy
import h5py

f = h5py.File('/mnt/analysis/e20009/e20009_Turi/run_0348.h5', 'r')
clouds = f['clouds']
event = clouds['evt147472_cloud']

plt.scatter(event[:,0], event[:,1])
plt.xlim([-292, 292])
plt.ylim([-292, 292])
plt.show()
