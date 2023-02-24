import matplotlib.pyplot as plt
import numpy
import h5py

f = h5py.File('/mnt/analysis/e20009/e20009_Turi/run_0348.h5', 'r')
clouds = f['clouds']
event = clouds['evt147472_cloud']

#plt.scatter(event[:,0], event[:,1])
#plt.xlim([-292, 292])
#plt.ylim([-292, 292])
#plt.show()

fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(projection='3d')

ax.scatter(event[:,0], 
           event[:,1], 
           event[:,2], 
           c = event[:,5], s = 10)

ax.set_xlim([-292, 292])
ax.set_ylim([-292, 292])
ax.set_zlim([0, 1000])

plt.show()
