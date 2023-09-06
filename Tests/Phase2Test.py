import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
sys.path.append('..')
from pcutils.core.point_cloud import PointCloud
from pcutils.hdf.TPCH5_utils import get_first_last_event_num

def PlotClusters(cluster_Path, event_num):
    cluster_file = h5py.File(cluster_Path, 'r')
    cluster_group = cluster_file.get('cluster')

    try:
        event = cluster_group[f'event_{event_num}']
    except:
        pass

    nclusters = event.attrs['nclusters']
    print(f'Event number: {event_num}')
    print(f'Number of clusters: {nclusters}')

    fig = plt.figure(figsize = (8, 6))
    ax = fig.add_subplot(projection = '3d')

    for cluster_i in range(nclusters):
        try:
            local_cluster = event[f'cluster_{cluster_i}']
            local_cloud = local_cluster['cloud']
            if len(local_cloud) < 10:
                continue
            print(f'Cluster {cluster_i} has {len(local_cloud)} points')
            ax.scatter(local_cloud[:,2], local_cloud[:,0], local_cloud[:,1], s = 5, label = cluster_i)
        except:
            print(f'Cluster {cluster_i} has 0 points')
            continue
        #print(f'Cluster {cluster_i} has {len(local_cloud)} points')
        #local_cloud = local_cluster['cloud']
        #if len(local_cloud) < 50:
            #continue
        #ax.scatter(local_cloud[:,2], local_cloud[:,0], local_cloud[:,1], s = 5, c = cluster_i*np.ones(len(local_cloud)), label = cluster_i)

    ax.set_box_aspect((1000/584, 1, 1))
    ax.set_xlim([0, 1000])
    ax.set_ylim([-292, 292])
    ax.set_zlim([-292, 292])
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.legend()
    plt.title(event_num)
    plt.show()
        

def main():
    cluster_Path = '/mnt/analysis/e20009/e20009_Turi/Workspace/clusters/run_0348.h5'
    event_num = np.random.randint(low = 146663, high = 228636)
    #event_num = 154522
    PlotClusters(cluster_Path = cluster_Path, event_num = event_num)

if __name__ == '__main__':
    main()
    print('Done!')
