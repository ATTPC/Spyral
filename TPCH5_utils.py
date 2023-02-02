import matplotlib.pyplot as plt
import numpy as np
import h5py
import pandas as pd

def get_first_last_event_num(PATH):
	'''
	Inputs:
		PATH            : Path to a specific HDF5 file.
	
	Returns:
		first_event_num : An integer that specifies the first event number in the HDF5 file.
		last_event_num  : An integer that specifies the last event number in the HDF5 file.
	'''
	f = h5py.File(PATH, 'r')
	
	meta = f['/meta'].get('meta')

	first_event_num = int(meta[0])
	last_event_num = int(meta[2])

	f.close()
	return first_event_num, last_event_num

def load_trace(PATH, event_num = 0):
	'''
	Inputs:
		PATH      : Path to a specified HDF5 file.
		event_num : The event number that you want to look at.

	Returns:
        meta      : A 2D array that contains the metadata for each trace (CoBo, AsAd, AGET, channel, pad number)
		trace     : A 2D array that contains the every trace for the specified event.
	'''
	f = h5py.File(PATH, 'r')
	
	events = f['/get']
	
	#first_event_num, last_event_num = get_first_last_event_num(PATH)
	#event_num = np.random.randint(low = first_event_num, high = last_event_num+1)

	dataset = events.get('evt'+str(event_num)+'_data')
	header = events.get('evt'+str(event_num)+'_header')
	
	first_event_num, last_event_num = get_first_last_event_num(PATH)

	if (event_num < first_event_num) or (event_num > last_event_num):
		raise Exception("Event number must be between "+str(first_event_num)+" and "+str(last_event_num)+" (default is 0)")

	CoBo = dataset[:, 0]
	AsAd = dataset[:, 1]
	AGET = dataset[:, 2]
	channel = dataset[:, 3]
	pad_num = dataset[:, 4]
	
	meta = np.transpose(np.array([CoBo, AsAd, AGET, channel, pad_num]))

	trace = dataset[:, 5:]
	
	return meta.astype(np.int64), trace.astype(np.int64)

def HDF5_LoadClouds(PATH, event_ind):
    f = h5py.File(PATH, 'r')
    meta = f['meta/meta']
    #print('First event: ', int(meta[0]), '\n Last event: ', int(meta[2]))
    if ((int(event_ind) >= int(meta[0])) and (int(event_ind) <= int(meta[2]))):
        cloud = f['/clouds'].get('evt'+str(int(event_ind))+'_cloud')[:,:]
    else:
        print('Invalid event number.', event_ind, ' must be between ', int(meta[0]), ' and ', int(meta[2]))
        cloud = 0
    f.close()
    return cloud

def main():
	PATH = '/mnt/research/attpc/e20009/h5/run_0231.h5'
	
	first_event_num, last_event_num = get_first_last_event_num(PATH)	

	meta, trace = load_trace(PATH, first_event_num)

	print(np.shape(meta))
	print(np.shape(trace))
	
if __name__ == '__main__':
	main()
	print('Done!')
