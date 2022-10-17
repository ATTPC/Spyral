import numpy as np
import h5py

if __name__ == '__main__':
	PATH = '/mnt/analysis/e20009/e20009_Turi/run_0231.h5'

	f = h5py.File(PATH, 'r+')
	#print(list(f['clouds'].keys()))
	clouds = f['clouds']
	print(np.shape(clouds.get('evt9999_cloud')))
	print(list(clouds.keys()))
	'''
	try:
		f.create_group('bingbong')
	except ValueError:
		print('Group already exists.')
	
	try:
		f['bingbong'].create_dataset('a', data = np.random.random((10, 10, 10)))
	except OSError:
		print('Dataset already exists.')
		del f['bingbong/a']
		f['bingbong'].create_dataset('a', data = np.random.random((2, 2, 2)))

	print(np.shape(f['bingbong/a']))	
	'''
	f.close()
