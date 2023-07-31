import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



if __name__ == '__main__':
    params = np.loadtxt('../params.txt', dtype = str, delimiter = ':')
    PATH = params[0, 1]
    #ntuple_PATH = params[1, 1]
    ntuple_PATH = '../ntuple_testing/Be10dp_12/all_ntuple_run0348_Turi.txt'

    ntuple = pd.read_csv(ntuple_PATH, delimiter = ',')

    #ntuple = ntuple[ntuple['mass'] == 1]

    plt.figure(figsize = (8, 6))
    plt.scatter(ntuple['dEdx'], ntuple['fbrho'], s = 0.25)
    plt.xlim(0, 3000)
    plt.ylim(0, 3)
    plt.grid()
    plt.show()
