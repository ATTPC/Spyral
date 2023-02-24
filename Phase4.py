import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.cluster import DBSCAN
import h5py
import circle_fit # circle_fit found from: https://www.sciencedirect.com/science/article/pii/S0167947310004809?via%3Dihub
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from IPython.display import clear_output

# PHASE 4 (Track fitting)

def motionIVP(t, vec):
    # Takes a vector and returns a list containing the diff eq information according to the Lorenz force
    # In vector:  vec = [x, y, z, vx, vy, vz]
    # Out vector: vec = [vx, vy, vz, dvx/dt, dvy/dt, dvz/dt]
    
    x, y, z, vx, vy, vz = vec
    rr = np.sqrt(vx**2 + vy**2 + vz**2)
    azi = np.arctan2(vy, vx)
    pol = np.arccos(vz / rr)
    
    vv = np.sqrt(vx**2 + vy**2 + vz**2)
    E = amuev * (1/np.sqrt(1 - (vv / C)**2) - 1)
    if E < 0.001 or E > 50:
        return 1
    
    st = dEdx_interp(E) * 1000 # In MeV/(g/cm^2)
    st *= 1.6021773349e-13 # Converts to J/(g/cm^2)
    st *= dens*100 # Converts to kg*m/s^2
    st /= m # Converts to m/s^2
    dvecdt = [vx,
              vy,
              vz,
              (q/m)*(Efield[0] + vy*Bfield[2] - vz*Bfield[1]) - st*np.sin(pol)*np.cos(azi),
              (q/m)*(Efield[1] + vz*Bfield[0] - vx*Bfield[2]) - st*np.sin(pol)*np.sin(azi),
              (q/m)*(Efield[2] + vx*Bfield[1] - vy*Bfield[0]) - st*np.cos(pol)]
    return dvecdt

def SolveIVP(t, polar, azimuth, brho, xvert, yvert, zvert):
#def SolveIVP(t, polar, azimuth, energy, brho, xvert, yvert, zvert):
    if np.isnan(polar) and np.isnan(azimuth) and np.isnan(brho) and np.isnan(xvert) and np.isnan(yvert) and np.isnan(zvert):
        return 0
    energy = amuev * (np.sqrt((brho / 3.107 * ch / ma)**2 + 1) - 1)
    gamma = energy / amuev + 1
    beta = np.sqrt(1 - 1/gamma**2)

    y0 = [xvert/1000,
          yvert/1000,
          zvert/1000,
          beta*C*np.sin(polar*np.pi/180)*np.cos(azimuth*np.pi/180),
          beta*C*np.sin(polar*np.pi/180)*np.sin(azimuth*np.pi/180),
          beta*C*np.cos(polar*np.pi/180)]
    
    sol = solve_ivp(motionIVP, 
                    t_span = [0, 1e-6], 
                    y0 = y0, 
                    t_eval = t, 
                    method='RK45', 
                    max_step=0.1)
    
    sol.y[:3] *= 1000 # Converts positions to mm
    
    return sol.y.T[:,:3]

def objective(guess):        
    obj = np.sum(np.array([np.min(np.sqrt((subset[i, 0] - guess[:,0])**2 + (subset[i, 1] - guess[:,1])**2 + (subset[i, 2] - guess[:,2])**2)) for i in range(len(subset))]))
    obj /= np.shape(data)[0]
    
    return obj

def FunkyKongODE(params):
    obj = objective(SolveIVP(t, *params))
    
    return obj

def Phase4():
    return 0

if __name__ == '__main__':
    
    print('Phase 4 finished successfully')
