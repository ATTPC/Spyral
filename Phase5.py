import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from TPCH5_utils import HDF5_LoadClouds
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import multiprocessing
import multiprocessing.pool
from multiprocessing import Pool, cpu_count

# PHASE 4 (Track fitting)

def motionIVP(t, vec):
    '''
    Parameters:
        t      : Timesteps to evaluate the motion on.
        vec    : Input vector that contains the current state of the ODE (has structure [x, y, z, vx, vy, vz]).

    Returns:
        dvecdt : Output vector that contains the next step for the ODE (has structure [vx, vy, vz, dvx/dt, dvy/dt, dvz/dt]).
    '''
    
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
    '''
    Parameters:
        t             : Timesteps to evaluate the motion on.
        polar         : Polar angle of the outgoing particle.
        azimuth       : Azimuthal angle of the outgoing particle.
        brho          : BRho of the outgoing particle.
        xvert         : X-coordinate of the reaction vertex.
        yvert         : Y-coordinate of the reaction vertex.
        zvert         : Z-coordinate of the reaction vertex.

    Returns:
        sol.y.T[:,:3] : Solution to the IVP where each column corresponds to the x-y-z positions of the particle respectively.
    '''

    if np.isnan(polar) or np.isnan(azimuth) or np.isnan(brho) or np.isnan(xvert) or np.isnan(yvert) or np.isnan(zvert):
        return 0
    energy = amuev * (np.sqrt((brho / 3.107 * ch / ma)**2 + 1) - 1) # Calculates the energy of the track in eV
    gamma = energy / amuev + 1 # Calculates the relativistic gamma factor
    beta = np.sqrt(1 - 1/gamma**2) # Calculates the relativistic beta factor

    y0 = [xvert/1000, # Converts x-vertex position to m
          yvert/1000, # Converts y-vertex position to m
          zvert/1000, # Converts z-vertex position to m
          beta*C*np.sin(polar*np.pi/180)*np.cos(azimuth*np.pi/180), # Calculates the x-velocity in m/s
          beta*C*np.sin(polar*np.pi/180)*np.sin(azimuth*np.pi/180), # Calculates the y-velocity in m/s
          beta*C*np.cos(polar*np.pi/180)] # Calculates the z-velocity in m/s
    
    sol = solve_ivp(motionIVP, 
                    t_span = [0, 1e-6], 
                    y0 = y0, 
                    t_eval = t, 
                    method='RK45', 
                    max_step=0.1)
    
    sol.y[:3] *= 1000 # Converts positions to mm
    
    return sol.y.T[:,:3]

def objective(guess, subset):
    '''
    Parameters:
        guess  : The solution to the ODE with the given fit parameters.
        subset : All of the points in the point cloud associated with a given track_id which are being fitted to.

    Returns:
        obj    : The value of the objective function between the data and fit.
    '''
  
    obj = np.sum(np.array([np.min(np.linalg.norm(guess[:,:3]-subset[i, :3], axis = 1)) for i in range(len(subset))]))
    obj /= np.shape(subset)[0]
    return obj

def FunkyKongODE(params, subset):
    '''
    Parameters:
        params : The current estimate of each fit parameter.
        subset : All of the points in the point cloud associated with a given track_id which are being fitted to.

    Returns:
        obj    : The value fo the objective function between the data and the fit.
    '''
    obj = objective(SolveIVP(t, *params), subset)
    
    return obj

def Phase4(evt_num_array):
    all_results_seg = []

    for event_num_i in tqdm(range(len(evt_num_array))):
        event_num = evt_num_array[event_num_i]

        ntuple_sub = ntuple[ntuple['evt'] == event_num].reset_index(drop = True)

        data = HDF5_LoadClouds(PATH, event_num)

        for track_id_i in range(len(ntuple_sub['track_id'])):

            if (ntuple_sub['gpolar'][track_id_i] != max(ntuple_sub['gpolar'])):
                all_results_seg.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
                continue

            track_id = ntuple_sub['track_id'][track_id_i]
            results = np.array([ntuple_sub['gpolar'][track_id_i],
                                ntuple_sub['gazimuth'][track_id_i],
                                ntuple_sub['gbrho'][track_id_i],
                                ntuple_sub['gxvert'][track_id_i],
                                ntuple_sub['gyvert'][track_id_i],
                                ntuple_sub['gzvert'][track_id_i],
                                ntuple_sub['direction'][track_id_i]])

            subset = data[data[:,5] == track_id]
            global t
            t = np.arange(0, 1e-6, 1e-10)
            t = t[:len(subset)]

            res = minimize(FunkyKongODE, 
               x0 = results[:-1], 
               method = 'Nelder-Mead',
               args = (subset),
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
    # Constants and conversions
    C = 2.99792E8 # Speed of light in m/s
    amuev = 931.494028 # Conversion from amu to eV

    # Experiment set-up specific info
    tilt = 0
    Emag = 60000  # E field in V/m
    Efield = [0, 0, -Emag]
    Bmag = 2.991 # B field in T
    Bfield = [0, -Bmag*np.sin(tilt*np.pi/180), -Bmag*np.cos(tilt*np.pi/180)]
    dens = 0.00013136 # Density of the gas in g/cm^3

    micromegas = 66.0045 # Time bucket of the micromega edge
    window = 399.455 # Time bucket of the window edge
    length = 1000 # Length of the detector in mm

    ch = -1
    ma = 1
    q = ch * 1.6021773349e-19
    m = ma * 1.660538782e-27

    all_cores = cpu_count()
    evt_cores = 40

    if evt_cores > all_cores:
        raise ValueError('Number of cores used cannot exceed ', str(all_cores))

    PATH = '/mnt/analysis/e20009/e20009_Turi/run_0348.h5'
    ntuple_PATH = 'all_ntuple_run0348_Turi.txt'
    #PATH = '/mnt/analysis/e20009/e20009_Turi/Be10dp178.h5'
    #ntuple_PATH = 'all_ntuple_Be10dp178_Turi.txt'

    dEdxSRIM = pd.read_csv('dEdx/Be10dpSRIM_Proton.txt', delimiter = ',')
    dEdx_interp = interp1d(dEdxSRIM['Ion Energy (MeV)'], dEdxSRIM['dE/dx (MeV/(mg/cm2))'])
 
    ntuple = pd.read_csv(ntuple_PATH, delimiter = ',')

    evt_parts = np.array_split(np.unique(ntuple['evt']), evt_cores)

    with Pool(evt_cores)as evt_p:
        run_parts = evt_p.map(Phase4, evt_parts)

    all_results = np.vstack(run_parts)

    if len(all_results) != len(ntuple):
        raise IndexError('Ntuple and additions are not the same length.')

    ntuple['fxvert'] = all_results[:,3]
    ntuple['fyvert'] = all_results[:,4]
    ntuple['fzvert'] = all_results[:,5]
    ntuple['fpolar'] = all_results[:,0]
    ntuple['fazimuth'] = all_results[:,1]
    ntuple['fbrho'] = all_results[:,2]

    ntuple.to_csv(ntuple_PATH, ',', index = False)
   
    print('Phase 4 finished successfully')
