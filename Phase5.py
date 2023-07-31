import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from TPCH5_utils import get_first_last_event_num, HDF5_LoadClouds
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import multiprocessing
import multiprocessing.pool
from multiprocessing import Pool, cpu_count

# PHASE 5 (Track fitting)

# Define constants
global C, amuev
C = 2.99792e8
amuev = 931.494028

def motionIVP(t, vec):
    '''
    Parameters:
        t      : Timesteps to evaluate the motion on.
        vec    : Input vector that contains the current state of the ODE (has structure [x, y, z, vx, vy, vz]).

    Returns:
        dvecdt : Output vector that contains the next step for the ODE (has structure [vx, vy, vz, dvx/dt, dvy/dt, dvz/dt]).
    '''
 
    #C = 2.99792e8
    #amuev = 931.494028
   
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

    #C = 2.99792e8
    #amuev = 931.494028

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
  
    obj = np.sum(np.array([np.min(np.linalg.norm(guess[:,:3]-subset[i,:3], axis = 1)) for i in range(len(subset))]))
    obj /= np.shape(subset)[0]
    return obj

def FunkyKongODE(params, subset, t):
    '''
    Parameters:
        params : The current estimate of each fit parameter.
        subset : All of the points in the point cloud associated with a given track_id which are being fitted to.

    Returns:
        obj    : The value fo the objective function between the data and the fit.
    '''
    obj = objective(SolveIVP(t, *params), subset)
    
    return obj

def Phase5(evt_num_array):

    params = np.loadtxt('/user/turi/PointCloud-utils/params.txt', dtype = str, delimiter = ':')
    PATH = params[0, 1]
    ntuple_PATH = params[1, 1]

    all_results_seg = []
    ntuple = pd.read_csv(ntuple_PATH, delimiter = ',')

    for event_num_i in tqdm(range(len(evt_num_array))):
        data = HDF5_LoadClouds(PATH, evt_num_array[event_num_i])
        ntuple_i = ntuple[ntuple['evt'] == evt_num_array[event_num_i]]
        results = np.hstack(np.array([ntuple_i['gpolar'],
                                      ntuple_i['gazimuth'],
                                      ntuple_i['gbrho'],
                                      ntuple_i['gxvert'],
                                      ntuple_i['gyvert'],
                                      ntuple_i['gzvert'],
                                      ntuple_i['direction']]))

        global ch, ma, q, m
        ch = int(ntuple_i['charge'])
        ma = int(ntuple_i['mass'])
        q = ch * 1.6021773349e-19
        m = ma * 1.660538782e-27

        if (ma == 0) or (ch == 0):
            all_results_seg.append(np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]))
            continue

        subset = data[data[:,6] == int(ntuple_i['track_id'])]

        global t
        t = np.arange(0, 1e-6, 1e-10)[:len(subset)]

        res = minimize(FunkyKongODE,
                       x0 = results[:6],
                       method = 'Nelder-Mead',
                       args = (subset, t),
                       bounds = ((0, 180),
                                 (0, 360),
                                 (0, 5),
                                 (results[3], results[3]),
                                 (results[4], results[4]),
                                 (results[5], results[5])),
                       options = {'maxiter':2000},
                       tol = 1e-3)

        all_results_seg.append(np.append(res.x, objective(SolveIVP(t, *res.x), subset)))

    return all_results_seg

if __name__ == '__main__':
    # Constants and conversions
    C = 2.99792e8 # Speed of light in m/s
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

    all_cores = cpu_count()
    evt_cores = 20

    if evt_cores > all_cores:
        raise ValueError('Number of cores used cannot exceed ', str(all_cores))

    params = np.loadtxt('params.txt', dtype = str, delimiter = ':')
    PATH = params[0, 1]
    ntuple_PATH = params[1, 1]

    first_event_num, last_event_num = get_first_last_event_num(PATH)
    print('First event number: ', first_event_num, '\nLast event num: ', last_event_num)

    dEdxSRIM = pd.read_csv('dEdx/Be10dpSRIM_Proton.txt', delimiter = ',')
    dEdx_interp = interp1d(dEdxSRIM['Ion Energy (MeV)'], dEdxSRIM['dE/dx (MeV/(mg/cm2))'])
 
    ntuple = pd.read_csv(ntuple_PATH, delimiter = ',')

    evt_parts = np.array_split(np.unique(ntuple['evt']), evt_cores)

    with Pool(evt_cores)as evt_p:
        run_parts = evt_p.map(Phase5, evt_parts)

    all_results = np.vstack(run_parts)

    if len(all_results) != len(ntuple):
        raise IndexError('Ntuple and additions are not the same length.')

    ntuple['fxvert'] = all_results[:,3]
    ntuple['fyvert'] = all_results[:,4]
    ntuple['fzvert'] = all_results[:,5]
    ntuple['fpolar'] = all_results[:,0]
    ntuple['fazimuth'] = all_results[:,1]
    ntuple['fbrho'] = all_results[:,2]
    ntuple['fenergy'] = amuev * (np.sqrt((ntuple['fbrho'] / 3.107 * ntuple['charge'] / ntuple['mass'])**2 + 1) - 1)
    ntuple['fobj'] = all_results[:,6]    

    ntuple.to_csv(ntuple_PATH, ',', index = False)
   
    print('Phase 5 finished successfully')
