import numpy as np
from scipy.integrate import solve_ivp
import h5py as h5
from ..core.target import Target
from ..core.nuclear_data import NucleusData
from dataclasses import dataclass
from pathlib import Path
from scipy import constants
from ..interpolate import BilinearInterpolator, LinearInterpolator
from scipy.interpolate import CubicSpline
from ..core.constants import MEV_2_JOULE, MEV_2_KG
import math

from numba import float64, int32
from numba.extending import as_numba_type
from numba.types import string, ListType
from numba.typed import List
from numba.experimental import jitclass

TIME_WINDOW: float = 1.0e-6 #1us window
QBRHO_2_P: float = 1.0e-9 * constants.speed_of_light #kG * cm -> MeV
SAMPLING_PERIOD: float = 2.0e-9 # seconds, converts time bucket interval to time
SAMPLING_RANGE: np.ndarray = np.arange(0., TIME_WINDOW, SAMPLING_PERIOD)
PRECISION: float = 2.0e-6
DEG2RAD: float = np.pi / 180.0

@dataclass
class InitialState:
    '''
    Wrapper for the initial state used to generate a trajectory
    '''
    vertex_x: float = 0.0 #m
    vertex_y: float = 0.0
    vertex_z: float = 0.0
    polar: float = 0.0 #rad
    azimuthal: float = 0.0
    kinetic_energy: float = 0.0 # MeV

    def to_array(self) -> np.ndarray:
        np.array([self.vertex_x, self.vertex_y, self.vertex_z, self.polar, self.azimuthal, self.kinetic_energy])

@dataclass
class GeneratorParams:
    '''
    Wrapper for the general parameters required to generate an interpolation scheme
    '''
    target: Target
    particle: NucleusData
    bfield: float #T
    efield: float #V
    ke_min: float #MeV
    ke_max: float #MeV
    ke_bins: int 
    polar_min: float #deg
    polar_max: float #deg
    polar_bins: int

#State = [x, y, z, vx, vy, vz]
#Derivative = [vx, vy, vz, ax, ay, az] (returns)
def equation_of_motion(t: float, state: np.ndarray, Bfield: float, Efield: float, target: Target, ejectile: NucleusData) -> np.ndarray:
    '''
    The equations of motion for a charged particle in a static electromagnetic field which experiences energy loss through some material.

    ## Parameters
    t: float, time step
    state: ndarray, the state of the particle (x,y,z,vx,vy,vz)
    Bfield: float, the magnitude of the magnetic field
    Efield: float, the magnitude of the electric field
    target: Target, the material through which the particle travels
    ejectile: NucleusData, data on the particle

    ## Returns
    ndarray: the derivatives of the state
    '''

    speed = math.sqrt(state[3]**2.0 + state[4]**2.0 + state[5]**2.0)
    unit_vector = state[3:] / speed # direction
    kinetic_energy = ejectile.mass * (1.0/math.sqrt(1.0 - (speed / constants.speed_of_light)**2.0) - 1.0) #MeV
    if kinetic_energy < PRECISION:
        return np.zeros(6)
    mass_kg = ejectile.mass * MEV_2_KG
    charge_c = ejectile.Z * constants.elementary_charge
    qm = charge_c/mass_kg

    deceleration = (target.get_dedx(ejectile, kinetic_energy) * MEV_2_JOULE * target.density * 100.0) / mass_kg
    results = np.zeros(6)
    results[0] = state[3]
    results[1] = state[4]
    results[2] = state[5]
    results[3] = qm * state[4] * Bfield - deceleration * unit_vector[0]
    results[4] = qm * (-1.0 * state[3] * Bfield) - deceleration * unit_vector[1]
    results[5] = qm * Efield - deceleration * unit_vector[2]

    return results

def jacobian(t, state: np.ndarray, Bfield: float, Efield: float, target: Target, ejectile: NucleusData) -> np.ndarray:
    '''
    Computes the jacobian of the charged particle in a static electromagnetic field which experiences energy loss through some material

    ## Parameters
    t: float, time step
    state: ndarray, the state of the particle (x,y,z,vx,vy,vz)
    Bfield: float, the magnitude of the magnetic field
    Efield: float, the magnitude of the electric field
    target: Target, the material through which the particle travels
    ejectile: NucleusData, data on the particle

    ## Returns
    ndarray: the jacobian
    '''
    jac = np.zeros((len(state), len(state)))
    mass_kg = ejectile.mass * MEV_2_KG
    charge_c = ejectile.Z * constants.elementary_charge
    qm = charge_c/mass_kg
    jac[0, 3] = 1.0
    jac[1, 4] = 1.0
    jac[2, 5] = 1.0
    jac[3, 4] = qm * Bfield
    jac[4, 3] = -1.0 * qm * Bfield

    return jac

def check_tracks_exist(trackpath: Path) -> bool:
    '''
    Simple file-existance checker with a nice print message
    '''
    if trackpath.exists():
        print(f'Track file {trackpath} detected, this file will be used. If new tracks are needed, please delete the original file.')
        return True
    else:
        return False

def generate_tracks(params: GeneratorParams, trackpath: Path):
    '''
    Generate a set of tracks given some parameters and write them to an h5 file at a path.
    
    ## Parameters
    params: GeneratorParams, parameters which control the tracks
    trackpath: Path, where to write the tracks to
    '''
    kes = np.linspace(params.ke_min, params.ke_max, params.ke_bins)
    polars = np.linspace(params.polar_min * DEG2RAD, params.polar_max * DEG2RAD, params.polar_bins)

    track_file = h5.File(trackpath, 'w')
    track_group: h5.Group = track_file.create_group('tracks')
    track_group.attrs['particle'] = params.particle.isotopic_symbol
    track_group.attrs['gas'] = params.target.pretty_string
    track_group.attrs['bfield'] = params.bfield
    track_group.attrs['efield'] = params.efield
    track_group.attrs['ke_min'] = params.ke_min
    track_group.attrs['ke_max'] = params.ke_max
    track_group.attrs['ke_bins'] = params.ke_bins
    track_group.attrs['polar_min'] = params.polar_min
    track_group.attrs['polar_max'] = params.polar_max
    track_group.attrs['polar_bins'] = params.polar_bins
    data = track_group.create_dataset('data', shape=(len(SAMPLING_RANGE), 3,  params.ke_bins, params.polar_bins), dtype=np.float64)  #time x (x, y, z, vx, vy, vz) x ke x polar

    initial_state = np.zeros(6)

    bfield = -1.0 * params.bfield
    efield = -1.0 * params.efield

    total_iters = params.polar_bins * params.ke_bins
    count = 0
    flush_percent = 0.01
    flush_val = flush_percent * total_iters
    flush_count = 0

    for eidx, e in enumerate(kes):
        for pidx, p in enumerate(polars):

            count += 1
            if count == flush_val:
                count = 0
                flush_count += 1
                print(f'\rPercent of data generated: {int(flush_count * flush_percent * 100)}%', end='')
            
            initial_state[:] = 0.0
            momentum = math.sqrt(e * (e + 2.0*params.particle.mass))
            speed = momentum / params.particle.mass * constants.speed_of_light
            initial_state[3] = speed * math.sin(p)
            initial_state[5] = speed * math.cos(p)
            result = solve_ivp(equation_of_motion, (0.0, TIME_WINDOW), initial_state, method='BDF', args=(bfield, efield, params.target, params.particle), t_eval=SAMPLING_RANGE, jac=jacobian)
            data[:, :, eidx, pidx] = result.y.T[:, :3]

    print('\n')

@jitclass([
    ('filepath', string), 
    ('particle_name', string), 
    ('gas_name', string), 
    ('bfield', float64), 
    ('efield', float64), 
    ('ke_min', float64), 
    ('ke_max', float64), 
    ('ke_bins', int32), 
    ('polar_min', float64), 
    ('polar_max', float64), 
    ('polar_bins', int32), 
    ('interpolators', ListType(BilinearInterpolator.class_type.instance_type))
])
class TrackInterpolator:
    '''
    # TrackInterpolator
    Represents an interpolation scheme used to generate trajectories. Solving ODE's can be expensive,
    so to save time pre-generate a range of solutions and then interpolate on these solutions. TrackInterpolator 
    uses bilinear interpolation to interpolate on the energy and polar angle (reaction angle) of the trajectory.
    '''
    def __init__(self, track_path: str, interpolators: ListType(BilinearInterpolator.class_type.instance_type), particle_name: str, gas_name: str, bfield: float, efield: float, ke_min: float, ke_max: float, ke_bins: int, polar_min: float, polar_max: float, polar_bins: int):
        self.filepath = track_path
        self.particle_name: str = particle_name
        self.gas_name: str = gas_name
        self.bfield: float = bfield
        self.efield: float = efield
        self.ke_min: float = ke_min
        self.ke_max: float = ke_max
        self.ke_bins: int = ke_bins
        self.polar_min: float = polar_min
        self.polar_max: float = polar_max
        self.polar_bins: int = polar_bins

        self.interpolators = interpolators

    def get_interpolated_trajectory(self, vx: float, vy: float, vz: float, polar: float, azim: float, ke: float) -> CubicSpline | None:
        '''
        Get an interpolated trajectory given some initial state.

        ## Parameters: 
        initial_state: InitialState

        ## Returns
        CubicSpline | None: Returns a CubicSpline, which interpolates the trajectory upon z for x,y or None when the algorithm fails
        '''

        is_backwards = False
        if polar > np.pi*0.5:
            is_backwards = True
            polar -= np.pi*0.5

        trajectory = np.zeros((len(self.interpolators), 3))
        for idx, _ in enumerate(trajectory):
            trajectory[idx] = self.interpolators[idx].interpolate(polar, ke)

        #Rotate polar by 90 degrees if nescessary
        #Since we rotate the entire coordinate system, is as simple as flip in z
        #Order here is important! Must be polar, azimuthal, translate!
        if is_backwards:
            trajectory[:, 2] *= -1
        #Rotate the trajectory in azimuthal (around z) to match data
        z_rot = np.array([[np.cos(azim), -np.sin(azim), 0.0], [np.sin(azim), np.cos(azim), 0.0], [0., 0., 1.0]])
        trajectory = (z_rot @ trajectory.T).T
        #Translate to vertex
        trajectory[:, 0] += vx
        trajectory[:, 1] += vy
        trajectory[:, 2] += vz
        #Rotate
        #Trim stopped region
        removal = np.full(len(trajectory), True)
        previous_element = np.full(3, -1.0)
        for idx, element in enumerate(trajectory):
            if element[0] == previous_element[0] and element[1] == previous_element[1] and element[2] == previous_element[2]:
                removal[idx] = False
            previous_element = element

        trajectory = trajectory[removal]
        if len(trajectory) < 2:
            return None

        return LinearInterpolator(trajectory[:, 2], trajectory[:, :2].T)
    
    def check_interpolator(self, params: GeneratorParams) -> bool:
        '''
        Check to see if this interpolator matches the given parameters
        '''
        is_valid = (
                        params.particle.isotopic_symbol == self.particle_name 
                        and params.bfield == self.bfield and params.efield == self.efield 
                        and params.target.pretty_string == self.gas_name
                        and params.ke_min == self.ke_min and params.ke_max == self.ke_max and params.ke_bins == self.ke_bins 
                        and params.polar_min == self.polar_min and params.polar_max == self.polar_max and params.polar_bins == self.polar_bins
                    )
        
        if is_valid:
            return True
        else:
            return False
        
    def check_values_in_range(self, ke: float, polar: float) -> bool:
        '''
        Check if these values of energy, angle are within the interpolation range
        '''
        polar_deg = polar / DEG2RAD
        if ke > self.ke_max or ke < self.ke_min or polar_deg < self.polar_min or polar_deg > self.polar_max:
            return False
        else:
            return True
        

def create_interpolator(track_path: Path) -> TrackInterpolator:
    track_file = h5.File(track_path, 'r')
    track_group: h5.Group = track_file['tracks']
    particle_name = track_group.attrs['particle']
    gas_name = track_group.attrs['gas']
    bfield = track_group.attrs['bfield']
    efield = track_group.attrs['efield']
    ke_min = track_group.attrs['ke_min']
    ke_max = track_group.attrs['ke_max']
    ke_bins = track_group.attrs['ke_bins']
    polar_min = track_group.attrs['polar_min']
    polar_max = track_group.attrs['polar_max']
    polar_bins = track_group.attrs['polar_bins']
    data = track_group['data'][:].copy()

    pmin_rad = polar_min * DEG2RAD
    pmax_rad = polar_max * DEG2RAD

    typed_interpolators = List()

    [typed_interpolators.append(BilinearInterpolator(pmin_rad, pmax_rad, polar_bins, ke_min, ke_max, ke_bins, time.T[:, :, :3])) for time in data]

    return TrackInterpolator(str(track_path), typed_interpolators, particle_name, gas_name, bfield, efield, ke_min, ke_max, ke_bins, polar_min, polar_max, polar_bins)