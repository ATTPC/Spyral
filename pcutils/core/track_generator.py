import numpy as np
from scipy.integrate import solve_ivp
import h5py as h5
from .target import Target
from .nuclear_data import NucleusData
from dataclasses import dataclass
from pathlib import Path
from scipy import constants, spatial
from ..interpolate import BilinearInterpolator
from scipy.interpolate import CubicSpline
from .constants import MEV_2_JOULE, MEV_2_KG
import math

TIME_WINDOW: float = 1.0e-6 #1us window
QBRHO_2_P: float = 1.0e-9 * constants.speed_of_light #kG * cm -> MeV
SAMPLING_PERIOD: float = 2.0e-9 # seconds, converts time bucket interval to time
SAMPLING_RANGE: np.ndarray = np.arange(0., TIME_WINDOW, SAMPLING_PERIOD)
PRECISION: float = 2.0e-6
DEG2RAD: float = np.pi / 180.0

@dataclass
class InitialState:
    vertex_x: float = 0.0 #m
    vertex_y: float = 0.0
    vertex_z: float = 0.0
    polar: float = 0.0 #rad
    azimuthal: float = 0.0
    kinetic_energy: float = 0.0 # MeV

@dataclass
class GeneratorParams:
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
    if trackpath.exists():
        print(f'Track file {trackpath} detected, this file will be used. If new tracks are needed, please delete the original file.')
        return True
    else:
        return False

def generate_tracks(params: GeneratorParams, trackpath: Path):
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

    

class TrackInterpolator:
    def __init__(self, track_path: Path):
        self.filepath = track_path
        self.data: np.ndarray | None = None
        self.particle_name: str = 'Invalid'
        self.gas_name: str = 'Invalid'
        self.bfield: float = 0.0
        self.efield: float = 0.0
        self.ke_min: float = 0.0
        self.ke_max: float = 0.0
        self.ke_binsx: int = 0
        self.polar_min: float = 0.0
        self.polar_max: float = 0.0
        self.polar_bins: int = 0
        self.read_file()

    
    def read_file(self):
        track_file = h5.File(self.filepath, 'r')
        track_group: h5.Group = track_file['tracks']
        self.particle_name = track_group.attrs['particle']
        self.gas_name = track_group.attrs['gas']
        self.bfield = track_group.attrs['bfield']
        self.efield = track_group.attrs['efield']
        self.ke_min = track_group.attrs['ke_min']
        self.ke_max = track_group.attrs['ke_max']
        self.ke_bins = track_group.attrs['ke_bins']
        self.polar_min = track_group.attrs['polar_min']
        self.polar_max = track_group.attrs['polar_max']
        self.polar_bins = track_group.attrs['polar_bins']
        self.data = track_group['data'][:].copy()

        pmin_rad = self.polar_min * DEG2RAD
        pmax_rad = self.polar_max * DEG2RAD

        self.interpolators = [BilinearInterpolator(pmin_rad, pmax_rad, self.polar_bins, self.ke_min, self.ke_max, self.ke_bins, time.T[:, :, :3]) for time in self.data]

    def get_interpolated_trajectory(self, initial_state: InitialState) -> CubicSpline | None:

        is_backwards = False
        if initial_state.polar > np.pi*0.5:
            is_backwards = True
            initial_state.polar -= np.pi*0.5

        trajectory = np.asarray([interp(initial_state.polar, initial_state.kinetic_energy) for interp in self.interpolators])

        #Rotate polar by 90 degrees if nescessary
        #Since we rotate the entire coordinate system, is as simple as flip in z
        #Order here is important! Must be polar, azimuthal, translate!
        if is_backwards:
            trajectory[:, 2] *= -1
        #Rotate the trajectory in azimuthal (around z) to match data
        z_rot = spatial.transform.Rotation.from_rotvec([0.0, 0.0, initial_state.azimuthal])
        trajectory = z_rot.apply(trajectory)
        #Translate to vertex
        trajectory[:, 0] += initial_state.vertex_x
        trajectory[:, 1] += initial_state.vertex_y
        trajectory[:, 2] += initial_state.vertex_z
        #Rotate
        #Trim stopped region
        _, indicies = np.unique(trajectory[:, 2], return_index=True)
        trajectory = trajectory[indicies]
        if len(trajectory) < 2:
            return None

        return CubicSpline(trajectory[:, 2], trajectory[:, :2], extrapolate=False)
    
    def check_interpolator(self, params: GeneratorParams) -> bool:
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
        polar_deg = polar / DEG2RAD
        if ke > self.ke_max or ke < self.ke_min or polar_deg < self.polar_min or polar_deg > self.polar_max:
            return False
        else:
            return True
        