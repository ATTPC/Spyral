import pycatima as catima
from .nuclear_data import NuclearDataMap, NucleusData
from .constants import AMU_2_MEV
from dataclasses import dataclass, field
from pathlib import Path
from json import load
from typing import Optional
from scipy.interpolate import CubicSpline
import numpy as np

@dataclass
class TargetData:
    compound: list[tuple[int, int, int]] = field(default_factory=list) #(Z, A, S)
    density: float = 0.0 #g/cm^3

def load_target_data(target_path: Path) -> Optional[TargetData]:
    with open(target_path, 'r') as target_file:
        json_data = load(target_file)
        if 'compound' not in json_data or 'density' not in json_data:
            return None
        else:
            return TargetData(json_data['compound'], json_data['density'])
        
@dataclass
class ElossInterpolater:
    spline: CubicSpline
    min_energy: float
    max_energy: float
    projectile_Z: int
    projectile_A: int

class Target:

    def __init__(self, target_file: Path, nuclear_data: NuclearDataMap):
        self.data: Optional[TargetData] = load_target_data(target_file)
        if self.data is None:
            print(f'Could not load target data in file {target_file}. Prepare for a crash.')

        self.pretty_string: str = ''.join(f'{nuclear_data.get_data(z, a).pretty_iso_symbol}<sub>{s}</sub>' for (z, a, s) in self.data.compound)
        
        #Construct the target material
        self.material = catima.Material()
        for z, a, s, in self.data.compound:
            self.material.add_element(nuclear_data.get_data(z, a).atomic_mass, z, float(s))
        self.material.density(self.data.density)
        print(f'density: ', self.material.density())
        print(f'Material: {self.pretty_string}')

        self.interpolater: ElossInterpolater | None = None
    
    def __str__(self) -> str:
        return self.pretty_string
    
    #in MeV
    def generate_interpolator(self, projectile_data: NucleusData, max_energy: float, min_energy: float, step_size: float):
        x = np.arange(start=min_energy, stop=max_energy, step=step_size)
        y = np.zeros(len(x))
        mass_u = projectile_data.mass / AMU_2_MEV
        projectile = catima.Projectile(mass_u, projectile_data.Z)
        for idx, energy in enumerate(x):
            projectile.T(energy/mass_u)
            y[idx] = catima.dedx(projectile, self.material)
        self.interpolater = ElossInterpolater(CubicSpline(x, y), min_energy, max_energy, projectile_data.Z, projectile_data.A)

    def get_dedx_interpolated(self, projectile_data: NucleusData, projectile_energy: float) -> Optional[float]:
        if self.interpolater is None:
            print('Warning tried to interpolate dEdx without having generated an interpolater!')
            return None
        elif projectile_data.Z != self.interpolater.projectile_Z or projectile_data.A != self.interpolater.projectile_A:
            print('Mismatched nucleus at get_dedx_interpolated!')
            return None
        elif projectile_energy > self.interpolater.max_energy or projectile_energy < self.interpolater.min_energy:
            print('Warning projectile energy outside of interpolater range at get_dedx_interpolated!')
            return None
        
        return self.interpolater.spline(projectile_energy)


    def get_dedx(self, projectile_data: NucleusData, projectile_energy: float) -> float:
        '''
        
        ## Returns
        float: dEdx in MeV/g/cm^2
        '''
        mass_u = projectile_data.mass / AMU_2_MEV # convert to u
        projectile = catima.Projectile(mass_u, projectile_data.Z)
        projectile.T(projectile_energy/mass_u)
        return catima.dedx(projectile, self.material)
    
    def get_angular_straggling(self, projectile_data: NucleusData, projectile_energy: float) -> float:
        '''
        Heavier

        ## Returns
        float: angular straggling in radians
        '''
        mass_u = projectile_data.mass / AMU_2_MEV # convert to u
        projectile = catima.Projectile(mass_u, projectile_data.Z,T=projectile_energy/mass_u)
        return catima.calculate(projectile, self.material).get_dict()['sigma_a']
