import pycatima as catima
from .nuclear_data import NuclearDataMap, NucleusData
from .constants import AMU_2_MEV, GAS_CONSTANT, ROOM_TEMPERATURE
from dataclasses import dataclass, field
from pathlib import Path
from json import load
from typing import Optional
import numpy as np

@dataclass
class TargetData:
    compound: list[tuple[int, int, int]] = field(default_factory=list) #(Z, A, S)
    pressure: float = 0.0 #torr

    def density(self):
        molar_mass: float = 0.0 
        for (z, a, s) in self.compound:
            molar_mass += a*s
        return molar_mass * self.pressure / (GAS_CONSTANT * ROOM_TEMPERATURE)



def load_target_data(target_path: Path) -> Optional[TargetData]:
    with open(target_path, 'r') as target_file:
        json_data = load(target_file)
        if 'compound' not in json_data or 'pressure(Torr)' not in json_data:
            return None
        else:
            return TargetData(json_data['compound'], json_data['pressure(Torr)'])

class Target:

    def __init__(self, target_file: Path, nuclear_data: NuclearDataMap):
        self.data: Optional[TargetData] = load_target_data(target_file)
        if self.data is None:
            print(f'Could not load target data in file {target_file}. Prepare for a crash.')

        self.pretty_string: str = ''.join(f'{nuclear_data.get_data(z, a).pretty_iso_symbol}<sub>{s}</sub>' for (z, a, s) in self.data.compound)
        self.ugly_string: str = ''.join(f'{nuclear_data.get_data(z, a).isotopic_symbol}{s}' for (z, a, s) in self.data.compound)
        
        #Construct the target material
        self.material = catima.Material()
        for z, a, s, in self.data.compound:
            self.material.add_element(nuclear_data.get_data(z, a).atomic_mass, z, float(s))
        self.density: float = self.data.density()
        self.material.density(self.density)

    def __str__(self) -> str:
        return self.pretty_string
    
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
    
    def get_energy_loss(self, projectile_data: NucleusData, projectile_energy: float, distances: np.ndarray) -> np.ndarray:
        mass_u = projectile_data.mass / AMU_2_MEV # convert to u
        projectile = catima.Projectile(mass_u, projectile_data.Z, T=projectile_energy/mass_u)
        eloss = np.zeros(len(distances))
        for idx, distance in enumerate(distances):
            self.material.thickness_cm(distance * 100.0)
            projectile.T(projectile_energy/mass_u)
            eloss[idx] = catima.calculate(projectile, self.material).get_dict()['Eloss']
        return eloss
