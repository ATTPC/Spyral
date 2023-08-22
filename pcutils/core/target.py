import pycatima as catima
from .nuclear_data import NuclearDataMap, NucleusData
from .constants import AMU_2_MEV
from dataclasses import dataclass, field
from pathlib import Path
from json import load
from typing import Optional

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
