
import numpy as np
from dataclasses import dataclass
from constants import AMU_2_MEV, ELECTRON_MASS_U

PATH_TO_MASSFILE = "./etc/amdc2016_mass.txt"

@dataclass
class NucleusData:
    mass: float = 0.0 #nuclear mass, MeV
    atomic_mass: float = 0.0 #atomic mass (includes electrons), amu
    element_symbol: str = "" #Element symbol (H, He, Li, etc.)
    isotopic_symbol: str = "" #Isotopic symbol w/o formating (1H, 2H, 3H, 4He, etc.)
    pretty_iso_symbol: str = "" #Isotopic symbol w/ rich text formating (<sup>1</sup>H, etc.)
    Z: int = 0
    A: int = 0

    def __str__(self):
        return self.isotopic_symbol

    def get_latex_rep(self):
        '''
        LaTeX formated isotopic symbol, useful for plotting
        '''
        return "$^{" + str(self.A) + "}$" + self.element_symbol

#Szudzik pairing function, requires use of unsigned integers
def generate_nucleus_id(z: np.uint32, a: np.uint32) -> np.uint32 :
    return z*z + z + a if z > a else a*a + z

class NuclearDataMap:
    def __init__(self):
        self.map = {}

        with open(PATH_TO_MASSFILE) as massfile:
            massfile.readline() # Header
            for line in massfile:
                entries = line.split()
                data = NucleusData()
                data.Z = int(entries[0]) #Column 1: Z
                data.A = int(entries[1]) #Column 2: A
                data.element_symbol = entries[3] #Column 3: Element
                data.atomic_mass = float(entries[4])
                data.mass = (float(entries[4]) - float(data.Z) * ELECTRON_MASS_U) * AMU_2_MEV #Remove electron masses to obtain nuclear masses, Column 4
                data.isotopic_symbol = f"{data.A}{entries[3]}"
                data.pretty_iso_symbol = f"<sup>{data.A}</sup>{entries[3]}"
                self.map[generate_nucleus_id(data.Z, data.A)] = data

    def get_data(self, z: np.uint32, a: np.uint32) -> NucleusData:
        return self.map[generate_nucleus_id(z, a)]