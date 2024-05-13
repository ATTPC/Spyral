"""Constants defined for Spyral

Some of these are just short aliases to scipy constants to avoid long keystrings

Attributes
----------
INVALID_PEAK_CENTROID: float
    Value is -1.0, used for peak finding
INVALID_EVENT_NAME: str
    Value is "InvalidEvent"
INVALID_EVENT_NUMBER: int
    Value is -1
INVALID_PAD_ID: int
    Value is -1
NUMBER_OF_TIME_BUCKETS: int
    For the GET system, 512
DEG2RAD: float
    angular degrees -> radians
FRIB_TRACE_LENGTH: int
    For FRIBDAQ system, 2048
AMU_2_MEV: float 
    scipy alias, MeV/c^2/u
ELECTRON_MASS_U: float
    scipy alias, mass of an electron in u
MEV_2_JOULE: float
    scipy alias, J/MeV 
MEV_2_KG: float
    scipy alias, kg/MeV/c^2
ROOM_TEMPERATURE: float
    In Kelvin
GAS_CONSTANT: float
    scipy alias, cm^3 Torr / (K mol)
QBRHO_2_P: float
    Shortcut for converting Q*B*Rho to momentum, T m / MeV
C: float
    scipy alias, speed_of_light m/s
E_CHARGE: float
    scipy alias, elementary_charge in Coulombs
"""

from scipy.constants import physical_constants
from scipy.constants import speed_of_light, elementary_charge
from numpy import pi

INVALID_PEAK_CENTROID: float = -1.0

INVALID_EVENT_NAME: str = "InvalidEvent"

INVALID_EVENT_NUMBER: int = -1

INVALID_PAD_ID: int = -1

NUMBER_OF_TIME_BUCKETS: int = 512

DEG2RAD: float = pi / 180.0

FRIB_TRACE_LENGTH: int = 2048

# Alias some scipy constants to avoid long key strings
AMU_2_MEV: float = physical_constants["atomic mass constant energy equivalent in MeV"][
    0
]  # CODATA 2018, convert u to MeV/c^2

ELECTRON_MASS_U: float = physical_constants["electron mass in u"][
    0
]  # CODATA 2018, evaluated by scipy

MEV_2_JOULE: float = (
    physical_constants["electron volt-joule relationship"][0] * 1.0e6
)  # J/ev * ev/MeV = J/MeV

MEV_2_KG: float = (
    physical_constants["electron volt-kilogram relationship"][0] * 1.0e6
)  # kg/ev * ev/MeV = kg/MeV (per c^2)

ROOM_TEMPERATURE: float = 293.0  # Kelvin

GAS_CONSTANT: float = (
    physical_constants["molar gas constant"][0] * 0.0075 * ((100.0) ** 3.0)
)  # m^3 Pa / K mol -> m^3 Torr / K mol -> cm^3 Torr / K mol

QBRHO_2_P: float = 1.0e-9 * 10.0 * 100.0 * speed_of_light  # T * m -> MeV/c^2

C = speed_of_light  # m/s

E_CHARGE = elementary_charge  # Coulombs
