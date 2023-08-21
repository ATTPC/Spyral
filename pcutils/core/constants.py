from scipy.constants import physical_constants

INVALID_PEAK_CENTROID: int = -1.0

INVALID_EVENT_NAME: str = "InvalidEvent"

INVALID_EVENT_NUMBER: int = -1

INVALID_PAD_ID: int = -1

NUMBER_OF_TIME_BUCKETS: int = 512

#Alias some scipy constants to avoid long key strings
AMU_2_MEV: float = physical_constants['atomic mass constant energy equivalent in MeV'] #CODATA 2018, convert u to MeV/c^2

ELECTRON_MASS_U: float = physical_constants['electron mass in u'] #CODATA 2018, evaluated by scipy

MEV_2_JOULE: float = physical_constants['electron volt-joule relationship'] * 1.0e6 # J/ev * ev/MeV = J/MeV

MEV_2_KG: float = physical_constants['electron volt-kilogram relationship'] * 1.0e6 # kg/ev * ev/MeV = kg/MeV (per c^2)
