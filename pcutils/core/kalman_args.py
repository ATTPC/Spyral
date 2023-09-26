from .target import Target
from .nuclear_data import NucleusData

class KalmanArgs:
    def __init__(self):
        self.target: Target | None = None
        self.ejectile: NucleusData | None = None
        self.bfield: float | None = None
        self.efield: float | None = None



g_args = KalmanArgs()

def set_kalman_args(target: Target, eject: NucleusData, bfield: float, efield: float):
    global g_args

    g_args.target = target
    g_args.ejectile = eject
    g_args.bfield = bfield
    g_args.efield = efield

def get_kalman_args() -> KalmanArgs:
    global g_args
    return g_args