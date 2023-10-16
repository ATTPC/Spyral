import numpy as np
import math
from numba import njit, int32, float64, boolean
from numba.experimental import jitclass

@njit
def clamp(value: float | int, low: float | int, hi: float | int) -> float | int:
    return max(low, min(value, hi))

# To use numba with a class we need to declare the types of all members of the class
# and use the @jitclass decorator
@jitclass([('x_min', float64), 
           ('x_max', float64), 
           ('x_bins', int32), 
           ('x_width', float64), 
           ('y_min', float64), 
           ('y_max', float64), 
           ('y_bins', int32), 
           ('y_width', float64), 
           ('values', float64[:,:,:]), 
           ('nan', boolean)])
class BilinearInterpolator:
    '''
    # BilinearInterpolator
    Interpolator for regularly spaced grid, where axes are given in strictly ascending order.
    Values outside the interpolation range can either be clamped to the interpolation range or result in a
    NaN value.

    We use numba to just-in-time compile these methods, which results in dramatic speed increases, on the order of a factor of 50
    '''
    def __init__(self, x_min: float, x_max: float, x_bins: int, y_min: float, y_max: float, y_bins: int, data: np.ndarray, nan: bool = True):
        self.x_min: float = x_min
        self.x_max: float = x_max
        self.x_bins: float = x_bins
        self.x_width: float = (self.x_max - self.x_min) / float(self.x_bins)
        self.y_min: float = y_min
        self.y_max: float = y_max
        self.y_bins: float = y_bins
        self.y_width: float = (self.y_max - self.y_min) / float(self.y_bins)
        self.values: np.ndarray = data
        self.nan: bool = nan
        self.check_values_shape()

    def check_values_shape(self):
        values_shape = self.values.shape
        if len(values_shape) < 2:
            print(f'The values given to BilinearInterpolator do not have the correct dimensionality! Given {values_shape}, requires a minimum 2 dimensions')
            raise Exception
        if values_shape[0] != self.x_bins:
            print(f'The shape of the values given to BilinearInterpolator along the x-axis does not match the given x-axis! axis={self.x_bins} values={values_shape[0]}')
            raise Exception
        if values_shape[1] != self.y_bins:
            print(f'The shape of the values given to BilinearInterpolator along the y-axis does not match the given y-axis! axis={self.y_bins} values={values_shape[1]}')
            raise Exception
        
    def get_edges_x(self, value: float) -> (int, float, int, float):
        bin_low = math.floor((value - self.x_min)/float(self.x_width))
        edge_low = self.x_min + bin_low*self.x_width
        bin_hi = min(bin_low + 1, self.x_bins-1)
        edge_hi = self.x_min + bin_hi*self.x_width
        return (bin_low, edge_low, bin_hi, edge_hi)
    
    def get_edges_y(self, value: float) -> (int, float):
        bin_low = math.floor((value - self.y_min)/float(self.y_width))
        edge_low = self.y_min + bin_low*self.y_width
        bin_hi = min(bin_low + 1, self.y_bins-1)
        edge_hi = self.y_min + bin_hi*self.y_width
        return (bin_low, edge_low, bin_hi, edge_hi)
    
    def interpolate(self, x: float, y: float) -> float:

        if self.nan and (x  > self.x_max or x < self.x_min or y < self.y_min or y > self.y_max):
            return np.array([np.nan, np.nan, np.nan])
        
        
        x = clamp(x, self.x_min, self.x_max)
        y = clamp(y, self.y_min, self.y_max)
        x1_bin, x1, x2_bin, x2 = self.get_edges_x(x)
        y1_bin, y1, y2_bin, y2 = self.get_edges_y(y)

        q11 = self.values[x1_bin, y1_bin]
        q12 = self.values[x1_bin, y2_bin]
        q21 = self.values[x2_bin, y1_bin]
        q22 = self.values[x2_bin, y2_bin]
        x2x = x2 - x
        y2y = y2 - y
        xx1 = x - x1
        yy1 = y - y1

        if x2 == x1 and y1 == y2: #On a corner
            return q11
        elif x2 == x1: #At xlim
            return (q11 * yy1 + q12 * y2y) / (y2 - y1)
        elif y1 == y2: #At ylim
            return (q11 * xx1 + q21 * x2x) / (x1 - x2)
        else: #In a coord
            return (q11 * (x2x) * (y2y) +
                    q21 * (xx1) * (y2y) +
                    q12 * (x2x) * (yy1) +
                    q22 * (xx1) * (yy1)
                   ) / ((x2 - x1) * (y2 - y1))
