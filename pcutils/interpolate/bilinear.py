import numpy as np
import math

def clamp(value: float | int, low: float | int, hi: float | int) -> float | int:
    return max(low, min(value, hi))

class BilinearInterpolator:
    '''
    Interpolator for regularly spaced grid, where axes are given in strictly ascending order.
    Values outside the interpolation range can either be clamped to the interpolation range or result in a
    NaN value.
    '''
    def __init__(self, x_min: float, x_max: float, x_bins: int, y_min: float, y_max: float, y_bins: int, data: np.ndarray, nan: bool = True):
        self.x_min = x_min
        self.x_max = x_max
        self.x_bins = x_bins
        self.x_width = (self.x_max - self.x_min) / float(self.x_bins)
        self.y_min = y_min
        self.y_max = y_max
        self.y_bins = y_bins
        self.y_width = (self.y_max - self.y_min) / float(self.y_bins)
        self.values = data
        self.nan = nan
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
    
    def __call__(self, x: float, y: float) -> np.ndarray:

        if self.nan and (x  > self.x_max or x < self.x_min or y < self.y_min or y > self.y_max):
            return np.nan
        
        
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

        return (q11 * (x2x) * (y2y) +
                q21 * (xx1) * (y2y) +
                q12 * (x2x) * (yy1) +
                q22 * (xx1) * (yy1)
               ) / ((x2 - x1) * (y2 - y1))
