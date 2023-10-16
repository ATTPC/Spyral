import numpy as np
from numba import float64
from numba.experimental import jitclass

# To use numba with a class we need to declare the types of all members of the class
# and use the @jitclass decorator
@jitclass([('x', float64[:]), ('y', float64[:, :]), ('x_min', float64), ('x_max', float64)])
class LinearInterpolator:
    '''
    # LinearInterpolator
    Simple wrapper around numpy.interp for use with vector valued functions (i.e. f(z) -> [x,y])

    We use numba to just-in-time compile these methods which results in a dramatic speed up on the order of 
    a factor of 50.
    '''
    def __init__(self, x_vals: np.ndarray, y_vals: np.ndarray):
        self.x = x_vals
        self.y = y_vals
        self.x_min = x_vals[0]
        self.x_max = x_vals[-1]
        self.check_values()

    def check_values(self):
        if len(self.y.shape) < 2:
            print(f'The y values have the wrong shape for LinearInterpolator! Shape {self.y.shape} must have at minimum 2 dimensions.')
            print('If you just want to interpolate a simple one dimensional function, use numpy.interp.')
            raise Exception
        if len(self.x) != len(self.y[1]):
            print(f'The values given to LinearInterpolator do not have the correct dimensionality! x: {len(self.x)} y: {len(self.y[1])}')
            raise Exception
        
    def interpolate(self, xs: np.ndarray) -> np.ndarray:
        results = np.empty((len(xs), len(self.y)))
        for idx in range(len(self.y)):
            results[:, idx] = np.interp(xs, self.x, self.y[idx])
        for idx, x in enumerate(xs):
            if x < self.x_min or x > self.x_max:
                results[idx] = np.array([np.nan, np.nan])
        return results


