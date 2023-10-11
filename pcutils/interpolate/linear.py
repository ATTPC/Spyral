import numpy as np

class LinearInterpolator:
    def __init__(self, x_vals: np.ndarray, y_vals: np.ndarray):
        self.x = x_vals
        self.y = y_vals
        self.check_values()

    def check_values(self):
        if len(self.y.shape) < 2:
            print(f'The y values have the wrong shape for LinearInterpolator! Shape {self.y.shape} must have at minimum 2 dimensions.')
            print('If you just want to interpolate a simple one dimensional function, use numpy.interp.')
            raise Exception
        if len(self.x) != len(self.y[1]):
            print(f'The values given to LinearInterpolator do not have the correct dimensionality! x: {len(self.x)} y: {len(self.y[1])}')
            raise Exception
        
    def __call__(self, xs: float) -> np.ndarray:
        results = np.empty((len(xs), len(self.y)))
        for idx in range(len(self.y)):
            results[:, idx] = np.interp(xs, self.x, self.y[idx], left=np.nan, right=np.nan)
        return results

