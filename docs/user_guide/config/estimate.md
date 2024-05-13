# Estimation Configuration

The Estimate parameters control the estimation phase. The default estimate parameter object is:

```python
estimate_params = EstimateParameters(
    min_total_trajectory_points=30, smoothing_factor=100.0
)
```

A break down of each parameter:

## minimum_total_trajectory_points

Minimum number of points in a cluster to be considered a valid trajectory. Any clusters smaller will be ignored.

## smoothing_factor

This is the parameter which controls the "degree" of smoothing which is applied to the cluster in x, y, and charge as a function of z. Each of these coordinates is smoothed using smoothing splines from the scipy toolkit. Our `smoothing_factor` parameter is a re-exposure of scipy's `lam` parameter in the function `make_smoothing_splines` (see [the docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_smoothing_spline.html#scipy.interpolate.make_smoothing_spline) for more details). Note that we cannot use scipy's default estimation feature for the smoothing factor; th estimation of `lam` is extremely slow, as it is an optimization problem. `smoothing_factor` cannot be negative, and in general seems to be around 100.0 for most of the data. This parameter needs more testing across more datasets, so please let us know if you find anyting interesting!
