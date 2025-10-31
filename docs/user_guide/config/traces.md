# GET & FRIB DAQ Trace Analysis Configuration

The GET and FRIB parameters control the trace analysis for each acquisition. The default trace parameter objects are:

```python
get_params = GetParameters(
    baseline_window_scale=20.0,
    peak_separation=50.0,
    peak_prominence=20.0,
    peak_max_width=50.0,
    peak_threshold=40.0,
)

frib_params = FribParameters(
    baseline_window_scale=100.0,
    peak_separation=50.0,
    peak_prominence=20.0,
    peak_max_width=500.0,
    peak_threshold=100.0,
    ic_delay_time_bucket=1100,
    ic_multiplicity=1,
)
```

A break down of each parameter:

## baseline_window_scale

The size of the window used to create a moving average of the baseline in the baseline removal algorithm. Baselines are removed using fourier analysis, passing the traces through a low-pass filter. This parameter is available to both FRIB and GET data.

## peak_separation

The minimum space between peaks in the peak finding algorithm. This is a re-exposure of the `distance` parameter in [scipy's find_peaks](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html#scipy.signal.find_peaks). See their documetation for more details. This parameter is available to both FRIB and GET data.

## peak_prominence

The minimum prominence of a peak (distance to the lowest contour). This is a re-exposure of the `prominence` parameter in [scipy's find_peaks](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html#scipy.signal.find_peaks). See their documentation for more details. This parameter is available to both FRIB and GET data.

## peak_max_width

The maximum width of a peak (at the base of the peak). This is a re-exposure of the `width` parameter in [scipy's find_peaks](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html#scipy.signal.find_peaks). See their documentation for more details. This parameter is available to both FRIB and GET data.

## peak_threshold

The minimum amplitude of a peak. This parameter is available to both FRIB and GET data.

## ic_delay_time_bucket

This is the delay to the ion chamber singal in units of FRIBDAQ time buckets. Any peaks in the ion chamber singal before this time bucket are ignored and not included in subsequent ion chamber analysis. The first peak after this time bucket is the triggering ion chamber singal, and all peaks after this time bucket are counted towards the ion chamber multiplicity (except in the case of `event_correct_ic_time` set to true, where the auxilary silicon detector is used to eliminate some of the peaks). In general, this parameter does not need changed from experiment to experiment, as it is a direct result of the electronic delay in the AT-TPC triggering scheme.

## event_ic_multiplicity

The maximum allowed ion chamber multiplicity for an event. In general, AT-TPC experiments would only allow for one ion chamber hit per event. This is only available to FRIB data.
