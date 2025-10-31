# Pad Plane Parameters

The AT-TPC Pad Plane mapping of electronics channel to physical pad is controlled by several map files that are passed as the following data structure:

```python
pad_params = PadParameters(
    pad_geometry_path=DEFAULT_MAP,
    pad_time_path=DEFAULT_MAP,
    pad_scale_path=DEFAULT_MAP,
)
```

Below we will define each individual parameter. If a path is set to `DEFAULT_MAP` the current AT-TPC default pad mapping is used for that specific path. If a path is set to `DEFAULT_LEGACY_MAP` the default pre-FRIBDAQ AT-TPC mapping is used for that specific path. In this way some paths may be set to defaults while others can be customized.

## pad_geometry_path

This is the path to a `.csv` file containing the mapping of pad number to X-Y coordinate location in millimeters. 

## pad_time_path

This is the path to a `.csv` file containing the mapping of pad number to a time offset. This corrects for small jitters in the timestamp on a channel by channel basis. This is parameter is currently under investigation for it's impact.

## pad_scale_path

This is the path to a `.csv` file containing the mapping of pad number to scale relative to a Big Pad. This information is used to estimate relative positional errors.
