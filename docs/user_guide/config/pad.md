# Pad Plane Parameters

The AT-TPC Pad Plane mapping of electronics channel to physical pad is controlled by several map files that are passed as the following data structure:

```python
pad_params = PadParameters(
    is_default=True,
    is_default_legacy=False,
    pad_geometry_path=INVALID_PATH,
    pad_gain_path=INVALID_PATH,
    pad_time_path=INVALID_PATH,
    pad_electronics_path=INVALID_PATH,
    pad_scale_path=INVALID_PATH,
)
```

Below we will define each individual parameter.

## is_default

`is_default` is a switch which indicates if the default pad maps, which are shipped with Spyral, should be used. The default maps should be up to date with the latest Spyral experiments; as such older data may not be compatible with the defaults. If this is set to `True` all other parameters in the group are ignored.

## is_default_legacy

A switch which indicates if the default legacy (pre-FRIBDAQ), which are shipped with Spyral, pad maps should be used. These maps are contemporary with the e20009 experiment. If this is set to `True` all other parameters in the group are ignored.

## pad_geometry_path

This is the path to a `.csv` file containing the mapping of pad number to X-Y coordinate location in millimeters.

## pad_gain_path

This is the path to a `.csv` file containing the mapping of pad number to a gain correction factor. It corrects the relative gain offset of the small and large pads. This currently has essentially no impact on the Spyral analysis and is in the process of being removed.

## pad_time_path

This is the path to a `.csv` file containing the mapping of pad number to a time offset. This corrects for small jitters in the timestamp on a channel by channel basis. This is parameter is currently under investigation for it's impact.

## pad_electronics_path

This is the path to a `.csv` file containing the mapping of pad number to electronics hardware. This file is more of a utility for validating checking data and uses in plotting.

## pad_scale_path

This is the path to a `.csv` file containing the mapping of pad number to scale relative to a Big Pad. This information is used to estimate relative positional errors.
