# Detector Configuration

The Detector parameters control the hardware setup of the AT-TPC used for the analysis. The default detector parameters in `config.json` are:

```json
"Detector":
{
    "magnetic_field(T)": 2.85,
    "electric_field(V/m)": 45000.0,
    "detector_length(mm)": 1000.0,
    "beam_region_radius(mm)": 25.0,
    "micromegas_time_bucket": 10.0,
    "window_time_bucket": 560.0,
    "get_frequency(MHz)": 6.25,
    "electric_field_garfield_path": "/path/to/some/garfield.txt",
    "do_electric_field_correction": false
},
```

Note that these default values *will not work*. In particular the `electric_field_*_path` parameters must be modified.

A break down of each parameter:

## magnetic_field(T)

The magnetic field strength in Tesla. The default value of 2.85 T is for the Helios magnet at Argonne.

## electric_field(V/m)

The electric field strength in V/m. The default value is 45 kV/m, which is a fairly common value for gases at around 300 Torr.

## detector_length(mm)

The length of AT-TPC in millimeters. This value does not change often.

## beam_region_radius(mm)

The radius of the beam region in millimeters. This value does not change often. It is often used to validate trajectories.

## micromegas_time_bucket

The GET timebucket of the micromegas. This is used as a reference point to calibrate the electron drift time (in GET timebuckets) to z-position (in millimeters). This changes from experiment to experiment, and is typically found by examining window events (events where the beam reacted with the window). This is a *critical* parameter.

## window_time_bucket

The GET timebucket of the window. This is used as a reference point to calibrate the electron drift time (in GET timebuckets) to z-position (in millimeters). This changes from experiment to experiment, and is typically found by examining window events (events where the beam reacted with the window). This is a *critical* parameter.

## get_frequency(MHz)

The sampling frequency of the GET acquistion. This is used to correlate data between the FRIBDAQ and GET DAQ setups. The valid values are 6.25 MHz or 3.125 MHz. This is a *critical* parameter.

## electric_field_garfield_path

The path to a file containing electron drift calculations from [Garfield++](https://gitlab.cern.ch/garfield/garfieldpp). The default value from `config.json` *is not valid* and should be modified if the `do_electric_field_correction` parameter is set to true. Spyral does ship with an example file in the `etc/` directory, `electrons.txt`. The example file is not always safe to use. It was calculated for 300 Torr of H<sub>2</sub> gas, and as such may not be accurate. It is also not well tested what impact this correction has in various detector configurations.

## do_electric_field_correction

This is a flag which indiciates whether or not to apply the electric field correction to the point clouds
