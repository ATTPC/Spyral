# Quick Start Guide

Here we'll walk through the basic installation and running of Spyral. This won't cover all the details of how Spyral works, but should allow you to get up and running. Spyral requires Python version 3.10 or greater, as well as the pip tool. Please make sure these requirements are satisfied before proceeding!

## Installation

To download Spyral it is recommend to use `git`

```bash
git clone https://github.com/turinath/Spyral.git
```

This will download the Spyral source code and install it to a directory named `Spyral`. To install all of the necessary dependencies it is recommended to create a virtual environment and use pip. To create a new virtual environment navigate to the Spyral directory and run the following command:

```bash
python -m venv .venv
```

Note that on some operating systems you may need to use `python3` instead of `python`. Additionally on some Linux distributions you may need to install the correct pip using

```bash
sudo apt install python3-pip
```

Once your virtual environment is created, activate it by running

```bash
source .venv/bin/activate
```

The above example is for Linux/MacOS. Windows users will need to use the slightly different commands. More details on virtualenvs can be found [here](https://docs.python.org/3/library/venv.html). Once the virtualenv is active, use pip to install the dependencies.

```bash
pip install -r requirements.txt
```

To make sure everything went ok you can then run

```bash
python main.py --help
```

from the top level of the repository. This should display the Spyral help message if everything went well.

## Configuration

Spyral parameters are configured using a JSON file. Spyral ships with an example of the configuration in the `config.json` file. It comes with some useful defaults that have been found to provide generally good results. However, you must provide a few paths that are used to tell Spyral where to find and write data. It is also recommended to make a copy of `config.json` (i.e. `local_config.json`) for actual use. This way you can use `config.json` as a reference.

Navigate to the `Workspace` group of the configuration, and change the paths to reflect your data. As an example, a valid set of workspace parameters might look like:

```json
"Workspace":
{
    "trace_data_path": "/media/MyDrive/a1975/h5/",
    "workspace_path": "/media/MyDrive/a1975/spyral/",
    "pad_geometry_path": "/home/user/Spyral/etc/padxy.csv",
    "pad_gain_path": "/home/user/Spyral/etc/pad_gain_map.csv",
    "pad_time_path": "/home/user/Spyral/etc/pad_time_correction.csv",
    "pad_electronics_path": "/home/user/Spyral/etc/pad_electronics.csv"
},
```

Some highlights here:

- "trace_data_path" is where you specify the path at which Spyral can find raw traces from the AT-TPC. These are files typically produced by the [merger](https://github.com/gwm17/rusted_graw/).
- "workspace_path" is a location where Spyral can write data. Spyral will make all of the subdirectories it needs. Be sure to pick a good location for this!
- For all of the other fields we're using the Spyral defaults that come shipped with the framework (found in the `etc/` directory).

Next we need to configure the `Run` parameters. These control what runs Spyral will analyze and what level of analysis will be performed. An example set of paramters would be

```json
"Run":
{
    "run_min": 17,
    "run_max": 17,
    "n_processes": 1,
    "phase_pointcloud": true,
    "phase_cluster": false,
    "phase_estimate": false,
    "phase_solve": false
},
```

Here we've set Spyral to analyze one run (17) with one process, and to only perform the point cloud phase. Finally we need to set some `Detector` parameters

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
    "electric_field_garfield_path": "/home/user/Spyral/etc/electrons.txt",
    "electric_field_correction_file_name": "h2_300torr.npy"
},
```

Here we've set some of the parameters to reasonable defaults for a 6.25 MHz sampling frequency. We also elected to use the default electric field correction which is for 300 Torr of H<sub>2</sub> gas.
All of the other parameters we will leave as defaults for now.

## Running Spyral

To run Spyral simply use

```bash
python main.py local_config.json
```

If everything worked correctly you should see the Spyral interface and analysis should begin!
