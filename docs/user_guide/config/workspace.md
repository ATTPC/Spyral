# Workspace Configuration

The Workspace refers to the location to which Spyral will generally write and read data. It is a directory to which Spyral has control and essentially owns. The default workspace parameters given in `config.json` are

```json
"Workspace":
{
    "trace_data_path": "/path/to/some/traces/",
    "workspace_path": "/path/to/some/workspace/",
    "pad_geometry_path": "/path/to/some/geometry.csv",
    "pad_gain_path": "/path/to/some/gain.csv",
    "pad_time_path": "/path/to/some/time.csv",
    "pad_electronics_path": "/path/to/some/electronics.csv"
},
```

Note that these defaults *will not work*. You *must* modify these values to run Spyral.

A break down of each parameter:

## trace_data_path

This is the path to the raw trace data. This is a directory which contains many [HDF5](https://www.hdfgroup.org/solutions/hdf5/) files with names formated as `run_0001.h5`. These files are typically produced by the AT-TPC [merger](https://github.com/gwm17/rusted_graw.git). See that link for more details on the trace data. The default value from `config.json` *is not valid* and should *always* be modified.

An important note: Spyral is only configured to work with data which uses the dual AT-TPC DAQ / FRIBDAQ configuration. Older data may have been taken using the AT-TPC DAQ only configuration. This older data is not compatible with Spyral out of the box and will require modification (either to the raw data or to Spyral, dealer's choice).

## workspace_path

This is the path at which Spyral will create it's workspace. This is one of the most important configuration parameters. Inside the workspace, a directory structure will be created

```txt
|-- workspace
   |-- clouds
   |-- clusters
   |-- correction
   |-- estimates
   |-- physics
   |-- log
   |-- tracks
   |-- gates
```

The `clouds`, `clusters`, `estimates`, and `physics` directories are where the output of the various analysis phases are stored (either HDF5 files or Apache [parquet](https://parquet.apache.org/) files). `correction` and `tracks` store the result of interpolation meshes generated for the electric field correction and ODE fitting respectively. `gates` stores particle ID gates, typically generated using the plotter tool. Finally, `log` contains log files which Spyral writes to. The default value from `config.json` *is not valid* and should *always* be modified. It is recommended to create a workspace for each experiment.

Note: Spyral will *always* create this directory structure, even if you accidentally use the wrong path or some invalid path and it can be annoying to cleanup if done incorrectly. So be careful to set the workspace location correctly before running Spyral!

## pad_geometry_path

This is the path to a `.csv` file containing the mapping of pad number to X-Y coordinate location in millimeters. The default value from `config.json` *is not valid* and should *always* be modified. Spyral does ship with an example file in the `etc/` directory, `padxy.csv`. This example file should be safe to use; the pad geometry does not change often.

## pad_gain_path

This is the path to a `.csv` file containing the mapping of pad number to a gain correction factor. It corrects the relative gain offset of the small and large pads. The default value from `config.json` *is not valid* and should *always* be modified. Spyral does ship with an example file in the `etc/` directory, `pad_gain.csv`. This example file should be safe to use; the pad gain factor does not change often.

## pad_time_path

This is the path to a `.csv` file containing the mapping of pad number to a time offset. This corrects for small jitters in the timestamp on a channel by channel basis. The default value from `config.json` *is not valid* and should *always* be modified. Spyral does ship with an example file in the `etc/` directory, `pad_time_correction.csv`. This example file should be safe to use; the pad time offset does not change often. Note: the offset is not well tested across multiple detector configurations.

## pad_electronics_path

This is the path to a `.csv` file containing the mapping of pad number to electronics hardware. The default value from `config.json` *is not valid* and should *always* be modified. Spyral does ship with an example file in the `etc/` directory, `pad_electronics.csv`. This example file should be safe to use; the pad electronics does not change often. Note: this file is currently unused and this parameter may be removed.
