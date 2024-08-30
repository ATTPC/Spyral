# Getting Started

With Spyral installed to your environment, we can now get started using Spyral to analyze some data! Below is an example python script that would run the default Spyral analysis Pipeline for data from the a1975 experiment run with AT-TPC at Argonne National Lab.

```python
import dotenv
dotenv.load_dotenv()

from spyral import (
    Pipeline,
    start_pipeline,
    PointcloudPhase,
    ClusterPhase,
    EstimationPhase,
    InterpSolverPhase,
)
from spyral import (
    PadParameters,
    GetParameters,
    FribParameters,
    DetectorParameters,
    ClusterParameters,
    SolverParameters,
    EstimateParameters,
    DEFAULT_MAP,
    DEFAULT_LEGACY_MAP,
)

from pathlib import Path
import multiprocessing

workspace_path = Path("/path/to/your/workspace/")
trace_path = Path("/path/to/your/attpc/traces/")

run_min = 94
run_max = 94
n_processes = 4

pad_params = PadParameters(
    pad_geometry_path=DEFAULT_MAP,
    pad_time_path=DEFAULT_MAP,
    pad_electronics_path=DEFAULT_MAP,
    pad_scale_path=DEFAULT_MAP,
)

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

det_params = DetectorParameters(
    magnetic_field=2.85,
    electric_field=45000.0,
    detector_length=1000.0,
    beam_region_radius=25.0,
    micromegas_time_bucket=10.0,
    window_time_bucket=560.0,
    get_frequency=6.25,
    garfield_file_path=Path("/path/to/some/garfield.txt"),
    do_garfield_correction=False,
)

cluster_params = ClusterParameters(
    min_cloud_size=50,
    min_points=3,
    min_size_scale_factor=0.05,
    min_size_lower_cutoff=10,
    cluster_selection_epsilon=10.0,
    min_cluster_size_join=15,
    circle_overlap_ratio=0.25,
    outlier_scale_factor=0.05,
)

estimate_params = EstimateParameters(
    min_total_trajectory_points=30, smoothing_factor=100.0
)

solver_params = SolverParameters(
    gas_data_path=Path("/path/to/some/gas/data.json"),
    particle_id_filename=Path("/path/to/some/particle/id.json"),
    ic_min_val=900.0,
    ic_max_val=1350.0,
    n_time_steps=1000,
    interp_ke_min=0.1,
    interp_ke_max=70.0,
    interp_ke_bins=350,
    interp_polar_min=2.0,
    interp_polar_max=88.0,
    interp_polar_bins=166,
    fit_vertex_rho=True,
    fit_vertex_phi=True,
    fit_azimuthal=True,
)

pipe = Pipeline(
    [
        PointcloudPhase(
            get_params,
            frib_params,
            det_params,
            pad_params,
        ),
        ClusterPhase(cluster_params, det_params),
        EstimationPhase(estimate_params, det_params),
        InterpSolverPhase(solver_params, det_params),
    ],
    [True, True, True, True],
    workspace_path,
    trace_path,
)


def main():
    start_pipeline(pipe, run_min, run_max, n_processes)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()

```

If that looks overwhelming, don't worry, we'll break down each section step by step and explain what is going on here.

## Imports and environments

The very first import might look kind of weird, we `import dotenv` and then simply call a fuction `load_dotenv` from the library. These lines are important because they're how we control the number of threads that our dependencies can use (numpy, scipy, polars all use multithreading in their backends). `load_dotenv` checks if there is a file named `.env` in the same folder as the script you are running, and loads if it exists. See this [section](./parallel.md#turning-off-numpy-scipy-etc-threads) which goes over how to control the threading of libraries. For now it is only important to say that these lines of code should exist at the top of your script.

Getting to the good stuff, we import all the Spyral functionality we need using the `from spyral import ...` statements. We also import the Python standard library `Path` object and a function from `multiprocessing` to make our code compatible with specific flavors of Linux. We then define the paths to the Spyral workspace as well as AT-TPC trace data.

## Workspace and Trace Data

Let's start by discussing the workspace which is defined in the example. The workspace path is a path to a directory where Spyral is allowed to write data. Spyral stores intermediate results as well as necessary assets for performing analysis. In the example we defined it as

```python
workspace_path = Path("/path/to/your/workspace/")
```

Note that we defined it as a `Path` object. This is necessary. Path objects can be inspected for whether or not the path exists. If the workspace does not exist, Spyral will attempt to create it. Obviously, the path we defined here is not a real path, so you should change this for your specific use case.

The other path we need to define, is the path to the AT-TPC trace data. AT-TPC trace data is the raw AT-TPC data after it has been merged (see [attpc_merger](https://github.com/attpc/attpc_merger)) into single run files. In the example we define it as

```python
trace_path = Path("/path/to/your/attpc/traces/")
```

Again, it is important that this is a `Path` object. And similarly to the workspace path, the example dummy value should be replaced with a real path when you write your script. The runs stored should be named with the standard AT-TPC convention (`run_0001.h5`, etc.)

## Run Range and Processors

Next we define our range of run numbers that we want to analyze. In the example we set it as

```python
run_min = 94
run_max = 94
```

These values are inclusive, so in this case our range covers exactly one run, run 94. The range can have gaps; say you have runs 54, 55, 57, 59 you can put the range to 54 to 59 and Spyral will just skip the runs that don't exist.

Spyral can also support skipping specific run numbers in the range. For example, you may want to do something like:

```python
run_min = 54
run_max = 59
runs_to_skip = [57, 55]
```

Note that the runs to skip must be in a list.

Next we set the number of processors. Spyral will analyze the runs in parallel, dividing up the tasks amongst a set of independent parallel processes. For more details on how this works and how to determine a good number of processes, see the [Parallel](parallel.md) docs. In the example we set it as

```python
n_processes = 4
```

It is important to note that in this case, because we only have one run, even though we request four processes, only one will be created.

## Parameters, Phases, and the Pipeline

Most of the remaining script pertains to the heart of Spyral: Phases and the Pipeline. A Phase (or in Spyral code a `PhaseLike`) is a unit of work to be done upon the data. Typically, a Phase will alter the data in some way, generating a new representation or organization of data. Phases can be chained together into a Pipeline, where a Pipeline represents a complete analysis. Each Phase emits a result (a `PhaseResult`) containing information about what data it output and where it can be found. Phases operate at the run level. That is, each `PhaseLike` implements a `run` function which performs that analysis upon a single run. The Pipeline can accept a run list and will exhaustively call the Phases, in order, to analyze the list. Spyral ships with default Phases, which have been used to analyze AT-TPC data and have been found to be generally successful.

Now that we've laid the ground work here, let's look at the example. First we define a bunch of parameter groups. This essentially represents the configuration of Spyral. We group the parameters into classes to keep things more organized and more efficiently pass them around. We won't go through the details of each individual group here; you can find documentation on the different parameters in the [Configuration](config/about.md) section. With our parameters defined we then create our Pipeline

```python
pipe = Pipeline(
    [
        PointcloudPhase(
            get_params,
            frib_params,
            det_params,
            pad_params,
        ),
        ClusterPhase(cluster_params, det_params),
        EstimationPhase(estimate_params, det_params),
        InterpSolverPhase(solver_params, det_params),
    ],
    [True, True, True, True],
    workspace_path,
    trace_path,
)
```

The first argument to the Pipeline constructor is a list of `PhaseLike` objects. Here we use the Spyral provided defaults. There are some important things to note here. First is that the **order of the phases matters**! The PointcloudPhase operates on AT-TPC trace data, the ClusterPhase operates on PointcloudPhase data, and so on. The Pipeline will check to make sure that you put things in order though, so don't worry too much (unless you're using custom Phases). For details on each default phases and info on custom Phases see the [Phases](phases/about.md) section of the documentation.

The next argument is a list of booleans of the same length as the list of Phases. These are switches that can selectively turn on or off a Phase in the Pipeline. If the switch is True, the corresponding Phase will be run. If False, the Phase will be skipped.

We then also pass along our workspace and trace paths.

Finally we define a main function for our script

```python
def main():
    start_pipeline(pipe, run_min, run_max, n_processes)
```

If you want to skip some runs like we mentioned before, pass the skip list like

```python
def main():
    start_pipeline(pipe, run_min, run_max, runs_to_skip=runs_to_skip)
```

And set this up as an entry point

```python
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
```

Note the `multiprocessing` function we use here is for compatibility with some Linux flavors.

## Run the Script

Now you can run your script as

```bash
python <your_script_file.py>
```

and begin your analysis!
