# Spyral

![CI](https://github.com/ATTPC/Spyral/actions/workflows/ci.yml/badge.svg)
[![PyPI version shields.io](https://img.shields.io/pypi/v/attpc_spyral.svg)](https://pypi.python.org/pypi/attpc_spyral/)
[![PyPI license](https://img.shields.io/pypi/l/attpc_spyral.svg)](https://pypi.python.org/pypi/attpc_spyral/)

Spyral is an analysis library for data from the Active Target Time Projection Chamber (AT-TPC). Spyral provides a flexible analysis pipeline, transforming the raw trace data into physical observables over several tunable steps. The analysis pipeline is also extensible, supporting a diverse array of datasets. Sypral can process multiple data files in parallel, allowing for scalable performance over larger experiment datasets.

## Installation

Install using pip:

```bash
pip install attpc_spyral
```

It is recommended to install Spyral to a virtual environment.

## Requirements

Python >= 3.10, < 3.13

Spyral aims to be cross platform and to support Linux, MacOS, and Windows. Currently Spyral has been tested and confirmed on MacOS and Ubuntu 22.04 Linux. Other platforms
are not guaranteed to work; if there is a problem please make an issue on the GitHub page, and it will be resolved as quickly as possible.

## Usage

For a full user guide and documentation with examples, see [our docs](https://attpc.github.io/Spyral/). Below is an example script of using Spyral with the default pipeline

```python
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
    INVALID_PATH,
)

from pathlib import Path
import multiprocessing

workspace_path = Path("/some/workspace/path/")
trace_path = Path("/some/trace/path/")

run_min = 94
run_max = 94
n_processes = 4

pad_params = PadParameters(
    is_default=True,
    is_default_legacy=False,
    pad_geometry_path=INVALID_PATH,
    pad_time_path=INVALID_PATH,
    pad_electronics_path=INVALID_PATH,
    pad_scale_path=INVALID_PATH,
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
    correct_ic_time=True,
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
    circle_overlap_ratio=0.5,
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

### Pipeline

The core of Spyral is the Pipeline. A Pipeline in a complete description of an analysis, made up of individual Phases. Each Phase is a unit of analysis to be performed on data. Spyral provides a complete set of default Phases which can be used to completely analyze an AT-TPC dataset. Custom Phases can also be created to extend the functionality of Spyral.

### Parallel Processing

Spyral is capable of running multiple data files in parallel. This is acheived through the python `multiprocessing` library. In the `start_pipeline` function a parameter named `n_processors indicates to Spyral the *maximum* number of processors which can be spawned. Spyral will then inspect the data load that was submitted in the configuration and attempt to balance the load across the processors as equally as possible.

Some notes about parallel processing:

- In job environments (SLURM, etc.), you won't want to have the typical progress display provided by Spyral. Set the `disable_display` argument of `start_pipeline` to `False` in this case.
- In general, it is best if the number of data files to be processed is evenly divisible by the number of processors. Otherwise, by necessity, the work load will be uneven across the processors.
- Spyral will sometimes run fewer processes than requested. This is usually in the case where the number of requested processors is greater than the number of files to be processed.

### Logs and Output

Spyral creates a set of logfiles when it is run (located in the log directory of the workspace). These logfiles can contain critical information describing the state of Spyral. In particular, if Spyral has a crash, the logfiles can be useful for determining what went wrong. A logfile is created for each process (including the parent process). The files are labeld by process number (or as parent in the case of the parent).

## Notebooks

See the [spyral_notebooks](https://github.com/attpc/spyral_notebooks) repository for notebooks which demonstrate the behavior of the default Phases of Spyral.
