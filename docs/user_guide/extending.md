# Extending Spyral

The AT-TPC is a powerful tool that can measure many different kinds of data at many different scales. It is, therefore, expected that there will be experiments for which the default Spyral Phases do not work very well, or more Phases may need to be injected into the Pipeline. Below is a brief description on how to extend Spyral. In this example we will add a new solving phase to Spyral.

## Project Layout

First, lets define how to layout a project to extend Spyral:

```txt
|---my_project
|   |---my_phases
|   |   |---__init__.py
|   |   |---my_solver_phase.py
|   |   |---schema.py
|   |   |---config.py
|   |---script.py
|   |---requirements.txt
|   |---.venv
```

Within our parent project folder (`my_project`) we have a subdirectory (`my_phases`) for our extension code, our script to run spyral (`script.py`), a requirements file (`requirements.txt`), and our virutal environment. Within the `phases` folder, we then have our acutal extension code (a phase, schema, and configuration).

## Extending the Configuration

Extending the configuration is very simple. In the above layout we would add something like this to the `my_phases/config.py` file:

```python
from dataclasses import dataclass

@dataclass
class MySolverParameters:
    a: float = 0.0
    b: float = 0.0
    c: str = ""
```

Easy as that! Obviously, you will want to give your class a good descriptive name, and each parameter should be named based on the functionality it will provide.

## Extending the Schema

Until now, we have not discussed Schema in Spyral and what they are for. Schema are used to give the Pipeline a way to make sure that the Phases correctly link together (data from the previous phase matches the input of the next phase). Schema are defined through JSON, and have the follwoing structure:

```json
{
    "phase":
    {
        "extension": ...,
        "data": 
        {

        }
    },
    "phase2":...
}
```

The key ideas are the following: a schema can contain artifacts from multiple phases, and describes the format for each artifact. Each artifact describes the file extension (format) that was used and the data format that was stored. You have a lot of flexibility in how to write schemas; this is necessary to cover the range of artifacts produced by the pipeline. Below is an example `schema.py` with a single artifact defined. 

```python

MY_SOLVER_SCHEMA = """
{
    "my_solver":
    {
        "exetension": ".parquet",
        "data":
        {
            "angle": "float",
            "radius": "float",
            "energy": "float"
        }
    }
}
"""
```

Still not anything too crazy. But it is important to define schema, otherwise the Pipeline can't make sure that everything is ok! Technically, you don't have to make them though (see the [docs](../api/core/phase.md)). The API documentation for schema can be found [here](../api/core/schema.md).

## Adding a new Phase

All Phases inherit from the [`PhaseLike`](../api/core/phase.md) abstract base class. As such your Phase will have to adhere to the predefined strutrue of `PhaseLike`. Lets look at what we would put in `my_phases/my_sovler_phase.py`.

```python
from spyral import PhaseLike, PhaseResult, ResultSchema, ESTIMATE_SCHEMA
from spyral.core.run_stacks import form_run_string

from .schema import MY_SOLVER_SCHEMA
from .config import MySolverParameters

from pathlib import Path
from multiprocessing import SimpleQueue

def generate_asset(path: Path) -> bool:
    # do some stuff
    return True

def load_asset(path: Path) -> SomeType:
    # load the asset in
    return SomeType

class MySolverPhase(PhaseLike):

    def __init__(self, solver_params: MySolverParameters):
        super().__init__(name="MySolver", incoming_schema=ResultSchema(ESTIMATION_SCHEMA), outgoing_schema=ResultSchema(MY_SOLVER_SCHEMA))
        self.params = solver_params

    def create_assets(self, workspace_path: Path) -> bool:
        asset_path = self.get_asset_storage_path(workspace_path) # Defined by PhaseLike
        self.my_asset = asset_path / self.params.c # Make a path to a specific asset
        return generate_asset(self.my_asset)

    def construct_artifact(
        self, payload: PhaseResult, workspace_path: Path
    ) -> PhaseResult:
        result = PhaseResult(
            artifacts={
                "my_solver": self.get_artifact_path(workspace_path)
            / f"{form_run_string(payload.run_number)}.h5", # form_run_string makes the standard AT-TPC run_#### string
            },
            successful=True,
            run_number=payload.run_number,
        )
        return result

    def run(
        self,
        payload: PhaseResult,
        workspace_path: Path,
        msg_queue: SimpleQueue,
        rng: np.random.Generator,
    ) -> PhaseResult:

        estimation_path = payload.artifact_path # This is the previous phase result

        result = self.construct_artifact(payload, workspace_path) # This is where we'll put our result

        # Load an asset
        asset = load_asset(self.my_asset)

        # Do a whole bunch of stuff!

        return result

```

This is a bit more complicated. There are four methods that **must** be defined for any phase. Most obvious is the constructor (`__init__`), where we take in our new parameters and store them to the Phase. In the constructor we also pass arguments to the base class constructor. The name is the name of the Phase, and **must** be unique for every phase of the Pipeline. The name is used to make folders in the workspace as well as for reporting progress. We also pass in our schema, where incoming in the expected schema of data from the previous phase and outgoing is the expected schema produced by this Phase.

Next is `create_assets`. This is where we would generate any extra data (interpolation schemes, electric field corrections, etc.) needed by this phase. This is called by the Pipeline before running. Assets can be stored as a memeber variable, or written to disk and loaded on the fly (as shown here).

Next is `construct_artifact`. This is a wrapper around creating a valid [`PhaseResult`](../api/core/schema.md), including a valid artifact path. This function is also what allows us to skip Phases! We can always ask a Phase where it would expect data from a given run to be stored.

Finally, the main function `run`. Here we take in the previous Phase's `PhaseResult` (here we're assuming it is the default EstimationPhase). We create an artifact using `self.construct_artifact`, load some assets, and away we go! We then return our artifact, for another phase to use if needed!

## Tying it all together

Now that we have our phase defined, it's time to implement it in a pipeline. Here is a modified version of our example script from [Getting Started](getting_started.md), which in our project we would save in `script.py`.

```python
from spyral import (
    Pipeline,
    start_pipeline,
    PointcloudPhase,
    ClusterPhase,
    EstimationPhase,
)
from spyral import (
    PadParameters,
    GetParameters,
    FribParameters,
    DetectorParameters,
    ClusterParameters,
    EstimateParameters,
    DEFAULT_MAP,
)

from pathlib import Path
import multiprocessing

# Import our custom stuff
from my_phases.my_solver_phase import MySolverPhase
from my_phases.config import MySolverParameters

workspace_path = Path("/path/to/your/workspace/")
trace_path = Path("/path/to/your/attpc/traces/")

run_min = 94
run_max = 94
n_processes = 4

pad_params = PadParameters(
    pad_geometry_path=DEFAULT_MAP,
    pad_gain_path=DEFAULT_MAP,
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
    cluster_selection_epsilon=0.3,
    min_cluster_size_join=15,
    circle_overlap_ratio=0.5,
    outlier_scale_factor=0.05,
)

estimate_params = EstimateParameters(
    min_total_trajectory_points=30, smoothing_factor=100.0
)

# Our custom parameters
solver_params = MySolverParameters(
    a=1.0,
    b=2.0,
    c="some_stuff"
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
        MySolverPhase(solver_params), # Our custom phase!
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

And that's it! We just added a custom phase to Spyral.

## Advanced topics

### Injecting a Phase

Adding Phases to the end of Spyral is relatively easy, because you don't have to cooperate with already existing phases. Things get slightly more complicated when you want to inject a Phase into the middle of the default pipeline. You have two options: modfiy every downstream phase, or force your new phase to mimic the schema of the previous phase.

### PhaseResult Metadata

Sort of taging along to the previous section, sometimes PhaseResults need to link two artifacts together. For example, the default InterpSolverPhase needs the cluster data  *and* the estimate data. To facilitate this, PhaseResult has a metadata field, which is a dictionary. You can shove basically anything you want into there, but it comes at a cost: the metadata can't be validated by the pipeline. We have no way to guarantee what is in there, it's all up to you the developer to verify that. With great power comes great responsibility or whatever.
