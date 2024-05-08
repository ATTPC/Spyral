from spyral import Pipeline, start_pipeline, PointcloudPhase, ClusterPhase
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

workspace_path = Path("/media/gordon/ThesisData/NewData/Pipeline/")
trace_path = Path("/media/gordon/ThesisData/NewData/a1975/h5/")

run_min = 94
run_max = 94
n_processes = 4

pad_params = PadParameters(
    is_default=True,
    is_default_legacy=False,
    pad_geometry_path=INVALID_PATH,
    pad_gain_path=INVALID_PATH,
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
    cluster_selection_epsilon=0.3,
    circle_overlap_ratio=0.5,
    fractional_charge_threshold=0.8,
    outlier_scale_factor=0.05,
)

estimate_params = EstimateParameters(
    min_total_trajectory_points=30, smoothing_factor=1.0
)

solver_params = SolverParameters(
    gas_data_path=Path("/path/to/some/gas/data.json"),
    particle_id_filename=Path("/path/to/some/particle/id.json"),
    ic_min_val=900.0,
    ic_max_val=1350.0,
    n_time_steps=10000,
    interp_ke_min=0.05,
    interp_ke_max=70.0,
    interp_ke_bins=400,
    interp_polar_min=2.0,
    interp_polar_max=88.0,
    interp_polar_bins=170,
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
    ],
    workspace_path,
    trace_path,
)


def main():
    start_pipeline(pipe, run_min, run_max, n_processes)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
