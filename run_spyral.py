import dotenv
dotenv.load_dotenv()

from spyral import (
    Pipeline,
    start_pipeline,
    PointcloudPhase,
    ClusterPhase,
    OverlapJoinParameters,
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
)

from pathlib import Path
import multiprocessing

workspace_path = Path("/Volumes/researchEXT/O16/no_efield/no_field_only_1_2_tracks/")
trace_path = Path("/path/to/your/attpc/traces/")

run_min = 53
run_max = 54
n_processes = 5

pad_params = PadParameters(
    pad_geometry_path=DEFAULT_MAP,
    pad_time_path=DEFAULT_MAP,
    pad_scale_path=DEFAULT_MAP,
)

get_params = GetParameters(
    baseline_window_scale=20.0,
    peak_separation=50.0, #set to 5 in Zach's version 
    peak_prominence=20.0,
    peak_max_width=100.0, #changed from 50.0 to 100.0
    peak_threshold=100.0, #changed from 40.0 to 100.0
)

frib_params = FribParameters(
    baseline_window_scale=100.0,
    peak_separation=50.0,  #set to 5 in Zach's version 
    peak_prominence=30.0, #changed from 20.0 to 30.0
    peak_max_width=20.0, #changed from 500.0 to 20.0
    peak_threshold=300.0, #changed from 100.0 to 300.0
    ic_delay_time_bucket=1100,
    ic_multiplicity=1,
)

det_params = DetectorParameters(
    magnetic_field=3.0,
    electric_field=57260.0,
    detector_length=1000.0,
    beam_region_radius=20.0,
    micromegas_time_bucket=91.98,
    window_time_bucket=469.21,
    get_frequency=3.125, #6.25 wasn't for this experiment
    garfield_file_path=Path("/mnt/home/singhp19/O16_driftvel_analysis/e20020_analysis/e20009_parameters/electrons_e20020.txt"),
    do_garfield_correction=False, #False for now 
)

cluster_params = ClusterParameters(
    min_cloud_size=20,
    min_points=3,
    min_size_scale_factor=0.05,
    min_size_lower_cutoff=10,
    cluster_selection_epsilon=10.0,
    overlap=OverlapJoinParameters(
        min_cluster_size_join=15,
        circle_overlap_ratio=0.25,
    ),
    continuity=None,
    outlier_scale_factor=0.05,
)

estimate_params = EstimateParameters(
    min_total_trajectory_points=20, smoothing_factor=100.0
)

solver_params = SolverParameters(
    gas_data_path=Path("/mnt/home/singhp19/O16_driftvel_analysis/e20020_analysis/solver_gas_16O.json"),
    particle_id_filename=Path("/mnt/home/singhp19/O16_driftvel_analysis/e20020_analysis/solver_particle_16O.json"),
    ic_min_val=300.0,
    ic_max_val=850.0,
    n_time_steps=1300,
    interp_ke_min=0.01,
    interp_ke_max=40.0,
    interp_ke_bins=800,
    interp_polar_min=0.1,
    interp_polar_max=179.9,
    interp_polar_bins=500,
    fit_vertex_rho=True,
    fit_vertex_phi=True,
    fit_azimuthal=True,
    fit_method="lbfgsb",
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
    [False, True, True, False],
    workspace_path,
    trace_path,
)


def main():
    start_pipeline(pipe, run_min, run_max, n_processes)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()