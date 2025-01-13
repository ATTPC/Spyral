from ..core.phase import PhaseLike
from ..core.schema import PhaseResult, ResultSchema
from ..core.config import SolverParameters, DetectorParameters
from ..core.status_message import StatusMessage
from ..core.cluster import Cluster
from ..core.spy_log import spyral_warn, spyral_error, spyral_info
from ..core.run_stacks import form_run_string
from ..core.track_generator import (
    generate_track_mesh,
    MeshParameters,
    check_mesh_metadata,
)
from ..core.estimator import Direction
from ..interpolate.track_interpolator import (
    create_interpolator_from_array,
)
from ..solvers.guess import Guess
from ..solvers.solver_interp import solve_physics_interp
from ..solvers.solver_interp_leastsq import solve_physics_interp_leastsq
from .schema import ESTIMATE_SCHEMA, INTERP_SOLVER_SCHEMA

from spyral_utils.nuclear.target import load_target, GasTarget, GasMixtureTarget
from spyral_utils.nuclear.particle_id import deserialize_particle_id, ParticleID
from spyral_utils.nuclear.nuclear_map import NuclearDataMap
import h5py as h5
import polars as pl
from pathlib import Path
from multiprocessing import SimpleQueue
from numpy.random import Generator
import numpy as np
from multiprocessing.shared_memory import SharedMemory

DEFAULT_PID_XAXIS = "dEdx"
DEFAULT_PID_YAXIS = "brho"
METHOD_LBFGSB = "lbfgsb"
METHOD_LEASTSQ = "leastsq"


class InterpSolverError(Exception):
    pass


def form_physics_file_name(run_number: int, particle: ParticleID) -> str:
    """Form a physics file string

    Physics files are run number + solved for isotope

    Parameters
    ----------
    run_number: int
        The run number
    particle: ParticleID
        The particle ID used by the solver.

    Returns
    -------
    str
        The run file string

    """
    return f"{form_run_string(run_number)}_{particle.nucleus.isotopic_symbol}.parquet"


class InterpSolverPhase(PhaseLike):
    """The default Spyral solver phase, inheriting from PhaseLike

    The goal of the solver phase is to get exact (or as exact as possible) values
    for the physical observables of a trajectory. InterpSolverPhase uses a pre-calculated
    mesh of ODE solutions to interpolate a model particle trajectory for a given set of
    kinematics (energy, angles, vertex) and fit the best model trajectory to the data. InterpSolverPhase
    is expected to be run after the EstimationPhase.

    Parameters
    ----------
    solver_params: SolverParameters
        Parameters controlling the interpolation mesh and fitting
    det_params: DetectorParameters
        Parameters describing the detector

    Attributes
    ----------
    solver_params: SolverParameters
        Parameters controlling the interpolation mesh and fitting
    det_params: DetectorParameters
        Parameters describing the detector
    nuclear_map: spyral_utils.nuclear.NuclearDataMap
        A map containing isotopic information
    track_path: pathlib.Path
        Path to the ODE solution mesh
    """

    def __init__(self, solver_params: SolverParameters, det_params: DetectorParameters):
        super().__init__(
            "InterpSolver",
            incoming_schema=ResultSchema(ESTIMATE_SCHEMA),
            outgoing_schema=ResultSchema(INTERP_SOLVER_SCHEMA),
        )
        self.solver_func = None
        if solver_params.fit_method == METHOD_LBFGSB:
            self.solver_func = solve_physics_interp
        elif solver_params.fit_method == METHOD_LEASTSQ:
            self.solver_func = solve_physics_interp_leastsq
        else:
            raise InterpSolverError(
                f"Fit method {solver_params.fit_method} is not a valid method. Please chose '{METHOD_LBFGSB}' or '{METHOD_LEASTSQ}'"
            )

        self.solver_params = solver_params
        self.det_params = det_params
        self.nuclear_map = NuclearDataMap()
        self.shared_mesh_shape: tuple | None = None
        self.shared_mesh_name: str = "interp_shared_mesh"
        self.shared_mesh_type: np.dtype | None = None

    def create_assets(self, workspace_path: Path) -> bool:
        target = load_target(Path(self.solver_params.gas_data_path), self.nuclear_map)
        pid = deserialize_particle_id(
            Path(self.solver_params.particle_id_filename), self.nuclear_map
        )
        if pid is None:
            raise InterpSolverError(
                "Could not create trajectory mesh, particle ID is not formatted correctly!"
            )
        if not isinstance(target, GasTarget | GasMixtureTarget):
            raise InterpSolverError(
                "Could not create trajectory mesh, target is not a GasTarget | GasMixtureTarget!"
            )
        mesh_params = MeshParameters(
            target,
            pid.nucleus,
            self.det_params.magnetic_field,
            self.det_params.electric_field,
            self.solver_params.n_time_steps,
            self.solver_params.interp_ke_min,
            self.solver_params.interp_ke_max,
            self.solver_params.interp_ke_bins,
            self.solver_params.interp_polar_min,
            self.solver_params.interp_polar_max,
            self.solver_params.interp_polar_bins,
        )
        self.track_path = (
            self.get_asset_storage_path(workspace_path)
            / mesh_params.get_track_file_name()
        )
        meta_path = (
            self.get_asset_storage_path(workspace_path)
            / mesh_params.get_track_meta_file_name()
        )
        matches = check_mesh_metadata(self.track_path, mesh_params)
        if not matches:
            generate_track_mesh(mesh_params, self.track_path, meta_path)
        # the shape and type attrs are filled in either by the checked metadata
        # or during mesh generation
        # Side-effecty-ness is annoying, but currently unavoidable
        self.shared_mesh_shape = mesh_params.shape
        self.shared_mesh_type = np.dtype(mesh_params.dtype)

        return True

    def create_shared_data(
        self, workspace_path: Path, handles: dict[str, SharedMemory]
    ) -> None:
        mesh_data: np.ndarray = np.load(self.track_path)
        # Check expected mesh properties
        if mesh_data.shape != self.shared_mesh_shape:
            raise InterpSolverError(
                f"Mesh shape {mesh_data.shape} did not match metadata shape {self.shared_mesh_shape}!"
            )
        if mesh_data.dtype != self.shared_mesh_type:
            raise InterpSolverError(
                f"Mesh type {mesh_data.dtype} did not match metadata type {self.shared_mesh_type}!"
            )
        # Create a block of shared memory of the same total size as the mesh
        # Note that we don't have a lock on the shared memory as the mesh is
        # used read-only
        # Dragon hack: Dragon does not currently implement SharedMemory, so for now
        # we fall back to the base multiprocessing impl when we receive a
        # NotImplementedError
        handle = None
        try:
            handle = SharedMemory(
                name=self.shared_mesh_name, create=True, size=mesh_data.nbytes
            )
        except NotImplementedError:
            # Hacky
            MultiprocessingSharedMemory = SharedMemory.__bases__[0]
            handle = MultiprocessingSharedMemory(
                name=self.shared_mesh_name, create=True, size=mesh_data.nbytes
            )

        handles[handle.name] = handle
        spyral_info(
            __name__,
            f"Allocated {mesh_data.nbytes * 1.0e-9:.2} GB of memory for shared mesh.",
        )
        memory_array = np.ndarray(
            mesh_data.shape, dtype=mesh_data.dtype, buffer=handle.buf
        )
        memory_array[:, :, :, :] = mesh_data[:, :, :, :]

    def construct_artifact(
        self, payload: PhaseResult, workspace_path: Path
    ) -> PhaseResult:
        if not self.solver_params.particle_id_filename.exists():
            spyral_warn(
                __name__,
                f"Particle ID {self.solver_params.particle_id_filename} does not exist, Solver will not run!",
            )
            return PhaseResult.invalid_result(payload.run_number)

        pid: ParticleID | None = deserialize_particle_id(
            Path(self.solver_params.particle_id_filename), self.nuclear_map
        )
        if pid is None:
            spyral_warn(
                __name__,
                f"Particle ID {self.solver_params.particle_id_filename} is not valid, Solver will not run!",
            )
            return PhaseResult.invalid_result(payload.run_number)

        result = PhaseResult(
            artifacts={
                "interp_solver": self.get_artifact_path(workspace_path)
                / form_physics_file_name(payload.run_number, pid)
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
        rng: Generator,
    ) -> PhaseResult:
        if self.solver_func is None:
            raise InterpSolverError(
                "Solver function was not set at Pipeline run for InterpSolverPhase!"
            )
        # Need particle ID the correct data subset
        pid: ParticleID | None = deserialize_particle_id(
            Path(self.solver_params.particle_id_filename), self.nuclear_map
        )
        if pid is None:
            msg_queue.put(StatusMessage("Waiting", 0, 0, payload.run_number))
            spyral_warn(
                __name__,
                f"Particle ID {self.solver_params.particle_id_filename} does not exist, Solver will not run!",
            )
            return PhaseResult.invalid_result(payload.run_number)

        # Check the cluster phase and estimate phase data
        cluster_path: Path = payload.artifacts["cluster"]
        estimate_path = payload.artifacts["estimation"]
        if not cluster_path.exists() or not estimate_path.exists():
            msg_queue.put(StatusMessage("Waiting", 0, 0, payload.run_number))
            spyral_warn(
                __name__,
                f"Either clusters or esitmates do not exist for run {payload.run_number} at phase 4. Skipping.",
            )
            return PhaseResult.invalid_result(payload.run_number)

        # Setup files
        result = self.construct_artifact(payload, workspace_path)
        cluster_file = h5.File(cluster_path, "r")
        estimate_df = pl.scan_parquet(estimate_path)

        cluster_group: h5.Group = cluster_file["cluster"]  # type: ignore
        if not isinstance(cluster_group, h5.Group):
            spyral_error(
                __name__,
                f"Cluster group does not eixst for run {payload.run_number} at phase 4!",
            )
            return PhaseResult.invalid_result(payload.run_number)

        # Extract cut axis names if present. Otherwise default to dEdx, brho
        xaxis = DEFAULT_PID_XAXIS
        yaxis = DEFAULT_PID_YAXIS
        if not pid.cut.is_default_x_axis() and not pid.cut.is_default_y_axis():
            xaxis = pid.cut.get_x_axis()
            yaxis = pid.cut.get_y_axis()

        # Select the particle group data, beam region of ic, convert to dictionary for row-wise operations
        estimates_gated = (
            estimate_df.filter(
                pl.struct([xaxis, yaxis]).map_batches(pid.cut.is_cols_inside)
                & (pl.col("ic_amplitude") > self.solver_params.ic_min_val)
                & (pl.col("ic_amplitude") < self.solver_params.ic_max_val)
            )
            .collect()
            .to_dict()
        )

        # Check that data actually exists for given PID
        if len(estimates_gated["event"]) == 0:
            msg_queue.put(StatusMessage("Waiting", 0, 0, payload.run_number))
            spyral_warn(__name__, f"No events within PID for run {payload.run_number}!")
            return PhaseResult.invalid_result(payload.run_number)

        nevents = len(estimates_gated["event"])
        total: int
        flush_val: int
        if nevents < 100:
            total = nevents
            flush_val = 0
        else:
            flush_percent = 0.01
            flush_val = int(flush_percent * (nevents))
            total = 100

        count = 0

        msg = StatusMessage(
            "Interp. Solver", 1, total, payload.run_number
        )  # We always increment by 1

        # Result storage
        phys_results = []

        # load the ODE solution interpolator
        if self.shared_mesh_shape is None or self.shared_mesh_name is None:
            spyral_warn(
                __name__,
                "Could not run the interpolation scheme as there is no shared memory!",
            )
            return PhaseResult.invalid_result(payload.run_number)
        # Ask for the shared memory by name and cast it to a numpy array of the correct shape
        # Hacking continues
        mesh_buffer = None
        try:
            mesh_buffer = SharedMemory(self.shared_mesh_name)
        except NotImplementedError:
            MultiprocessingSharedMemory = SharedMemory.__bases__[0]
            mesh_buffer = MultiprocessingSharedMemory(self.shared_mesh_name)

        mesh_handle = np.ndarray(
            self.shared_mesh_shape, dtype=self.shared_mesh_type, buffer=mesh_buffer.buf
        )
        mesh_handle.setflags(write=False)
        # Create our interpolator
        interpolator = create_interpolator_from_array(self.track_path, mesh_handle)

        # Process the data
        for row, event in enumerate(estimates_gated["event"]):
            count += 1
            if count > flush_val:
                count = 0
                msg_queue.put(msg)

            event_group = cluster_group[f"event_{event}"]
            cidx = estimates_gated["cluster_index"][row]
            local_cluster: h5.Dataset = event_group[f"cluster_{cidx}"]  # type: ignore
            cluster = Cluster(
                event,
                local_cluster.attrs["label"],  # type: ignore
                local_cluster["cloud"][:].copy(),  # type: ignore
            )
            orig_run = estimates_gated["orig_run"][row]
            orig_event = estimates_gated["orig_event"][row]

            # Do the solver
            guess = Guess(
                estimates_gated["polar"][row],
                estimates_gated["azimuthal"][row],
                estimates_gated["brho"][row],
                estimates_gated["vertex_x"][row],
                estimates_gated["vertex_y"][row],
                estimates_gated["vertex_z"][row],
                Direction.NONE,  # type: ignore
            )
            res = self.solver_func(
                cidx,
                orig_run,
                orig_event,
                cluster,
                guess,
                pid.nucleus,
                interpolator,
                self.det_params,
                self.solver_params,
            )
            if res is not None:
                phys_results.append(vars(res))

        # Write out the results
        physics_df = pl.DataFrame(phys_results)
        physics_df.write_parquet(result.artifacts["interp_solver"])
        spyral_info(
            __name__, f"Phase InterpSolver complete for run {payload.run_number}"
        )
        # Close the handle
        mesh_buffer.close()
        return result
