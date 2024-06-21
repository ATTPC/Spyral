from ..core.phase import PhaseLike, PhaseResult
from ..core.config import EstimateParameters, DetectorParameters
from ..core.status_message import StatusMessage
from ..core.cluster import Cluster
from ..core.estimator import estimate_physics
from ..core.spy_log import spyral_warn, spyral_error, spyral_info
from ..core.run_stacks import form_run_string
from .schema import CLUSTER_SCHEMA, ESTIMATE_SCHEMA

import h5py as h5
import polars as pl
from pathlib import Path
from multiprocessing import SimpleQueue
from numpy.random import Generator


class EstimationPhase(PhaseLike):
    """The default Spyral estimation phase, inheriting from PhaseLike

    The goal of the estimation phase is to get reasonable estimations of
    the physical properties of a particle trajectory (B&rho; , reaction angle, etc.)
    for use in the more complex solving phase to follow. EstimationPhase should come
    after ClusterPhase and before InterpSolverPhase in the Pipeline.

    Parameters
    ----------
    estimate_params: EstimateParameters
        Parameters controlling the estimation algorithm
    det_params: DetectorParameters
        Parameters describing the detector

    Attributes
    ----------
    estimate_params: EstimateParameters
        Parameters controlling the estimation algorithm
    det_params: DetectorParameters
        Parameters describing the detector

    """

    def __init__(
        self, estimate_params: EstimateParameters, det_params: DetectorParameters
    ):
        super().__init__(
            "Estimation",
            incoming_schema=CLUSTER_SCHEMA,
            outgoing_schema=ESTIMATE_SCHEMA,
        )
        self.estimate_params = estimate_params
        self.det_params = det_params

    def create_assets(self, workspace_path: Path) -> bool:
        return True

    def construct_artifact(
        self, payload: PhaseResult, workspace_path: Path
    ) -> PhaseResult:
        result = PhaseResult(
            artifact_path=self.get_artifact_path(workspace_path)
            / f"{form_run_string(payload.run_number)}.parquet",
            successful=True,
            run_number=payload.run_number,
            metadata={"cluster_path": payload.artifact_path},
        )
        return result

    def run(
        self,
        payload: PhaseResult,
        workspace_path: Path,
        msg_queue: SimpleQueue,
        rng: Generator,
    ) -> PhaseResult:
        # Check that clusters exist
        cluster_path = payload.artifact_path
        if not cluster_path.exists() or not payload.successful:
            spyral_warn(
                __name__,
                f"Cluster file for run {payload.run_number} not present for phase 3. Skipping.",
            )
            return PhaseResult.invalid_result(payload.run_number)

        result = self.construct_artifact(payload, workspace_path)

        cluster_file = h5.File(cluster_path, "r")
        cluster_group: h5.Group = cluster_file["cluster"]  # type: ignore
        if not isinstance(cluster_group, h5.Group):
            spyral_error(
                __name__, f"Cluster group not present for run {payload.run_number}!"
            )
            return PhaseResult.invalid_result(payload.run_number)

        min_event: int = cluster_group.attrs["min_event"]  # type: ignore
        max_event: int = cluster_group.attrs["max_event"]  # type: ignore

        nevents = max_event - min_event + 1
        total: int
        flush_val: int
        if nevents < 100:
            total = nevents
            flush_val = 0
        else:
            flush_percent = 0.01
            flush_val = int(flush_percent * (max_event - min_event))
            total = 100

        count = 0

        # estimation results
        data: dict[str, list] = {
            "event": [],
            "cluster_index": [],
            "cluster_label": [],
            "ic_amplitude": [],
            "ic_centroid": [],
            "ic_integral": [],
            "ic_multiplicity": [],
            "vertex_x": [],
            "vertex_y": [],
            "vertex_z": [],
            "center_x": [],
            "center_y": [],
            "center_z": [],
            "polar": [],
            "azimuthal": [],
            "brho": [],
            "dEdx": [],
            "dE": [],
            "arclength": [],
            "direction": [],
        }

        msg = StatusMessage(
            self.name, 1, total, payload.run_number
        )  # We always increment by 1
        # Process data
        for idx in range(min_event, max_event + 1):
            count += 1
            if count > flush_val:
                count = 0
                msg_queue.put(msg)

            event: h5.Group | None = None
            event_name = f"event_{idx}"
            if event_name not in cluster_group:
                continue
            else:
                event = cluster_group[event_name]  # type: ignore

            nclusters: int = event.attrs["nclusters"]  # type: ignore
            ic_amp = float(event.attrs["ic_amplitude"])  # type: ignore
            ic_cent = float(event.attrs["ic_centroid"])  # type: ignore
            ic_int = float(event.attrs["ic_integral"])  # type: ignore
            ic_mult = float(event.attrs["ic_multiplicity"])  # type: ignore
            # Go through every cluster in each event
            for cidx in range(0, nclusters):
                local_cluster: h5.Group | None = None
                cluster_name = f"cluster_{cidx}"
                if cluster_name not in event:  # type: ignore
                    continue
                else:
                    local_cluster = event[cluster_name]  # type: ignore

                cluster = Cluster(
                    idx, local_cluster.attrs["label"], local_cluster["cloud"][:].copy()  # type: ignore
                )

                # Cluster is loaded do some analysis
                estimate_physics(
                    cidx,
                    cluster,
                    ic_amp,
                    ic_cent,
                    ic_int,
                    ic_mult,
                    self.estimate_params,
                    self.det_params,
                    data,
                )

        # Write the results to a DataFrame
        df = pl.DataFrame(data)
        df.write_parquet(result.artifact_path)
        spyral_info(__name__, "Phase 3 complete.")
        # Next step also needs to know where to find the clusters
        return result
