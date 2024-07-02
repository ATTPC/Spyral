from ..core.phase import PhaseLike, PhaseResult
from ..core.config import ClusterParameters, DetectorParameters
from ..core.status_message import StatusMessage
from ..core.point_cloud import PointCloud
from ..core.clusterize import form_clusters, join_clusters, cleanup_clusters
from ..core.spy_log import spyral_warn, spyral_error, spyral_info
from ..core.run_stacks import form_run_string
from .schema import POINTCLOUD_SCHEMA, CLUSTER_SCHEMA

import h5py as h5
from pathlib import Path
from multiprocessing import SimpleQueue
from numpy.random import Generator


class ClusterPhase(PhaseLike):
    """The default Spyral clustering phase, inheriting from PhaseLike

    The goal of the clustering phase is to take in a point cloud
    and separate the points into individual particle trajectories. In
    the default version here, we use scikit-learn's HDBSCAN clustering
    algorithm. The clustering phase should come after the Pointcloud/PointcloudLegacy
    Phase in the Pipeline and before the EstimationPhase.

    Parameters
    ----------
    cluster_params: ClusterParameters
        Parameters controlling the clustering algorithm
    det_params: DetectorParameters
        Parameters describing the detector

    Attributes
    ----------
    cluster_params: ClusterParameters
        Parameters controlling the clustering algorithm
    det_params: DetectorParameters
        Parameters describing the detector

    """

    def __init__(
        self, cluster_params: ClusterParameters, det_params: DetectorParameters
    ) -> None:
        super().__init__(
            "Cluster", incoming_schema=POINTCLOUD_SCHEMA, outgoing_schema=CLUSTER_SCHEMA
        )
        self.cluster_params = cluster_params
        self.det_params = det_params

    def create_assets(self, workspace_path: Path) -> bool:
        return True

    def construct_artifact(
        self, payload: PhaseResult, workspace_path: Path
    ) -> PhaseResult:
        result = PhaseResult(
            artifact_path=self.get_artifact_path(workspace_path)
            / f"{form_run_string(payload.run_number)}.h5",
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
        # Check that point clouds exist
        point_path = payload.artifact_path
        if not point_path.exists() or not payload.successful:
            spyral_warn(
                __name__,
                f"Point cloud data does not exist for run {payload.run_number} at phase 2. Skipping.",
            )
            return PhaseResult.invalid_result(payload.run_number)

        result = self.construct_artifact(payload, workspace_path)

        point_file = h5.File(point_path, "r")
        cluster_file = h5.File(result.artifact_path, "w")

        cloud_group: h5.Group = point_file["cloud"]  # type: ignore
        if not isinstance(cloud_group, h5.Group):
            spyral_error(
                __name__, f"Point cloud group not present in run {payload.run_number}!"
            )
            return PhaseResult.invalid_result(payload.run_number)

        min_event: int = cloud_group.attrs["min_event"]  # type: ignore
        max_event: int = cloud_group.attrs["max_event"]  # type: ignore
        cluster_group: h5.Group = cluster_file.create_group("cluster")
        cluster_group.attrs["min_event"] = min_event
        cluster_group.attrs["max_event"] = max_event

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

        msg = StatusMessage(
            self.name, 1, total, payload.run_number
        )  # we always increment by 1

        # Process the data
        for idx in range(min_event, max_event + 1):
            count += 1
            if count > flush_val:
                count = 0
                msg_queue.put(msg)

            cloud_data: h5.Dataset | None = None
            cloud_name = f"cloud_{idx}"
            if cloud_name not in cloud_group:
                continue
            else:
                cloud_data = cloud_group[cloud_name]  # type: ignore

            if cloud_data is None:
                continue

            cloud = PointCloud()
            cloud.load_cloud_from_hdf5_data(cloud_data[:].copy(), idx)

            # Here we don't need to use the labels array.
            # We just pass it along as needed.
            clusters, labels = form_clusters(cloud, self.cluster_params)
            joined, labels = join_clusters(clusters, self.cluster_params, labels)
            cleaned, _ = cleanup_clusters(joined, self.cluster_params, labels)

            # Each event can contain many clusters
            cluster_event_group = cluster_group.create_group(f"event_{idx}")
            cluster_event_group.attrs["nclusters"] = len(cleaned)
            cluster_event_group.attrs["ic_amplitude"] = cloud_data.attrs["ic_amplitude"]
            cluster_event_group.attrs["ic_centroid"] = cloud_data.attrs["ic_centroid"]
            cluster_event_group.attrs["ic_integral"] = cloud_data.attrs["ic_integral"]
            cluster_event_group.attrs["ic_multiplicity"] = cloud_data.attrs[
                "ic_multiplicity"
            ]
            for cidx, cluster in enumerate(cleaned):
                local_group = cluster_event_group.create_group(f"cluster_{cidx}")
                local_group.attrs["label"] = cluster.label
                local_group.create_dataset("cloud", data=cluster.data)

        spyral_info(__name__, "Phase 2 complete.")
        return result
