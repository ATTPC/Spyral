from ..core.phase import ArtifactSchema

HDF5_EXTENSION = ".h5"
PARQUET_EXTENSION = ".parquet"

TRACE_SCHEMA = ArtifactSchema(HDF5_EXTENSION, {"get": "evt", "frib": {"evt", "scaler"}})

POINTCLOUD_SCHEMA = ArtifactSchema(HDF5_EXTENSION, {"cloud": "cloud"})

CLUSTER_SCHEMA = ArtifactSchema(HDF5_EXTENSION, {"cluster": {"event": "cluster"}})

ESTIMATE_SCHEMA = ArtifactSchema(
    PARQUET_EXTENSION,
    [
        "event" "cluster_index",
        "cluster_label",
        "ic_amplitude",
        "ic_centroid",
        "ic_integral",
        "ic_multiplicity",
        "vertex_x",
        "vertex_y",
        "vertex_z",
        "center_x",
        "center_y",
        "center_z",
        "polar",
        "azimuthal",
        "brho",
        "dEdx",
        "dE",
        "arclength",
        "direction",
    ],
)

INTERP_SOLVER_SCHEMA = ArtifactSchema(
    PARQUET_EXTENSION,
    [
        "event",
        "cluster_index",
        "cluster_label",
        "vertex_x",
        "sigma_vx",
        "vertex_y",
        "sigma_vy",
        "vertex_z",
        "sigma_vz",
        "brho",
        "sigma_brho",
        "polar",
        "sigma_polar",
        "azimuthal",
        "sigma_azimuthal",
        "redchisq",
    ],
)
