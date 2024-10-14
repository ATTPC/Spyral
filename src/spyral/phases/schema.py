HDF5_EXTENSION = ".h5"
PARQUET_EXTENSION = ".parquet"

TRACE_SCHEMA = """
{
    "trace":
    {
        "extension": ".h5",
        "data":
        {
            "get": "array",
            "frib": "array"
        }
    }
}
"""

POINTCLOUD_SCHEMA = """
{
    "pointcloud":
    {
        "extension": ".h5",
        "data":
        {
            "cloud":
            {
                "cloud": "array"
            }
        }
    }
}
"""

CLUSTER_SCHEMA = """
{
    "cluster":
    {
        "extension": ".h5",
        "data":
        {
            "cluster":
            {
                "event": 
                {
                    "cluster": "array"
                }
            }
        }
    }
}
"""

ESTIMATE_SCHEMA = """
{
    "estimation": 
    {
        "extension": ".parquet",
        "data":
        {
            "event": "int", 
            "cluster_index": "int",
            "cluster_label": "int",
            "ic_amplitude": "float",
            "ic_centroid": "float",
            "ic_integral": "float",
            "ic_multiplicity": "float",
            "vertex_x": "float",
            "vertex_y": "float",
            "vertex_z": "float",
            "center_x": "float",
            "center_y": "float",
            "center_z": "float",
            "polar": "float",
            "azimuthal": "float",
            "brho": "float",
            "dEdx": "float",
            "sqrt_dEdx": "float",
            "dE": "float",
            "arclength": "float",
            "direction": "int"
        }
    },
    "cluster":
    {
        "extension": ".h5",
        "data":
        {
            "cluster":
            {
                "event": 
                {
                    "cluster": "array"
                }
            }
        }
    }
}
"""

INTERP_SOLVER_SCHEMA = """
{
    "interp_solver":
    {
        "extension": ".parquet",
        "data":
        {
            "event": "int",
            "cluster_index": "int",
            "cluster_label": "int",
            "vertex_x": "float",
            "sigma_vx": "float",
            "vertex_y": "float",
            "sigma_vy": "float",
            "vertex_z": "float",
            "sigma_vz": "float",
            "brho": "float",
            "sigma_brho": "float",
            "ke": "float",
            "sigma_ke": "float",
            "polar": "float",
            "sigma_polar": "float",
            "azimuthal": "float",
            "sigma_azimuthal": "float",
            "redchisq": "float"
        }
    }
}
"""
