# Solver Configuration

The Solver parameters control the solving phase of the analysis.

```python
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
    fit_method="lbfgsb",
)
```

Note that these default values *will not work*. In particular the `gas_data_path` and `particle_id_file` parameters must be modified.

A break down of each parameter:

## gas_data_path

Path to a JSON file containing the following [spyral-utils](https://attpc.github.io/spyral-utils) format

## particle_id_filename

Name of a JSON file containing the following [spyral-utils](https://attpc.github.io/spyral-utils) format
Typically gates are made using the [particle ID notebook](https://github.com/ATTPC/spyral_notebooks).

## ic_min_val

The minimum value for the ion chamber amplitude. Used to make a gate on beam species.

## ic_max_val

The maximum value for the ion chamber amplitude. Used to make a gate on beam species.

## ode_times_steps

The number of timesteps for each individual solution. More timesteps is a finer-grained solution, providing more precision at the cost of higher memory
usage.

## interp_ke_min(MeV)

Minimum kinetic energy (MeV) of the ODE mesh for the solver. This is the minimum energy of a particle for which this solver will work. In units of
MeV.

## interp_ke_max(MeV)

Maximum kinetic energy (MeV) of the ODE mesh for the solver. This is the maximum energy of a particle for which this solver will work. In units of MeV.

## interp_ke_bins

Number of kinetic energy bins of the ODE mesh for the solver. This is the coarseness/fineness of the mesh in kinetic energy.

## interp_polar_min

Minimum polar angle (degrees) of the ODE mesh for the solver. This is the minimum polar angle of a particle for which this solver will work. In units
of degrees.

## interp_polar_max

Maximum polar angle (degrees) of the ODE mesh for the solver. This is the maximum polar angle of a particle for which this solver will work. In units
of degrees.

## interp_polar_bins

Number of polar angle bins of the ODE mesh for the solver. This is the coarseness/fineness of the mesh in polar angle.

## fit_vertex_rho

A boolean switch telling the solver whether or not the vertex &rho; should be included in the fit. In testing the simulation has indicated that the
poisition of the vertex in the x-y plane (for the fit we use rho and phi because they are easier to define bounds for) is not well constrained and may
cause overfitting. Data analysis has indicated *somewhat* the other direction, that it does seem to have some impact. User discretion should be used to
determine if this parameter is important for your data. True turns on this variable for the fit, False holds it constant.

## fit_vertex_phi

A boolean switch telling the solver whether or not the vertex &phi; should be included in the fit. In testing the simulation has indicated that the
poisition of the vertex in the x-y plane (for the fit we use rho and phi because they are easier to define bounds for) is not well constrained and may
cause overfitting. Data analysis has indicated *somewhat* the other direction, that it does seem to have some impact. User discretion should be used
to determine if this parameter is important for your data. True turns on this variable for the fit, False holds it constant.

## fit_azimuthal

A boolean switch telling the solver whether or not the trajectory azimuthal angle; should be included in the fit. In testing the simulation has
indicated that the azimuthal angle is not impacted by the fit and may cause overfitting. Data analysis has indicated *somewhat* the other direction,
that it does seem to have some impact. User discretion should be used to determine if this parameter is important for your data. True turns on this
variable for the fit, False holds it constant.

## fit_method

Which fit method to use. Options are "lbfgsb" or "leastsq". See the [phase docs](../phases/solve.md) for more details.

