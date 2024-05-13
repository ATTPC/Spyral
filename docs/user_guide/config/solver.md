# Solver Configuration

The Solver parameters control the solving phase of the analysis. The default solver parameters in `config.json` are:

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
)
```

Note that these default values *will not work*. In particular the `gas_data_path` and `particle_id_file` parameters must be modified.

A break down of each parameter:

## gas_data_path

Path to a JSON file containing the following [spyral-utils](https://attpc.github.io/spyral-utils) format

```json
{
    "compound": [
        [1, 2, 2]
    ],
    "pressure(Torr)": 300.0,
    "thickness(ug/cm^2)": null
}
```

`compound` is a list of [Z, A, S] (atomic number, mass number, stoichiometry) specifying the compound of the gas. `pressure(Torr)` is the gas pressure in Torr. `thickness` is not used and should be set to `null`. This completely specifies the active target gas to Spyral. The above data describes <sup>1</sup>H<sub>2</sub> gas at 300 Torr.

## particle_id_filename

Name of a JSON file containing the following [spyral-utils](https://attpc.github.io/spyral-utils) format

```json
{
    "name": "my_pid",
    "Z": 1,
    "A": 1,
    "vertices": [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.0, 0.0]
    ]
}
```

`name` is the name of the gate, `Z` is the atomic number of the particle in the gate, `A` is the mass number, and `verticies` is a list of points which form a closed polygon in B&rho;-dEdx. The above data describes protons in a very silly box. Note that the first and last point in verticies should be the same. Typically gates are made using the particle ID notebook.

## ic_min_val

The minimum value for the ion chamber amplitude. Used to make a gate on beam species.

## ic_max_val

The maximum value for the ion chamber amplitude. Used to make a gate on beam species.

## ode_times_steps

The number of timesteps for each individual solution. More timesteps is a finer-grained solution, providing more precision at the cost of higher memory usage.

## interp_ke_min(MeV)

Minimum kinetic energy (MeV) of the ODE mesh for the solver. This is the minimum energy of a particle for which this solver will work. In units of MeV.

## interp_ke_max(MeV)

Maximum kinetic energy (MeV) of the ODE mesh for the solver. This is the maximum energy of a particle for which this solver will work. In units of MeV.

## interp_ke_bins

Number of kinetic energy bins of the ODE mesh for the solver. This is the coarseness/fineness of the mesh in kinetic energy.

## interp_polar_min

Minimum polar angle (degrees) of the ODE mesh for the solver. This is the minimum polar angle of a particle for which this solver will work. In units of degrees.

## interp_polar_max

Maximum polar angle (degrees) of the ODE mesh for the solver. This is the maximum polar angle of a particle for which this solver will work. In units of degrees.

## interp_polar_bins

Number of polar angle bins of the ODE mesh for the solver. This is the coarseness/fineness of the mesh in polar angle.
