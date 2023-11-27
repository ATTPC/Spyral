# Solver Configuration

The Solver parameters control the solving phase of the analysis. The default solver parameters in `config.json` are:

```json
"Solver":
{
    "gas_data_path": "/path/to/some/gas.json",
    "particle_id_file": "some_pid.json",
    "ic_min": 950.0,
    "ic_max": 1350.0,
    "interp_file_name": "some_tracks.npy",
    "interp_ke_min(MeV)": 0.1,
    "interp_ke_max(MeV)": 70.0,
    "interp_ke_bins": 350,
    "interp_polar_min(deg)": 5.0,
    "interp_polar_max(deg)": 88.0,
    "interp_polar_bins": 166
}
```

Note that these default values *will not work*. In particular the `gas_data_path`, `particle_id_file`, and `interp_file_name` parameters must be modified.

A break down of each parameter:

## gas_data_path

Path to a JSON file containing the following format

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

## particle_id_file

Name of a JSON file containing the following format

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

`name` is the name of the gate, `Z` is the atomic number of the particle in the gate, `A` is the mass number, and `verticies` is a list of points which form a closed polygon in B&rho;-dEdx. The above data describes protons in a very silly box. Note that the first and last point in verticies should be the same. Typically gates are made using the plotter tool.

## ic_min

The minimum value for the ion chamber amplitude. Used to make a gate on beam species.

## ic_max

The maximum value for the ion chamber amplitude. Used to make a gate on beam species.

## interp_file_name

Name of the interpolation ODE mesh for the solver. If the file exists, it must have the same mesh grid as specified by the other `interp` parameters, otherwise an error will be thrown. If the file does not exist it will be generated using the other `interp` parameters. This name should reflect the type of analysis performed (i.e. `protons_in_300torrH2_500keV_halfDeg.npy`) and should have the `.npy` extension.

## interp_ke_min(MeV)

Minimum kinetic energy of the ODE mesh for the solver. This is the minimum energy of a particle for which this solver will work. In units of MeV.

## interp_ke_max(MeV)

Maximum kinetic energy of the ODE mesh for the solver. This is the maximum energy of a particle for which this solver will work. In units of MeV.

## interp_ke_bins

Number of kinetic energy bins of the ODE mesh for the solver. This is the coarseness/fineness of the mesh in kinetic energy.

## interp_polar_min(deg)

Minimum polar angle of the ODE mesh for the solver. This is the minimum polar angle of a particle for which this solver will work. In units of degrees.

## interp_polar_max(deg)

Maximum polar angle of the ODE mesh for the solver. This is the maximum polar angle of a particle for which this solver will work. In units of degrees.

## interp_polar_bins

Number of polar angle bins of the ODE mesh for the solver. This is the coarseness/fineness of the mesh in polar angle.
