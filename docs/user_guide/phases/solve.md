# Solving for Physics

The final phase of Spyral is solving for the physical observables using our estimates from the estimation phase. Solving is a two step process outlined below:

1. Generate an interpolation mesh of ODE solutions
2. Fit the solutions to the data trajectory

Below we'll break down each of these steps

## Generating the Interpolation Mesh

Before Spyral does anything, it will check to see if you're planning to run the solve phase. If you are, it will check to see if the interpolation mesh you requested (see the [configuration](../config/solver.md)); if the mesh does not exist, it will generate one.

In the [configuration](../config/solver.md) you can specify the coarseness of the mesh; the particle species is taken from the given particle ID gate. Spyral walks through the kinetic energy, polar angle range at the steps needed for the number of bins requested, making a complete trajectory for each energy, angle. Care is taken to handle the time range of the solving. First, Spyral will walk through the phase space of the configuration and find the time range of the longest trajectory (in time) using an inital coarse time window of 1 &mu;s. Then Spyral generates the acutal mesh using this refined time window. The number of time steps is a configuration parameter and can be used to optimize resolution. Each step saves 3 double precision floats (8 bytes) for 24 bytes per point. Multiply this by the total number of bins (energy bins x angle bins x timesteps) to get the total size of the mesh in bytes. Even trajectories which stop before the 1.0 &mu;s time window have the same total data size. This will give you an idea of what the size of your mesh will be.

The mesh size is important for several reasons. First is, obviously, a finer mesh (more bins) gives better results, as it reduces the degree of interpolation. But since a finer mesh is bigger, the computational strain is larger. The interpolation is a bilinear interpolation; it requires the *entire* mesh be loaded into memory. To alleviate this, Spyral shares the mesh memory between processes. You should always have more than 2x as much memory (read: RAM) available compared to the size of the mesh.

Finally, when a trajectory is requested, Spyral interpolates on energy and angle to get a trajectory points or fewer (points where the particle stopped are trimmed).

When a trajectory is requested, the full energy, polar angle, azimuthal angle, and reaction vertex must be given. The azimuthal angle and vertex are used to rotate and translate the trajectory to the proper orientation (this reduces the phase space of the interpolation mesh).

In general, it has been found that 0.5 degree steps in polar angle are best, while 200 keV steps in energy are acceptable. However, this may need tuning from experiment to experiment.

## Output

The output of the solver phase is a dataframe written to a parquet file in the workspace. Files are named by run and the particle species from the particle ID gate. The available values in the data frame are as follows:

- event: The event number associated with this data
- cluster_index: the cluster index associated with this data
- cluster_label: the cluster label associated with this data
- vertex_x: the fitted vertex x-coordinate
- sigma_vx: the error on the fitted vertex x-coordinate in meters
- vertex_y: the fitted vertex y-coordinate in meters
- sigma_vy: the error on the fitted vertex y-coordinate in meters
- vertex_z: the fitted vertex z-coordinate in meters
- sigma_vz: the error on the fitted vertex z-coordinate in meters
- brho: the fitted (total) B&rho; in Tm
- sigma_brho: the error on the fitted total B&rho; in Tm
- ke: the kinetic energy of the particle in MeV calculated from the fitted B&rho; and the known mass
- sigma_ke: the error on kinetic energy of the particle in MeV calculated from the fitted B&rho; and its uncertainty
- polar: the fitted polar angle of the trajectory in radians
- sigma_polar: the error on the fitted polar angle of the trajectory in radians
- azimuthal: the resulting azimuthal angle (not fitted) in radians. This calculated from the best fit vertex position.
- sigma_azimuthal: error on the resulting azimuthal angle (not fitted) in radians
- redchisq: the reduced &chi;<sup>2</sup> of the fit (error)

## Choice of Fitting Method

Spyral has two solving phases available: `InterpSolverPhase` and `InterpLeastSqSolverPhase`. `InterpSolverPhase` uses the L-BFGS-B optimization routine to determine the best trajectory from the *mean* error between the points on the ODE solution and the data. `InterpLeastSqSolverPhase` uses non-linear least-squares with the Levenberg-Marquardt algorithm to optimize based on the residuals. In both cases error bars based on the size of the pads in the pad plane and the width of a time bucket are included with the data (in L-BFGS-B these errors are used to form an error-weighted average). The key difference is that `InterpSolverPhase` cannot estimate uncertainties on the fit parameters while `InterpLeastSqSolverPhase` can. However, this is somewhat more nuanced in practice.

- Non-linear least-squares is very sensitive to outliers. Noisy data will often cause this method to fail dramatically. L-BFGS-B is less sensitive to outliers in general, and will typically still arrive at a sane value even in the presence of noisy data. *However*, because L-BFGS-B doesn't report parameter uncertainties, it is hard to quantify how *well* it performs on this data and testing has shown that *sane* does not always mean *good*.
- Non-linear least-squares makes many approximations about the underlying error distributions of the data. It is not clear how well these approximations apply to AT-TPC data.
- In general, when testing least-squares, it was observed to determine at the same conclusion as our L-BFGS-B solution *except* in the case of low-energy data near 90 degrees in the lab polar angle. This is data where we have always suspected we perform poorly, but if this region is critical for your analysis, it may be best to use L-BFGS-B, which is less rejective.

In general, the offical Spyral recommendation is to use `InterpLeastSqSolverPhase`; reporting errors is a critical part of understanding a fit to data, and provides a more robust understanding of the analysis performance. But given the wide range of datasets and configurations of the AT-TPC, we can not say that this is the correct method for *every* dataset.

## Control over Fit parameters

In general, 6 parameters completely describe a trajectory for a given particle species: particle kinetic energy (mass and momentum magnitude), polar angle, azimuthal angle, vertex x,y,z. Both solver types are set up to fit each of these parameters. However, in testing against simulated values it was found that there was little improvement relative to truth values when fitting vertex x, y, and the azimuthal angle. It is possible that for some data sets, including these parameters could be an overfit relative to the amount of information available. As a result, the solver configuration [parameters](../config/solver.md) exposes switches to turn these parameters on and off. It is up to the user to determine if these parameters are appropriate to use or not for a given dataset. Feedback is welcome on experiences using these paraemters, this is an active area of research for Spryal! Note that turning off parameters will generally also make the solving phase run faster.

## Final Thoughts

Generating a good interpolation mesh is key. If the mesh is too coarse the results will become equally coarse.

To make performance reasonable [Numba](../numba.md) is used to just-in-time compile the interpolation scheme and objective function. Checkout that section of the docs for more details.
