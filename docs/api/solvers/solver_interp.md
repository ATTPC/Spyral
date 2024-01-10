# solver_interp Module

This module defines an interpolation based ODE solving-fitting routine. The ODE for the AT-TPC system is solved ahead of time for many different initial values. These solutions are then interpolated and fit to the data to determine a best set of parameters for a given trajectory.

::: spyral.solvers.solver_interp