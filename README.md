# Linear System Solvers
Linear System Solvers library. Written in C++17. It contains a few wrappers around CUDA cuSolver library functions plus some other well known solvers.
It also contains some PDE solvers.

## PDE Solver
* 1D general Heat equation with variable coefficients in space dimension (supports all Dirichlet, Neumann, Robin boundary conditions)

## Some surfaces from PDE solver

Heat equation
![Pure heat equation](/outputs/temp_heat_equ_numerical.png)

Advection equation
![Advection equation](/outputs/temp_advection_equ_numerical.png)

Black-Scholes equation
![Black-Scholes equation](/outputs/call_option_price_surface_numerical.png)

## Usage
Just started.
To see how to use this library see header files ending with _t.
I will soon describe in detail how to use this library.

## Note
It is not yet a library. It will be made into one, once I start working on the interface 