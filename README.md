# Linear System Solvers
Linear System Solvers library. Written in C++17. It contains a few wrappers around CUDA cuSolver library functions plus some other well known solvers.
It also contains some PDE solvers.

## ODE Solver
* general 2nd degree equation with variable coefficients (supports all Dirichlet, Neumann, Robin boundary conditions)

## PDE Solver
* 1D general heat equation with variable coefficients in space dimension (supports all Dirichlet, Neumann, Robin boundary conditions)
* 1D general wave equation with variable coefficients in space dimension (supports all Dirichlet, Neumann, Robin boundary conditions) - IN PROGRESS (Testing stage)

## Some curves from ODE solver
Simple Two-point BVP (u''(x) = -2 with Dirichlet and Robin BC)
![Simple ODE equation](/outputs/simple_ode_numerical.png)

## Some surfaces from PDE solver

Heat equation (Dirichlet BC)
![Pure heat equation](/outputs/temp_heat_equ_numerical.png)

Wave equation (Dirichlet BC)
![Pure heat equation](/outputs/wave_pure_dir_equ_numerical.png)

Damped wave equation (Dirichlet BC)
![Pure heat equation](/outputs/damped_wave_dir_equ_numerical.png)

Heat equation (Dirichlet and Neumann BC)
![Pure heat equation](/outputs/temp_heat_neu_equ_numerical.png)

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