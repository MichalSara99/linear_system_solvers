# Linear System Solvers
Linear System Solvers library. Written in C++17. It contains a few wrappers around CUDA cuSolver library functions plus some other well known solvers.
It also contains some PDE solvers.

## ODE Solver
* general 2nd degree equation with variable coefficients (supports all Dirichlet, Neumann, Robin boundary conditions)
* All solvers support uniform and non-uniform grid with variable scaling


## PDE Solver
* 1D general heat equation with variable coefficients in space dimension (supports all Dirichlet, Neumann, Robin boundary conditions)
* 1D general wave equation with variable coefficients in space dimension (supports all Dirichlet, Neumann, Robin boundary conditions)
* 2D general Heston type model with variable coefficients in space dimensions
 (support for Douglas-Rachford ADI (weighted scheme also available), Craig-Sneyd ADI, Modified Craig-Sneyd ADI, Hundsdorfer-Verwer ADI)
* All solvers support uniform and non-uniform grid with variable scaling


## Some curves from ODE solver
Simple Two-point BVP (u''(x) = -2 with Dirichlet and Robin BC)
![Simple ODE equation](/outputs/simple_ode_numerical.png)

## Some surfaces from PDE solver

Heat equation (Dirichlet BC) from implicit solver
![Pure heat equation](/outputs/temp_heat_equ_numerical_nonuniform.png)

Wave equation (Dirichlet BC) from implicit solver
![Pure wave equation](/outputs/wave_pure_dir_equ_numerical.png)

Wave equation (Neumann BC) from implicit solver
![Pure wave equation - neumann](/outputs/wave_neu_equ_numerical.png)

Damped wave equation (Dirichlet BC) from implicit solver
![Damped wave equation](/outputs/damped_wave_dir_equ_numerical.png)

Heat equation (Dirichlet and Neumann BC) from explicit solver
![Pure heat equation](/outputs/temp_heat_neu_equ_numerical.png)

Advection equation from implicit solver
![Advection equation](/outputs/temp_advection_equ_numerical.png)

Black-Scholes equation from implicit solver
![Black-Scholes equation](/outputs/call_option_price_surface_numerical_nonuniform.png)

Heston equation DR from implicit solver
![Heston equation DR](/outputs/impl_heston_dr_numerical.png)

Heston equation CS from implicit solver
![Heston equation CS](/outputs/impl_heston_cs_numerical.png)

Heston equation MCS from implicit solver
![Heston equation MCS](/outputs/impl_heston_mcs_numerical.png)

Heston equation HV from implicit solver
![Heston equation HV](/outputs/impl_heston_hv_numerical.png)

SABR equation from implicit solver
![SABR equation](/outputs/impl_sabr_dr_numerical.png)

Heston equation from explicit solver
![Heston equation expl](/outputs/expl_heston_euler_host_numerical.png)

SABR equation from explicit solver
![SABR equation expl](/outputs/expl_sabr_euler_host_numerical.png)

## Usage
Just started.
To see how to use this library see header files ending with _t.
I will soon describe in detail how to use this library.

## Note
It is not yet a library. It will be made into one, once I start working on the interface 