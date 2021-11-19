#if !defined(_LSS_ADVECTION_EQUATION_T_HPP_)
#define _LSS_ADVECTION_EQUATION_T_HPP_

#include "pde_solvers/one_dimensional/heat_type/lss_1d_general_svc_heat_equation.hpp"
#include <map>

#define PI 3.14159265359

// ///////////////////////////////////////////////////////////////////////////
//							ADVECTION PROBLEMS
// ///////////////////////////////////////////////////////////////////////////

// ===========================================================================
// ========================== IMPLICIT SOLVERS ===============================
// ===========================================================================

// ===========================================================================
// ====== Advection Diffusion problem with homogeneous boundary conditions ===
// ===========================================================================

template <typename T> void testImplAdvDiffEquationDirichletBCCUDASolverDeviceQREuler()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_grids::grid_config_hints_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::dev_fwd_cusolver_qr_euler_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
    using lss_utility::pi;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Advection Diffusion equation: \n\n";
    std::cout << " Using CUDA solver with QR (DEVICE) and implicit Euler method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = 1, x in <0,1> \n\n";
    std::cout << "============================================================\n";

    // typedef the general_heat_equation
    typedef general_svc_heat_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(1.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(0.1));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [](T x) { return 1.0; };
    auto b = [](T x) { return -1.0; };
    auto other = [](T x) { return 0.0; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, b, other);
    // initial condition:
    auto initial_condition = [](T x) { return 1.0; };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(initial_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // grid config:
    auto const &grid_config_hints_ptr = std::make_shared<grid_config_hints_1d<T>>();
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         dev_fwd_cusolver_qr_euler_solver_config_ptr);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        T const first = 2.0 / pi<T>();
        T const exp_0p5x = std::exp(0.5 * x);
        T const exp_m0p5 = std::exp(-0.5);
        T np_sqr{};
        T sum{};
        T num{}, den{}, var{};
        T lambda{};
        for (std::size_t i = 1; i <= n; ++i)
        {
            np_sqr = (i * i * pi<T>() * pi<T>());
            lambda = 0.25 + np_sqr;
            num = (1.0 - std::pow(-1.0, i) * exp_m0p5) * exp_0p5x * std::exp(-1.0 * lambda * t) *
                  std::sin(i * pi<T>() * x);
            den = i * (1.0 + (0.25 / np_sqr));
            var = num / den;
            sum += var;
        }
        return (first * sum);
    };

    T const h = discretization_ptr->space_step();
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        benchmark = exact(j * h, time_range.upper(), 30);
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

template <typename T> void testImplAdvDiffEquationDirichletBCCUDASolverDeviceQRCrankNicolson()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_grids::grid_config_hints_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::dev_fwd_cusolver_qr_cn_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
    using lss_utility::pi;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Advection Diffusion equation: \n\n";
    std::cout << " Using CUDA solver with QR (DEVICE) and implicit CN method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = 1, x in <0,1> \n\n";
    std::cout << "============================================================\n";

    // typedef the general_heat_equation
    typedef general_svc_heat_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(1.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(0.1));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [](T x) { return 1.0; };
    auto b = [](T x) { return -1.0; };
    auto other = [](T x) { return 0.0; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, b, other);
    // initial condition:
    auto initial_condition = [](T x) { return 1.0; };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(initial_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // grid config:
    auto const &grid_config_hints_ptr = std::make_shared<grid_config_hints_1d<T>>();
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         dev_fwd_cusolver_qr_cn_solver_config_ptr);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        T const first = 2.0 / pi<T>();
        T const exp_0p5x = std::exp(0.5 * x);
        T const exp_m0p5 = std::exp(-0.5);
        T np_sqr{};
        T sum{};
        T num{}, den{}, var{};
        T lambda{};
        for (std::size_t i = 1; i <= n; ++i)
        {
            np_sqr = (i * i * pi<T>() * pi<T>());
            lambda = 0.25 + np_sqr;
            num = (1.0 - std::pow(-1.0, i) * exp_m0p5) * exp_0p5x * std::exp(-1.0 * lambda * t) *
                  std::sin(i * pi<T>() * x);
            den = i * (1.0 + (0.25 / np_sqr));
            var = num / den;
            sum += var;
        }
        return (first * sum);
    };

    T const h = discretization_ptr->space_step();
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        benchmark = exact(j * h, time_range.upper(), 30);
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

void testImplAdvDiffEquationDirichletBCCUDASolverDeviceQR()
{
    std::cout << "============================================================\n";
    std::cout << " Implicit Advection (CUDA QR DEVICE) Equation (Dirichlet BC)\n";
    std::cout << "============================================================\n";

    testImplAdvDiffEquationDirichletBCCUDASolverDeviceQREuler<double>();
    testImplAdvDiffEquationDirichletBCCUDASolverDeviceQREuler<float>();
    testImplAdvDiffEquationDirichletBCCUDASolverDeviceQRCrankNicolson<double>();
    testImplAdvDiffEquationDirichletBCCUDASolverDeviceQRCrankNicolson<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplAdvDiffEquationDirichletBCSORSolverDeviceEuler()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_grids::grid_config_hints_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::dev_fwd_sorsolver_euler_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
    using lss_utility::pi;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Advection Diffusion equation: \n\n";
    std::cout << " Using CUDA SOR solver (DEVICE) with implicit Euler method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = 1, x in <0,1> \n\n";
    std::cout << "============================================================\n";

    // typedef the general_heat_equation
    typedef general_svc_heat_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(1.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(0.1));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [](T x) { return 1.0; };
    auto b = [](T x) { return -1.0; };
    auto other = [](T x) { return 0.0; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, b, other);
    // initial condition:
    auto initial_condition = [](T x) { return 1.0; };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(initial_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // grid config:
    auto const &grid_config_hints_ptr = std::make_shared<grid_config_hints_1d<T>>();
    // details:
    std::map<std::string, T> details;
    details["sor_omega"] = static_cast<T>(1.0);
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         dev_fwd_sorsolver_euler_solver_config_ptr, details);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        T const first = 2.0 / pi<T>();
        T const exp_0p5x = std::exp(0.5 * x);
        T const exp_m0p5 = std::exp(-0.5);
        T np_sqr{};
        T sum{};
        T num{}, den{}, var{};
        T lambda{};
        for (std::size_t i = 1; i <= n; ++i)
        {
            np_sqr = (i * i * pi<T>() * pi<T>());
            lambda = 0.25 + np_sqr;
            num = (1.0 - std::pow(-1.0, i) * exp_m0p5) * exp_0p5x * std::exp(-1.0 * lambda * t) *
                  std::sin(i * pi<T>() * x);
            den = i * (1.0 + (0.25 / np_sqr));
            var = num / den;
            sum += var;
        }
        return (first * sum);
    };

    T const h = discretization_ptr->space_step();
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        benchmark = exact(j * h, time_range.upper(), 20);
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

template <typename T> void testImplAdvDiffEquationDirichletBCSORSolverDeviceCrankNicolson()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_grids::grid_config_hints_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::dev_fwd_sorsolver_cn_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
    using lss_utility::pi;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Advection Diffusion equation: \n\n";
    std::cout << " Using CUDA SOR solver (DEVICE) with implicit CN method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = 1, x in <0,1> \n\n";
    std::cout << "============================================================\n";

    // typedef the general_heat_equation
    typedef general_svc_heat_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(1.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(0.1));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [](T x) { return 1.0; };
    auto b = [](T x) { return -1.0; };
    auto other = [](T x) { return 0.0; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, b, other);
    // initial condition:
    auto initial_condition = [](T x) { return 1.0; };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(initial_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // grid config:
    auto const &grid_config_hints_ptr = std::make_shared<grid_config_hints_1d<T>>();
    // details:
    std::map<std::string, T> details;
    details["sor_omega"] = static_cast<T>(1.0);
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         dev_fwd_sorsolver_cn_solver_config_ptr, details);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        T const first = 2.0 / pi<T>();
        T const exp_0p5x = std::exp(0.5 * x);
        T const exp_m0p5 = std::exp(-0.5);
        T np_sqr{};
        T sum{};
        T num{}, den{}, var{};
        T lambda{};
        for (std::size_t i = 1; i <= n; ++i)
        {
            np_sqr = (i * i * pi<T>() * pi<T>());
            lambda = 0.25 + np_sqr;
            num = (1.0 - std::pow(-1.0, i) * exp_m0p5) * exp_0p5x * std::exp(-1.0 * lambda * t) *
                  std::sin(i * pi<T>() * x);
            den = i * (1.0 + (0.25 / np_sqr));
            var = num / den;
            sum += var;
        }
        return (first * sum);
    };

    T const h = discretization_ptr->space_step();
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        benchmark = exact(j * h, time_range.upper(), 20);
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

void testImplAdvDiffEquationDirichletBCSORSolverDevice()
{
    std::cout << "============================================================\n";
    std::cout << " Implicit Advection  (SOR QR DEVICE) Equation (Dirichlet BC)\n";
    std::cout << "============================================================\n";

    testImplAdvDiffEquationDirichletBCSORSolverDeviceEuler<double>();
    testImplAdvDiffEquationDirichletBCSORSolverDeviceEuler<float>();
    testImplAdvDiffEquationDirichletBCSORSolverDeviceCrankNicolson<double>();
    testImplAdvDiffEquationDirichletBCSORSolverDeviceCrankNicolson<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplAdvDiffEquationDirichletBCSORSolverHostEuler()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_grids::grid_config_hints_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::host_fwd_sorsolver_euler_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
    using lss_utility::pi;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Advection Diffusion equation: \n\n";
    std::cout << " Using SOR solver with implicit Euler method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = 1, x in <0,1> \n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_svc_heat_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(1.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(0.1));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [](T x) { return 1.0; };
    auto b = [](T x) { return -1.0; };
    auto other = [](T x) { return 0.0; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, b, other);
    // initial condition:
    auto initial_condition = [](T x) { return 1.0; };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(initial_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // grid config:
    auto const &grid_config_hints_ptr = std::make_shared<grid_config_hints_1d<T>>();
    // details:
    std::map<std::string, T> details;
    details["sor_omega"] = static_cast<T>(1.0);
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         host_fwd_sorsolver_euler_solver_config_ptr, details);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        T const first = 2.0 / pi<T>();
        T const exp_0p5x = std::exp(0.5 * x);
        T const exp_m0p5 = std::exp(-0.5);
        T np_sqr{};
        T sum{};
        T num{}, den{}, var{};
        T lambda{};
        for (std::size_t i = 1; i <= n; ++i)
        {
            np_sqr = (i * i * pi<T>() * pi<T>());
            lambda = 0.25 + np_sqr;
            num = (1.0 - std::pow(-1.0, i) * exp_m0p5) * exp_0p5x * std::exp(-1.0 * lambda * t) *
                  std::sin(i * pi<T>() * x);
            den = i * (1.0 + (0.25 / np_sqr));
            var = num / den;
            sum += var;
        }
        return (first * sum);
    };

    T const h = discretization_ptr->space_step();
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        benchmark = exact(j * h, time_range.upper(), 20);
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

template <typename T> void testImplAdvDiffEquationDirichletBCSORSolverHostCrankNicolson()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_grids::grid_config_hints_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::host_fwd_sorsolver_cn_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
    using lss_utility::pi;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Advection Diffusion equation: \n\n";
    std::cout << " Using SOR solver with implicit CN method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = 1, x in <0,1> \n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_svc_heat_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(1.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(0.1));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [](T x) { return 1.0; };
    auto b = [](T x) { return -1.0; };
    auto other = [](T x) { return 0.0; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, b, other);
    // initial condition:
    auto initial_condition = [](T x) { return 1.0; };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(initial_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // grid config:
    auto const &grid_config_hints_ptr = std::make_shared<grid_config_hints_1d<T>>();
    // details:
    std::map<std::string, T> details;
    details["sor_omega"] = static_cast<T>(1.0);
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         host_fwd_sorsolver_cn_solver_config_ptr, details);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        T const first = 2.0 / pi<T>();
        T const exp_0p5x = std::exp(0.5 * x);
        T const exp_m0p5 = std::exp(-0.5);
        T np_sqr{};
        T sum{};
        T num{}, den{}, var{};
        T lambda{};
        for (std::size_t i = 1; i <= n; ++i)
        {
            np_sqr = (i * i * pi<T>() * pi<T>());
            lambda = 0.25 + np_sqr;
            num = (1.0 - std::pow(-1.0, i) * exp_m0p5) * exp_0p5x * std::exp(-1.0 * lambda * t) *
                  std::sin(i * pi<T>() * x);
            den = i * (1.0 + (0.25 / np_sqr));
            var = num / den;
            sum += var;
        }
        return (first * sum);
    };

    T const h = discretization_ptr->space_step();
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        benchmark = exact(j * h, time_range.upper(), 20);
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

void testImplAdvDiffEquationDirichletBCSORSolverHost()
{
    std::cout << "============================================================\n";
    std::cout << "= Implicit Advection (SOR QR HOST) Equation (Dirichlet BC) =\n";
    std::cout << "============================================================\n";

    testImplAdvDiffEquationDirichletBCSORSolverHostEuler<double>();
    testImplAdvDiffEquationDirichletBCSORSolverHostEuler<float>();
    testImplAdvDiffEquationDirichletBCSORSolverHostCrankNicolson<double>();
    testImplAdvDiffEquationDirichletBCSORSolverHostCrankNicolson<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplAdvDiffEquationDirichletBCCUDASolverHostQREuler()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_grids::grid_config_hints_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::host_fwd_cusolver_qr_euler_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
    using lss_utility::pi;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Advection Diffusion equation: \n\n";
    std::cout << " Using CUDA Solver on HOST with QR and implicit Euler method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = 1, x in <0,1> \n\n";
    std::cout << "============================================================\n";

    typedef general_svc_heat_equation<T, std::vector, std::allocator<T>> pde_solver;
    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(1.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(0.1));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [](T x) { return 1.0; };
    auto b = [](T x) { return -1.0; };
    auto other = [](T x) { return 0.0; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, b, other);
    // initial condition:
    auto initial_condition = [](T x) { return 1.0; };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(initial_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // grid config:
    auto const &grid_config_hints_ptr = std::make_shared<grid_config_hints_1d<T>>();
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         host_fwd_cusolver_qr_euler_solver_config_ptr);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        T const first = 2.0 / pi<T>();
        T const exp_0p5x = std::exp(0.5 * x);
        T const exp_m0p5 = std::exp(-0.5);
        T np_sqr{};
        T sum{};
        T num{}, den{}, var{};
        T lambda{};
        for (std::size_t i = 1; i <= n; ++i)
        {
            np_sqr = (i * i * pi<T>() * pi<T>());
            lambda = 0.25 + np_sqr;
            num = (1.0 - std::pow(-1.0, i) * exp_m0p5) * exp_0p5x * std::exp(-1.0 * lambda * t) *
                  std::sin(i * pi<T>() * x);
            den = i * (1.0 + (0.25 / np_sqr));
            var = num / den;
            sum += var;
        }
        return (first * sum);
    };

    T const h = discretization_ptr->space_step();
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        benchmark = exact(j * h, time_range.upper(), 20);
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

template <typename T> void testImplAdvDiffEquationDirichletBCCUDASolverHostQRCrankNicolson()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_grids::grid_config_hints_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::host_fwd_cusolver_qr_cn_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
    using lss_utility::pi;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Advection Diffusion equation: \n\n";
    std::cout << " Using CUDA Solver on HOST with QR and implicit CN method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = 1, x in <0,1> \n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_svc_heat_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(1.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(0.1));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [](T x) { return 1.0; };
    auto b = [](T x) { return -1.0; };
    auto other = [](T x) { return 0.0; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, b, other);
    // initial condition:
    auto initial_condition = [](T x) { return 1.0; };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(initial_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // grid config:
    auto const &grid_config_hints_ptr = std::make_shared<grid_config_hints_1d<T>>();
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         host_fwd_cusolver_qr_cn_solver_config_ptr);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        T const first = 2.0 / pi<T>();
        T const exp_0p5x = std::exp(0.5 * x);
        T const exp_m0p5 = std::exp(-0.5);
        T np_sqr{};
        T sum{};
        T num{}, den{}, var{};
        T lambda{};
        for (std::size_t i = 1; i <= n; ++i)
        {
            np_sqr = (i * i * pi<T>() * pi<T>());
            lambda = 0.25 + np_sqr;
            num = (1.0 - std::pow(-1.0, i) * exp_m0p5) * exp_0p5x * std::exp(-1.0 * lambda * t) *
                  std::sin(i * pi<T>() * x);
            den = i * (1.0 + (0.25 / np_sqr));
            var = num / den;
            sum += var;
        }
        return (first * sum);
    };

    T const h = discretization_ptr->space_step();
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        benchmark = exact(j * h, time_range.upper(), 20);
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

void testImplAdvDiffEquationDirichletBCCUDASolverHostQR()
{
    std::cout << "============================================================\n";
    std::cout << "= Implicit Advection(CUDA QR HOST) Equation (Dirichlet BC) =\n";
    std::cout << "============================================================\n";

    testImplAdvDiffEquationDirichletBCCUDASolverHostQREuler<double>();
    testImplAdvDiffEquationDirichletBCCUDASolverHostQREuler<float>();
    testImplAdvDiffEquationDirichletBCCUDASolverHostQRCrankNicolson<double>();
    testImplAdvDiffEquationDirichletBCCUDASolverHostQRCrankNicolson<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplAdvDiffEquationDirichletBCDoubleSweepSolverEuler()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_grids::grid_config_hints_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::host_fwd_dssolver_euler_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
    using lss_utility::pi;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Advection Diffusion equation: \n\n";
    std::cout << " Using Double Sweep on HOST with implicit Euler method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = 1, x in <0,1> \n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_svc_heat_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(1.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(0.1));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [](T x) { return 1.0; };
    auto b = [](T x) { return -1.0; };
    auto other = [](T x) { return 0.0; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, b, other);
    // initial condition:
    auto initial_condition = [](T x) { return 1.0; };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(initial_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // grid config:
    auto const &grid_config_hints_ptr = std::make_shared<grid_config_hints_1d<T>>();
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         host_fwd_dssolver_euler_solver_config_ptr);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        T const first = 2.0 / pi<T>();
        T const exp_0p5x = std::exp(0.5 * x);
        T const exp_m0p5 = std::exp(-0.5);
        T np_sqr{};
        T sum{};
        T num{}, den{}, var{};
        T lambda{};
        for (std::size_t i = 1; i <= n; ++i)
        {
            np_sqr = (i * i * pi<T>() * pi<T>());
            lambda = 0.25 + np_sqr;
            num = (1.0 - std::pow(-1.0, i) * exp_m0p5) * exp_0p5x * std::exp(-1.0 * lambda * t) *
                  std::sin(i * pi<T>() * x);
            den = i * (1.0 + (0.25 / np_sqr));
            var = num / den;
            sum += var;
        }
        return (first * sum);
    };

    T const h = discretization_ptr->space_step();
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        benchmark = exact(j * h, time_range.upper(), 20);
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

template <typename T> void testImplAdvDiffEquationDirichletBCDoubleSweepSolverCrankNicolson()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_grids::grid_config_hints_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::host_fwd_dssolver_cn_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
    using lss_utility::pi;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Advection Diffusion equation: \n\n";
    std::cout << " Using Double Sweep on HOST with implicit CN method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = 1, x in <0,1> \n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_svc_heat_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(1.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(0.1));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [](T x) { return 1.0; };
    auto b = [](T x) { return -1.0; };
    auto other = [](T x) { return 0.0; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, b, other);
    // initial condition:
    auto initial_condition = [](T x) { return 1.0; };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(initial_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // grid config:
    auto const &grid_config_hints_ptr = std::make_shared<grid_config_hints_1d<T>>();
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         host_fwd_dssolver_cn_solver_config_ptr);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        T const first = 2.0 / pi<T>();
        T const exp_0p5x = std::exp(0.5 * x);
        T const exp_m0p5 = std::exp(-0.5);
        T np_sqr{};
        T sum{};
        T num{}, den{}, var{};
        T lambda{};
        for (std::size_t i = 1; i <= n; ++i)
        {
            np_sqr = (i * i * pi<T>() * pi<T>());
            lambda = 0.25 + np_sqr;
            num = (1.0 - std::pow(-1.0, i) * exp_m0p5) * exp_0p5x * std::exp(-1.0 * lambda * t) *
                  std::sin(i * pi<T>() * x);
            den = i * (1.0 + (0.25 / np_sqr));
            var = num / den;
            sum += var;
        }
        return (first * sum);
    };

    T const h = discretization_ptr->space_step();
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        benchmark = exact(j * h, time_range.upper(), 20);
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

void testImplAdvDiffEquationDirichletBCDoubleSweepSolver()
{
    std::cout << "============================================================\n";
    std::cout << "= Implicit Advection(Double Sweep) Equation (Dirichlet BC) =\n";
    std::cout << "============================================================\n";

    testImplAdvDiffEquationDirichletBCDoubleSweepSolverEuler<double>();
    testImplAdvDiffEquationDirichletBCDoubleSweepSolverEuler<float>();
    testImplAdvDiffEquationDirichletBCDoubleSweepSolverCrankNicolson<double>();
    testImplAdvDiffEquationDirichletBCDoubleSweepSolverCrankNicolson<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplAdvDiffEquationDirichletBCThomasLUSolverEuler()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_grids::grid_config_hints_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::host_fwd_tlusolver_euler_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
    using lss_utility::pi;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Advection Diffusion equation: \n\n";
    std::cout << " Using Thomas LU algorithm with implicit Euler method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = 1, x in <0,1> \n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_svc_heat_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(1.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(0.1));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [](T x) { return 1.0; };
    auto b = [](T x) { return -1.0; };
    auto other = [](T x) { return 0.0; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, b, other);
    // initial condition:
    auto initial_condition = [](T x) { return 1.0; };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(initial_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // grid config:
    auto const &grid_config_hints_ptr = std::make_shared<grid_config_hints_1d<T>>();
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         host_fwd_tlusolver_euler_solver_config_ptr);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        T const first = 2.0 / pi<T>();
        T const exp_0p5x = std::exp(0.5 * x);
        T const exp_m0p5 = std::exp(-0.5);
        T np_sqr{};
        T sum{};
        T num{}, den{}, var{};
        T lambda{};
        for (std::size_t i = 1; i <= n; ++i)
        {
            np_sqr = (i * i * pi<T>() * pi<T>());
            lambda = 0.25 + np_sqr;
            num = (1.0 - std::pow(-1.0, i) * exp_m0p5) * exp_0p5x * std::exp(-1.0 * lambda * t) *
                  std::sin(i * pi<T>() * x);
            den = i * (1.0 + (0.25 / np_sqr));
            var = num / den;
            sum += var;
        }
        return (first * sum);
    };
    T const h = discretization_ptr->space_step();
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        benchmark = exact(j * h, time_range.upper(), 20);
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

template <typename T> void testImplAdvDiffEquationDirichletBCThomasLUSolverCrankNicolson()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_grids::grid_config_hints_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::host_fwd_tlusolver_cn_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
    using lss_utility::pi;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Advection Diffusion equation: \n\n";
    std::cout << " Using Thomas LU algorithm with implicit CN method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = 1, x in <0,1> \n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_svc_heat_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(1.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(0.1));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [](T x) { return 1.0; };
    auto b = [](T x) { return -1.0; };
    auto other = [](T x) { return 0.0; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, b, other);
    // initial condition:
    auto initial_condition = [](T x) { return 1.0; };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(initial_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // grid config:
    auto const &grid_config_hints_ptr = std::make_shared<grid_config_hints_1d<T>>();
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         host_fwd_tlusolver_cn_solver_config_ptr);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        T const first = 2.0 / pi<T>();
        T const exp_0p5x = std::exp(0.5 * x);
        T const exp_m0p5 = std::exp(-0.5);
        T np_sqr{};
        T sum{};
        T num{}, den{}, var{};
        T lambda{};
        for (std::size_t i = 1; i <= n; ++i)
        {
            np_sqr = (i * i * pi<T>() * pi<T>());
            lambda = 0.25 + np_sqr;
            num = (1.0 - std::pow(-1.0, i) * exp_m0p5) * exp_0p5x * std::exp(-1.0 * lambda * t) *
                  std::sin(i * pi<T>() * x);
            den = i * (1.0 + (0.25 / np_sqr));
            var = num / den;
            sum += var;
        }
        return (first * sum);
    };

    T const h = discretization_ptr->space_step();
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        benchmark = exact(j * h, time_range.upper(), 20);
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

void testImplAdvDiffEquationDirichletBCThomasLUSolver()
{
    std::cout << "============================================================\n";
    std::cout << "== Implicit Advection (Thomas LU) Equation (Dirichlet BC) ==\n";
    std::cout << "============================================================\n";

    testImplAdvDiffEquationDirichletBCThomasLUSolverEuler<double>();
    testImplAdvDiffEquationDirichletBCThomasLUSolverEuler<float>();
    testImplAdvDiffEquationDirichletBCThomasLUSolverCrankNicolson<double>();
    testImplAdvDiffEquationDirichletBCThomasLUSolverCrankNicolson<float>();

    std::cout << "============================================================\n";
}

#endif //_LSS_ADVECTION_EQUATION_T_HPP_
