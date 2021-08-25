#if !defined(_LSS_PURE_WAVE_EQUATION_T_HPP_)
#define _LSS_PURE_WAVE_EQUATION_T_HPP_

#include "pde_solvers/one_dimensional/wave_type/lss_1d_general_svc_wave_equation.hpp"
#include <map>

// ///////////////////////////////////////////////////////////////////////////
//							PURE WAVE PROBLEMS
// ///////////////////////////////////////////////////////////////////////////

// ===========================================================================
// ========================== IMPLICIT SOLVERS ===============================
// ===========================================================================

// ===========================================================================
// =========== Wave problem with homogeneous boundary conditions =============
// ===========================================================================

// Dirichlet boundaries:

template <typename T> void testImplPureWaveEquationDirichletBCCUDASolverDeviceQRDetail()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::wave_coefficient_data_config_1d;
    using lss_pde_solvers::wave_data_config_1d;
    using lss_pde_solvers::wave_implicit_solver_config;
    using lss_pde_solvers::wave_initial_data_config_1d;
    using lss_pde_solvers::default_wave_solver_configs::dev_fwd_cusolver_qr_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_wave_equation;
    using lss_utility::pi;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Wave equation: \n\n";
    std::cout << " Using CUDA solver with QR (DEVICE) method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_tt(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = sin(pi*x), x in <0,1> \n\n";
    std::cout << " U_x(x,0) = 0, x in <0,1> \n\n";
    std::cout << "============================================================\n";

    // typedef the general_svc_wave_equation
    typedef general_svc_wave_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(1.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(0.8));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto b = [](T x) { return 1.0; };
    auto other = [](T x) { return 0.0; };
    auto const wave_coeffs_data_ptr = std::make_shared<wave_coefficient_data_config_1d<T>>(other, b, other, other);
    // initial condition:
    auto initial_condition = [](T x) { return std::sin(pi<T>() * x); };
    auto const wave_init_data_ptr = std::make_shared<wave_initial_data_config_1d<T>>(initial_condition, other);
    // wave data config:
    auto const wave_data_ptr = std::make_shared<wave_data_config_1d<T>>(wave_coeffs_data_ptr, wave_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // initialize pde solver
    pde_solver pdesolver(wave_data_ptr, discretization_ptr, boundary_pair, dev_fwd_cusolver_qr_solver_config_ptr);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        const T var1 = std::sin(pi<T>() * x);
        const T var2 = std::cos(pi<T>() * t);
        return (var1 * var2);
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

void testImplPureWaveEquationDirichletBCCUDASolverDeviceQR()
{
    std::cout << "============================================================\n";
    std::cout << " Implicit Pure Wave (CUDA QR DEVICE) Equation (Dirichlet BC)\n";
    std::cout << "============================================================\n";

    testImplPureWaveEquationDirichletBCCUDASolverDeviceQRDetail<double>();
    testImplPureWaveEquationDirichletBCCUDASolverDeviceQRDetail<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplPureWaveEquationDirichletBCCUDASolverDeviceSORDetail()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::wave_coefficient_data_config_1d;
    using lss_pde_solvers::wave_data_config_1d;
    using lss_pde_solvers::wave_implicit_solver_config;
    using lss_pde_solvers::wave_initial_data_config_1d;
    using lss_pde_solvers::default_wave_solver_configs::dev_fwd_sorsolver_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_wave_equation;
    using lss_utility::pi;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Wave equation: \n\n";
    std::cout << " Using CUDA solver with LU (DEVICE) method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_tt(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = sin(pi*x), x in <0,1> \n\n";
    std::cout << " U_x(x,0) = 0, x in <0,1> \n\n";
    std::cout << "============================================================\n";

    // typedef the general_svc_wave_equation
    typedef general_svc_wave_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(1.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(0.7));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto b = [](T x) { return 1.0; };
    auto other = [](T x) { return 0.0; };
    auto const wave_coeffs_data_ptr = std::make_shared<wave_coefficient_data_config_1d<T>>(other, b, other, other);
    // initial condition:
    auto initial_condition = [](T x) { return std::sin(pi<T>() * x); };
    auto const wave_init_data_ptr = std::make_shared<wave_initial_data_config_1d<T>>(initial_condition, other);
    // wave data config:
    auto const wave_data_ptr = std::make_shared<wave_data_config_1d<T>>(wave_coeffs_data_ptr, wave_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // details:
    std::map<std::string, T> details;
    details["sor_omega"] = static_cast<T>(1.0);
    // initialize pde solver
    pde_solver pdesolver(wave_data_ptr, discretization_ptr, boundary_pair, dev_fwd_sorsolver_solver_config_ptr,
                         details);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        const T var1 = std::sin(pi<T>() * x);
        const T var2 = std::cos(pi<T>() * t);
        return (var1 * var2);
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

void testImplPureWaveEquationDirichletBCCUDASolverDeviceSOR()
{
    std::cout << "============================================================\n";
    std::cout << "Implicit Pure Wave (CUDA SOR DEVICE) Equation (Dirichlet BC)\n";
    std::cout << "============================================================\n";

    testImplPureWaveEquationDirichletBCCUDASolverDeviceSORDetail<double>();
    testImplPureWaveEquationDirichletBCCUDASolverDeviceSORDetail<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplPureWaveEquationDirichletBCCUDASolverHostSORDetail()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::wave_coefficient_data_config_1d;
    using lss_pde_solvers::wave_data_config_1d;
    using lss_pde_solvers::wave_implicit_solver_config;
    using lss_pde_solvers::wave_initial_data_config_1d;
    using lss_pde_solvers::default_wave_solver_configs::host_fwd_sorsolver_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_wave_equation;
    using lss_utility::pi;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Wave equation: \n\n";
    std::cout << " Using CUDA solver with LU (DEVICE) method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_tt(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = sin(pi*x), x in <0,1> \n\n";
    std::cout << " U_x(x,0) = 0, x in <0,1> \n\n";
    std::cout << "============================================================\n";

    // typedef the general_svc_wave_equation
    typedef general_svc_wave_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(1.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(0.7));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto b = [](T x) { return 1.0; };
    auto other = [](T x) { return 0.0; };
    auto const wave_coeffs_data_ptr = std::make_shared<wave_coefficient_data_config_1d<T>>(other, b, other, other);
    // initial condition:
    auto initial_condition = [](T x) { return std::sin(pi<T>() * x); };
    auto const wave_init_data_ptr = std::make_shared<wave_initial_data_config_1d<T>>(initial_condition, other);
    // wave data config:
    auto const wave_data_ptr = std::make_shared<wave_data_config_1d<T>>(wave_coeffs_data_ptr, wave_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // details:
    std::map<std::string, T> details;
    details["sor_omega"] = static_cast<T>(1.0);
    // initialize pde solver
    pde_solver pdesolver(wave_data_ptr, discretization_ptr, boundary_pair, host_fwd_sorsolver_solver_config_ptr,
                         details);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        const T var1 = std::sin(pi<T>() * x);
        const T var2 = std::cos(pi<T>() * t);
        return (var1 * var2);
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

void testImplPureWaveEquationDirichletBCCUDASolverHostSOR()
{
    std::cout << "============================================================\n";
    std::cout << "Implicit Pure Wave (CUDA SOR Host) Equation (Dirichlet BC)\n";
    std::cout << "============================================================\n";

    testImplPureWaveEquationDirichletBCCUDASolverHostSORDetail<double>();
    testImplPureWaveEquationDirichletBCCUDASolverHostSORDetail<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplPureWaveEquationDirichletBCSolverHostDoubleSweepDetail()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::wave_coefficient_data_config_1d;
    using lss_pde_solvers::wave_data_config_1d;
    using lss_pde_solvers::wave_implicit_solver_config;
    using lss_pde_solvers::wave_initial_data_config_1d;
    using lss_pde_solvers::default_wave_solver_configs::host_fwd_dssolver_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_wave_equation;
    using lss_utility::pi;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Wave equation: \n\n";
    std::cout << " Using Double Sweep Solver method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_tt(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = sin(pi*x), x in <0,1> \n\n";
    std::cout << " U_x(x,0) = 0, x in <0,1> \n\n";
    std::cout << "============================================================\n";

    // typedef the general_svc_wave_equation
    typedef general_svc_wave_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(1.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(0.7));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto b = [](T x) { return 1.0; };
    auto other = [](T x) { return 0.0; };
    auto const wave_coeffs_data_ptr = std::make_shared<wave_coefficient_data_config_1d<T>>(other, b, other, other);
    // initial condition:
    auto initial_condition = [](T x) { return std::sin(pi<T>() * x); };
    auto const wave_init_data_ptr = std::make_shared<wave_initial_data_config_1d<T>>(initial_condition, other);
    // wave data config:
    auto const wave_data_ptr = std::make_shared<wave_data_config_1d<T>>(wave_coeffs_data_ptr, wave_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // initialize pde solver
    pde_solver pdesolver(wave_data_ptr, discretization_ptr, boundary_pair, host_fwd_dssolver_solver_config_ptr);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        const T var1 = std::sin(pi<T>() * x);
        const T var2 = std::cos(pi<T>() * t);
        return (var1 * var2);
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

void testImplPureWaveEquationDirichletBCSolverDoubleSweep()
{
    std::cout << "============================================================\n";
    std::cout << "Implicit Pure Wave (Double Sweep) Equation (Dirichlet BC)\n";
    std::cout << "============================================================\n";

    testImplPureWaveEquationDirichletBCSolverHostDoubleSweepDetail<double>();
    testImplPureWaveEquationDirichletBCSolverHostDoubleSweepDetail<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplPureWaveEquationDirichletBCSolverHostLUDetail()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::wave_coefficient_data_config_1d;
    using lss_pde_solvers::wave_data_config_1d;
    using lss_pde_solvers::wave_implicit_solver_config;
    using lss_pde_solvers::wave_initial_data_config_1d;
    using lss_pde_solvers::default_wave_solver_configs::host_fwd_tlusolver_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_wave_equation;
    using lss_utility::pi;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Wave equation: \n\n";
    std::cout << " Using Thomas LU Solver method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_tt(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = sin(pi*x), x in <0,1> \n\n";
    std::cout << " U_x(x,0) = 0, x in <0,1> \n\n";
    std::cout << "============================================================\n";

    // typedef the general_svc_wave_equation
    typedef general_svc_wave_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(1.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(0.7));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto b = [](T x) { return 1.0; };
    auto other = [](T x) { return 0.0; };
    auto const wave_coeffs_data_ptr = std::make_shared<wave_coefficient_data_config_1d<T>>(other, b, other, other);
    // initial condition:
    auto initial_condition = [](T x) { return std::sin(pi<T>() * x); };
    auto const wave_init_data_ptr = std::make_shared<wave_initial_data_config_1d<T>>(initial_condition, other);
    // wave data config:
    auto const wave_data_ptr = std::make_shared<wave_data_config_1d<T>>(wave_coeffs_data_ptr, wave_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // initialize pde solver
    pde_solver pdesolver(wave_data_ptr, discretization_ptr, boundary_pair, host_fwd_tlusolver_solver_config_ptr);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        const T var1 = std::sin(pi<T>() * x);
        const T var2 = std::cos(pi<T>() * t);
        return (var1 * var2);
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

void testImplPureWaveEquationDirichletBCSolverLU()
{
    std::cout << "============================================================\n";
    std::cout << "== Implicit Pure Wave (Thomas LU) Equation (Dirichlet BC) ==\n";
    std::cout << "============================================================\n";

    testImplPureWaveEquationDirichletBCSolverHostLUDetail<double>();
    testImplPureWaveEquationDirichletBCSolverHostLUDetail<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplWaveEquationDirichletBCSolverHostLUDetail()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::wave_coefficient_data_config_1d;
    using lss_pde_solvers::wave_data_config_1d;
    using lss_pde_solvers::wave_implicit_solver_config;
    using lss_pde_solvers::wave_initial_data_config_1d;
    using lss_pde_solvers::default_wave_solver_configs::host_fwd_tlusolver_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_wave_equation;
    using lss_utility::pi;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Wave equation: \n\n";
    std::cout << " Using Thomas LU Solver method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_tt(x,t) = 4*U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = -4*t*t, t > 0 \n\n";
    std::cout << " U(x,0) = x*(1 - x), x in <0,1> \n\n";
    std::cout << " U_x(x,0) = 8*x, x in <0,1> \n\n";
    std::cout << "============================================================\n";

    // typedef the general_svc_wave_equation
    typedef general_svc_wave_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(1.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(0.7));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto b = [](T x) { return 4.0; };
    auto other = [](T x) { return 0.0; };
    auto const wave_coeffs_data_ptr = std::make_shared<wave_coefficient_data_config_1d<T>>(other, b, other, other);
    // initial condition:
    auto first_initial_condition = [](T x) { return (x * (1.0 - x)); };
    auto second_initial_condition = [](T x) { return (8 * x); };
    auto const wave_init_data_ptr =
        std::make_shared<wave_initial_data_config_1d<T>>(first_initial_condition, second_initial_condition);
    // wave data config:
    auto const wave_data_ptr = std::make_shared<wave_data_config_1d<T>>(wave_coeffs_data_ptr, wave_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet_0 = [](T t) { return (-4.0 * t * t); };
    auto const &dirichlet_1 = [](T t) { return (-4.0 * t * t + 8.0 * t); };
    auto const &boundary_low_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_0);
    auto const &boundary_high_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_1);
    auto const &boundary_pair = std::make_pair(boundary_low_ptr, boundary_high_ptr);
    // initialize pde solver
    pde_solver pdesolver(wave_data_ptr, discretization_ptr, boundary_pair, host_fwd_tlusolver_solver_config_ptr);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t) {
        const T res = x - x * x - 4.0 * t * t + 8.0 * t * x;
        return (res);
    };

    T const h = discretization_ptr->space_step();
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        benchmark = exact(j * h, time_range.upper());
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

void testImplWaveEquationDirichletBCSolverLU()
{
    std::cout << "============================================================\n";
    std::cout << "===== Implicit Wave (Thomas LU) Equation (Dirichlet BC) ====\n";
    std::cout << "============================================================\n";

    testImplWaveEquationDirichletBCSolverHostLUDetail<double>();
    testImplWaveEquationDirichletBCSolverHostLUDetail<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplDampedWaveEquationDirichletBCSolverHostDoubleSweepDetail()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::wave_coefficient_data_config_1d;
    using lss_pde_solvers::wave_data_config_1d;
    using lss_pde_solvers::wave_implicit_solver_config;
    using lss_pde_solvers::wave_initial_data_config_1d;
    using lss_pde_solvers::default_wave_solver_configs::host_fwd_dssolver_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_wave_equation;
    using lss_utility::pi;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Wave equation: \n\n";
    std::cout << " Using Double Sweep Solver method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_tt(x,t)  + U_t(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,pi> and t > 0,\n";
    std::cout << " U(0,t) = U(pi,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = sin(x), x in <0,1> \n\n";
    std::cout << " U_x(x,0) = 0, x in <0,1> \n\n";
    std::cout << "============================================================\n";

    // typedef the general_svc_wave_equation
    typedef general_svc_wave_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(pi<T>()));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(0.7));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [](T x) { return 1.0; };
    auto b = [](T x) { return 1.0; };
    auto other = [](T x) { return 0.0; };
    auto const wave_coeffs_data_ptr = std::make_shared<wave_coefficient_data_config_1d<T>>(a, b, other, other);
    // initial condition:
    auto first_initial_condition = [](T x) { return std::sin(x); };
    auto const wave_init_data_ptr = std::make_shared<wave_initial_data_config_1d<T>>(first_initial_condition, other);
    // wave data config:
    auto const wave_data_ptr = std::make_shared<wave_data_config_1d<T>>(wave_coeffs_data_ptr, wave_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet_0 = [](T t) { return 0.0; };
    auto const &dirichlet_1 = [](T t) { return 0.0; };
    auto const &boundary_low_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_0);
    auto const &boundary_high_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_1);
    auto const &boundary_pair = std::make_pair(boundary_low_ptr, boundary_high_ptr);
    // initialize pde solver
    pde_solver pdesolver(wave_data_ptr, discretization_ptr, boundary_pair, host_fwd_dssolver_solver_config_ptr);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t) {
        const T exp_half = std::exp(-0.5 * t);
        const T sqrt_3 = std::sqrt(3.0);
        const T arg = 0.5 * sqrt_3;
        const T res = exp_half * std::sin(x) * (std::cos(arg * t) + (sin(arg * t) / sqrt_3));
        return (res);
    };

    T const h = discretization_ptr->space_step();
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        benchmark = exact(j * h, time_range.upper());
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

void testImplDampedWaveEquationDirichletBCSolverDoubleSweep()
{
    std::cout << "============================================================\n";
    std::cout << "==== Implicit Wave (Double Sweep) Equation (Dirichlet BC) ==\n";
    std::cout << "============================================================\n";

    testImplDampedWaveEquationDirichletBCSolverHostDoubleSweepDetail<double>();
    testImplDampedWaveEquationDirichletBCSolverHostDoubleSweepDetail<float>();

    std::cout << "============================================================\n";
}

// Neumann BC:

template <typename T> void testImplPureWaveEquationNeumannBCCUDASolverDeviceQRDetail()
{
    using lss_boundary::neumann_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::wave_coefficient_data_config_1d;
    using lss_pde_solvers::wave_data_config_1d;
    using lss_pde_solvers::wave_implicit_solver_config;
    using lss_pde_solvers::wave_initial_data_config_1d;
    using lss_pde_solvers::default_wave_solver_configs::dev_fwd_cusolver_qr_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_wave_equation;
    using lss_utility::pi;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Wave equation: \n\n";
    std::cout << " Using CUDA solver with QR (DEVICE) method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_tt(x,t) = 4*U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,pi> and t > 0,\n";
    std::cout << " U_x(0,t) = U_x(pi,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = 3*cos(x), x in <0,pi> \n\n";
    std::cout << " U_x(x,0) = 1 - cos(4*x), x in <0,1> \n\n";
    std::cout << "============================================================\n";

    // typedef the general_svc_wave_equation
    typedef general_svc_wave_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(pi<T>()));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(0.8));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto b = [](T x) { return 4.0; };
    auto other = [](T x) { return 0.0; };
    auto const wave_coeffs_data_ptr = std::make_shared<wave_coefficient_data_config_1d<T>>(other, b, other, other);
    // initial condition:
    auto first_initial_condition = [](T x) { return 3.0 * std::cos(x); };
    auto second_initial_condition = [](T x) { return (1.0 - std::cos(4.0 * x)); };
    auto const wave_init_data_ptr =
        std::make_shared<wave_initial_data_config_1d<T>>(first_initial_condition, second_initial_condition);
    // wave data config:
    auto const wave_data_ptr = std::make_shared<wave_data_config_1d<T>>(wave_coeffs_data_ptr, wave_init_data_ptr);
    // boundary conditions:
    auto const &neumann = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<neumann_boundary_1d<T>>(neumann);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // initialize pde solver
    pde_solver pdesolver(wave_data_ptr, discretization_ptr, boundary_pair, dev_fwd_cusolver_qr_solver_config_ptr);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t) {
        const T var1 = 3.0 * std::cos(2.0 * t) * std::cos(x);
        const T var2 = -0.125 * std::sin(8.0 * t) * std::cos(4.0 * x);
        return (t + var1 + var2);
    };

    T const h = discretization_ptr->space_step();
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        benchmark = exact(j * h, time_range.upper());
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

void testImplPureWaveEquationNeumannBCCUDASolverDeviceQR()
{
    std::cout << "============================================================\n";
    std::cout << "= Implicit Pure Wave (CUDA QR DEVICE) Equation (Neumann BC)=\n";
    std::cout << "============================================================\n";

    testImplPureWaveEquationNeumannBCCUDASolverDeviceQRDetail<double>();
    testImplPureWaveEquationNeumannBCCUDASolverDeviceQRDetail<float>();

    std::cout << "============================================================\n";
}

// ===========================================================================
// ========================== EXPLICIT SOLVERS ===============================
// ===========================================================================

// ===========================================================================
// =========== Wave problem with homogeneous boundary conditions =============
// ===========================================================================

// Dirichlet boundaries:

template <typename T> void testExplPureWaveEquationDirichletBCCUDAHostSolverDetail()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::wave_coefficient_data_config_1d;
    using lss_pde_solvers::wave_data_config_1d;
    using lss_pde_solvers::wave_explicit_solver_config;
    using lss_pde_solvers::wave_initial_data_config_1d;
    using lss_pde_solvers::default_wave_solver_configs::host_expl_fwd_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::explicit_solvers::general_svc_wave_equation;
    using lss_utility::pi;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Wave equation: \n\n";
    std::cout << " Using CUDA solver method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_tt(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = sin(pi*x), x in <0,1> \n\n";
    std::cout << " U_x(x,0) = 0, x in <0,1> \n\n";
    std::cout << "============================================================\n";

    // typedef the general_svc_wave_equation
    typedef general_svc_wave_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 150;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(1.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(0.8));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto b = [](T x) { return 1.0; };
    auto other = [](T x) { return 0.0; };
    auto const wave_coeffs_data_ptr = std::make_shared<wave_coefficient_data_config_1d<T>>(other, b, other, other);
    // initial condition:
    auto initial_condition = [](T x) { return std::sin(pi<T>() * x); };
    auto const wave_init_data_ptr = std::make_shared<wave_initial_data_config_1d<T>>(initial_condition, other);
    // wave data config:
    auto const wave_data_ptr = std::make_shared<wave_data_config_1d<T>>(wave_coeffs_data_ptr, wave_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // initialize pde solver
    pde_solver pdesolver(wave_data_ptr, discretization_ptr, boundary_pair, host_expl_fwd_solver_config_ptr);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t) {
        const T var1 = std::sin(pi<T>() * x);
        const T var2 = std::cos(pi<T>() * t);
        return (var1 * var2);
    };

    T const h = discretization_ptr->space_step();
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        benchmark = exact(j * h, time_range.upper());
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

void testExplPureWaveEquationDirichletBCCUDAHostSolver()
{
    std::cout << "============================================================\n";
    std::cout << "= Implicit Pure Wave (CUDA DEVICE) Equation (Dirichlet BC) =\n";
    std::cout << "============================================================\n";

    testExplPureWaveEquationDirichletBCCUDAHostSolverDetail<double>();
    testExplPureWaveEquationDirichletBCCUDAHostSolverDetail<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testExplPureWaveEquationDirichletBCCUDADeviceSolverDetail()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::wave_coefficient_data_config_1d;
    using lss_pde_solvers::wave_data_config_1d;
    using lss_pde_solvers::wave_explicit_solver_config;
    using lss_pde_solvers::wave_initial_data_config_1d;
    using lss_pde_solvers::default_wave_solver_configs::dev_expl_fwd_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::explicit_solvers::general_svc_wave_equation;
    using lss_utility::pi;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Wave equation: \n\n";
    std::cout << " Using CUDA solver method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_tt(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = sin(pi*x), x in <0,1> \n\n";
    std::cout << " U_x(x,0) = 0, x in <0,1> \n\n";
    std::cout << "============================================================\n";

    // typedef the general_svc_wave_equation
    typedef general_svc_wave_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 150;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(1.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(0.8));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto b = [](T x) { return 1.0; };
    auto other = [](T x) { return 0.0; };
    auto const wave_coeffs_data_ptr = std::make_shared<wave_coefficient_data_config_1d<T>>(other, b, other, other);
    // initial condition:
    auto initial_condition = [](T x) { return std::sin(pi<T>() * x); };
    auto const wave_init_data_ptr = std::make_shared<wave_initial_data_config_1d<T>>(initial_condition, other);
    // wave data config:
    auto const wave_data_ptr = std::make_shared<wave_data_config_1d<T>>(wave_coeffs_data_ptr, wave_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // initialize pde solver
    pde_solver pdesolver(wave_data_ptr, discretization_ptr, boundary_pair, dev_expl_fwd_solver_config_ptr);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t) {
        const T var1 = std::sin(pi<T>() * x);
        const T var2 = std::cos(pi<T>() * t);
        return (var1 * var2);
    };

    T const h = discretization_ptr->space_step();
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        benchmark = exact(j * h, time_range.upper());
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

void testExplPureWaveEquationDirichletBCCUDADeviceSolver()
{
    std::cout << "============================================================\n";
    std::cout << "= Implicit Pure Wave (CUDA DEVICE) Equation (Dirichlet BC) =\n";
    std::cout << "============================================================\n";

    testExplPureWaveEquationDirichletBCCUDADeviceSolverDetail<double>();
    testExplPureWaveEquationDirichletBCCUDADeviceSolverDetail<float>();

    std::cout << "============================================================\n";
}

#endif //_LSS_PURE_WAVE_EQUATION_T_HPP_
