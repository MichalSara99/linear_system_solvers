#if !defined(_LSS_BLACK_SCHOLES_EQUATION_T_HPP_)
#define _LSS_BLACK_SCHOLES_EQUATION_T_HPP_

#include "pde_solvers/one_dimensional/heat_type/lss_1d_general_heat_equation.hpp"
#include <map>

#define PI 3.14159265359

// ///////////////////////////////////////////////////////////////////////////
//							BLACK SCHOLES PROBLEMS
// ///////////////////////////////////////////////////////////////////////////

// ===========================================================================
// ========================== IMPLICIT SOLVERS ===============================
// ===========================================================================

// Dirichlet boundaries:

template <typename T> void testImplBlackScholesEquationDirichletBCCUDASolverDeviceQREuler()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::grid_enum;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_grids::grid_config_1d;
    using lss_grids::grid_config_hints_1d;
    using lss_grids::grid_transform_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::dev_bwd_cusolver_qr_euler_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_heat_equation;
    using lss_utility::black_scholes_exact;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Black-Scholes Call equation: \n\n";
    std::cout << " Using CUDA on DEVICE with QR implicit Euler method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = 0.5*sig*sig*x*x*U_xx(x,t) + r*x*U_x(x,t) - "
                 "r*U(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " 0 < x < 20 and 0 < t < 1,\n";
    std::cout << " U(0,t) = 0 and  U(20,t) = 20-K*exp(-r*(1-t)),0 < t < 1 \n\n";
    std::cout << " U(x,T) = max(0,x-K), x in <0,20> \n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_heat_equation<T, std::vector, std::allocator<T>> pde_solver;
    // set up call option parameters:
    auto const &strike = 10;
    auto const &maturity = 1.0;
    auto const &rate = 0.2;
    auto const &sig = 0.25;
    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(20.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(maturity));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [=](T t, T x) { return 0.5 * sig * sig * x * x; };
    auto b = [=](T t, T x) { return rate * x; };
    auto c = [=](T t, T x) { return -rate; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, b, c);
    // terminal condition:
    auto terminal_condition = [=](T x) { return std::max<T>(0.0, x - strike); };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(terminal_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet_low = [=](T t) { return 0.0; };
    auto const &dirichlet_high = [=](T t) { return (20.0 - strike * std::exp(-rate * (maturity - t))); };
    auto const &boundary_low_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_low);
    auto const &boundary_high_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_high);
    auto const &boundary_pair = std::make_pair(boundary_low_ptr, boundary_high_ptr);
    // grid config:
    auto const alpha_scale = 3.0;
    auto const &grid_config_hints_ptr =
        std::make_shared<grid_config_hints_1d<T>>(strike, alpha_scale, grid_enum::Nonuniform);
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         dev_bwd_cusolver_qr_euler_solver_config_ptr);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    black_scholes_exact<T> bs_exact(0.0, strike, rate, sig, maturity);

    T x{};
    auto const grid_cfg = std::make_shared<grid_config_1d<T>>(discretization_ptr);
    auto const grid_trans_cfg =
        std::make_shared<grid_transform_config_1d<T>>(discretization_ptr, grid_config_hints_ptr);
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        x = grid_1d<T>::transformed_value(grid_trans_cfg, grid_1d<T>::value(grid_cfg, j));
        benchmark = bs_exact.call(x);
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

template <typename T> void testImplBlackScholesEquationDirichletBCCUDASolverDeviceQRCrankNicolson()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::grid_enum;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_grids::grid_config_1d;
    using lss_grids::grid_config_hints_1d;
    using lss_grids::grid_transform_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::dev_bwd_cusolver_qr_cn_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_heat_equation;
    using lss_utility::black_scholes_exact;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Black-Scholes Call equation: \n\n";
    std::cout << " Using CUDA on DEVICE with QR implicit Euler method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = 0.5*sig*sig*x*x*U_xx(x,t) + r*x*U_x(x,t) - "
                 "r*U(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " 0 < x < 20 and 0 < t < 1,\n";
    std::cout << " U(0,t) = 0 and  U(20,t) = 20-K*exp(-r*(1-t)),0 < t < 1 \n\n";
    std::cout << " U(x,T) = max(0,x-K), x in <0,20> \n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_heat_equation<T, std::vector, std::allocator<T>> pde_solver;
    // set up call option parameters:
    auto const &strike = 10;
    auto const &maturity = 1.0;
    auto const &rate = 0.2;
    auto const &sig = 0.25;
    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(20.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(maturity));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [=](T t, T x) { return 0.5 * sig * sig * x * x; };
    auto b = [=](T t, T x) { return rate * x; };
    auto c = [=](T t, T x) { return -rate; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, b, c);
    // terminal condition:
    auto terminal_condition = [=](T x) { return std::max<T>(0.0, x - strike); };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(terminal_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet_low = [=](T t) { return 0.0; };
    auto const &dirichlet_high = [=](T t) { return (20.0 - strike * std::exp(-rate * (maturity - t))); };
    auto const &boundary_low_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_low);
    auto const &boundary_high_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_high);
    auto const &boundary_pair = std::make_pair(boundary_low_ptr, boundary_high_ptr);
    // grid config:
    auto const alpha_scale = 3.0;
    auto const &grid_config_hints_ptr =
        std::make_shared<grid_config_hints_1d<T>>(strike, alpha_scale, grid_enum::Nonuniform);
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         dev_bwd_cusolver_qr_cn_solver_config_ptr);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    black_scholes_exact<T> bs_exact(0.0, strike, rate, sig, maturity);

    T x{};
    auto const grid_cfg = std::make_shared<grid_config_1d<T>>(discretization_ptr);
    auto const grid_trans_cfg =
        std::make_shared<grid_transform_config_1d<T>>(discretization_ptr, grid_config_hints_ptr);
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        x = grid_1d<T>::transformed_value(grid_trans_cfg, grid_1d<T>::value(grid_cfg, j));
        benchmark = bs_exact.call(x);
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

void testImplBlackScholesEquationDirichletBCCUDASolverDeviceQR()
{
    std::cout << "============================================================\n";
    std::cout << " Implicit Black-Scholes (CUDA QR DEVICE) Equation (Dir BC) \n";
    std::cout << "============================================================\n";

    testImplBlackScholesEquationDirichletBCCUDASolverDeviceQREuler<double>();
    testImplBlackScholesEquationDirichletBCCUDASolverDeviceQREuler<float>();
    testImplBlackScholesEquationDirichletBCCUDASolverDeviceQRCrankNicolson<double>();
    testImplBlackScholesEquationDirichletBCCUDASolverDeviceQRCrankNicolson<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplBlackScholesEquationDirichletBCSORSolverDeviceEuler()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::grid_enum;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_grids::grid_config_1d;
    using lss_grids::grid_config_hints_1d;
    using lss_grids::grid_transform_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::dev_bwd_sorsolver_euler_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_heat_equation;
    using lss_utility::black_scholes_exact;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Black-Scholes Call equation: \n\n";
    std::cout << " Using SOR on DEVICE with implicit Euler method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = 0.5*sig*sig*x*x*U_xx(x,t) + r*x*U_x(x,t) - "
                 "r*U(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " 0 < x < 20 and 0 < t < 1,\n";
    std::cout << " U(0,t) = 0 and  U(20,t) = 20-K*exp(-r*(1-t)),0 < t < 1 \n\n";
    std::cout << " U(x,T) = max(0,x-K), x in <0,20> \n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_heat_equation<T, std::vector, std::allocator<T>> pde_solver;
    // set up call option parameters:
    auto const &strike = 10;
    auto const &maturity = 1.0;
    auto const &rate = 0.2;
    auto const &sig = 0.25;
    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(20.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(maturity));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [=](T t, T x) { return 0.5 * sig * sig * x * x; };
    auto b = [=](T t, T x) { return rate * x; };
    auto c = [=](T t, T x) { return -rate; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, b, c);
    // terminal condition:
    auto terminal_condition = [=](T x) { return std::max<T>(0.0, x - strike); };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(terminal_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet_low = [=](T t) { return 0.0; };
    auto const &dirichlet_high = [=](T t) { return (20.0 - strike * std::exp(-rate * (maturity - t))); };
    auto const &boundary_low_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_low);
    auto const &boundary_high_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_high);
    auto const &boundary_pair = std::make_pair(boundary_low_ptr, boundary_high_ptr);
    // grid config:
    auto const alpha_scale = 3.0;
    auto const &grid_config_hints_ptr =
        std::make_shared<grid_config_hints_1d<T>>(strike, alpha_scale, grid_enum::Nonuniform);
    // details:
    std::map<std::string, T> details;
    details["sor_omega"] = static_cast<T>(1.0);
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         dev_bwd_sorsolver_euler_solver_config_ptr, details);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    black_scholes_exact<T> bs_exact(0.0, strike, rate, sig, maturity);

    T x{};
    auto const grid_cfg = std::make_shared<grid_config_1d<T>>(discretization_ptr);
    auto const grid_trans_cfg =
        std::make_shared<grid_transform_config_1d<T>>(discretization_ptr, grid_config_hints_ptr);
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        x = grid_1d<T>::transformed_value(grid_trans_cfg, grid_1d<T>::value(grid_cfg, j));
        benchmark = bs_exact.call(x);
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

template <typename T> void testImplBlackScholesEquationDirichletBCSORSolverDeviceCrankNicolson()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::grid_enum;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_grids::grid_config_1d;
    using lss_grids::grid_config_hints_1d;
    using lss_grids::grid_transform_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::dev_bwd_sorsolver_cn_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_heat_equation;
    using lss_utility::black_scholes_exact;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Black-Scholes Call equation: \n\n";
    std::cout << " Using SOR on DEVICE with implicit CN method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = 0.5*sig*sig*x*x*U_xx(x,t) + r*x*U_x(x,t) - "
                 "r*U(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " 0 < x < 20 and 0 < t < 1,\n";
    std::cout << " U(0,t) = 0 and  U(20,t) = 20-K*exp(-r*(1-t)),0 < t < 1 \n\n";
    std::cout << " U(x,T) = max(0,x-K), x in <0,20> \n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_heat_equation<T, std::vector, std::allocator<T>> pde_solver;
    // set up call option parameters:
    auto const &strike = 10;
    auto const &maturity = 1.0;
    auto const &rate = 0.2;
    auto const &sig = 0.25;
    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(20.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(maturity));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [=](T t, T x) { return 0.5 * sig * sig * x * x; };
    auto b = [=](T t, T x) { return rate * x; };
    auto c = [=](T t, T x) { return -rate; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, b, c);
    // terminal condition:
    auto terminal_condition = [=](T x) { return std::max<T>(0.0, x - strike); };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(terminal_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet_low = [=](T t) { return 0.0; };
    auto const &dirichlet_high = [=](T t) { return (20.0 - strike * std::exp(-rate * (maturity - t))); };
    auto const &boundary_low_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_low);
    auto const &boundary_high_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_high);
    auto const &boundary_pair = std::make_pair(boundary_low_ptr, boundary_high_ptr);
    // grid config:
    auto const alpha_scale = 3.0;
    auto const &grid_config_hints_ptr =
        std::make_shared<grid_config_hints_1d<T>>(strike, alpha_scale, grid_enum::Nonuniform);
    // details:
    std::map<std::string, T> details;
    details["sor_omega"] = static_cast<T>(1.0);
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         dev_bwd_sorsolver_cn_solver_config_ptr, details);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    black_scholes_exact<T> bs_exact(0.0, strike, rate, sig, maturity);

    T x{};
    auto const grid_cfg = std::make_shared<grid_config_1d<T>>(discretization_ptr);
    auto const grid_trans_cfg =
        std::make_shared<grid_transform_config_1d<T>>(discretization_ptr, grid_config_hints_ptr);
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        x = grid_1d<T>::transformed_value(grid_trans_cfg, grid_1d<T>::value(grid_cfg, j));
        benchmark = bs_exact.call(x);
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

void testImplBlackScholesEquationDirichletBCSORSolverDevice()
{
    std::cout << "============================================================\n";
    std::cout << "= Implicit Black-Scholes (SOR DEVICE) Equation (Dir BC) ====\n";
    std::cout << "============================================================\n";

    testImplBlackScholesEquationDirichletBCSORSolverDeviceEuler<double>();
    testImplBlackScholesEquationDirichletBCSORSolverDeviceEuler<float>();
    testImplBlackScholesEquationDirichletBCSORSolverDeviceCrankNicolson<double>();
    testImplBlackScholesEquationDirichletBCSORSolverDeviceCrankNicolson<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplBlackScholesEquationDirichletBCSORSolverHostEuler()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::grid_enum;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_grids::grid_config_1d;
    using lss_grids::grid_config_hints_1d;
    using lss_grids::grid_transform_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::host_bwd_sorsolver_euler_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_heat_equation;
    using lss_utility::black_scholes_exact;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Black-Scholes Call equation: \n\n";
    std::cout << " Using SOR on HOST with implicit Euler method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = 0.5*sig*sig*x*x*U_xx(x,t) + r*x*U_x(x,t) - "
                 "r*U(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " 0 < x < 20 and 0 < t < 1,\n";
    std::cout << " U(0,t) = 0 and  U(20,t) = 20-K*exp(-r*(1-t)),0 < t < 1 \n\n";
    std::cout << " U(x,T) = max(0,x-K), x in <0,20> \n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_heat_equation<T, std::vector, std::allocator<T>> pde_solver;
    // set up call option parameters:
    auto const &strike = 10;
    auto const &maturity = 1.0;
    auto const &rate = 0.2;
    auto const &sig = 0.25;
    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(20.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(maturity));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [=](T t, T x) { return 0.5 * sig * sig * x * x; };
    auto b = [=](T t, T x) { return rate * x; };
    auto c = [=](T t, T x) { return -rate; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, b, c);
    // terminal condition:
    auto terminal_condition = [=](T x) { return std::max<T>(0.0, x - strike); };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(terminal_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet_low = [=](T t) { return 0.0; };
    auto const &dirichlet_high = [=](T t) { return (20.0 - strike * std::exp(-rate * (maturity - t))); };
    auto const &boundary_low_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_low);
    auto const &boundary_high_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_high);
    auto const &boundary_pair = std::make_pair(boundary_low_ptr, boundary_high_ptr);
    // grid config:
    auto const alpha_scale = 3.0;
    auto const &grid_config_hints_ptr =
        std::make_shared<grid_config_hints_1d<T>>(strike, alpha_scale, grid_enum::Nonuniform);
    // details:
    std::map<std::string, T> details;
    details["sor_omega"] = static_cast<T>(1.0);
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         host_bwd_sorsolver_euler_solver_config_ptr, details);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    black_scholes_exact<T> bs_exact(0.0, strike, rate, sig, maturity);

    T x{};
    auto const grid_cfg = std::make_shared<grid_config_1d<T>>(discretization_ptr);
    auto const grid_trans_cfg =
        std::make_shared<grid_transform_config_1d<T>>(discretization_ptr, grid_config_hints_ptr);
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        x = grid_1d<T>::transformed_value(grid_trans_cfg, grid_1d<T>::value(grid_cfg, j));
        benchmark = bs_exact.call(x);
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

template <typename T> void testImplBlackScholesEquationDirichletBCSORSolverHostCrankNicolson()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::grid_enum;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_grids::grid_config_1d;
    using lss_grids::grid_config_hints_1d;
    using lss_grids::grid_transform_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::host_bwd_sorsolver_cn_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_heat_equation;
    using lss_utility::black_scholes_exact;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Black-Scholes Call equation: \n\n";
    std::cout << " Using SOR on HOST with implicit CN method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = 0.5*sig*sig*x*x*U_xx(x,t) + r*x*U_x(x,t) - "
                 "r*U(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " 0 < x < 20 and 0 < t < 1,\n";
    std::cout << " U(0,t) = 0 and  U(20,t) = 20-K*exp(-r*(1-t)),0 < t < 1 \n\n";
    std::cout << " U(x,T) = max(0,x-K), x in <0,20> \n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_heat_equation<T, std::vector, std::allocator<T>> pde_solver;
    // set up call option parameters:
    auto const &strike = 10;
    auto const &maturity = 1.0;
    auto const &rate = 0.2;
    auto const &sig = 0.25;
    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(20.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(maturity));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [=](T t, T x) { return 0.5 * sig * sig * x * x; };
    auto b = [=](T t, T x) { return rate * x; };
    auto c = [=](T t, T x) { return -rate; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, b, c);
    // terminal condition:
    auto terminal_condition = [=](T x) { return std::max<T>(0.0, x - strike); };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(terminal_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet_low = [=](T t) { return 0.0; };
    auto const &dirichlet_high = [=](T t) { return (20.0 - strike * std::exp(-rate * (maturity - t))); };
    auto const &boundary_low_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_low);
    auto const &boundary_high_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_high);
    auto const &boundary_pair = std::make_pair(boundary_low_ptr, boundary_high_ptr);
    // grid config:
    auto const alpha_scale = 3.0;
    auto const &grid_config_hints_ptr =
        std::make_shared<grid_config_hints_1d<T>>(strike, alpha_scale, grid_enum::Nonuniform);
    // details:
    std::map<std::string, T> details;
    details["sor_omega"] = static_cast<T>(1.0);
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         host_bwd_sorsolver_cn_solver_config_ptr, details);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    black_scholes_exact<T> bs_exact(0.0, strike, rate, sig, maturity);

    T x{};
    auto const grid_cfg = std::make_shared<grid_config_1d<T>>(discretization_ptr);
    auto const grid_trans_cfg =
        std::make_shared<grid_transform_config_1d<T>>(discretization_ptr, grid_config_hints_ptr);
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        x = grid_1d<T>::transformed_value(grid_trans_cfg, grid_1d<T>::value(grid_cfg, j));
        benchmark = bs_exact.call(x);
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

void testImplBlackScholesEquationDirichletBCSORSolverHost()
{
    std::cout << "============================================================\n";
    std::cout << "=== Implicit Black-Scholes (SOR HOST) Equation (Dir BC) ====\n";
    std::cout << "============================================================\n";

    testImplBlackScholesEquationDirichletBCSORSolverHostEuler<double>();
    testImplBlackScholesEquationDirichletBCSORSolverHostEuler<float>();
    testImplBlackScholesEquationDirichletBCSORSolverHostCrankNicolson<double>();
    testImplBlackScholesEquationDirichletBCSORSolverHostCrankNicolson<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplBlackScholesEquationDirichletBCDoubleSweepSolverEuler()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::grid_enum;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_grids::grid_config_1d;
    using lss_grids::grid_config_hints_1d;
    using lss_grids::grid_transform_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::host_bwd_dssolver_euler_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_heat_equation;
    using lss_utility::black_scholes_exact;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Black-Scholes Call equation: \n\n";
    std::cout << " Using Double Sweep on HOST with implicit Euler method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = 0.5*sig*sig*x*x*U_xx(x,t) + r*x*U_x(x,t) - "
                 "r*U(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " 0 < x < 20 and 0 < t < 1,\n";
    std::cout << " U(0,t) = 0 and  U(20,t) = 20-K*exp(-r*(1-t)),0 < t < 1 \n\n";
    std::cout << " U(x,T) = max(0,x-K), x in <0,20> \n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_heat_equation<T, std::vector, std::allocator<T>> pde_solver;
    // set up call option parameters:
    auto const &strike = 10;
    auto const &maturity = 1.0;
    auto const &rate = 0.2;
    auto const &sig = 0.25;
    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(20.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(maturity));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [=](T t, T x) { return 0.5 * sig * sig * x * x; };
    auto b = [=](T t, T x) { return rate * x; };
    auto c = [=](T t, T x) { return -rate; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, b, c);
    // terminal condition:
    auto terminal_condition = [=](T x) { return std::max<T>(0.0, x - strike); };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(terminal_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet_low = [=](T t) { return 0.0; };
    auto const &dirichlet_high = [=](T t) { return (20.0 - strike * std::exp(-rate * (maturity - t))); };
    auto const &boundary_low_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_low);
    auto const &boundary_high_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_high);
    auto const &boundary_pair = std::make_pair(boundary_low_ptr, boundary_high_ptr);
    // grid config:
    auto const alpha_scale = 3.0;
    auto const &grid_config_hints_ptr =
        std::make_shared<grid_config_hints_1d<T>>(strike, alpha_scale, grid_enum::Nonuniform);
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         host_bwd_dssolver_euler_solver_config_ptr);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    black_scholes_exact<T> bs_exact(0.0, strike, rate, sig, maturity);

    T x{};
    auto const grid_cfg = std::make_shared<grid_config_1d<T>>(discretization_ptr);
    auto const grid_trans_cfg =
        std::make_shared<grid_transform_config_1d<T>>(discretization_ptr, grid_config_hints_ptr);
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        x = grid_1d<T>::transformed_value(grid_trans_cfg, grid_1d<T>::value(grid_cfg, j));
        benchmark = bs_exact.call(x);
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

template <typename T> void testImplBlackScholesEquationDirichletBCDoubleSweepSolverCrankNicolson()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::grid_enum;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_grids::grid_config_1d;
    using lss_grids::grid_config_hints_1d;
    using lss_grids::grid_transform_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::host_bwd_dssolver_cn_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_heat_equation;
    using lss_utility::black_scholes_exact;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Black-Scholes Call equation: \n\n";
    std::cout << " Using Double Sweep on HOST with implicit CN method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = 0.5*sig*sig*x*x*U_xx(x,t) + r*x*U_x(x,t) - "
                 "r*U(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " 0 < x < 20 and 0 < t < 1,\n";
    std::cout << " U(0,t) = 0 and  U(20,t) = 20-K*exp(-r*(1-t)),0 < t < 1 \n\n";
    std::cout << " U(x,T) = max(0,x-K), x in <0,20> \n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_heat_equation<T, std::vector, std::allocator<T>> pde_solver;
    // set up call option parameters:
    auto const &strike = 10;
    auto const &maturity = 1.0;
    auto const &rate = 0.2;
    auto const &sig = 0.25;
    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(20.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(maturity));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [=](T t, T x) { return 0.5 * sig * sig * x * x; };
    auto b = [=](T t, T x) { return rate * x; };
    auto c = [=](T t, T x) { return -rate; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, b, c);
    // terminal condition:
    auto terminal_condition = [=](T x) { return std::max<T>(0.0, x - strike); };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(terminal_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet_low = [=](T t) { return 0.0; };
    auto const &dirichlet_high = [=](T t) { return (20.0 - strike * std::exp(-rate * (maturity - t))); };
    auto const &boundary_low_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_low);
    auto const &boundary_high_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_high);
    auto const &boundary_pair = std::make_pair(boundary_low_ptr, boundary_high_ptr);
    // grid config:
    auto const alpha_scale = 3.0;
    auto const &grid_config_hints_ptr =
        std::make_shared<grid_config_hints_1d<T>>(strike, alpha_scale, grid_enum::Nonuniform);
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         host_bwd_dssolver_cn_solver_config_ptr);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    black_scholes_exact<T> bs_exact(0.0, strike, rate, sig, maturity);

    T x{};
    auto const grid_cfg = std::make_shared<grid_config_1d<T>>(discretization_ptr);
    auto const grid_trans_cfg =
        std::make_shared<grid_transform_config_1d<T>>(discretization_ptr, grid_config_hints_ptr);
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        x = grid_1d<T>::transformed_value(grid_trans_cfg, grid_1d<T>::value(grid_cfg, j));
        benchmark = bs_exact.call(x);
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

void testImplBlackScholesEquationDirichletBCDoubleSweepSolver()
{
    std::cout << "============================================================\n";
    std::cout << "= Implicit Black-Scholes (Double Sweep) Equation (Dir BC) ==\n";
    std::cout << "============================================================\n";

    testImplBlackScholesEquationDirichletBCDoubleSweepSolverEuler<double>();
    testImplBlackScholesEquationDirichletBCDoubleSweepSolverEuler<float>();
    testImplBlackScholesEquationDirichletBCDoubleSweepSolverCrankNicolson<double>();
    testImplBlackScholesEquationDirichletBCDoubleSweepSolverCrankNicolson<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplBlackScholesEquationDirichletBCThomasLUSolverEuler()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_grids::grid_config_1d;
    using lss_grids::grid_config_hints_1d;
    using lss_grids::grid_transform_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::host_bwd_tlusolver_euler_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_heat_equation;
    using lss_utility::black_scholes_exact;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Black-Scholes Call equation: \n\n";
    std::cout << " Using Thomas LU on HOST with implicit Euler method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = 0.5*sig*sig*x*x*U_xx(x,t) + r*x*U_x(x,t) - "
                 "r*U(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " 0 < x < 20 and 0 < t < 1,\n";
    std::cout << " U(0,t) = 0 and  U(20,t) = 20-K*exp(-r*(1-t)),0 < t < 1 \n\n";
    std::cout << " U(x,T) = max(0,x-K), x in <0,20> \n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_heat_equation<T, std::vector, std::allocator<T>> pde_solver;
    // set up call option parameters:
    auto const &strike = 10;
    auto const &maturity = 1.0;
    auto const &rate = 0.2;
    auto const &sig = 0.25;
    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(20.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(maturity));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [=](T t, T x) { return 0.5 * sig * sig * x * x; };
    auto b = [=](T t, T x) { return rate * x; };
    auto c = [=](T t, T x) { return -rate; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, b, c);
    // terminal condition:
    auto terminal_condition = [=](T x) { return std::max<T>(0.0, x - strike); };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(terminal_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet_low = [=](T t) { return 0.0; };
    auto const &dirichlet_high = [=](T t) { return (20.0 - strike * std::exp(-rate * (maturity - t))); };
    auto const &boundary_low_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_low);
    auto const &boundary_high_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_high);
    auto const &boundary_pair = std::make_pair(boundary_low_ptr, boundary_high_ptr);
    // grid config:
    auto const &grid_config_hints_ptr = std::make_shared<grid_config_hints_1d<T>>();
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         host_bwd_tlusolver_euler_solver_config_ptr);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    black_scholes_exact<T> bs_exact(0.0, strike, rate, sig, maturity);

    T x{};
    auto const grid_cfg = std::make_shared<grid_config_1d<T>>(discretization_ptr);
    auto const grid_trans_cfg =
        std::make_shared<grid_transform_config_1d<T>>(discretization_ptr, grid_config_hints_ptr);
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        x = grid_1d<T>::transformed_value(grid_trans_cfg, grid_1d<T>::value(grid_cfg, j));
        benchmark = bs_exact.call(x);
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

template <typename T> void testImplBlackScholesEquationDirichletBCThomasLUSolverCrankNicolson()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_grids::grid_config_1d;
    using lss_grids::grid_config_hints_1d;
    using lss_grids::grid_transform_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::host_bwd_tlusolver_cn_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_heat_equation;
    using lss_utility::black_scholes_exact;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Black-Scholes Call equation: \n\n";
    std::cout << " Using Thomas LU on HOST with implicit CN method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = 0.5*sig*sig*x*x*U_xx(x,t) + r*x*U_x(x,t) - "
                 "r*U(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " 0 < x < 20 and 0 < t < 1,\n";
    std::cout << " U(0,t) = 0 and  U(20,t) = 20-K*exp(-r*(1-t)),0 < t < 1 \n\n";
    std::cout << " U(x,T) = max(0,x-K), x in <0,20> \n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_heat_equation<T, std::vector, std::allocator<T>> pde_solver;
    // set up call option parameters:
    auto const &strike = 10;
    auto const &maturity = 1.0;
    auto const &rate = 0.2;
    auto const &sig = 0.25;
    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(20.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(maturity));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [=](T t, T x) { return 0.5 * sig * sig * x * x; };
    auto b = [=](T t, T x) { return rate * x; };
    auto c = [=](T t, T x) { return -rate; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, b, c);
    // terminal condition:
    auto terminal_condition = [=](T x) { return std::max<T>(0.0, x - strike); };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(terminal_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet_low = [=](T t) { return 0.0; };
    auto const &dirichlet_high = [=](T t) { return (20.0 - strike * std::exp(-rate * (maturity - t))); };
    auto const &boundary_low_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_low);
    auto const &boundary_high_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_high);
    auto const &boundary_pair = std::make_pair(boundary_low_ptr, boundary_high_ptr);
    // grid config:
    auto const &grid_config_hints_ptr = std::make_shared<grid_config_hints_1d<T>>();
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         host_bwd_tlusolver_cn_solver_config_ptr);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    black_scholes_exact<T> bs_exact(0.0, strike, rate, sig, maturity);

    T x{};
    auto const grid_cfg = std::make_shared<grid_config_1d<T>>(discretization_ptr);
    auto const grid_trans_cfg =
        std::make_shared<grid_transform_config_1d<T>>(discretization_ptr, grid_config_hints_ptr);
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        x = grid_1d<T>::transformed_value(grid_trans_cfg, grid_1d<T>::value(grid_cfg, j));
        benchmark = bs_exact.call(x);
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

void testImplBlackScholesEquationDirichletBCThomasLUSolver()
{
    std::cout << "============================================================\n";
    std::cout << "== Implicit Black-Scholes (Thomas LU) Equation (Dir BC) ====\n";
    std::cout << "============================================================\n";

    testImplBlackScholesEquationDirichletBCThomasLUSolverEuler<double>();
    testImplBlackScholesEquationDirichletBCThomasLUSolverEuler<float>();
    testImplBlackScholesEquationDirichletBCThomasLUSolverCrankNicolson<double>();
    testImplBlackScholesEquationDirichletBCThomasLUSolverCrankNicolson<float>();

    std::cout << "============================================================\n";
}

// forward starting call:
template <typename T> void testImplFwdBlackScholesEquationDirichletBCCUDASolverDeviceQREuler()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_grids::grid_config_1d;
    using lss_grids::grid_config_hints_1d;
    using lss_grids::grid_transform_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::dev_bwd_cusolver_qr_euler_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_heat_equation;
    using lss_utility::black_scholes_exact;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Black-Scholes Call equation: \n\n";
    std::cout << " Using CUDA on DEVICE with QR implicit Euler method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = 0.5*sig*sig*x*x*U_xx(x,t) + r*x*U_x(x,t) - "
                 "r*U(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " 0 < x < 20 and 0.5 < t < 1,\n";
    std::cout << " U(0,t) = 0 and  U(20,t) = 20-K*exp(-r*(1-t)),0.5 < t < 1 \n\n";
    std::cout << " U(x,T) = max(0,x-K), x in <0,20> \n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_heat_equation<T, std::vector, std::allocator<T>> pde_solver;
    // set up call option parameters:
    auto const &strike = 10;
    auto const &maturity = 1.0;
    auto const &rate = 0.2;
    auto const &sig = 0.25;
    auto const &fwd_start = 0.5;
    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(20.0));
    // time range
    range<T> time_range(static_cast<T>(fwd_start), static_cast<T>(maturity));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [=](T t, T x) { return 0.5 * sig * sig * x * x; };
    auto b = [=](T t, T x) { return rate * x; };
    auto c = [=](T t, T x) { return -rate; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, b, c);
    // terminal condition:
    auto terminal_condition = [=](T x) { return std::max<T>(0.0, x - strike); };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(terminal_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet_low = [=](T t) { return 0.0; };
    auto const &dirichlet_high = [=](T t) { return (20.0 - strike * std::exp(-rate * (maturity - t))); };
    auto const &boundary_low_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_low);
    auto const &boundary_high_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_high);
    auto const &boundary_pair = std::make_pair(boundary_low_ptr, boundary_high_ptr);
    // grid config:
    auto const &grid_config_hints_ptr = std::make_shared<grid_config_hints_1d<T>>();
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         dev_bwd_cusolver_qr_euler_solver_config_ptr);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    black_scholes_exact<T> bs_exact(fwd_start, strike, rate, sig, maturity);

    T x{};
    auto const grid_cfg = std::make_shared<grid_config_1d<T>>(discretization_ptr);
    auto const grid_trans_cfg =
        std::make_shared<grid_transform_config_1d<T>>(discretization_ptr, grid_config_hints_ptr);
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        x = grid_1d<T>::transformed_value(grid_trans_cfg, grid_1d<T>::value(grid_cfg, j));
        benchmark = bs_exact.call(x);
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

template <typename T> void testImplFwdBlackScholesEquationDirichletBCCUDASolverDeviceQRCrankNicolson()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_grids::grid_config_1d;
    using lss_grids::grid_config_hints_1d;
    using lss_grids::grid_transform_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::dev_bwd_cusolver_qr_cn_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_heat_equation;
    using lss_utility::black_scholes_exact;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Black-Scholes Call equation: \n\n";
    std::cout << " Using CUDA on DEVICE with QR implicit Euler method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = 0.5*sig*sig*x*x*U_xx(x,t) + r*x*U_x(x,t) - "
                 "r*U(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " 0 < x < 20 and 0.5 < t < 1,\n";
    std::cout << " U(0,t) = 0 and  U(20,t) = 20-K*exp(-r*(1-t)),0.5 < t < 1 \n\n";
    std::cout << " U(x,T) = max(0,x-K), x in <0,20> \n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_heat_equation<T, std::vector, std::allocator<T>> pde_solver;
    // set up call option parameters:
    auto const &strike = 10;
    auto const &maturity = 1.0;
    auto const &rate = 0.2;
    auto const &sig = 0.25;
    auto const &fwd_start = 0.5;
    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(20.0));
    // time range
    range<T> time_range(static_cast<T>(fwd_start), static_cast<T>(maturity));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [=](T t, T x) { return 0.5 * sig * sig * x * x; };
    auto b = [=](T t, T x) { return rate * x; };
    auto c = [=](T t, T x) { return -rate; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, b, c);
    // terminal condition:
    auto terminal_condition = [=](T x) { return std::max<T>(0.0, x - strike); };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(terminal_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet_low = [=](T t) { return 0.0; };
    auto const &dirichlet_high = [=](T t) { return (20.0 - strike * std::exp(-rate * (maturity - t))); };
    auto const &boundary_low_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_low);
    auto const &boundary_high_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_high);
    auto const &boundary_pair = std::make_pair(boundary_low_ptr, boundary_high_ptr);
    // grid config:
    auto const &grid_config_hints_ptr = std::make_shared<grid_config_hints_1d<T>>();
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         dev_bwd_cusolver_qr_cn_solver_config_ptr);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    black_scholes_exact<T> bs_exact(fwd_start, strike, rate, sig, maturity);

    T x{};
    auto const grid_cfg = std::make_shared<grid_config_1d<T>>(discretization_ptr);
    auto const grid_trans_cfg =
        std::make_shared<grid_transform_config_1d<T>>(discretization_ptr, grid_config_hints_ptr);
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        x = grid_1d<T>::transformed_value(grid_trans_cfg, grid_1d<T>::value(grid_cfg, j));
        benchmark = bs_exact.call(x);
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

void testImplFwdBlackScholesEquationDirichletBCCUDASolverDeviceQR()
{
    std::cout << "============================================================\n";
    std::cout << " Implicit Black-Scholes (CUDA QR DEVICE) Equation (Dir BC) \n";
    std::cout << "============================================================\n";

    testImplFwdBlackScholesEquationDirichletBCCUDASolverDeviceQREuler<double>();
    testImplFwdBlackScholesEquationDirichletBCCUDASolverDeviceQREuler<float>();
    testImplFwdBlackScholesEquationDirichletBCCUDASolverDeviceQRCrankNicolson<double>();
    testImplFwdBlackScholesEquationDirichletBCCUDASolverDeviceQRCrankNicolson<float>();

    std::cout << "============================================================\n";
}

// Uisng stepping = getting the whole surface
template <typename T> void testImplBlackScholesEquationDirichletBCThomasLUSolverEulerStepping()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_containers::container_2d;
    using lss_enumerations::by_enum;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_grids::grid_config_1d;
    using lss_grids::grid_config_hints_1d;
    using lss_grids::grid_transform_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::host_bwd_tlusolver_euler_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_heat_equation;
    using lss_utility::black_scholes_exact;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Black-Scholes Call equation: \n\n";
    std::cout << " Using Thomas LU on HOST with implicit Euler method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = 0.5*sig*sig*x*x*U_xx(x,t) + r*x*U_x(x,t) - "
                 "r*U(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " 0 < x < 20 and 0 < t < 1,\n";
    std::cout << " U(0,t) = 0 and  U(20,t) = 20-K*exp(-r*(1-t)),0 < t < 1 \n\n";
    std::cout << " U(x,T) = max(0,x-K), x in <0,20> \n\n";
    std::cout << "============================================================\n";

    // typedef the constiner_2d:
    typedef container_2d<by_enum::Row, T, std::vector, std::allocator<T>> container_2d_t;
    // typedef the Implicit1DHeatEquation
    typedef general_heat_equation<T, std::vector, std::allocator<T>> pde_solver;
    // set up call option parameters:
    auto const &strike = 10;
    auto const &maturity = 1.0;
    auto const &rate = 0.2;
    auto const &sig = 0.25;
    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(20.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(maturity));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [=](T t, T x) { return 0.5 * sig * sig * x * x; };
    auto b = [=](T t, T x) { return rate * x; };
    auto c = [=](T t, T x) { return -rate; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, b, c);
    // terminal condition:
    auto terminal_condition = [=](T x) { return std::max<T>(0.0, x - strike); };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(terminal_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet_low = [=](T t) { return 0.0; };
    auto const &dirichlet_high = [=](T t) { return (20.0 - strike * std::exp(-rate * (maturity - t))); };
    auto const &boundary_low_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_low);
    auto const &boundary_high_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_high);
    auto const &boundary_pair = std::make_pair(boundary_low_ptr, boundary_high_ptr);
    // grid config:
    auto const &grid_config_hints_ptr = std::make_shared<grid_config_hints_1d<T>>();
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         host_bwd_tlusolver_euler_solver_config_ptr);
    // prepare container for solutions:
    container_2d_t solutions(Td, Sd);
    // get the solution:
    pdesolver.solve(solutions);
    // get exact solution:
    black_scholes_exact<T> bs_exact(0.0, strike, rate, sig, maturity);

    T const k = discretization_ptr->time_step();
    T x{};
    auto const grid_cfg = std::make_shared<grid_config_1d<T>>(discretization_ptr);
    auto const grid_trans_cfg =
        std::make_shared<grid_transform_config_1d<T>>(discretization_ptr, grid_config_hints_ptr);
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t t = 0; t < solutions.rows(); ++t)
    {
        std::cout << "time: " << t * k << ":\n";
        for (std::size_t j = 0; j < solutions.columns(); ++j)
        {
            x = grid_1d<T>::transformed_value(grid_trans_cfg, grid_1d<T>::value(grid_cfg, j));
            benchmark = bs_exact.call(x, maturity - t * k);
            std::cout << "t_" << j << ": " << solutions(t, j) << " |  " << benchmark << " | "
                      << (solutions(t, j) - benchmark) << '\n';
        }
    }
}

template <typename T> void testImplBlackScholesEquationDirichletBCThomasLUSolverCrankNicolsonStepping()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_containers::container_2d;
    using lss_enumerations::by_enum;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_grids::grid_config_1d;
    using lss_grids::grid_config_hints_1d;
    using lss_grids::grid_transform_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::host_bwd_tlusolver_cn_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_heat_equation;
    using lss_utility::black_scholes_exact;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Black-Scholes Call equation: \n\n";
    std::cout << " Using Thomas LU on HOST with implicit CN method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = 0.5*sig*sig*x*x*U_xx(x,t) + r*x*U_x(x,t) - "
                 "r*U(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " 0 < x < 20 and 0 < t < 1,\n";
    std::cout << " U(0,t) = 0 and  U(20,t) = 20-K*exp(-r*(1-t)),0 < t < 1 \n\n";
    std::cout << " U(x,T) = max(0,x-K), x in <0,20> \n\n";
    std::cout << "============================================================\n";

    // typedef the constiner_2d:
    typedef container_2d<by_enum::Row, T, std::vector, std::allocator<T>> container_2d_t;

    // typedef the Implicit1DHeatEquation
    typedef general_heat_equation<T, std::vector, std::allocator<T>> pde_solver;
    // set up call option parameters:
    auto const &strike = 10;
    auto const &maturity = 1.0;
    auto const &rate = 0.2;
    auto const &sig = 0.25;
    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(20.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(maturity));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [=](T t, T x) { return 0.5 * sig * sig * x * x; };
    auto b = [=](T t, T x) { return rate * x; };
    auto c = [=](T t, T x) { return -rate; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, b, c);
    // terminal condition:
    auto terminal_condition = [=](T x) { return std::max<T>(0.0, x - strike); };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(terminal_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet_low = [=](T t) { return 0.0; };
    auto const &dirichlet_high = [=](T t) { return (20.0 - strike * std::exp(-rate * (maturity - t))); };
    auto const &boundary_low_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_low);
    auto const &boundary_high_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_high);
    auto const &boundary_pair = std::make_pair(boundary_low_ptr, boundary_high_ptr);
    // grid config:
    auto const &grid_config_hints_ptr = std::make_shared<grid_config_hints_1d<T>>();
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         host_bwd_tlusolver_cn_solver_config_ptr);
    // prepare container for solutions:
    container_2d_t solutions(Td, Sd);
    // get the solution:
    pdesolver.solve(solutions);
    // get exact solution:
    black_scholes_exact<T> bs_exact(0.0, strike, rate, sig, maturity);

    T const k = discretization_ptr->time_step();
    T x{};
    auto const grid_cfg = std::make_shared<grid_config_1d<T>>(discretization_ptr);
    auto const grid_trans_cfg =
        std::make_shared<grid_transform_config_1d<T>>(discretization_ptr, grid_config_hints_ptr);
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t t = 0; t < solutions.rows(); ++t)
    {
        std::cout << "time: " << t * k << ":\n";
        for (std::size_t j = 0; j < solutions.columns(); ++j)
        {
            x = grid_1d<T>::transformed_value(grid_trans_cfg, grid_1d<T>::value(grid_cfg, j));
            benchmark = bs_exact.call(x, maturity - t * k);
            std::cout << "t_" << j << ": " << solutions(t, j) << " |  " << benchmark << " | "
                      << (solutions(t, j) - benchmark) << '\n';
        }
    }
}

void testImplBlackScholesEquationDirichletBCThomasLUSolverStepping()
{
    std::cout << "============================================================\n";
    std::cout << "== Implicit Black-Scholes (Thomas LU) Equation (Dir BC) ====\n";
    std::cout << "============================================================\n";

    testImplBlackScholesEquationDirichletBCThomasLUSolverEulerStepping<double>();
    testImplBlackScholesEquationDirichletBCThomasLUSolverEulerStepping<float>();
    testImplBlackScholesEquationDirichletBCThomasLUSolverCrankNicolsonStepping<double>();
    testImplBlackScholesEquationDirichletBCThomasLUSolverCrankNicolsonStepping<float>();

    std::cout << "============================================================\n";
}

// ===========================================================================
// ========================== EXPLICIT SOLVERS ===============================
// ===========================================================================

// Dirichlet boundaries:

template <typename T> void testExplBlackScholesEquationDirichletBCBarakatClark()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::explicit_pde_schemes_enum;
    using lss_grids::grid_config_1d;
    using lss_grids::grid_config_hints_1d;
    using lss_grids::grid_transform_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_explicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::host_expl_bwd_bc_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::explicit_solvers::general_heat_equation;
    using lss_utility::black_scholes_exact;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Black-Scholes Call equation: \n\n";
    std::cout << " Using Barakat-Clark ADE method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = 0.5*sig*sig*x*x*U_xx(x,t) + r*x*U_x(x,t) - "
                 "r*U(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " 0 < x < 20 and 0 < t < 1,\n";
    std::cout << " U(0,t) = 0 and  U(20,t) = 20-K*exp(-r*(1-t)),0 < t < 1 \n\n";
    std::cout << " U(x,T) = max(0,x-K), x in <0,20> \n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_heat_equation<T, std::vector, std::allocator<T>> pde_solver;
    // set up call option parameters:
    auto const &strike = 10;
    auto const &maturity = 1.0;
    auto const &rate = 0.2;
    auto const &sig = 0.25;
    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 300;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(20.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(maturity));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [=](T t, T x) { return 0.5 * sig * sig * x * x; };
    auto b = [=](T t, T x) { return rate * x; };
    auto c = [=](T t, T x) { return -rate; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, b, c);
    // terminal condition:
    auto terminal_condition = [=](T x) { return std::max<T>(0.0, x - strike); };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(terminal_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet_low = [=](T t) { return 0.0; };
    auto const &dirichlet_high = [=](T t) { return (20.0 - strike * std::exp(-rate * (maturity - t))); };
    auto const &boundary_low_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_low);
    auto const &boundary_high_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_high);
    auto const &boundary_pair = std::make_pair(boundary_low_ptr, boundary_high_ptr);
    // grid config:
    auto const &grid_config_hints_ptr = std::make_shared<grid_config_hints_1d<T>>();
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         host_expl_bwd_bc_solver_config_ptr);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    black_scholes_exact<T> bs_exact(0.0, strike, rate, sig, maturity);

    T x{};
    auto const grid_cfg = std::make_shared<grid_config_1d<T>>(discretization_ptr);
    auto const grid_trans_cfg =
        std::make_shared<grid_transform_config_1d<T>>(discretization_ptr, grid_config_hints_ptr);
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        x = grid_1d<T>::transformed_value(grid_trans_cfg, grid_1d<T>::value(grid_cfg, j));
        benchmark = bs_exact.call(x);
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

template <typename T> void testExplBlackScholesEquationDirichletBCSaulyev()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::explicit_pde_schemes_enum;
    using lss_grids::grid_config_1d;
    using lss_grids::grid_config_hints_1d;
    using lss_grids::grid_transform_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_explicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::host_expl_bwd_s_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::explicit_solvers::general_heat_equation;
    using lss_utility::black_scholes_exact;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Black-Scholes Call equation: \n\n";
    std::cout << " Using Saulyev ADE method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = 0.5*sig*sig*x*x*U_xx(x,t) + r*x*U_x(x,t) - "
                 "r*U(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " 0 < x < 20 and 0 < t < 1,\n";
    std::cout << " U(0,t) = 0 and  U(20,t) = 20-K*exp(-r*(1-t)),0 < t < 1 \n\n";
    std::cout << " U(x,T) = max(0,x-K), x in <0,20> \n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_heat_equation<T, std::vector, std::allocator<T>> pde_solver;
    // set up call option parameters:
    auto const &strike = 10;
    auto const &maturity = 1.0;
    auto const &rate = 0.2;
    auto const &sig = 0.25;
    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 300;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(20.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(maturity));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [=](T t, T x) { return 0.5 * sig * sig * x * x; };
    auto b = [=](T t, T x) { return rate * x; };
    auto c = [=](T t, T x) { return -rate; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, b, c);
    // terminal condition:
    auto terminal_condition = [=](T x) { return std::max<T>(0.0, x - strike); };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(terminal_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet_low = [=](T t) { return 0.0; };
    auto const &dirichlet_high = [=](T t) { return (20.0 - strike * std::exp(-rate * (maturity - t))); };
    auto const &boundary_low_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_low);
    auto const &boundary_high_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_high);
    auto const &boundary_pair = std::make_pair(boundary_low_ptr, boundary_high_ptr);
    // grid config:
    auto const &grid_config_hints_ptr = std::make_shared<grid_config_hints_1d<T>>();
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         host_expl_bwd_s_solver_config_ptr);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    black_scholes_exact<T> bs_exact(0.0, strike, rate, sig, maturity);

    T x{};
    auto const grid_cfg = std::make_shared<grid_config_1d<T>>(discretization_ptr);
    auto const grid_trans_cfg =
        std::make_shared<grid_transform_config_1d<T>>(discretization_ptr, grid_config_hints_ptr);
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        x = grid_1d<T>::transformed_value(grid_trans_cfg, grid_1d<T>::value(grid_cfg, j));
        benchmark = bs_exact.call(x);
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

template <typename T> void testExplBlackScholesEquationDirichletBCEuler()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::explicit_pde_schemes_enum;
    using lss_grids::grid_config_1d;
    using lss_grids::grid_config_hints_1d;
    using lss_grids::grid_transform_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_explicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::host_expl_bwd_euler_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::explicit_solvers::general_heat_equation;
    using lss_utility::black_scholes_exact;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Black-Scholes Call equation: \n\n";
    std::cout << " Using Euler method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = 0.5*sig*sig*x*x*U_xx(x,t) + r*x*U_x(x,t) - "
                 "r*U(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " 0 < x < 20 and 0 < t < 1,\n";
    std::cout << " U(0,t) = 0 and  U(20,t) = 20-K*exp(-r*(1-t)),0 < t < 1 \n\n";
    std::cout << " U(x,T) = max(0,x-K), x in <0,20> \n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_heat_equation<T, std::vector, std::allocator<T>> pde_solver;
    // set up call option parameters:
    auto const &strike = 10;
    auto const &maturity = 1.0;
    auto const &rate = 0.2;
    auto const &sig = 0.25;
    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 6000;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(20.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(maturity));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [=](T t, T x) { return 0.5 * sig * sig * x * x; };
    auto b = [=](T t, T x) { return rate * x; };
    auto c = [=](T t, T x) { return -rate; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, b, c);
    // terminal condition:
    auto terminal_condition = [=](T x) { return std::max<T>(0.0, x - strike); };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(terminal_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet_low = [=](T t) { return 0.0; };
    auto const &dirichlet_high = [=](T t) { return (20.0 - strike * std::exp(-rate * (maturity - t))); };
    auto const &boundary_low_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_low);
    auto const &boundary_high_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_high);
    auto const &boundary_pair = std::make_pair(boundary_low_ptr, boundary_high_ptr);
    // grid config:
    auto const &grid_config_hints_ptr = std::make_shared<grid_config_hints_1d<T>>();
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         host_expl_bwd_euler_solver_config_ptr);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    black_scholes_exact<T> bs_exact(0.0, strike, rate, sig, maturity);

    T x{};
    auto const grid_cfg = std::make_shared<grid_config_1d<T>>(discretization_ptr);
    auto const grid_trans_cfg =
        std::make_shared<grid_transform_config_1d<T>>(discretization_ptr, grid_config_hints_ptr);
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        x = grid_1d<T>::transformed_value(grid_trans_cfg, grid_1d<T>::value(grid_cfg, j));
        benchmark = bs_exact.call(x);
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

void testExplBlackScholesEquationDirichletBCADE()
{
    std::cout << "============================================================\n";
    std::cout << "========= Explicit Black-Scholes Equation (Dir BC) =========\n";
    std::cout << "============================================================\n";

    testExplBlackScholesEquationDirichletBCBarakatClark<double>();
    testExplBlackScholesEquationDirichletBCBarakatClark<float>();
    testExplBlackScholesEquationDirichletBCSaulyev<double>();
    testExplBlackScholesEquationDirichletBCSaulyev<float>();
    testExplBlackScholesEquationDirichletBCEuler<double>();
    testExplBlackScholesEquationDirichletBCEuler<float>();

    std::cout << "============================================================\n";
}

#endif //_LSS_BLACK_SCHOLES_EQUATION_T_HPP_
