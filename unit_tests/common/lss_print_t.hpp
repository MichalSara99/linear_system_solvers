#if !defined(_LSS_PRINT_T_HPP_)
#define _LSS_PRINT_T_HPP_

#include <cmath>
#include <sstream>

#include "common/lss_print.hpp"
#include "containers/lss_container_2d.hpp"
#include "ode_solvers/second_degree/lss_general_ode_equation.hpp"
#include "pde_solvers/one_dimensional/heat_type/lss_1d_general_heat_equation.hpp"
#include "pde_solvers/one_dimensional/wave_type/lss_1d_general_svc_wave_equation.hpp"
#include "pde_solvers/two_dimensional/heat_type/lss_2d_general_heston_equation.hpp"

// ODEs

template <typename T> void testImplSimpleODEThomesLUQRPrint()
{
    using lss_boundary::neumann_boundary_1d;
    using lss_boundary::robin_boundary_1d;
    using lss_enumerations::grid_enum;
    using lss_grids::grid_config_1d;
    using lss_grids::grid_config_hints_1d;
    using lss_grids::grid_transform_config_1d;
    using lss_ode_solvers::ode_coefficient_data_config;
    using lss_ode_solvers::ode_data_config;
    using lss_ode_solvers::ode_discretization_config;
    using lss_ode_solvers::ode_nonhom_data_config;
    using lss_ode_solvers::default_ode_solver_configs::dev_cusolver_qr_solver_config_ptr;
    using lss_ode_solvers::implicit_solvers::general_ode_equation;
    using lss_print::print;
    using lss_utility::range;

    std::cout << "=================================\n";
    std::cout << "Solving Boundary-value problem: \n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " u''(t) = -2, \n\n";
    std::cout << " where\n\n";
    std::cout << " t in <0,1>,\n";
    std::cout << " u'(0) - 1 = 0 \n";
    std::cout << " u'(1) + 2*u(1) = 0\n\n";
    std::cout << "Exact solution is:\n\n";
    std::cout << " u(t) = -t*t + t + 0.5\n";
    std::cout << "=================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_ode_equation<T, std::vector, std::allocator<T>> ode_solver;

    // number of space subdivisions:
    std::size_t Sd{100};
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(1.0));
    // discretization config:
    auto const discretization_ptr = std::make_shared<ode_discretization_config<T>>(space_range, Sd);
    // coeffs:
    auto a = [](T x) { return 0.0; };
    auto b = [](T x) { return 0.0; };
    auto const ode_coeffs_data_ptr = std::make_shared<ode_coefficient_data_config<T>>(a, b);
    // nonhom data:
    auto two = [](T x) { return -2.0; };
    auto const ode_nonhom_data_ptr = std::make_shared<ode_nonhom_data_config<T>>(two);
    // ode data config:
    auto const ode_data_ptr = std::make_shared<ode_data_config<T>>(ode_coeffs_data_ptr, ode_nonhom_data_ptr);
    // boundary conditions:
    auto const &neumann = [](T t) { return -1.0; };
    auto const &robin_first = [](T t) { return 2.0; };
    auto const &robin_second = [](T t) { return 0.0; };
    auto const &boundary_low_ptr = std::make_shared<neumann_boundary_1d<T>>(neumann);
    auto const &boundary_high_ptr = std::make_shared<robin_boundary_1d<T>>(robin_first, robin_second);
    auto const &boundary_pair = std::make_pair(boundary_low_ptr, boundary_high_ptr);
    // grid config:
    auto const &alpha_scale = 3.0;
    auto const &grid_config_hints_ptr =
        std::make_shared<grid_config_hints_1d<T>>(0.5, alpha_scale, grid_enum::Nonuniform);
    // initialize ode solver
    ode_solver odesolver(ode_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         dev_cusolver_qr_solver_config_ptr);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    odesolver.solve(solution);

    // print both of these
    std::stringstream ssa;
    ssa << "outputs/simple_ode_approx_" << typeid(T).name() << ".txt";
    std::string file_name_approx{ssa.str()};
    std::ofstream approx(file_name_approx);
    print(discretization_ptr, solution, approx);
    approx.close();
    std::cout << "approx saved to file: " << file_name_approx << "\n";
}

void testImplSimpleODEThomesLUPrint()
{
    std::cout << "============================================================\n";
    std::cout << "=========== Implicit Simple ODE (Thomas LU)  ===============\n";
    std::cout << "============================================================\n";

    testImplSimpleODEThomesLUQRPrint<double>();
    testImplSimpleODEThomesLUQRPrint<float>();

    std::cout << "============================================================\n";
}

// PDEs
template <typename T> void testImplBlackScholesEquationDirichletBCThomasLUSolverEulerPrint()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_grids::grid_1d;
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
    using lss_print::print;
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
    // grid hints:
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
    // get the benchmark:
    auto const grid_cfg = std::make_shared<grid_config_1d<T>>(discretization_ptr);
    auto const grid_trans_cfg =
        std::make_shared<grid_transform_config_1d<T>>(discretization_ptr, grid_config_hints_ptr);
    std::vector<T> benchmark(solution.size());
    T x{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        x = grid_1d<T>::transformed_value(grid_trans_cfg, grid_1d<T>::value(grid_cfg, j));
        benchmark[j] = bs_exact.call(x);
    }
    // print both of these
    std::stringstream ssa;
    std::string name = "ImplBlackScholesEquationDirichletBCThomasLUSolverEuler";
    ssa << "outputs/" << name << "_approx_" << typeid(T).name() << ".txt";
    std::string file_name_approx{ssa.str()};
    std::ofstream approx(file_name_approx);
    print(discretization_ptr, grid_config_hints_ptr, solution, approx);
    approx.close();
    std::cout << "approx saved to file: " << file_name_approx << "\n";
    std::stringstream ssb;
    ssb << "outputs/" << name << "_bench_" << typeid(T).name() << ".txt";
    std::string file_name_bench{ssb.str()};
    std::ofstream bench(file_name_bench);
    print(discretization_ptr, grid_config_hints_ptr, benchmark, bench);
    bench.close();
    std::cout << "bench saved to file: " << file_name_bench << "\n";
}

template <typename T> void testImplBlackScholesEquationDirichletBCThomasLUSolverCrankNicolsonPrint()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_grids::grid_1d;
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
    using lss_print::print;
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
    // grid hints:
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

    auto const grid_cfg = std::make_shared<grid_config_1d<T>>(discretization_ptr);
    auto const grid_trans_cfg =
        std::make_shared<grid_transform_config_1d<T>>(discretization_ptr, grid_config_hints_ptr);
    std::vector<T> benchmark(solution.size());
    T x{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        x = grid_1d<T>::transformed_value(grid_trans_cfg, grid_1d<T>::value(grid_cfg, j));
        benchmark[j] = bs_exact.call(x);
    }
    // print both of these
    std::stringstream ssa;
    std::string name = "ImplBlackScholesEquationDirichletBCThomasLUSolverCrankNicolson";
    ssa << "outputs/" << name << "_approx_" << typeid(T).name() << ".txt";
    std::string file_name_approx{ssa.str()};
    std::ofstream approx(file_name_approx);
    print(discretization_ptr, grid_config_hints_ptr, solution, approx);
    approx.close();
    std::cout << "approx saved to file: " << file_name_approx << "\n";
    std::stringstream ssb;
    ssb << "outputs/" << name << "_bench_" << typeid(T).name() << ".txt";
    std::string file_name_bench{ssb.str()};
    std::ofstream bench(file_name_bench);
    print(discretization_ptr, grid_config_hints_ptr, benchmark, bench);
    bench.close();
    std::cout << "bench saved to file: " << file_name_bench << "\n";
}

void testImplBlackScholesEquationDirichletBCThomasLUSolverPrint()
{
    std::cout << "============================================================\n";
    std::cout << "== Implicit Black-Scholes (Thomas LU) Equation (Dir BC) ====\n";
    std::cout << "============================================================\n";

    testImplBlackScholesEquationDirichletBCThomasLUSolverEulerPrint<double>();
    testImplBlackScholesEquationDirichletBCThomasLUSolverEulerPrint<float>();
    testImplBlackScholesEquationDirichletBCThomasLUSolverCrankNicolsonPrint<double>();
    testImplBlackScholesEquationDirichletBCThomasLUSolverCrankNicolsonPrint<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplBlackScholesEquationDirichletBCThomasLUSolverEulerPrintSurf()
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
    using lss_print::print;
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
    // grid hints:
    auto const &grid_config_hints_ptr = std::make_shared<grid_config_hints_1d<T>>(strike);
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         host_bwd_tlusolver_euler_solver_config_ptr);
    // prepare container for solution:
    container_2d_t solutions(Td, Sd);
    // get the solution:
    pdesolver.solve(solutions);
    // get exact solution:
    black_scholes_exact<T> bs_exact(0.0, strike, rate, sig, maturity);

    // get the benchmark:
    T x{};
    T const k = discretization_ptr->time_step();
    container_2d_t benchmark(Td, Sd);
    auto const grid_cfg = std::make_shared<grid_config_1d<T>>(discretization_ptr);
    auto const grid_trans_cfg =
        std::make_shared<grid_transform_config_1d<T>>(discretization_ptr, grid_config_hints_ptr);
    for (std::size_t t = 0; t < solutions.rows(); ++t)
    {
        for (std::size_t j = 0; j < solutions.columns(); ++j)
        {
            x = grid_1d<T>::transformed_value(grid_trans_cfg, grid_1d<T>::value(grid_cfg, j));
            benchmark(t, j, bs_exact.call(x, maturity - t * k));
        }
    }
    // print both of these
    std::stringstream ssa;
    std::string name = "ImplBlackScholesEquationDirichletBCThomasLUSolverEulerSurf";
    ssa << "outputs/" << name << "_approx_" << typeid(T).name() << ".txt";
    std::string file_name_approx{ssa.str()};
    std::ofstream approx(file_name_approx);
    print(discretization_ptr, grid_config_hints_ptr, solutions, approx);
    approx.close();
    std::cout << "approx saved to file: " << file_name_approx << "\n";
    std::stringstream ssb;
    ssb << "outputs/" << name << "_bench_" << typeid(T).name() << ".txt";
    std::string file_name_bench{ssb.str()};
    std::ofstream bench(file_name_bench);
    print(discretization_ptr, grid_config_hints_ptr, benchmark, bench);
    bench.close();
    std::cout << "bench saved to file: " << file_name_bench << "\n";
}

template <typename T> void testImplBlackScholesEquationDirichletBCThomasLUSolverCrankNicolsonPrintSurf()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_containers::container_2d;
    using lss_enumerations::by_enum;
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
    using lss_pde_solvers::default_heat_solver_configs::host_bwd_tlusolver_cn_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_heat_equation;
    using lss_print::print;
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
    // grid hints:
    auto const alpha_scale = static_cast<T>(3.);
    auto const &grid_config_hints_ptr =
        std::make_shared<grid_config_hints_1d<T>>(strike, alpha_scale, grid_enum::Nonuniform);
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         host_bwd_tlusolver_cn_solver_config_ptr);
    // prepare container for solution:
    container_2d_t solutions(Td, Sd);
    // get the solution:
    pdesolver.solve(solutions);
    // get exact solution:
    black_scholes_exact<T> bs_exact(0.0, strike, rate, sig, maturity);

    T x{};
    T const k = discretization_ptr->time_step();
    auto const grid_cfg = std::make_shared<grid_config_1d<T>>(discretization_ptr);
    auto const grid_trans_cfg =
        std::make_shared<grid_transform_config_1d<T>>(discretization_ptr, grid_config_hints_ptr);
    container_2d_t benchmark(Td, Sd);
    for (std::size_t t = 0; t < solutions.rows(); ++t)
    {
        for (std::size_t j = 0; j < solutions.columns(); ++j)
        {
            x = grid_1d<T>::transformed_value(grid_trans_cfg, grid_1d<T>::value(grid_cfg, j));
            benchmark(t, j, bs_exact.call(x, maturity - t * k));
        }
    }
    // print both of these
    std::string name = "ImplBlackScholesEquationDirichletBCThomasLUSolverSurf";
    std::stringstream ssa;
    ssa << "outputs/" << name << "_approx_" << typeid(T).name() << ".txt";
    std::string file_name_approx{ssa.str()};
    std::ofstream approx(file_name_approx);
    print(discretization_ptr, grid_config_hints_ptr, solutions, approx);
    approx.close();
    std::cout << "approx saved to file: " << file_name_approx << "\n";
    std::stringstream ssb;
    ssb << "outputs/" << name << "_bench_" << typeid(T).name() << ".txt";
    std::string file_name_bench{ssb.str()};
    std::ofstream bench(file_name_bench);
    print(discretization_ptr, grid_config_hints_ptr, benchmark, bench);
    bench.close();
    std::cout << "bench saved to file: " << file_name_bench << "\n";
}

void testImplBlackScholesEquationDirichletBCThomasLUSolverPrintSurf()
{
    std::cout << "============================================================\n";
    std::cout << "== Implicit Black-Scholes (Thomas LU) Equation (Dir BC) ====\n";
    std::cout << "============================================================\n";

    testImplBlackScholesEquationDirichletBCThomasLUSolverEulerPrintSurf<double>();
    testImplBlackScholesEquationDirichletBCThomasLUSolverEulerPrintSurf<float>();
    testImplBlackScholesEquationDirichletBCThomasLUSolverCrankNicolsonPrintSurf<double>();
    testImplBlackScholesEquationDirichletBCThomasLUSolverCrankNicolsonPrintSurf<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplPureHeatEquationDirichletBCCUDASolverDeviceQREulerPrintSurface()
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
    using lss_pde_solvers::default_heat_solver_configs::dev_fwd_cusolver_qr_euler_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_heat_equation;
    using lss_print::print;
    using lss_utility::pi;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Heat equation: \n\n";
    std::cout << " Using CUDA solver with QR (DEVICE) and implicit Euler method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = x, x in <0,1> \n\n";
    std::cout << "============================================================\n";

    // typedef 2D container
    typedef container_2d<by_enum::Row, T, std::vector, std::allocator<T>> container_2d_t;

    // typedef the general_heat_equation
    typedef general_heat_equation<T, std::vector, std::allocator<T>> pde_solver;

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
    auto a = [](T t, T x) { return 1.0; };
    auto other = [](T t, T x) { return 0.0; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, other, other);
    // initial condition:
    auto initial_condition = [](T x) { return x; };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(initial_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // grid hints:
    auto const &grid_config_hints_ptr = std::make_shared<grid_config_hints_1d<T>>();
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         dev_fwd_cusolver_qr_euler_solver_config_ptr);
    // prepare container for solution:
    container_2d_t solutions(Td, Sd);
    // get the solution:
    pdesolver.solve(solutions);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        T const first = static_cast<T>(2.0 / pi<T>());
        T sum{};
        T var1{};
        T var2{};
        for (std::size_t i = 1; i <= n; ++i)
        {
            var1 = std::pow(-1.0, i + 1) * std::exp(-1.0 * (i * pi<T>()) * (i * pi<T>()) * t);
            var2 = std::sin(i * pi<T>() * x) / i;
            sum += (var1 * var2);
        }
        return (first * sum);
    };

    T x{};
    T const k = discretization_ptr->time_step();
    auto const grid_cfg = std::make_shared<grid_config_1d<T>>(discretization_ptr);
    auto const grid_trans_cfg =
        std::make_shared<grid_transform_config_1d<T>>(discretization_ptr, grid_config_hints_ptr);
    container_2d_t benchmark(Td, Sd);
    for (std::size_t t = 0; t < solutions.rows(); ++t)
    {
        for (std::size_t j = 0; j < solutions.columns(); ++j)
        {
            x = grid_1d<T>::transformed_value(grid_trans_cfg, grid_1d<T>::value(grid_cfg, j));
            benchmark(t, j, exact(x, t * k, 20));
        }
    }

    // print both of these
    std::stringstream ssa;
    std::string name = "ImplPureHeatEquationDirichletBCCUDASolverDeviceQREulerSurf";
    ssa << "outputs/" << name << "_approx_" << typeid(T).name() << ".txt";
    std::string file_name_approx{ssa.str()};
    std::ofstream approx(file_name_approx);
    print(discretization_ptr, grid_config_hints_ptr, solutions, approx);
    approx.close();
    std::cout << "approx saved to file: " << file_name_approx << "\n";
    std::stringstream ssb;
    ssb << "outputs/" << name << "_bench_" << typeid(T).name() << ".txt";
    std::string file_name_bench{ssb.str()};
    std::ofstream bench(file_name_bench);
    print(discretization_ptr, grid_config_hints_ptr, benchmark, bench);
    bench.close();
    std::cout << "bench saved to file: " << file_name_bench << "\n";
}

template <typename T> void testImplPureHeatEquationDirichletBCCUDASolverDeviceQRCrankNicolsonPrintSurface()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_containers::container_2d;
    using lss_enumerations::by_enum;
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
    using lss_pde_solvers::default_heat_solver_configs::dev_fwd_cusolver_qr_cn_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_heat_equation;
    using lss_print::print;
    using lss_utility::pi;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Heat equation: \n\n";
    std::cout << " Using CUDA solver with QR (DEVICE) and implicit CN method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = x, x in <0,1> \n\n";
    std::cout << "============================================================\n";

    // typedef 2D container
    typedef container_2d<by_enum::Row, T, std::vector, std::allocator<T>> container_2d_t;
    // typedef the general_heat_equation
    typedef general_heat_equation<T, std::vector, std::allocator<T>> pde_solver;

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
    auto a = [](T t, T x) { return 1.0; };
    auto other = [](T t, T x) { return 0.0; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, other, other);
    // initial condition:
    auto initial_condition = [](T x) { return x; };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(initial_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // grid hints:
    auto const alpha_scale = static_cast<T>(3.);
    auto const &grid_config_hints_ptr =
        std::make_shared<grid_config_hints_1d<T>>(T(.5), alpha_scale, grid_enum::Nonuniform);
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         dev_fwd_cusolver_qr_cn_solver_config_ptr);
    // prepare container for solution:
    container_2d_t solutions(Td, Sd);
    // get the solution:
    pdesolver.solve(solutions);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        T const first = static_cast<T>(2.0 / pi<T>());
        T sum{};
        T var1{};
        T var2{};
        for (std::size_t i = 1; i <= n; ++i)
        {
            var1 = std::pow(-1.0, i + 1) * std::exp(-1.0 * (i * pi<T>()) * (i * pi<T>()) * t);
            var2 = std::sin(i * pi<T>() * x) / i;
            sum += (var1 * var2);
        }
        return (first * sum);
    };

    T x{};
    T const k = discretization_ptr->time_step();
    auto const grid_cfg = std::make_shared<grid_config_1d<T>>(discretization_ptr);
    auto const grid_trans_cfg =
        std::make_shared<grid_transform_config_1d<T>>(discretization_ptr, grid_config_hints_ptr);
    container_2d_t benchmark(Td, Sd);
    for (std::size_t t = 0; t < solutions.rows(); ++t)
    {
        for (std::size_t j = 0; j < solutions.columns(); ++j)
        {
            x = grid_1d<T>::transformed_value(grid_trans_cfg, grid_1d<T>::value(grid_cfg, j));
            benchmark(t, j, exact(x, t * k, 20));
        }
    }

    // print both of these
    std::stringstream ssa;
    std::string name = "ImplPureHeatEquationDirichletBCCUDASolverDeviceQRCrankNicolsonSurf";
    ssa << "outputs/" << name << "_approx_" << typeid(T).name() << ".txt";
    std::string file_name_approx{ssa.str()};
    std::ofstream approx(file_name_approx);
    print(discretization_ptr, grid_config_hints_ptr, solutions, approx);
    approx.close();
    std::cout << "approx saved to file: " << file_name_approx << "\n";
    std::stringstream ssb;
    ssb << "outputs/" << name << "_bench_" << typeid(T).name() << ".txt";
    std::string file_name_bench{ssb.str()};
    std::ofstream bench(file_name_bench);
    print(discretization_ptr, grid_config_hints_ptr, benchmark, bench);
    bench.close();
    std::cout << "bench saved to file: " << file_name_bench << "\n";
}

void testImplPureHeatEquationDirichletBCCUDASolverDeviceQRPrintSurface()
{
    std::cout << "============================================================\n";
    std::cout << " Implicit Pure Heat (CUDA QR DEVICE) Equation (Dirichlet BC)\n";
    std::cout << "============================================================\n";

    testImplPureHeatEquationDirichletBCCUDASolverDeviceQREulerPrintSurface<double>();
    testImplPureHeatEquationDirichletBCCUDASolverDeviceQREulerPrintSurface<float>();
    testImplPureHeatEquationDirichletBCCUDASolverDeviceQRCrankNicolsonPrintSurface<double>();
    testImplPureHeatEquationDirichletBCCUDASolverDeviceQRCrankNicolsonPrintSurface<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testExplPureHeatEquationNeumannNeumannBCEulerPrintSurf()
{
    using lss_boundary::neumann_boundary_1d;
    using lss_containers::container_2d;
    using lss_enumerations::by_enum;
    using lss_enumerations::explicit_pde_schemes_enum;
    using lss_grids::grid_config_1d;
    using lss_grids::grid_config_hints_1d;
    using lss_grids::grid_transform_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_explicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::dev_expl_fwd_euler_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::explicit_solvers::general_heat_equation;
    using lss_print::print;
    using lss_utility::pi;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Heat equation: \n\n";
    std::cout << " Using Euler method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = x, x in <0,1> \n\n";
    std::cout << "============================================================\n";

    // typedef 2D container
    typedef container_2d<by_enum::Row, T, std::vector, std::allocator<T>> container_2d_t;
    // typedef the Implicit1DHeatEquation
    typedef general_heat_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 50;
    // number of time subdivisions:
    std::size_t const Td = 2000;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(1.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(0.3));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [](T t, T x) { return 1.0; };
    auto other = [](T t, T x) { return 0.0; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, other, other);
    // initial condition:
    auto initial_condition = [](T x) { return x; };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(initial_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &neumann = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<neumann_boundary_1d<T>>(neumann);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // grid hints:
    auto const &grid_config_hints_ptr = std::make_shared<grid_config_hints_1d<T>>();
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         dev_expl_fwd_euler_solver_config_ptr);
    // prepare container for solution:
    container_2d_t solutions(Td, Sd);
    // get the solution:
    pdesolver.solve(solutions);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        T const pipi = static_cast<T>(pi<T>() * pi<T>());
        T const first = static_cast<T>(4.0) / pipi;
        T sum{};
        T var0{};
        T var1{};
        T var2{};
        for (std::size_t i = 1; i <= n; ++i)
        {
            var0 = static_cast<T>(2 * i - 1);
            var1 = std::exp(static_cast<T>(-1.0) * pipi * var0 * var0 * t);
            var2 = std::cos(var0 * pi<T>() * x) / (var0 * var0);
            sum += (var1 * var2);
        }
        return (static_cast<T>(0.5) - first * sum);
    };

    T x{};
    T const k = discretization_ptr->time_step();
    auto const grid_cfg = std::make_shared<grid_config_1d<T>>(discretization_ptr);
    auto const grid_trans_cfg =
        std::make_shared<grid_transform_config_1d<T>>(discretization_ptr, grid_config_hints_ptr);
    container_2d_t benchmark(Td, Sd);
    for (std::size_t t = 0; t < solutions.rows(); ++t)
    {
        for (std::size_t j = 0; j < solutions.columns(); ++j)
        {
            x = grid_1d<T>::transformed_value(grid_trans_cfg, grid_1d<T>::value(grid_cfg, j));
            benchmark(t, j, exact(x, t * k, 20));
        }
    }

    // print both of these
    std::stringstream ssa;
    std::string name = "ExplPureHeatEquationNeumannNeumannBCEulerSurf";
    ssa << "outputs/" << name << "_approx_" << typeid(T).name() << ".txt";
    std::string file_name_approx{ssa.str()};
    std::ofstream approx(file_name_approx);
    print(discretization_ptr, grid_config_hints_ptr, solutions, approx);
    approx.close();
    std::cout << "approx saved to file: " << file_name_approx << "\n";
    std::stringstream ssb;
    ssb << "outputs/" << name << "_bench_" << typeid(T).name() << ".txt";
    std::string file_name_bench{ssb.str()};
    std::ofstream bench(file_name_bench);
    print(discretization_ptr, grid_config_hints_ptr, benchmark, bench);
    bench.close();
    std::cout << "bench saved to file: " << file_name_bench << "\n";
}

void testExplPureHeatEquationNeumannBCEulerPrintSurface()
{
    std::cout << "============================================================\n";
    std::cout << "===== Explicit Pure Heat (ADE ) Equation (Non-Dir BC) ======\n";
    std::cout << "============================================================\n";

    testExplPureHeatEquationNeumannNeumannBCEulerPrintSurf<double>();
    testExplPureHeatEquationNeumannNeumannBCEulerPrintSurf<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplAdvDiffEquationDirichletBCThomasLUSolverCNPrintSurf()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_containers::container_2d;
    using lss_enumerations::by_enum;
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
    using lss_pde_solvers::default_heat_solver_configs::host_fwd_tlusolver_cn_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_heat_equation;
    using lss_print::print;
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

    // typedef 2D container
    typedef container_2d<by_enum::Row, T, std::vector, std::allocator<T>> container_2d_t;

    // typedef the Implicit1DHeatEquation
    typedef general_heat_equation<T, std::vector, std::allocator<T>> pde_solver;

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
    auto a = [](T t, T x) { return 1.0; };
    auto b = [](T t, T x) { return -1.0; };
    auto other = [](T t, T x) { return 0.0; };
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
    // grid hints:
    auto const alpha_scale = static_cast<T>(3.);
    auto const &grid_config_hints_ptr =
        std::make_shared<grid_config_hints_1d<T>>(T(0.5), alpha_scale, grid_enum::Nonuniform);
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         host_fwd_tlusolver_cn_solver_config_ptr);
    // prepare container for solution:
    container_2d_t solutions(Td, Sd);
    // get the solution:
    pdesolver.solve(solutions);
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

    T x{};
    T const k = discretization_ptr->time_step();
    auto const grid_cfg = std::make_shared<grid_config_1d<T>>(discretization_ptr);
    auto const grid_trans_cfg =
        std::make_shared<grid_transform_config_1d<T>>(discretization_ptr, grid_config_hints_ptr);
    container_2d_t benchmark(Td, Sd);
    for (std::size_t t = 0; t < solutions.rows(); ++t)
    {
        for (std::size_t j = 0; j < solutions.columns(); ++j)
        {
            x = grid_1d<T>::transformed_value(grid_trans_cfg, grid_1d<T>::value(grid_cfg, j));
            benchmark(t, j, exact(x, t * k, 20));
        }
    }

    // print both of these
    std::stringstream ssa;
    std::string name = "ImplAdvDiffEquationDirichletBCThomasLUSolverCN";
    ssa << "outputs/" << name << "_approx_" << typeid(T).name() << ".txt";
    std::string file_name_approx{ssa.str()};
    std::ofstream approx(file_name_approx);
    print(discretization_ptr, grid_config_hints_ptr, solutions, approx);
    approx.close();
    std::cout << "approx saved to file: " << file_name_approx << "\n";
    std::stringstream ssb;
    ssb << "outputs/" << name << "_bench_" << typeid(T).name() << ".txt";
    std::string file_name_bench{ssb.str()};
    std::ofstream bench(file_name_bench);
    print(discretization_ptr, grid_config_hints_ptr, benchmark, bench);
    bench.close();
    std::cout << "bench saved to file: " << file_name_bench << "\n";
}

void testImplAdvDiffEquationDirichletBCThomasLUSolverPrintSurface()
{
    std::cout << "============================================================\n";
    std::cout << "========= Implicit Advection Equation (Non-Dir BC) =========\n";
    std::cout << "============================================================\n";

    testImplAdvDiffEquationDirichletBCThomasLUSolverCNPrintSurf<double>();
    testImplAdvDiffEquationDirichletBCThomasLUSolverCNPrintSurf<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplPureWaveEquationDirichletBCCUDASolverDeviceQRDetailPrintSurf()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_containers::container_2d;
    using lss_enumerations::by_enum;
    using lss_enumerations::grid_enum;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_grids::grid_config_1d;
    using lss_grids::grid_config_hints_1d;
    using lss_grids::grid_transform_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::wave_coefficient_data_config_1d;
    using lss_pde_solvers::wave_data_config_1d;
    using lss_pde_solvers::wave_implicit_solver_config;
    using lss_pde_solvers::wave_initial_data_config_1d;
    using lss_pde_solvers::default_wave_solver_configs::dev_fwd_cusolver_qr_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_wave_equation;
    using lss_print::print;
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
    // typedef 2D container
    typedef container_2d<by_enum::Row, T, std::vector, std::allocator<T>> container_2d_t;

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
    // grid hints:
    auto const alpha_scale = static_cast<T>(3.);
    auto const &grid_config_hints_ptr =
        std::make_shared<grid_config_hints_1d<T>>(T(0.5), alpha_scale, grid_enum::Nonuniform);
    // initialize pde solver
    pde_solver pdesolver(wave_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         dev_fwd_cusolver_qr_solver_config_ptr);
    // prepare container for solution:
    container_2d_t solutions(Td, Sd);
    // get the solution:
    pdesolver.solve(solutions);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        const T var1 = std::sin(pi<T>() * x);
        const T var2 = std::cos(pi<T>() * t);
        return (var1 * var2);
    };

    T x{};
    T const k = discretization_ptr->time_step();
    auto const grid_cfg = std::make_shared<grid_config_1d<T>>(discretization_ptr);
    auto const grid_trans_cfg =
        std::make_shared<grid_transform_config_1d<T>>(discretization_ptr, grid_config_hints_ptr);
    container_2d_t benchmark(Td, Sd);
    for (std::size_t t = 0; t < solutions.rows(); ++t)
    {
        for (std::size_t j = 0; j < solutions.columns(); ++j)
        {
            x = grid_1d<T>::transformed_value(grid_trans_cfg, grid_1d<T>::value(grid_cfg, j));
            benchmark(t, j, exact(x, t * k, 20));
        }
    }

    // print both of these
    std::stringstream ssa;
    std::string name = "ImplPureWaveEquationDirichletBCCUDASolverDeviceQRSurf";
    ssa << "outputs/" << name << "_approx_" << typeid(T).name() << ".txt";
    std::string file_name_approx{ssa.str()};
    std::ofstream approx(file_name_approx);
    print(discretization_ptr, grid_config_hints_ptr, solutions, approx);
    approx.close();
    std::cout << "approx saved to file: " << file_name_approx << "\n";
    std::stringstream ssb;
    ssb << "outputs/" << name << "_bench_" << typeid(T).name() << ".txt";
    std::string file_name_bench{ssb.str()};
    std::ofstream bench(file_name_bench);
    print(discretization_ptr, grid_config_hints_ptr, benchmark, bench);
    bench.close();
    std::cout << "bench saved to file: " << file_name_bench << "\n";
}

void testImplPureWaveEquationDirichletBCCUDASolverDeviceQRPrintSurf()
{
    std::cout << "============================================================\n";
    std::cout << "========= Implicit Pure Wave Equation (Non-Dir BC) =========\n";
    std::cout << "============================================================\n";

    testImplPureWaveEquationDirichletBCCUDASolverDeviceQRDetailPrintSurf<double>();
    testImplPureWaveEquationDirichletBCCUDASolverDeviceQRDetailPrintSurf<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplWaveEquationDirichletBCSolverHostLUDetailPrintSurf()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_containers::container_2d;
    using lss_enumerations::by_enum;
    using lss_enumerations::grid_enum;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_grids::grid_config_1d;
    using lss_grids::grid_config_hints_1d;
    using lss_grids::grid_transform_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::wave_coefficient_data_config_1d;
    using lss_pde_solvers::wave_data_config_1d;
    using lss_pde_solvers::wave_implicit_solver_config;
    using lss_pde_solvers::wave_initial_data_config_1d;
    using lss_pde_solvers::default_wave_solver_configs::host_fwd_tlusolver_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_wave_equation;
    using lss_print::print;
    using lss_utility::pi;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Wave equation: \n\n";
    std::cout << " Using Thomas LU Solver method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_tt(x,t) = 4*U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = -4*t*t,U(1,t) = -4*t*t + 8*t, t > 0 \n\n";
    std::cout << " U(x,0) = x*(1 - x), x in <0,1> \n\n";
    std::cout << " U_x(x,0) = 0, x in <0,1> \n\n";
    std::cout << "============================================================\n";

    // typedef 2D container
    typedef container_2d<by_enum::Row, T, std::vector, std::allocator<T>> container_2d_t;

    // typedef the general_svc_wave_equation
    typedef general_svc_wave_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(1.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(5.7));
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
    // grid hints:
    auto const alpha_scale = static_cast<T>(3.);
    auto const &grid_config_hints_ptr =
        std::make_shared<grid_config_hints_1d<T>>(T(0.5), alpha_scale, grid_enum::Nonuniform);
    // initialize pde solver
    pde_solver pdesolver(wave_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         host_fwd_tlusolver_solver_config_ptr);
    // prepare container for solution:
    container_2d_t solutions(Td, Sd);
    // get the solution:
    pdesolver.solve(solutions);
    // get exact solution:
    auto exact = [](T x, T t) {
        const T res = x - x * x - 4.0 * t * t + 8.0 * t * x;
        return (res);
    };

    T x{};
    T const k = discretization_ptr->time_step();
    auto const grid_cfg = std::make_shared<grid_config_1d<T>>(discretization_ptr);
    auto const grid_trans_cfg =
        std::make_shared<grid_transform_config_1d<T>>(discretization_ptr, grid_config_hints_ptr);
    container_2d_t benchmark(Td, Sd);
    for (std::size_t t = 0; t < solutions.rows(); ++t)
    {
        for (std::size_t j = 0; j < solutions.columns(); ++j)
        {
            x = grid_1d<T>::transformed_value(grid_trans_cfg, grid_1d<T>::value(grid_cfg, j));
            benchmark(t, j, exact(x, t * k));
        }
    }

    // print both of these
    std::stringstream ssa;
    std::string name = "ImplWaveEquationDirichletBCSolverHostLUSurf";
    ssa << "outputs/" << name << "_approx_" << typeid(T).name() << ".txt";
    std::string file_name_approx{ssa.str()};
    std::ofstream approx(file_name_approx);
    print(discretization_ptr, grid_config_hints_ptr, solutions, approx);
    approx.close();
    std::cout << "approx saved to file: " << file_name_approx << "\n";
    std::stringstream ssb;
    ssb << "outputs/" << name << "_bench_" << typeid(T).name() << ".txt";
    std::string file_name_bench{ssb.str()};
    std::ofstream bench(file_name_bench);
    print(discretization_ptr, grid_config_hints_ptr, benchmark, bench);
    bench.close();
    std::cout << "bench saved to file: " << file_name_bench << "\n";
}

void testImplWaveEquationDirichletBCSolverHostLUPrintSurf()
{
    std::cout << "============================================================\n";
    std::cout << "===== Implicit Wave (Thomas LU) Equation (Dirichlet BC) ====\n";
    std::cout << "============================================================\n";

    testImplWaveEquationDirichletBCSolverHostLUDetailPrintSurf<double>();
    testImplWaveEquationDirichletBCSolverHostLUDetailPrintSurf<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplWaveEquationDirichletBCSolverHostDoubleSweepDetailPrintSurf()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_containers::container_2d;
    using lss_enumerations::by_enum;
    using lss_enumerations::grid_enum;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_grids::grid_config_1d;
    using lss_grids::grid_config_hints_1d;
    using lss_grids::grid_transform_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::wave_coefficient_data_config_1d;
    using lss_pde_solvers::wave_data_config_1d;
    using lss_pde_solvers::wave_implicit_solver_config;
    using lss_pde_solvers::wave_initial_data_config_1d;
    using lss_pde_solvers::default_wave_solver_configs::host_fwd_dssolver_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_wave_equation;
    using lss_print::print;
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

    // typedef 2D container
    typedef container_2d<by_enum::Row, T, std::vector, std::allocator<T>> container_2d_t;

    // typedef the general_svc_wave_equation
    typedef general_svc_wave_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(pi<T>()));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(5.7));
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
    // grid hints:
    auto const alpha_scale = static_cast<T>(3.);
    auto const &grid_config_hints_ptr =
        std::make_shared<grid_config_hints_1d<T>>(T(0.5), alpha_scale, grid_enum::Nonuniform);
    // initialize pde solver
    pde_solver pdesolver(wave_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         host_fwd_dssolver_solver_config_ptr);
    // prepare container for solution:
    container_2d_t solutions(Td, Sd);
    // get the solution:
    pdesolver.solve(solutions);
    // get exact solution:
    auto exact = [](T x, T t) {
        const T exp_half = std::exp(-0.5 * t);
        const T sqrt_3 = std::sqrt(3.0);
        const T arg = 0.5 * sqrt_3;
        const T res = exp_half * std::sin(x) * (std::cos(arg * t) + (sin(arg * t) / sqrt_3));
        return (res);
    };

    T x{};
    T const k = discretization_ptr->time_step();
    auto const grid_cfg = std::make_shared<grid_config_1d<T>>(discretization_ptr);
    auto const grid_trans_cfg =
        std::make_shared<grid_transform_config_1d<T>>(discretization_ptr, grid_config_hints_ptr);
    container_2d_t benchmark(Td, Sd);
    for (std::size_t t = 0; t < solutions.rows(); ++t)
    {
        for (std::size_t j = 0; j < solutions.columns(); ++j)
        {
            x = grid_1d<T>::transformed_value(grid_trans_cfg, grid_1d<T>::value(grid_cfg, j));
            benchmark(t, j, exact(x, t * k));
        }
    }

    // print both of these
    std::stringstream ssa;
    std::string name = "ImplWaveEquationDirichletBCSolverHostDoubleSweepSurf";
    ssa << "outputs/" << name << "_approx_" << typeid(T).name() << ".txt";
    std::string file_name_approx{ssa.str()};
    std::ofstream approx(file_name_approx);
    print(discretization_ptr, grid_config_hints_ptr, solutions, approx);
    approx.close();
    std::cout << "approx saved to file: " << file_name_approx << "\n";
    std::stringstream ssb;
    ssb << "outputs/" << name << "_bench_" << typeid(T).name() << ".txt";
    std::string file_name_bench{ssb.str()};
    std::ofstream bench(file_name_bench);
    print(discretization_ptr, grid_config_hints_ptr, benchmark, bench);
    bench.close();
    std::cout << "bench saved to file: " << file_name_bench << "\n";
}

void testImplWaveEquationDirichletBCSolverHostDoubleSweepPrintSurf()
{
    std::cout << "============================================================\n";
    std::cout << "=== Implicit Wave (Double Sweep) Equation (Dirichlet BC) ===\n";
    std::cout << "============================================================\n";

    testImplWaveEquationDirichletBCSolverHostDoubleSweepDetailPrintSurf<double>();
    testImplWaveEquationDirichletBCSolverHostDoubleSweepDetailPrintSurf<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplPureWaveEquationNeumannBCCUDASolverDeviceQRDetailPrintSurf()
{
    using lss_boundary::neumann_boundary_1d;
    using lss_containers::container_2d;
    using lss_enumerations::by_enum;
    using lss_enumerations::grid_enum;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_grids::grid_config_1d;
    using lss_grids::grid_config_hints_1d;
    using lss_grids::grid_transform_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::wave_coefficient_data_config_1d;
    using lss_pde_solvers::wave_data_config_1d;
    using lss_pde_solvers::wave_implicit_solver_config;
    using lss_pde_solvers::wave_initial_data_config_1d;
    using lss_pde_solvers::default_wave_solver_configs::dev_fwd_cusolver_qr_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_wave_equation;
    using lss_print::print;
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
    std::cout << " U_x(x,0) = 1 - cos(4*x), x in <0,pi> \n\n";
    std::cout << "============================================================\n";

    // typedef 2D container
    typedef container_2d<by_enum::Row, T, std::vector, std::allocator<T>> container_2d_t;
    // typedef the general_svc_wave_equation
    typedef general_svc_wave_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(pi<T>()));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(3.8));
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
    // grid hints:
    auto const alpha_scale = static_cast<T>(3.);
    auto const &grid_config_hints_ptr =
        std::make_shared<grid_config_hints_1d<T>>(T(0.5), alpha_scale, grid_enum::Nonuniform);
    // initialize pde solver
    pde_solver pdesolver(wave_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         dev_fwd_cusolver_qr_solver_config_ptr);
    // prepare container for solution:
    container_2d_t solutions(Td, Sd);
    // get the solution:
    pdesolver.solve(solutions);
    // get exact solution:
    auto exact = [](T x, T t) {
        const T var1 = 3.0 * std::cos(2.0 * t) * std::cos(x);
        const T var2 = -0.125 * std::sin(8.0 * t) * std::cos(4.0 * x);
        return (t + var1 + var2);
    };

    T x{};
    T const k = discretization_ptr->time_step();
    auto const grid_cfg = std::make_shared<grid_config_1d<T>>(discretization_ptr);
    auto const grid_trans_cfg =
        std::make_shared<grid_transform_config_1d<T>>(discretization_ptr, grid_config_hints_ptr);
    container_2d_t benchmark(Td, Sd);
    for (std::size_t t = 0; t < solutions.rows(); ++t)
    {
        for (std::size_t j = 0; j < solutions.columns(); ++j)
        {
            x = grid_1d<T>::transformed_value(grid_trans_cfg, grid_1d<T>::value(grid_cfg, j));
            benchmark(t, j, exact(x, t * k));
        }
    }

    // print both of these
    std::stringstream ssa;
    std::string name = "ImplPureWaveEquationNeumannBCCUDASolverDeviceQRSurf";
    ssa << "outputs/" << name << "_approx_" << typeid(T).name() << ".txt";
    std::string file_name_approx{ssa.str()};
    std::ofstream approx(file_name_approx);
    print(discretization_ptr, grid_config_hints_ptr, solutions, approx);
    approx.close();
    std::cout << "approx saved to file: " << file_name_approx << "\n";
    std::stringstream ssb;
    ssb << "outputs/" << name << "_bench_" << typeid(T).name() << ".txt";
    std::string file_name_bench{ssb.str()};
    std::ofstream bench(file_name_bench);
    print(discretization_ptr, grid_config_hints_ptr, benchmark, bench);
    bench.close();
    std::cout << "bench saved to file: " << file_name_bench << "\n";
}

void testImplPureWaveEquationNeumannBCCUDASolverDeviceQRPrintSurf()
{
    std::cout << "============================================================\n";
    std::cout << "= Implicit Pure Wave (CUDA QR DEVICE) Equation (Neumann BC)=\n";
    std::cout << "============================================================\n";

    testImplPureWaveEquationNeumannBCCUDASolverDeviceQRDetailPrintSurf<double>();
    testImplPureWaveEquationNeumannBCCUDASolverDeviceQRDetailPrintSurf<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testExplPureWaveEquationDirichletBCCUDAHostSolverDetailPrintSurf()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_containers::container_2d;
    using lss_enumerations::by_enum;
    using lss_enumerations::grid_enum;
    using lss_grids::grid_config_1d;
    using lss_grids::grid_config_hints_1d;
    using lss_grids::grid_transform_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::wave_coefficient_data_config_1d;
    using lss_pde_solvers::wave_data_config_1d;
    using lss_pde_solvers::wave_explicit_solver_config;
    using lss_pde_solvers::wave_initial_data_config_1d;
    using lss_pde_solvers::default_wave_solver_configs::host_expl_fwd_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::explicit_solvers::general_svc_wave_equation;
    using lss_print::print;
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

    // typedef 2D container
    typedef container_2d<by_enum::Row, T, std::vector, std::allocator<T>> container_2d_t;

    // typedef the general_svc_wave_equation
    typedef general_svc_wave_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 50;
    // number of time subdivisions:
    std::size_t const Td = 2000;
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
    // grid hints:
    auto const alpha_scale = static_cast<T>(3.);
    auto const &grid_config_hints_ptr =
        std::make_shared<grid_config_hints_1d<T>>(T(0.5), alpha_scale, grid_enum::Nonuniform);
    // initialize pde solver
    pde_solver pdesolver(wave_data_ptr, discretization_ptr, boundary_pair, grid_config_hints_ptr,
                         host_expl_fwd_solver_config_ptr);
    // prepare container for solution:
    container_2d_t solutions(Td, Sd);
    // get the solution:
    pdesolver.solve(solutions);
    // get exact solution:
    auto exact = [](T x, T t) {
        const T var1 = std::sin(pi<T>() * x);
        const T var2 = std::cos(pi<T>() * t);
        return (var1 * var2);
    };

    T x{};
    T const k = discretization_ptr->time_step();
    auto const grid_cfg = std::make_shared<grid_config_1d<T>>(discretization_ptr);
    auto const grid_trans_cfg =
        std::make_shared<grid_transform_config_1d<T>>(discretization_ptr, grid_config_hints_ptr);
    container_2d_t benchmark(Td, Sd);
    for (std::size_t t = 0; t < solutions.rows(); ++t)
    {
        for (std::size_t j = 0; j < solutions.columns(); ++j)
        {
            x = grid_1d<T>::transformed_value(grid_trans_cfg, grid_1d<T>::value(grid_cfg, j));
            benchmark(t, j, exact(x, t * k));
        }
    }

    // print both of these
    std::stringstream ssa;
    std::string name = "ExplPureWaveEquationDirichletBCCUDAHostSolverSurf";
    ssa << "outputs/" << name << "_approx_" << typeid(T).name() << ".txt";
    std::string file_name_approx{ssa.str()};
    std::ofstream approx(file_name_approx);
    print(discretization_ptr, grid_config_hints_ptr, solutions, approx);
    approx.close();
    std::cout << "approx saved to file: " << file_name_approx << "\n";
    std::stringstream ssb;
    ssb << "outputs/" << name << "_bench_" << typeid(T).name() << ".txt";
    std::string file_name_bench{ssb.str()};
    std::ofstream bench(file_name_bench);
    print(discretization_ptr, grid_config_hints_ptr, benchmark, bench);
    bench.close();
    std::cout << "bench saved to file: " << file_name_bench << "\n";
}

void testExplPureWaveEquationDirichletBCCUDAHostSolverPrintSurf()
{
    std::cout << "============================================================\n";
    std::cout << "= Implicit Pure Wave (CUDA DEVICE) Equation (Dirichlet BC) =\n";
    std::cout << "============================================================\n";

    testExplPureWaveEquationDirichletBCCUDAHostSolverDetailPrintSurf<double>();
    testExplPureWaveEquationDirichletBCCUDAHostSolverDetailPrintSurf<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplHestonEquationCUDAQRSolverCrankNicolsonPrintSurf()
{
    using lss_boundary::dirichlet_boundary_2d;
    using lss_boundary::neumann_boundary_2d;
    using lss_containers::container_2d;
    using lss_enumerations::by_enum;
    using lss_enumerations::grid_enum;
    using lss_enumerations::splitting_method_enum;
    using lss_grids::grid_config_1d;
    using lss_grids::grid_config_hints_1d;
    using lss_grids::grid_config_hints_2d;
    using lss_grids::grid_transform_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_2d;
    using lss_pde_solvers::heat_data_config_2d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_2d;
    using lss_pde_solvers::pde_discretization_config_2d;
    using lss_pde_solvers::splitting_method_config;
    using lss_pde_solvers::default_heat_solver_configs::dev_bwd_cusolver_qr_cn_solver_config_ptr;
    using lss_pde_solvers::two_dimensional::implicit_solvers::general_heston_equation;
    using lss_print::print;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Heston Call equation: \n\n";
    std::cout << " Using CUDA QR algo with implicit Crank-Nicolson method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(s,v,t) = 0.5*v*s*s*U_ss(s,v,t) + 0.5*sig*sig*v*U_vv(s,v,t)"
                 " + rho*sig*v*s*U_sv(s,v,t) + r*s*U_s(s,v,t)"
                 " + [k*(theta-v)-lambda*v]*U_v(s,v,t) - r*U(s,v,t)\n\n";
    std::cout << " where\n\n";
    std::cout << " 0 < s < 20, 0 < v < 1, and 0 < t < 1,\n";
    std::cout << " U(0,v,t) = 0 and  U_s(20,v,t) - 1 = 0, 0 < t < 1\n";
    std::cout << " r*s*U_s(s,0,t)+k*theta*U_v(s,0,t)-rU(s,0,t)-U_t(s,0,t) = 0,"
                 "0 < t < 1\n";
    std::cout << " U(s,1,t) = s, 0 < t < 1\n";
    std::cout << " U(s,v,T) = max(0,s - K), s in <0,20> \n\n";
    std::cout << "============================================================\n";

    typedef container_2d<by_enum::Row, T, std::vector, std::allocator<T>> rcontainer_2d_t;
    // typedef the Implicit1DHeatEquation
    typedef general_heston_equation<T, std::vector, std::allocator<T>> pde_solver;
    // set up call option parameters:
    auto const &strike = 10.0;
    auto const &maturity = 1.0;
    auto const &rate = 0.03;
    auto const &sig_sig = 0.3;
    auto const &sig_kappa = 2.0;
    auto const &sig_theta = 0.2;
    auto const &rho = 0.2;
    // number of space subdivisions for spot:
    std::size_t const Sd = 100;
    // number of space subdivision for volatility:
    std::size_t const Vd = 50;
    // number of time subdivisions:
    std::size_t const Td = 150;
    // space Spot range:
    range<T> spacex_range(static_cast<T>(0.0), static_cast<T>(20.0));
    // space Vol range:
    range<T> spacey_range(static_cast<T>(0.0), static_cast<T>(1.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(maturity));
    // discretization config:
    auto const discretization_ptr =
        std::make_shared<pde_discretization_config_2d<T>>(spacex_range, spacey_range, Sd, Vd, time_range, Td);
    // coeffs:
    auto a = [=](T t, T s, T v) { return (0.5 * v * s * s); };
    auto b = [=](T t, T s, T v) { return (0.5 * sig_sig * sig_sig * v); };
    auto c = [=](T t, T s, T v) { return (rho * sig_sig * v * s); };
    auto d = [=](T t, T s, T v) { return (rate * s); };
    auto e = [=](T t, T s, T v) { return (sig_kappa * (sig_theta - v)); };
    auto f = [=](T t, T s, T v) { return (-rate); };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_2d<T>>(a, b, c, d, e, f);
    // terminal condition:
    auto terminal_condition = [=](T s, T v) { return std::max<T>(0.0, s - strike); };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_2d<T>>(terminal_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_2d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // horizontal spot boundary conditions:
    auto const &dirichlet_low = [=](T t, T v) { return 0.0; };
    auto const &neumann_high = [=](T t, T s) { return -1.0; };
    auto const &boundary_low_ptr = std::make_shared<dirichlet_boundary_2d<T>>(dirichlet_low);
    auto const &boundary_high_ptr = std::make_shared<neumann_boundary_2d<T>>(neumann_high);
    auto const &horizontal_boundary_pair = std::make_pair(boundary_low_ptr, boundary_high_ptr);
    // vertical upper vol boundary:
    auto const &dirichlet_high = [=](T t, T s) { return s; };
    auto const &vertical_upper_boundary_ptr = std::make_shared<dirichlet_boundary_2d<T>>(dirichlet_high);
    // splitting method configuration:
    auto const &splitting_config_ptr =
        std::make_shared<splitting_method_config<T>>(splitting_method_enum::DouglasRachford);
    // grid config:
    auto const alpha = static_cast<T>(10. / 3.);
    auto const beta = static_cast<T>(1.0);
    auto const &grid_config_hints_ptr =
        std::make_shared<grid_config_hints_2d<T>>(strike, alpha, beta, grid_enum::Nonuniform);

    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, vertical_upper_boundary_ptr, horizontal_boundary_pair,
                         splitting_config_ptr, grid_config_hints_ptr, dev_bwd_cusolver_qr_cn_solver_config_ptr);
    // prepare container for solution:
    rcontainer_2d_t solution(Sd, Vd, T{});
    // get the solution:
    pdesolver.solve(solution);

    // print approx only:
    std::stringstream ssa;
    std::string name = "ImplHestonEquationCUDAQRSolverCrankNicolson_";
    ssa << "outputs/" << name << typeid(T).name() << ".txt";
    std::string file_name_approx{ssa.str()};
    std::ofstream approx(file_name_approx);
    print(discretization_ptr, grid_config_hints_ptr, solution, approx);
    approx.close();
}

void testImplHestonEquationCUDAQRSolverCrankNicolsonPrint()
{
    std::cout << "============================================================\n";
    std::cout << "=========== Implicit Heston Equation (CUDA DEVICE) =========\n";
    std::cout << "============================================================\n";

    testImplHestonEquationCUDAQRSolverCrankNicolsonPrintSurf<double>();
    testImplHestonEquationCUDAQRSolverCrankNicolsonPrintSurf<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplHestonEquationThomasLUSolverCrankNicolsonPrintSurf()
{
    using lss_boundary::dirichlet_boundary_2d;
    using lss_boundary::neumann_boundary_2d;
    using lss_containers::container_2d;
    using lss_enumerations::by_enum;
    using lss_enumerations::grid_enum;
    using lss_enumerations::splitting_method_enum;
    using lss_grids::grid_config_1d;
    using lss_grids::grid_config_hints_1d;
    using lss_grids::grid_config_hints_2d;
    using lss_grids::grid_transform_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_2d;
    using lss_pde_solvers::heat_data_config_2d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_2d;
    using lss_pde_solvers::pde_discretization_config_2d;
    using lss_pde_solvers::splitting_method_config;
    using lss_pde_solvers::default_heat_solver_configs::host_bwd_tlusolver_cn_solver_config_ptr;
    using lss_pde_solvers::two_dimensional::implicit_solvers::general_heston_equation;
    using lss_print::print;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Heston Call equation: \n\n";
    std::cout << " Using Thomas LU algo with implicit Crank-Nicolson method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(s,v,t) = 0.5*v*s*s*U_ss(s,v,t) + 0.5*sig*sig*v*U_vv(s,v,t)"
                 " + rho*sig*v*s*U_sv(s,v,t) + r*s*U_s(s,v,t)"
                 " + [k*(theta-v)-lambda*v]*U_v(s,v,t) - r*U(s,v,t)\n\n";
    std::cout << " where\n\n";
    std::cout << " 50 < s < 200, 0 < v < 1, and 0 < t < 1,\n";
    std::cout << " U(50,v,t) = 0 and  U_s(200,v,t) - 1 = 0, 0 < t < 1\n";
    std::cout << " r*s*U_s(s,0,t)+k*theta*U_v(s,0,t)-rU(s,0,t)-U_t(s,0,t) = 0,"
                 "0 < t < 1\n";
    std::cout << " U(s,1,t) = s, 0 < t < 1\n";
    std::cout << " U(s,v,T) = max(0,s - K), s in <50,200> \n\n";
    std::cout << "============================================================\n";

    typedef container_2d<by_enum::Row, T, std::vector, std::allocator<T>> rcontainer_2d_t;
    // typedef the Implicit1DHeatEquation
    typedef general_heston_equation<T, std::vector, std::allocator<T>> pde_solver;
    // set up call option parameters:
    auto const &strike = 100.0;
    auto const &maturity = 1.0;
    auto const &rate = 0.03;
    auto const &sig_sig = 0.3;
    auto const &sig_kappa = 2.0;
    auto const &sig_theta = 0.2;
    auto const &rho = 0.2;
    // number of space subdivisions for spot:
    std::size_t const Sd = 150;
    // number of space subdivision for volatility:
    std::size_t const Vd = 50;
    // number of time subdivisions:
    std::size_t const Td = 200;
    // space Spot range:
    range<T> spacex_range(static_cast<T>(30.0), static_cast<T>(200.0));
    // space Vol range:
    range<T> spacey_range(static_cast<T>(0.0), static_cast<T>(1.2));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(maturity));
    // discretization config:
    auto const discretization_ptr =
        std::make_shared<pde_discretization_config_2d<T>>(spacex_range, spacey_range, Sd, Vd, time_range, Td);
    // coeffs:
    auto a = [=](T t, T s, T v) { return (0.5 * v * s * s); };
    auto b = [=](T t, T s, T v) { return (0.5 * sig_sig * sig_sig * v); };
    auto c = [=](T t, T s, T v) { return (rho * sig_sig * v * s); };
    auto d = [=](T t, T s, T v) { return (rate * s); };
    auto e = [=](T t, T s, T v) { return (sig_kappa * (sig_theta - v)); };
    auto f = [=](T t, T s, T v) { return (-rate); };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_2d<T>>(a, b, c, d, e, f);
    // terminal condition:
    auto terminal_condition = [=](T s, T v) { return std::max<T>(0.0, s - strike); };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_2d<T>>(terminal_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_2d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // horizontal spot boundary conditions:
    auto const &dirichlet_low = [=](T t, T v) { return 0.0; };
    auto const &neumann_high = [=](T t, T s) { return -1.0; };
    auto const &boundary_low_ptr = std::make_shared<dirichlet_boundary_2d<T>>(dirichlet_low);
    auto const &boundary_high_ptr = std::make_shared<neumann_boundary_2d<T>>(neumann_high);
    auto const &horizontal_boundary_pair = std::make_pair(boundary_low_ptr, boundary_high_ptr);
    // vertical upper vol boundary:
    auto const &dirichlet_high = [=](T t, T s) { return s; };
    auto const &vertical_upper_boundary_ptr = std::make_shared<dirichlet_boundary_2d<T>>(dirichlet_high);
    // splitting method configuration:
    auto const &splitting_config_ptr =
        std::make_shared<splitting_method_config<T>>(splitting_method_enum::DouglasRachford);
    // grid config:
    auto const alpha_scale = static_cast<T>(3.);
    auto const beta_scale = static_cast<T>(50.);
    auto const &grid_config_hints_ptr =
        std::make_shared<grid_config_hints_2d<T>>(strike, alpha_scale, beta_scale, grid_enum::Nonuniform);

    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, vertical_upper_boundary_ptr, horizontal_boundary_pair,
                         splitting_config_ptr, grid_config_hints_ptr, host_bwd_tlusolver_cn_solver_config_ptr);
    // prepare container for solution:
    rcontainer_2d_t solution(Sd, Vd, T{});
    // get the solution:
    pdesolver.solve(solution);

    // print approx only:
    std::stringstream ssa;
    std::string name = "ImplHestonEquationThomasLUSolverCrankNicolson_";
    ssa << "outputs/" << name << typeid(T).name() << ".txt";
    std::string file_name_approx{ssa.str()};
    std::ofstream approx(file_name_approx);
    print(discretization_ptr, grid_config_hints_ptr, solution, approx);
    approx.close();
}

void testImplHestonEquationThomasLUSolverCrankNicolsonPrint()
{
    std::cout << "============================================================\n";
    std::cout << "======== Implicit Heston Equation (Thomas LU Solver) =======\n";
    std::cout << "============================================================\n";

    testImplHestonEquationThomasLUSolverCrankNicolsonPrintSurf<double>();
    testImplHestonEquationThomasLUSolverCrankNicolsonPrintSurf<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplSABREquationDoubleSweepSolverCrankNicolsonPrintSurf()
{
    using lss_boundary::dirichlet_boundary_2d;
    using lss_boundary::neumann_boundary_2d;
    using lss_containers::container_2d;
    using lss_enumerations::by_enum;
    using lss_enumerations::grid_enum;
    using lss_enumerations::splitting_method_enum;
    using lss_grids::grid_config_hints_2d;
    using lss_pde_solvers::heat_coefficient_data_config_2d;
    using lss_pde_solvers::heat_data_config_2d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_2d;
    using lss_pde_solvers::pde_discretization_config_2d;
    using lss_pde_solvers::splitting_method_config;
    using lss_pde_solvers::default_heat_solver_configs::host_bwd_dssolver_cn_solver_config_ptr;
    using lss_pde_solvers::two_dimensional::implicit_solvers::general_heston_equation;
    using lss_print::print;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value SABR Call equation: \n\n";
    std::cout << " Using Double Sweep algo with implicit Crank-Nicolson method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(s,a,t) = 0.5*a*a*s^(2b)*D^(2*(1-b))*U_ss(s,a,t) "
                 " + 0.5*sig*sig*a*a*U_vv(s,v,t)"
                 " + rho*sig*s^b*D^(1-b)*a*a*U_sv(s,a,t) + r*s*U_s(s,a,t)"
                 " - r*U(s,a,t)\n\n";
    std::cout << " where\n\n";
    std::cout << " 50 < s < 200, 0 < v < 1, and 0 < t < 1,\n";
    std::cout << " U(0,a,t) = 0 and  U_s(200,a,t) - 1 = 0, 0 < t < 1\n";
    std::cout << " r*s*U_s(s,0,t) - rU(s,0,t) - U_t(s,0,t) = 0,"
                 "0 < t < 1\n";
    std::cout << " U(s,1,t) = s, 0 < t < 1\n";
    std::cout << " U(s,a,T) = max(0,s - K), s in <50,200> \n\n";
    std::cout << "============================================================\n";

    typedef container_2d<by_enum::Row, T, std::vector, std::allocator<T>> rcontainer_2d_t;
    // typedef the Implicit1DHeatEquation
    typedef general_heston_equation<T, std::vector, std::allocator<T>> pde_solver;
    // set up call option parameters:
    auto const &strike = 100.0;
    auto const &maturity = 1.0;
    auto const &rate = 0.03;
    auto const &sig_sig = 0.081;
    auto const &rho = 0.6;
    auto const &beta = 0.7;
    // number of space subdivisions for spot:
    std::size_t const Sd = 100;
    // number of space subdivision for volatility:
    std::size_t const Vd = 50;
    // number of time subdivisions:
    std::size_t const Td = 200;
    // space Spot range:
    range<T> spacex_range(static_cast<T>(50.0), static_cast<T>(200.0));
    // space Vol range:
    range<T> spacey_range(static_cast<T>(0.0), static_cast<T>(1.2));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(maturity));
    // discretization config:
    auto const discretization_ptr =
        std::make_shared<pde_discretization_config_2d<T>>(spacex_range, spacey_range, Sd, Vd, time_range, Td);
    // coeffs:
    auto D = [=](T t, T s, T alpha) { return std::exp(-rate * (maturity - t)); };
    auto a = [=](T t, T s, T alpha) {
        return (0.5 * alpha * alpha * std::pow(s, 2.0 * beta) * std::pow(D(t, s, alpha), 2.0 * (1.0 - beta)));
    };
    auto b = [=](T t, T s, T alpha) { return (0.5 * sig_sig * sig_sig * alpha * alpha); };
    auto c = [=](T t, T s, T alpha) {
        return (rho * sig_sig * alpha * alpha * std::pow(s, beta) * std::pow(D(t, s, alpha), (1.0 - beta)));
    };
    auto d = [=](T t, T s, T alpha) { return (rate * s); };
    auto e = [=](T t, T s, T alpha) { return 0.0; };
    auto f = [=](T t, T s, T alpha) { return (-rate); };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_2d<T>>(a, b, c, d, e, f);
    // terminal condition:
    auto terminal_condition = [=](T s, T v) { return std::max<T>(0.0, s - strike); };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_2d<T>>(terminal_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_2d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // horizontal spot boundary conditions:
    auto const &dirichlet_low = [=](T t, T v) { return 0.0; };
    auto const &neumann_high = [=](T t, T s) { return -1.0; };
    auto const &boundary_low_ptr = std::make_shared<dirichlet_boundary_2d<T>>(dirichlet_low);
    auto const &boundary_high_ptr = std::make_shared<neumann_boundary_2d<T>>(neumann_high);
    auto const &horizontal_boundary_pair = std::make_pair(boundary_low_ptr, boundary_high_ptr);
    // vertical upper vol boundary:
    auto const &dirichlet_high = [=](T t, T s) { return s; };
    auto const &vertical_upper_boundary_ptr = std::make_shared<dirichlet_boundary_2d<T>>(dirichlet_high);
    // splitting method configuration:
    auto const &splitting_config_ptr =
        std::make_shared<splitting_method_config<T>>(splitting_method_enum::DouglasRachford);
    // grid config:
    auto const alpha_scale = static_cast<T>(3.);
    auto const beta_scale = static_cast<T>(50.);
    auto const &grid_config_hints_ptr =
        std::make_shared<grid_config_hints_2d<T>>(strike, alpha_scale, beta_scale, grid_enum::Nonuniform);

    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, vertical_upper_boundary_ptr, horizontal_boundary_pair,
                         splitting_config_ptr, grid_config_hints_ptr, host_bwd_dssolver_cn_solver_config_ptr);
    // prepare container for solution:
    rcontainer_2d_t solution(Sd, Vd, T{});
    // get the solution:
    pdesolver.solve(solution);

    // print approx only:
    std::stringstream ssa;
    std::string name = "ImplSABREquationDoubleSweepSolverCrankNicolson_";
    ssa << "outputs/" << name << typeid(T).name() << ".txt";
    std::string file_name_approx{ssa.str()};
    std::ofstream approx(file_name_approx);
    print(discretization_ptr, grid_config_hints_ptr, solution, approx);
    approx.close();
}

void testImplSABREquationDoubleSweepSolverCrankNicolsonPrint()
{
    std::cout << "============================================================\n";
    std::cout << "====== Implicit SABR Equation (Double Sweep Solver) ========\n";
    std::cout << "============================================================\n";

    testImplSABREquationDoubleSweepSolverCrankNicolsonPrintSurf<double>();
    testImplSABREquationDoubleSweepSolverCrankNicolsonPrintSurf<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplHestonEquationThomasLUSolverDouglasRachfordCrankNicolsonPrintSurf()
{
    using lss_boundary::dirichlet_boundary_2d;
    using lss_boundary::neumann_boundary_2d;
    using lss_containers::container_2d;
    using lss_enumerations::by_enum;
    using lss_enumerations::grid_enum;
    using lss_enumerations::splitting_method_enum;
    using lss_grids::grid_config_hints_2d;
    using lss_pde_solvers::heat_coefficient_data_config_2d;
    using lss_pde_solvers::heat_data_config_2d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_2d;
    using lss_pde_solvers::pde_discretization_config_2d;
    using lss_pde_solvers::splitting_method_config;
    using lss_pde_solvers::default_heat_solver_configs::host_bwd_tlusolver_cn_solver_config_ptr;
    using lss_pde_solvers::two_dimensional::implicit_solvers::general_heston_equation;
    using lss_print::print;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Heston Call equation: \n\n";
    std::cout << " Using Thomas LU algo with implicit Crank-Nicolson method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(s,v,t) = 0.5*v*s*s*U_ss(s,v,t) + 0.5*sig*sig*v*U_vv(s,v,t)"
                 " + rho*sig*v*s*U_sv(s,v,t) + r*s*U_s(s,v,t)"
                 " + [k*(theta-v)-lambda*v]*U_v(s,v,t) - r*U(s,v,t)\n\n";
    std::cout << " where\n\n";
    std::cout << " 50 < s < 200, 0 < v < 1, and 0 < t < 1,\n";
    std::cout << " U(50,v,t) = 0 and  U_s(200,v,t) - 1 = 0, 0 < t < 1\n";
    std::cout << " r*s*U_s(s,0,t)+k*theta*U_v(s,0,t)-rU(s,0,t)-U_t(s,0,t) = 0,"
                 "0 < t < 1\n";
    std::cout << " U(s,1,t) = s, 0 < t < 1\n";
    std::cout << " U(s,v,T) = max(0,s - K), s in <50,200> \n\n";
    std::cout << "============================================================\n";

    typedef container_2d<by_enum::Row, T, std::vector, std::allocator<T>> rcontainer_2d_t;
    // typedef the Implicit1DHeatEquation
    typedef general_heston_equation<T, std::vector, std::allocator<T>> pde_solver;
    // set up call option parameters:
    auto const &strike = 100.0;
    auto const &maturity = 1.0;
    auto const &rate = 0.03;
    auto const &sig_sig = 0.3;
    auto const &sig_kappa = 2.0;
    auto const &sig_theta = 0.2;
    auto const &rho = 0.2;
    // number of space subdivisions for spot:
    std::size_t const Sd = 150;
    // number of space subdivision for volatility:
    std::size_t const Vd = 50;
    // number of time subdivisions:
    std::size_t const Td = 200;
    // space Spot range:
    range<T> spacex_range(static_cast<T>(50.0), static_cast<T>(150.0));
    // space Vol range:
    range<T> spacey_range(static_cast<T>(0.0), static_cast<T>(1.2));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(maturity));
    // discretization config:
    auto const discretization_ptr =
        std::make_shared<pde_discretization_config_2d<T>>(spacex_range, spacey_range, Sd, Vd, time_range, Td);
    // coeffs:
    auto a = [=](T t, T s, T v) { return (0.5 * v * s * s); };
    auto b = [=](T t, T s, T v) { return (0.5 * sig_sig * sig_sig * v); };
    auto c = [=](T t, T s, T v) { return (rho * sig_sig * v * s); };
    auto d = [=](T t, T s, T v) { return (rate * s); };
    auto e = [=](T t, T s, T v) { return (sig_kappa * (sig_theta - v)); };
    auto f = [=](T t, T s, T v) { return (-rate); };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_2d<T>>(a, b, c, d, e, f);
    // terminal condition:
    auto terminal_condition = [=](T s, T v) { return std::max<T>(0.0, s - strike); };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_2d<T>>(terminal_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_2d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // horizontal spot boundary conditions:
    auto const &dirichlet_low = [=](T t, T v) { return 0.0; };
    auto const &neumann_high = [=](T t, T s) { return -1.0; };
    auto const &boundary_low_ptr = std::make_shared<dirichlet_boundary_2d<T>>(dirichlet_low);
    auto const &boundary_high_ptr = std::make_shared<neumann_boundary_2d<T>>(neumann_high);
    auto const &horizontal_boundary_pair = std::make_pair(boundary_low_ptr, boundary_high_ptr);
    // vertical upper vol boundary:
    auto const &dirichlet_high = [=](T t, T s) { return s; };
    auto const &vertical_upper_boundary_ptr = std::make_shared<dirichlet_boundary_2d<T>>(dirichlet_high);
    // splitting method configuration:
    auto const &splitting_config_ptr =
        std::make_shared<splitting_method_config<T>>(splitting_method_enum::DouglasRachford);
    // grid config:
    auto const &grid_config_hints_ptr = std::make_shared<grid_config_hints_2d<T>>();

    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, vertical_upper_boundary_ptr, horizontal_boundary_pair,
                         splitting_config_ptr, grid_config_hints_ptr, host_bwd_tlusolver_cn_solver_config_ptr);
    // prepare container for solution:
    rcontainer_2d_t solution(Sd, Vd, T{});
    // get the solution:
    pdesolver.solve(solution);

    // print approx only:
    std::stringstream ssa;
    std::string name = "ImplHestonEquationThomasLUSolverDouglasRachfordCrankNicolson_";
    ssa << "outputs/" << name << typeid(T).name() << ".txt";
    std::string file_name_approx{ssa.str()};
    std::ofstream approx(file_name_approx);
    print(discretization_ptr, grid_config_hints_ptr, solution, approx);
    approx.close();
}

void testImplHestonEquationThomasLUSolverDouglasRachfordCrankNicolsonPrint()
{
    std::cout << "============================================================\n";
    std::cout << "===== Implicit Heston Equation (Thomas LU Solver => DR) ====\n";
    std::cout << "============================================================\n";

    testImplHestonEquationThomasLUSolverDouglasRachfordCrankNicolsonPrintSurf<double>();
    testImplHestonEquationThomasLUSolverDouglasRachfordCrankNicolsonPrintSurf<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplHestonEquationThomasLUSolverCraigSneydCrankNicolsonPrintSurf()
{
    using lss_boundary::dirichlet_boundary_2d;
    using lss_boundary::neumann_boundary_2d;
    using lss_containers::container_2d;
    using lss_enumerations::by_enum;
    using lss_enumerations::grid_enum;
    using lss_enumerations::splitting_method_enum;
    using lss_grids::grid_config_hints_2d;
    using lss_pde_solvers::heat_coefficient_data_config_2d;
    using lss_pde_solvers::heat_data_config_2d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_2d;
    using lss_pde_solvers::pde_discretization_config_2d;
    using lss_pde_solvers::splitting_method_config;
    using lss_pde_solvers::default_heat_solver_configs::host_bwd_tlusolver_o8_solver_config_ptr;
    using lss_pde_solvers::two_dimensional::implicit_solvers::general_heston_equation;
    using lss_print::print;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Heston Call equation: \n\n";
    std::cout << " Using Thomas LU algo with implicit Crank-Nicolson method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(s,v,t) = 0.5*v*s*s*U_ss(s,v,t) + 0.5*sig*sig*v*U_vv(s,v,t)"
                 " + rho*sig*v*s*U_sv(s,v,t) + r*s*U_s(s,v,t)"
                 " + [k*(theta-v)-lambda*v]*U_v(s,v,t) - r*U(s,v,t)\n\n";
    std::cout << " where\n\n";
    std::cout << " 50 < s < 200, 0 < v < 1, and 0 < t < 1,\n";
    std::cout << " U(50,v,t) = 0 and  U_s(200,v,t) - 1 = 0, 0 < t < 1\n";
    std::cout << " r*s*U_s(s,0,t)+k*theta*U_v(s,0,t)-rU(s,0,t)-U_t(s,0,t) = 0,"
                 "0 < t < 1\n";
    std::cout << " U(s,1,t) = s, 0 < t < 1\n";
    std::cout << " U(s,v,T) = max(0,s - K), s in <50,200> \n\n";
    std::cout << "============================================================\n";

    typedef container_2d<by_enum::Row, T, std::vector, std::allocator<T>> rcontainer_2d_t;
    // typedef the Implicit1DHeatEquation
    typedef general_heston_equation<T, std::vector, std::allocator<T>> pde_solver;
    // set up call option parameters:
    auto const &strike = 100.0;
    auto const &maturity = 1.0;
    auto const &rate = 0.03;
    auto const &sig_sig = 0.3;
    auto const &sig_kappa = 2.0;
    auto const &sig_theta = 0.2;
    auto const &rho = 0.2;
    // number of space subdivisions for spot:
    std::size_t const Sd = 150;
    // number of space subdivision for volatility:
    std::size_t const Vd = 50;
    // number of time subdivisions:
    std::size_t const Td = 200;
    // space Spot range:
    range<T> spacex_range(static_cast<T>(50.0), static_cast<T>(200.0));
    // space Vol range:
    range<T> spacey_range(static_cast<T>(0.0), static_cast<T>(1.2));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(maturity));
    // discretization config:
    auto const discretization_ptr =
        std::make_shared<pde_discretization_config_2d<T>>(spacex_range, spacey_range, Sd, Vd, time_range, Td);
    // coeffs:
    auto a = [=](T t, T s, T v) { return (0.5 * v * s * s); };
    auto b = [=](T t, T s, T v) { return (0.5 * sig_sig * sig_sig * v); };
    auto c = [=](T t, T s, T v) { return (rho * sig_sig * v * s); };
    auto d = [=](T t, T s, T v) { return (rate * s); };
    auto e = [=](T t, T s, T v) { return (sig_kappa * (sig_theta - v)); };
    auto f = [=](T t, T s, T v) { return (-rate); };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_2d<T>>(a, b, c, d, e, f);
    // terminal condition:
    auto terminal_condition = [=](T s, T v) { return std::max<T>(0.0, s - strike); };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_2d<T>>(terminal_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_2d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // horizontal spot boundary conditions:
    auto const &dirichlet_low = [=](T t, T v) { return 0.0; };
    auto const &neumann_high = [=](T t, T s) { return -1.0; };
    auto const &boundary_low_ptr = std::make_shared<dirichlet_boundary_2d<T>>(dirichlet_low);
    auto const &boundary_high_ptr = std::make_shared<neumann_boundary_2d<T>>(neumann_high);
    auto const &horizontal_boundary_pair = std::make_pair(boundary_low_ptr, boundary_high_ptr);
    // vertical upper vol boundary:
    auto const &dirichlet_high = [=](T t, T s) { return s; };
    auto const &vertical_upper_boundary_ptr = std::make_shared<dirichlet_boundary_2d<T>>(dirichlet_high);
    // splitting method configuration:
    auto const &splitting_config_ptr =
        std::make_shared<splitting_method_config<T>>(splitting_method_enum::CraigSneyd, T{0.8});
    // grid:
    auto const alpha_scale = static_cast<T>(3.);
    auto const beta_scale = static_cast<T>(50.);
    auto const &grid_config_hints_ptr =
        std::make_shared<grid_config_hints_2d<T>>(strike, alpha_scale, beta_scale, grid_enum::Nonuniform);

    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, vertical_upper_boundary_ptr, horizontal_boundary_pair,
                         splitting_config_ptr, grid_config_hints_ptr, host_bwd_tlusolver_o8_solver_config_ptr);
    // prepare container for solution:
    rcontainer_2d_t solution(Sd, Vd, T{});
    // get the solution:
    pdesolver.solve(solution);

    // print approx only:
    std::stringstream ssa;
    std::string name = "ImplHestonEquationThomasLUSolverCraigSneydCrankNicolson_";
    ssa << "outputs/" << name << typeid(T).name() << ".txt";
    std::string file_name_approx{ssa.str()};
    std::ofstream approx(file_name_approx);
    print(discretization_ptr, grid_config_hints_ptr, solution, approx);
    approx.close();
}

void testImplHestonEquationThomasLUSolverCraigSneydCrankNicolsonPrint()
{
    std::cout << "============================================================\n";
    std::cout << "===== Implicit Heston Equation (Thomas LU Solver => CS) ====\n";
    std::cout << "============================================================\n";

    testImplHestonEquationThomasLUSolverCraigSneydCrankNicolsonPrintSurf<double>();
    testImplHestonEquationThomasLUSolverCraigSneydCrankNicolsonPrintSurf<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplHestonEquationThomasLUSolverModCraigSneydCrankNicolsonPrintSurf()
{
    using lss_boundary::dirichlet_boundary_2d;
    using lss_boundary::neumann_boundary_2d;
    using lss_containers::container_2d;
    using lss_enumerations::by_enum;
    using lss_enumerations::grid_enum;
    using lss_enumerations::splitting_method_enum;
    using lss_grids::grid_config_hints_2d;
    using lss_pde_solvers::heat_coefficient_data_config_2d;
    using lss_pde_solvers::heat_data_config_2d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_2d;
    using lss_pde_solvers::pde_discretization_config_2d;
    using lss_pde_solvers::splitting_method_config;
    using lss_pde_solvers::default_heat_solver_configs::host_bwd_tlusolver_cn_solver_config_ptr;
    using lss_pde_solvers::two_dimensional::implicit_solvers::general_heston_equation;
    using lss_print::print;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Heston Call equation: \n\n";
    std::cout << " Using Thomas LU algo with implicit Crank-Nicolson method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(s,v,t) = 0.5*v*s*s*U_ss(s,v,t) + 0.5*sig*sig*v*U_vv(s,v,t)"
                 " + rho*sig*v*s*U_sv(s,v,t) + r*s*U_s(s,v,t)"
                 " + [k*(theta-v)-lambda*v]*U_v(s,v,t) - r*U(s,v,t)\n\n";
    std::cout << " where\n\n";
    std::cout << " 50 < s < 200, 0 < v < 1, and 0 < t < 1,\n";
    std::cout << " U(50,v,t) = 0 and  U_s(200,v,t) - 1 = 0, 0 < t < 1\n";
    std::cout << " r*s*U_s(s,0,t)+k*theta*U_v(s,0,t)-rU(s,0,t)-U_t(s,0,t) = 0,"
                 "0 < t < 1\n";
    std::cout << " U(s,1,t) = s, 0 < t < 1\n";
    std::cout << " U(s,v,T) = max(0,s - K), s in <50,200> \n\n";
    std::cout << "============================================================\n";

    typedef container_2d<by_enum::Row, T, std::vector, std::allocator<T>> rcontainer_2d_t;
    // typedef the Implicit1DHeatEquation
    typedef general_heston_equation<T, std::vector, std::allocator<T>> pde_solver;
    // set up call option parameters:
    auto const &strike = 100.0;
    auto const &maturity = 1.0;
    auto const &rate = 0.03;
    auto const &sig_sig = 0.3;
    auto const &sig_kappa = 2.0;
    auto const &sig_theta = 0.2;
    auto const &rho = 0.2;
    // number of space subdivisions for spot:
    std::size_t const Sd = 150;
    // number of space subdivision for volatility:
    std::size_t const Vd = 50;
    // number of time subdivisions:
    std::size_t const Td = 200;
    // space Spot range:
    range<T> spacex_range(static_cast<T>(50.0), static_cast<T>(180.0));
    // space Vol range:
    range<T> spacey_range(static_cast<T>(0.0), static_cast<T>(1.2));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(maturity));
    // discretization config:
    auto const discretization_ptr =
        std::make_shared<pde_discretization_config_2d<T>>(spacex_range, spacey_range, Sd, Vd, time_range, Td);
    // coeffs:
    auto a = [=](T t, T s, T v) { return (0.5 * v * s * s); };
    auto b = [=](T t, T s, T v) { return (0.5 * sig_sig * sig_sig * v); };
    auto c = [=](T t, T s, T v) { return (rho * sig_sig * v * s); };
    auto d = [=](T t, T s, T v) { return (rate * s); };
    auto e = [=](T t, T s, T v) { return (sig_kappa * (sig_theta - v)); };
    auto f = [=](T t, T s, T v) { return (-rate); };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_2d<T>>(a, b, c, d, e, f);
    // terminal condition:
    auto terminal_condition = [=](T s, T v) { return std::max<T>(0.0, s - strike); };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_2d<T>>(terminal_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_2d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // horizontal spot boundary conditions:
    auto const &dirichlet_low = [=](T t, T v) { return 0.0; };
    auto const &neumann_high = [=](T t, T s) { return -1.0; };
    auto const &boundary_low_ptr = std::make_shared<dirichlet_boundary_2d<T>>(dirichlet_low);
    auto const &boundary_high_ptr = std::make_shared<neumann_boundary_2d<T>>(neumann_high);
    auto const &horizontal_boundary_pair = std::make_pair(boundary_low_ptr, boundary_high_ptr);
    // vertical upper vol boundary:
    auto const &dirichlet_high = [=](T t, T s) { return s; };
    auto const &vertical_upper_boundary_ptr = std::make_shared<dirichlet_boundary_2d<T>>(dirichlet_high);
    // splitting method configuration:
    auto const &splitting_config_ptr =
        std::make_shared<splitting_method_config<T>>(splitting_method_enum::ModifiedCraigSneyd, T{0.5});
    // grid:
    auto const alpha_scale = static_cast<T>(3.);
    auto const beta_scale = static_cast<T>(50.);
    auto const &grid_config_hints_ptr =
        std::make_shared<grid_config_hints_2d<T>>(strike, alpha_scale, beta_scale, grid_enum::Nonuniform);

    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, vertical_upper_boundary_ptr, horizontal_boundary_pair,
                         splitting_config_ptr, grid_config_hints_ptr, host_bwd_tlusolver_cn_solver_config_ptr);
    // prepare container for solution:
    rcontainer_2d_t solution(Sd, Vd, T{});
    // get the solution:
    pdesolver.solve(solution);

    // print approx only:
    std::stringstream ssa;
    std::string name = "ImplHestonEquationThomasLUSolverModCraigSneydCrankNicolson_";
    ssa << "outputs/" << name << typeid(T).name() << ".txt";
    std::string file_name_approx{ssa.str()};
    std::ofstream approx(file_name_approx);
    print(discretization_ptr, grid_config_hints_ptr, solution, approx);
    approx.close();
}

void testImplHestonEquationThomasLUSolverModCraigSneydCrankNicolsonPrint()
{
    std::cout << "============================================================\n";
    std::cout << "==== Implicit Heston Equation (Thomas LU Solver => MCS) ====\n";
    std::cout << "============================================================\n";

    testImplHestonEquationThomasLUSolverModCraigSneydCrankNicolsonPrintSurf<double>();
    testImplHestonEquationThomasLUSolverModCraigSneydCrankNicolsonPrintSurf<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplHestonEquationThomasLUSolverHundsdorferVerwerCrankNicolsonPrintSurf()
{
    using lss_boundary::dirichlet_boundary_2d;
    using lss_boundary::neumann_boundary_2d;
    using lss_containers::container_2d;
    using lss_enumerations::by_enum;
    using lss_enumerations::grid_enum;
    using lss_enumerations::splitting_method_enum;
    using lss_grids::grid_config_hints_2d;
    using lss_pde_solvers::heat_coefficient_data_config_2d;
    using lss_pde_solvers::heat_data_config_2d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_2d;
    using lss_pde_solvers::pde_discretization_config_2d;
    using lss_pde_solvers::splitting_method_config;
    using lss_pde_solvers::default_heat_solver_configs::host_bwd_tlusolver_cn_solver_config_ptr;
    using lss_pde_solvers::two_dimensional::implicit_solvers::general_heston_equation;
    using lss_print::print;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Heston Call equation: \n\n";
    std::cout << " Using Thomas LU algo with implicit Crank-Nicolson method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(s,v,t) = 0.5*v*s*s*U_ss(s,v,t) + 0.5*sig*sig*v*U_vv(s,v,t)"
                 " + rho*sig*v*s*U_sv(s,v,t) + r*s*U_s(s,v,t)"
                 " + [k*(theta-v)-lambda*v]*U_v(s,v,t) - r*U(s,v,t)\n\n";
    std::cout << " where\n\n";
    std::cout << " 50 < s < 200, 0 < v < 1, and 0 < t < 1,\n";
    std::cout << " U(50,v,t) = 0 and  U_s(200,v,t) - 1 = 0, 0 < t < 1\n";
    std::cout << " r*s*U_s(s,0,t)+k*theta*U_v(s,0,t)-rU(s,0,t)-U_t(s,0,t) = 0,"
                 "0 < t < 1\n";
    std::cout << " U(s,1,t) = s, 0 < t < 1\n";
    std::cout << " U(s,v,T) = max(0,s - K), s in <50,200> \n\n";
    std::cout << "============================================================\n";

    typedef container_2d<by_enum::Row, T, std::vector, std::allocator<T>> rcontainer_2d_t;
    // typedef the Implicit1DHeatEquation
    typedef general_heston_equation<T, std::vector, std::allocator<T>> pde_solver;
    // set up call option parameters:
    auto const &strike = 100.0;
    auto const &maturity = 1.0;
    auto const &rate = 0.03;
    auto const &sig_sig = 0.3;
    auto const &sig_kappa = 2.0;
    auto const &sig_theta = 0.2;
    auto const &rho = 0.2;
    // number of space subdivisions for spot:
    std::size_t const Sd = 150;
    // number of space subdivision for volatility:
    std::size_t const Vd = 50;
    // number of time subdivisions:
    std::size_t const Td = 200;
    // space Spot range:
    range<T> spacex_range(static_cast<T>(50.0), static_cast<T>(200.0));
    // space Vol range:
    range<T> spacey_range(static_cast<T>(0.0), static_cast<T>(1.2));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(maturity));
    // discretization config:
    auto const discretization_ptr =
        std::make_shared<pde_discretization_config_2d<T>>(spacex_range, spacey_range, Sd, Vd, time_range, Td);
    // coeffs:
    auto a = [=](T t, T s, T v) { return (0.5 * v * s * s); };
    auto b = [=](T t, T s, T v) { return (0.5 * sig_sig * sig_sig * v); };
    auto c = [=](T t, T s, T v) { return (rho * sig_sig * v * s); };
    auto d = [=](T t, T s, T v) { return (rate * s); };
    auto e = [=](T t, T s, T v) { return (sig_kappa * (sig_theta - v)); };
    auto f = [=](T t, T s, T v) { return (-rate); };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_2d<T>>(a, b, c, d, e, f);
    // terminal condition:
    auto terminal_condition = [=](T s, T v) { return std::max<T>(0.0, s - strike); };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_2d<T>>(terminal_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_2d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // horizontal spot boundary conditions:
    auto const &dirichlet_low = [=](T t, T v) { return 0.0; };
    auto const &neumann_high = [=](T t, T s) { return -1.0; };
    auto const &boundary_low_ptr = std::make_shared<dirichlet_boundary_2d<T>>(dirichlet_low);
    auto const &boundary_high_ptr = std::make_shared<neumann_boundary_2d<T>>(neumann_high);
    auto const &horizontal_boundary_pair = std::make_pair(boundary_low_ptr, boundary_high_ptr);
    // vertical upper vol boundary:
    auto const &dirichlet_high = [=](T t, T s) { return s; };
    auto const &vertical_upper_boundary_ptr = std::make_shared<dirichlet_boundary_2d<T>>(dirichlet_high);
    // splitting method configuration:
    auto const &splitting_config_ptr =
        std::make_shared<splitting_method_config<T>>(splitting_method_enum::HundsdorferVerwer);
    // grid config:
    auto const alpha_scale = static_cast<T>(3.);
    auto const beta_scale = static_cast<T>(50.);
    auto const &grid_config_hints_ptr =
        std::make_shared<grid_config_hints_2d<T>>(strike, alpha_scale, beta_scale, grid_enum::Nonuniform);

    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, vertical_upper_boundary_ptr, horizontal_boundary_pair,
                         splitting_config_ptr, grid_config_hints_ptr, host_bwd_tlusolver_cn_solver_config_ptr);
    // prepare container for solution:
    rcontainer_2d_t solution(Sd, Vd, T{});
    // get the solution:
    pdesolver.solve(solution);

    // print approx only:
    std::stringstream ssa;
    std::string name = "ImplHestonEquationThomasLUSolverHundsdorferVerwerCrankNicolson_";
    ssa << "outputs/" << name << typeid(T).name() << ".txt";
    std::string file_name_approx{ssa.str()};
    std::ofstream approx(file_name_approx);
    print(discretization_ptr, grid_config_hints_ptr, solution, approx);
    approx.close();
}

void testImplHestonEquationThomasLUSolverHundsdorferVerwerCrankNicolsonPrint()
{
    std::cout << "============================================================\n";
    std::cout << "===== Implicit Heston Equation (Thomas LU Solver => HV) ====\n";
    std::cout << "============================================================\n";

    testImplHestonEquationThomasLUSolverHundsdorferVerwerCrankNicolsonPrintSurf<double>();
    testImplHestonEquationThomasLUSolverHundsdorferVerwerCrankNicolsonPrintSurf<float>();

    std::cout << "============================================================\n";
}

#endif ///_LSS_PRINT_T_HPP_
