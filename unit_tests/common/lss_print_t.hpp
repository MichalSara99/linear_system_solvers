#if !defined(_LSS_PRINT_T_HPP_)
#define _LSS_PRINT_T_HPP_

#include <sstream>

#include "common/lss_print.hpp"
#include "containers/lss_container_2d.hpp"
#include "ode_solvers/second_degree/lss_general_ode_equation.hpp"
#include "pde_solvers/one_dimensional/heat_type/lss_1d_general_svc_heat_equation.hpp"
#include "pde_solvers/one_dimensional/wave_type/lss_1d_general_svc_wave_equation.hpp"

// ODEs

template <typename T> void testImplSimpleODEThomesLUQRPrint()
{
    using lss_boundary::neumann_boundary_1d;
    using lss_boundary::robin_boundary_1d;
    using lss_ode_solvers::dev_cusolver_qr_solver_config_ptr;
    using lss_ode_solvers::ode_coefficient_data_config;
    using lss_ode_solvers::ode_data_config;
    using lss_ode_solvers::ode_discretization_config;
    using lss_ode_solvers::ode_nonhom_data_config;
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
    // initialize ode solver
    ode_solver odesolver(ode_data_ptr, discretization_ptr, boundary_pair, dev_cusolver_qr_solver_config_ptr);
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
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::host_bwd_tlusolver_euler_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
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
    typedef general_svc_heat_equation<T, std::vector, std::allocator<T>> pde_solver;
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
    auto a = [=](T x) { return 0.5 * sig * sig * x * x; };
    auto b = [=](T x) { return rate * x; };
    auto c = [=](T x) { return -rate; };
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
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, host_bwd_tlusolver_euler_solver_config_ptr);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    black_scholes_exact<T> bs_exact(0.0, strike, rate, sig, maturity);
    // get the benchmark:
    T const h = discretization_ptr->space_step();
    std::vector<T> benchmark(solution.size());
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        benchmark[j] = bs_exact.call(j * h);
    }
    // print both of these
    std::stringstream ssa;
    ssa << "outputs/call_approx_" << typeid(T).name() << ".txt";
    std::string file_name_approx{ssa.str()};
    std::ofstream approx(file_name_approx);
    print(discretization_ptr, solution, approx);
    approx.close();
    std::cout << "approx saved to file: " << file_name_approx << "\n";
    std::stringstream ssb;
    ssb << "outputs/call_bench_" << typeid(T).name() << ".txt";
    std::string file_name_bench{ssb.str()};
    std::ofstream bench(file_name_bench);
    print(discretization_ptr, benchmark, bench);
    bench.close();
    std::cout << "bench saved to file: " << file_name_bench << "\n";
}

template <typename T> void testImplBlackScholesEquationDirichletBCThomasLUSolverCrankNicolsonPrint()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::host_bwd_tlusolver_cn_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
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
    typedef general_svc_heat_equation<T, std::vector, std::allocator<T>> pde_solver;
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
    auto a = [=](T x) { return 0.5 * sig * sig * x * x; };
    auto b = [=](T x) { return rate * x; };
    auto c = [=](T x) { return -rate; };
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
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, host_bwd_tlusolver_cn_solver_config_ptr);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    black_scholes_exact<T> bs_exact(0.0, strike, rate, sig, maturity);

    T const h = discretization_ptr->space_step();
    std::vector<T> benchmark(solution.size());
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        benchmark[j] = bs_exact.call(j * h);
    }
    // print both of these
    std::stringstream ssa;
    ssa << "outputs/call_approx_" << typeid(T).name() << ".txt";
    std::string file_name_approx{ssa.str()};
    std::ofstream approx(file_name_approx);
    print(discretization_ptr, solution, approx);
    approx.close();
    std::cout << "approx saved to file: " << file_name_approx << "\n";
    std::stringstream ssb;
    ssb << "outputs/call_bench_" << typeid(T).name() << ".txt";
    std::string file_name_bench{ssb.str()};
    std::ofstream bench(file_name_bench);
    print(discretization_ptr, benchmark, bench);
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
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::host_bwd_tlusolver_euler_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
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
    typedef container_2d<T, std::vector, std::allocator<T>> container_2d_t;

    // typedef the Implicit1DHeatEquation
    typedef general_svc_heat_equation<T, std::vector, std::allocator<T>> pde_solver;
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
    auto a = [=](T x) { return 0.5 * sig * sig * x * x; };
    auto b = [=](T x) { return rate * x; };
    auto c = [=](T x) { return -rate; };
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
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, host_bwd_tlusolver_euler_solver_config_ptr);
    // prepare container for solution:
    container_2d_t solutions(Td, Sd);
    // get the solution:
    pdesolver.solve(solutions);
    // get exact solution:
    black_scholes_exact<T> bs_exact(0.0, strike, rate, sig, maturity);

    // get the benchmark:
    T const h = discretization_ptr->space_step();
    T const k = discretization_ptr->time_step();
    container_2d_t benchmark(Td, Sd);
    for (std::size_t t = 0; t < solutions.rows(); ++t)
    {
        for (std::size_t j = 0; j < solutions.columns(); ++j)
        {
            benchmark(t, j, bs_exact.call(j * h, maturity - t * k));
        }
    }
    // print both of these
    std::stringstream ssa;
    ssa << "outputs/call_approx_surf_" << typeid(T).name() << ".txt";
    std::string file_name_approx{ssa.str()};
    std::ofstream approx(file_name_approx);
    print(discretization_ptr, solutions, approx);
    approx.close();
    std::cout << "approx saved to file: " << file_name_approx << "\n";
    std::stringstream ssb;
    ssb << "outputs/call_bench_surf_" << typeid(T).name() << ".txt";
    std::string file_name_bench{ssb.str()};
    std::ofstream bench(file_name_bench);
    print(discretization_ptr, benchmark, bench);
    bench.close();
    std::cout << "bench saved to file: " << file_name_bench << "\n";
}

template <typename T> void testImplBlackScholesEquationDirichletBCThomasLUSolverCrankNicolsonPrintSurf()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_containers::container_2d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::host_bwd_tlusolver_cn_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
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
    typedef container_2d<T, std::vector, std::allocator<T>> container_2d_t;

    // typedef the Implicit1DHeatEquation
    typedef general_svc_heat_equation<T, std::vector, std::allocator<T>> pde_solver;
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
    auto a = [=](T x) { return 0.5 * sig * sig * x * x; };
    auto b = [=](T x) { return rate * x; };
    auto c = [=](T x) { return -rate; };
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
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, host_bwd_tlusolver_cn_solver_config_ptr);
    // prepare container for solution:
    container_2d_t solutions(Td, Sd);
    // get the solution:
    pdesolver.solve(solutions);
    // get exact solution:
    black_scholes_exact<T> bs_exact(0.0, strike, rate, sig, maturity);

    T const h = discretization_ptr->space_step();
    T const k = discretization_ptr->time_step();
    container_2d_t benchmark(Td, Sd);
    for (std::size_t t = 0; t < solutions.rows(); ++t)
    {
        for (std::size_t j = 0; j < solutions.columns(); ++j)
        {
            benchmark(t, j, bs_exact.call(j * h, maturity - t * k));
        }
    }
    // print both of these
    std::stringstream ssa;
    ssa << "outputs/call_approx_surf_" << typeid(T).name() << ".txt";
    std::string file_name_approx{ssa.str()};
    std::ofstream approx(file_name_approx);
    print(discretization_ptr, solutions, approx);
    approx.close();
    std::cout << "approx saved to file: " << file_name_approx << "\n";
    std::stringstream ssb;
    ssb << "outputs/call_bench_surf_" << typeid(T).name() << ".txt";
    std::string file_name_bench{ssb.str()};
    std::ofstream bench(file_name_bench);
    print(discretization_ptr, benchmark, bench);
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
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::dev_fwd_cusolver_qr_euler_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
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
    typedef container_2d<T, std::vector, std::allocator<T>> container_2d_t;

    // typedef the general_svc_heat_equation
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
    auto other = [](T x) { return 0.0; };
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
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, dev_fwd_cusolver_qr_euler_solver_config_ptr);
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

    T const h = discretization_ptr->space_step();
    T const k = discretization_ptr->time_step();
    container_2d_t benchmark(Td, Sd);
    for (std::size_t t = 0; t < solutions.rows(); ++t)
    {
        for (std::size_t j = 0; j < solutions.columns(); ++j)
        {
            benchmark(t, j, exact(j * h, t * k, 20));
        }
    }

    // print both of these
    std::stringstream ssa;
    ssa << "outputs/pheat_euler_approx_surf_" << typeid(T).name() << ".txt";
    std::string file_name_approx{ssa.str()};
    std::ofstream approx(file_name_approx);
    print(discretization_ptr, solutions, approx);
    approx.close();
    std::cout << "approx saved to file: " << file_name_approx << "\n";
    std::stringstream ssb;
    ssb << "outputs/pheat_euler_bench_surf_" << typeid(T).name() << ".txt";
    std::string file_name_bench{ssb.str()};
    std::ofstream bench(file_name_bench);
    print(discretization_ptr, benchmark, bench);
    bench.close();
    std::cout << "bench saved to file: " << file_name_bench << "\n";
}

template <typename T> void testImplPureHeatEquationDirichletBCCUDASolverDeviceQRCrankNicolsonPrintSurface()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_containers::container_2d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::dev_fwd_cusolver_qr_cn_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
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
    typedef container_2d<T, std::vector, std::allocator<T>> container_2d_t;
    // typedef the general_svc_heat_equation
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
    auto other = [](T x) { return 0.0; };
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
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, dev_fwd_cusolver_qr_cn_solver_config_ptr);
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

    T const h = discretization_ptr->space_step();
    T const k = discretization_ptr->time_step();
    container_2d_t benchmark(Td, Sd);
    for (std::size_t t = 0; t < solutions.rows(); ++t)
    {
        for (std::size_t j = 0; j < solutions.columns(); ++j)
        {
            benchmark(t, j, exact(j * h, t * k, 20));
        }
    }

    // print both of these
    std::stringstream ssa;
    ssa << "outputs/pheat_cn_approx_surf_" << typeid(T).name() << ".txt";
    std::string file_name_approx{ssa.str()};
    std::ofstream approx(file_name_approx);
    print(discretization_ptr, solutions, approx);
    approx.close();
    std::cout << "approx saved to file: " << file_name_approx << "\n";
    std::stringstream ssb;
    ssb << "outputs/pheat_cn_bench_surf_" << typeid(T).name() << ".txt";
    std::string file_name_bench{ssb.str()};
    std::ofstream bench(file_name_bench);
    print(discretization_ptr, benchmark, bench);
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
    using lss_enumerations::explicit_pde_schemes_enum;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_explicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::dev_expl_fwd_euler_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::explicit_solvers::general_svc_heat_equation;
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
    typedef container_2d<T, std::vector, std::allocator<T>> container_2d_t;
    // typedef the Implicit1DHeatEquation
    typedef general_svc_heat_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 6000;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(1.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(0.3));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [](T x) { return 1.0; };
    auto other = [](T x) { return 0.0; };
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
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, dev_expl_fwd_euler_solver_config_ptr);
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

    T const h = discretization_ptr->space_step();
    T const k = discretization_ptr->time_step();
    container_2d_t benchmark(Td, Sd);
    for (std::size_t t = 0; t < solutions.rows(); ++t)
    {
        for (std::size_t j = 0; j < solutions.columns(); ++j)
        {
            benchmark(t, j, exact(j * h, t * k, 20));
        }
    }

    // print both of these
    std::stringstream ssa;
    ssa << "outputs/pheatneu_e_approx_surf_" << typeid(T).name() << ".txt";
    std::string file_name_approx{ssa.str()};
    std::ofstream approx(file_name_approx);
    print(discretization_ptr, solutions, approx);
    approx.close();
    std::cout << "approx saved to file: " << file_name_approx << "\n";
    std::stringstream ssb;
    ssb << "outputs/pheatneu_e_bench_surf_" << typeid(T).name() << ".txt";
    std::string file_name_bench{ssb.str()};
    std::ofstream bench(file_name_bench);
    print(discretization_ptr, benchmark, bench);
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
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::default_heat_solver_configs::host_fwd_tlusolver_cn_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
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
    typedef container_2d<T, std::vector, std::allocator<T>> container_2d_t;

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
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, host_fwd_tlusolver_cn_solver_config_ptr);
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

    T const h = discretization_ptr->space_step();
    T const k = discretization_ptr->time_step();
    container_2d_t benchmark(Td, Sd);
    for (std::size_t t = 0; t < solutions.rows(); ++t)
    {
        for (std::size_t j = 0; j < solutions.columns(); ++j)
        {
            benchmark(t, j, exact(j * h, t * k, 20));
        }
    }

    // print both of these
    std::stringstream ssa;
    ssa << "outputs/advection_cn_approx_surf_" << typeid(T).name() << ".txt";
    std::string file_name_approx{ssa.str()};
    std::ofstream approx(file_name_approx);
    print(discretization_ptr, solutions, approx);
    approx.close();
    std::cout << "approx saved to file: " << file_name_approx << "\n";
    std::stringstream ssb;
    ssb << "outputs/advection_cn_bench_surf_" << typeid(T).name() << ".txt";
    std::string file_name_bench{ssb.str()};
    std::ofstream bench(file_name_bench);
    print(discretization_ptr, benchmark, bench);
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
    using lss_enumerations::implicit_pde_schemes_enum;
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
    typedef container_2d<T, std::vector, std::allocator<T>> container_2d_t;

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
    container_2d_t solutions(Sd, Td);
    // get the solution:
    pdesolver.solve(solutions);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        const T var1 = std::sin(pi<T>() * x);
        const T var2 = std::cos(pi<T>() * t);
        return (var1 * var2);
    };

    T const h = discretization_ptr->space_step();
    T const k = discretization_ptr->time_step();
    container_2d_t benchmark(Td, Sd);
    for (std::size_t t = 0; t < solutions.rows(); ++t)
    {
        for (std::size_t j = 0; j < solutions.columns(); ++j)
        {
            benchmark(t, j, exact(j * h, t * k, 20));
        }
    }

    // print both of these
    std::stringstream ssa;
    ssa << "outputs/pure_wave_approx_surf_" << typeid(T).name() << ".txt";
    std::string file_name_approx{ssa.str()};
    std::ofstream approx(file_name_approx);
    print(discretization_ptr, solutions, approx);
    approx.close();
    std::cout << "approx saved to file: " << file_name_approx << "\n";
    std::stringstream ssb;
    ssb << "outputs/pure_wave_bench_surf_" << typeid(T).name() << ".txt";
    std::string file_name_bench{ssb.str()};
    std::ofstream bench(file_name_bench);
    print(discretization_ptr, benchmark, bench);
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
    using lss_enumerations::implicit_pde_schemes_enum;
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
    typedef container_2d<T, std::vector, std::allocator<T>> container_2d_t;

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
    // initialize pde solver
    pde_solver pdesolver(wave_data_ptr, discretization_ptr, boundary_pair, host_fwd_tlusolver_solver_config_ptr);
    // prepare container for solution:
    container_2d_t solutions(Sd, Td);
    // get the solution:
    pdesolver.solve(solutions);
    // get exact solution:
    auto exact = [](T x, T t) {
        const T res = x - x * x - 4.0 * t * t + 8.0 * t * x;
        return (res);
    };

    T const h = discretization_ptr->space_step();
    T const k = discretization_ptr->time_step();
    container_2d_t benchmark(Td, Sd);
    for (std::size_t t = 0; t < solutions.rows(); ++t)
    {
        for (std::size_t j = 0; j < solutions.columns(); ++j)
        {
            benchmark(t, j, exact(j * h, t * k));
        }
    }

    // print both of these
    std::stringstream ssa;
    ssa << "outputs/wave_approx_surf_" << typeid(T).name() << ".txt";
    std::string file_name_approx{ssa.str()};
    std::ofstream approx(file_name_approx);
    print(discretization_ptr, solutions, approx);
    approx.close();
    std::cout << "approx saved to file: " << file_name_approx << "\n";
    std::stringstream ssb;
    ssb << "outputs/wave_bench_surf_" << typeid(T).name() << ".txt";
    std::string file_name_bench{ssb.str()};
    std::ofstream bench(file_name_bench);
    print(discretization_ptr, benchmark, bench);
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
    using lss_enumerations::implicit_pde_schemes_enum;
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
    typedef container_2d<T, std::vector, std::allocator<T>> container_2d_t;

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
    // initialize pde solver
    pde_solver pdesolver(wave_data_ptr, discretization_ptr, boundary_pair, host_fwd_dssolver_solver_config_ptr);
    // prepare container for solution:
    container_2d_t solutions(Sd, Td);
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

    T const h = discretization_ptr->space_step();
    T const k = discretization_ptr->time_step();
    container_2d_t benchmark(Td, Sd);
    for (std::size_t t = 0; t < solutions.rows(); ++t)
    {
        for (std::size_t j = 0; j < solutions.columns(); ++j)
        {
            benchmark(t, j, exact(j * h, t * k));
        }
    }

    // print both of these
    std::stringstream ssa;
    ssa << "outputs/damped_wave_approx_surf_" << typeid(T).name() << ".txt";
    std::string file_name_approx{ssa.str()};
    std::ofstream approx(file_name_approx);
    print(discretization_ptr, solutions, approx);
    approx.close();
    std::cout << "approx saved to file: " << file_name_approx << "\n";
    std::stringstream ssb;
    ssb << "outputs/damped_wave_bench_surf_" << typeid(T).name() << ".txt";
    std::string file_name_bench{ssb.str()};
    std::ofstream bench(file_name_bench);
    print(discretization_ptr, benchmark, bench);
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

#endif ///_LSS_PRINT_T_HPP_
