#if !defined(_LSS_HESTON_EQUATION_T_HPP_)
#define _LSS_HESTON_EQUATION_T_HPP_

#include "pde_solvers/two_dimensional/heat_type/lss_2d_general_svc_heston_equation.hpp"
#include "sparse_solvers/tridiagonal/double_sweep_solver/lss_double_sweep_solver.hpp"
#include <map>

#define PI 3.14159265359

// ///////////////////////////////////////////////////////////////////////////
//							HESTON PROBLEMS
// ///////////////////////////////////////////////////////////////////////////

// ===========================================================================
// ========================== IMPLICIT SOLVERS ===============================
// ===========================================================================

// Dirichlet boundaries:

template <typename T> void testImplHestonEquationCUDAQRSolverCrankNicolson()
{
    using lss_boundary::dirichlet_boundary_2d;
    using lss_boundary::neumann_boundary_2d;
    using lss_containers::container_2d;
    using lss_enumerations::by_enum;
    using lss_enumerations::splitting_method_enum;
    using lss_pde_solvers::heat_coefficient_data_config_2d;
    using lss_pde_solvers::heat_data_config_2d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_2d;
    using lss_pde_solvers::pde_discretization_config_2d;
    using lss_pde_solvers::splitting_method_config;
    using lss_pde_solvers::default_heat_solver_configs::dev_bwd_cusolver_qr_cn_solver_config_ptr;
    using lss_pde_solvers::two_dimensional::implicit_solvers::general_svc_heston_equation;
    using lss_print::print;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Heston Call equation: \n\n";
    std::cout << " Using Thomas LU algo with implicit Euler method\n\n";
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
    typedef general_svc_heston_equation<T, std::vector, std::allocator<T>> pde_solver;
    // set up call option parameters:
    auto const &strike = 10.0;
    auto const &maturity = 1.0;
    auto const &rate = 0.03;
    auto const &sig_sig = 0.3;
    auto const &sig_kappa = 2.0;
    auto const &sig_theta = 0.2;
    auto const &rho = 0.2;
    // number of space subdivisions for spot:
    std::size_t const Sd = 50;
    // number of space subdivision for volatility:
    std::size_t const Vd = 50;
    // number of time subdivisions:
    std::size_t const Td = 100;
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
    auto a = [=](T s, T v) { return (0.5 * v * s * s); };
    auto b = [=](T s, T v) { return (0.5 * sig_sig * sig_sig * v); };
    auto c = [=](T s, T v) { return (rho * sig_sig * v * s); };
    auto d = [=](T s, T v) { return (rate * s); };
    auto e = [=](T s, T v) { return (sig_kappa * (sig_theta - v)); };
    auto f = [=](T s, T v) { return (-rate); };
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

    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, vertical_upper_boundary_ptr, horizontal_boundary_pair,
                         splitting_config_ptr, dev_bwd_cusolver_qr_cn_solver_config_ptr);
    // prepare container for solution:
    rcontainer_2d_t solution(Sd, Vd, T{});
    // get the solution:
    pdesolver.solve(solution);

    print(discretization_ptr, solution);
}

void testImplHestonEquationCUDAQRSolver()
{
    std::cout << "============================================================\n";
    std::cout << "============ Implicit Hetson Equation (Dir BC) =============\n";
    std::cout << "============================================================\n";

    testImplHestonEquationCUDAQRSolverCrankNicolson<double>();
    testImplHestonEquationCUDAQRSolverCrankNicolson<float>();

    std::cout << "============================================================\n";
}

template <typename T> void completeTest()
{
    using lss_boundary::boundary_2d_pair;
    using lss_boundary::dirichlet_boundary_2d;
    using lss_boundary::neumann_boundary_2d;
    using lss_containers::container_2d;
    using lss_double_sweep_solver::double_sweep_solver;
    using lss_enumerations::by_enum;
    using lss_pde_solvers::heat_coefficient_data_config_2d;
    using lss_pde_solvers::heat_data_config_2d;
    using lss_pde_solvers::heat_implicit_solver_config;
    using lss_pde_solvers::heat_initial_data_config_2d;
    using lss_pde_solvers::pde_discretization_config_2d;
    using lss_pde_solvers::default_heat_solver_configs::dev_bwd_cusolver_qr_cn_solver_config_ptr;
    using lss_pde_solvers::two_dimensional::implicit_solvers::general_svc_heston_equation;
    using lss_print::print;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Heston Call equation: \n\n";
    std::cout << " Using Thomas LU algo with implicit Euler method\n\n";
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

    typedef discretization<dimension_enum::Two, T, std::vector, std::allocator<T>> d_2d;
    typedef discretization<dimension_enum::One, T, std::vector, std::allocator<T>> d_1d;
    typedef container_2d<by_enum::Row, T, std::vector, std::allocator<T>> rcontainer_2d_t;
    typedef container_2d<by_enum::Column, T, std::vector, std::allocator<T>> ccontainer_2d_t;
    typedef std::vector<T, std::allocator<T>> container_t;
    typedef double_sweep_solver<T, std::vector, std::allocator<T>> ds_solver;

    // typedef the Implicit1DHeatEquation
    typedef general_svc_heston_equation<T, std::vector, std::allocator<T>> pde_solver;
    // set up call option parameters:
    auto const &strike = 10.0;
    auto const &maturity = 1.0;
    auto const &rate = 0.03;
    auto const &sig_sig = 0.3;
    auto const &sig_kappa = 2.0;
    auto const &sig_theta = 0.2;
    auto const &sig_rho = 0.1;
    // number of space subdivisions for spot:
    std::size_t const Sd = 40;
    // number of space subdivision for volatility:
    std::size_t const Vd = 40;
    // number of time subdivisions:
    std::size_t const Td = 100;
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
    auto a = [=](T s, T v) { return (0.5 * v * s * s); };
    auto b = [=](T s, T v) { return (0.5 * sig_sig * sig_sig * v); };
    auto c = [=](T s, T v) { return (sig_rho * sig_sig * v * s); };
    auto d = [=](T s, T v) { return (rate * s); };
    auto e = [=](T s, T v) { return (sig_kappa * (sig_theta - v)); };
    auto f = [=](T s, T v) { return (-rate); };
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

    // computation:

    // get space ranges:
    const auto &spaces = discretization_ptr->space_range();
    // get time range:
    const auto time_rng = discretization_ptr->time_range();
    // across X:
    const auto space_x = spaces.first;
    // across Y:
    const auto space_y = spaces.second;
    // get space steps:
    const auto &hs = discretization_ptr->space_step();
    // across X:
    const T h_1 = hs.first;
    // across Y:
    const T h_2 = hs.second;
    // get space steps:
    const auto k = discretization_ptr->time_step();
    // size of spaces discretization:
    const auto &space_sizes = discretization_ptr->number_of_space_points();
    // size of time discretization:
    const auto &time_size = discretization_ptr->number_of_time_points();
    const std::size_t space_size_x = std::get<0>(space_sizes);
    const std::size_t space_size_y = std::get<1>(space_sizes);

    // create container to carry previous solution:
    rcontainer_2d_t prev_sol(space_size_x, space_size_y, T{});
    // create container to carry next solution:
    rcontainer_2d_t next_sol(space_size_x, space_size_y, T{});
    // discretize initial condition
    d_2d::of_function(space_x.lower(), space_y.lower(), h_1, h_2, heat_data_ptr->initial_condition(), prev_sol);

    // calculate scheme coefficients:
    const T half = static_cast<T>(0.5);
    const T quarter = static_cast<T>(0.25);
    const T one = static_cast<T>(1.0);
    const T two = static_cast<T>(2.0);
    const T three = static_cast<T>(3.0);
    const T four = static_cast<T>(4.0);

    auto const alpha = k / (h_1 * h_1);
    auto const beta = k / (h_2 * h_2);
    auto const gamma = quarter * k / (h_1 * h_2);
    auto const delta = half * k / h_1;
    auto const ni = half * k / h_2;
    auto const rho = k;

    // save coefficients locally:
    // auto a = heat_data_ptr->a_coefficient();
    // auto b = heat_data_ptr->b_coefficient();
    // auto c = heat_data_ptr->c_coefficient();
    // auto d = heat_data_ptr->d_coefficient();
    // auto e = heat_data_ptr->e_coefficient();
    // auto f = heat_data_ptr->f_coefficient();

    auto M = [=](T x, T y) { return (alpha * a(x, y) - delta * d(x, y)); };
    auto M_tilde = [=](T x, T y) { return (beta * b(x, y) - ni * e(x, y)); };
    auto P = [=](T x, T y) { return (alpha * a(x, y) + delta * d(x, y)); };
    auto P_tilde = [=](T x, T y) { return (beta * b(x, y) + ni * e(x, y)); };
    auto Z = [=](T x, T y) { return (two * alpha * a(x, y) - half * rho * f(x, y)); };
    auto W = [=](T x, T y) { return (two * beta * b(x, y) - half * rho * f(x, y)); };
    auto C = [=](T x, T y) { return c(x, y); };
    auto S = [=](T x, T y) { return (one - Z(x, y) - W(x, y)); };

    /// ============= vertical boundaries =====================

    ccontainer_2d_t csolution(next_sol);
    // 1D container for intermediate solution:
    container_t solution(space_size_x, T{});
    const std::size_t N = solution.size() - 1;
    auto horizontal_bc = horizontal_boundary_pair;

    const T time_max = time_rng.upper();
    const T start_x = space_x.lower();
    const T start_y = space_y.lower();
    const T end_x = space_x.upper();
    const T end_y = space_y.upper();
    T x{};
    T y{};

    T time = time_max;
    T next_time{};

    for (std::size_t l = 0; l < time_size; ++l)
    {
        time = time_max - l * k;
        next_time = time - k;
        if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_2d<T>>(std::get<1>(horizontal_bc)))
        {
            const T del = two * h_1 * ptr->value(time, start_y);
            x = start_x + static_cast<T>(N) * h_1;
            solution[N] = ((one - three * ni * e(x, start_y) + rho * f(x, start_y)) * prev_sol(N, 0)) +
                          (four * ni * e(x, start_y) * prev_sol(N, 1)) - (ni * e(x, start_y) * prev_sol(N, 2)) -
                          (delta * del * d(x, start_y));
        }

        for (std::size_t t = 1; t < N; ++t)
        {
            x = start_x + static_cast<T>(t) * h_1;
            solution[t] = (-delta * d(x, start_y) * prev_sol(t - 1, 0)) +
                          ((one - three * ni * e(x, start_y) + rho * f(x, start_y)) * prev_sol(t, 0)) +
                          (delta * d(x, start_y) * prev_sol(t + 1, 0)) - (ni * e(x, start_y) * prev_sol(t, 2)) +
                          (four * ni * e(x, start_y) * prev_sol(t, 1));
        }

        csolution(0, solution);

        auto const &upper_bnd_ptr = std::dynamic_pointer_cast<dirichlet_boundary_2d<T>>(vertical_upper_boundary_ptr);
        auto const &upper_bnd = [=](T s, T t) { return upper_bnd_ptr->value(t, s); };
        d_1d::of_function(start_x, h_1, next_time, upper_bnd, solution);
        csolution(space_size_y - 1, solution);
        next_sol = csolution;

        // intermediate solution:
        // containers for first split solver:
        container_t low(space_size_x, T{});
        container_t diag(space_size_x, T{});
        container_t high(space_size_x, T{});
        container_t rhs(space_size_x, T{});

        auto solver_y = std::make_shared<ds_solver>(space_x, space_size_x);
        auto split_0 = [=](T const &y, container_t &low, container_t &diag, container_t &high) {
            T x{};
            for (std::size_t t = 0; t < low.size(); ++t)
            {
                x = start_x + static_cast<T>(t) * h_1;
                low[t] = (-half * M(x, y));
                diag[t] = (one + half * Z(x, y));
                high[t] = (-half * P(x, y));
            }
        };

        auto scheme_fun = [=](std::size_t const &y_index, T const &y, rcontainer_2d_t const &input,
                              container_t &solution) {
            auto const theta = 0.5;
            const std::size_t N = solution.size() - 1;
            T x{};
            for (std::size_t t = 1; t < N; ++t)
            {
                x = start_x + static_cast<T>(t) * h_1;
                solution[t] =
                    (gamma * C(x, y) * input(t - 1, y_index - 1)) + ((one - theta) * M(x, y) * input(t - 1, y_index)) -
                    (gamma * C(x, y) * input(t - 1, y_index + 1)) + (M_tilde(x, y) * input(t, y_index - 1)) +
                    ((one - W(x, y) - (one - theta) * Z(x, y)) * input(t, y_index)) +
                    (P_tilde(x, y) * input(t, y_index + 1)) - (gamma * C(x, y) * input(t + 1, y_index - 1)) +
                    ((one - theta) * P(x, y) * input(t + 1, y_index)) + (gamma * C(x, y) * input(t + 1, y_index + 1));
            }
        };

        auto y_lower_bc = [=](T t, T y) { return 0.0; };

        auto horizontal_bc_upper = std::get<1>(horizontal_bc);
        const std::size_t lri = prev_sol.rows() - 1;
        const std::size_t lci = prev_sol.columns() - 1;

        auto y_upper_bc = [=](T t, T y) {
            std::size_t j = static_cast<std::size_t>((y - start_y) / h_2);
            return prev_sol(lri, j);
        };
        auto y_horizontal_low = std::make_shared<dirichlet_boundary_2d<T>>(y_lower_bc);
        auto y_horizontal_high = std::make_shared<dirichlet_boundary_2d<T>>(y_upper_bc);
        auto y_horizontal_bc = std::make_pair(y_horizontal_low, y_horizontal_high);
        ccontainer_2d_t inter_solution(space_size_x, space_size_y, T{});
        for (std::size_t j = 1; j < space_size_y - 1; ++j)
        {
            y = start_y + static_cast<T>(j) * h_2;
            split_0(y, low, diag, high);
            scheme_fun(j, y, prev_sol, rhs);
            solver_y->set_diagonals(low, diag, high);
            solver_y->set_rhs(rhs);
            solver_y->solve(y_horizontal_bc, solution, next_time, y);
            inter_solution(j, solution);
        }

        // final solution:
        auto solver_u = std::make_shared<ds_solver>(space_y, space_size_y);
        auto split_1 = [=](T const &x, container_t &low, container_t &diag, container_t &high) {
            T y{};
            for (std::size_t t = 0; t < low.size(); ++t)
            {
                y = start_y + static_cast<T>(t) * h_2;
                low[t] = (-half * M_tilde(x, y));
                diag[t] = (one + half * W(x, y));
                high[t] = (-half * P_tilde(x, y));
            }
        };

        auto scheme_fun_u = [=](std::size_t const &x_index, T const &x, rcontainer_2d_t const &input,
                                rcontainer_2d_t const &intermed, container_t &solution) {
            auto const theta = 0.5;
            const std::size_t N = solution.size() - 1;
            T y{};
            for (std::size_t t = 1; t < N; ++t)
            {
                y = start_y + static_cast<T>(t) * h_2;
                solution[t] = -theta * M_tilde(x, y) * input(x_index, t - 1) + theta * W(x, y) * input(x_index, t) -
                              theta * P_tilde(x, y) * input(x_index, t + 1) + intermed(x_index, t);
            }
        };

        auto u_lower_bc = [=](T t, T x) {
            const std::size_t i = static_cast<std::size_t>((x - start_x) / h_1);
            return next_sol(i, 0);
        };

        auto u_upper_bc = [=](T t, T x) {
            const std::size_t i = static_cast<std::size_t>((x - start_x) / h_1);
            return next_sol(i, lci);
        };
        auto u_vertical_low = std::make_shared<dirichlet_boundary_2d<T>>(u_lower_bc);
        auto u_vertical_high = std::make_shared<dirichlet_boundary_2d<T>>(u_upper_bc);
        auto u_horizontal_bc = std::make_pair(u_vertical_low, u_vertical_high);
        for (std::size_t i = 1; i < space_size_x - 1; ++i)
        {
            x = start_x + static_cast<T>(i) * h_1;
            split_1(x, low, diag, high);
            scheme_fun_u(i, x, prev_sol, inter_solution, rhs);
            solver_u->set_diagonals(low, diag, high);
            solver_u->set_rhs(rhs);
            solver_u->solve(u_horizontal_bc, solution, next_time, x);
            next_sol(i, solution);
        }

        // upper stock boundary:
        auto upper_horizontal = [=](boundary_2d_pair<T> const &boundary, rcontainer_2d_t const &next_solution, T time,
                                    container_t &solution) {
            auto const &second_bnd = boundary.second;
            T const three = static_cast<T>(3.0);
            T const four = static_cast<T>(4.0);
            T const two = static_cast<T>(2.0);
            T y{};
            for (std::size_t t = 0; t < solution.size(); ++t)
            {
                y = start_y + static_cast<T>(t) * h_2;
                solution[t] = ((four * next_solution(lri - 1, t)) - next_solution(lri - 2, t) -
                               (two * h_1 * second_bnd->value(time, y))) /
                              three;
            }
        };

        // lower stock boundary:
        auto lower_horizontal = [=](boundary_2d_pair<T> const &boundary, rcontainer_2d_t const &next_solution, T time,
                                    container_t &solution) {
            auto const &first_bnd = boundary.first;
            T y{};
            for (std::size_t t = 0; t < solution.size(); ++t)
            {
                y = start_y + static_cast<T>(t) * h_2;
                solution[t] = first_bnd->value(time, y);
            }
        };

        lower_horizontal(horizontal_bc, next_sol, next_time, solution);
        next_sol(0, solution);
        upper_horizontal(horizontal_bc, next_sol, next_time, solution);
        next_sol(space_size_x - 1, solution);
        prev_sol = next_sol;
    }
}

void completeTestTest()
{
    completeTest<double>();
}

#endif //_LSS_HESTON_EQUATION_T_HPP_
