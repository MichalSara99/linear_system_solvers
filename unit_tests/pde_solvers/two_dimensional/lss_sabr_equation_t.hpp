#if !defined(_LSS_SABR_EQUATION_T_HPP_)
#define _LSS_SABR_EQUATION_T_HPP_

#include <cmath>
#include <map>
#include <sstream>

#include "pde_solvers/two_dimensional/heat_type/lss_2d_general_heston_equation.hpp"
#include "sparse_solvers/tridiagonal/double_sweep_solver/lss_double_sweep_solver.hpp"

// ///////////////////////////////////////////////////////////////////////////
//							SABR PROBLEMS
// ///////////////////////////////////////////////////////////////////////////

// ===========================================================================
// ========================== IMPLICIT SOLVERS ===============================
// ===========================================================================

template <typename T> void testImplSABREquationDoubleSweepSolverCrankNicolson()
{
    using lss_boundary::dirichlet_boundary_2d;
    using lss_boundary::neumann_boundary_2d;
    using lss_containers::container_2d;
    using lss_enumerations::by_enum;
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
    auto const &sig_sig = 0.041;
    auto const &rho = 0.6;
    auto const &beta = 0.7;
    // number of space subdivisions for spot:
    std::size_t const Sd = 50;
    // number of space subdivision for volatility:
    std::size_t const Vd = 50;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space Spot range:
    range<T> spacex_range(static_cast<T>(50.0), static_cast<T>(200.0));
    // space Vol range:
    range<T> spacey_range(static_cast<T>(0.0), static_cast<T>(1.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(maturity));
    auto const half = static_cast<T>(0.5);
    auto const one = static_cast<T>(1.0);
    auto const two = static_cast<T>(2.0);

    // discretization config:
    auto const discretization_ptr =
        std::make_shared<pde_discretization_config_2d<T>>(spacex_range, spacey_range, Sd, Vd, time_range, Td);
    // coeffs:
    auto D = [=](T s, T alpha) { return std::exp(-rate * half); };
    auto a = [=](T t, T s, T alpha) {
        return (0.5 * alpha * alpha * std::pow(s, two * beta) * std::pow(D(s, alpha), two * (one - beta)));
    };
    auto b = [=](T t, T s, T alpha) { return (half * sig_sig * sig_sig * alpha * alpha); };
    auto c = [=](T t, T s, T alpha) {
        return (rho * sig_sig * alpha * alpha * std::pow(s, beta) * std::pow(D(s, alpha), (one - beta)));
    };
    auto d = [=](T t, T s, T alpha) { return (rate * s); };
    auto e = [=](T t, T s, T alpha) { return 0.0; };
    auto f = [=](T t, T s, T alpha) { return (-rate); };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_2d<T>>(a, b, c, d, e, f);
    // terminal condition:
    auto terminal_condition = [=](T s, T v) -> T { return std::max<T>(0.0, s - strike); };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_2d<T>>(terminal_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_2d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // horizontal spot boundary conditions:
    auto const &dirichlet_low = [=](T t, T v) -> T { return 0.0; };
    auto const &neumann_high = [=](T t, T s) -> T { return -1.0; };
    auto const &boundary_low_ptr = std::make_shared<dirichlet_boundary_2d<T>>(dirichlet_low);
    auto const &boundary_high_ptr = std::make_shared<neumann_boundary_2d<T>>(neumann_high);
    auto const &horizontal_boundary_pair = std::make_pair(boundary_low_ptr, boundary_high_ptr);
    // vertical upper vol boundary:
    auto const &dirichlet_high = [=](T t, T s) -> T { return s; };
    auto const &vertical_upper_boundary_ptr = std::make_shared<dirichlet_boundary_2d<T>>(dirichlet_high);
    // splitting method configuration:
    auto const &splitting_config_ptr =
        std::make_shared<splitting_method_config<T>>(splitting_method_enum::DouglasRachford);
    // grid config:
    auto const &grid_config_hints_ptr = std::make_shared<grid_config_hints_2d<T>>(strike);

    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, vertical_upper_boundary_ptr, horizontal_boundary_pair,
                         splitting_config_ptr, grid_config_hints_ptr, host_bwd_dssolver_cn_solver_config_ptr);
    // prepare container for solution:
    rcontainer_2d_t solution(Sd, Vd, T{});
    // get the solution:
    pdesolver.solve(solution);

    print(discretization_ptr, grid_config_hints_ptr, solution);
}

void testImplSABREquationDoubleSweepSolver()
{
    std::cout << "============================================================\n";
    std::cout << "============== Implicit SABR Equation (Dir BC) =============\n";
    std::cout << "============================================================\n";

    testImplSABREquationDoubleSweepSolverCrankNicolson<double>();
    testImplSABREquationDoubleSweepSolverCrankNicolson<float>();

    std::cout << "============================================================\n";
}

#endif //_LSS_SABR_EQUATION_T_HPP_
