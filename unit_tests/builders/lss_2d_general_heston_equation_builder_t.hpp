#if !defined(_LSS_2D_GENERAL_HESTON_EQUATION_BUILDER_T_HPP_)
#define _LSS_2D_GENERAL_HESTON_EQUATION_BUILDER_T_HPP_

#include <functional>
#include <map>
#include <sstream>

#include "builders/lss_2d_general_heston_equation_builder.hpp"
#include "builders/lss_dirichlet_boundary_builder.hpp"
#include "builders/lss_grid_config_hints_builder.hpp"
#include "builders/lss_heat_data_config_builder.hpp"
#include "builders/lss_heat_solver_config_builder.hpp"
#include "builders/lss_neumann_boundary_builder.hpp"
#include "builders/lss_pde_discretization_config_builder.hpp"
#include "builders/lss_range_builder.hpp"
#include "builders/lss_splitting_method_config_builder.hpp"
#include "common/lss_print.hpp"

template <typename T> void test_heston_equation_builder_t()
{
    using lss_boundary::dirichlet_boundary_2d_builder;
    using lss_boundary::neumann_boundary_2d_builder;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_enumerations::splitting_method_enum;
    using lss_grids::grid_config_hints_2d;
    using lss_grids::grid_config_hints_2d_builder;
    using lss_grids::grid_enum;
    using lss_pde_solvers::heat_coefficient_data_config_2d_builder;
    using lss_pde_solvers::heat_data_config_2d_builder;
    using lss_pde_solvers::heat_implicit_solver_config_builder;
    using lss_pde_solvers::heat_initial_data_config_2d_builder;
    using lss_pde_solvers::pde_discretization_config_2d_builder;
    using lss_pde_solvers::splitting_method_config_builder;
    using lss_pde_solvers::default_heat_solver_configs::host_bwd_dssolver_cn_solver_config_ptr;
    using lss_pde_solvers::two_dimensional::implicit_solvers::general_heston_equation_builder;
    using lss_print::print;
    using lss_utility::pi;
    using lss_utility::range_builder;

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
    auto const &stock_range = range_builder<T>().lower(T(0.0)).upper(T(20.0)).build();
    // space Vol range:
    auto const &vol_range = range_builder<T>().lower(T(0.0)).upper(T(1.5)).build();
    // time range
    auto const &time_range = range_builder<T>().lower(T(0.0)).upper(maturity).build();
    // discretization config:
    auto const &discretization_ptr = pde_discretization_config_2d_builder<T>()
                                         .space_range_1(*stock_range)
                                         .space_range_2(*vol_range)
                                         .number_of_space_points_1(Sd)
                                         .number_of_space_points_2(Vd)
                                         .time_range(*time_range)
                                         .number_of_time_points(Td)
                                         .build();
    // coefficient builder:
    auto a = [=](T t, T s, T v) { return (0.5 * v * s * s); };
    auto b = [=](T t, T s, T v) { return (0.5 * sig_sig * sig_sig * v); };
    auto c = [=](T t, T s, T v) { return (rho * sig_sig * v * s); };
    auto d = [=](T t, T s, T v) { return (rate * s); };
    auto e = [=](T t, T s, T v) { return (sig_kappa * (sig_theta - v)); };
    auto f = [=](T t, T s, T v) { return (-rate); };
    auto const &coefficients_ptr = heat_coefficient_data_config_2d_builder<T>()
                                       .a_coefficient(a)
                                       .b_coefficient(b)
                                       .c_coefficient(c)
                                       .d_coefficient(d)
                                       .e_coefficient(e)
                                       .f_coefficient(f)
                                       .build();

    // initial condition builder:
    // terminal condition:
    auto terminal_condition = [=](T s, T v) { return std::max<T>(0.0, s - strike); };
    auto const &init_data_ptr = heat_initial_data_config_2d_builder<T>().initial_condition(terminal_condition).build();
    // heat data config builder:
    auto const &data_ptr = heat_data_config_2d_builder<T>()
                               .coefficient_data_config(coefficients_ptr)
                               .initial_data_config(init_data_ptr)
                               .build();

    // grid config:
    auto const alpha = static_cast<T>(3.);
    auto const beta = static_cast<T>(60.);
    auto const &grid_config_hints_ptr = grid_config_hints_2d_builder<T>()
                                            .accumulation_point(strike)
                                            .alpha_scale(alpha)
                                            .beta_scale(beta)
                                            .grid(grid_enum::Nonuniform)
                                            .build();

    // horizontal boundary conditions builder:
    auto const &dirichlet_low = [=](T t, T v) { return 0.0; };
    auto const &neumann_high = [=](T t, T v) { return -1.0; };
    auto const &boundary_low_ptr = dirichlet_boundary_2d_builder<T>().value(dirichlet_low).build();
    auto const &boundary_high_ptr = neumann_boundary_2d_builder<T>().value(neumann_high).build();
    auto const &horizontal_boundary_pair = std::make_pair(boundary_low_ptr, boundary_high_ptr);

    // vertical upper vol boundary builder:
    auto const &dirichlet_high = [=](T t, T s) { return s; };
    auto const &vertical_upper_boundary_ptr = dirichlet_boundary_2d_builder<T>().value(dirichlet_high).build();
    // splitting method configuration:
    auto const &splitting_config_ptr = splitting_method_config_builder<T>()
                                           .splitting_method(splitting_method_enum::DouglasRachford)
                                           .weighting_value(T(0.5))
                                           .build();

    // pde solver builder:
    auto const &pde_solver = general_heston_equation_builder<T, std::vector, std::allocator<T>>()
                                 .heat_data_config(data_ptr)
                                 .discretization_config(discretization_ptr)
                                 .vertical_upper_boundary(vertical_upper_boundary_ptr)
                                 .horizontal_boundary(horizontal_boundary_pair)
                                 .splitting_method_config(splitting_config_ptr)
                                 .grid_hints(grid_config_hints_ptr)
                                 .solver_config(host_bwd_dssolver_cn_solver_config_ptr)
                                 .build();

    // prepare container for solution:
    rcontainer_2d_t solution(Sd, Vd, T{});
    // get the solution:
    pde_solver->solve(solution);

    // print approx only:
    std::stringstream ssa;
    std::string name = "ImplHestonEquationDSSolverCrankNicolsonFromBuilder_";
    ssa << "outputs/" << name << typeid(T).name() << ".txt";
    std::string file_name_approx{ssa.str()};
    std::ofstream approx(file_name_approx);
    print(discretization_ptr, grid_config_hints_ptr, solution, approx);
    approx.close();
}

void test_heston_equation_builder()
{
    test_heston_equation_builder_t<float>();
    test_heston_equation_builder_t<double>();
}

#endif ///_LSS_2D_GENERAL_HESTON_EQUATION_BUILDER_T_HPP_
