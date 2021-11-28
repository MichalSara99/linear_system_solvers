#if !defined(_LSS_GENERAL_2ND_ODE_EQUATION_BUILDER_T_HPP_)
#define _LSS_GENERAL_2ND_ODE_EQUATION_BUILDER_T_HPP_

#include <functional>
#include <map>
#include <sstream>

#include "builders/lss_general_2nd_ode_equation_builder.hpp"
#include "builders/lss_grid_config_hints_builder.hpp"
#include "builders/lss_neumann_boundary_builder.hpp"
#include "builders/lss_ode_data_config_builder.hpp"
#include "builders/lss_ode_discretization_config_builder.hpp"
#include "builders/lss_ode_solver_config_builder.hpp"
#include "builders/lss_robin_boundary_builder.hpp"
#include "common/lss_print.hpp"
#include "common/lss_utility.hpp"
#include "discretization/lss_discretization.hpp"

template <typename T> void test_general_2nd_ode_equation_builder_t()
{

    using lss_boundary::neumann_boundary_1d_builder;
    using lss_boundary::robin_boundary_1d_builder;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_grids::grid_config_hints_1d_builder;
    using lss_grids::grid_enum;
    using lss_ode_solvers::ode_coefficient_data_config_builder;
    using lss_ode_solvers::ode_data_config_builder;
    using lss_ode_solvers::ode_discretization_config_builder;
    using lss_ode_solvers::ode_implicit_solver_config_builder;
    using lss_ode_solvers::ode_nonhom_data_config_builder;
    using lss_ode_solvers::default_ode_solver_configs::host_tlusolver_solver_config_ptr;
    using lss_ode_solvers::implicit_solvers::general_2nd_ode_equation_builder;
    using lss_print::print;
    using lss_utility::pi;
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

    // number of space subdivisions:
    std::size_t Sd{100};
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(1.0));
    // discretization config:
    auto const &discretization_ptr =
        ode_discretization_config_builder<T>().space_range(space_range).number_of_space_points(Sd).build();
    // coeffs:
    auto a = [](T x) { return 0.0; };
    auto b = [](T x) { return 0.0; };
    auto const ode_coeffs_data_ptr = ode_coefficient_data_config_builder<T>().a_coefficient(a).b_coefficient(b).build();
    // nonhom data:
    auto two = [](T x) { return -2.0; };
    auto const ode_nonhom_data_ptr = ode_nonhom_data_config_builder<T>().nonhom_function(two).build();
    // ode data config:
    auto const ode_data_ptr = ode_data_config_builder<T>()
                                  .coefficient_data_config(ode_coeffs_data_ptr)
                                  .nonhom_data_config(ode_nonhom_data_ptr)
                                  .build();

    // boundary conditions:
    auto const &neumann = [](T t) { return -1.0; };
    auto const &robin_first = [](T t) { return 2.0; };
    auto const &robin_second = [](T t) { return 0.0; };
    auto const &boundary_low_ptr = neumann_boundary_1d_builder<T>().value(neumann).build();
    auto const &boundary_high_ptr =
        robin_boundary_1d_builder<T>().linear_value(robin_first).value(robin_second).build();
    auto const &boundary_pair = std::make_pair(boundary_low_ptr, boundary_high_ptr);
    // grid config:
    auto const &alpha_scale = 3.0;
    auto const &grid_config_hints_ptr = grid_config_hints_1d_builder<T>()
                                            .accumulation_point(0.5)
                                            .alpha_scale(alpha_scale)
                                            .grid(grid_enum::Nonuniform)
                                            .build();
    // ode solver builder
    auto const &ode_solver = general_2nd_ode_equation_builder<T, std::vector, std::allocator<T>>()
                                 .ode_data_config(ode_data_ptr)
                                 .discretization_config(discretization_ptr)
                                 .boundary_pair(boundary_pair)
                                 .grid_hints(grid_config_hints_ptr)
                                 .solver_config(host_tlusolver_solver_config_ptr)
                                 .build();

    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    ode_solver->solve(solution);

    // print both of these
    std::stringstream ssa;
    ssa << "outputs/SimpleODEFromBuilder_" << typeid(T).name() << ".txt";
    std::string file_name_approx{ssa.str()};
    std::ofstream approx(file_name_approx);
    print(discretization_ptr, solution, approx);
    approx.close();
    std::cout << "approx saved to file: " << file_name_approx << "\n";
}

void test_general_2nd_ode_equation_builder()
{
    test_general_2nd_ode_equation_builder_t<float>();
    test_general_2nd_ode_equation_builder_t<double>();
}

#endif ///_LSS_GENERAL_2ND_ODE_EQUATION_BUILDER_T_HPP_
