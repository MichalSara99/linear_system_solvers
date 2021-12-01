#if !defined(_LSS_1D_GENERAL_WAVE_EQUATION_BUILDER_T_HPP_)
#define _LSS_1D_GENERAL_WAVE_EQUATION_BUILDER_T_HPP_

#include <functional>
#include <map>

#include "builders/lss_1d_general_wave_equation_builder.hpp"
#include "builders/lss_dirichlet_boundary_builder.hpp"
#include "builders/lss_pde_discretization_config_builder.hpp"
#include "builders/lss_wave_data_config_builder.hpp"
#include "builders/lss_wave_solver_config_builder.hpp"
#include "common/lss_utility.hpp"
#include "discretization/lss_discretization.hpp"

template <typename T> void test_pure_wave_equation_builder_t()
{

    using lss_boundary::dirichlet_boundary_1d_builder;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_grids::grid_config_1d;
    using lss_grids::grid_config_hints_1d;
    using lss_grids::grid_transform_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d_builder;
    using lss_pde_solvers::wave_coefficient_data_config_1d_builder;
    using lss_pde_solvers::wave_data_config_1d_builder;
    using lss_pde_solvers::wave_implicit_solver_config_builder;
    using lss_pde_solvers::wave_initial_data_config_1d_builder;
    using lss_pde_solvers::default_wave_solver_configs::dev_fwd_cusolver_qr_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_wave_equation_builder;
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

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(1.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(0.8));
    // discretization config:
    auto const &discretization_ptr = pde_discretization_config_1d_builder<T>()
                                         .space_range(space_range)
                                         .number_of_space_points(Sd)
                                         .time_range(time_range)
                                         .number_of_time_points(Td)
                                         .build();
    // coefficient builder:
    // coeffs:
    auto b = [](T t, T x) { return 1.0; };
    auto zero = [](T t, T x) { return 0.0; };
    auto const &coefficients_ptr = wave_coefficient_data_config_1d_builder<T>()
                                       .a_coefficient(zero)
                                       .b_coefficient(b)
                                       .c_coefficient(zero)
                                       .d_coefficient(zero)
                                       .build();
    // initial condition builder:
    auto const &init_data_ptr = wave_initial_data_config_1d_builder<T>()
                                    .first_initial_condition([](T x) { return std::sin(pi<T>() * x); })
                                    .second_initial_condition([](T x) { return 0.0; })
                                    .build();
    // wave data config builder:
    auto const &data_ptr = wave_data_config_1d_builder<T>()
                               .coefficient_data_config(coefficients_ptr)
                               .initial_data_config(init_data_ptr)
                               .build();

    // grid config:
    auto const &grid_config_hints_ptr = std::make_shared<grid_config_hints_1d<T>>();

    // boundary conditions builder:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_low = dirichlet_boundary_1d_builder<T>().value(dirichlet).build();
    auto const &boundary_high = dirichlet_boundary_1d_builder<T>().value(dirichlet).build();
    auto const &boundary_pair = std::make_pair(boundary_low, boundary_high);

    // pde solver builder:
    auto const &pde_solver = general_wave_equation_builder<T, std::vector, std::allocator<T>>()
                                 .boundary_pair(boundary_pair)
                                 .discretization_config(discretization_ptr)
                                 .grid_config_hints(grid_config_hints_ptr)
                                 .solver_config(dev_fwd_cusolver_qr_solver_config_ptr)
                                 .wave_data_config(data_ptr)
                                 .build();

    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pde_solver->solve(solution);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        const T var1 = std::sin(pi<T>() * x);
        const T var2 = std::cos(pi<T>() * t);
        return (var1 * var2);
    };

    T x{};
    auto const grid_cfg = std::make_shared<grid_config_1d<T>>(discretization_ptr);
    auto const grid_trans_cfg =
        std::make_shared<grid_transform_config_1d<T>>(discretization_ptr, grid_config_hints_ptr);
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        x = grid_1d<T>::transformed_value(grid_trans_cfg, grid_1d<T>::value(grid_cfg, j));
        benchmark = exact(x, time_range.upper(), 20);
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

void test_pure_wave_equation_builder()
{
    test_pure_wave_equation_builder_t<float>();
    test_pure_wave_equation_builder_t<double>();
}

template <typename T> void test_expl_pure_wave_equation_builder_t()
{

    using lss_boundary::dirichlet_boundary_1d_builder;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_grids::grid_config_1d;
    using lss_grids::grid_config_hints_1d;
    using lss_grids::grid_transform_config_1d;
    using lss_pde_solvers::pde_discretization_config_1d_builder;
    using lss_pde_solvers::wave_coefficient_data_config_1d_builder;
    using lss_pde_solvers::wave_data_config_1d_builder;
    using lss_pde_solvers::wave_initial_data_config_1d_builder;
    using lss_pde_solvers::default_wave_solver_configs::dev_expl_fwd_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::explicit_solvers::general_wave_equation_builder;
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

    // number of space subdivisions:
    std::size_t const Sd = 50;
    // number of time subdivisions:
    std::size_t const Td = 2000;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(1.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(0.8));
    // discretization config:
    auto const &discretization_ptr = pde_discretization_config_1d_builder<T>()
                                         .space_range(space_range)
                                         .number_of_space_points(Sd)
                                         .time_range(time_range)
                                         .number_of_time_points(Td)
                                         .build();
    // coefficient builder:
    // coeffs:
    auto b = [](T t, T x) { return 1.0; };
    auto zero = [](T t, T x) { return 0.0; };
    auto const &coefficients_ptr = wave_coefficient_data_config_1d_builder<T>()
                                       .a_coefficient(zero)
                                       .b_coefficient(b)
                                       .c_coefficient(zero)
                                       .d_coefficient(zero)
                                       .build();
    // initial condition builder:
    auto const &init_data_ptr = wave_initial_data_config_1d_builder<T>()
                                    .first_initial_condition([](T x) { return std::sin(pi<T>() * x); })
                                    .second_initial_condition([](T x) { return 0.0; })
                                    .build();
    // wave data config builder:
    auto const &data_ptr = wave_data_config_1d_builder<T>()
                               .coefficient_data_config(coefficients_ptr)
                               .initial_data_config(init_data_ptr)
                               .build();

    // grid config:
    auto const &grid_config_hints_ptr = std::make_shared<grid_config_hints_1d<T>>();

    // boundary conditions builder:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_low = dirichlet_boundary_1d_builder<T>().value(dirichlet).build();
    auto const &boundary_high = dirichlet_boundary_1d_builder<T>().value(dirichlet).build();
    auto const &boundary_pair = std::make_pair(boundary_low, boundary_high);

    // pde solver builder:
    auto const &pde_solver = general_wave_equation_builder<T, std::vector, std::allocator<T>>()
                                 .boundary_pair(boundary_pair)
                                 .discretization_config(discretization_ptr)
                                 .grid_config_hints(grid_config_hints_ptr)
                                 .solver_config(dev_expl_fwd_solver_config_ptr)
                                 .wave_data_config(data_ptr)
                                 .build();
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pde_solver->solve(solution);
    // get exact solution:
    auto exact = [](T x, T t) {
        const T var1 = std::sin(pi<T>() * x);
        const T var2 = std::cos(pi<T>() * t);
        return (var1 * var2);
    };

    T x{};
    auto const grid_cfg = std::make_shared<grid_config_1d<T>>(discretization_ptr);
    auto const grid_trans_cfg =
        std::make_shared<grid_transform_config_1d<T>>(discretization_ptr, grid_config_hints_ptr);
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        x = grid_1d<T>::transformed_value(grid_trans_cfg, grid_1d<T>::value(grid_cfg, j));
        benchmark = exact(x, time_range.upper());
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

void test_expl_pure_wave_equation_builder()
{
    test_expl_pure_wave_equation_builder_t<float>();
    test_expl_pure_wave_equation_builder_t<double>();
}

#endif ///_LSS_1D_GENERAL_WAVE_EQUATION_BUILDER_T_HPP_
