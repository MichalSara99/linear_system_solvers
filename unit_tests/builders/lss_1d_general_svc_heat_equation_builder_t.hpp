#if !defined(_LSS_1D_GENERAL_SVC_HEAT_EQUATION_BUILDER_T_HPP_)
#define _LSS_1D_GENERAL_SVC_HEAT_EQUATION_BUILDER_T_HPP_

#include <functional>
#include <map>

#include "builders/lss_1d_general_svc_heat_equation_builder.hpp"
#include "builders/lss_dirichlet_boundary_builder.hpp"
#include "builders/lss_heat_data_config_builder.hpp"
#include "builders/lss_heat_solver_config_builder.hpp"
#include "builders/lss_pde_discretization_config_builder.hpp"
#include "common/lss_utility.hpp"
#include "discretization/lss_discretization.hpp"

template <typename T> void test_pure_heat_equation_builder_t()
{
    using lss_boundary::dirichlet_boundary_1d_builder;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_pde_solvers::heat_coefficient_data_config_1d_builder;
    using lss_pde_solvers::heat_data_config_1d_builder;
    using lss_pde_solvers::heat_implicit_solver_config_builder;
    using lss_pde_solvers::heat_initial_data_config_1d_builder;
    using lss_pde_solvers::pde_discretization_config_1d_builder;
    using lss_pde_solvers::default_heat_solver_configs::dev_fwd_cusolver_qr_euler_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation_builder;
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

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(1.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(0.1));
    // discretization config:
    auto const &discretization_ptr = pde_discretization_config_1d_builder<T>()
                                         .space_range(space_range)
                                         .number_of_space_points(Sd)
                                         .time_range(time_range)
                                         .number_of_time_points(Td)
                                         .build();
    // coefficient builder:
    // coeffs:
    auto a = [](T x) { return 1.0; };
    auto zero = [](T x) { return 0.0; };
    auto const &coefficients_ptr =
        heat_coefficient_data_config_1d_builder<T>().a_coefficient(a).b_coefficient(zero).c_coefficient(zero).build();

    // initial condition builder:
    auto const &init_data_ptr =
        heat_initial_data_config_1d_builder<T>().initial_condition([](T x) { return x; }).build();
    // heat data config builder:
    auto const &data_ptr = heat_data_config_1d_builder<T>()
                               .coefficient_data_config(coefficients_ptr)
                               .initial_data_config(init_data_ptr)
                               .build();

    // boundary conditions builder:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_low = dirichlet_boundary_1d_builder<T>().value(dirichlet).build();
    auto const &boundary_high = dirichlet_boundary_1d_builder<T>().value(dirichlet).build();
    auto const &boundary_pair = std::make_pair(boundary_low, boundary_high);

    // pde solver builder:
    auto const &pde_solver = general_svc_heat_equation_builder<T, std::vector, std::allocator<T>>()
                                 .boundary_pair(boundary_pair)
                                 .discretization_config(discretization_ptr)
                                 .solver_config(dev_fwd_cusolver_qr_euler_solver_config_ptr)
                                 .heat_data_config(data_ptr)
                                 .build();

    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pde_solver->solve(solution);
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
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        benchmark = exact(j * h, time_range.upper(), 20);
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

void test_pure_heat_equation_builder()
{
    test_pure_heat_equation_builder_t<float>();
    test_pure_heat_equation_builder_t<double>();
}

#endif ///_LSS_1D_GENERAL_SVC_HEAT_EQUATION_BUILDER_T_HPP_
