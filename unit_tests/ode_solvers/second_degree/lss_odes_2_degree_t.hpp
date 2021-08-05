#pragma once
#if !defined(_LSS_ODE_2_DEGREE_T_HPP_)
#define _LSS_ODE_2_DEGREE_T_HPP_

#include <vector>

#include "boundaries/lss_dirichlet_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include "ode_solvers/second_degree/lss_general_ode_equation.hpp"

template <typename T> void testImplSimpleODEDirichletBCCUDASolverDeviceQR()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_ode_solvers::dev_cusolver_qr_solver_config_ptr;
    using lss_ode_solvers::ode_coefficient_data_config;
    using lss_ode_solvers::ode_data_config;
    using lss_ode_solvers::ode_discretization_config;
    using lss_ode_solvers::ode_nonhom_data_config;
    using lss_ode_solvers::implicit_solvers::general_ode_equation;
    using lss_utility::range;

    std::cout << "=================================\n";
    std::cout << "Solving Boundary-value problem: \n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " u''(t) = -2, \n\n";
    std::cout << " where\n\n";
    std::cout << " t in <0,1>,\n";
    std::cout << " u(0) = u(1) = 0\n\n";
    std::cout << "Exact solution is:\n\n";
    std::cout << " u(t) = t(1-t)\n";
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
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // initialize ode solver
    ode_solver odesolver(ode_data_ptr, discretization_ptr, boundary_pair, dev_cusolver_qr_solver_config_ptr);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    odesolver.solve(solution);

    // exact value:
    auto exact = [](T x) { return x * (static_cast<T>(1.0) - x); };
    T const h = discretization_ptr->space_step();
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        benchmark = exact(j * h);
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

void testImplSimpleODEDirichletBCCUDASolverDevice()
{
    std::cout << "==================================================\n";
    std::cout << "==================== (Dir-Dir BC) ================\n";
    std::cout << "==================================================\n";

    testImplSimpleODEDirichletBCCUDASolverDeviceQR<double>();
    testImplSimpleODEDirichletBCCUDASolverDeviceQR<float>();

    std::cout << "==================================================\n";
}

template <typename T> void testImplSimpleODEDirichletNeumannBCCUDASolverDeviceQR()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_boundary::neumann_boundary_1d;
    using lss_ode_solvers::dev_cusolver_qr_solver_config_ptr;
    using lss_ode_solvers::ode_coefficient_data_config;
    using lss_ode_solvers::ode_data_config;
    using lss_ode_solvers::ode_discretization_config;
    using lss_ode_solvers::ode_nonhom_data_config;
    using lss_ode_solvers::implicit_solvers::general_ode_equation;
    using lss_utility::range;

    std::cout << "=================================\n";
    std::cout << "Solving Boundary-value problem: \n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " u''(t) = 6*t, \n\n";
    std::cout << " where\n\n";
    std::cout << " t in <0,2>,\n";
    std::cout << " u(0) = 1 \n";
    std::cout << " u'(2) = 0\n\n";
    std::cout << "Exact solution is:\n\n";
    std::cout << " u(t) = t*t*t - 12*t + 1\n";
    std::cout << "=================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_ode_equation<T, std::vector, std::allocator<T>> ode_solver;

    // number of space subdivisions:
    std::size_t Sd{100};
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(2.0));
    // discretization config:
    auto const discretization_ptr = std::make_shared<ode_discretization_config<T>>(space_range, Sd);
    // coeffs:
    auto a = [](T x) { return 0.0; };
    auto b = [](T x) { return 0.0; };
    auto const ode_coeffs_data_ptr = std::make_shared<ode_coefficient_data_config<T>>(a, b);
    // nonhom data:
    auto two = [](T x) { return 6.0 * x; };
    auto const ode_nonhom_data_ptr = std::make_shared<ode_nonhom_data_config<T>>(two);
    // ode data config:
    auto const ode_data_ptr = std::make_shared<ode_data_config<T>>(ode_coeffs_data_ptr, ode_nonhom_data_ptr);
    // boundary conditions:
    auto const &dirichlet = [](T t) { return 1.0; };
    auto const &neumann = [](T t) { return 0.0; };
    auto const &boundary_low_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_high_ptr = std::make_shared<neumann_boundary_1d<T>>(neumann);
    auto const &boundary_pair = std::make_pair(boundary_low_ptr, boundary_high_ptr);
    // initialize ode solver
    ode_solver odesolver(ode_data_ptr, discretization_ptr, boundary_pair, dev_cusolver_qr_solver_config_ptr);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    odesolver.solve(solution);

    // exact value:
    auto exact = [](T x) { return (x * x * x - 12.0 * x + 1.0); };

    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T const h = discretization_ptr->space_step();
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        benchmark = exact(j * h);
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

void testImplSimpleODEDirichletNeumannBCCUDASolverDevice()
{
    std::cout << "==================================================\n";
    std::cout << "================= (Dir-Neu BC) ===================\n";
    std::cout << "==================================================\n";

    testImplSimpleODEDirichletNeumannBCCUDASolverDeviceQR<double>();
    testImplSimpleODEDirichletNeumannBCCUDASolverDeviceQR<float>();

    std::cout << "==================================================\n";
}

template <typename T> void testImplSimpleODEDirichletRobinBCCUDASolverDeviceQR()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_boundary::robin_boundary_1d;
    using lss_ode_solvers::dev_cusolver_qr_solver_config_ptr;
    using lss_ode_solvers::ode_coefficient_data_config;
    using lss_ode_solvers::ode_data_config;
    using lss_ode_solvers::ode_discretization_config;
    using lss_ode_solvers::ode_nonhom_data_config;
    using lss_ode_solvers::implicit_solvers::general_ode_equation;
    using lss_utility::range;

    std::cout << "=================================\n";
    std::cout << "Solving Boundary-value problem: \n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " u''(t) = -2, \n\n";
    std::cout << " where\n\n";
    std::cout << " t in <0,1>,\n";
    std::cout << " u(0) = 1 \n";
    std::cout << " u'(1) + u(1) = 0\n\n";
    std::cout << "Exact solution is:\n\n";
    std::cout << " u(t) = -t*t + t + 1\n";
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
    auto const &dirichlet = [](T t) { return 1.0; };
    auto const &robin_first = [](T t) { return 1.0; };
    auto const &robin_second = [](T t) { return 0.0; };
    auto const &boundary_low_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_high_ptr = std::make_shared<robin_boundary_1d<T>>(robin_first, robin_second);
    auto const &boundary_pair = std::make_pair(boundary_low_ptr, boundary_high_ptr);
    // initialize ode solver
    ode_solver odesolver(ode_data_ptr, discretization_ptr, boundary_pair, dev_cusolver_qr_solver_config_ptr);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    odesolver.solve(solution);

    // exact value:
    auto exact = [](T x) { return (-x * x + x + static_cast<T>(1.0)); };

    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T const h = discretization_ptr->space_step();
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        benchmark = exact(j * h);
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

void testImplSimpleODEDirichletRobinBCCUDASolverDevice()
{
    std::cout << "==================================================\n";
    std::cout << "================= (Dir-Rob BC) ===================\n";
    std::cout << "==================================================\n";

    testImplSimpleODEDirichletRobinBCCUDASolverDeviceQR<double>();
    testImplSimpleODEDirichletRobinBCCUDASolverDeviceQR<float>();

    std::cout << "==================================================\n";
}

template <typename T> void testImplSimpleODENeumannRobinBCCUDASolverDeviceQR()
{
    using lss_boundary::neumann_boundary_1d;
    using lss_boundary::robin_boundary_1d;
    using lss_ode_solvers::dev_cusolver_qr_solver_config_ptr;
    using lss_ode_solvers::ode_coefficient_data_config;
    using lss_ode_solvers::ode_data_config;
    using lss_ode_solvers::ode_discretization_config;
    using lss_ode_solvers::ode_nonhom_data_config;
    using lss_ode_solvers::implicit_solvers::general_ode_equation;
    using lss_utility::range;

    std::cout << "=================================\n";
    std::cout << "Solving Boundary-value problem: \n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " u''(t) = 6*t, \n\n";
    std::cout << " where\n\n";
    std::cout << " t in <0,2>,\n";
    std::cout << " u'(0) = 0 \n";
    std::cout << " u'(2) + 2*u(t) = 0\n\n";
    std::cout << "Exact solution is:\n\n";
    std::cout << " u(t) = t*t*t - 14\n";
    std::cout << "=================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_ode_equation<T, std::vector, std::allocator<T>> ode_solver;

    // number of space subdivisions:
    std::size_t Sd{100};
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(2.0));
    // discretization config:
    auto const discretization_ptr = std::make_shared<ode_discretization_config<T>>(space_range, Sd);
    // coeffs:
    auto a = [](T x) { return 0.0; };
    auto b = [](T x) { return 0.0; };
    auto const ode_coeffs_data_ptr = std::make_shared<ode_coefficient_data_config<T>>(a, b);
    // nonhom data:
    auto two = [](T x) { return 6 * x; };
    auto const ode_nonhom_data_ptr = std::make_shared<ode_nonhom_data_config<T>>(two);
    // ode data config:
    auto const ode_data_ptr = std::make_shared<ode_data_config<T>>(ode_coeffs_data_ptr, ode_nonhom_data_ptr);
    // boundary conditions:
    auto const &neumann = [](T t) { return 0.0; };
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

    // exact value:
    auto exact = [](T x) { return (x * x * x - static_cast<T>(14.0)); };

    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T const h = discretization_ptr->space_step();
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        benchmark = exact(j * h);
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

void testImplSimpleODENeumannRobinBCCUDASolverDevice()
{
    std::cout << "==================================================\n";
    std::cout << "=================  (Neu-Rob BC) ==================\n";
    std::cout << "==================================================\n";

    testImplSimpleODENeumannRobinBCCUDASolverDeviceQR<double>();
    testImplSimpleODENeumannRobinBCCUDASolverDeviceQR<float>();

    std::cout << "==================================================\n";
}

template <typename T> void testImplSimpleODE1NeumannRobinBCCUDASolverDeviceQR()
{
    using lss_boundary::neumann_boundary_1d;
    using lss_boundary::robin_boundary_1d;
    using lss_ode_solvers::dev_cusolver_qr_solver_config_ptr;
    using lss_ode_solvers::ode_coefficient_data_config;
    using lss_ode_solvers::ode_data_config;
    using lss_ode_solvers::ode_discretization_config;
    using lss_ode_solvers::ode_nonhom_data_config;
    using lss_ode_solvers::implicit_solvers::general_ode_equation;
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

    // exact value:
    auto exact = [](T x) { return (-x * x + x + static_cast<T>(0.5)); };

    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T const h = discretization_ptr->space_step();
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        benchmark = exact(j * h);
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

void testImplSimpleODE1NeumannRobinBCCUDASolverDevice()
{
    std::cout << "==================================================\n";
    std::cout << "================ (Neu - Rob) =====================\n";
    std::cout << "==================================================\n";

    testImplSimpleODE1NeumannRobinBCCUDASolverDeviceQR<double>();
    testImplSimpleODE1NeumannRobinBCCUDASolverDeviceQR<float>();

    std::cout << "==================================================\n";
}

#endif ///_LSS_ODE_2_DEGREE_T_HPP_
