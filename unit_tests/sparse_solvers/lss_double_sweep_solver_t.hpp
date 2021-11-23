#pragma once
#if !defined(_LSS_DOUBLE_SWEEP_SOLVER_T_HPP_)
#define _LSS_DOUBLE_SWEEP_SOLVER_T_HPP_

#include <vector>

#include "boundaries/lss_dirichlet_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include "discretization/lss_grid_config.hpp"
#include "ode_solvers/lss_ode_discretization_config.hpp"
#include "sparse_solvers/tridiagonal/double_sweep_solver/lss_double_sweep_solver.hpp"

template <typename T> void testBVPDoubleSweepDirichletBC()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_double_sweep_solver::double_sweep_solver;
    using lss_utility::range;

    /*

    Solve BVP:

    u''(t) = - 2,

    where

    t \in (0, 1)
    u(0) = 0 ,  u(1) = 0


    Exact solution is

    u(t) = t(1-t)

    */
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

    // discretization:
    std::size_t N{100};
    // step size:
    double h = 1.0 / static_cast<T>(N - 1);
    // upper,mid, and lower diagonal:
    std::vector<T> upper_diag(N, static_cast<T>(1.0));
    std::vector<T> diagonal(N, static_cast<T>(-2.0));
    std::vector<T> lower_diag(N, static_cast<T>(1.0));

    // right-hand side:
    std::vector<T> rhs(N, static_cast<T>(-2.0) * h * h);

    // boundary conditions:
    auto const &lower_ptr = std::make_shared<dirichlet_boundary_1d<T>>([](T t) { return 0.0; });
    auto const &upper_ptr = std::make_shared<dirichlet_boundary_1d<T>>([](T t) { return 0.0; });

    auto dss = std::make_shared<double_sweep_solver<T, std::vector, std::allocator<T>>>(N);
    dss->set_diagonals(std::move(lower_diag), std::move(diagonal), std::move(upper_diag));
    dss->set_rhs(rhs);
    // get the solution:
    std::vector<T> solution(N);
    dss->solve(std::make_pair(lower_ptr, upper_ptr), solution);

    // exact value:
    auto exact = [](T x) { return x * (static_cast<T>(1.0) - x); };

    std::cout << "tp : FDM | Exact | Diff\n";
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        std::cout << "t_" << j << ": " << solution[j] << " |  " << exact(j * h) << " | " << (solution[j] - exact(j * h))
                  << '\n';
    }
}

void testDoubleSweepDirichletBC()
{
    std::cout << "==================================================\n";
    std::cout << "=========== Double Sweep (Dirichlet BC) ==========\n";
    std::cout << "==================================================\n";

    testBVPDoubleSweepDirichletBC<double>();
    testBVPDoubleSweepDirichletBC<float>();

    std::cout << "==================================================\n";
}

template <typename T> void testBVPDoubleSweepRobinBC()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_boundary::robin_boundary_1d;
    using lss_double_sweep_solver::double_sweep_solver;
    using lss_utility::range;

    /*

    Solve BVP:

    u''(t) = - 2,

    where

    t \in (0, 1)
    u(0) = 1 ,  u'(1) + u(1) = 0


    Exact solution is

    u(t) = -t*t + t + 1

    */
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

    // discretization:
    std::size_t N{100};
    // step size:
    T h = static_cast<T>(1.0) / static_cast<T>(N - 1);
    // upper,mid, and lower diagonal:
    std::vector<T> upper_diag(N, static_cast<T>(1.0));
    std::vector<T> diagonal(N, static_cast<T>(-2.0));
    std::vector<T> lower_diag(N, static_cast<T>(1.0));

    // right-hand side:
    std::vector<T> rhs(N, static_cast<T>(-2.0) * h * h);

    // boundary conditions:
    auto const &lower_ptr = std::make_shared<dirichlet_boundary_1d<T>>([](T t) { return 1.0; });
    auto const &upper_ptr = std::make_shared<robin_boundary_1d<T>>([](T t) { return 1.0; }, [](T t) { return 0.0; });

    auto dss = std::make_shared<double_sweep_solver<T, std::vector, std::allocator<T>>>(N);
    dss->set_diagonals(std::move(lower_diag), std::move(diagonal), std::move(upper_diag));
    dss->set_rhs(rhs);
    // get the solution:
    std::vector<T> solution(N);
    dss->solve(std::make_pair(lower_ptr, upper_ptr), solution);

    // exact value:
    auto exact = [](T x) { return (-x * x + x + static_cast<T>(1.0)); };

    std::cout << "tp : FDM | Exact | Abs Diff\n";
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        std::cout << "t_" << j << ": " << solution[j] << " |  " << exact(j * h) << " | " << (solution[j] - exact(j * h))
                  << '\n';
    }
}

void testDoubleSweepRobinBC()
{
    std::cout << "==================================================\n";
    std::cout << "=========== Double Sweep (Robin BC) ==============\n";
    std::cout << "==================================================\n";

    testBVPDoubleSweepRobinBC<double>();
    testBVPDoubleSweepRobinBC<float>();

    std::cout << "==================================================\n";
}

template <typename T> void testBVPDoubleSweepMixBC()
{
    using lss_boundary::neumann_boundary_1d;
    using lss_boundary::robin_boundary_1d;
    using lss_double_sweep_solver::double_sweep_solver;
    using lss_utility::range;

    /*

    Solve BVP:

    u''(t) = - 2,

    where

    t \in (0, 1)
    u'(0) - 1  = 0,  u'(1) + 2*u(1) = 0


    Exact solution is

    u(t) = -t*t + t + 0.5

    */
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

    // discretization:
    std::size_t N{1000};
    // step size:
    T h = static_cast<T>(1.0) / static_cast<T>(N - 1);
    // upper,mid, and lower diagonal:
    std::vector<T> upper_diag(N, static_cast<T>(1.0));
    std::vector<T> diagonal(N, static_cast<T>(-2.0));
    std::vector<T> lower_diag(N, static_cast<T>(1.0));

    // right-hand side:
    std::vector<T> rhs(N, static_cast<T>(-2.0) * h * h);

    // boundary conditions:
    auto const &lower_ptr = std::make_shared<neumann_boundary_1d<T>>([](T t) { return -1.0; });
    auto const &upper_ptr = std::make_shared<robin_boundary_1d<T>>([](T t) { return 2.0; }, [](T t) { return 0.0; });
    // constriuct space range:
    auto dss = std::make_shared<double_sweep_solver<T, std::vector, std::allocator<T>>>(N);
    dss->set_diagonals(std::move(lower_diag), std::move(diagonal), std::move(upper_diag));
    dss->set_rhs(rhs);
    // get the solution:
    std::vector<T> solution(N);
    dss->solve(std::make_pair(lower_ptr, upper_ptr), solution);

    // exact value:
    auto exact = [](T x) { return (-x * x + x + static_cast<T>(0.5)); };

    std::cout << "tp : FDM | Exact | Abs Diff\n";
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        std::cout << "t_" << j << ": " << solution[j] << " |  " << exact(j * h) << " | " << (solution[j] - exact(j * h))
                  << '\n';
    }
}

void testDoubleSweepMixBC()
{
    std::cout << "==================================================\n";
    std::cout << "=========== Double Sweep (Mix BC) ==============\n";
    std::cout << "==================================================\n";

    testBVPDoubleSweepMixBC<double>();
    testBVPDoubleSweepMixBC<float>();

    std::cout << "==================================================\n";
}

#endif ///_LSS_DOUBLE_SWEEP_SOLVER_T_HPP_
