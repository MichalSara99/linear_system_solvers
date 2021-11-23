#pragma once
#if !defined(_LSS_THOMAS_LU_SOLVER_T_HPP_)
#define _LSS_THOMAS_LU_SOLVER_T_HPP_

#include <vector>

#include "boundaries/lss_dirichlet_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include "sparse_solvers/tridiagonal/thomas_lu_solver/lss_thomas_lu_solver.hpp"

template <typename T> void testBVPThomasLUDirichletBC()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_boundary::dirichlet_boundary_2d;
    using lss_thomas_lu_solver::thomas_lu_solver;
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

    auto dss = std::make_shared<thomas_lu_solver<T, std::vector, std::allocator<T>>>(N);
    dss->set_diagonals(std::move(lower_diag), std::move(diagonal), std::move(upper_diag));
    dss->set_rhs(rhs);
    // get the solution:
    std::vector<T> solution(N);
    dss->solve(std::make_pair(lower_ptr, upper_ptr), solution);

    // exact value:
    auto exact = [](T x) { return x * (static_cast<T>(1.0) - x); };

    std::cout << "tp : FDM | Exact | Abs Diff\n";
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        std::cout << "t_" << j << ": " << solution[j] << " |  " << exact(j * h) << " | " << (solution[j] - exact(j * h))
                  << '\n';
    }
}

void testThomasLUDirichletBC()
{
    std::cout << "==================================================\n";
    std::cout << "======= Thomas LU Solver (Dirichlet BC) ==========\n";
    std::cout << "==================================================\n";

    testBVPThomasLUDirichletBC<double>();
    testBVPThomasLUDirichletBC<float>();

    std::cout << "==================================================\n";
}

template <typename T> void testBVPThomasLURobinBC()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_boundary::robin_boundary_1d;
    using lss_thomas_lu_solver::thomas_lu_solver;
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

    auto dss = std::make_shared<thomas_lu_solver<T, std::vector, std::allocator<T>>>(N);
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

void testThomasLURobinBC()
{
    std::cout << "==================================================\n";
    std::cout << "======= Thomas LU Solver (Robin BC) ==============\n";
    std::cout << "==================================================\n";

    testBVPThomasLURobinBC<double>();
    testBVPThomasLURobinBC<float>();

    std::cout << "==================================================\n";
}

template <typename T> void testBVPThomasLUMixBC()
{
    using lss_boundary::neumann_boundary_1d;
    using lss_boundary::robin_boundary_1d;
    using lss_thomas_lu_solver::thomas_lu_solver;
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

    auto dss = std::make_shared<thomas_lu_solver<T, std::vector, std::allocator<T>>>(N);
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

void testThomasLUMixBC()
{
    std::cout << "==================================================\n";
    std::cout << "========= Thomas LU Solver (Mix BC) ==============\n";
    std::cout << "==================================================\n";

    testBVPThomasLUMixBC<double>();
    testBVPThomasLUMixBC<float>();

    std::cout << "==================================================\n";
}

#endif ///_LSS_THOMAS_LU_SOLVER_T_HPP_
