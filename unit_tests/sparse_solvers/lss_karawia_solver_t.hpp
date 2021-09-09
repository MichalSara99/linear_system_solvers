#pragma once
#if !defined(_LSS_KARAWIA_SOLVER_T_HPP_)
#define _LSS_KARAWIA_SOLVER_T_HPP_

#include <vector>

#include "boundaries/lss_dirichlet_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include "sparse_solvers/pentadiagonal/karawia_solver/lss_karawia_solver.hpp"

template <typename T> void testBVPKarawiaDirichletBC()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_karawia_solver::karawia_solver;
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
    std::vector<T> uppest_diag(N, static_cast<T>(-1.0));
    std::vector<T> upper_diag(N, static_cast<T>(16.0));
    std::vector<T> diagonal(N, static_cast<T>(-30.0));
    std::vector<T> lower_diag(N, static_cast<T>(16.0));
    std::vector<T> lowest_diag(N, static_cast<T>(-1.0));

    // right-hand side:
    std::vector<T> rhs(N, static_cast<T>(-2.0) * 12.0 * h * h);

    // exact value:
    auto exact = [](T x) { return x * (static_cast<T>(1.0) - x); };

    // boundary conditions:
    auto const &lower_ptr = std::make_shared<dirichlet_boundary_1d<T>>([](T t) { return 0.0; });
    auto const &upper_ptr = std::make_shared<dirichlet_boundary_1d<T>>([](T t) { return 0.0; });
    auto const &other_lower_ptr = std::make_shared<dirichlet_boundary_1d<T>>([&](T t) { return exact(h); });
    auto const &other_upper_ptr = std::make_shared<dirichlet_boundary_1d<T>>([&](T t) { return exact(1.0 - h); });
    // constriuct space range:
    range<T> space_range(0.0, 1.0);
    auto dss = std::make_shared<karawia_solver<T, std::vector, std::allocator<T>>>(space_range, N);
    dss->set_diagonals(std::move(lowest_diag), std::move(lower_diag), std::move(diagonal), std::move(upper_diag),
                       std::move(uppest_diag));
    dss->set_rhs(rhs);
    // get the solution:
    std::vector<T> solution(N);
    //
    auto const &boundary = std::make_pair(lower_ptr, upper_ptr);
    auto const &other_boundary = std::make_pair(other_lower_ptr, other_upper_ptr);
    dss->solve(boundary, other_boundary, solution);

    std::cout << "tp : FDM | Exact | Diff\n";
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        std::cout << "t_" << j << ": " << solution[j] << " |  " << exact(j * h) << " | " << (solution[j] - exact(j * h))
                  << '\n';
    }
}

void testKarawiaDirichletBC()
{
    std::cout << "==================================================\n";
    std::cout << "============== Karawia (Dirichlet BC) ============\n";
    std::cout << "==================================================\n";

    testBVPKarawiaDirichletBC<double>();
    testBVPKarawiaDirichletBC<float>();

    std::cout << "==================================================\n";
}

#endif ///_LSS_KARAWIA_SOLVER_T_HPP_
