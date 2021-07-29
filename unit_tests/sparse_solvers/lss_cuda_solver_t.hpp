#pragma once
#if !defined(_LSS_CUDA_SOLVER_T_HPP_)
#define _LSS_CUDA_SOLVER_T_HPP_

#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include "pde_solvers/lss_discretization.hpp"
#include "sparse_solvers/general/core_cuda_solver/lss_core_cuda_solver_policy.hpp"
#include "sparse_solvers/tridiagonal/cuda_solver/lss_cuda_solver.hpp"

template <typename T> void testBVPCUDADirichletBC()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_cuda_solver::cuda_solver;
    using lss_enumerations::memory_space_enum;
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
    // constriuct space range:
    range<T> space_range(0.0, 1.0);
    auto dss =
        std::make_shared<cuda_solver<memory_space_enum::Device, T, std::vector, std::allocator<T>>>(space_range, N);
    dss->set_diagonals(std::move(lower_diag), std::move(diagonal), std::move(upper_diag));
    dss->set_rhs(rhs);
    // get the solution:
    std::vector<T> solution(N);
    dss->solve(std::make_pair(lower_ptr, upper_ptr), solution);

    // exact value:
    auto exact = [](T x) { return x * (static_cast<T>(1.0) - x); };

    std::cout << "tp : FDM | Exact\n";
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        std::cout << "t_" << j << ": " << solution[j] << " |  " << exact(j * h) << '\n';
    }
}

void testCUDADirichletBC()
{
    std::cout << "==================================================\n";
    std::cout << "=========== CUDA solver (Dirichlet BC) ===========\n";
    std::cout << "==================================================\n";

    testBVPCUDADirichletBC<double>();
    testBVPCUDADirichletBC<float>();

    std::cout << "==================================================\n";
}

template <typename T> void testBVPCUDARobinBC()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_boundary::robin_boundary_1d;
    using lss_cuda_solver::cuda_solver;
    using lss_enumerations::memory_space_enum;
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
    // constriuct space range:
    range<T> space_range(0.0, 1.0);
    auto dss =
        std::make_shared<cuda_solver<memory_space_enum::Device, T, std::vector, std::allocator<T>>>(space_range, N);
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

void testCUDARobinBC()
{
    std::cout << "==================================================\n";
    std::cout << "============ CUDA solver (Robin BC) ==============\n";
    std::cout << "==================================================\n";

    testBVPCUDARobinBC<double>();
    testBVPCUDARobinBC<float>();

    std::cout << "==================================================\n";
}

template <typename T> void testBVPCUDADirichletNeumannBC()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_boundary::neumann_boundary_1d;
    using lss_cuda_solver::cuda_solver;
    using lss_enumerations::memory_space_enum;
    using lss_pde_solvers::discretization_1d;
    using lss_utility::range;

    typedef discretization_1d<T, std::vector, std::allocator<T>> d_1d;

    /*

    Solve BVP:

    u''(t) =  6 * t,

    where

    t \in (0, 2)
    u(0) = 1 ,  u'(2) = 0


    Exact solution is

    u(t) = t*t*t-12*t+1

    */
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

    // discretization:
    std::size_t N{100};
    // constriuct space range:
    range<T> space_range(0.0, 2.0);
    // step size:
    T h = space_range.spread() / static_cast<T>(N - 1);
    // upper,mid, and lower diagonal:
    std::vector<T> upper_diag(N, static_cast<T>(1.0));
    std::vector<T> diagonal(N, static_cast<T>(-2.0));
    std::vector<T> lower_diag(N, static_cast<T>(1.0));
    // right-hand side:
    auto const &rhs_fun = [=](T t) { return static_cast<T>(h * h * 6.0) * t; };
    std::vector<T> rhs(N, T{});
    d_1d::of_function(space_range.lower(), h, rhs_fun, rhs);

    // boundary conditions:
    auto const &lower_ptr = std::make_shared<dirichlet_boundary_1d<T>>([](T t) { return 1.0; });
    auto const &upper_ptr = std::make_shared<neumann_boundary_1d<T>>([](T t) { return 0.0; });

    auto dss =
        std::make_shared<cuda_solver<memory_space_enum::Device, T, std::vector, std::allocator<T>>>(space_range, N);
    dss->set_diagonals(std::move(lower_diag), std::move(diagonal), std::move(upper_diag));
    dss->set_rhs(rhs);
    // get the solution:
    std::vector<T> solution(N);
    dss->solve(std::make_pair(lower_ptr, upper_ptr), solution);

    // exact value:
    auto exact = [](T x) { return (x * x * x - 12.0 * x + 1.0); };

    std::cout << "tp : FDM | Exact | Abs Diff\n";
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        std::cout << "t_" << j << ": " << solution[j] << " |  " << exact(j * h) << " | " << (solution[j] - exact(j * h))
                  << '\n';
    }
}

void testCUDADirichletNeumannBC()
{
    std::cout << "==================================================\n";
    std::cout << "========== CUDA solver (Dir-Neu BC) ==============\n";
    std::cout << "==================================================\n";

    testBVPCUDADirichletNeumannBC<double>();
    testBVPCUDADirichletNeumannBC<float>();

    std::cout << "==================================================\n";
}

template <typename T> void testBVPCUDANeumannDirichletBC()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_boundary::neumann_boundary_1d;
    using lss_cuda_solver::cuda_solver;
    using lss_enumerations::memory_space_enum;
    using lss_pde_solvers::discretization_1d;
    using lss_utility::range;

    typedef discretization_1d<T, std::vector, std::allocator<T>> d_1d;

    /*

    Solve BVP:

    u''(t) =  6 * t,

    where

    t \in (0, 2)
    u(0) = 1 ,  u'(2) = 0


    Exact solution is

    u(t) = t*t*t-12*t+1

    */
    std::cout << "=================================\n";
    std::cout << "Solving Boundary-value problem: \n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " u''(t) = 6*t, \n\n";
    std::cout << " where\n\n";
    std::cout << " t in <0,2>,\n";
    std::cout << " u'(0) = 1 \n";
    std::cout << " u(2) = 0\n\n";
    std::cout << "Exact solution is:\n\n";
    std::cout << " u(t) = t*t*t + t - 10\n";
    std::cout << "=================================\n";

    // discretization:
    std::size_t N{100};
    // constriuct space range:
    range<T> space_range(0.0, 2.0);
    // step size:
    T h = space_range.spread() / static_cast<T>(N - 1);
    // upper,mid, and lower diagonal:
    std::vector<T> upper_diag(N, static_cast<T>(1.0));
    std::vector<T> diagonal(N, static_cast<T>(-2.0));
    std::vector<T> lower_diag(N, static_cast<T>(1.0));
    // right-hand side:
    auto const &rhs_fun = [=](T t) { return static_cast<T>(h * h * 6.0) * t; };
    std::vector<T> rhs(N, T{});
    d_1d::of_function(space_range.lower(), h, rhs_fun, rhs);

    // boundary conditions:
    auto const &upper_ptr = std::make_shared<dirichlet_boundary_1d<T>>([](T t) { return 0.0; });
    auto const &lower_ptr = std::make_shared<neumann_boundary_1d<T>>([](T t) { return -1.0; });

    auto dss =
        std::make_shared<cuda_solver<memory_space_enum::Device, T, std::vector, std::allocator<T>>>(space_range, N);
    dss->set_diagonals(std::move(lower_diag), std::move(diagonal), std::move(upper_diag));
    dss->set_rhs(rhs);
    // get the solution:
    std::vector<T> solution(N);
    dss->solve(std::make_pair(lower_ptr, upper_ptr), solution);

    // exact value:
    auto exact = [](T x) { return (x * x * x + x - 10.0); };

    std::cout << "tp : FDM | Exact | Abs Diff\n";
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        std::cout << "t_" << j << ": " << solution[j] << " |  " << exact(j * h) << " | " << (solution[j] - exact(j * h))
                  << '\n';
    }
}

void testCUDANeumannDirichletBC()
{
    std::cout << "==================================================\n";
    std::cout << "========== CUDA solver (Neu-Dir BC) ==============\n";
    std::cout << "==================================================\n";

    testBVPCUDANeumannDirichletBC<double>();
    testBVPCUDANeumannDirichletBC<float>();

    std::cout << "==================================================\n";
}

template <typename T> void testBVPCUDANeumannRobinBC()
{
    using lss_boundary::neumann_boundary_1d;
    using lss_boundary::robin_boundary_1d;
    using lss_cuda_solver::cuda_solver;
    using lss_enumerations::memory_space_enum;
    using lss_pde_solvers::discretization_1d;
    using lss_utility::range;

    typedef discretization_1d<T, std::vector, std::allocator<T>> d_1d;

    /*

    Solve BVP:

    u''(t) =  6 * t,

    where

    t \in (0, 2)

    u'(0) = 0
    u'(2) + 2 * u(2) = 0 ,


    Exact solution is

    u(t) = t*t*t - 14

    */
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

    // discretization:
    std::size_t N{100};
    // constriuct space range:
    range<T> space_range(0.0, 2.0);
    // step size:
    T h = space_range.spread() / static_cast<T>(N - 1);
    // upper,mid, and lower diagonal:
    std::vector<T> upper_diag(N, static_cast<T>(1.0));
    std::vector<T> diagonal(N, static_cast<T>(-2.0));
    std::vector<T> lower_diag(N, static_cast<T>(1.0));
    // right-hand side:
    auto const &rhs_fun = [=](T t) { return static_cast<T>(h * h * 6.0) * t; };
    std::vector<T> rhs(N, T{});
    d_1d::of_function(space_range.lower(), h, rhs_fun, rhs);

    // boundary conditions:
    auto const &lower_ptr = std::make_shared<neumann_boundary_1d<T>>([](T t) { return 0.0; });
    auto const &upper_ptr = std::make_shared<robin_boundary_1d<T>>([](T t) { return 2.0; }, [](T t) { return 0.0; });

    auto dss =
        std::make_shared<cuda_solver<memory_space_enum::Device, T, std::vector, std::allocator<T>>>(space_range, N);
    dss->set_diagonals(std::move(lower_diag), std::move(diagonal), std::move(upper_diag));
    dss->set_rhs(rhs);
    // get the solution:
    std::vector<T> solution(N);
    dss->solve(std::make_pair(lower_ptr, upper_ptr), solution);

    // exact value:
    auto exact = [](T x) { return (x * x * x - static_cast<T>(14.0)); };

    std::cout << "tp : FDM | Exact | Abs Diff\n";
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        std::cout << "t_" << j << ": " << solution[j] << " |  " << exact(j * h) << " | " << (solution[j] - exact(j * h))
                  << '\n';
    }
}

void testCUDANeumannRobinBC()
{
    std::cout << "==================================================\n";
    std::cout << "=============  CUDA solver (Neu-Rob BC) ==========\n";
    std::cout << "==================================================\n";

    testBVPCUDANeumannRobinBC<double>();
    testBVPCUDANeumannRobinBC<float>();

    std::cout << "==================================================\n";
}

template <typename T> void testBVPCUDAMixBC()
{
    using lss_boundary::neumann_boundary_1d;
    using lss_boundary::robin_boundary_1d;
    using lss_cuda_solver::cuda_solver;
    using lss_enumerations::memory_space_enum;
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
    auto const &lower_ptr = std::make_shared<neumann_boundary_1d<T>>([](T t) { return -1.0; });
    auto const &upper_ptr = std::make_shared<robin_boundary_1d<T>>([](T t) { return 2.0; }, [](T t) { return 0.0; });
    // constriuct space range:
    range<T> space_range(0.0, 1.0);
    auto dss =
        std::make_shared<cuda_solver<memory_space_enum::Device, T, std::vector, std::allocator<T>>>(space_range, N);
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

void testCUDAMixBC()
{
    std::cout << "==================================================\n";
    std::cout << "============= CUDA Solver (Mix BC) ===============\n";
    std::cout << "==================================================\n";

    testBVPCUDAMixBC<double>();
    testBVPCUDAMixBC<float>();

    std::cout << "==================================================\n";
}
#endif ///_LSS_CUDA_SOLVER_T_HPP_
