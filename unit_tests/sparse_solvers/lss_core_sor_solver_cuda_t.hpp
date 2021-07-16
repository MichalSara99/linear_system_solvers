#pragma once
#if !defined(_LSS_CORE_SOR_SOLVER_CUDA_T)
#define _LSS_CORE_SOR_SOLVER_CUDA_T

#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include "sparse_solvers/general/core_sor_solver_cuda/lss_core_sor_solver_cuda.hpp"

template <typename T> void testSORSolverCUDA1()
{
    using lss_core_sor_solver::core_sor_solver_cuda;
    using lss_core_sor_solver::flat_matrix;

    std::cout << "===================================\n";
    std::cout << "Solving sparse system of equations:\n";
    std::cout << "Ax = b,\n";
    std::cout << "where\n\n";
    std::cout << "     [10, -1, 2, 0] \n";
    std::cout << " A = [-1, 11, -1, 3] \n";
    std::cout << "     [2, -1, 10, -1] \n";
    std::cout << "	 [0, 3, -1, 8]\n";
    std::cout << "\n";
    std::cout << "b = [6, 25, -11, 15]^T \n";
    std::cout << "\n";

    /*

    Solving for x in

    Ax = b,
    where


    A =
    [10, -1, 2, 0]
    [-1, 11, -1, 3]
    [2, -1, 10, -1]
    [0, 3, -1, 8]

    b = [6, 25, -11, 15]^T

    */
    // size of the system:
    int const m = 4;
    // first create and populate the sparse matrix:
    flat_matrix<T> fm(m, m);

    // populate the matrix:
    fm.emplace_back(0, 0, static_cast<T>(10.0));
    fm.emplace_back(0, 1, static_cast<T>(-1.));
    fm.emplace_back(0, 2, static_cast<T>(2.));

    fm.emplace_back(1, 0, static_cast<T>(-1.));
    fm.emplace_back(1, 1, static_cast<T>(11.));
    fm.emplace_back(1, 2, static_cast<T>(-1.));
    fm.emplace_back(1, 3, static_cast<T>(3.));

    fm.emplace_back(2, 0, static_cast<T>(2.0));
    fm.emplace_back(2, 1, static_cast<T>(-1.0));
    fm.emplace_back(2, 2, static_cast<T>(10.0));
    fm.emplace_back(2, 3, static_cast<T>(-1.));

    fm.emplace_back(3, 1, static_cast<T>(3.0));
    fm.emplace_back(3, 2, static_cast<T>(-1.0));
    fm.emplace_back(3, 3, static_cast<T>(8.));

    // lets use std::vector to populate vector b:
    std::vector<T> b = {static_cast<T>(6.), static_cast<T>(25.0), static_cast<T>(-11.), static_cast<T>(15.)};

    // create sparse solver on DEVICE:
    core_sor_solver_cuda<T> sor(m);

    // insert sparse matrix A and vector b:
    sor.set_flat_sparse_matrix(std::move(fm));
    sor.set_rhs(b);
    sor.set_omega(static_cast<T>(0.5));

    auto solution = sor.solve();
    std::cout << "Solution is: \n[";
    for (auto const &e : solution)
    {
        std::cout << e << " ";
    }
    std::cout << "]\n";
}

void testSORCUDA()
{
    std::cout << "============================================================\n";
    std::cout << "=============== Initialise Flat-Raw-Matrix =================\n";
    std::cout << "============================================================\n";

    testSORSolverCUDA1<float>();
    testSORSolverCUDA1<double>();

    std::cout << "============================================================\n";
}

template <typename T> void testBVPDirichletBCSORCUDATest()
{
    using lss_core_sor_solver::core_sor_solver_cuda;
    using lss_core_sor_solver::flat_matrix;

    std::cout << "=================================\n";
    std::cout << " Using SOR algorithm to \n";
    std::cout << " solve Boundary Value Problem: \n\n";
    std::cout << " type: " << typeid(T).name() << "\n\n";
    std::cout << " u''(t) = -2, \n\n";
    std::cout << " where\n\n";
    std::cout << " t in <0,1>,\n";
    std::cout << " u(0) = u(1) = 0\n\n";
    std::cout << "Exact solution is:\n\n";
    std::cout << " u(t) = t(1-t)\n";
    std::cout << "=================================\n";

    // discretization:
    // t_0,t_1,t_2,...,t_20
    int const N = 20;
    // step size:
    double h = 1.0 / static_cast<double>(N);
    // set number of columns and rows:
    // because we already know the boundary values
    // at t_0 = 0 and t_20 = 0:

    int const m = N - 1;
    // first create and populate the sparse matrix:
    flat_matrix<T> fm(m, m);

    // populate the matrix:
    fm.emplace_back(0, 0, static_cast<T>(-2.0));
    fm.emplace_back(0, 1, static_cast<T>(1.0));
    for (std::size_t t = 1; t < m - 1; ++t)
    {
        fm.emplace_back(t, t - 1, static_cast<T>(1.0));
        fm.emplace_back(t, t, static_cast<T>(-2.0));
        fm.emplace_back(t, t + 1, static_cast<T>(1.0));
    }
    fm.emplace_back(m - 1, m - 2, static_cast<T>(1.0));
    fm.emplace_back(m - 1, m - 1, static_cast<T>(-2.0));

    // lets use std::vector to populate vector b:
    std::vector<T> b(m, static_cast<T>(-2.0 * h * h));
    // set the Dirichlet boundary conditions:
    T left = 0.0;
    T right = 0.0;
    b[0] = b[0] - left;
    b[b.size() - 1] = b[b.size() - 1] - right;

    // create sor  solver:
    core_sor_solver_cuda<T> sor(m);

    // insert sparse matrix A and vector b:
    sor.set_flat_sparse_matrix(std::move(fm));
    sor.set_rhs(b);
    sor.set_omega(static_cast<T>(0.5));

    std::vector<T> solution(m);
    sor.solve(solution);

    // exact value:
    auto exact = [](T x) { return x * (static_cast<T>(1.0) - x); };

    std::cout << "tp : FDM | Exact\n";
    std::cout << "t_" << 0 << ": " << left << " |  " << exact(0) << '\n';
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        std::cout << "t_" << j + 1 << ": " << solution[j] << " |  " << exact((j + 1) * h) << '\n';
    }
    std::cout << "t_" << N << ": " << right << " |  " << exact(N * h) << '\n';
}

void testBVPDirichletBCSORCUDA()
{
    std::cout << "============================================================\n";
    std::cout << "=========================== BVP ============================\n";
    std::cout << "============================================================\n";

    testBVPDirichletBCSORCUDATest<float>();
    testBVPDirichletBCSORCUDATest<double>();

    std::cout << "============================================================\n";
}

#endif ///_LSS_CORE_SOR_SOLVER_CUDA_T
