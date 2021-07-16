#pragma once
#if !defined(_LSS_CORE_CUDA_SOLVER_T_HPP_)
#define _LSS_CORE_CUDA_SOLVER_T_HPP_

#include "sparse_solvers/general/core_cuda_solver/lss_core_cuda_solver.hpp"

template <typename T> void deviceSparseQRTest()
{
    using lss_core_cuda_solver::factorization_enum;
    using lss_core_cuda_solver::flat_matrix;
    using lss_core_cuda_solver::memory_space_enum;
    using lss_core_cuda_solver::real_sparse_solver_cuda;

    std::cout << "===================================\n";
    std::cout << "Solving sparse system of equations:\n";
    std::cout << "Ax = b,\n";
    std::cout << "where\n\n";
    std::cout << "    [  1.0  2.0  0.0  0.0  0.0  0.0 ] \n";
    std::cout << "    [  3.0  4.0  5.0  0.0  0.0  0.0 ] \n";
    std::cout << "    [  0.0  6.0  7.0  8.0  0.0  0.0 ] \n";
    std::cout << "A = [  0.0  0.0  9.0 10.0 11.0  0.0 ] \n";
    std::cout << "    [  0.0  0.0  0.0 12.0 13.0 14.0 ] \n";
    std::cout << "    [  0.0  0.0  0.0  0.0 15.0 16.0 ] \n";
    std::cout << "\n";
    std::cout << "b = [  0.0  2.0  4.0  6.0  8.0  10.0 ]^T \n";
    std::cout << "\n";

    /*

    Solving for x in

    Ax = b,
    where


    A =
    [  1.0  2.0  0.0  0.0  0.0  0.0 ]
    [  3.0  4.0  5.0  0.0  0.0  0.0 ]
    [  0.0  6.0  7.0  8.0  0.0  0.0 ]
    [  0.0  0.0  9.0 10.0 11.0  0.0 ]
    [  0.0  0.0  0.0 12.0 13.0 14.0 ]
    [  0.0  0.0  0.0  0.0 15.0 16.0 ]

    b = [0.0,2.0,4.0,6.0,8.0,10.0]^T

    */

    // size of the system:
    int const m = 6;
    // first create and populate the sparse matrix:
    flat_matrix<T> fsm(m, m);
    // populate the matrix:
    fsm.emplace_back(0, 0, static_cast<T>(1.0));
    fsm.emplace_back(0, 1, static_cast<T>(2.0));
    fsm.emplace_back(1, 0, static_cast<T>(3.0));
    fsm.emplace_back(1, 1, static_cast<T>(4.0));
    fsm.emplace_back(1, 2, static_cast<T>(5.0));
    fsm.emplace_back(2, 1, static_cast<T>(6.0));
    fsm.emplace_back(2, 2, static_cast<T>(7.0));
    fsm.emplace_back(2, 3, static_cast<T>(8.0));
    fsm.emplace_back(3, 2, static_cast<T>(9.0));
    fsm.emplace_back(3, 3, static_cast<T>(10.0));
    fsm.emplace_back(3, 4, static_cast<T>(11.0));
    fsm.emplace_back(4, 3, static_cast<T>(12.0));
    fsm.emplace_back(4, 4, static_cast<T>(13.0));
    fsm.emplace_back(4, 5, static_cast<T>(14.0));
    fsm.emplace_back(5, 4, static_cast<T>(15.0));
    fsm.emplace_back(5, 5, static_cast<T>(16.0));

    // lets use std::vector to populate vector b:
    std::vector<T> b = {0.0, 2.0, 4.0, 6.0, 8.0, 10.0};

    // create sparse solver on DEVICE:
    real_sparse_solver_cuda<memory_space_enum::Device, T, std::vector, std::allocator<T>> rss(m);

    // because we used default cstor we need to call initialize
    rss.initialize(m);

    // insert sparse matrix A and vector b:
    rss.set_flat_sparse_matrix(std::move(fsm));
    rss.set_rhs(b);

    auto solution = rss.solve();
    std::cout << "Solution is: \n[";
    for (auto const &e : solution)
    {
        std::cout << e << " ";
    }
    std::cout << "]\n";
}

template <typename T> void deviceSparseQRPointerTest()
{
    using lss_core_cuda_solver::flat_matrix;
    using lss_core_cuda_solver::memory_space_enum;
    using lss_core_cuda_solver::real_sparse_solver_cuda;
    using lss_core_cuda_solver_policy::sparse_solver_device_qr;

    std::cout << "===================================\n";
    std::cout << "Solving sparse system of equations:\n";
    std::cout << "Ax = b,\n";
    std::cout << "where\n\n";
    std::cout << "    [  1.0  2.0  0.0  0.0  0.0  0.0 ] \n";
    std::cout << "    [  3.0  4.0  5.0  0.0  0.0  0.0 ] \n";
    std::cout << "    [  0.0  6.0  7.0  8.0  0.0  0.0 ] \n";
    std::cout << "A = [  0.0  0.0  9.0 10.0 11.0  0.0 ] \n";
    std::cout << "    [  0.0  0.0  0.0 12.0 13.0 14.0 ] \n";
    std::cout << "    [  0.0  0.0  0.0  0.0 15.0 16.0 ] \n";
    std::cout << "\n";
    std::cout << "b = [  0.0  2.0  4.0  6.0  8.0  10.0 ]^T \n";
    std::cout << "\n";

    /*

    Solving for x in

    Ax = b,
    where


    A =
    [  1.0  2.0  0.0  0.0  0.0  0.0 ]
    [  3.0  4.0  5.0  0.0  0.0  0.0 ]
    [  0.0  6.0  7.0  8.0  0.0  0.0 ]
    [  0.0  0.0  9.0 10.0 11.0  0.0 ]
    [  0.0  0.0  0.0 12.0 13.0 14.0 ]
    [  0.0  0.0  0.0  0.0 15.0 16.0 ]

    b = [0.0,2.0,4.0,6.0,8.0,10.0]^T

    */

    // size of the system:
    int const m = 6;
    // first create and populate the sparse matrix:
    flat_matrix<T> fsm(m, m);
    // populate the matrix:
    fsm.emplace_back(0, 0, static_cast<T>(1.0));
    fsm.emplace_back(0, 1, static_cast<T>(2.0));
    fsm.emplace_back(1, 0, static_cast<T>(3.0));
    fsm.emplace_back(1, 1, static_cast<T>(4.0));
    fsm.emplace_back(1, 2, static_cast<T>(5.0));
    fsm.emplace_back(2, 1, static_cast<T>(6.0));
    fsm.emplace_back(2, 2, static_cast<T>(7.0));
    fsm.emplace_back(2, 3, static_cast<T>(8.0));
    fsm.emplace_back(3, 2, static_cast<T>(9.0));
    fsm.emplace_back(3, 3, static_cast<T>(10.0));
    fsm.emplace_back(3, 4, static_cast<T>(11.0));
    fsm.emplace_back(4, 3, static_cast<T>(12.0));
    fsm.emplace_back(4, 4, static_cast<T>(13.0));
    fsm.emplace_back(4, 5, static_cast<T>(14.0));
    fsm.emplace_back(5, 4, static_cast<T>(15.0));
    fsm.emplace_back(5, 5, static_cast<T>(16.0));

    // lets use std::vector to populate vector b:
    std::vector<T> b = {0.0, 2.0, 4.0, 6.0, 8.0, 10.0};

    // create sparse solver on DEVICE:
    real_sparse_solver_cuda<memory_space_enum::Device, T, std::vector, std::allocator<T>> rss(m);

    // because we used default cstor we need to call initialize
    rss.initialize(m);

    // insert sparse matrix A and vector b:
    rss.set_flat_sparse_matrix(std::move(fsm));
    rss.set_rhs(b);

    std::vector<T> solution(m);
    rss.solve(solution);

    std::cout << "Solution is: \n[";
    for (std::size_t t = 0; t < m; ++t)
    {
        std::cout << solution[t] << " ";
    }
    std::cout << "]\n";
}

void deviceSparseQRTest()
{
    std::cout << "==================================================\n";
    std::cout << "=========== Sparse QR Solver - DEVICE ============\n";
    std::cout << "==================================================\n";

    deviceSparseQRTest<double>();
    deviceSparseQRTest<float>();
    deviceSparseQRPointerTest<double>();
    deviceSparseQRPointerTest<float>();

    std::cout << "==================================================\n";
}

template <typename T> void hostSparseQRTest()
{
    using lss_core_cuda_solver::flat_matrix;
    using lss_core_cuda_solver::memory_space_enum;
    using lss_core_cuda_solver::real_sparse_solver_cuda;

    std::cout << "===================================\n";
    std::cout << "Solving sparse system of equations:\n";
    std::cout << "Ax = b,\n";
    std::cout << "where\n\n";
    std::cout << "    [  1.0  2.0  0.0  0.0  0.0  0.0 ] \n";
    std::cout << "    [  3.0  4.0  5.0  0.0  0.0  0.0 ] \n";
    std::cout << "    [  0.0  6.0  7.0  8.0  0.0  0.0 ] \n";
    std::cout << "A = [  0.0  0.0  9.0 10.0 11.0  0.0 ] \n";
    std::cout << "    [  0.0  0.0  0.0 12.0 13.0 14.0 ] \n";
    std::cout << "    [  0.0  0.0  0.0  0.0 15.0 16.0 ] \n";
    std::cout << "\n";
    std::cout << "b = [  0.0  2.0  4.0  6.0  8.0  10.0 ]^T \n";
    std::cout << "\n";

    /*

    Solving for x in

    Ax = b,
    where


    A =
    [  1.0  2.0  0.0  0.0  0.0  0.0 ]
    [  3.0  4.0  5.0  0.0  0.0  0.0 ]
    [  0.0  6.0  7.0  8.0  0.0  0.0 ]
    [  0.0  0.0  9.0 10.0 11.0  0.0 ]
    [  0.0  0.0  0.0 12.0 13.0 14.0 ]
    [  0.0  0.0  0.0  0.0 15.0 16.0 ]

    b = [0.0,2.0,4.0,6.0,8.0,10.0]^T

    */

    // size of the system:
    int const m = 6;
    // first create and populate the sparse matrix:
    flat_matrix<T> fsm(m, m);
    // populate the matrix:
    fsm.emplace_back(0, 0, static_cast<T>(1.0));
    fsm.emplace_back(0, 1, static_cast<T>(2.0));
    fsm.emplace_back(1, 0, static_cast<T>(3.0));
    fsm.emplace_back(1, 1, static_cast<T>(4.0));
    fsm.emplace_back(1, 2, static_cast<T>(5.0));
    fsm.emplace_back(2, 1, static_cast<T>(6.0));
    fsm.emplace_back(2, 2, static_cast<T>(7.0));
    fsm.emplace_back(2, 3, static_cast<T>(8.0));
    fsm.emplace_back(3, 2, static_cast<T>(9.0));
    fsm.emplace_back(3, 3, static_cast<T>(10.0));
    fsm.emplace_back(3, 4, static_cast<T>(11.0));
    fsm.emplace_back(4, 3, static_cast<T>(12.0));
    fsm.emplace_back(4, 4, static_cast<T>(13.0));
    fsm.emplace_back(4, 5, static_cast<T>(14.0));
    fsm.emplace_back(5, 4, static_cast<T>(15.0));
    fsm.emplace_back(5, 5, static_cast<T>(16.0));

    // lets use std::vector to populate vector b:
    std::vector<T> b = {0.0, 2.0, 4.0, 6.0, 8.0, 10.0};

    // create sparse solver on DEVICE:
    real_sparse_solver_cuda<memory_space_enum::Host, T, std::vector, std::allocator<T>> rss(m);

    // because we used default cstor we need to call initialize
    rss.initialize(m);

    // insert sparse matrix A and vector b:
    rss.set_flat_sparse_matrix(std::move(fsm));
    rss.set_rhs(b);

    auto solution = rss.solve();
    std::cout << "Solution is: \n[";
    for (auto const &e : solution)
    {
        std::cout << e << " ";
    }
    std::cout << "]\n";
}

template <typename T> void hostSparseQRPointerTest()
{
    using lss_core_cuda_solver::flat_matrix;
    using lss_core_cuda_solver::memory_space_enum;
    using lss_core_cuda_solver::real_sparse_solver_cuda;

    std::cout << "===================================\n";
    std::cout << "Solving sparse system of equations:\n";
    std::cout << "Ax = b,\n";
    std::cout << "where\n\n";
    std::cout << "    [  1.0  2.0  0.0  0.0  0.0  0.0 ] \n";
    std::cout << "    [  3.0  4.0  5.0  0.0  0.0  0.0 ] \n";
    std::cout << "    [  0.0  6.0  7.0  8.0  0.0  0.0 ] \n";
    std::cout << "A = [  0.0  0.0  9.0 10.0 11.0  0.0 ] \n";
    std::cout << "    [  0.0  0.0  0.0 12.0 13.0 14.0 ] \n";
    std::cout << "    [  0.0  0.0  0.0  0.0 15.0 16.0 ] \n";
    std::cout << "\n";
    std::cout << "b = [  0.0  2.0  4.0  6.0  8.0  10.0 ]^T \n";
    std::cout << "\n";

    /*

    Solving for x in

    Ax = b,
    where


    A =
    [  1.0  2.0  0.0  0.0  0.0  0.0 ]
    [  3.0  4.0  5.0  0.0  0.0  0.0 ]
    [  0.0  6.0  7.0  8.0  0.0  0.0 ]
    [  0.0  0.0  9.0 10.0 11.0  0.0 ]
    [  0.0  0.0  0.0 12.0 13.0 14.0 ]
    [  0.0  0.0  0.0  0.0 15.0 16.0 ]

    b = [0.0,2.0,4.0,6.0,8.0,10.0]^T

    */

    // size of the system:
    int const m = 6;

    // first create and populate the sparse matrix:
    flat_matrix<T> fsm(m, m);
    // populate the matrix:
    fsm.emplace_back(0, 0, static_cast<T>(1.0));
    fsm.emplace_back(0, 1, static_cast<T>(2.0));
    fsm.emplace_back(1, 0, static_cast<T>(3.0));
    fsm.emplace_back(1, 1, static_cast<T>(4.0));
    fsm.emplace_back(1, 2, static_cast<T>(5.0));
    fsm.emplace_back(2, 1, static_cast<T>(6.0));
    fsm.emplace_back(2, 2, static_cast<T>(7.0));
    fsm.emplace_back(2, 3, static_cast<T>(8.0));
    fsm.emplace_back(3, 2, static_cast<T>(9.0));
    fsm.emplace_back(3, 3, static_cast<T>(10.0));
    fsm.emplace_back(3, 4, static_cast<T>(11.0));
    fsm.emplace_back(4, 3, static_cast<T>(12.0));
    fsm.emplace_back(4, 4, static_cast<T>(13.0));
    fsm.emplace_back(4, 5, static_cast<T>(14.0));
    fsm.emplace_back(5, 4, static_cast<T>(15.0));
    fsm.emplace_back(5, 5, static_cast<T>(16.0));

    // lets use std::vector to populate vector b:
    std::vector<T> b = {0.0, 2.0, 4.0, 6.0, 8.0, 10.0};

    // create sparse solver on DEVICE:
    real_sparse_solver_cuda<memory_space_enum::Host, T, std::vector, std::allocator<T>> rss(m);

    // because we used default cstor we need to call initialize
    rss.initialize(m);

    // insert sparse matrix A and vector b:
    rss.set_flat_sparse_matrix(std::move(fsm));
    rss.set_rhs(b);

    auto solution = rss.solve();
    std::cout << "Solution is: \n[";
    for (auto const &e : solution)
    {
        std::cout << e << " ";
    }
    std::cout << "]\n";
}

void hostSparseQRTest()
{
    std::cout << "==================================================\n";
    std::cout << "=========== Sparse QR Solver - HOST ==============\n";
    std::cout << "==================================================\n";

    hostSparseQRTest<double>();
    hostSparseQRTest<float>();
    hostSparseQRPointerTest<double>();
    hostSparseQRPointerTest<float>();

    std::cout << "==================================================\n";
}

template <typename T> void hostBVPDirichletBCQRTest()
{
    using lss_core_cuda_solver::flat_matrix;
    using lss_core_cuda_solver::memory_space_enum;
    using lss_core_cuda_solver::real_sparse_solver_cuda;

    std::cout << "=================================\n";
    std::cout << " Using QR decomposition to \n";
    std::cout << " solve Boundary Value Problem: \n\n";
    std::cout << " type: double					\n\n";
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
    const T h = static_cast<T>(1.0) / static_cast<T>(N);
    // set number of columns and rows:
    // because we already know the boundary values
    // at t_0 = 0 and t_20 = 0:
    int const m = N - 1;
    // first create and populate the sparse matrix:
    flat_matrix<T> fsm(m, m);
    // populate the matrix:
    fsm.emplace_back(0, 0, static_cast<T>(-2.0));
    fsm.emplace_back(0, 1, static_cast<T>(1.0));
    for (std::size_t t = 1; t < m - 1; ++t)
    {
        fsm.emplace_back(t, t - 1, static_cast<T>(1.0));
        fsm.emplace_back(t, t, static_cast<T>(-2.0));
        fsm.emplace_back(t, t + 1, static_cast<T>(1.0));
    }
    fsm.emplace_back(m - 1, m - 2, static_cast<T>(1.0));
    fsm.emplace_back(m - 1, m - 1, static_cast<T>(-2.0));

    // lets use std::vector to populate vector b:
    std::vector<T> b(m, static_cast<T>(-2.0) * h * h);
    // set the Dirichlet boundary conditions:
    T left = static_cast<T>(0.0);
    T right = static_cast<T>(0.0);
    b[0] = b[0] - left;
    b[b.size() - 1] = b[b.size() - 1] - right;

    // create sparse solver on HOST:
    real_sparse_solver_cuda<memory_space_enum::Host, T, std::vector, std::allocator<T>> rss(m);

    // because we used default cstor we need to call initialize
    rss.initialize(m);

    // insert sparse matrix A and vector b:
    rss.set_flat_sparse_matrix(std::move(fsm));
    rss.set_rhs(b);

    std::vector<T> solution(m);
    rss.solve(solution);

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

template <typename T> void hostBVPDirichletBCLUTest()
{
    using lss_core_cuda_solver::factorization_enum;
    using lss_core_cuda_solver::flat_matrix;
    using lss_core_cuda_solver::memory_space_enum;
    using lss_core_cuda_solver::real_sparse_solver_cuda;

    std::cout << "=================================\n";
    std::cout << " Using LU decomposition to \n";
    std::cout << " solve Boundary Value Problem: \n\n";
    std::cout << " type: double					\n\n";
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
    const T h = static_cast<T>(1.0) / static_cast<T>(N);
    // set number of columns and rows:
    // because we already know the boundary values
    // at t_0 = 0 and t_20 = 0:
    int const m = N - 1;
    // first create and populate the sparse matrix:
    flat_matrix<T> fsm(m, m);
    // populate the matrix:
    fsm.emplace_back(0, 0, static_cast<T>(-2.0));
    fsm.emplace_back(0, 1, static_cast<T>(1.0));
    for (std::size_t t = 1; t < m - 1; ++t)
    {
        fsm.emplace_back(t, t - 1, static_cast<T>(1.0));
        fsm.emplace_back(t, t, static_cast<T>(-2.0));
        fsm.emplace_back(t, t + 1, static_cast<T>(1.0));
    }
    fsm.emplace_back(m - 1, m - 2, static_cast<T>(1.0));
    fsm.emplace_back(m - 1, m - 1, static_cast<T>(-2.0));

    // lets use std::vector to populate vector b:
    std::vector<T> b(m, static_cast<T>(-2.0) * h * h);
    // set the Dirichlet boundary conditions:
    const T left = static_cast<T>(0.0);
    const T right = static_cast<T>(0.0);
    b[0] = b[0] - left;
    b[b.size() - 1] = b[b.size() - 1] - right;

    // create sparse solver on HOST:
    real_sparse_solver_cuda<memory_space_enum::Host, T, std::vector, std::allocator<T>> rss(m);

    // because we used default cstor we need to call initialize
    rss.initialize(m);

    // insert sparse matrix A and vector b:
    rss.set_flat_sparse_matrix(std::move(fsm));
    rss.set_rhs(b);

    std::vector<T> solution(m);
    rss.solve(solution, factorization_enum::LUMethod);

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

template <typename T> void hostBVPDirichletBCCholeskyTest()
{
    using lss_core_cuda_solver::factorization_enum;
    using lss_core_cuda_solver::flat_matrix;
    using lss_core_cuda_solver::memory_space_enum;
    using lss_core_cuda_solver::real_sparse_solver_cuda;

    std::cout << "=================================\n";
    std::cout << " Using Cholesky decomposition to \n";
    std::cout << " solve Boundary Value Problem: \n\n";
    std::cout << " type: double					\n\n";
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
    T h = static_cast<T>(1.0) / static_cast<T>(N);
    // set number of columns and rows:
    // because we already know the boundary values
    // at t_0 = 0 and t_20 = 0:
    int const m = N - 1;
    // first create and populate the sparse matrix:
    flat_matrix<T> fsm(m, m);
    // populate the matrix:
    fsm.emplace_back(0, 0, static_cast<T>(-2.0));
    fsm.emplace_back(0, 1, static_cast<T>(1.0));
    for (std::size_t t = 1; t < m - 1; ++t)
    {
        fsm.emplace_back(t, t - 1, static_cast<T>(1.0));
        fsm.emplace_back(t, t, static_cast<T>(-2.0));
        fsm.emplace_back(t, t + 1, static_cast<T>(1.0));
    }
    fsm.emplace_back(m - 1, m - 2, static_cast<T>(1.0));
    fsm.emplace_back(m - 1, m - 1, static_cast<T>(-2.0));

    // lets use std::vector to populate vector b:
    std::vector<T> b(m, static_cast<T>(-2.0) * h * h);
    // set the Dirichlet boundary conditions:
    T left = static_cast<T>(0.0);
    T right = static_cast<T>(0.0);
    b[0] = b[0] - left;
    b[b.size() - 1] = b[b.size() - 1] - right;

    // create sparse solver on HOST:
    real_sparse_solver_cuda<memory_space_enum::Host, T, std::vector, std::allocator<T>> rss(m);

    // because we used default cstor we need to call initialize
    rss.initialize(m);

    // insert sparse matrix A and vector b:
    rss.set_flat_sparse_matrix(std::move(fsm));
    rss.set_rhs(b);

    std::vector<T> solution(m);
    rss.solve(solution, factorization_enum::CholeskyMethod);

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

void testDirichletBCBVPOnHost()
{
    std::cout << "==================================================\n";
    std::cout << "============ Dirichlet BC BVP - HOST ´´============\n";
    std::cout << "==================================================\n";

    hostBVPDirichletBCQRTest<double>();
    hostBVPDirichletBCQRTest<float>();
    hostBVPDirichletBCLUTest<double>();
    hostBVPDirichletBCLUTest<float>();
    hostBVPDirichletBCCholeskyTest<double>();
    hostBVPDirichletBCCholeskyTest<float>();

    std::cout << "==================================================\n";
}

template <typename T> void deviceBVPDirichletBCQRTest()
{
    using lss_core_cuda_solver::flat_matrix;
    using lss_core_cuda_solver::memory_space_enum;
    using lss_core_cuda_solver::real_sparse_solver_cuda;

    std::cout << "=================================\n";
    std::cout << " Using QR decomposition to \n";
    std::cout << " solve Boundary Value Problem: \n\n";
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
    T h = static_cast<T>(1.0) / static_cast<T>(N);
    // set number of columns and rows:
    // because we already know the boundary values
    // at timepoints t_0 and t_20:
    int const m = N - 1;
    // first create and populate the sparse matrix:
    flat_matrix<T> fsm(m, m);
    // populate the matrix:
    fsm.emplace_back(0, 0, static_cast<T>(-2.0));
    fsm.emplace_back(0, 1, static_cast<T>(1.0));
    for (std::size_t t = 1; t < m - 1; ++t)
    {
        fsm.emplace_back(t, t - 1, static_cast<T>(1.0));
        fsm.emplace_back(t, t, static_cast<T>(-2.0));
        fsm.emplace_back(t, t + 1, static_cast<T>(1.0));
    }
    fsm.emplace_back(m - 1, m - 2, static_cast<T>(1.0));
    fsm.emplace_back(m - 1, m - 1, static_cast<T>(-2.0));

    // lets use std::vector to populate vector b:
    std::vector<T> b(m, static_cast<T>(-2.0) * h * h);
    // set the Dirichlet boundary conditions:
    T left = static_cast<T>(0.0);
    T right = static_cast<T>(0.0);
    b[0] = b[0] - left;
    b[b.size() - 1] = b[b.size() - 1] - right;

    // create sparse solver on HOST:
    real_sparse_solver_cuda<memory_space_enum::Device, T, std::vector, std::allocator<T>> rss(m);

    // because we used default cstor we need to call initialize
    rss.initialize(m);

    // insert sparse matrix A and vector b:
    rss.set_flat_sparse_matrix(std::move(fsm));
    rss.set_rhs(b);

    std::vector<T> solution(m);
    rss.solve(solution);

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

template <typename T> void deviceBVPDirichletBCCholeskyTest()
{
    using lss_core_cuda_solver::factorization_enum;
    using lss_core_cuda_solver::flat_matrix;
    using lss_core_cuda_solver::memory_space_enum;
    using lss_core_cuda_solver::real_sparse_solver_cuda;

    std::cout << "=================================\n";
    std::cout << " Using Cholesky decomposition to \n";
    std::cout << " solve Boundary Value Problem: \n\n";
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
    T h = static_cast<T>(1.0) / static_cast<T>(N);
    // set number of columns and rows:
    // because we already know the boundary values
    // at t_0 = 0 and t_20 = 0:
    int const m = N - 1;
    // first create and populate the sparse matrix:
    flat_matrix<T> fsm(m, m);
    // populate the matrix:
    fsm.emplace_back(0, 0, static_cast<T>(-2.0));
    fsm.emplace_back(0, 1, static_cast<T>(1.0));
    for (std::size_t t = 1; t < m - 1; ++t)
    {
        fsm.emplace_back(t, t - 1, static_cast<T>(1.0));
        fsm.emplace_back(t, t, static_cast<T>(-2.0));
        fsm.emplace_back(t, t + 1, static_cast<T>(1.0));
    }
    fsm.emplace_back(m - 1, m - 2, static_cast<T>(1.0));
    fsm.emplace_back(m - 1, m - 1, static_cast<T>(-2.0));

    // lets use std::vector to populate vector b:
    std::vector<T> b(m, static_cast<T>(-2.0) * h * h);
    // set the Dirichlet boundary conditions:
    T left = static_cast<T>(0.0);
    T right = static_cast<T>(0.0);
    b[0] = b[0] - left;
    b[b.size() - 1] = b[b.size() - 1] - right;

    // create sparse solver on HOST:
    real_sparse_solver_cuda<memory_space_enum::Device, T, std::vector, std::allocator<T>> rss(m);

    // because we used default cstor we need to call initialize
    rss.initialize(m);

    // insert sparse matrix A and vector b:
    rss.set_flat_sparse_matrix(std::move(fsm));
    rss.set_rhs(b);

    std::vector<T> solution(m);
    rss.solve(solution, factorization_enum::CholeskyMethod);

    // exact value:
    auto exact = [](double x) { return x * (static_cast<T>(1.0) - x); };

    std::cout << "tp : FDM | Exact\n";
    std::cout << "t_" << 0 << ": " << left << " |  " << exact(0) << '\n';
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        std::cout << "t_" << j + 1 << ": " << solution[j] << " |  " << exact((j + 1) * h) << '\n';
    }
    std::cout << "t_" << N << ": " << right << " |  " << exact(N * h) << '\n';
}

void testDirichletBCBVPOnDevice()
{
    std::cout << "==================================================\n";
    std::cout << "============ Dirichlet BC BVP - DEVICE ===========\n";
    std::cout << "==================================================\n";

    deviceBVPDirichletBCQRTest<double>();
    deviceBVPDirichletBCQRTest<float>();
    deviceBVPDirichletBCCholeskyTest<double>();
    deviceBVPDirichletBCCholeskyTest<float>();

    std::cout << "==================================================\n";
}

template <typename T> void hostBVPRobinBCQRTest()
{
    using lss_core_cuda_solver::flat_matrix;
    using lss_core_cuda_solver::memory_space_enum;
    using lss_core_cuda_solver::real_sparse_solver_cuda;
    using lss_core_cuda_solver_policy::sparse_solver_host_qr;

    std::cout << "=================================\n";
    std::cout << " Using QR decomposition to \n";
    std::cout << " solve Boundary-value problem: \n\n";
    std::cout << " u''(t) = -2, \n\n";
    std::cout << " where\n\n";
    std::cout << " t in <0,1>,\n";
    std::cout << " u(0) = 1 \n";
    std::cout << " u'(1) + u(1) = 0\n\n";
    std::cout << "Exact solution is:\n\n";
    std::cout << " u(t) = -t*t + t + 1\n";
    std::cout << "=================================\n";

    // discretization:
    // t_0,t_1,t_2,...,t_20
    int const N = 100;
    // step size:
    T h = static_cast<T>(1.0) / static_cast<T>(N);
    // set the Robin boundary conditions:
    T alpha = static_cast<T>(.00);
    T phi = static_cast<T>(1.0);
    T beta = (static_cast<T>(2.0) + h) / (static_cast<T>(2.0) - h);
    T psi = static_cast<T>(0.0);
    // set number of columns and rows:
    // because we already know the boundary values
    // at t_0 = 0 and t_20 = 0:
    int const m = N - 1;
    // first create and populate the sparse matrix:
    flat_matrix<T> fsm(m, m);
    // populate the matrix:
    fsm.emplace_back(0, 0, static_cast<T>((alpha * 1.0 - 2.0)));
    fsm.emplace_back(0, 1, static_cast<T>(1.0));
    for (std::size_t t = 1; t < m - 1; ++t)
    {
        fsm.emplace_back(t, t - 1, static_cast<T>(1.0));
        fsm.emplace_back(t, t, static_cast<T>(-2.0));
        fsm.emplace_back(t, t + 1, static_cast<T>(1.0));
    }
    fsm.emplace_back(m - 1, m - 2, static_cast<T>(1.0));
    fsm.emplace_back(m - 1, m - 1, static_cast<T>((-2.0 + (1.0 / beta))));

    // lets use std::vector to populate vector b:
    std::vector<T> b(m, static_cast<T>(-2.0) * h * h);
    // adjust first and last elements due to the Robin BC
    b[0] = b[0] - static_cast<T>(1.0) * phi;
    b[b.size() - 1] = b[b.size() - 1] + psi * (static_cast<T>(1.0) / beta);

    // create sparse solver on HOST:
    real_sparse_solver_cuda<memory_space_enum::Host, T, std::vector, std::allocator<T>> rss(m);

    // because we used default cstor we need to call initialize
    rss.initialize(m);

    // insert sparse matrix A and vector b:
    rss.set_flat_sparse_matrix(std::move(fsm));
    rss.set_rhs(b);

    std::vector<T> solution(m);
    rss.solve(solution);

    // exact value:
    auto exact = [](T x) { return (-x * x + x + static_cast<T>(1.0)); };

    std::cout << "tp : FDM | Exact | Abs Diff\n";
    std::cout << "t_" << 0 << ": " << (alpha * solution.front() + phi) << " |  " << exact(0) << " | "
              << ((alpha * solution.front() + phi) - exact(0)) << '\n';
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        std::cout << "t_" << j + 1 << ": " << solution[j] << " |  " << exact((j + 1) * h) << " | "
                  << (solution[j] - exact((j + 1) * h)) << '\n';
    }
    std::cout << "t_" << N << ": " << ((solution.back() - psi) / beta) << " |  " << exact(N * h) << " | "
              << (((solution.back() - psi) / beta) - exact(N * h)) << '\n';
}

template <typename T> void hostBVPRobinBCLUTest()
{
    using lss_core_cuda_solver::factorization_enum;
    using lss_core_cuda_solver::flat_matrix;
    using lss_core_cuda_solver::memory_space_enum;
    using lss_core_cuda_solver::real_sparse_solver_cuda;

    std::cout << "=================================\n";
    std::cout << " Using LU decomposition to \n";
    std::cout << " solve Boundary-value problem: \n\n";
    std::cout << " u''(t) = -2, \n\n";
    std::cout << " where\n\n";
    std::cout << " t in <0,1>,\n";
    std::cout << " u(0) = 1 \n";
    std::cout << " u'(1) + u(1) = 0\n\n";
    std::cout << "Exact solution is:\n\n";
    std::cout << " u(t) = -t*t + t + 1\n";
    std::cout << "=================================\n";

    // discretization:
    // t_0,t_1,t_2,...,t_20
    int const N = 100;
    // step size:
    T h = static_cast<T>(1.0) / static_cast<T>(N);
    // set the Robin boundary conditions:
    T alpha = static_cast<T>(.00);
    T phi = static_cast<T>(1.0);
    T beta = (static_cast<T>(2.0) + h) / (static_cast<T>(2.0) - h);
    T psi = static_cast<T>(0.0);
    // set number of columns and rows:
    // because we already know the boundary values
    // at t_0 = 0 and t_20 = 0:
    int const m = N - 1;
    // first create and populate the sparse matrix:
    flat_matrix<T> fsm(m, m);
    // populate the matrix:
    fsm.emplace_back(0, 0, static_cast<T>((alpha * 1.0 - 2.0)));
    fsm.emplace_back(0, 1, static_cast<T>(1.0));
    for (std::size_t t = 1; t < m - 1; ++t)
    {
        fsm.emplace_back(t, t - 1, static_cast<T>(1.0));
        fsm.emplace_back(t, t, static_cast<T>(-2.0));
        fsm.emplace_back(t, t + 1, static_cast<T>(1.0));
    }
    fsm.emplace_back(m - 1, m - 2, static_cast<T>(1.0));
    fsm.emplace_back(m - 1, m - 1, static_cast<T>((-2.0 + (1.0 / beta))));

    // lets use std::vector to populate vector b:
    std::vector<T> b(m, static_cast<T>(-2.0) * h * h);
    // adjust first and last elements due to the Robin BC
    b[0] = b[0] - static_cast<T>(1.0) * phi;
    b[b.size() - 1] = b[b.size() - 1] + psi * (static_cast<T>(1.0) / beta);

    // create sparse solver on HOST:
    real_sparse_solver_cuda<memory_space_enum::Host, T, std::vector, std::allocator<T>> rss(m);

    // because we used default cstor we need to call initialize
    rss.initialize(m);

    // insert sparse matrix A and vector b:
    rss.set_flat_sparse_matrix(std::move(fsm));
    rss.set_rhs(b);

    std::vector<T> solution(m);
    rss.solve(solution, factorization_enum::LUMethod);

    // exact value:
    auto exact = [](T x) { return (-x * x + x + static_cast<T>(1.0)); };

    std::cout << "tp : FDM | Exact | Abs Diff\n";
    std::cout << "t_" << 0 << ": " << (alpha * solution.front() + phi) << " |  " << exact(0) << " | "
              << ((alpha * solution.front() + phi) - exact(0)) << '\n';
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        std::cout << "t_" << j + 1 << ": " << solution[j] << " |  " << exact((j + 1) * h) << " | "
                  << (solution[j] - exact((j + 1) * h)) << '\n';
    }
    std::cout << "t_" << N << ": " << ((solution.back() - psi) / beta) << " |  " << exact(N * h) << " | "
              << (((solution.back() - psi) / beta) - exact(N * h)) << '\n';
}

template <typename T> void hostBVPRobinBCCholeskyTest()
{
    using lss_core_cuda_solver::factorization_enum;
    using lss_core_cuda_solver::flat_matrix;
    using lss_core_cuda_solver::memory_space_enum;
    using lss_core_cuda_solver::real_sparse_solver_cuda;

    std::cout << "=================================\n";
    std::cout << " Using Cholesky decomposition to \n";
    std::cout << " solve Boundary-value problem: \n\n";
    std::cout << " u''(t) = -2, \n\n";
    std::cout << " where\n\n";
    std::cout << " t in <0,1>,\n";
    std::cout << " u(0) = 1 \n";
    std::cout << " u'(1) + u(1) = 0\n\n";
    std::cout << "Exact solution is:\n\n";
    std::cout << " u(t) = -t*t + t + 1\n";
    std::cout << "=================================\n";

    // discretization:
    // t_0,t_1,t_2,...,t_20
    int const N = 100;
    // step size:
    T h = static_cast<T>(1.0) / static_cast<T>(N);
    // set the Robin boundary conditions:
    T alpha = static_cast<T>(.00);
    T phi = static_cast<T>(1.0);
    T beta = (static_cast<T>(2.0) + h) / (static_cast<T>(2.0) - h);
    T psi = static_cast<T>(0.0);
    // set number of columns and rows:
    // because we already know the boundary values
    // at t_0 = 0 and t_20 = 0:
    int const m = N - 1;
    // first create and populate the sparse matrix:
    flat_matrix<T> fsm(m, m);
    // populate the matrix:
    fsm.emplace_back(0, 0, static_cast<T>((alpha * 1.0 - 2.0)));
    fsm.emplace_back(0, 1, static_cast<T>(1.0));
    for (std::size_t t = 1; t < m - 1; ++t)
    {
        fsm.emplace_back(t, t - 1, static_cast<T>(1.0));
        fsm.emplace_back(t, t, static_cast<T>(-2.0));
        fsm.emplace_back(t, t + 1, static_cast<T>(1.0));
    }
    fsm.emplace_back(m - 1, m - 2, static_cast<T>(1.0));
    fsm.emplace_back(m - 1, m - 1, static_cast<T>((-2.0 + (1.0 / beta))));

    // lets use std::vector to populate vector b:
    std::vector<T> b(m, static_cast<T>(-2.0) * h * h);
    // adjust first and last elements due to the Robin BC
    b[0] = b[0] - static_cast<T>(1.0) * phi;
    b[b.size() - 1] = b[b.size() - 1] + psi * (static_cast<T>(1.0) / beta);

    // create sparse solver on HOST:
    real_sparse_solver_cuda<memory_space_enum::Host, T, std::vector, std::allocator<T>> rss(m);

    // because we used default cstor we need to call initialize
    rss.initialize(m);

    // insert sparse matrix A and vector b:
    rss.set_flat_sparse_matrix(std::move(fsm));
    rss.set_rhs(b);

    std::vector<T> solution(m);
    rss.solve(solution, factorization_enum::CholeskyMethod);

    // exact value:
    auto exact = [](T x) { return (-x * x + x + static_cast<T>(1.0)); };

    std::cout << "tp : FDM | Exact | Abs Diff\n";
    std::cout << "t_" << 0 << ": " << (alpha * solution.front() + phi) << " |  " << exact(0) << " | "
              << ((alpha * solution.front() + phi) - exact(0)) << '\n';
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        std::cout << "t_" << j + 1 << ": " << solution[j] << " |  " << exact((j + 1) * h) << " | "
                  << (solution[j] - exact((j + 1) * h)) << '\n';
    }
    std::cout << "t_" << N << ": " << ((solution.back() - psi) / beta) << " |  " << exact(N * h) << " | "
              << (((solution.back() - psi) / beta) - exact(N * h)) << '\n';
}

void testRobinBCBVPOnHost()
{
    std::cout << "==================================================\n";
    std::cout << "============ Robin BC BVP - HOST =================\n";
    std::cout << "==================================================\n";

    hostBVPRobinBCQRTest<double>();
    hostBVPRobinBCQRTest<float>();
    hostBVPRobinBCLUTest<double>();
    hostBVPRobinBCLUTest<float>();
    hostBVPRobinBCCholeskyTest<double>();
    hostBVPRobinBCCholeskyTest<float>();

    std::cout << "==================================================\n";
}

template <typename T> void deviceBVPRobinBCQRTest()
{
    using lss_core_cuda_solver::flat_matrix;
    using lss_core_cuda_solver::memory_space_enum;
    using lss_core_cuda_solver::real_sparse_solver_cuda;

    std::cout << "=================================\n";
    std::cout << " Using QR decomposition to \n";
    std::cout << " solve Boundary-value problem: \n\n";
    std::cout << " u''(t) = -2, \n\n";
    std::cout << " where\n\n";
    std::cout << " t in <0,1>,\n";
    std::cout << " u(0) = 1 \n";
    std::cout << " u'(1) + u(1) = 0\n\n";
    std::cout << "Exact solution is:\n\n";
    std::cout << " u(t) = -t*t + t + 1\n";
    std::cout << "=================================\n";

    // discretization:
    // t_0,t_1,t_2,...,t_20
    int const N = 100;
    // step size:
    T h = static_cast<T>(1.0) / static_cast<T>(N);
    // set the Robin boundary conditions:
    T alpha = static_cast<T>(.00);
    T phi = static_cast<T>(1.0);
    T beta = (static_cast<T>(2.0) + h) / (static_cast<T>(2.0) - h);
    T psi = static_cast<T>(0.0);
    // set number of columns and rows:
    // because we already know the boundary values
    // at timepoints t_0 and t_20:
    int const m = N - 1;
    // first create and populate the sparse matrix:
    flat_matrix<T> fsm(m, m);
    // populate the matrix:
    fsm.emplace_back(0, 0, static_cast<T>((alpha * 1.0 - 2.0)));
    fsm.emplace_back(0, 1, static_cast<T>(1.0));
    for (std::size_t t = 1; t < m - 1; ++t)
    {
        fsm.emplace_back(t, t - 1, static_cast<T>(1.0));
        fsm.emplace_back(t, t, static_cast<T>(-2.0));
        fsm.emplace_back(t, t + 1, static_cast<T>(1.0));
    }
    fsm.emplace_back(m - 1, m - 2, static_cast<T>(1.0));
    fsm.emplace_back(m - 1, m - 1, static_cast<T>((-2.0 + (1.0 / beta))));

    // lets use std::vector to populate vector b:
    std::vector<T> b(m, static_cast<T>(-2.0) * h * h);
    // adjust first and last elements due to the Robin BC
    b[0] = b[0] - static_cast<T>(1.0) * phi;
    b[b.size() - 1] = b[b.size() - 1] + psi * (static_cast<T>(1.0) / beta);

    // create sparse solver on HOST:
    real_sparse_solver_cuda<memory_space_enum::Device, T, std::vector, std::allocator<T>> rss(m);

    // because we used default cstor we need to call initialize
    rss.initialize(m);

    // insert sparse matrix A and vector b:
    rss.set_flat_sparse_matrix(std::move(fsm));
    rss.set_rhs(b);

    std::vector<T> solution(m);
    rss.solve(solution);

    // exact value:
    auto exact = [](T x) { return (-x * x + x + static_cast<T>(1.0)); };

    std::cout << "tp : FDM | Exact | Abs Diff\n";
    std::cout << "t_" << 0 << ": " << (alpha * solution.front() + phi) << " |  " << exact(0) << " | "
              << ((alpha * solution.front() + phi) - exact(0)) << '\n';
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        std::cout << "t_" << j + 1 << ": " << solution[j] << " |  " << exact((j + 1) * h) << " | "
                  << (solution[j] - exact((j + 1) * h)) << '\n';
    }
    std::cout << "t_" << N << ": " << ((solution.back() - psi) / beta) << " |  " << exact(N * h) << " | "
              << (((solution.back() - psi) / beta) - exact(N * h)) << '\n';
}

template <typename T> void deviceBVPRobinBCCholeskyTest()
{
    using lss_core_cuda_solver::factorization_enum;
    using lss_core_cuda_solver::flat_matrix;
    using lss_core_cuda_solver::memory_space_enum;
    using lss_core_cuda_solver::real_sparse_solver_cuda;

    std::cout << "=================================\n";
    std::cout << " Using Cholesky decomposition to \n";
    std::cout << " solve Boundary-value problem: \n\n";
    std::cout << " u''(t) = -2, \n\n";
    std::cout << " where\n\n";
    std::cout << " t in <0,1>,\n";
    std::cout << " u(0) = 1 \n";
    std::cout << " u'(1) + u(1) = 0\n\n";
    std::cout << "Exact solution is:\n\n";
    std::cout << " u(t) = -t*t + t + 1\n";
    std::cout << "=================================\n";

    // discretization:
    // t_0,t_1,t_2,...,t_20
    int const N = 100;
    // step size:
    T h = static_cast<T>(1.0) / static_cast<T>(N);
    // set the Robin boundary conditions:
    T alpha = static_cast<T>(.00);
    T phi = static_cast<T>(1.0);
    T beta = (static_cast<T>(2.0) + h) / (static_cast<T>(2.0) - h);
    T psi = static_cast<T>(0.0);
    // set number of columns and rows:
    // because we already know the boundary values
    // at t_0 = 0 and t_20 = 0:
    int const m = N - 1;
    // first create and populate the sparse matrix:
    flat_matrix<T> fsm(m, m);
    // populate the matrix:
    fsm.emplace_back(0, 0, static_cast<T>((alpha * 1.0 - 2.0)));
    fsm.emplace_back(0, 1, static_cast<T>(1.0));
    for (std::size_t t = 1; t < m - 1; ++t)
    {
        fsm.emplace_back(t, t - 1, static_cast<T>(1.0));
        fsm.emplace_back(t, t, static_cast<T>(-2.0));
        fsm.emplace_back(t, t + 1, static_cast<T>(1.0));
    }
    fsm.emplace_back(m - 1, m - 2, static_cast<T>(1.0));
    fsm.emplace_back(m - 1, m - 1, static_cast<T>((-2.0 + (1.0 / beta))));

    // lets use std::vector to populate vector b:
    std::vector<T> b(m, static_cast<T>(-2.0) * h * h);
    // adjust first and last elements due to the Robin BC
    b[0] = b[0] - static_cast<T>(1.0) * phi;
    b[b.size() - 1] = b[b.size() - 1] + psi * (static_cast<T>(1.0) / beta);

    // create sparse solver on HOST:
    real_sparse_solver_cuda<memory_space_enum::Device, T, std::vector, std::allocator<T>> rss(m);

    // because we used default cstor we need to call initialize
    rss.initialize(m);

    // insert sparse matrix A and vector b:
    rss.set_flat_sparse_matrix(std::move(fsm));
    rss.set_rhs(b);

    std::vector<T> solution(m);
    rss.solve(solution, factorization_enum::CholeskyMethod);

    // exact value:
    auto exact = [](T x) { return (-x * x + x + static_cast<T>(1.0)); };

    std::cout << "tp : FDM | Exact | Abs Diff\n";
    std::cout << "t_" << 0 << ": " << (alpha * solution.front() + phi) << " |  " << exact(0) << " | "
              << ((alpha * solution.front() + phi) - exact(0)) << '\n';
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        std::cout << "t_" << j + 1 << ": " << solution[j] << " |  " << exact((j + 1) * h) << " | "
                  << (solution[j] - exact((j + 1) * h)) << '\n';
    }
    std::cout << "t_" << N << ": " << ((solution.back() - psi) / beta) << " |  " << exact(N * h) << " | "
              << (((solution.back() - psi) / beta) - exact(N * h)) << '\n';
}

void testRobinBCBVPOnDevice()
{
    std::cout << "==================================================\n";
    std::cout << "============ Robin BC BVP - DEVICE ===============\n";
    std::cout << "==================================================\n";

    deviceBVPRobinBCQRTest<double>();
    deviceBVPRobinBCQRTest<float>();
    deviceBVPRobinBCCholeskyTest<double>();
    deviceBVPRobinBCCholeskyTest<float>();

    std::cout << "==================================================\n";
}

#endif ///_lss_core_cuda_solver_T
