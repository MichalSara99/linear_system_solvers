#pragma once
#if !defined(_LSS_DENSE_SOLVERS_CUDA_T_HPP_)
#define _LSS_DENSE_SOLVERS_CUDA_T_HPP_

#include "dense_solvers/lss_dense_solvers_cuda.hpp"

template <typename T> void deviceDenseQRTest()
{
    using lss_dense_solvers::flat_matrix;
    using lss_dense_solvers::real_dense_solver_cuda;

    std::cout << "===================================\n";
    std::cout << "Solving system of equations:\n";
    std::cout << "Ax = b,\n";
    std::cout << "where\n\n";
    std::cout << "    [ 1 2 3 ] \n";
    std::cout << "A = [ 4 5 6 ] \n";
    std::cout << "    [ 2 1 1 ] \n";
    std::cout << "\n";
    std::cout << "b = [ 6 15 4]^T \n";
    std::cout << "\n";

    /*

    Solving for x in

    Ax = b,
    where

    [ 1 2 3 ]
    A = [ 4 5 6 ]
    [ 2 1 1 ]

    x = [1 1 1]^T
    b = [6 15 4]^T


    */

    // size of the system:
    int const m = 3;
    // first create and populate the dense matrix:
    flat_matrix<T> fsm(m, m);

    // populate the matrix:
    fsm.emplace_back(0, 0, static_cast<T>(1.0));
    fsm.emplace_back(0, 1, static_cast<T>(2.0));
    fsm.emplace_back(0, 2, static_cast<T>(3.0));
    fsm.emplace_back(1, 0, static_cast<T>(4.0));
    fsm.emplace_back(1, 1, static_cast<T>(5.0));
    fsm.emplace_back(1, 2, static_cast<T>(6.0));
    fsm.emplace_back(2, 0, static_cast<T>(2.0));
    fsm.emplace_back(2, 1, static_cast<T>(1.0));
    fsm.emplace_back(2, 2, static_cast<T>(1.0));

    // lets use std::vector to populate vector b:
    std::vector<T> b = {6.0, 15.0, 4.0};

    // create dense solver:
    real_dense_solver_cuda<T> rds(m, m);

    // because we used default cstor we need to call initialize
    rds.initialize(m, m);

    // insert sparse matrix A and vector b:
    rds.set_flat_dense_matrix(std::move(fsm));
    rds.set_rhs(b);

    auto solution = rds.solve();

    std::cout << "Solution is: \n[";
    for (auto const &e : solution)
    {
        std::cout << e << " ";
    }
    std::cout << "]\n";
}

template <typename T> void deviceDenseQRPointersTest()
{
    using lss_dense_solvers::flat_matrix;
    using lss_dense_solvers::real_dense_solver_cuda;

    std::cout << "===================================\n";
    std::cout << "Solving system of equations:\n";
    std::cout << "Ax = b,\n";
    std::cout << "where\n\n";
    std::cout << "    [ 1 2 3 ] \n";
    std::cout << "A = [ 4 5 6 ] \n";
    std::cout << "    [ 2 1 1 ] \n";
    std::cout << "\n";
    std::cout << "b = [ 6 15 4]^T \n";
    std::cout << "\n";

    /*

    Solving for x in

    Ax = b,
    where

    [ 1 2 3 ]
    A = [ 4 5 6 ]
    [ 2 1 1 ]

    x = [1 1 1]^T
    b = [6 15 4]^T


    */

    // size of the system:
    int const m = 3;
    // first create and populate the dense matrix:
    flat_matrix<T> fsm(m, m);

    // populate the matrix:
    fsm.emplace_back(0, 0, static_cast<T>(1.0));
    fsm.emplace_back(0, 1, static_cast<T>(2.0));
    fsm.emplace_back(0, 2, static_cast<T>(3.0));
    fsm.emplace_back(1, 0, static_cast<T>(4.0));
    fsm.emplace_back(1, 1, static_cast<T>(5.0));
    fsm.emplace_back(1, 2, static_cast<T>(6.0));
    fsm.emplace_back(2, 0, static_cast<T>(2.0));
    fsm.emplace_back(2, 1, static_cast<T>(1.0));
    fsm.emplace_back(2, 2, static_cast<T>(1.0));

    // lets use std::vector to populate vector b:
    std::vector<T> b = {6.0, 15.0, 4.0};

    // create dense solver:
    real_dense_solver_cuda<T> rds(m, m);

    // because we used default cstor we need to call initialize
    rds.initialize(m, m);

    // insert sparse matrix A and vector b:
    rds.set_flat_dense_matrix(std::move(fsm));
    rds.set_rhs(b);

    std::vector<T> solution(m);
    rds.solve(solution);

    std::cout << "Solution is: \n[";
    for (std::size_t t = 0; t < m; ++t)
    {
        std::cout << solution[t] << " ";
    }
    std::cout << "]\n";
}

void deviceDenseQRTest()
{
    std::cout << "==================================================\n";
    std::cout << "=========== Dense QR Solver - DEVICE =============\n";
    std::cout << "==================================================\n";

    deviceDenseQRTest<double>();
    deviceDenseQRTest<float>();
    deviceDenseQRPointersTest<double>();
    deviceDenseQRPointersTest<float>();

    std::cout << "==================================================\n";
}

template <typename T> void deviceDenseLUTest()
{
    using lss_dense_solvers::factorization_enum;
    using lss_dense_solvers::flat_matrix;
    using lss_dense_solvers::real_dense_solver_cuda;

    std::cout << "===================================\n";
    std::cout << "Solving system of equations:\n";
    std::cout << "Ax = b,\n";
    std::cout << "where\n\n";
    std::cout << "    [ 1 2 3 ] \n";
    std::cout << "A = [ 4 5 6 ] \n";
    std::cout << "    [ 2 1 1 ] \n";
    std::cout << "\n";
    std::cout << "b = [ 6 15 4]^T \n";
    std::cout << "\n";

    /*

    Solving for x in

    Ax = b,
    where

    [ 1 2 3 ]
    A = [ 4 5 6 ]
    [ 2 1 1 ]

    x = [1 1 1]^T
    b = [6 15 4]^T


    */

    // size of the system:
    int const m = 3;
    // first create and populate the dense matrix:
    flat_matrix<T> fsm(m, m);

    // populate the matrix:
    fsm.emplace_back(0, 0, static_cast<T>(1.0));
    fsm.emplace_back(0, 1, static_cast<T>(2.0));
    fsm.emplace_back(0, 2, static_cast<T>(3.0));
    fsm.emplace_back(1, 0, static_cast<T>(4.0));
    fsm.emplace_back(1, 1, static_cast<T>(5.0));
    fsm.emplace_back(1, 2, static_cast<T>(6.0));
    fsm.emplace_back(2, 0, static_cast<T>(2.0));
    fsm.emplace_back(2, 1, static_cast<T>(1.0));
    fsm.emplace_back(2, 2, static_cast<T>(1.0));

    // lets use std::vector to populate vector b:
    std::vector<T> b = {6.0, 15.0, 4.0};

    // create dense solver:
    real_dense_solver_cuda<T> rds(m, m);

    // because we used default cstor we need to call initialize
    rds.initialize(m, m);

    // insert sparse matrix A and vector b:
    rds.set_flat_dense_matrix(std::move(fsm));
    rds.set_rhs(b);

    auto solution = rds.solve(factorization_enum::LUMethod);
    std::cout << "Solution is: \n[";
    for (auto const &e : solution)
    {
        std::cout << e << " ";
    }
    std::cout << "]\n";
}

template <typename T> void deviceDenseLUPointersTest()
{
    using lss_dense_solvers::factorization_enum;
    using lss_dense_solvers::flat_matrix;
    using lss_dense_solvers::real_dense_solver_cuda;

    std::cout << "===================================\n";
    std::cout << "Solving system of equations:\n";
    std::cout << "Ax = b,\n";
    std::cout << "where\n\n";
    std::cout << "    [ 1 2 3 ] \n";
    std::cout << "A = [ 4 5 6 ] \n";
    std::cout << "    [ 2 1 1 ] \n";
    std::cout << "\n";
    std::cout << "b = [ 6 15 4]^T \n";
    std::cout << "\n";

    /*

    Solving for x in

    Ax = b,
    where

    [ 1 2 3 ]
    A = [ 4 5 6 ]
    [ 2 1 1 ]

    x = [1 1 1]^T
    b = [6 15 4]^T


    */

    // size of the system:
    int const m = 3;
    // first create and populate the dense matrix:
    flat_matrix<T> fsm(m, m);

    // populate the matrix:
    fsm.emplace_back(0, 0, static_cast<T>(1.0));
    fsm.emplace_back(0, 1, static_cast<T>(2.0));
    fsm.emplace_back(0, 2, static_cast<T>(3.0));
    fsm.emplace_back(1, 0, static_cast<T>(4.0));
    fsm.emplace_back(1, 1, static_cast<T>(5.0));
    fsm.emplace_back(1, 2, static_cast<T>(6.0));
    fsm.emplace_back(2, 0, static_cast<T>(2.0));
    fsm.emplace_back(2, 1, static_cast<T>(1.0));
    fsm.emplace_back(2, 2, static_cast<T>(1.0));

    // lets use std::vector to populate vector b:
    std::vector<T> b = {6.0, 15.0, 4.0};

    // create dense solver:
    real_dense_solver_cuda<T> rds(m, m);

    // because we used default cstor we need to call initialize
    rds.initialize(m, m);

    // insert sparse matrix A and vector b:
    rds.set_flat_dense_matrix(std::move(fsm));
    rds.set_rhs(b);

    std::vector<T> solution(m);
    rds.solve(solution, factorization_enum::LUMethod);

    std::cout << "Solution is: \n[";
    for (std::size_t t = 0; t < m; ++t)
    {
        std::cout << solution[t] << " ";
    }
    std::cout << "]\n";
}

void deviceDenseLUTest()
{
    std::cout << "==================================================\n";
    std::cout << "=========== Dense LU Solver - DEVICE =============\n";
    std::cout << "==================================================\n";

    deviceDenseLUTest<double>();
    deviceDenseLUTest<float>();
    deviceDenseLUPointersTest<double>();
    deviceDenseLUPointersTest<float>();

    std::cout << "==================================================\n";
}

#endif ///_LSS_DENSE_SOLVERS_CUDA_T_HPP_
