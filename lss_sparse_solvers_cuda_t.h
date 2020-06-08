#pragma once
#if !defined(_LSS_SPARSE_SOLVERS_CUDA_T)
#define _LSS_SPARSE_SOLVERS_CUDA_T


#include"lss_sparse_solvers_cuda.h"
#include"lss_sparse_solvers_policy.h"

void deviceSparseDefaultQRTest() {

    using lss_sparse_solvers_cuda::FlatMatrix;
    using lss_sparse_solvers_cuda::MemorySpace;
    using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
    using lss_sparse_solvers_policy::SparseSolverDeviceQR;
    using lss_sparse_solvers_policy::SparseSolverHostQR;
    using lss_sparse_solvers_policy::SparseSolverHostLU;


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

    // first create and populate the sparse matrix:
    FlatMatrix<double> fsm;
    // size of the system:
    int const m = 6;
    // set number of columns and rows:
    fsm.setColumns(m); fsm.setRows(m);
    // populate the matrix:
    fsm.emplace_back(0, 0, 1.0); fsm.emplace_back(0, 1, 2.0);
    fsm.emplace_back(1, 0, 3.0); fsm.emplace_back(1, 1, 4.0); fsm.emplace_back(1, 2, 5.0);
    fsm.emplace_back(2, 1, 6.0); fsm.emplace_back(2, 2, 7.0); fsm.emplace_back(2, 3, 8.0);
    fsm.emplace_back(3, 2, 9.0); fsm.emplace_back(3, 3, 10.0); fsm.emplace_back(3, 4, 11.0);
    fsm.emplace_back(4, 3, 12.0); fsm.emplace_back(4, 4, 13.0); fsm.emplace_back(4, 5, 14.0);
    fsm.emplace_back(5, 4, 15.0); fsm.emplace_back(5, 5, 16.0);

    // lets use std::vector to populate vector b:
    std::vector<double> b = { 0.0,2.0,4.0,6.0,8.0,10.0 };

    // create sparse solver on DEVICE:
    RealSparseSolverCUDA<MemorySpace::Device, double> rss;

    // because we used default cstor we need to call initialize
    rss.initialize(m);

    // insert sparse matrix A and vector b:
    rss.setFlatSparseMatrix(std::move(fsm));
    rss.setRhs(b);


    auto solution = rss.solve<SparseSolverDeviceQR>();
    std::cout << "Solution is: \n[";
    for (auto const& e : solution) {
        std::cout << e << " ";
    }
    std::cout << "]\n";

}

void deviceSparseFloatQRTest() {

    using lss_sparse_solvers_cuda::FlatMatrix;
    using lss_sparse_solvers_cuda::MemorySpace;
    using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
    using lss_sparse_solvers_policy::SparseSolverDeviceQR;



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

    // first create and populate the sparse matrix:
    FlatMatrix<float> fsm;
    // size of the system:
    int const m = 6;
    // set number of columns and rows:
    fsm.setColumns(m); fsm.setRows(m);
    // populate the matrix:
    fsm.emplace_back(0, 0, 1.0f); fsm.emplace_back(0, 1, 2.0f);
    fsm.emplace_back(1, 0, 3.0f); fsm.emplace_back(1, 1, 4.0f); fsm.emplace_back(1, 2, 5.0f);
    fsm.emplace_back(2, 1, 6.0f); fsm.emplace_back(2, 2, 7.0f); fsm.emplace_back(2, 3, 8.0f);
    fsm.emplace_back(3, 2, 9.0f); fsm.emplace_back(3, 3, 10.0f); fsm.emplace_back(3, 4, 11.0f);
    fsm.emplace_back(4, 3, 12.0f); fsm.emplace_back(4, 4, 13.0f); fsm.emplace_back(4, 5, 14.0f);
    fsm.emplace_back(5, 4, 15.0f); fsm.emplace_back(5, 5, 16.0f);

    // lets use std::vector to populate vector b:
    std::vector<float> b = { 0.0f,2.0f,4.0f,6.0f,8.0f,10.0f };

    // create sparse solver on DEVICE:
    RealSparseSolverCUDA<MemorySpace::Device, float> rss;

    // because we used default cstor we need to call initialize
    rss.initialize(m);

    // insert sparse matrix A and vector b:
    rss.setFlatSparseMatrix(std::move(fsm));
    rss.setRhs(b);


    auto solution = rss.solve<SparseSolverDeviceQR>();
    std::cout << "Solution is: \n[";
    for (auto const& e : solution) {
        std::cout << e << " ";
    }
    std::cout << "]\n";

}

void deviceSparseDefaultQRPointerTest() {

    using lss_sparse_solvers_cuda::FlatMatrix;
    using lss_sparse_solvers_cuda::MemorySpace;
    using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
    using lss_sparse_solvers_policy::SparseSolverDeviceQR;



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

    // first create and populate the sparse matrix:
    FlatMatrix<double> fsm;
    // size of the system:
    int const m = 6;
    // set number of columns and rows:
    fsm.setColumns(m); fsm.setRows(m);
    // populate the matrix:
    fsm.emplace_back(0, 0, 1.0); fsm.emplace_back(0, 1, 2.0);
    fsm.emplace_back(1, 0, 3.0); fsm.emplace_back(1, 1, 4.0); fsm.emplace_back(1, 2, 5.0);
    fsm.emplace_back(2, 1, 6.0); fsm.emplace_back(2, 2, 7.0); fsm.emplace_back(2, 3, 8.0);
    fsm.emplace_back(3, 2, 9.0); fsm.emplace_back(3, 3, 10.0); fsm.emplace_back(3, 4, 11.0);
    fsm.emplace_back(4, 3, 12.0); fsm.emplace_back(4, 4, 13.0); fsm.emplace_back(4, 5, 14.0);
    fsm.emplace_back(5, 4, 15.0); fsm.emplace_back(5, 5, 16.0);

    // lets use std::vector to populate vector b:
    std::vector<double> b = { 0.0,2.0,4.0,6.0,8.0,10.0 };

    // create sparse solver on DEVICE:
    RealSparseSolverCUDA<MemorySpace::Device, double> rss;

    // because we used default cstor we need to call initialize
    rss.initialize(m);

    // insert sparse matrix A and vector b:
    rss.setFlatSparseMatrix(std::move(fsm));
    rss.setRhs(b);

    double* solution = (double*)malloc(sizeof(double) * m);
    rss.solve<SparseSolverDeviceQR>(solution);
    
    std::cout << "Solution is: \n[";
    for (std::size_t t = 0; t < m;++t) {
        std::cout << solution[t] << " ";
    }
    std::cout << "]\n";

    free(solution);
}


void deviceSparseFloatQRPointerTest() {

    using lss_sparse_solvers_cuda::FlatMatrix;
    using lss_sparse_solvers_cuda::MemorySpace;
    using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
    using lss_sparse_solvers_policy::SparseSolverDeviceQR;



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

    // first create and populate the sparse matrix:
    FlatMatrix<float> fsm;
    // size of the system:
    int const m = 6;
    // set number of columns and rows:
    fsm.setColumns(m); fsm.setRows(m);
    // populate the matrix:
    fsm.emplace_back(0, 0, 1.0f); fsm.emplace_back(0, 1, 2.0f);
    fsm.emplace_back(1, 0, 3.0f); fsm.emplace_back(1, 1, 4.0f); fsm.emplace_back(1, 2, 5.0f);
    fsm.emplace_back(2, 1, 6.0f); fsm.emplace_back(2, 2, 7.0f); fsm.emplace_back(2, 3, 8.0f);
    fsm.emplace_back(3, 2, 9.0f); fsm.emplace_back(3, 3, 10.0f); fsm.emplace_back(3, 4, 11.0f);
    fsm.emplace_back(4, 3, 12.0f); fsm.emplace_back(4, 4, 13.0f); fsm.emplace_back(4, 5, 14.0f);
    fsm.emplace_back(5, 4, 15.0f); fsm.emplace_back(5, 5, 16.0f);

    // lets use std::vector to populate vector b:
    std::vector<float> b = { 0.0f,2.0f,4.0f,6.0f,8.0f,10.0f };

    // create sparse solver on DEVICE:
    RealSparseSolverCUDA<MemorySpace::Device, float> rss;

    // because we used default cstor we need to call initialize
    rss.initialize(m);

    // insert sparse matrix A and vector b:
    rss.setFlatSparseMatrix(std::move(fsm));
    rss.setRhs(b);

    float* solution = (float*)malloc(sizeof(float) * m);
    rss.solve<SparseSolverDeviceQR>(solution);
    std::cout << "Solution is: \n[";
    for (std::size_t t = 0; t < m;++t) {
        std::cout << solution[t] << " ";
    }
    std::cout << "]\n";

}


void deviceSparseQRTest() {
    std::cout << "==================================================\n";
    std::cout << "=========== Sparse QR Solver - DEVICE ============\n";
    std::cout << "==================================================\n";

    deviceSparseDefaultQRTest();
    deviceSparseDefaultQRPointerTest();
    deviceSparseFloatQRTest();
    deviceSparseFloatQRPointerTest();

    std::cout << "==================================================\n";
}



void hostSparseDefaultQRTest() {

    using lss_sparse_solvers_cuda::FlatMatrix;
    using lss_sparse_solvers_cuda::MemorySpace;
    using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
    using lss_sparse_solvers_policy::SparseSolverDeviceQR;
    using lss_sparse_solvers_policy::SparseSolverHostQR;
    using lss_sparse_solvers_policy::SparseSolverHostLU;


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

    // first create and populate the sparse matrix:
    FlatMatrix<double> fsm;
    // size of the system:
    int const m = 6;
    // set number of columns and rows:
    fsm.setColumns(m); fsm.setRows(m);
    // populate the matrix:
    fsm.emplace_back(0, 0, 1.0); fsm.emplace_back(0, 1, 2.0);
    fsm.emplace_back(1, 0, 3.0); fsm.emplace_back(1, 1, 4.0); fsm.emplace_back(1, 2, 5.0);
    fsm.emplace_back(2, 1, 6.0); fsm.emplace_back(2, 2, 7.0); fsm.emplace_back(2, 3, 8.0);
    fsm.emplace_back(3, 2, 9.0); fsm.emplace_back(3, 3, 10.0); fsm.emplace_back(3, 4, 11.0);
    fsm.emplace_back(4, 3, 12.0); fsm.emplace_back(4, 4, 13.0); fsm.emplace_back(4, 5, 14.0);
    fsm.emplace_back(5, 4, 15.0); fsm.emplace_back(5, 5, 16.0);

    // lets use std::vector to populate vector b:
    std::vector<double> b = { 0.0,2.0,4.0,6.0,8.0,10.0 };

    // create sparse solver on DEVICE:
    RealSparseSolverCUDA<MemorySpace::Host, double> rss;

    // because we used default cstor we need to call initialize
    rss.initialize(m);

    // insert sparse matrix A and vector b:
    rss.setFlatSparseMatrix(std::move(fsm));
    rss.setRhs(b);


    auto solution = rss.solve<SparseSolverHostQR>();
    std::cout << "Solution is: \n[";
    for (auto const& e : solution) {
        std::cout << e << " ";
    }
    std::cout << "]\n";

}

void hostSparseFloatQRTest() {

    using lss_sparse_solvers_cuda::FlatMatrix;
    using lss_sparse_solvers_cuda::MemorySpace;
    using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
    using lss_sparse_solvers_policy::SparseSolverHostQR;



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

    // first create and populate the sparse matrix:
    FlatMatrix<float> fsm;
    // size of the system:
    int const m = 6;
    // set number of columns and rows:
    fsm.setColumns(m); fsm.setRows(m);
    // populate the matrix:
    fsm.emplace_back(0, 0, 1.0f); fsm.emplace_back(0, 1, 2.0f);
    fsm.emplace_back(1, 0, 3.0f); fsm.emplace_back(1, 1, 4.0f); fsm.emplace_back(1, 2, 5.0f);
    fsm.emplace_back(2, 1, 6.0f); fsm.emplace_back(2, 2, 7.0f); fsm.emplace_back(2, 3, 8.0f);
    fsm.emplace_back(3, 2, 9.0f); fsm.emplace_back(3, 3, 10.0f); fsm.emplace_back(3, 4, 11.0f);
    fsm.emplace_back(4, 3, 12.0f); fsm.emplace_back(4, 4, 13.0f); fsm.emplace_back(4, 5, 14.0f);
    fsm.emplace_back(5, 4, 15.0f); fsm.emplace_back(5, 5, 16.0f);

    // lets use std::vector to populate vector b:
    std::vector<float> b = { 0.0f,2.0f,4.0f,6.0f,8.0f,10.0f };

    // create sparse solver on DEVICE:
    RealSparseSolverCUDA<MemorySpace::Host, float> rss;

    // because we used default cstor we need to call initialize
    rss.initialize(m);

    // insert sparse matrix A and vector b:
    rss.setFlatSparseMatrix(std::move(fsm));
    rss.setRhs(b);


    auto solution = rss.solve<SparseSolverHostQR>();
    std::cout << "Solution is: \n[";
    for (auto const& e : solution) {
        std::cout << e << " ";
    }
    std::cout << "]\n";

}

void hostSparseDefaultQRPointerTest() {

    using lss_sparse_solvers_cuda::FlatMatrix;
    using lss_sparse_solvers_cuda::MemorySpace;
    using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
    using lss_sparse_solvers_policy::SparseSolverHostQR;



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

    // first create and populate the sparse matrix:
    FlatMatrix<double> fsm;
    // size of the system:
    int const m = 6;
    // set number of columns and rows:
    fsm.setColumns(m); fsm.setRows(m);
    // populate the matrix:
    fsm.emplace_back(0, 0, 1.0); fsm.emplace_back(0, 1, 2.0);
    fsm.emplace_back(1, 0, 3.0); fsm.emplace_back(1, 1, 4.0); fsm.emplace_back(1, 2, 5.0);
    fsm.emplace_back(2, 1, 6.0); fsm.emplace_back(2, 2, 7.0); fsm.emplace_back(2, 3, 8.0);
    fsm.emplace_back(3, 2, 9.0); fsm.emplace_back(3, 3, 10.0); fsm.emplace_back(3, 4, 11.0);
    fsm.emplace_back(4, 3, 12.0); fsm.emplace_back(4, 4, 13.0); fsm.emplace_back(4, 5, 14.0);
    fsm.emplace_back(5, 4, 15.0); fsm.emplace_back(5, 5, 16.0);

    // lets use std::vector to populate vector b:
    std::vector<double> b = { 0.0,2.0,4.0,6.0,8.0,10.0 };

    // create sparse solver on DEVICE:
    RealSparseSolverCUDA<MemorySpace::Host, double> rss;

    // because we used default cstor we need to call initialize
    rss.initialize(m);

    // insert sparse matrix A and vector b:
    rss.setFlatSparseMatrix(std::move(fsm));
    rss.setRhs(b);

    double* solution = (double*)malloc(sizeof(double) * m);
    rss.solve<SparseSolverHostQR>(solution);

    std::cout << "Solution is: \n[";
    for (std::size_t t = 0; t < m; ++t) {
        std::cout << solution[t] << " ";
    }
    std::cout << "]\n";

    free(solution);
}


void hostSparseFloatQRPointerTest() {

    using lss_sparse_solvers_cuda::FlatMatrix;
    using lss_sparse_solvers_cuda::MemorySpace;
    using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
    using lss_sparse_solvers_policy::SparseSolverHostQR;



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

    // first create and populate the sparse matrix:
    FlatMatrix<float> fsm;
    // size of the system:
    int const m = 6;
    // set number of columns and rows:
    fsm.setColumns(m); fsm.setRows(m);
    // populate the matrix:
    fsm.emplace_back(0, 0, 1.0f); fsm.emplace_back(0, 1, 2.0f);
    fsm.emplace_back(1, 0, 3.0f); fsm.emplace_back(1, 1, 4.0f); fsm.emplace_back(1, 2, 5.0f);
    fsm.emplace_back(2, 1, 6.0f); fsm.emplace_back(2, 2, 7.0f); fsm.emplace_back(2, 3, 8.0f);
    fsm.emplace_back(3, 2, 9.0f); fsm.emplace_back(3, 3, 10.0f); fsm.emplace_back(3, 4, 11.0f);
    fsm.emplace_back(4, 3, 12.0f); fsm.emplace_back(4, 4, 13.0f); fsm.emplace_back(4, 5, 14.0f);
    fsm.emplace_back(5, 4, 15.0f); fsm.emplace_back(5, 5, 16.0f);

    // lets use std::vector to populate vector b:
    std::vector<float> b = { 0.0f,2.0f,4.0f,6.0f,8.0f,10.0f };

    // create sparse solver on DEVICE:
    RealSparseSolverCUDA<MemorySpace::Host, float> rss;

    // because we used default cstor we need to call initialize
    rss.initialize(m);

    // insert sparse matrix A and vector b:
    rss.setFlatSparseMatrix(std::move(fsm));
    rss.setRhs(b);

    float* solution = (float*)malloc(sizeof(float) * m);
    rss.solve<SparseSolverHostQR>(solution);
    std::cout << "Solution is: \n[";
    for (std::size_t t = 0; t < m; ++t) {
        std::cout << solution[t] << " ";
    }
    std::cout << "]\n";

}


void hostSparseQRTest() {
    std::cout << "==================================================\n";
    std::cout << "=========== Sparse QR Solver - HOST ==============\n";
    std::cout << "==================================================\n";

    hostSparseDefaultQRTest();
    hostSparseDefaultQRPointerTest();
    hostSparseFloatQRTest();
    hostSparseFloatQRPointerTest();

    std::cout << "==================================================\n";
}

#endif ///_LSS_SPARSE_SOLVERS_CUDA_T