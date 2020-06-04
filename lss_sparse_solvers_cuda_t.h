#pragma once
#if !defined(_LSS_SPARSE_SOLVERS_CUDA_T)
#define _LSS_SPARSE_SOLVERS_CUDA_T


#include"lss_sparse_solvers_cuda.h"


void deviceQRtest() {

    using lss_sparse_solvers_cuda::FlatMatrix;
    using lss_sparse_solvers_cuda::MemorySpace;
    using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
    using lss_sparse_solvers_cuda::SparseSolverFactorizationDevice;
    using lss_sparse_solvers_cuda::SparseSolverFactorizationHost;

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

    auto solution = rss.solve(SparseSolverFactorizationDevice::QR);

    for (auto const& e : solution) {
        std::cout << e << ", ";
    }


}










#endif ///_LSS_SPARSE_SOLVERS_CUDA_T