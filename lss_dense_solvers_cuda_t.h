#pragma once
#if !defined(_LSS_DENSE_SOLVERS_CUDA_T)
#define _LSS_DENSE_SOLVERS_CUDA_T


#include"lss_dense_solvers_cuda.h"
#include"lss_dense_solvers_policy.h"


void deviceDenseQRtest() {

    using lss_dense_solvers_cuda::FlatMatrix;
    using lss_dense_solvers_cuda::RealDenseSolverCUDA;
    using lss_dense_solvers_policy::DenseSolverLU;


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

    // first create and populate the dense matrix:
    FlatMatrix<double> fsm;
    // size of the system:
    int const m = 3;
    // set number of columns and rows:
    fsm.setColumns(m); fsm.setRows(m);
    // populate the matrix:
    fsm.emplace_back(0, 0, 1.0); fsm.emplace_back(0, 1, 2.0); fsm.emplace_back(0, 2, 3.0);
    fsm.emplace_back(1, 0, 4.0); fsm.emplace_back(1, 1, 5.0); fsm.emplace_back(1, 2, 6.0);
    fsm.emplace_back(2, 0, 2.0); fsm.emplace_back(2, 1, 1.0); fsm.emplace_back(2, 2, 1.0);


    // lets use std::vector to populate vector b:
    std::vector<double> b = { 6.0,15.0,4.0 };

    // create dense solver:
    RealDenseSolverCUDA<double> rds;

    // because we used default cstor we need to call initialize
    rds.initialize(m,m);

    // insert sparse matrix A and vector b:
    rds.setFlatDenseMatrix(std::move(fsm));
    rds.setRhs(b);

    auto solution = rds.solve<DenseSolverLU>();

    for (auto const& e : solution) {
        std::cout << e << ", ";
    }


}








#endif ///_LSS_DENSE_SOLVERS_CUDA_T