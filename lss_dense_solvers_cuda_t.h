#pragma once
#if !defined(_LSS_DENSE_SOLVERS_CUDA_T)
#define _LSS_DENSE_SOLVERS_CUDA_T


#include"lss_dense_solvers_cuda.h"
#include"lss_dense_solvers_policy.h"


void deviceDenseDefaultQRTest() {

    using lss_dense_solvers_cuda::FlatMatrix;
    using lss_dense_solvers_cuda::RealDenseSolverCUDA;
    using lss_dense_solvers_policy::DenseSolverQR;

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


    auto solution = rds.solve<DenseSolverQR>();

    std::cout << "Solution is: \n[";
    for (auto const& e : solution) {
        std::cout << e << " ";
    }
    std::cout << "]\n";

}


void deviceDenseDefaultQRPointersTest() {

    using lss_dense_solvers_cuda::FlatMatrix;
    using lss_dense_solvers_cuda::RealDenseSolverCUDA;
    using lss_dense_solvers_policy::DenseSolverQR;

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
    rds.initialize(m, m);

    // insert sparse matrix A and vector b:
    rds.setFlatDenseMatrix(std::move(fsm));
    rds.setRhs(b);

    double* solution = (double*)malloc(sizeof(double)*m);
    rds.solve<DenseSolverQR>(solution);

    std::cout << "Solution is: \n[";
    for (std::size_t t = 0; t < m;++t) {
        std::cout << solution[t] << " ";
    }
    std::cout << "]\n";


    free(solution);
}

void deviceDenseFloatQRTest() {

    using lss_dense_solvers_cuda::FlatMatrix;
    using lss_dense_solvers_cuda::RealDenseSolverCUDA;
    using lss_dense_solvers_policy::DenseSolverQR;

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

    // first create and populate the dense matrix:
    FlatMatrix<float> fsm;
    // size of the system:
    int const m = 3;
    // set number of columns and rows:
    fsm.setColumns(m); fsm.setRows(m);
    // populate the matrix:
    fsm.emplace_back(0, 0, 1.0f); fsm.emplace_back(0, 1, 2.0f); fsm.emplace_back(0, 2, 3.0f);
    fsm.emplace_back(1, 0, 4.0f); fsm.emplace_back(1, 1, 5.0f); fsm.emplace_back(1, 2, 6.0f);
    fsm.emplace_back(2, 0, 2.0f); fsm.emplace_back(2, 1, 1.0f); fsm.emplace_back(2, 2, 1.0f);


    // lets use std::vector to populate vector b:
    std::vector<float> b = { 6.0f,15.0f,4.0f };

    // create dense solver:
    RealDenseSolverCUDA<float> rds;

    // because we used default cstor we need to call initialize
    rds.initialize(m, m);

    // insert sparse matrix A and vector b:
    rds.setFlatDenseMatrix(std::move(fsm));
    rds.setRhs(b);

    auto solution = rds.solve<DenseSolverQR>();

    std::cout << "Solution is: \n[";
    for (auto const& e : solution) {
        std::cout << e << " ";
    }
    std::cout << "]\n";

}

void deviceDenseFloatQRPointersTest() {

    using lss_dense_solvers_cuda::FlatMatrix;
    using lss_dense_solvers_cuda::RealDenseSolverCUDA;
    using lss_dense_solvers_policy::DenseSolverQR;

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

    // first create and populate the dense matrix:
    FlatMatrix<float> fsm;
    // size of the system:
    int const m = 3;
    // set number of columns and rows:
    fsm.setColumns(m); fsm.setRows(m);
    // populate the matrix:
    fsm.emplace_back(0, 0, 1.0f); fsm.emplace_back(0, 1, 2.0f); fsm.emplace_back(0, 2, 3.0f);
    fsm.emplace_back(1, 0, 4.0f); fsm.emplace_back(1, 1, 5.0f); fsm.emplace_back(1, 2, 6.0f);
    fsm.emplace_back(2, 0, 2.0f); fsm.emplace_back(2, 1, 1.0f); fsm.emplace_back(2, 2, 1.0f);


    // lets use std::vector to populate vector b:
    std::vector<float> b = { 6.0f,15.0f,4.0f };

    // create dense solver:
    RealDenseSolverCUDA<float> rds;

    // because we used default cstor we need to call initialize
    rds.initialize(m, m);

    // insert sparse matrix A and vector b:
    rds.setFlatDenseMatrix(std::move(fsm));
    rds.setRhs(b);

    float* solution = (float*)malloc(sizeof(float) * m);
    rds.solve<DenseSolverQR>(solution);

    std::cout << "Solution is: \n[";
    for (std::size_t t = 0; t < m;++t) {
        std::cout << solution[t] << " ";
    }
    std::cout << "]\n";

    free(solution);
}

void deviceDenseQRTest() {
    std::cout << "==================================================\n";
    std::cout << "=========== Dense QR Solver - DEVICE =============\n";
    std::cout << "==================================================\n";

    deviceDenseDefaultQRTest();
    deviceDenseFloatQRTest();
    deviceDenseDefaultQRPointersTest();
    deviceDenseFloatQRPointersTest();

    std::cout << "==================================================\n";
}


void deviceDenseDefaultLUTest() {

    using lss_dense_solvers_cuda::FlatMatrix;
    using lss_dense_solvers_cuda::RealDenseSolverCUDA;
    using lss_dense_solvers_policy::DenseSolverLU;


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
    rds.initialize(m, m);

    // insert sparse matrix A and vector b:
    rds.setFlatDenseMatrix(std::move(fsm));
    rds.setRhs(b);

    auto solution = rds.solve<DenseSolverLU>();
    std::cout << "Solution is: \n[";
    for (auto const& e : solution) {
        std::cout << e << " ";
    }
    std::cout << "]\n";

}


void deviceDenseDefaultLUPointersTest() {

    using lss_dense_solvers_cuda::FlatMatrix;
    using lss_dense_solvers_cuda::RealDenseSolverCUDA;
    using lss_dense_solvers_policy::DenseSolverLU;


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
    rds.initialize(m, m);

    // insert sparse matrix A and vector b:
    rds.setFlatDenseMatrix(std::move(fsm));
    rds.setRhs(b);

    double* solution = (double*)malloc(sizeof(double) * m);
    rds.solve<DenseSolverLU>(solution);
    
    std::cout << "Solution is: \n[";
    for (std::size_t t = 0; t < m; ++t) {
        std::cout << solution[t] << " ";
    }
    std::cout << "]\n";

    free(solution);
}


void deviceDenseFloatLUTest() {

    using lss_dense_solvers_cuda::FlatMatrix;
    using lss_dense_solvers_cuda::RealDenseSolverCUDA;
    using lss_dense_solvers_policy::DenseSolverLU;

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

    // first create and populate the dense matrix:
    FlatMatrix<float> fsm;
    // size of the system:
    int const m = 3;
    // set number of columns and rows:
    fsm.setColumns(m); fsm.setRows(m);
    // populate the matrix:
    fsm.emplace_back(0, 0, 1.0f); fsm.emplace_back(0, 1, 2.0f); fsm.emplace_back(0, 2, 3.0f);
    fsm.emplace_back(1, 0, 4.0f); fsm.emplace_back(1, 1, 5.0f); fsm.emplace_back(1, 2, 6.0f);
    fsm.emplace_back(2, 0, 2.0f); fsm.emplace_back(2, 1, 1.0f); fsm.emplace_back(2, 2, 1.0f);


    // lets use std::vector to populate vector b:
    std::vector<float> b = { 6.0f,15.0f,4.0f };

    // create dense solver:
    RealDenseSolverCUDA<float> rds;

    // because we used default cstor we need to call initialize
    rds.initialize(m, m);

    // insert sparse matrix A and vector b:
    rds.setFlatDenseMatrix(std::move(fsm));
    rds.setRhs(b);

    auto solution = rds.solve<DenseSolverLU>();

    std::cout << "Solution is: \n[";
    for (auto const& e : solution) {
        std::cout << e << " ";
    }
    std::cout << "]\n";

}



void deviceDenseFloatLUPointersTest() {

    using lss_dense_solvers_cuda::FlatMatrix;
    using lss_dense_solvers_cuda::RealDenseSolverCUDA;
    using lss_dense_solvers_policy::DenseSolverLU;

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

    // first create and populate the dense matrix:
    FlatMatrix<float> fsm;
    // size of the system:
    int const m = 3;
    // set number of columns and rows:
    fsm.setColumns(m); fsm.setRows(m);
    // populate the matrix:
    fsm.emplace_back(0, 0, 1.0f); fsm.emplace_back(0, 1, 2.0f); fsm.emplace_back(0, 2, 3.0f);
    fsm.emplace_back(1, 0, 4.0f); fsm.emplace_back(1, 1, 5.0f); fsm.emplace_back(1, 2, 6.0f);
    fsm.emplace_back(2, 0, 2.0f); fsm.emplace_back(2, 1, 1.0f); fsm.emplace_back(2, 2, 1.0f);


    // lets use std::vector to populate vector b:
    std::vector<float> b = { 6.0f,15.0f,4.0f };

    // create dense solver:
    RealDenseSolverCUDA<float> rds;

    // because we used default cstor we need to call initialize
    rds.initialize(m, m);

    // insert sparse matrix A and vector b:
    rds.setFlatDenseMatrix(std::move(fsm));
    rds.setRhs(b);

    float* solution = (float*)malloc(sizeof(float) * m);
    rds.solve<DenseSolverLU>(solution);

    std::cout << "Solution is: \n[";
    for (std::size_t t = 0; t < m; ++t) {
        std::cout << solution[t] << " ";
    }
    std::cout << "]\n";

    free(solution);
}

void deviceDenseLUTest() {
    std::cout << "==================================================\n";
    std::cout << "=========== Dense LU Solver - DEVICE =============\n";
    std::cout << "==================================================\n";

    deviceDenseDefaultLUTest();
    deviceDenseDefaultLUPointersTest();
    deviceDenseFloatLUTest();
    deviceDenseFloatLUPointersTest();

    std::cout << "==================================================\n";
}





#endif ///_LSS_DENSE_SOLVERS_CUDA_T