#pragma once
#if !defined(_LSS_DENSE_SOLVERS_CUDA_T)
#define _LSS_DENSE_SOLVERS_CUDA_T

#include "dense_solvers/lss_dense_solvers_cuda.h"
#include "dense_solvers/lss_dense_solvers_policy.h"

void deviceDenseDefaultQRTest() {
  using lss_dense_solvers::flat_matrix;
  using lss_dense_solvers::real_dense_solver_cuda;
  using lss_dense_solvers_policy::dense_solver_qr;

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
  flat_matrix<double> fsm;
  // size of the system:
  int const m = 3;
  // set number of columns and rows:
  fsm.set_columns(m);
  fsm.set_rows(m);
  // populate the matrix:
  fsm.emplace_back(0, 0, 1.0);
  fsm.emplace_back(0, 1, 2.0);
  fsm.emplace_back(0, 2, 3.0);
  fsm.emplace_back(1, 0, 4.0);
  fsm.emplace_back(1, 1, 5.0);
  fsm.emplace_back(1, 2, 6.0);
  fsm.emplace_back(2, 0, 2.0);
  fsm.emplace_back(2, 1, 1.0);
  fsm.emplace_back(2, 2, 1.0);

  // lets use std::vector to populate vector b:
  std::vector<double> b = {6.0, 15.0, 4.0};

  // create dense solver:
  real_dense_solver_cuda<double> rds;

  // because we used default cstor we need to call initialize
  rds.initialize(m, m);

  // insert sparse matrix A and vector b:
  rds.set_flat_dense_matrix(std::move(fsm));
  rds.set_rhs(b);

  auto solution = rds.solve<dense_solver_qr>();

  std::cout << "Solution is: \n[";
  for (auto const& e : solution) {
    std::cout << e << " ";
  }
  std::cout << "]\n";
}

void deviceDenseDefaultQRPointersTest() {
  using lss_dense_solvers::flat_matrix;
  using lss_dense_solvers::real_dense_solver_cuda;
  using lss_dense_solvers_policy::dense_solver_qr;

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
  flat_matrix<double> fsm;
  // size of the system:
  int const m = 3;
  // set number of columns and rows:
  fsm.set_columns(m);
  fsm.set_rows(m);
  // populate the matrix:
  fsm.emplace_back(0, 0, 1.0);
  fsm.emplace_back(0, 1, 2.0);
  fsm.emplace_back(0, 2, 3.0);
  fsm.emplace_back(1, 0, 4.0);
  fsm.emplace_back(1, 1, 5.0);
  fsm.emplace_back(1, 2, 6.0);
  fsm.emplace_back(2, 0, 2.0);
  fsm.emplace_back(2, 1, 1.0);
  fsm.emplace_back(2, 2, 1.0);

  // lets use std::vector to populate vector b:
  std::vector<double> b = {6.0, 15.0, 4.0};

  // create dense solver:
  real_dense_solver_cuda<double> rds;

  // because we used default cstor we need to call initialize
  rds.initialize(m, m);

  // insert sparse matrix A and vector b:
  rds.set_flat_dense_matrix(std::move(fsm));
  rds.set_rhs(b);

  std::vector<double> solution(m);
  rds.solve<dense_solver_qr>(solution);

  std::cout << "Solution is: \n[";
  for (std::size_t t = 0; t < m; ++t) {
    std::cout << solution[t] << " ";
  }
  std::cout << "]\n";
}

void deviceDenseFloatQRTest() {
  using lss_dense_solvers::flat_matrix;
  using lss_dense_solvers::real_dense_solver_cuda;
  using lss_dense_solvers_policy::dense_solver_qr;

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
  flat_matrix<float> fsm;
  // size of the system:
  int const m = 3;
  // set number of columns and rows:
  fsm.set_columns(m);
  fsm.set_rows(m);
  // populate the matrix:
  fsm.emplace_back(0, 0, 1.0f);
  fsm.emplace_back(0, 1, 2.0f);
  fsm.emplace_back(0, 2, 3.0f);
  fsm.emplace_back(1, 0, 4.0f);
  fsm.emplace_back(1, 1, 5.0f);
  fsm.emplace_back(1, 2, 6.0f);
  fsm.emplace_back(2, 0, 2.0f);
  fsm.emplace_back(2, 1, 1.0f);
  fsm.emplace_back(2, 2, 1.0f);

  // lets use std::vector to populate vector b:
  std::vector<float> b = {6.0f, 15.0f, 4.0f};

  // create dense solver:
  real_dense_solver_cuda<float> rds;

  // because we used default cstor we need to call initialize
  rds.initialize(m, m);

  // insert sparse matrix A and vector b:
  rds.set_flat_dense_matrix(std::move(fsm));
  rds.set_rhs(b);

  auto solution = rds.solve<dense_solver_qr>();

  std::cout << "Solution is: \n[";
  for (auto const& e : solution) {
    std::cout << e << " ";
  }
  std::cout << "]\n";
}

void deviceDenseFloatQRPointersTest() {
  using lss_dense_solvers::flat_matrix;
  using lss_dense_solvers::real_dense_solver_cuda;
  using lss_dense_solvers_policy::dense_solver_qr;

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
  flat_matrix<float> fsm;
  // size of the system:
  int const m = 3;
  // set number of columns and rows:
  fsm.set_columns(m);
  fsm.set_rows(m);
  // populate the matrix:
  fsm.emplace_back(0, 0, 1.0f);
  fsm.emplace_back(0, 1, 2.0f);
  fsm.emplace_back(0, 2, 3.0f);
  fsm.emplace_back(1, 0, 4.0f);
  fsm.emplace_back(1, 1, 5.0f);
  fsm.emplace_back(1, 2, 6.0f);
  fsm.emplace_back(2, 0, 2.0f);
  fsm.emplace_back(2, 1, 1.0f);
  fsm.emplace_back(2, 2, 1.0f);

  // lets use std::vector to populate vector b:
  std::vector<float> b = {6.0f, 15.0f, 4.0f};

  // create dense solver:
  real_dense_solver_cuda<float> rds;

  // because we used default cstor we need to call initialize
  rds.initialize(m, m);

  // insert sparse matrix A and vector b:
  rds.set_flat_dense_matrix(std::move(fsm));
  rds.set_rhs(b);

  std::vector<float> solution(m);
  rds.solve<dense_solver_qr>(solution);

  std::cout << "Solution is: \n[";
  for (std::size_t t = 0; t < m; ++t) {
    std::cout << solution[t] << " ";
  }
  std::cout << "]\n";
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
  using lss_dense_solvers::flat_matrix;
  using lss_dense_solvers::real_dense_solver_cuda;
  using lss_dense_solvers_policy::dense_solver_lu;

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
  flat_matrix<double> fsm;
  // size of the system:
  int const m = 3;
  // set number of columns and rows:
  fsm.set_columns(m);
  fsm.set_rows(m);
  // populate the matrix:
  fsm.emplace_back(0, 0, 1.0);
  fsm.emplace_back(0, 1, 2.0);
  fsm.emplace_back(0, 2, 3.0);
  fsm.emplace_back(1, 0, 4.0);
  fsm.emplace_back(1, 1, 5.0);
  fsm.emplace_back(1, 2, 6.0);
  fsm.emplace_back(2, 0, 2.0);
  fsm.emplace_back(2, 1, 1.0);
  fsm.emplace_back(2, 2, 1.0);

  // lets use std::vector to populate vector b:
  std::vector<double> b = {6.0, 15.0, 4.0};

  // create dense solver:
  real_dense_solver_cuda<double> rds;

  // because we used default cstor we need to call initialize
  rds.initialize(m, m);

  // insert sparse matrix A and vector b:
  rds.set_flat_dense_matrix(std::move(fsm));
  rds.set_rhs(b);

  auto solution = rds.solve<dense_solver_lu>();
  std::cout << "Solution is: \n[";
  for (auto const& e : solution) {
    std::cout << e << " ";
  }
  std::cout << "]\n";
}

void deviceDenseDefaultLUPointersTest() {
  using lss_dense_solvers::flat_matrix;
  using lss_dense_solvers::real_dense_solver_cuda;
  using lss_dense_solvers_policy::dense_solver_lu;

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
  flat_matrix<double> fsm;
  // size of the system:
  int const m = 3;
  // set number of columns and rows:
  fsm.set_columns(m);
  fsm.set_rows(m);
  // populate the matrix:
  fsm.emplace_back(0, 0, 1.0);
  fsm.emplace_back(0, 1, 2.0);
  fsm.emplace_back(0, 2, 3.0);
  fsm.emplace_back(1, 0, 4.0);
  fsm.emplace_back(1, 1, 5.0);
  fsm.emplace_back(1, 2, 6.0);
  fsm.emplace_back(2, 0, 2.0);
  fsm.emplace_back(2, 1, 1.0);
  fsm.emplace_back(2, 2, 1.0);

  // lets use std::vector to populate vector b:
  std::vector<double> b = {6.0, 15.0, 4.0};

  // create dense solver:
  real_dense_solver_cuda<double> rds;

  // because we used default cstor we need to call initialize
  rds.initialize(m, m);

  // insert sparse matrix A and vector b:
  rds.set_flat_dense_matrix(std::move(fsm));
  rds.set_rhs(b);

  std::vector<double> solution(m);
  rds.solve<dense_solver_lu>(solution);

  std::cout << "Solution is: \n[";
  for (std::size_t t = 0; t < m; ++t) {
    std::cout << solution[t] << " ";
  }
  std::cout << "]\n";
}

void deviceDenseFloatLUTest() {
  using lss_dense_solvers::flat_matrix;
  using lss_dense_solvers::real_dense_solver_cuda;
  using lss_dense_solvers_policy::dense_solver_lu;

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
  flat_matrix<float> fsm;
  // size of the system:
  int const m = 3;
  // set number of columns and rows:
  fsm.set_columns(m);
  fsm.set_rows(m);
  // populate the matrix:
  fsm.emplace_back(0, 0, 1.0f);
  fsm.emplace_back(0, 1, 2.0f);
  fsm.emplace_back(0, 2, 3.0f);
  fsm.emplace_back(1, 0, 4.0f);
  fsm.emplace_back(1, 1, 5.0f);
  fsm.emplace_back(1, 2, 6.0f);
  fsm.emplace_back(2, 0, 2.0f);
  fsm.emplace_back(2, 1, 1.0f);
  fsm.emplace_back(2, 2, 1.0f);

  // lets use std::vector to populate vector b:
  std::vector<float> b = {6.0f, 15.0f, 4.0f};

  // create dense solver:
  real_dense_solver_cuda<float> rds;

  // because we used default cstor we need to call initialize
  rds.initialize(m, m);

  // insert sparse matrix A and vector b:
  rds.set_flat_dense_matrix(std::move(fsm));
  rds.set_rhs(b);

  auto solution = rds.solve<dense_solver_lu>();

  std::cout << "Solution is: \n[";
  for (auto const& e : solution) {
    std::cout << e << " ";
  }
  std::cout << "]\n";
}

void deviceDenseFloatLUPointersTest() {
  using lss_dense_solvers::flat_matrix;
  using lss_dense_solvers::real_dense_solver_cuda;
  using lss_dense_solvers_policy::dense_solver_lu;

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
  flat_matrix<float> fsm;
  // size of the system:
  int const m = 3;
  // set number of columns and rows:
  fsm.set_columns(m);
  fsm.set_rows(m);
  // populate the matrix:
  fsm.emplace_back(0, 0, 1.0f);
  fsm.emplace_back(0, 1, 2.0f);
  fsm.emplace_back(0, 2, 3.0f);
  fsm.emplace_back(1, 0, 4.0f);
  fsm.emplace_back(1, 1, 5.0f);
  fsm.emplace_back(1, 2, 6.0f);
  fsm.emplace_back(2, 0, 2.0f);
  fsm.emplace_back(2, 1, 1.0f);
  fsm.emplace_back(2, 2, 1.0f);

  // lets use std::vector to populate vector b:
  std::vector<float> b = {6.0f, 15.0f, 4.0f};

  // create dense solver:
  real_dense_solver_cuda<float> rds;

  // because we used default cstor we need to call initialize
  rds.initialize(m, m);

  // insert sparse matrix A and vector b:
  rds.set_flat_dense_matrix(std::move(fsm));
  rds.set_rhs(b);

  std::vector<float> solution(m);
  rds.solve<dense_solver_lu>(solution);

  std::cout << "Solution is: \n[";
  for (std::size_t t = 0; t < m; ++t) {
    std::cout << solution[t] << " ";
  }
  std::cout << "]\n";
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

#endif  ///_LSS_DENSE_SOLVERS_CUDA_T
