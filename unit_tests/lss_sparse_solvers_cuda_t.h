#pragma once
#if !defined(_LSS_SPARSE_SOLVERS_CUDA_T)
#define _LSS_SPARSE_SOLVERS_CUDA_T

#include "sparse_solvers/lss_sparse_solvers_cuda.h"
#include "sparse_solvers/lss_sparse_solvers_policy.h"

void deviceSparseDefaultQRTest() {
  using lss_sparse_solvers_cuda::FlatMatrix;
  using lss_sparse_solvers_cuda::MemorySpace;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_sparse_solvers_policy::SparseSolverDeviceQR;
  using lss_sparse_solvers_policy::SparseSolverHostLU;
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
  fsm.setColumns(m);
  fsm.setRows(m);
  // populate the matrix:
  fsm.emplace_back(0, 0, 1.0);
  fsm.emplace_back(0, 1, 2.0);
  fsm.emplace_back(1, 0, 3.0);
  fsm.emplace_back(1, 1, 4.0);
  fsm.emplace_back(1, 2, 5.0);
  fsm.emplace_back(2, 1, 6.0);
  fsm.emplace_back(2, 2, 7.0);
  fsm.emplace_back(2, 3, 8.0);
  fsm.emplace_back(3, 2, 9.0);
  fsm.emplace_back(3, 3, 10.0);
  fsm.emplace_back(3, 4, 11.0);
  fsm.emplace_back(4, 3, 12.0);
  fsm.emplace_back(4, 4, 13.0);
  fsm.emplace_back(4, 5, 14.0);
  fsm.emplace_back(5, 4, 15.0);
  fsm.emplace_back(5, 5, 16.0);

  // lets use std::vector to populate vector b:
  std::vector<double> b = {0.0, 2.0, 4.0, 6.0, 8.0, 10.0};

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
  fsm.setColumns(m);
  fsm.setRows(m);
  // populate the matrix:
  fsm.emplace_back(0, 0, 1.0f);
  fsm.emplace_back(0, 1, 2.0f);
  fsm.emplace_back(1, 0, 3.0f);
  fsm.emplace_back(1, 1, 4.0f);
  fsm.emplace_back(1, 2, 5.0f);
  fsm.emplace_back(2, 1, 6.0f);
  fsm.emplace_back(2, 2, 7.0f);
  fsm.emplace_back(2, 3, 8.0f);
  fsm.emplace_back(3, 2, 9.0f);
  fsm.emplace_back(3, 3, 10.0f);
  fsm.emplace_back(3, 4, 11.0f);
  fsm.emplace_back(4, 3, 12.0f);
  fsm.emplace_back(4, 4, 13.0f);
  fsm.emplace_back(4, 5, 14.0f);
  fsm.emplace_back(5, 4, 15.0f);
  fsm.emplace_back(5, 5, 16.0f);

  // lets use std::vector to populate vector b:
  std::vector<float> b = {0.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.0f};

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
  fsm.setColumns(m);
  fsm.setRows(m);
  // populate the matrix:
  fsm.emplace_back(0, 0, 1.0);
  fsm.emplace_back(0, 1, 2.0);
  fsm.emplace_back(1, 0, 3.0);
  fsm.emplace_back(1, 1, 4.0);
  fsm.emplace_back(1, 2, 5.0);
  fsm.emplace_back(2, 1, 6.0);
  fsm.emplace_back(2, 2, 7.0);
  fsm.emplace_back(2, 3, 8.0);
  fsm.emplace_back(3, 2, 9.0);
  fsm.emplace_back(3, 3, 10.0);
  fsm.emplace_back(3, 4, 11.0);
  fsm.emplace_back(4, 3, 12.0);
  fsm.emplace_back(4, 4, 13.0);
  fsm.emplace_back(4, 5, 14.0);
  fsm.emplace_back(5, 4, 15.0);
  fsm.emplace_back(5, 5, 16.0);

  // lets use std::vector to populate vector b:
  std::vector<double> b = {0.0, 2.0, 4.0, 6.0, 8.0, 10.0};

  // create sparse solver on DEVICE:
  RealSparseSolverCUDA<MemorySpace::Device, double> rss;

  // because we used default cstor we need to call initialize
  rss.initialize(m);

  // insert sparse matrix A and vector b:
  rss.setFlatSparseMatrix(std::move(fsm));
  rss.setRhs(b);

  std::vector<double> solution(m);
  rss.solve<SparseSolverDeviceQR>(solution);

  std::cout << "Solution is: \n[";
  for (std::size_t t = 0; t < m; ++t) {
    std::cout << solution[t] << " ";
  }
  std::cout << "]\n";
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
  fsm.setColumns(m);
  fsm.setRows(m);
  // populate the matrix:
  fsm.emplace_back(0, 0, 1.0f);
  fsm.emplace_back(0, 1, 2.0f);
  fsm.emplace_back(1, 0, 3.0f);
  fsm.emplace_back(1, 1, 4.0f);
  fsm.emplace_back(1, 2, 5.0f);
  fsm.emplace_back(2, 1, 6.0f);
  fsm.emplace_back(2, 2, 7.0f);
  fsm.emplace_back(2, 3, 8.0f);
  fsm.emplace_back(3, 2, 9.0f);
  fsm.emplace_back(3, 3, 10.0f);
  fsm.emplace_back(3, 4, 11.0f);
  fsm.emplace_back(4, 3, 12.0f);
  fsm.emplace_back(4, 4, 13.0f);
  fsm.emplace_back(4, 5, 14.0f);
  fsm.emplace_back(5, 4, 15.0f);
  fsm.emplace_back(5, 5, 16.0f);

  // lets use std::vector to populate vector b:
  std::vector<float> b = {0.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.0f};

  // create sparse solver on DEVICE:
  RealSparseSolverCUDA<MemorySpace::Device, float> rss;

  // because we used default cstor we need to call initialize
  rss.initialize(m);

  // insert sparse matrix A and vector b:
  rss.setFlatSparseMatrix(std::move(fsm));
  rss.setRhs(b);

  std::vector<float> solution(m);
  rss.solve<SparseSolverDeviceQR>(solution);
  std::cout << "Solution is: \n[";
  for (std::size_t t = 0; t < m; ++t) {
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
  using lss_sparse_solvers_policy::SparseSolverHostLU;
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
  fsm.setColumns(m);
  fsm.setRows(m);
  // populate the matrix:
  fsm.emplace_back(0, 0, 1.0);
  fsm.emplace_back(0, 1, 2.0);
  fsm.emplace_back(1, 0, 3.0);
  fsm.emplace_back(1, 1, 4.0);
  fsm.emplace_back(1, 2, 5.0);
  fsm.emplace_back(2, 1, 6.0);
  fsm.emplace_back(2, 2, 7.0);
  fsm.emplace_back(2, 3, 8.0);
  fsm.emplace_back(3, 2, 9.0);
  fsm.emplace_back(3, 3, 10.0);
  fsm.emplace_back(3, 4, 11.0);
  fsm.emplace_back(4, 3, 12.0);
  fsm.emplace_back(4, 4, 13.0);
  fsm.emplace_back(4, 5, 14.0);
  fsm.emplace_back(5, 4, 15.0);
  fsm.emplace_back(5, 5, 16.0);

  // lets use std::vector to populate vector b:
  std::vector<double> b = {0.0, 2.0, 4.0, 6.0, 8.0, 10.0};

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
  fsm.setColumns(m);
  fsm.setRows(m);
  // populate the matrix:
  fsm.emplace_back(0, 0, 1.0f);
  fsm.emplace_back(0, 1, 2.0f);
  fsm.emplace_back(1, 0, 3.0f);
  fsm.emplace_back(1, 1, 4.0f);
  fsm.emplace_back(1, 2, 5.0f);
  fsm.emplace_back(2, 1, 6.0f);
  fsm.emplace_back(2, 2, 7.0f);
  fsm.emplace_back(2, 3, 8.0f);
  fsm.emplace_back(3, 2, 9.0f);
  fsm.emplace_back(3, 3, 10.0f);
  fsm.emplace_back(3, 4, 11.0f);
  fsm.emplace_back(4, 3, 12.0f);
  fsm.emplace_back(4, 4, 13.0f);
  fsm.emplace_back(4, 5, 14.0f);
  fsm.emplace_back(5, 4, 15.0f);
  fsm.emplace_back(5, 5, 16.0f);

  // lets use std::vector to populate vector b:
  std::vector<float> b = {0.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.0f};

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
  fsm.setColumns(m);
  fsm.setRows(m);
  // populate the matrix:
  fsm.emplace_back(0, 0, 1.0);
  fsm.emplace_back(0, 1, 2.0);
  fsm.emplace_back(1, 0, 3.0);
  fsm.emplace_back(1, 1, 4.0);
  fsm.emplace_back(1, 2, 5.0);
  fsm.emplace_back(2, 1, 6.0);
  fsm.emplace_back(2, 2, 7.0);
  fsm.emplace_back(2, 3, 8.0);
  fsm.emplace_back(3, 2, 9.0);
  fsm.emplace_back(3, 3, 10.0);
  fsm.emplace_back(3, 4, 11.0);
  fsm.emplace_back(4, 3, 12.0);
  fsm.emplace_back(4, 4, 13.0);
  fsm.emplace_back(4, 5, 14.0);
  fsm.emplace_back(5, 4, 15.0);
  fsm.emplace_back(5, 5, 16.0);

  // lets use std::vector to populate vector b:
  std::vector<double> b = {0.0, 2.0, 4.0, 6.0, 8.0, 10.0};

  // create sparse solver on DEVICE:
  RealSparseSolverCUDA<MemorySpace::Host, double> rss;

  // because we used default cstor we need to call initialize
  rss.initialize(m);

  // insert sparse matrix A and vector b:
  rss.setFlatSparseMatrix(std::move(fsm));
  rss.setRhs(b);

  std::vector<double> solution(m);
  rss.solve<SparseSolverHostQR>(solution);

  std::cout << "Solution is: \n[";
  for (std::size_t t = 0; t < m; ++t) {
    std::cout << solution[t] << " ";
  }
  std::cout << "]\n";
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
  fsm.setColumns(m);
  fsm.setRows(m);
  // populate the matrix:
  fsm.emplace_back(0, 0, 1.0f);
  fsm.emplace_back(0, 1, 2.0f);
  fsm.emplace_back(1, 0, 3.0f);
  fsm.emplace_back(1, 1, 4.0f);
  fsm.emplace_back(1, 2, 5.0f);
  fsm.emplace_back(2, 1, 6.0f);
  fsm.emplace_back(2, 2, 7.0f);
  fsm.emplace_back(2, 3, 8.0f);
  fsm.emplace_back(3, 2, 9.0f);
  fsm.emplace_back(3, 3, 10.0f);
  fsm.emplace_back(3, 4, 11.0f);
  fsm.emplace_back(4, 3, 12.0f);
  fsm.emplace_back(4, 4, 13.0f);
  fsm.emplace_back(4, 5, 14.0f);
  fsm.emplace_back(5, 4, 15.0f);
  fsm.emplace_back(5, 5, 16.0f);

  // lets use std::vector to populate vector b:
  std::vector<float> b = {0.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.0f};

  // create sparse solver on DEVICE:
  RealSparseSolverCUDA<MemorySpace::Host, float> rss;

  // because we used default cstor we need to call initialize
  rss.initialize(m);

  // insert sparse matrix A and vector b:
  rss.setFlatSparseMatrix(std::move(fsm));
  rss.setRhs(b);

  std::vector<float> solution(m);
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

void hostBVPDirichletBCDefaultQRTest() {
  using lss_sparse_solvers_cuda::FlatMatrix;
  using lss_sparse_solvers_cuda::MemorySpace;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_sparse_solvers_policy::SparseSolverHostQR;

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

  // first create and populate the sparse matrix:
  FlatMatrix<double> fsm;
  // discretization:
  // t_0,t_1,t_2,...,t_20
  int const N = 20;
  // step size:
  double h = 1.0 / static_cast<double>(N);
  // set number of columns and rows:
  // because we already know the boundary values
  // at t_0 = 0 and t_20 = 0:
  int const m = N - 1;
  fsm.setColumns(m);
  fsm.setRows(m);
  // populate the matrix:
  fsm.emplace_back(0, 0, -2.0);
  fsm.emplace_back(0, 1, 1.0);
  for (std::size_t t = 1; t < m - 1; ++t) {
    fsm.emplace_back(t, t - 1, 1.0);
    fsm.emplace_back(t, t, -2.0);
    fsm.emplace_back(t, t + 1, 1.0);
  }
  fsm.emplace_back(m - 1, m - 2, 1.0);
  fsm.emplace_back(m - 1, m - 1, -2.0);

  // lets use std::vector to populate vector b:
  std::vector<double> b(m, -2.0 * h * h);
  // set the Dirichlet boundary conditions:
  double left = 0.0;
  double right = 0.0;
  b[0] = b[0] - left;
  b[b.size() - 1] = b[b.size() - 1] - right;

  // create sparse solver on HOST:
  RealSparseSolverCUDA<MemorySpace::Host, double> rss;

  // because we used default cstor we need to call initialize
  rss.initialize(m);

  // insert sparse matrix A and vector b:
  rss.setFlatSparseMatrix(std::move(fsm));
  rss.setRhs(b);

  std::vector<double> solution(m);
  rss.solve<SparseSolverHostQR>(solution);

  // exact value:
  auto exact = [](double x) { return x * (1.0 - x); };

  std::cout << "tp : FDM | Exact\n";
  std::cout << "t_" << 0 << ": " << left << " |  " << exact(0) << '\n';
  for (std::size_t j = 0; j < solution.size(); ++j) {
    std::cout << "t_" << j + 1 << ": " << solution[j] << " |  "
              << exact((j + 1) * h) << '\n';
  }
  std::cout << "t_" << N << ": " << right << " |  " << exact(N * h) << '\n';
}

void hostBVPDirichletBCDefaultLUTest() {
  using lss_sparse_solvers_cuda::FlatMatrix;
  using lss_sparse_solvers_cuda::MemorySpace;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_sparse_solvers_policy::SparseSolverHostLU;

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

  // first create and populate the sparse matrix:
  FlatMatrix<double> fsm;
  // discretization:
  // t_0,t_1,t_2,...,t_20
  int const N = 20;
  // step size:
  double h = 1.0 / static_cast<double>(N);
  // set number of columns and rows:
  // because we already know the boundary values
  // at t_0 = 0 and t_20 = 0:
  int const m = N - 1;
  fsm.setColumns(m);
  fsm.setRows(m);
  // populate the matrix:
  fsm.emplace_back(0, 0, -2.0);
  fsm.emplace_back(0, 1, 1.0);
  for (std::size_t t = 1; t < m - 1; ++t) {
    fsm.emplace_back(t, t - 1, 1.0);
    fsm.emplace_back(t, t, -2.0);
    fsm.emplace_back(t, t + 1, 1.0);
  }
  fsm.emplace_back(m - 1, m - 2, 1.0);
  fsm.emplace_back(m - 1, m - 1, -2.0);

  // lets use std::vector to populate vector b:
  std::vector<double> b(m, -2.0 * h * h);
  // set the Dirichlet boundary conditions:
  double left = 0.0;
  double right = 0.0;
  b[0] = b[0] - left;
  b[b.size() - 1] = b[b.size() - 1] - right;

  // create sparse solver on HOST:
  RealSparseSolverCUDA<MemorySpace::Host, double> rss;

  // because we used default cstor we need to call initialize
  rss.initialize(m);

  // insert sparse matrix A and vector b:
  rss.setFlatSparseMatrix(std::move(fsm));
  rss.setRhs(b);

  std::vector<double> solution(m);
  rss.solve<SparseSolverHostLU>(solution);

  // exact value:
  auto exact = [](double x) { return x * (1.0 - x); };

  std::cout << "tp : FDM | Exact\n";
  std::cout << "t_" << 0 << ": " << left << " |  " << exact(0) << '\n';
  for (std::size_t j = 0; j < solution.size(); ++j) {
    std::cout << "t_" << j + 1 << ": " << solution[j] << " |  "
              << exact((j + 1) * h) << '\n';
  }
  std::cout << "t_" << N << ": " << right << " |  " << exact(N * h) << '\n';
}

void hostBVPDirichletBCDefaultCholeskyTest() {
  using lss_sparse_solvers_cuda::FlatMatrix;
  using lss_sparse_solvers_cuda::MemorySpace;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_sparse_solvers_policy::SparseSolverHostCholesky;

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

  // first create and populate the sparse matrix:
  FlatMatrix<double> fsm;
  // discretization:
  // t_0,t_1,t_2,...,t_20
  int const N = 20;
  // step size:
  double h = 1.0 / static_cast<double>(N);
  // set number of columns and rows:
  // because we already know the boundary values
  // at t_0 = 0 and t_20 = 0:
  int const m = N - 1;
  fsm.setColumns(m);
  fsm.setRows(m);
  // populate the matrix:
  fsm.emplace_back(0, 0, -2.0);
  fsm.emplace_back(0, 1, 1.0);
  for (std::size_t t = 1; t < m - 1; ++t) {
    fsm.emplace_back(t, t - 1, 1.0);
    fsm.emplace_back(t, t, -2.0);
    fsm.emplace_back(t, t + 1, 1.0);
  }
  fsm.emplace_back(m - 1, m - 2, 1.0);
  fsm.emplace_back(m - 1, m - 1, -2.0);

  // lets use std::vector to populate vector b:
  std::vector<double> b(m, -2.0 * h * h);
  // set the Dirichlet boundary conditions:
  double left = 0.0;
  double right = 0.0;
  b[0] = b[0] - left;
  b[b.size() - 1] = b[b.size() - 1] - right;

  // create sparse solver on HOST:
  RealSparseSolverCUDA<MemorySpace::Host, double> rss;

  // because we used default cstor we need to call initialize
  rss.initialize(m);

  // insert sparse matrix A and vector b:
  rss.setFlatSparseMatrix(std::move(fsm));
  rss.setRhs(b);

  std::vector<double> solution(m);
  rss.solve<SparseSolverHostCholesky>(solution);

  // exact value:
  auto exact = [](double x) { return x * (1.0 - x); };

  std::cout << "tp : FDM | Exact\n";
  std::cout << "t_" << 0 << ": " << left << " |  " << exact(0) << '\n';
  for (std::size_t j = 0; j < solution.size(); ++j) {
    std::cout << "t_" << j + 1 << ": " << solution[j] << " |  "
              << exact((j + 1) * h) << '\n';
  }
  std::cout << "t_" << N << ": " << right << " |  " << exact(N * h) << '\n';
}

void hostBVPDirichletBCFloatQRTest() {
  using lss_sparse_solvers_cuda::FlatMatrix;
  using lss_sparse_solvers_cuda::MemorySpace;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_sparse_solvers_policy::SparseSolverHostQR;

  std::cout << "=================================\n";
  std::cout << " Using QR decomposition to \n";
  std::cout << " solve Boundary Value Problem: \n\n";
  std::cout << " type: float					\n\n";
  std::cout << " u''(t) = -2, \n\n";
  std::cout << " where\n\n";
  std::cout << " t in <0,1>,\n";
  std::cout << " u(0) = u(1) = 0\n\n";
  std::cout << "Exact solution is:\n\n";
  std::cout << " u(t) = t(1-t)\n";
  std::cout << "=================================\n";

  // first create and populate the sparse matrix:
  FlatMatrix<float> fsm;
  // discretization:
  // t_0,t_1,t_2,...,t_20
  int const N = 20;
  // step size:
  float h = 1.0f / static_cast<float>(N);
  // set number of columns and rows:
  // because we already know the boundary values
  // at t_0 = 0 and t_20 = 0:
  int const m = N - 1;
  fsm.setColumns(m);
  fsm.setRows(m);
  // populate the matrix:
  fsm.emplace_back(0, 0, -2.0);
  fsm.emplace_back(0, 1, 1.0);
  for (std::size_t t = 1; t < m - 1; ++t) {
    fsm.emplace_back(t, t - 1, 1.0);
    fsm.emplace_back(t, t, -2.0);
    fsm.emplace_back(t, t + 1, 1.0);
  }
  fsm.emplace_back(m - 1, m - 2, 1.0);
  fsm.emplace_back(m - 1, m - 1, -2.0);

  // lets use std::vector to populate vector b:
  std::vector<float> b(m, -2.0f * h * h);
  // set the Dirichlet boundary conditions:
  float left = 0.0;
  float right = 0.0;
  b[0] = b[0] - left;
  b[b.size() - 1] = b[b.size() - 1] - right;

  // create sparse solver on HOST:
  RealSparseSolverCUDA<MemorySpace::Host, float> rss;

  // because we used default cstor we need to call initialize
  rss.initialize(m);

  // insert sparse matrix A and vector b:
  rss.setFlatSparseMatrix(std::move(fsm));
  rss.setRhs(b);

  std::vector<float> solution(m);
  rss.solve<SparseSolverHostQR>(solution);

  // exact value:
  auto exact = [](float x) { return x * (1.0 - x); };

  std::cout << "tp : FDM | Exact\n";
  std::cout << "t_" << 0 << ": " << left << " |  " << exact(0) << '\n';
  for (std::size_t j = 0; j < solution.size(); ++j) {
    std::cout << "t_" << j + 1 << ": " << solution[j] << " |  "
              << exact((j + 1) * h) << '\n';
  }
  std::cout << "t_" << N << ": " << right << " |  " << exact(N * h) << '\n';
}

void hostBVPDirichletBCFloatLUTest() {
  using lss_sparse_solvers_cuda::FlatMatrix;
  using lss_sparse_solvers_cuda::MemorySpace;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_sparse_solvers_policy::SparseSolverHostLU;

  std::cout << "=================================\n";
  std::cout << " Using LU decomposition to \n";
  std::cout << " solve Boundary Value Problem: \n\n";
  std::cout << " type: float					\n\n";
  std::cout << " u''(t) = -2, \n\n";
  std::cout << " where\n\n";
  std::cout << " t in <0,1>,\n";
  std::cout << " u(0) = u(1) = 0\n\n";
  std::cout << "Exact solution is:\n\n";
  std::cout << " u(t) = t(1-t)\n";
  std::cout << "=================================\n";

  // first create and populate the sparse matrix:
  FlatMatrix<float> fsm;
  // discretization:
  // t_0,t_1,t_2,...,t_20
  int const N = 20;
  // step size:
  float h = 1.0f / static_cast<float>(N);
  // set number of columns and rows:
  // because we already know the boundary values
  // at t_0 = 0 and t_20 = 0:
  int const m = N - 1;
  fsm.setColumns(m);
  fsm.setRows(m);
  // populate the matrix:
  fsm.emplace_back(0, 0, -2.0);
  fsm.emplace_back(0, 1, 1.0);
  for (std::size_t t = 1; t < m - 1; ++t) {
    fsm.emplace_back(t, t - 1, 1.0);
    fsm.emplace_back(t, t, -2.0);
    fsm.emplace_back(t, t + 1, 1.0);
  }
  fsm.emplace_back(m - 1, m - 2, 1.0);
  fsm.emplace_back(m - 1, m - 1, -2.0);

  // lets use std::vector to populate vector b:
  std::vector<float> b(m, -2.0f * h * h);
  // set the Dirichlet boundary conditions:
  float left = 0.0;
  float right = 0.0;
  b[0] = b[0] - left;
  b[b.size() - 1] = b[b.size() - 1] - right;

  // create sparse solver on HOST:
  RealSparseSolverCUDA<MemorySpace::Host, float> rss;

  // because we used default cstor we need to call initialize
  rss.initialize(m);

  // insert sparse matrix A and vector b:
  rss.setFlatSparseMatrix(std::move(fsm));
  rss.setRhs(b);

  std::vector<float> solution(m);
  rss.solve<SparseSolverHostLU>(solution);

  // exact value:
  auto exact = [](float x) { return x * (1.0 - x); };

  std::cout << "tp : FDM | Exact\n";
  std::cout << "t_" << 0 << ": " << left << " |  " << exact(0) << '\n';
  for (std::size_t j = 0; j < solution.size(); ++j) {
    std::cout << "t_" << j + 1 << ": " << solution[j] << " |  "
              << exact((j + 1) * h) << '\n';
  }
  std::cout << "t_" << N << ": " << right << " |  " << exact(N * h) << '\n';
}

void hostBVPDirichletBCFloatCholeskyTest() {
  using lss_sparse_solvers_cuda::FlatMatrix;
  using lss_sparse_solvers_cuda::MemorySpace;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_sparse_solvers_policy::SparseSolverHostCholesky;

  std::cout << "=================================\n";
  std::cout << " Using Cholesky decomposition to \n";
  std::cout << " solve Boundary Value Problem: \n\n";
  std::cout << " type: float					\n\n";
  std::cout << " u''(t) = -2, \n\n";
  std::cout << " where\n\n";
  std::cout << " t in <0,1>,\n";
  std::cout << " u(0) = u(1) = 0\n\n";
  std::cout << "Exact solution is:\n\n";
  std::cout << " u(t) = t(1-t)\n";
  std::cout << "=================================\n";

  // first create and populate the sparse matrix:
  FlatMatrix<float> fsm;
  // discretization:
  // t_0,t_1,t_2,...,t_20
  int const N = 20;
  // step size:
  float h = 1.0f / static_cast<float>(N);
  // set number of columns and rows:
  // because we already know the boundary values
  // at t_0 = 0 and t_20 = 0:
  int const m = N - 1;
  fsm.setColumns(m);
  fsm.setRows(m);
  // populate the matrix:
  fsm.emplace_back(0, 0, -2.0);
  fsm.emplace_back(0, 1, 1.0);
  for (std::size_t t = 1; t < m - 1; ++t) {
    fsm.emplace_back(t, t - 1, 1.0);
    fsm.emplace_back(t, t, -2.0);
    fsm.emplace_back(t, t + 1, 1.0);
  }
  fsm.emplace_back(m - 1, m - 2, 1.0);
  fsm.emplace_back(m - 1, m - 1, -2.0);

  // lets use std::vector to populate vector b:
  std::vector<float> b(m, -2.0f * h * h);
  // set the Dirichlet boundary conditions:
  float left = 0.0;
  float right = 0.0;
  b[0] = b[0] - left;
  b[b.size() - 1] = b[b.size() - 1] - right;

  // create sparse solver on HOST:
  RealSparseSolverCUDA<MemorySpace::Host, float> rss;

  // because we used default cstor we need to call initialize
  rss.initialize(m);

  // insert sparse matrix A and vector b:
  rss.setFlatSparseMatrix(std::move(fsm));
  rss.setRhs(b);

  std::vector<float> solution(m);
  rss.solve<SparseSolverHostCholesky>(solution);

  // exact value:
  auto exact = [](float x) { return x * (1.0 - x); };

  std::cout << "tp : FDM | Exact\n";
  std::cout << "t_" << 0 << ": " << left << " |  " << exact(0) << '\n';
  for (std::size_t j = 0; j < solution.size(); ++j) {
    std::cout << "t_" << j + 1 << ": " << solution[j] << " |  "
              << exact((j + 1) * h) << '\n';
  }
  std::cout << "t_" << N << ": " << right << " |  " << exact(N * h) << '\n';
}

void testDirichletBCBVPOnHost() {
  std::cout << "==================================================\n";
  std::cout << "============ Dirichlet BC BVP - HOST ´´============\n";
  std::cout << "==================================================\n";

  hostBVPDirichletBCDefaultQRTest();
  hostBVPDirichletBCFloatQRTest();
  hostBVPDirichletBCDefaultLUTest();
  hostBVPDirichletBCFloatLUTest();
  hostBVPDirichletBCDefaultCholeskyTest();
  hostBVPDirichletBCFloatCholeskyTest();

  std::cout << "==================================================\n";
}

void deviceBVPDirichletBCDefaultQRTest() {
  using lss_sparse_solvers_cuda::FlatMatrix;
  using lss_sparse_solvers_cuda::MemorySpace;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_sparse_solvers_policy::SparseSolverDeviceQR;

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

  // first create and populate the sparse matrix:
  FlatMatrix<double> fsm;
  // discretization:
  // t_0,t_1,t_2,...,t_20
  int const N = 20;
  // step size:
  double h = 1.0 / static_cast<double>(N);
  // set number of columns and rows:
  // because we already know the boundary values
  // at timepoints t_0 and t_20:
  int const m = N - 1;
  fsm.setColumns(m);
  fsm.setRows(m);
  // populate the matrix:
  fsm.emplace_back(0, 0, -2.0);
  fsm.emplace_back(0, 1, 1.0);
  for (std::size_t t = 1; t < m - 1; ++t) {
    fsm.emplace_back(t, t - 1, 1.0);
    fsm.emplace_back(t, t, -2.0);
    fsm.emplace_back(t, t + 1, 1.0);
  }
  fsm.emplace_back(m - 1, m - 2, 1.0);
  fsm.emplace_back(m - 1, m - 1, -2.0);

  // lets use std::vector to populate vector b:
  std::vector<double> b(m, -2.0 * h * h);
  // set the Dirichlet boundary conditions:
  double left = 0.0;
  double right = 0.0;
  b[0] = b[0] - left;
  b[b.size() - 1] = b[b.size() - 1] - right;

  // create sparse solver on HOST:
  RealSparseSolverCUDA<MemorySpace::Device, double> rss;

  // because we used default cstor we need to call initialize
  rss.initialize(m);

  // insert sparse matrix A and vector b:
  rss.setFlatSparseMatrix(std::move(fsm));
  rss.setRhs(b);

  std::vector<double> solution(m);
  rss.solve<SparseSolverDeviceQR>(solution);

  // exact value:
  auto exact = [](double x) { return x * (1.0 - x); };

  std::cout << "tp : FDM | Exact\n";
  std::cout << "t_" << 0 << ": " << left << " |  " << exact(0) << '\n';
  for (std::size_t j = 0; j < solution.size(); ++j) {
    std::cout << "t_" << j + 1 << ": " << solution[j] << " |  "
              << exact((j + 1) * h) << '\n';
  }
  std::cout << "t_" << N << ": " << right << " |  " << exact(N * h) << '\n';
}

void deviceBVPDirichletBCDefaultCholeskyTest() {
  using lss_sparse_solvers_cuda::FlatMatrix;
  using lss_sparse_solvers_cuda::MemorySpace;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_sparse_solvers_policy::SparseSolverDeviceCholesky;

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

  // first create and populate the sparse matrix:
  FlatMatrix<double> fsm;
  // discretization:
  // t_0,t_1,t_2,...,t_20
  int const N = 20;
  // step size:
  double h = 1.0 / static_cast<double>(N);
  // set number of columns and rows:
  // because we already know the boundary values
  // at t_0 = 0 and t_20 = 0:
  int const m = N - 1;
  fsm.setColumns(m);
  fsm.setRows(m);
  // populate the matrix:
  fsm.emplace_back(0, 0, -2.0);
  fsm.emplace_back(0, 1, 1.0);
  for (std::size_t t = 1; t < m - 1; ++t) {
    fsm.emplace_back(t, t - 1, 1.0);
    fsm.emplace_back(t, t, -2.0);
    fsm.emplace_back(t, t + 1, 1.0);
  }
  fsm.emplace_back(m - 1, m - 2, 1.0);
  fsm.emplace_back(m - 1, m - 1, -2.0);

  // lets use std::vector to populate vector b:
  std::vector<double> b(m, -2.0 * h * h);
  // set the Dirichlet boundary conditions:
  double left = 0.0;
  double right = 0.0;
  b[0] = b[0] - left;
  b[b.size() - 1] = b[b.size() - 1] - right;

  // create sparse solver on HOST:
  RealSparseSolverCUDA<MemorySpace::Device, double> rss;

  // because we used default cstor we need to call initialize
  rss.initialize(m);

  // insert sparse matrix A and vector b:
  rss.setFlatSparseMatrix(std::move(fsm));
  rss.setRhs(b);

  std::vector<double> solution(m);
  rss.solve<SparseSolverDeviceCholesky>(solution);

  // exact value:
  auto exact = [](double x) { return x * (1.0 - x); };

  std::cout << "tp : FDM | Exact\n";
  std::cout << "t_" << 0 << ": " << left << " |  " << exact(0) << '\n';
  for (std::size_t j = 0; j < solution.size(); ++j) {
    std::cout << "t_" << j + 1 << ": " << solution[j] << " |  "
              << exact((j + 1) * h) << '\n';
  }
  std::cout << "t_" << N << ": " << right << " |  " << exact(N * h) << '\n';
}

void testDirichletBCBVPOnDevice() {
  std::cout << "==================================================\n";
  std::cout << "============ Dirichlet BC BVP - DEVICE ===========\n";
  std::cout << "==================================================\n";

  deviceBVPDirichletBCDefaultQRTest();
  deviceBVPDirichletBCDefaultCholeskyTest();

  std::cout << "==================================================\n";
}

void hostBVPRobinBCDefaultQRTest() {
  using lss_sparse_solvers_cuda::FlatMatrix;
  using lss_sparse_solvers_cuda::MemorySpace;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_sparse_solvers_policy::SparseSolverHostQR;

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

  // first create and populate the sparse matrix:
  FlatMatrix<double> fsm;
  // discretization:
  // t_0,t_1,t_2,...,t_20
  int const N = 100;
  // step size:
  double h = 1.0 / static_cast<double>(N);
  // set the Robin boundary conditions:
  double alpha = .00;
  double phi = 1.0;
  double beta = (2.0 + h) / (2.0 - h);
  double psi = 0.0;
  // set number of columns and rows:
  // because we already know the boundary values
  // at t_0 = 0 and t_20 = 0:
  int const m = N - 1;
  fsm.setColumns(m);
  fsm.setRows(m);
  // populate the matrix:
  fsm.emplace_back(0, 0, (alpha * 1.0 - 2.0));
  fsm.emplace_back(0, 1, 1.0);
  for (std::size_t t = 1; t < m - 1; ++t) {
    fsm.emplace_back(t, t - 1, 1.0);
    fsm.emplace_back(t, t, -2.0);
    fsm.emplace_back(t, t + 1, 1.0);
  }
  fsm.emplace_back(m - 1, m - 2, 1.0);
  fsm.emplace_back(m - 1, m - 1, (-2.0 + (1.0 / beta)));

  // lets use std::vector to populate vector b:
  std::vector<double> b(m, -2.0 * h * h);
  // adjust first and last elements due to the Robin BC
  b[0] = b[0] - 1.0 * phi;
  b[b.size() - 1] = b[b.size() - 1] + psi * (1.0 / beta);

  // create sparse solver on HOST:
  RealSparseSolverCUDA<MemorySpace::Host, double> rss;

  // because we used default cstor we need to call initialize
  rss.initialize(m);

  // insert sparse matrix A and vector b:
  rss.setFlatSparseMatrix(std::move(fsm));
  rss.setRhs(b);

  std::vector<double> solution(m);
  rss.solve<SparseSolverHostQR>(solution);

  // exact value:
  auto exact = [](double x) { return (-x * x + x + 1.0); };

  std::cout << "tp : FDM | Exact | Abs Diff\n";
  std::cout << "t_" << 0 << ": " << (alpha * solution.front() + phi) << " |  "
            << exact(0) << " | "
            << ((alpha * solution.front() + phi) - exact(0)) << '\n';
  for (std::size_t j = 0; j < solution.size(); ++j) {
    std::cout << "t_" << j + 1 << ": " << solution[j] << " |  "
              << exact((j + 1) * h) << " | "
              << (solution[j] - exact((j + 1) * h)) << '\n';
  }
  std::cout << "t_" << N << ": " << ((solution.back() - psi) / beta) << " |  "
            << exact(N * h) << " | "
            << (((solution.back() - psi) / beta) - exact(N * h)) << '\n';
}

void hostBVPRobinBCDefaultLUTest() {
  using lss_sparse_solvers_cuda::FlatMatrix;
  using lss_sparse_solvers_cuda::MemorySpace;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_sparse_solvers_policy::SparseSolverHostLU;

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

  // first create and populate the sparse matrix:
  FlatMatrix<double> fsm;
  // discretization:
  // t_0,t_1,t_2,...,t_20
  int const N = 100;
  // step size:
  double h = 1.0 / static_cast<double>(N);
  // set the Robin boundary conditions:
  double alpha = .00;
  double phi = 1.0;
  double beta = (2.0 + h) / (2.0 - h);
  double psi = 0.0;
  // set number of columns and rows:
  // because we already know the boundary values
  // at t_0 = 0 and t_20 = 0:
  int const m = N - 1;
  fsm.setColumns(m);
  fsm.setRows(m);
  // populate the matrix:
  fsm.emplace_back(0, 0, (alpha * 1.0 - 2.0));
  fsm.emplace_back(0, 1, 1.0);
  for (std::size_t t = 1; t < m - 1; ++t) {
    fsm.emplace_back(t, t - 1, 1.0);
    fsm.emplace_back(t, t, -2.0);
    fsm.emplace_back(t, t + 1, 1.0);
  }
  fsm.emplace_back(m - 1, m - 2, 1.0);
  fsm.emplace_back(m - 1, m - 1, (-2.0 + (1.0 / beta)));

  // lets use std::vector to populate vector b:
  std::vector<double> b(m, -2.0 * h * h);
  // adjust first and last elements due to the Robin BC
  b[0] = b[0] - 1.0 * phi;
  b[b.size() - 1] = b[b.size() - 1] + psi * (1.0 / beta);

  // create sparse solver on HOST:
  RealSparseSolverCUDA<MemorySpace::Host, double> rss;

  // because we used default cstor we need to call initialize
  rss.initialize(m);

  // insert sparse matrix A and vector b:
  rss.setFlatSparseMatrix(std::move(fsm));
  rss.setRhs(b);

  std::vector<double> solution(m);
  rss.solve<SparseSolverHostLU>(solution);

  // exact value:
  auto exact = [](double x) { return (-x * x + x + 1.0); };

  std::cout << "tp : FDM | Exact | Abs Diff\n";
  std::cout << "t_" << 0 << ": " << (alpha * solution.front() + phi) << " |  "
            << exact(0) << " | "
            << ((alpha * solution.front() + phi) - exact(0)) << '\n';
  for (std::size_t j = 0; j < solution.size(); ++j) {
    std::cout << "t_" << j + 1 << ": " << solution[j] << " |  "
              << exact((j + 1) * h) << " | "
              << (solution[j] - exact((j + 1) * h)) << '\n';
  }
  std::cout << "t_" << N << ": " << ((solution.back() - psi) / beta) << " |  "
            << exact(N * h) << " | "
            << (((solution.back() - psi) / beta) - exact(N * h)) << '\n';
}

void hostBVPRobinBCDefaultCholeskyTest() {
  using lss_sparse_solvers_cuda::FlatMatrix;
  using lss_sparse_solvers_cuda::MemorySpace;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_sparse_solvers_policy::SparseSolverHostCholesky;

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

  // first create and populate the sparse matrix:
  FlatMatrix<double> fsm;
  // discretization:
  // t_0,t_1,t_2,...,t_20
  int const N = 100;
  // step size:
  double h = 1.0 / static_cast<double>(N);
  // set the Robin boundary conditions:
  double alpha = .00;
  double phi = 1.0;
  double beta = (2.0 + h) / (2.0 - h);
  double psi = 0.0;
  // set number of columns and rows:
  // because we already know the boundary values
  // at t_0 = 0 and t_20 = 0:
  int const m = N - 1;
  fsm.setColumns(m);
  fsm.setRows(m);
  // populate the matrix:
  fsm.emplace_back(0, 0, (alpha * 1.0 - 2.0));
  fsm.emplace_back(0, 1, 1.0);
  for (std::size_t t = 1; t < m - 1; ++t) {
    fsm.emplace_back(t, t - 1, 1.0);
    fsm.emplace_back(t, t, -2.0);
    fsm.emplace_back(t, t + 1, 1.0);
  }
  fsm.emplace_back(m - 1, m - 2, 1.0);
  fsm.emplace_back(m - 1, m - 1, (-2.0 + (1.0 / beta)));

  // lets use std::vector to populate vector b:
  std::vector<double> b(m, -2.0 * h * h);
  // adjust first and last elements due to the Robin BC
  b[0] = b[0] - 1.0 * phi;
  b[b.size() - 1] = b[b.size() - 1] + psi * (1.0 / beta);

  // create sparse solver on HOST:
  RealSparseSolverCUDA<MemorySpace::Host, double> rss;

  // because we used default cstor we need to call initialize
  rss.initialize(m);

  // insert sparse matrix A and vector b:
  rss.setFlatSparseMatrix(std::move(fsm));
  rss.setRhs(b);

  std::vector<double> solution(m);
  rss.solve<SparseSolverHostCholesky>(solution);

  // exact value:
  auto exact = [](double x) { return (-x * x + x + 1.0); };

  std::cout << "tp : FDM | Exact | Abs Diff\n";
  std::cout << "t_" << 0 << ": " << (alpha * solution.front() + phi) << " |  "
            << exact(0) << " | "
            << ((alpha * solution.front() + phi) - exact(0)) << '\n';
  for (std::size_t j = 0; j < solution.size(); ++j) {
    std::cout << "t_" << j + 1 << ": " << solution[j] << " |  "
              << exact((j + 1) * h) << " | "
              << (solution[j] - exact((j + 1) * h)) << '\n';
  }
  std::cout << "t_" << N << ": " << ((solution.back() - psi) / beta) << " |  "
            << exact(N * h) << " | "
            << (((solution.back() - psi) / beta) - exact(N * h)) << '\n';
}

void testRobinBCBVPOnHost() {
  std::cout << "==================================================\n";
  std::cout << "============ Robin BC BVP - HOST =================\n";
  std::cout << "==================================================\n";

  hostBVPRobinBCDefaultQRTest();
  hostBVPRobinBCDefaultLUTest();
  hostBVPRobinBCDefaultCholeskyTest();

  std::cout << "==================================================\n";
}

void deviceBVPRobinBCDefaultQRTest() {
  using lss_sparse_solvers_cuda::FlatMatrix;
  using lss_sparse_solvers_cuda::MemorySpace;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_sparse_solvers_policy::SparseSolverDeviceQR;

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

  // first create and populate the sparse matrix:
  FlatMatrix<double> fsm;
  // discretization:
  // t_0,t_1,t_2,...,t_20
  int const N = 100;
  // step size:
  double h = 1.0 / static_cast<double>(N);
  // set the Robin boundary conditions:
  double alpha = .00;
  double phi = 1.0;
  double beta = (2.0 + h) / (2.0 - h);
  double psi = 0.0;
  // set number of columns and rows:
  // because we already know the boundary values
  // at timepoints t_0 and t_20:
  int const m = N - 1;
  fsm.setColumns(m);
  fsm.setRows(m);
  // populate the matrix:
  fsm.emplace_back(0, 0, (alpha * 1.0 - 2.0));
  fsm.emplace_back(0, 1, 1.0);
  for (std::size_t t = 1; t < m - 1; ++t) {
    fsm.emplace_back(t, t - 1, 1.0);
    fsm.emplace_back(t, t, -2.0);
    fsm.emplace_back(t, t + 1, 1.0);
  }
  fsm.emplace_back(m - 1, m - 2, 1.0);
  fsm.emplace_back(m - 1, m - 1, (-2.0 + (1.0 / beta)));

  // lets use std::vector to populate vector b:
  std::vector<double> b(m, -2.0 * h * h);
  // adjust first and last elements due to the Robin BC
  b[0] = b[0] - 1.0 * phi;
  b[b.size() - 1] = b[b.size() - 1] + psi * (1.0 / beta);

  // create sparse solver on HOST:
  RealSparseSolverCUDA<MemorySpace::Device, double> rss;

  // because we used default cstor we need to call initialize
  rss.initialize(m);

  // insert sparse matrix A and vector b:
  rss.setFlatSparseMatrix(std::move(fsm));
  rss.setRhs(b);

  std::vector<double> solution(m);
  rss.solve<SparseSolverDeviceQR>(solution);

  // exact value:
  auto exact = [](double x) { return (-x * x + x + 1.0); };

  std::cout << "tp : FDM | Exact | Abs Diff\n";
  std::cout << "t_" << 0 << ": " << (alpha * solution.front() + phi) << " |  "
            << exact(0) << " | "
            << ((alpha * solution.front() + phi) - exact(0)) << '\n';
  for (std::size_t j = 0; j < solution.size(); ++j) {
    std::cout << "t_" << j + 1 << ": " << solution[j] << " |  "
              << exact((j + 1) * h) << " | "
              << (solution[j] - exact((j + 1) * h)) << '\n';
  }
  std::cout << "t_" << N << ": " << ((solution.back() - psi) / beta) << " |  "
            << exact(N * h) << " | "
            << (((solution.back() - psi) / beta) - exact(N * h)) << '\n';
}

void deviceBVPRobinBCDefaultCholeskyTest() {
  using lss_sparse_solvers_cuda::FlatMatrix;
  using lss_sparse_solvers_cuda::MemorySpace;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_sparse_solvers_policy::SparseSolverDeviceCholesky;

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

  // first create and populate the sparse matrix:
  FlatMatrix<double> fsm;
  // discretization:
  // t_0,t_1,t_2,...,t_20
  int const N = 100;
  // step size:
  double h = 1.0 / static_cast<double>(N);
  // set the Robin boundary conditions:
  double alpha = .00;
  double phi = 1.0;
  double beta = (2.0 + h) / (2.0 - h);
  double psi = 0.0;
  // set number of columns and rows:
  // because we already know the boundary values
  // at t_0 = 0 and t_20 = 0:
  int const m = N - 1;
  fsm.setColumns(m);
  fsm.setRows(m);
  // populate the matrix:
  fsm.emplace_back(0, 0, (alpha * 1.0 - 2.0));
  fsm.emplace_back(0, 1, 1.0);
  for (std::size_t t = 1; t < m - 1; ++t) {
    fsm.emplace_back(t, t - 1, 1.0);
    fsm.emplace_back(t, t, -2.0);
    fsm.emplace_back(t, t + 1, 1.0);
  }
  fsm.emplace_back(m - 1, m - 2, 1.0);
  fsm.emplace_back(m - 1, m - 1, (-2.0 + (1.0 / beta)));

  // lets use std::vector to populate vector b:
  std::vector<double> b(m, -2.0 * h * h);
  // adjust first and last elements due to the Robin BC
  b[0] = b[0] - 1.0 * phi;
  b[b.size() - 1] = b[b.size() - 1] + psi * (1.0 / beta);

  // create sparse solver on HOST:
  RealSparseSolverCUDA<MemorySpace::Device, double> rss;

  // because we used default cstor we need to call initialize
  rss.initialize(m);

  // insert sparse matrix A and vector b:
  rss.setFlatSparseMatrix(std::move(fsm));
  rss.setRhs(b);

  std::vector<double> solution(m);
  rss.solve<SparseSolverDeviceCholesky>(solution);

  // exact value:
  auto exact = [](double x) { return (-x * x + x + 1.0); };

  std::cout << "tp : FDM | Exact | Abs Diff\n";
  std::cout << "t_" << 0 << ": " << (alpha * solution.front() + phi) << " |  "
            << exact(0) << " | "
            << ((alpha * solution.front() + phi) - exact(0)) << '\n';
  for (std::size_t j = 0; j < solution.size(); ++j) {
    std::cout << "t_" << j + 1 << ": " << solution[j] << " |  "
              << exact((j + 1) * h) << " | "
              << (solution[j] - exact((j + 1) * h)) << '\n';
  }
  std::cout << "t_" << N << ": " << ((solution.back() - psi) / beta) << " |  "
            << exact(N * h) << " | "
            << (((solution.back() - psi) / beta) - exact(N * h)) << '\n';
}

void testRobinBCBVPOnDevice() {
  std::cout << "==================================================\n";
  std::cout << "============ Robin BC BVP - DEVICE ===============\n";
  std::cout << "==================================================\n";

  deviceBVPRobinBCDefaultQRTest();
  deviceBVPRobinBCDefaultCholeskyTest();

  std::cout << "==================================================\n";
}

#endif  ///_LSS_SPARSE_SOLVERS_CUDA_T
