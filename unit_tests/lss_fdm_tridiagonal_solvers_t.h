#pragma once
#if !defined(_LSS_FDM_TRIDIAGONAL_SOLVERS_T)
#define _LSS_FDM_TRIDIAGONAL_SOLVERS_T

#include <vector>

#include "common/lss_enumerations.h"
#include "sparse_solvers/lss_fdm_double_sweep_solver.h"
#include "sparse_solvers/lss_fdm_thomas_lu_solver.h"
#include "sparse_solvers/lss_fdm_tridiagonal_solvers.h"

template <typename T>
void testBVPDoubleSweepDirichletBC() {
  using lss_enumerations::boundary_condition_enum;
  using lss_fdm_double_sweep_solver::fdm_double_sweep_solver;

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
  std::size_t N{20};
  // step size:
  double h = 1.0 / static_cast<T>(N);
  // upper,mid, and lower diagonal:
  std::vector<T> upperDiag(N + 1, 1.0);
  std::vector<T> diagonal(N + 1, -2.0);
  std::vector<T> lowerDiag(N + 1, 1.0);

  // right-hand side:
  std::vector<T> rhs(N + 1, -2.0 * h * h);

  // boundary conditions:
  T left = 0.0;
  T right = 0.0;

  fdm_double_sweep_solver<T, boundary_condition_enum::Dirichlet, std::vector,
                          std::allocator<T>>
      dss{N + 1};
  dss.set_diagonals(std::move(lowerDiag), std::move(diagonal),
                    std::move(upperDiag));
  dss.set_boundary_condition(std::make_pair(left, right));
  dss.set_rhs(rhs);
  // get the solution:
  auto solution = dss.solve();

  // exact value:
  auto exact = [](T x) { return x * (1.0 - x); };

  std::cout << "tp : FDM | Exact\n";
  for (std::size_t j = 0; j < solution.size(); ++j) {
    std::cout << "t_" << j << ": " << solution[j] << " |  " << exact(j * h)
              << '\n';
  }
}

template <typename T>
void testBVPFDMSolverDirichletBC_0() {
  using lss_enumerations::boundary_condition_enum;
  using lss_fdm_double_sweep_solver::fdm_double_sweep_solver;
  using lss_fdm_tridiagonal_solvers::fdm_tridiagonal_solver;

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
  std::size_t N{20};
  // step size:
  T h = static_cast<T>(1.0) / static_cast<T>(N);
  // upper,mid, and lower diagonal:
  std::vector<T> upperDiag(N + 1, 1.0f);
  std::vector<T> diagonal(N + 1, -2.0f);
  std::vector<T> lowerDiag(N + 1, 1.0f);

  // right-hand side:
  std::vector<T> rhs(N + 1, -2.0 * h * h);

  // boundary conditions:
  T left = 0.0f;
  T right = 0.0f;

  // typedef the solver
  typedef fdm_tridiagonal_solver<T, boundary_condition_enum::Dirichlet,
                                 fdm_double_sweep_solver, std::vector,
                                 std::allocator<T>>
      DoubleSweep;

  DoubleSweep dss{N + 1};
  dss.set_diagonals(std::move(lowerDiag), std::move(diagonal),
                    std::move(upperDiag));
  dss.set_boundary_condition(std::make_pair(left, right));
  dss.set_rhs(rhs);
  // get the solution:
  auto solution = dss.solve();

  // exact value:
  auto exact = [](T x) { return x * (1.0 - x); };

  std::cout << "tp : FDM | Exact\n";
  for (std::size_t j = 0; j < solution.size(); ++j) {
    std::cout << "t_" << j << ": " << solution[j] << " |  " << exact(j * h)
              << '\n';
  }
}

void testDoubleSweepDirichletBC() {
  std::cout << "==================================================\n";
  std::cout << "=========== Double Sweep (Dirichlet BC) ==========\n";
  std::cout << "==================================================\n";

  testBVPDoubleSweepDirichletBC<double>();
  testBVPDoubleSweepDirichletBC<float>();
  testBVPFDMSolverDirichletBC_0<double>();
  testBVPFDMSolverDirichletBC_0<float>();

  std::cout << "==================================================\n";
}

template <typename T>
void testBVPThomasLUSolverDirichletBC() {
  using lss_enumerations::boundary_condition_enum;
  using lss_fdm_thomas_lu_solver::fdm_thomas_lu_solver;

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
  std::size_t N{20};
  // step size:
  T h = static_cast<T>(1.0) / static_cast<T>(N);
  // upper,mid, and lower diagonal:
  // must be of size N+1 because we need to add the t_0 point:
  std::vector<T> upperDiag(N + 1, 1.0);
  std::vector<T> diagonal(N + 1, -2.0);
  std::vector<T> lowerDiag(N + 1, 1.0);

  // right-hand side:
  std::vector<T> rhs(N + 1, -2.0 * h * h);

  // boundary conditions:
  T left = 0.0;
  T right = 0.0;

  fdm_thomas_lu_solver<T, boundary_condition_enum::Dirichlet, std::vector,
                       std::allocator<T>>
      dss{N + 1};
  dss.set_diagonals(std::move(lowerDiag), std::move(diagonal),
                    std::move(upperDiag));
  dss.set_boundary_condition(std::make_pair(left, right));
  dss.set_rhs(rhs);
  // get the solution:
  auto solution = dss.solve();

  // exact value:
  auto exact = [](T x) { return x * (1.0 - x); };

  std::cout << "tp : FDM | Exact\n";
  for (std::size_t j = 0; j < solution.size(); ++j) {
    std::cout << "t_" << j << ": " << solution[j] << " |  " << exact(j * h)
              << '\n';
  }
}

template <typename T>
void testBVPFDMSolverDirichletBC_1() {
  using lss_enumerations::boundary_condition_enum;
  using lss_fdm_thomas_lu_solver::fdm_thomas_lu_solver;
  using lss_fdm_tridiagonal_solvers::fdm_tridiagonal_solver;

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
  std::size_t N{20};
  // step size:
  T h = static_cast<T>(1.0) / static_cast<T>(N);
  // upper,mid, and lower diagonal:
  // must be of size N+1 because we need to add the t_0 point:
  std::vector<T> upperDiag(N + 1, 1.0);
  std::vector<T> diagonal(N + 1, -2.0);
  std::vector<T> lowerDiag(N + 1, 1.0);

  // right-hand side:
  std::vector<T> rhs(N + 1, -2.0 * h * h);

  // boundary conditions:
  T left = 0.0;
  T right = 0.0;

  // typedef the solver
  typedef fdm_tridiagonal_solver<T, boundary_condition_enum::Dirichlet,
                                 fdm_thomas_lu_solver, std::vector,
                                 std::allocator<T>>
      ThomasLU;

  ThomasLU ts{N + 1};
  ts.set_diagonals(std::move(lowerDiag), std::move(diagonal),
                   std::move(upperDiag));
  ts.set_boundary_condition(std::make_pair(left, right));
  ts.set_rhs(rhs);
  // get the solution:
  auto solution = ts.solve();

  // exact value:
  auto exact = [](T x) { return x * (1.0 - x); };

  std::cout << "tp : FDM | Exact\n";
  for (std::size_t j = 0; j < solution.size(); ++j) {
    std::cout << "t_" << j << ": " << solution[j] << " |  " << exact(j * h)
              << '\n';
  }
}

void testThomasLUSolverDirichletBC() {
  std::cout << "==================================================\n";
  std::cout << "========= Thomas LU Solver (Dirichlet BC) ========\n";
  std::cout << "==================================================\n";

  testBVPThomasLUSolverDirichletBC<double>();
  testBVPThomasLUSolverDirichletBC<float>();
  testBVPFDMSolverDirichletBC_1<double>();
  testBVPFDMSolverDirichletBC_1<float>();

  std::cout << "==================================================\n";
}
//
template <typename T>
void testBVPDoubleSweepDirichletBC1() {
  using lss_enumerations::boundary_condition_enum;
  using lss_fdm_double_sweep_solver::fdm_double_sweep_solver;

  /*

  Solve BVP:

  u''(t) = - 2,

  where

  t \in (0, 1)
  u(0) = 1 ,  u(1) = 1


  Exact solution is

  u(t) = -t*t + t + 1

  */
  std::cout << "=================================\n";
  std::cout << "Solving Boundary-value problem: \n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " u''(t) = -2, \n\n";
  std::cout << " where\n\n";
  std::cout << " t in <0,1>,\n";
  std::cout << " u(0) = u(1) = 1\n\n";
  std::cout << "Exact solution is:\n\n";
  std::cout << " u(t) = -t*t + t + 1\n";
  std::cout << "=================================\n";

  // discretization:
  std::size_t N{20};
  // step size:
  T h = static_cast<T>(1.0) / static_cast<T>(N);
  // upper,mid, and lower diagonal:
  std::vector<T> upperDiag(N + 1, 1.0);
  std::vector<T> diagonal(N + 1, -2.0);
  std::vector<T> lowerDiag(N + 1, 1.0);

  // right-hand side:
  std::vector<T> rhs(N + 1, -2.0 * h * h);

  // boundary conditions:
  T left = 1.0;
  T right = 1.0;

  fdm_double_sweep_solver<T, boundary_condition_enum::Dirichlet, std::vector,
                          std::allocator<T>>
      dss{N + 1};
  dss.set_diagonals(std::move(lowerDiag), std::move(diagonal),
                    std::move(upperDiag));
  dss.set_boundary_condition(std::make_pair(left, right));
  dss.set_rhs(rhs);
  // get the solution:
  auto solution = dss.solve();

  // exact value:
  auto exact = [](T x) { return (-x * x + x + 1.0); };

  std::cout << "tp : FDM | Exact\n";
  for (std::size_t j = 0; j < solution.size(); ++j) {
    std::cout << "t_" << j << ": " << solution[j] << " |  " << exact(j * h)
              << '\n';
  }
}

template <typename T>
void testBVPFDMSolverDirichletBC_2() {
  using lss_enumerations::boundary_condition_enum;
  using lss_fdm_double_sweep_solver::fdm_double_sweep_solver;
  using lss_fdm_tridiagonal_solvers::fdm_tridiagonal_solver;

  /*

  Solve BVP:

  u''(t) = - 2,

  where

  t \in (0, 1)
  u(0) = 1 ,  u(1) = 1


  Exact solution is

  u(t) = -t*t + t + 1

  */
  std::cout << "=================================\n";
  std::cout << "Solving Boundary-value problem: \n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " u''(t) = -2, \n\n";
  std::cout << " where\n\n";
  std::cout << " t in <0,1>,\n";
  std::cout << " u(0) = u(1) = 1\n\n";
  std::cout << "Exact solution is:\n\n";
  std::cout << " u(t) = -t*t + t + 1\n";
  std::cout << "=================================\n";

  // discretization:
  std::size_t N{20};
  // step size:
  T h = static_cast<T>(1.0) / static_cast<T>(N);
  // upper,mid, and lower diagonal:
  std::vector<T> upperDiag(N + 1, 1.0);
  std::vector<T> diagonal(N + 1, -2.0);
  std::vector<T> lowerDiag(N + 1, 1.0);

  // right-hand side:
  std::vector<T> rhs(N + 1, -2.0 * h * h);

  // boundary conditions:
  T left = 1.0;
  T right = 1.0;

  // typedef the solver
  typedef fdm_tridiagonal_solver<T, boundary_condition_enum::Dirichlet,
                                 fdm_double_sweep_solver, std::vector,
                                 std::allocator<T>>
      DoubleSweep;

  DoubleSweep dss{N + 1};
  dss.set_diagonals(std::move(lowerDiag), std::move(diagonal),
                    std::move(upperDiag));
  dss.set_boundary_condition(std::make_pair(left, right));
  dss.set_rhs(rhs);
  // get the solution:
  auto solution = dss.solve();

  // exact value:
  auto exact = [](T x) { return (-x * x + x + 1.0); };

  std::cout << "tp : FDM | Exact\n";
  for (std::size_t j = 0; j < solution.size(); ++j) {
    std::cout << "t_" << j << ": " << solution[j] << " |  " << exact(j * h)
              << '\n';
  }
}

void testDoubleSweepDirichletBC1() {
  std::cout << "==================================================\n";
  std::cout << "=========== Double Sweep (Dirichlet BC) ==========\n";
  std::cout << "==================================================\n";

  testBVPDoubleSweepDirichletBC1<double>();
  testBVPDoubleSweepDirichletBC1<float>();
  testBVPFDMSolverDirichletBC_2<double>();
  testBVPFDMSolverDirichletBC_2<float>();

  std::cout << "==================================================\n";
}
//
template <typename T>
void testBVPThomasLUSolverDirichletBC1() {
  using lss_enumerations::boundary_condition_enum;
  using lss_fdm_thomas_lu_solver::fdm_thomas_lu_solver;

  /*

  Solve BVP:

  u''(t) = - 2,

  where

  t \in (0, 1)
  u(0) = 0 ,  u(1) = 0


  Exact solution is

  u(t) = -t*t + t + 1

  */
  std::cout << "=================================\n";
  std::cout << "Solving Boundary-value problem: \n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " u''(t) = -2, \n\n";
  std::cout << " where\n\n";
  std::cout << " t in <0,1>,\n";
  std::cout << " u(0) = u(1) = 1\n\n";
  std::cout << "Exact solution is:\n\n";
  std::cout << " u(t) = -t*t + t + 1\n";
  std::cout << "=================================\n";

  // discretization:
  std::size_t N{20};
  // step size:
  T h = static_cast<T>(1.0) / static_cast<T>(N);
  // upper,mid, and lower diagonal:
  // must be of size N+1 because we need to add the t_0 point:
  std::vector<T> upperDiag(N + 1, 1.0);
  std::vector<T> diagonal(N + 1, -2.0);
  std::vector<T> lowerDiag(N + 1, 1.0);

  // right-hand side:
  std::vector<T> rhs(N + 1, -2.0 * h * h);

  // boundary conditions:
  T left = 1.0;
  T right = 1.0;

  fdm_thomas_lu_solver<T, boundary_condition_enum::Dirichlet, std::vector,
                       std::allocator<T>>
      dss{N + 1};
  dss.set_diagonals(std::move(lowerDiag), std::move(diagonal),
                    std::move(upperDiag));
  dss.set_boundary_condition(std::make_pair(left, right));
  dss.set_rhs(rhs);
  // get the solution:
  auto solution = dss.solve();

  // exact value:
  auto exact = [](T x) { return (-x * x + x + 1.0); };

  std::cout << "tp : FDM | Exact\n";
  for (std::size_t j = 0; j < solution.size(); ++j) {
    std::cout << "t_" << j << ": " << solution[j] << " |  " << exact(j * h)
              << '\n';
  }
}

template <typename T>
void testBVPFDMSolverDirichletBC_3() {
  using lss_enumerations::boundary_condition_enum;
  using lss_fdm_thomas_lu_solver::fdm_thomas_lu_solver;
  using lss_fdm_tridiagonal_solvers::fdm_tridiagonal_solver;

  /*

  Solve BVP:

  u''(t) = - 2,

  where

  t \in (0, 1)
  u(0) = 0 ,  u(1) = 0


  Exact solution is

  u(t) = -t*t + t + 1

  */
  std::cout << "=================================\n";
  std::cout << "Solving Boundary-value problem: \n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " u''(t) = -2, \n\n";
  std::cout << " where\n\n";
  std::cout << " t in <0,1>,\n";
  std::cout << " u(0) = u(1) = 1\n\n";
  std::cout << "Exact solution is:\n\n";
  std::cout << " u(t) = -t*t + t + 1\n";
  std::cout << "=================================\n";

  // discretization:
  std::size_t N{20};
  // step size:
  T h = static_cast<T>(1.0) / static_cast<T>(N);
  // upper,mid, and lower diagonal:
  // must be of size N+1 because we need to add the t_0 point:
  std::vector<T> upperDiag(N + 1, 1.0);
  std::vector<T> diagonal(N + 1, -2.0);
  std::vector<T> lowerDiag(N + 1, 1.0);

  // right-hand side:
  std::vector<T> rhs(N + 1, -2.0 * h * h);

  // boundary conditions:
  T left = 1.0;
  T right = 1.0;

  // typedef the solver
  typedef fdm_tridiagonal_solver<T, boundary_condition_enum::Dirichlet,
                                 fdm_thomas_lu_solver, std::vector,
                                 std::allocator<T>>
      ThomasLU;

  ThomasLU ts{N + 1};
  ts.set_diagonals(std::move(lowerDiag), std::move(diagonal),
                   std::move(upperDiag));
  ts.set_boundary_condition(std::make_pair(left, right));
  ts.set_rhs(rhs);
  // get the solution:
  auto solution = ts.solve();

  // exact value:
  auto exact = [](T x) { return (-x * x + x + 1.0); };

  std::cout << "tp : FDM | Exact\n";
  for (std::size_t j = 0; j < solution.size(); ++j) {
    std::cout << "t_" << j << ": " << solution[j] << " |  " << exact(j * h)
              << '\n';
  }
}

void testThomasLUSolverDirichletBC1() {
  std::cout << "==================================================\n";
  std::cout << "======== Thomas LU Solver (Dirichlet BC) =========\n";
  std::cout << "==================================================\n";

  testBVPThomasLUSolverDirichletBC1<double>();
  testBVPThomasLUSolverDirichletBC1<float>();
  testBVPFDMSolverDirichletBC_3<double>();
  testBVPFDMSolverDirichletBC_3<float>();

  std::cout << "==================================================\n";
}
//
template <typename T>
void testBVPDoubleSweepRobinBC() {
  using lss_enumerations::boundary_condition_enum;
  using lss_fdm_double_sweep_solver::fdm_double_sweep_solver;

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
  T h = static_cast<T>(1.0) / static_cast<T>(N);
  // upper,mid, and lower diagonal:
  std::vector<T> upperDiag(N + 1, 1.0);
  std::vector<T> diagonal(N + 1, -2.0);
  std::vector<T> lowerDiag(N + 1, 1.0);

  // right-hand side:
  std::vector<T> rhs(N + 1, -2.0 * h * h);

  // boundary conditions:
  T leftLin = 0.0;
  T leftConst = 1.0;
  auto left = std::make_pair(leftLin, leftConst);

  T rightLin = (2.0 + h) / (2.0 - h);
  T rightConst = 0.0;
  auto right = std::make_pair(rightLin, rightConst);

  fdm_double_sweep_solver<T, boundary_condition_enum::Robin, std::vector,
                          std::allocator<T>>
      dss{N + 1};
  dss.set_diagonals(std::move(lowerDiag), std::move(diagonal),
                    std::move(upperDiag));
  dss.set_boundary_condition(left, right);
  dss.set_rhs(rhs);
  // get the solution:
  auto solution = dss.solve();

  // exact value:
  auto exact = [](T x) { return (-x * x + x + 1.0); };

  std::cout << "tp : FDM | Exact | Abs Diff\n";
  for (std::size_t j = 0; j < solution.size(); ++j) {
    std::cout << "t_" << j << ": " << solution[j] << " |  " << exact(j * h)
              << " | " << (solution[j] - exact(j * h)) << '\n';
  }
}

template <typename T>
void testBVPFDMSolverRobinBC_0() {
  using lss_enumerations::boundary_condition_enum;
  using lss_fdm_double_sweep_solver::fdm_double_sweep_solver;
  using lss_fdm_tridiagonal_solvers::fdm_tridiagonal_solver;

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
  T h = static_cast<T>(1.0) / static_cast<T>(N);
  // upper,mid, and lower diagonal:
  std::vector<T> upperDiag(N + 1, 1.0f);
  std::vector<T> diagonal(N + 1, -2.0f);
  std::vector<T> lowerDiag(N + 1, 1.0f);

  // right-hand side:
  std::vector<T> rhs(N + 1, -2.0 * h * h);

  // boundary conditions:
  T leftLin = 0.0f;
  T leftConst = 1.0f;
  auto left = std::make_pair(leftLin, leftConst);

  T rightLin = (2.0 + h) / (2.0 - h);
  T rightConst = 0.0f;
  auto right = std::make_pair(rightLin, rightConst);

  // typedef the solver
  typedef fdm_tridiagonal_solver<T, boundary_condition_enum::Robin,
                                 fdm_double_sweep_solver, std::vector,
                                 std::allocator<T>>
      DoubleSweep;

  DoubleSweep dss{N + 1};
  dss.set_diagonals(std::move(lowerDiag), std::move(diagonal),
                    std::move(upperDiag));
  dss.set_boundary_condition(left, right);
  dss.set_rhs(rhs);
  // get the solution:
  auto solution = dss.solve();

  // exact value:
  auto exact = [](T x) { return (-x * x + x + 1.0); };

  std::cout << "tp : FDM | Exact | Abs Diff\n";
  for (std::size_t j = 0; j < solution.size(); ++j) {
    std::cout << "t_" << j << ": " << solution[j] << " |  " << exact(j * h)
              << " | " << (solution[j] - exact(j * h)) << '\n';
  }
}

void testDoubleSweepRobinBC() {
  std::cout << "==================================================\n";
  std::cout << "=========== Double Sweep (Robin BC) ==============\n";
  std::cout << "==================================================\n";

  testBVPDoubleSweepRobinBC<double>();
  testBVPDoubleSweepRobinBC<float>();
  testBVPFDMSolverRobinBC_0<double>();
  testBVPFDMSolverRobinBC_0<float>();

  std::cout << "==================================================\n";
}

//
template <typename T>
void testBVPThomasLUSolverRobinBC() {
  using lss_enumerations::boundary_condition_enum;
  using lss_fdm_thomas_lu_solver::fdm_thomas_lu_solver;

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
  T h = static_cast<T>(1.0) / static_cast<T>(N);
  // upper,mid, and lower diagonal:
  // must be of size N+1 because we need to add the t_0 point:
  std::vector<T> upperDiag(N + 1, 1.0);
  std::vector<T> diagonal(N + 1, -2.0);
  std::vector<T> lowerDiag(N + 1, 1.0);

  // right-hand side:
  std::vector<T> rhs(N + 1, -2.0 * h * h);

  // boundary conditions:
  T leftLin = 0.0;
  T leftConst = 1.0;
  auto left = std::make_pair(leftLin, leftConst);

  T rightLin = (2.0 + h) / (2.0 - h);
  T rightConst = 0.0;
  auto right = std::make_pair(rightLin, rightConst);

  fdm_thomas_lu_solver<T, boundary_condition_enum::Robin, std::vector,
                       std::allocator<T>>
      dss{N + 1};
  dss.set_diagonals(std::move(lowerDiag), std::move(diagonal),
                    std::move(upperDiag));
  dss.set_boundary_condition(left, right);
  dss.set_rhs(rhs);
  // get the solution:
  auto solution = dss.solve();

  // exact value:
  auto exact = [](T x) { return (-x * x + x + 1.0); };

  std::cout << "tp : FDM | Exact | Abs Diff\n";
  for (std::size_t j = 0; j < solution.size(); ++j) {
    std::cout << "t_" << j << ": " << solution[j] << " |  " << exact(j * h)
              << " | " << (solution[j] - exact(j * h)) << '\n';
  }
}

template <typename T>
void testBVPFDMSolverRobinBC_1() {
  using lss_enumerations::boundary_condition_enum;
  using lss_fdm_thomas_lu_solver::fdm_thomas_lu_solver;
  using lss_fdm_tridiagonal_solvers::fdm_tridiagonal_solver;

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
  T h = static_cast<T>(1.0) / static_cast<T>(N);
  // upper,mid, and lower diagonal:
  // must be of size N+1 because we need to add the t_0 point:
  std::vector<T> upperDiag(N + 1, 1.0);
  std::vector<T> diagonal(N + 1, -2.0);
  std::vector<T> lowerDiag(N + 1, 1.0);

  // right-hand side:
  std::vector<T> rhs(N + 1, -2.0 * h * h);

  // boundary conditions:
  T leftLin = 0.0;
  T leftConst = 1.0;
  auto left = std::make_pair(leftLin, leftConst);

  T rightLin = (2.0 + h) / (2.0 - h);
  T rightConst = 0.0;
  auto right = std::make_pair(rightLin, rightConst);

  // typedef the solver
  typedef fdm_tridiagonal_solver<T, boundary_condition_enum::Robin,
                                 fdm_thomas_lu_solver, std::vector,
                                 std::allocator<T>>
      ThomasLU;

  ThomasLU ts{N + 1};
  ts.set_diagonals(std::move(lowerDiag), std::move(diagonal),
                   std::move(upperDiag));
  ts.set_boundary_condition(left, right);
  ts.set_rhs(rhs);
  // get the solution:
  auto solution = ts.solve();

  // exact value:
  auto exact = [](T x) { return (-x * x + x + 1.0); };

  std::cout << "tp : FDM | Exact | Abs Diff\n";
  for (std::size_t j = 0; j < solution.size(); ++j) {
    std::cout << "t_" << j << ": " << solution[j] << " |  " << exact(j * h)
              << " | " << (solution[j] - exact(j * h)) << '\n';
  }
}

void testThomasLUSolverRobinBC() {
  std::cout << "==================================================\n";
  std::cout << "========= Thomas LU Solver (Robin BC) ========\n";
  std::cout << "==================================================\n";

  testBVPThomasLUSolverRobinBC<double>();
  testBVPThomasLUSolverRobinBC<float>();
  testBVPFDMSolverRobinBC_1<double>();
  testBVPFDMSolverRobinBC_1<float>();

  std::cout << "==================================================\n";
}

#endif  ///_LSS_FDM_TRIDIAGONAL_SOLVERS_T
