#pragma once
#if !defined(_LSS_ONE_DIM_PURE_HEAT_EQUATION_T)
#define _LSS_ONE_DIM_PURE_HEAT_EQUATION_T

#pragma warning(disable : 4305)

#include "common/lss_enumerations.h"
#include "common/lss_utility.h"
#include "pde_solvers/one_dim/classic/lss_one_dim_general_heat_equation_solvers.h"
#include "sparse_solvers/lss_fdm_double_sweep_solver.h"
#include "sparse_solvers/lss_fdm_thomas_lu_solver.h"

#define PI 3.14159

// ///////////////////////////////////////////////////////////////////////////
//							PURE HEAT PROBLEMS
// ///////////////////////////////////////////////////////////////////////////

// ===========================================================================
// ========================== IMPLICIT SOLVERS ===============================
// ===========================================================================

// ===========================================================================
// =========== Heat problem with homogeneous boundary conditions =============
// ===========================================================================

template <typename T>
void testImplPureHeatEquationDirichletBCDoubleSweepEuler() {
  using lss_enumerations::BoundaryConditionType;
  using lss_enumerations::ImplicitPDESchemes;
  using lss_fdm_double_sweep_solver::FDMDoubleSweepSolver;
  using lss_one_dim_general_heat_equation_solvers::implicit_solvers::
      Implicit1DGeneralHeatEquation;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) "
               "*sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DHeatEquation
  typedef Implicit1DGeneralHeatEquation<T, BoundaryConditionType::Dirichlet,
                                        FDMDoubleSweepSolver, std::vector,
                                        std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](T x) { return x; };
  // boundary conditions:
  auto const &dirichlet = [](T t) { return 0.0; };
  auto boundary = std::make_pair(dirichlet, dirichlet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.10, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::Euler);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T const first = static_cast<T>(2.0) / PI;
    T sum{};
    T var1{};
    T var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var1 = std::pow(-1.0, i + 1) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
      var2 = std::sin(i * PI * x) / i;
      sum += (var1 * var2);
    }
    return (first * sum);
  };

  T const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.10, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

template <typename T>
void testImplPureHeatEquationDirichletBCDoubleSweepCN() {
  using lss_enumerations::BoundaryConditionType;
  using lss_enumerations::ImplicitPDESchemes;
  using lss_fdm_double_sweep_solver::FDMDoubleSweepSolver;
  using lss_one_dim_general_heat_equation_solvers::implicit_solvers::
      Implicit1DGeneralHeatEquation;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Crank-Nicolson \n"
               "method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) "
               "*sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DHeatEquation
  typedef Implicit1DGeneralHeatEquation<T, BoundaryConditionType::Dirichlet,
                                        FDMDoubleSweepSolver, std::vector,
                                        std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](T x) { return x; };
  // boundary conditions:
  auto const &dirichlet = [](T t) { return 0.0; };
  auto boundary = std::make_pair(dirichlet, dirichlet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.10, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  impl_solver.solve(solution);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T const first = 2.0 / PI;
    T sum{};
    T var1{};
    T var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var1 = std::pow(-1.0, i + 1) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
      var2 = std::sin(i * PI * x) / i;
      sum += (var1 * var2);
    }
    return (first * sum);
  };

  T const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.10, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplPureHeatEquationDirichletBCDoubleSweep() {
  std::cout << "============================================================\n";
  std::cout << "======== Implicit Pure Heat Equation (Dirichlet BC) ========\n";
  std::cout << "============================================================\n";

  testImplPureHeatEquationDirichletBCDoubleSweepEuler<double>();
  testImplPureHeatEquationDirichletBCDoubleSweepEuler<float>();
  testImplPureHeatEquationDirichletBCDoubleSweepCN<double>();
  testImplPureHeatEquationDirichletBCDoubleSweepCN<float>();

  std::cout << "============================================================\n";
}

template <typename T>
void testImplPureHeatEquationDirichletBCThomasLUEuler() {
  using lss_enumerations::BoundaryConditionType;
  using lss_enumerations::ImplicitPDESchemes;
  using lss_fdm_thomas_lu_solver::FDMThomasLUSolver;
  using lss_one_dim_general_heat_equation_solvers::implicit_solvers::
      Implicit1DGeneralHeatEquation;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using Thomas LU algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) \n"
               "*sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DHeatEquation
  typedef Implicit1DGeneralHeatEquation<T, BoundaryConditionType::Dirichlet,
                                        FDMThomasLUSolver, std::vector,
                                        std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](T x) { return x; };
  // boundary conditions:
  auto const &dirichlet = [](T t) { return 0.0; };
  auto boundary = std::make_pair(dirichlet, dirichlet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.10, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::Euler);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T const first = 2.0 / PI;
    T sum{};
    T var1{};
    T var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var1 = std::pow(-1.0, i + 1) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
      var2 = std::sin(i * PI * x) / i;
      sum += (var1 * var2);
    }
    return (first * sum);
  };

  T const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.10, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

template <typename T>
void testImplPureHeatEquationDirichletBCThomasLUCN() {
  using lss_enumerations::BoundaryConditionType;
  using lss_enumerations::ImplicitPDESchemes;
  using lss_fdm_thomas_lu_solver::FDMThomasLUSolver;
  using lss_one_dim_general_heat_equation_solvers::implicit_solvers::
      Implicit1DGeneralHeatEquation;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using Thomas LU algorithm with \n";
  std::cout << " implicit Crank-Nicolson method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) \n"
               "*sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DHeatEquation
  typedef Implicit1DGeneralHeatEquation<T, BoundaryConditionType::Dirichlet,
                                        FDMThomasLUSolver, std::vector,
                                        std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 1000;
  // number of time subdivisions:
  std::size_t const Td = 1000;
  // initial condition:
  auto initialCondition = [](T x) { return x; };
  // boundary conditions:
  auto const &dirichlet = [](T t) { return 0.0; };
  auto boundary = std::make_pair(dirichlet, dirichlet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.10, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  impl_solver.solve(solution);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T const first = 2.0 / PI;
    T sum{};
    T var1{};
    T var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var1 = std::pow(-1.0, i + 1) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
      var2 = std::sin(i * PI * x) / i;
      sum += (var1 * var2);
    }
    return (first * sum);
  };

  T const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.10, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplPureHeatEquationDirichletBCThomasLU() {
  std::cout << "============================================================\n";
  std::cout << "======= Implicit Pure Heat Equation (Dirichlet BC) =========\n";
  std::cout << "============================================================\n";

  testImplPureHeatEquationDirichletBCThomasLUEuler<double>();
  testImplPureHeatEquationDirichletBCThomasLUEuler<float>();
  testImplPureHeatEquationDirichletBCThomasLUCN<double>();
  testImplPureHeatEquationDirichletBCThomasLUCN<float>();

  std::cout << "============================================================\n";
}

template <typename T>
void testImplPureHeatEquationRobinBCDoubleSweepEuler() {
  using lss_enumerations::BoundaryConditionType;
  using lss_enumerations::ImplicitPDESchemes;
  using lss_fdm_double_sweep_solver::FDMDoubleSweepSolver;
  using lss_one_dim_general_heat_equation_solvers::implicit_solvers::
      Implicit1DGeneralHeatEquation;
  using lss_one_dim_pde_utility::RobinBoundary;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout
      << " U(x,t) = 0.5 - (4/(pi*pi))*sum_1^infty{ exp(-((2n-1)*pi)^2*t)\n"
         "*cos((2n-1)*pi*x)/(2n-1)^2}\n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DHeatEquation
  typedef Implicit1DGeneralHeatEquation<T, BoundaryConditionType::Robin,
                                        FDMDoubleSweepSolver, std::vector,
                                        std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 400;
  // number of time subdivisions:
  std::size_t const Td = 150;
  // initial condition:
  auto initialCondition = [](T x) { return x; };
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.1, Sd, Td);
  // boundary conditions:
  // Robin boundaries are assumed to be of following form:
  //
  //				U_0 = leftLin * U_1 + leftConst
  //				U_{N-1} = rightLin * U_N + rightConst
  //
  // In our case discretizing the boundaries gives:
  //
  //				(U_1 - U_-1)/2h = 0
  //				(U_N+1 - U_{N-1})/2h = 0
  //
  // Therefore we have:
  //
  //				leftLin = 1.0, leftConst = 0.0
  //				rightLin = 1.0, rightConst = 0.0
  //
  // set boundary conditions:
  auto leftBoundary = std::make_pair(1.0, 0.0);
  auto rightBoundary = std::make_pair(1.0, 0.0);
  impl_solver.setBoundaryCondition(
      RobinBoundary<T>(leftBoundary, rightBoundary));
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::Euler);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T const pipi = PI * PI;
    T const first = 4.0 / pipi;
    T sum{};
    T var0{};
    T var1{};
    T var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var0 = (2 * i - 1);
      var1 = std::exp(-1.0 * pipi * var0 * var0 * t);
      var2 = std::cos(var0 * PI * x) / (var0 * var0);
      sum += (var1 * var2);
    }
    return (0.5 - first * sum);
  };

  T const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

template <typename T>
void testImplPureHeatEquationRobinBCDoubleSweepCN() {
  using lss_enumerations::BoundaryConditionType;
  using lss_enumerations::ImplicitPDESchemes;
  using lss_fdm_double_sweep_solver::FDMDoubleSweepSolver;
  using lss_one_dim_general_heat_equation_solvers::implicit_solvers::
      Implicit1DGeneralHeatEquation;
  using lss_one_dim_pde_utility::RobinBoundary;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Cranc-Nicolson \n"
               "method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout
      << " U(x,t) = 0.5 - (4/(pi*pi))*sum_1^infty{ exp(-((2n-1)*pi)^2*t) \n"
         "*cos((2n-1)*pi*x)/(2n-1)^2}\n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DHeatEquation
  typedef Implicit1DGeneralHeatEquation<T, BoundaryConditionType::Robin,
                                        FDMDoubleSweepSolver, std::vector,
                                        std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 1000;
  // number of time subdivisions:
  std::size_t const Td = 50;
  // initial condition:
  auto initialCondition = [](T x) { return x; };

  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.1, Sd, Td);
  // boundary conditions:
  // Robin boundaries are assumed to be of following form:
  //
  //				U_0 = leftLin * U_1 + leftConst
  //				U_{N-1} = rightLin * U_N + rightConst
  //
  // In our case discretizing the boundaries gives:
  //
  //				(U_1 - U_0)/h = 0
  //				(U_N - U_{N-1})/h = 0
  //
  // Therefore we have:
  //
  //				leftLin = 1.0, leftConst = 0.0
  //				rightLin = 1.0, rightConst = 0.0
  //
  // set boundary conditions:
  auto leftBoundary = std::make_pair(1.0, 0.0);
  auto rightBoundary = std::make_pair(1.0, 0.0);
  impl_solver.setBoundaryCondition(
      RobinBoundary<T>(leftBoundary, rightBoundary));
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  impl_solver.solve(solution);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T const pipi = PI * PI;
    T const first = 4.0 / pipi;
    T sum{};
    T var0{};
    T var1{};
    T var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var0 = (2 * i - 1);
      var1 = std::exp(-1.0 * pipi * var0 * var0 * t);
      var2 = std::cos(var0 * PI * x) / (var0 * var0);
      sum += (var1 * var2);
    }
    return (0.5 - first * sum);
  };

  T const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplPureHeatEquationRobinBCDoubleSweep() {
  std::cout << "============================================================\n";
  std::cout << "======= Implicit Pure Heat Equation (Robin BC) =============\n";
  std::cout << "============================================================\n";

  testImplPureHeatEquationRobinBCDoubleSweepEuler<double>();
  testImplPureHeatEquationRobinBCDoubleSweepEuler<float>();
  testImplPureHeatEquationRobinBCDoubleSweepCN<double>();
  testImplPureHeatEquationRobinBCDoubleSweepCN<float>();

  std::cout << "============================================================\n";
}

template <typename T>
void testImplPureHeatEquationRobinBCThomasLUEuler() {
  using lss_enumerations::BoundaryConditionType;
  using lss_enumerations::ImplicitPDESchemes;
  using lss_fdm_thomas_lu_solver::FDMThomasLUSolver;
  using lss_one_dim_general_heat_equation_solvers::implicit_solvers::
      Implicit1DGeneralHeatEquation;
  using lss_one_dim_pde_utility::RobinBoundary;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using Thomas LU algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout
      << " U(x,t) = 0.5 - (4/(pi*pi))*sum_1^infty{ exp(-((2n-1)*pi)^2*t) \n"
         "*cos((2n-1)*pi*x)/(2n-1)^2}\n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DHeatEquation
  typedef Implicit1DGeneralHeatEquation<T, BoundaryConditionType::Robin,
                                        FDMThomasLUSolver, std::vector,
                                        std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](T x) { return x; };
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.1, Sd, Td);
  // boundary conditions:
  // Robin boundaries are assumed to be of following form:
  //
  //				U_0 = leftLin * U_1 + leftConst
  //				U_{N-1} = rightLin * U_N + rightConst
  //
  // In our case discretizing the boundaries gives:
  //
  //				(U_1 - U_0)/h = 0
  //				(U_N - U_{N-1})/h = 0
  //
  // Therefore we have:
  //
  //				leftLin = 1.0, leftConst = 0.0
  //				rightLin = 1.0, rightConst = 0.0
  //
  // set boundary conditions:
  auto leftBoundary = std::make_pair(1.0, 0.0);
  auto rightBoundary = std::make_pair(1.0, 0.0);
  impl_solver.setBoundaryCondition(
      RobinBoundary<T>(leftBoundary, rightBoundary));
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::Euler);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T const pipi = PI * PI;
    T const first = 4.0 / pipi;
    T sum{};
    T var0{};
    T var1{};
    T var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var0 = (2 * i - 1);
      var1 = std::exp(-1.0 * pipi * var0 * var0 * t);
      var2 = std::cos(var0 * PI * x) / (var0 * var0);
      sum += (var1 * var2);
    }
    return (0.5 - first * sum);
  };

  T const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

template <typename T>
void testImplPureHeatEquationRobinBCThomasLUCN() {
  using lss_enumerations::BoundaryConditionType;
  using lss_enumerations::ImplicitPDESchemes;
  using lss_fdm_thomas_lu_solver::FDMThomasLUSolver;
  using lss_one_dim_general_heat_equation_solvers::implicit_solvers::
      Implicit1DGeneralHeatEquation;
  using lss_one_dim_pde_utility::RobinBoundary;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using Thomas LU algorithm with \n";
  std::cout << " implicit Crank-Nicolson method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout
      << " U(x,t) = 0.5 - (4/(pi*pi))*sum_1^infty{ exp(-((2n-1)*pi)^2*t) \n"
         "*cos((2n-1)*pi*x)/(2n-1)^2}\n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquation
  typedef Implicit1DGeneralHeatEquation<T, BoundaryConditionType::Robin,
                                        FDMThomasLUSolver, std::vector,
                                        std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 1000;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](T x) { return x; };
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.1, Sd, Td);
  // Robin boundaries are assumed to be of following form:
  //
  //				U_0 = leftLin * U_1 + leftConst
  //				U_{N-1} = rightLin * U_N + rightConst
  //
  // In our case discretizing the boundaries gives:
  //
  //				(U_1 - U_0)/h = 0
  //				(U_N - U_{N-1})/h = 0
  //
  // Therefore we have:
  //
  //				leftLin = 1.0, leftConst = 0.0
  //				rightLin = 1.0, rightConst = 0.0
  //
  // set boundary conditions:
  auto leftBoundary = std::make_pair(1.0, 0.0);
  auto rightBoundary = std::make_pair(1.0, 0.0);
  impl_solver.setBoundaryCondition(
      RobinBoundary<T>(leftBoundary, rightBoundary));
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  impl_solver.solve(solution);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T const pipi = PI * PI;
    T const first = 4.0 / pipi;
    T sum{};
    T var0{};
    T var1{};
    T var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var0 = (2 * i - 1);
      var1 = std::exp(-1.0 * pipi * var0 * var0 * t);
      var2 = std::cos(var0 * PI * x) / (var0 * var0);
      sum += (var1 * var2);
    }
    return (0.5 - first * sum);
  };

  T const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplPureHeatEquationRobinBCThomasLU() {
  std::cout << "============================================================\n";
  std::cout << "========== Implicit Pure Heat Equation (Robin BC) ==========\n";
  std::cout << "============================================================\n";

  testImplPureHeatEquationRobinBCThomasLUEuler<double>();
  testImplPureHeatEquationRobinBCThomasLUEuler<float>();
  testImplPureHeatEquationRobinBCThomasLUCN<double>();
  testImplPureHeatEquationRobinBCThomasLUCN<float>();

  std::cout << "============================================================\n";
}

// ===========================================================================
// ======= Heat problem with homogeneous boundary conditions and source ======
// ===========================================================================

template <typename T>
void testImplPureHeatEquationSourceDirichletBCDoubleSweepEuler() {
  using lss_enumerations::BoundaryConditionType;
  using lss_enumerations::ImplicitPDESchemes;
  using lss_fdm_double_sweep_solver::FDMDoubleSweepSolver;
  using lss_one_dim_general_heat_equation_solvers::implicit_solvers::
      Implicit1DGeneralHeatEquation;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with source: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = 1, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquation
  typedef Implicit1DGeneralHeatEquation<T, BoundaryConditionType::Dirichlet,
                                        FDMDoubleSweepSolver, std::vector,
                                        std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](T x) { return 1.0; };
  // boundary conditions:
  auto const &dirichlet = [](T t) { return 0.0; };
  auto boundary = std::make_pair(dirichlet, dirichlet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.10, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // set heat source:
  impl_solver.setHeatSource([](T x, T t) { return x; });
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::Euler);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T sum{};
    T q_n{};
    T f_n{};
    T lam_n{};
    T lam_2{};
    T var1{};
    for (std::size_t i = 1; i <= n; ++i) {
      q_n = (2.0 / (i * PI)) * std::pow(-1.0, i + 1);
      f_n = (2.0 / (i * PI)) * (1.0 - std::pow(-1.0, i));
      lam_n = i * PI;
      lam_2 = lam_n * lam_n;
      var1 =
          (q_n / lam_2 + (f_n - (q_n / lam_2)) * std::exp(-1.0 * lam_2 * t)) *
          std::sin(i * PI * x);
      sum += var1;
    }
    return sum;
  };

  T const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.10, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

template <typename T>
void testImplPureHeatEquationSourceDirichletBCDoubleSweepCN() {
  using lss_enumerations::BoundaryConditionType;
  using lss_enumerations::ImplicitPDESchemes;
  using lss_fdm_double_sweep_solver::FDMDoubleSweepSolver;
  using lss_one_dim_general_heat_equation_solvers::implicit_solvers::
      Implicit1DGeneralHeatEquation;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Crank-Nicolson \n"
               "method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = 1, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquation
  typedef Implicit1DGeneralHeatEquation<T, BoundaryConditionType::Dirichlet,
                                        FDMDoubleSweepSolver, std::vector,
                                        std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](T x) { return 1.0; };
  // boundary conditions:
  auto const &dirichlet = [](T t) { return 0.0; };
  auto boundary = std::make_pair(dirichlet, dirichlet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.10, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // set heat source:
  impl_solver.setHeatSource([](T x, T t) { return x; });
  // get the solution:
  impl_solver.solve(solution);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T sum{};
    T q_n{};
    T f_n{};
    T lam_n{};
    T lam_2{};
    T var1{};
    for (std::size_t i = 1; i <= n; ++i) {
      q_n = (2.0 / (i * PI)) * std::pow(-1.0, i + 1);
      f_n = (2.0 / (i * PI)) * (1.0 - std::pow(-1.0, i));
      lam_n = i * PI;
      lam_2 = lam_n * lam_n;
      var1 =
          ((q_n / lam_2) + (f_n - (q_n / lam_2)) * std::exp(-1.0 * lam_2 * t)) *
          std::sin(i * PI * x);
      sum += var1;
    }
    return sum;
  };

  T const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.10, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplPureHeatEquationSourceDirichletBCDoubleSweep() {
  std::cout << "============================================================\n";
  std::cout << "== Implicit Pure Heat Equation with source (Dirichlet BC) ==\n";
  std::cout << "============================================================\n";

  testImplPureHeatEquationSourceDirichletBCDoubleSweepEuler<double>();
  testImplPureHeatEquationSourceDirichletBCDoubleSweepEuler<float>();
  testImplPureHeatEquationSourceDirichletBCDoubleSweepCN<double>();
  testImplPureHeatEquationSourceDirichletBCDoubleSweepCN<float>();

  std::cout << "============================================================\n";
}

template <typename T>
void testImplPureHeatEquationSourceDirichletBCThomasLUEuler() {
  using lss_enumerations::BoundaryConditionType;
  using lss_enumerations::ImplicitPDESchemes;
  using lss_fdm_thomas_lu_solver::FDMThomasLUSolver;
  using lss_one_dim_general_heat_equation_solvers::implicit_solvers::
      Implicit1DGeneralHeatEquation;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using Thomas LU algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = 1, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquation
  typedef Implicit1DGeneralHeatEquation<T, BoundaryConditionType::Dirichlet,
                                        FDMThomasLUSolver, std::vector,
                                        std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](T x) { return 1.0; };
  // boundary conditions:
  auto const &dirichlet = [](T t) { return 0.0; };
  auto boundary = std::make_pair(dirichlet, dirichlet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.10, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // set heat source:
  impl_solver.setHeatSource([](T x, T t) { return x; });
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::Euler);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T sum{};
    T q_n{};
    T f_n{};
    T lam_n{};
    T lam_2{};
    T var1{};
    for (std::size_t i = 1; i <= n; ++i) {
      q_n = (2.0 / (i * PI)) * std::pow(-1.0, i + 1);
      f_n = (2.0 / (i * PI)) * (1.0 - std::pow(-1.0, i));
      lam_n = i * PI;
      lam_2 = lam_n * lam_n;
      var1 =
          ((q_n / lam_2) + (f_n - (q_n / lam_2)) * std::exp(-1.0 * lam_2 * t)) *
          std::sin(i * PI * x);
      sum += var1;
    }
    return sum;
  };

  T const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.10, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

template <typename T>
void testImplPureHeatEquationSourceDirichletBCThomasLUCN() {
  using lss_enumerations::BoundaryConditionType;
  using lss_enumerations::ImplicitPDESchemes;
  using lss_fdm_thomas_lu_solver::FDMThomasLUSolver;
  using lss_one_dim_general_heat_equation_solvers::implicit_solvers::
      Implicit1DGeneralHeatEquation;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with source: \n\n";
  std::cout << " Using Thomas LU algorithm with \n ";
  std::cout << " implicit Crank-Nicolson method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = 1, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquation
  typedef Implicit1DGeneralHeatEquation<T, BoundaryConditionType::Dirichlet,
                                        FDMThomasLUSolver, std::vector,
                                        std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 1000;
  // number of time subdivisions:
  std::size_t const Td = 1000;
  // initial condition:
  auto initialCondition = [](T x) { return 1.0; };
  // boundary conditions:
  auto const &dirichlet = [](T t) { return 0.0; };
  auto boundary = std::make_pair(dirichlet, dirichlet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.10, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // set heat source:
  impl_solver.setHeatSource([](T x, T t) { return x; });
  // get the solution:
  impl_solver.solve(solution);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T sum{};
    T q_n{};
    T f_n{};
    T lam_n{};
    T lam_2{};
    T var1{};
    for (std::size_t i = 1; i <= n; ++i) {
      q_n = (2.0 / (i * PI)) * std::pow(-1.0, i + 1);
      f_n = (2.0 / (i * PI)) * (1.0 - std::pow(-1.0, i));
      lam_n = i * PI;
      lam_2 = lam_n * lam_n;
      var1 =
          ((q_n / lam_2) + (f_n - (q_n / lam_2)) * std::exp(-1.0 * lam_2 * t)) *
          std::sin(i * PI * x);
      sum += var1;
    }
    return sum;
  };

  T const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.10, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplPureHeatEquationSourceDirichletBCThomasLU() {
  std::cout << "============================================================\n";
  std::cout << "=== Implicit Pure Heat Equation with source (Dirichlet BC) =\n";
  std::cout << "============================================================\n";

  testImplPureHeatEquationSourceDirichletBCThomasLUEuler<double>();
  testImplPureHeatEquationSourceDirichletBCThomasLUEuler<float>();
  testImplPureHeatEquationSourceDirichletBCThomasLUCN<double>();
  testImplPureHeatEquationSourceDirichletBCThomasLUCN<float>();

  std::cout << "============================================================\n";
}

// ===========================================================================
// ==== Heat problem with homogeneous Robin boundary conditions and source ===
// ===========================================================================

template <typename T>
void testImplPureHeatEquationSourceRobinBCDoubleSweepEuler() {
  using lss_enumerations::BoundaryConditionType;
  using lss_enumerations::ImplicitPDESchemes;
  using lss_fdm_double_sweep_solver::FDMDoubleSweepSolver;
  using lss_one_dim_general_heat_equation_solvers::implicit_solvers::
      Implicit1DGeneralHeatEquation;
  using lss_one_dim_pde_utility::RobinBoundary;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with Robin boundaries\n";
  std::cout << " and source: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquation
  typedef Implicit1DGeneralHeatEquation<T, BoundaryConditionType::Robin,
                                        FDMDoubleSweepSolver, std::vector,
                                        std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](T x) { return x; };
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.1, Sd, Td);
  // boundary conditions:
  auto leftBoundary = std::make_pair(1.0, 0.0);
  auto rightBoundary = std::make_pair(1.0, 0.0);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(
      RobinBoundary<T>(leftBoundary, rightBoundary));
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // set heat source:
  impl_solver.setHeatSource([](T x, T t) { return x; });
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::Euler);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T sum{};
    T q_n{};
    T f_n{};
    T lam_n{};
    T lam_2{};
    T var1{};
    for (std::size_t i = 1; i <= n; ++i) {
      lam_n = i * PI;
      lam_2 = lam_n * lam_n;
      q_n = (2.0 / (lam_2)) * (std::pow(-1.0, i) - 1.0);
      f_n = q_n;

      var1 =
          ((q_n / lam_2) + (f_n - (q_n / lam_2)) * std::exp(-1.0 * lam_2 * t)) *
          std::cos(lam_n * x);
      sum += var1;
    }
    return ((0.5 + 0.5 * t) + sum);
  };

  T const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

template <typename T>
void testImplPureHeatEquationSourceRobinBCDoubleSweepCN() {
  using lss_enumerations::BoundaryConditionType;
  using lss_enumerations::ImplicitPDESchemes;
  using lss_fdm_double_sweep_solver::FDMDoubleSweepSolver;
  using lss_one_dim_general_heat_equation_solvers::implicit_solvers::
      Implicit1DGeneralHeatEquation;
  using lss_one_dim_pde_utility::RobinBoundary;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with Robin boundaries \n ";
  std::cout << " and source: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Crank-Nicolson \n";
  std::cout << " method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquation
  typedef Implicit1DGeneralHeatEquation<T, BoundaryConditionType::Robin,
                                        FDMDoubleSweepSolver, std::vector,
                                        std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](T x) { return x; };
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.1, Sd, Td);
  // boundary conditions:
  auto leftBoundary = std::make_pair(1.0, 0.0);
  auto rightBoundary = std::make_pair(1.0, 0.0);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(
      RobinBoundary<T>(leftBoundary, rightBoundary));
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // set heat source:
  impl_solver.setHeatSource([](T x, T t) { return x; });
  // get the solution:
  impl_solver.solve(solution);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T sum{};
    T q_n{};
    T f_n{};
    T lam_n{};
    T lam_2{};
    T var1{};
    for (std::size_t i = 1; i <= n; ++i) {
      lam_n = i * PI;
      lam_2 = lam_n * lam_n;
      q_n = (2.0 / (lam_2)) * (std::pow(-1.0, i) - 1.0);
      f_n = q_n;

      var1 =
          ((q_n / lam_2) + (f_n - (q_n / lam_2)) * std::exp(-1.0 * lam_2 * t)) *
          std::cos(lam_n * x);
      sum += var1;
    }
    return ((0.5 + 0.5 * t) + sum);
  };

  T const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplPureHeatEquationSourceRobinBCDoubleSweep() {
  std::cout << "============================================================\n";
  std::cout << "=== Implicit Pure Heat Equation with source (Robin BC) =====\n";
  std::cout << "============================================================\n";

  testImplPureHeatEquationSourceRobinBCDoubleSweepEuler<double>();
  testImplPureHeatEquationSourceRobinBCDoubleSweepEuler<float>();
  testImplPureHeatEquationSourceRobinBCDoubleSweepCN<double>();
  testImplPureHeatEquationSourceRobinBCDoubleSweepCN<float>();

  std::cout << "============================================================\n";
}

template <typename T>
void testImplPureHeatEquationSourceRobinBCThomasLUEuler() {
  using lss_enumerations::BoundaryConditionType;
  using lss_enumerations::ImplicitPDESchemes;
  using lss_fdm_thomas_lu_solver::FDMThomasLUSolver;
  using lss_one_dim_general_heat_equation_solvers::implicit_solvers::
      Implicit1DGeneralHeatEquation;
  using lss_one_dim_pde_utility::RobinBoundary;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with Robin boundaries\n ";
  std::cout << " and source: \n\n";
  std::cout << " Using Thomas LU algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquation
  typedef Implicit1DGeneralHeatEquation<T, BoundaryConditionType::Robin,
                                        FDMThomasLUSolver, std::vector,
                                        std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](T x) { return x; };
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.1, Sd, Td);
  // boundary conditions:
  auto leftBoundary = std::make_pair(1.0, 0.0);
  auto rightBoundary = std::make_pair(1.0, 0.0);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(
      RobinBoundary<T>(leftBoundary, rightBoundary));
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // set heat source:
  impl_solver.setHeatSource([](T x, T t) { return x; });
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::Euler);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T sum{};
    T q_n{};
    T f_n{};
    T lam_n{};
    T lam_2{};
    T var1{};
    for (std::size_t i = 1; i <= n; ++i) {
      lam_n = i * PI;
      lam_2 = lam_n * lam_n;
      q_n = (2.0 / (lam_2)) * (std::pow(-1.0, i) - 1.0);
      f_n = q_n;

      var1 =
          ((q_n / lam_2) + (f_n - (q_n / lam_2)) * std::exp(-1.0 * lam_2 * t)) *
          std::cos(lam_n * x);
      sum += var1;
    }
    return ((0.5 + 0.5 * t) + sum);
  };

  T const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

template <typename T>
void testImplPureHeatEquationSourceRobinBCThomasLUCN() {
  using lss_enumerations::BoundaryConditionType;
  using lss_enumerations::ImplicitPDESchemes;
  using lss_fdm_thomas_lu_solver::FDMThomasLUSolver;
  using lss_one_dim_general_heat_equation_solvers::implicit_solvers::
      Implicit1DGeneralHeatEquation;
  using lss_one_dim_pde_utility::RobinBoundary;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with Robin boundaries \n";
  std::cout << " and source: \n\n";
  std::cout << " Using Thomas LU algorithm with \n";
  std::cout << " implicit Crank-Nicolson method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquation
  typedef Implicit1DGeneralHeatEquation<T, BoundaryConditionType::Robin,
                                        FDMThomasLUSolver, std::vector,
                                        std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 1000;
  // number of time subdivisions:
  std::size_t const Td = 1000;
  // initial condition:
  auto initialCondition = [](T x) { return x; };
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.1, Sd, Td);
  // boundary conditions:
  auto leftBoundary = std::make_pair(1.0, 0.0);
  auto rightBoundary = std::make_pair(1.0, 0.0);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(
      RobinBoundary<T>(leftBoundary, rightBoundary));
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // set heat source:
  impl_solver.setHeatSource([](T x, T t) { return x; });
  // get the solution:
  impl_solver.solve(solution);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T sum{};
    T q_n{};
    T f_n{};
    T lam_n{};
    T lam_2{};
    T var1{};
    for (std::size_t i = 1; i <= n; ++i) {
      lam_n = i * PI;
      lam_2 = lam_n * lam_n;
      q_n = (2.0 / (lam_2)) * (std::pow(-1.0, i) - 1.0);
      f_n = q_n;

      var1 =
          ((q_n / lam_2) + (f_n - (q_n / lam_2)) * std::exp(-1.0 * lam_2 * t)) *
          std::cos(lam_n * x);
      sum += var1;
    }
    return ((0.5 + 0.5 * t) + sum);
  };

  T const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplPureHeatEquationSourceRobinBCThomasLU() {
  std::cout << "============================================================\n";
  std::cout << "==== Implicit Pure Heat Equation with source (Robin BC) ====\n";
  std::cout << "============================================================\n";

  testImplPureHeatEquationSourceRobinBCThomasLUEuler<double>();
  testImplPureHeatEquationSourceRobinBCThomasLUEuler<float>();
  testImplPureHeatEquationSourceRobinBCThomasLUCN<double>();
  testImplPureHeatEquationSourceRobinBCThomasLUCN<float>();

  std::cout << "============================================================\n";
}

// ===========================================================================
// ============ Heat problem with nonhomogeneous boundary conditions =========
// ===========================================================================

template <typename T>
void testImplNonHomPureHeatEquationDirichletBCDoubleSweepEuler() {
  using lss_enumerations::BoundaryConditionType;
  using lss_enumerations::ImplicitPDESchemes;
  using lss_fdm_double_sweep_solver::FDMDoubleSweepSolver;
  using lss_one_dim_general_heat_equation_solvers::implicit_solvers::
      Implicit1DGeneralHeatEquation;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with Non-hom. BC: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = 0, U(1,t) = 100, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = 100*x + (198/pi)*sum_0^infty{ "
               "(-1)^(n)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquation
  typedef Implicit1DGeneralHeatEquation<T, BoundaryConditionType::Dirichlet,
                                        FDMDoubleSweepSolver, std::vector,
                                        std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](T x) { return x; };
  // boundary conditions:
  auto const &dirichletLeft = [](T t) { return 0.0; };
  auto const &dirichletRight = [](T t) { return 100.0; };
  auto boundary = std::make_pair(dirichletLeft, dirichletRight);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::Euler);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T const first = 198.0 / PI;
    T sum{};
    T var1{};
    T var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var1 = std::pow(-1.0, i) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
      var2 = std::sin(i * PI * x) / i;
      sum += (var1 * var2);
    }
    return (100 * x + first * sum);
  };

  T const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

template <typename T>
void testImplNonHomPureHeatEquationDirichletBCDoubleSweepCN() {
  using lss_enumerations::BoundaryConditionType;
  using lss_enumerations::ImplicitPDESchemes;
  using lss_fdm_double_sweep_solver::FDMDoubleSweepSolver;
  using lss_one_dim_general_heat_equation_solvers::implicit_solvers::
      Implicit1DGeneralHeatEquation;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with Non-hom. BC:: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Crank-Nicolson \n"
               "method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(0,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = 100*x + (198/pi)*sum_0^infty{ "
               "(-1)^(n)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquation
  typedef Implicit1DGeneralHeatEquation<T, BoundaryConditionType::Dirichlet,
                                        FDMDoubleSweepSolver, std::vector,
                                        std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](T x) { return x; };
  // boundary conditions:
  auto const &dirichletLeft = [](T t) { return 0.0; };
  auto const &dirichletRight = [](T t) { return 100.0; };
  auto boundary = std::make_pair(dirichletLeft, dirichletRight);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  impl_solver.solve(solution);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T const first = 198.0 / PI;
    T sum{};
    T var1{};
    T var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var1 = std::pow(-1.0, i) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
      var2 = std::sin(i * PI * x) / i;
      sum += (var1 * var2);
    }
    return (100 * x + first * sum);
  };

  T const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplNonHomPureHeatEquationDirichletBCDoubleSweep() {
  std::cout << "============================================================\n";
  std::cout << "= Implicit Pure Heat Equation (non-homogenous Dirichlet BC) \n";
  std::cout << "============================================================\n";

  testImplNonHomPureHeatEquationDirichletBCDoubleSweepEuler<double>();
  testImplNonHomPureHeatEquationDirichletBCDoubleSweepEuler<float>();
  testImplNonHomPureHeatEquationDirichletBCDoubleSweepCN<double>();
  testImplNonHomPureHeatEquationDirichletBCDoubleSweepCN<float>();

  std::cout << "============================================================\n";
}

template <typename T>
void testImplNonHomPureHeatEquationDirichletBCThomasLUEuler() {
  using lss_enumerations::BoundaryConditionType;
  using lss_enumerations::ImplicitPDESchemes;
  using lss_fdm_thomas_lu_solver::FDMThomasLUSolver;
  using lss_one_dim_general_heat_equation_solvers::implicit_solvers::
      Implicit1DGeneralHeatEquation;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with non-hom BC: \n\n";
  std::cout << " Using Thomas LU algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(0,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = 100.0*x + (198/pi)*sum_0^infty{ \n"
               "(-1)^(n)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquation
  typedef Implicit1DGeneralHeatEquation<T, BoundaryConditionType::Dirichlet,
                                        FDMThomasLUSolver, std::vector,
                                        std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](T x) { return x; };
  // boundary conditions:
  auto const &dirichletLeft = [](T t) { return 0.0; };
  auto const &dirichletRight = [](T t) { return 100.0; };
  auto boundary = std::make_pair(dirichletLeft, dirichletRight);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::Euler);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T const first = 198.0 / PI;
    T sum{};
    T var1{};
    T var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var1 = std::pow(-1.0, i) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
      var2 = std::sin(i * PI * x) / i;
      sum += (var1 * var2);
    }
    return (100.0 * x + first * sum);
  };

  T const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

template <typename T>
void testImplNonHomPureHeatEquationDirichletBCThomasLUCN() {
  using lss_enumerations::BoundaryConditionType;
  using lss_enumerations::ImplicitPDESchemes;
  using lss_fdm_thomas_lu_solver::FDMThomasLUSolver;
  using lss_one_dim_general_heat_equation_solvers::implicit_solvers::
      Implicit1DGeneralHeatEquation;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with non-hom BC: \n\n";
  std::cout << " Using Thomas LU algorithm with \n";
  std::cout << "implicit Crank-Nicolson method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = 0, U(1,t) = 100, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = 100*x + (198/pi)*sum_0^infty{ "
               "(-1)^(n)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquation
  typedef Implicit1DGeneralHeatEquation<T, BoundaryConditionType::Dirichlet,
                                        FDMThomasLUSolver, std::vector,
                                        std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](T x) { return x; };
  // boundary conditions:
  auto const &dirichletLeft = [](T t) { return 0.0; };
  auto const &dirichletRight = [](T t) { return 100.0; };
  auto boundary = std::make_pair(dirichletLeft, dirichletRight);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  impl_solver.solve(solution);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T const first = 198.0 / PI;
    T sum{};
    T var1{};
    T var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var1 = std::pow(-1.0, i) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
      var2 = std::sin(i * PI * x) / i;
      sum += (var1 * var2);
    }
    return (100.0 * x + first * sum);
  };

  T const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplNonHomPureHeatEquationDirichletBCThomasLU() {
  std::cout << "============================================================\n";
  std::cout << "===== Implicit Pure Heat Equation (with non-homogeneous \n"
               "Dirichlet BC) ==========\n";
  std::cout << "============================================================\n";

  testImplNonHomPureHeatEquationDirichletBCThomasLUEuler<double>();
  testImplNonHomPureHeatEquationDirichletBCThomasLUEuler<float>();
  testImplNonHomPureHeatEquationDirichletBCThomasLUCN<double>();
  testImplNonHomPureHeatEquationDirichletBCThomasLUCN<float>();

  std::cout << "============================================================\n";
}

// ============================================================================
// ====================== EPLICIT SOLVERS =====================================
// ============================================================================

// ============================================================================
// =============== Heat problem with homogeneous boundary conditions ==========
// ============================================================================

template <typename T>
void testExplPureHeatEquationDirichletBCEuler() {
  using lss_enumerations::BoundaryConditionType;
  using lss_enumerations::ExplicitPDESchemes;
  using lss_one_dim_general_heat_equation_solvers::explicit_solvers::
      Explicit1DGeneralHeatEquation;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using explicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) \n"
               "*sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Explicit1DGeneralHeatEquation
  typedef Explicit1DGeneralHeatEquation<T, BoundaryConditionType::Dirichlet,
                                        std::vector, std::allocator<T>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 10000;
  // initial condition:
  auto initialCondition = [](T x) { return x; };
  // boundary conditions:
  auto const &dirichlet = [](T x) { return 0.0; };
  auto boundary = std::make_pair(dirichlet, dirichlet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  explicit_solver expl_solver(Range<T>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  expl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  expl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  expl_solver.solve(solution, ExplicitPDESchemes::Euler);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T const first = 2.0 / PI;
    T sum{};
    T var1{};
    T var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var1 = std::pow(-1.0, i + 1) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
      var2 = std::sin(i * PI * x) / i;
      sum += (var1 * var2);
    }
    return (first * sum);
  };

  T const h = expl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

template <typename T>
void testExplPureHeatEquationDirichletBCADEBC() {
  using lss_enumerations::BoundaryConditionType;
  using lss_enumerations::ExplicitPDESchemes;
  using lss_one_dim_general_heat_equation_solvers::explicit_solvers::
      Explicit1DGeneralHeatEquation;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using explicit ADE Barakat Clark method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) "
               "*sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Explicit1DGeneralHeatEquation
  typedef Explicit1DGeneralHeatEquation<T, BoundaryConditionType::Dirichlet,
                                        std::vector, std::allocator<T>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 10000;
  // initial condition:
  auto initialCondition = [](T x) { return x; };
  // boundary conditions:
  auto const &dirichlet = [](T x) { return 0.0; };
  auto boundary = std::make_pair(dirichlet, dirichlet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  explicit_solver expl_solver(Range<T>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  expl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  expl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  expl_solver.solve(solution);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T const first = 2.0 / PI;
    T sum{};
    T var1{};
    T var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var1 = std::pow(-1.0, i + 1) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
      var2 = std::sin(i * PI * x) / i;
      sum += (var1 * var2);
    }
    return (first * sum);
  };

  T const h = expl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

template <typename T>
void testExplPureHeatEquationDirichletBCADES() {
  using lss_enumerations::BoundaryConditionType;
  using lss_enumerations::ExplicitPDESchemes;
  using lss_one_dim_general_heat_equation_solvers::explicit_solvers::
      Explicit1DGeneralHeatEquation;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using explicit ADE Saulyev method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) \n"
               "*sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Explicit1DGeneralHeatEquation
  typedef Explicit1DGeneralHeatEquation<T, BoundaryConditionType::Dirichlet,
                                        std::vector, std::allocator<T>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 10000;
  // initial condition:
  auto initialCondition = [](T x) { return x; };
  // boundary conditions:
  auto const &dirichlet = [](T x) { return 0.0; };
  auto boundary = std::make_pair(dirichlet, dirichlet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  explicit_solver expl_solver(Range<T>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  expl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  expl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  expl_solver.solve(solution, ExplicitPDESchemes::ADESaulyev);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T const first = 2.0 / PI;
    T sum{};
    T var1{};
    T var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var1 = std::pow(-1.0, i + 1) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
      var2 = std::sin(i * PI * x) / i;
      sum += (var1 * var2);
    }
    return (first * sum);
  };

  T const h = expl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testExplPureHeatEquationDirichletBC() {
  std::cout << "============================================================\n";
  std::cout << "====== Explicit Pure Heat Equation (Dirichlet BC) ==========\n";
  std::cout << "============================================================\n";

  testExplPureHeatEquationDirichletBCEuler<double>();
  testExplPureHeatEquationDirichletBCEuler<float>();
  testExplPureHeatEquationDirichletBCADEBC<double>();
  testExplPureHeatEquationDirichletBCADEBC<float>();
  testExplPureHeatEquationDirichletBCADES<double>();
  testExplPureHeatEquationDirichletBCADES<float>();

  std::cout << "============================================================\n";
}

// ============================================================================
// ====== Heat problem with homogeneous boundary conditions and source ========
// ============================================================================

template <typename T>
void testExplPureHeatEquationSourceDirichletBCEuler() {
  using lss_enumerations::BoundaryConditionType;
  using lss_enumerations::ExplicitPDESchemes;
  using lss_one_dim_general_heat_equation_solvers::explicit_solvers::
      Explicit1DGeneralHeatEquation;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with source: \n\n";
  std::cout << " Using explicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = 1, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Explicit1DGeneralHeatEquation
  typedef Explicit1DGeneralHeatEquation<T, BoundaryConditionType::Dirichlet,
                                        std::vector, std::allocator<T>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 10000;
  // initial condition:
  auto initialCondition = [](T x) { return 1.0; };
  // boundary conditions:
  auto const &dirichlet = [](T x) { return 0.0; };
  auto boundary = std::make_pair(dirichlet, dirichlet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  explicit_solver expl_solver(Range<T>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  expl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  expl_solver.setInitialCondition(initialCondition);
  // set heat source:
  expl_solver.setHeatSource([](T x, T t) { return x; });
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  expl_solver.solve(solution, ExplicitPDESchemes::Euler);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T sum{};
    T q_n{};
    T f_n{};
    T lam_n{};
    T lam_2{};
    T var1{};
    for (std::size_t i = 1; i <= n; ++i) {
      q_n = (2.0 / (i * PI)) * std::pow(-1.0, i + 1);
      f_n = (2.0 / (i * PI)) * (1.0 - std::pow(-1.0, i));
      lam_n = i * PI;
      lam_2 = lam_n * lam_n;
      var1 =
          (q_n / lam_2 + (f_n - (q_n / lam_2)) * std::exp(-1.0 * lam_2 * t)) *
          std::sin(i * PI * x);
      sum += var1;
    }
    return sum;
  };

  T const h = expl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

template <typename T>
void testExplPureHeatEquationSourceDirichletBCADEBC() {
  using lss_enumerations::BoundaryConditionType;
  using lss_enumerations::ExplicitPDESchemes;
  using lss_one_dim_general_heat_equation_solvers::explicit_solvers::
      Explicit1DGeneralHeatEquation;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with source: \n\n";
  std::cout << " Using explicit ADE Barakat Clark method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = 1, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Explicit1DGeneralHeatEquation
  typedef Explicit1DGeneralHeatEquation<T, BoundaryConditionType::Dirichlet,
                                        std::vector, std::allocator<T>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 10000;
  // initial condition:
  auto initialCondition = [](T x) { return 1.0; };
  // boundary conditions:
  auto const &dirichlet = [](T x) { return 0.0; };
  auto boundary = std::make_pair(dirichlet, dirichlet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  explicit_solver expl_solver(Range<T>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  expl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  expl_solver.setInitialCondition(initialCondition);
  // set heat source:
  expl_solver.setHeatSource([](T x, T t) { return x; });
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  expl_solver.solve(solution);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T sum{};
    T q_n{};
    T f_n{};
    T lam_n{};
    T lam_2{};
    T var1{};
    for (std::size_t i = 1; i <= n; ++i) {
      q_n = (2.0 / (i * PI)) * std::pow(-1.0, i + 1);
      f_n = (2.0 / (i * PI)) * (1.0 - std::pow(-1.0, i));
      lam_n = i * PI;
      lam_2 = lam_n * lam_n;
      var1 =
          (q_n / lam_2 + (f_n - (q_n / lam_2)) * std::exp(-1.0 * lam_2 * t)) *
          std::sin(i * PI * x);
      sum += var1;
    }
    return sum;
  };

  T const h = expl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

template <typename T>
void testExplPureHeatEquationSourceDirichletBCADES() {
  using lss_enumerations::BoundaryConditionType;
  using lss_enumerations::ExplicitPDESchemes;
  using lss_one_dim_general_heat_equation_solvers::explicit_solvers::
      Explicit1DGeneralHeatEquation;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with source: \n\n";
  std::cout << " Using explicit ADE Saulyev method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = 1.0, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Explicit1DGeneralHeatEquation
  typedef Explicit1DGeneralHeatEquation<T, BoundaryConditionType::Dirichlet,
                                        std::vector, std::allocator<T>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 10000;
  // initial condition:
  auto initialCondition = [](T x) { return 1.0; };
  // boundary conditions:
  auto const &dirichlet = [](T x) { return 0.0; };
  auto boundary = std::make_pair(dirichlet, dirichlet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  explicit_solver expl_solver(Range<T>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  expl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  expl_solver.setInitialCondition(initialCondition);
  // set heat source:
  expl_solver.setHeatSource([](T x, T t) { return x; });
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  expl_solver.solve(solution, ExplicitPDESchemes::ADESaulyev);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T sum{};
    T q_n{};
    T f_n{};
    T lam_n{};
    T lam_2{};
    T var1{};
    for (std::size_t i = 1; i <= n; ++i) {
      q_n = (2.0 / (i * PI)) * std::pow(-1.0, i + 1);
      f_n = (2.0 / (i * PI)) * (1.0 - std::pow(-1.0, i));
      lam_n = i * PI;
      lam_2 = lam_n * lam_n;
      var1 =
          (q_n / lam_2 + (f_n - (q_n / lam_2)) * std::exp(-1.0 * lam_2 * t)) *
          std::sin(i * PI * x);
      sum += var1;
    }
    return sum;
  };

  T const h = expl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testExplPureHeatEquationSourceDirichletBC() {
  std::cout << "============================================================\n";
  std::cout << "== Explicit Pure Heat Equation with source (Dirichlet BC) ==\n";
  std::cout << "============================================================\n";

  testExplPureHeatEquationSourceDirichletBCEuler<double>();
  testExplPureHeatEquationSourceDirichletBCEuler<float>();
  testExplPureHeatEquationSourceDirichletBCADEBC<double>();
  testExplPureHeatEquationSourceDirichletBCADEBC<float>();
  testExplPureHeatEquationSourceDirichletBCADES<double>();
  testExplPureHeatEquationSourceDirichletBCADES<float>();

  std::cout << "============================================================\n";
}

// ============================================================================
// ========== Heat problem with nonhomogeneous boundary conditions ============
// ============================================================================

template <typename T>
void testExplNonHomPureHeatEquationDirichletBCEuler() {
  using lss_enumerations::BoundaryConditionType;
  using lss_enumerations::ExplicitPDESchemes;
  using lss_one_dim_general_heat_equation_solvers::explicit_solvers::
      Explicit1DGeneralHeatEquation;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with Non-hom BC: \n\n";
  std::cout << " Using explicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = 0, U(1,t) = 100, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = 100*x + (198/pi)*sum_0^infty{ "
               "(-1)^(n)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Explicit1DGeneralHeatEquation
  typedef Explicit1DGeneralHeatEquation<T, BoundaryConditionType::Dirichlet,
                                        std::vector, std::allocator<T>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 10000;
  // initial condition:
  auto initialCondition = [](T x) { return x; };
  // boundary conditions:
  auto const &dirichletLeft = [](T x) { return 0.0; };
  auto const &dirichletRight = [](T x) { return 100.0; };
  auto boundary = std::make_pair(dirichletLeft, dirichletRight);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  explicit_solver expl_solver(Range<T>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  expl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  expl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  expl_solver.solve(solution, ExplicitPDESchemes::Euler);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T const first = 198.0 / PI;
    T sum{};
    T var1{};
    T var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var1 = std::pow(-1.0, i) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
      var2 = std::sin(i * PI * x) / i;
      sum += (var1 * var2);
    }
    return (100.0 * x + first * sum);
  };

  T const h = expl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

template <typename T>
void testExplNonHomPureHeatEquationDirichletBCADEBC() {
  using lss_enumerations::BoundaryConditionType;
  using lss_enumerations::ExplicitPDESchemes;
  using lss_one_dim_general_heat_equation_solvers::explicit_solvers::
      Explicit1DGeneralHeatEquation;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with non-hom BC: \n\n";
  std::cout << " Using explicit ADE Barakat Clark method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = 0, U(1,t) = 100, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = 100*x +  (198/pi)*sum_0^infty{ "
               "(-1)^(n+1)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Explicit1DGeneralHeatEquation
  typedef Explicit1DGeneralHeatEquation<T, BoundaryConditionType::Dirichlet,
                                        std::vector, std::allocator<T>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 10000;
  // initial condition:
  auto initialCondition = [](T x) { return x; };
  // boundary conditions:
  auto const &dirichletLeft = [](T x) { return 0.0; };
  auto const &dirichletRight = [](T x) { return 100.0; };
  auto boundary = std::make_pair(dirichletLeft, dirichletRight);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  explicit_solver expl_solver(Range<T>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  expl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  expl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  expl_solver.solve(solution);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T const first = 198.0 / PI;
    T sum{};
    T var1{};
    T var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var1 = std::pow(-1.0, i) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
      var2 = std::sin(i * PI * x) / i;
      sum += (var1 * var2);
    }
    return (100.0 * x + first * sum);
  };

  T const h = expl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

template <typename T>
void testExplNonHomPureHeatEquationDirichletBCADES() {
  using lss_enumerations::BoundaryConditionType;
  using lss_enumerations::ExplicitPDESchemes;
  using lss_one_dim_general_heat_equation_solvers::explicit_solvers::
      Explicit1DGeneralHeatEquation;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with non-hom BC: \n\n";
  std::cout << " Using explicit ADE Saulyev method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = 0, U(1,t) = 100, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = 100.0*x + (198/pi)*sum_0^infty{ \n"
               "(-1)^(n)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Explicit1DGeneralHeatEquation
  typedef Explicit1DGeneralHeatEquation<T, BoundaryConditionType::Dirichlet,
                                        std::vector, std::allocator<T>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 10000;
  // initial condition:
  auto initialCondition = [](T x) { return x; };
  // boundary conditions:
  auto const &dirichletLeft = [](T x) { return 0.0; };
  auto const &dirichletRight = [](T x) { return 100.0; };
  auto boundary = std::make_pair(dirichletLeft, dirichletRight);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  explicit_solver expl_solver(Range<T>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  expl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  expl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  expl_solver.solve(solution, ExplicitPDESchemes::ADESaulyev);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T const first = 198.0 / PI;
    T sum{};
    T var1{};
    T var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var1 = std::pow(-1.0, i) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
      var2 = std::sin(i * PI * x) / i;
      sum += (var1 * var2);
    }
    return (100.0 * x + first * sum);
  };

  T const h = expl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testExplNonHomPureHeatEquationDirichletBC() {
  std::cout << "============================================================\n";
  std::cout << "========= Explicit Pure Heat Equation (with non-homogeneous \n"
               "Dirichlet BC) ======\n";
  std::cout << "============================================================\n";

  testExplNonHomPureHeatEquationDirichletBCEuler<double>();
  testExplNonHomPureHeatEquationDirichletBCEuler<float>();
  testExplNonHomPureHeatEquationDirichletBCADEBC<double>();
  testExplNonHomPureHeatEquationDirichletBCADEBC<float>();
  testExplNonHomPureHeatEquationDirichletBCADES<double>();
  testExplNonHomPureHeatEquationDirichletBCADES<float>();

  std::cout << "============================================================\n";
}

// ============================================================================
// =========== Heat problem with homogeneous Robin boundary conditions ========
// ============================================================================

template <typename T>
void testExplHomPureHeatEquationRobinBCEuler() {
  using lss_enumerations::BoundaryConditionType;
  using lss_one_dim_general_heat_equation_solvers::explicit_solvers::
      Explicit1DGeneralHeatEquation;
  using lss_one_dim_pde_utility::RobinBoundary;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using explicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout
      << " U(x,t) = 0.5 - (4/(pi*pi))*sum_1^infty{ exp(-((2n-1)*pi)^2*t) \n"
         "*cos((2n-1)*pi*x)/(2n-1)^2}\n\n";
  std::cout << "============================================================\n";

  // typedef the Explicit1DGeneralHeatEquation
  typedef Explicit1DGeneralHeatEquation<T, BoundaryConditionType::Robin,
                                        std::vector, std::allocator<T>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 5000;
  // initial condition:
  auto initialCondition = [](T x) { return x; };
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, 0.0);
  // initialize solver
  explicit_solver expl_solver(Range<T>(0.0, 1.0), 0.1, Sd, Td);
  // boundary conditions:
  // Robin boundaries are assumed to be of following form:
  //
  //				U_0 = leftLin * U_1 + leftConst
  //				U_{N-1} = rightLin * U_N + rightConst
  //
  // In our case discretizing the boundaries gives:
  //
  //				(U_1 - U_-1)/2h = 0
  //				(U_N+1 - U_{N-1})/2h = 0
  //
  // Therefore we have:
  //
  //				leftLin = 1.0, leftConst = 0.0
  //				rightLin = 1.0, rightConst = 0.0
  //
  auto const h = expl_solver.spaceStep();
  auto leftBoundary = std::make_pair(1.0, 0.0);
  auto rightBoundary = std::make_pair(1.0, 0.0);
  // set boundary conditions:
  expl_solver.setBoundaryCondition(
      RobinBoundary<T>(leftBoundary, rightBoundary));
  // set initial condition:
  expl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  expl_solver.solve(solution);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T const pipi = PI * PI;
    T const first = 4.0 / pipi;
    T sum{};
    T var0{};
    T var1{};
    T var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var0 = (2 * i - 1);
      var1 = std::exp(-1.0 * pipi * var0 * var0 * t);
      var2 = std::cos(var0 * PI * x) / (var0 * var0);
      sum += (var1 * var2);
    }
    return (0.5 - first * sum);
  };

  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testExplHomPureHeatEquationRobinBC() {
  std::cout << "============================================================\n";
  std::cout << "======= Explicit Pure Heat Equation (with homogeneous Robin \n"
               "BC) ==============\n";
  std::cout << "============================================================\n";

  testExplHomPureHeatEquationRobinBCEuler<double>();
  testExplHomPureHeatEquationRobinBCEuler<float>();

  std::cout << "============================================================\n";
}

// ============================================================================
// === Heat problem with homogeneous Robin boundary conditions and source =====
// ============================================================================

template <typename T>
void testExplHomPureHeatEquationSourceRobinBCEuler() {
  using lss_enumerations::BoundaryConditionType;
  using lss_one_dim_general_heat_equation_solvers::explicit_solvers::
      Explicit1DGeneralHeatEquation;
  using lss_one_dim_pde_utility::RobinBoundary;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with heat source: \n\n";
  std::cout << " Using explicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Explicit1DGeneralHeatEquation
  typedef Explicit1DGeneralHeatEquation<T, BoundaryConditionType::Robin,
                                        std::vector, std::allocator<T>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 5000;
  // initial condition:
  auto initialCondition = [](T x) { return x; };
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, 0.0);
  // initialize solver
  explicit_solver expl_solver(Range<T>(0.0, 1.0), 0.1, Sd, Td);
  // boundary conditions:
  // Robin boundaries are assumed to be of following form:
  //
  //				U_0 = leftLin * U_1 + leftConst
  //				U_{N-1} = rightLin * U_N + rightConst
  //
  // In our case discretizing the boundaries gives:
  //
  //				(U_1 - U_-1)/2h = 0
  //				(U_N+1 - U_{N-1})/2h = 0
  //
  // Therefore we have:
  //
  //				leftLin = 1.0, leftConst = 0.0
  //				rightLin = 1.0, rightConst = 0.0
  //
  auto const h = expl_solver.spaceStep();
  auto leftBoundary = std::make_pair(1.0, 0.0);
  auto rightBoundary = std::make_pair(1.0, 0.0);
  // set boundary conditions:
  expl_solver.setBoundaryCondition(
      RobinBoundary<T>(leftBoundary, rightBoundary));
  // set initial condition:
  expl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set2OrderCoefficient(1.0);
  // set heat source:
  expl_solver.setHeatSource([](T x, T t) { return x; });
  // get the solution:
  expl_solver.solve(solution);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T sum{};
    T q_n{};
    T f_n{};
    T lam_n{};
    T lam_2{};
    T var1{};
    for (std::size_t i = 1; i <= n; ++i) {
      lam_n = i * PI;
      lam_2 = lam_n * lam_n;
      q_n = (2.0 / (lam_2)) * (std::pow(-1.0, i) - 1.0);
      f_n = q_n;

      var1 =
          ((q_n / lam_2) + (f_n - (q_n / lam_2)) * std::exp(-1.0 * lam_2 * t)) *
          std::cos(lam_n * x);
      sum += var1;
    }
    return ((0.5 + 0.5 * t) + sum);
  };

  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testExplHomPureHeatEquationSourceRobinBC() {
  std::cout << "============================================================\n";
  std::cout << "====== Explicit Pure Heat Equation with Source (with \n"
               "homogeneous Robin BC) =====\n";
  std::cout << "============================================================\n";

  testExplHomPureHeatEquationSourceRobinBCEuler<double>();
  testExplHomPureHeatEquationSourceRobinBCEuler<float>();

  std::cout << "============================================================\n";
}

#endif  ///_LSS_ONE_DIM_PURE_HEAT_EQUATION_T
