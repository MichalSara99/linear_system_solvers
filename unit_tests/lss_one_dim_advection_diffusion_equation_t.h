#pragma once
#if !defined(_LSS_ONE_DIM_ADVECTION_DIFFUSION_EQUATION_T)
#define _LSS_ONE_DIM_ADVECTION_DIFFUSION_EQUATION_T

#pragma warning(disable : 4305)

#include "common/lss_types.h"
#include "common/lss_utility.h"
#include "pde_solvers/one_dim/classic/lss_one_dim_general_heat_equation_solvers.h"
#include "sparse_solvers/lss_fdm_double_sweep_solver.h"
#include "sparse_solvers/lss_fdm_thomas_lu_solver.h"

#define PI 3.14159

// ///////////////////////////////////////////////////////////////////////////
//					ADVECTION-DIFFUSION	PROBLEMS
// ///////////////////////////////////////////////////////////////////////////

// ===========================================================================
// ======================== IMPLICIT SOLVERS =================================
// ===========================================================================

// ===========================================================================
// ====== Advection Diffusion problem with homogeneous boundary conditions ===
// ===========================================================================

template <typename T>
void testImplAdvDiffEquationDirichletBCDoubleSweepEuler() {
  using lss_fdm_double_sweep_solver::FDMDoubleSweepSolver;
  using lss_one_dim_general_heat_equation_solvers::implicit_solvers::
      Implicit1DGeneralHeatEquation;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Advection Diffusion equation: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
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
  auto boundary = std::make_pair(0.0, 0.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.08, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // set convection term
  impl_solver.set1OrderCoefficient(-1.0);
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::Euler);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T const first = 2.0 / PI;
    T const exp_0p5x = std::exp(0.5 * x);
    T const exp_m0p5 = std::exp(-0.5);
    T np_sqr{};
    T sum{};
    T num{}, den{}, var{};
    T lambda{};
    for (std::size_t i = 1; i <= n; ++i) {
      np_sqr = (i * i * PI * PI);
      lambda = 0.25 + np_sqr;
      num = (1.0 - std::pow(-1.0, i) * exp_m0p5) * exp_0p5x *
            std::exp(-1.0 * lambda * t) * std::sin(i * PI * x);
      den = i * (1.0 + (0.25 / np_sqr));
      var = num / den;
      sum += var;
    }
    return (first * sum);
  };

  T const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.08, 30);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

template <typename T>
void testImplAdvDiffEquationDirichletBCDoubleSweepCN() {
  using lss_fdm_double_sweep_solver::FDMDoubleSweepSolver;
  using lss_one_dim_general_heat_equation_solvers::implicit_solvers::
      Implicit1DGeneralHeatEquation;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Advection Diffusion equation: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Crank-Nicolson \n"
               "method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = -sin(pi*x), x in <0,1> \n\n";
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
  auto boundary = std::make_pair(0.0, 0.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.09, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // set convection term
  impl_solver.set1OrderCoefficient(-1.0);
  // get the solution:
  impl_solver.solve(solution);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T const first = 2.0 / PI;
    T const exp_0p5x = std::exp(0.5 * x);
    T const exp_m0p5 = std::exp(-0.5);
    T np_sqr{};
    T sum{};
    T num{}, den{}, var{};
    T lambda{};
    for (std::size_t i = 1; i <= n; ++i) {
      np_sqr = (i * i * PI * PI);
      lambda = 0.25 + np_sqr;
      num = (1.0 - std::pow(-1.0, i) * exp_m0p5) * exp_0p5x *
            std::exp(-1.0 * lambda * t) * std::sin(i * PI * x);
      den = i * (1.0 + (0.25 / np_sqr));
      var = num / den;
      sum += var;
    }
    return (first * sum);
  };

  T const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.09, 40);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplAdvDiffEquationDirichletBCDoubleSweep() {
  std::cout << "============================================================\n";
  std::cout << "=== Implicit Advection Diffusion Equation (Dirichlet BC) ===\n";
  std::cout << "============================================================\n";

  testImplAdvDiffEquationDirichletBCDoubleSweepEuler<double>();
  testImplAdvDiffEquationDirichletBCDoubleSweepEuler<float>();
  testImplAdvDiffEquationDirichletBCDoubleSweepCN<double>();
  testImplAdvDiffEquationDirichletBCDoubleSweepCN<float>();

  std::cout << "============================================================\n";
}

// ===========================================================================
// ====== Advection Diffusion problem with homogeneous Dirichlet boundary ====
// =========================== conditions and Source =========================
// ===========================================================================

template <typename T>
void testImplAdvDiffEquationSourceDirichletBCDoubleSweepEuler() {
  using lss_fdm_double_sweep_solver::FDMDoubleSweepSolver;
  using lss_one_dim_general_heat_equation_solvers::implicit_solvers::
      Implicit1DGeneralHeatEquation;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Advection Diffusion equation with "
               "source: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t) + x, \n\n";
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
  auto boundary = std::make_pair(0.0, 0.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.08, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // set convection term
  impl_solver.set1OrderCoefficient(-1.0);
  // set source function:
  impl_solver.setHeatSource([](T x, T t) { return x; });
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::Euler);

  T const h = impl_solver.spaceStep();
  std::cout << "tp : FDM \n";
  for (std::size_t j = 0; j < solution.size(); ++j) {
    std::cout << "t_" << j << ": " << solution[j] << '\n';
  }
}

template <typename T>
void testImplAdvDiffEquationSourceDirichletBCDoubleSweepCN() {
  using lss_fdm_double_sweep_solver::FDMDoubleSweepSolver;
  using lss_one_dim_general_heat_equation_solvers::implicit_solvers::
      Implicit1DGeneralHeatEquation;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Advection Diffusion equation with \n"
               "source: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Crank-Nicolson \n"
               "method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = 1.0, x in <0,1> \n\n";
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
  auto boundary = std::make_pair(0.0, 0.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.08, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // set convection term
  impl_solver.set1OrderCoefficient(-1.0);
  // set source function:
  impl_solver.setHeatSource([](T x, T t) { return x; });
  // get the solution:
  impl_solver.solve(solution);

  T const h = impl_solver.spaceStep();
  std::cout << "tp : FDM \n";
  for (std::size_t j = 0; j < solution.size(); ++j) {
    std::cout << "t_" << j << ": " << solution[j] << '\n';
  }
}

void testImplAdvDiffEquationSourceDirichletBCDoubleSweep() {
  std::cout << "============================================================\n";
  std::cout << "= Implicit Advection Diffusion Equation with source ========\n"
               "=================== (Dirichlet BC) =========================\n";
  std::cout << "============================================================\n";

  testImplAdvDiffEquationSourceDirichletBCDoubleSweepEuler<double>();
  testImplAdvDiffEquationSourceDirichletBCDoubleSweepEuler<float>();
  testImplAdvDiffEquationSourceDirichletBCDoubleSweepCN<double>();
  testImplAdvDiffEquationSourceDirichletBCDoubleSweepCN<float>();

  std::cout << "============================================================\n";
}

template <typename T>
void testImplAdvDiffEquationSourceDirichletBCThomasLUEuler() {
  using lss_fdm_thomas_lu_solver::FDMThomasLUSolver;
  using lss_one_dim_general_heat_equation_solvers::implicit_solvers::
      Implicit1DGeneralHeatEquation;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Advection Diffusion equation with "
               "source: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t) + x, \n\n";
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
  auto boundary = std::make_pair(0.0, 0.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.08, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // set convection term
  impl_solver.set1OrderCoefficient(-1.0);
  // set source function:
  impl_solver.setHeatSource([](T x, T t) { return x; });
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::Euler);

  T const h = impl_solver.spaceStep();
  std::cout << "tp : FDM \n";
  for (std::size_t j = 0; j < solution.size(); ++j) {
    std::cout << "t_" << j << ": " << solution[j] << '\n';
  }
}

template <typename T>
void testImplAdvDiffEquationSourceDirichletBCThomasLUCN() {
  using lss_fdm_thomas_lu_solver::FDMThomasLUSolver;
  using lss_one_dim_general_heat_equation_solvers::implicit_solvers::
      Implicit1DGeneralHeatEquation;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Advection Diffusion equation with \n"
               "source: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Crank-Nicolson \n"
               "method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = 1.0, x in <0,1> \n\n";
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
  auto boundary = std::make_pair(0.0, 0.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.08, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // set convection term
  impl_solver.set1OrderCoefficient(-1.0);
  // set source function:
  impl_solver.setHeatSource([](T x, T t) { return x; });
  // get the solution:
  impl_solver.solve(solution);

  T const h = impl_solver.spaceStep();
  std::cout << "tp : FDM \n";
  for (std::size_t j = 0; j < solution.size(); ++j) {
    std::cout << "t_" << j << ": " << solution[j] << '\n';
  }
}

void testImplAdvDiffEquationSourceDirichletBCThomasLU() {
  std::cout << "============================================================\n";
  std::cout << "==== Implicit Advection Diffusion Equation with source =====\n"
               "======================== Dirichlet BC ======================\n";
  std::cout << "============================================================\n";

  testImplAdvDiffEquationSourceDirichletBCThomasLUEuler<double>();
  testImplAdvDiffEquationSourceDirichletBCThomasLUEuler<float>();
  testImplAdvDiffEquationSourceDirichletBCThomasLUCN<double>();
  testImplAdvDiffEquationSourceDirichletBCThomasLUCN<float>();

  std::cout << "============================================================\n";
}

// ===========================================================================
// = Advection Diffusion problem with homogeneous Robin boundary conditions ==
// ===========================================================================

template <typename T>
void testImplAdvDiffEquationRobinBCDoubleSweepEuler() {
  using lss_fdm_double_sweep_solver::FDMDoubleSweepSolver;
  using lss_one_dim_general_heat_equation_solvers::implicit_solvers::
      Implicit1DGeneralHeatEquation;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Advection Diffusion equation: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = exp(0.5*x), x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquation
  typedef Implicit1DGeneralHeatEquation<T, BoundaryConditionType::Robin,
                                        FDMDoubleSweepSolver, std::vector,
                                        std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 400;
  // number of time subdivisions:
  std::size_t const Td = 150;
  // initial condition:
  auto initialCondition = [](T x) { return std::exp(0.5 * x); };
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
  impl_solver.setBoundaryCondition(leftBoundary, rightBoundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // set convection term
  impl_solver.set1OrderCoefficient(-1.0);
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::Euler);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T const zero = 2.0 * (1.0 - std::exp(-0.5)) / (1.0 - std::exp(-1.0));
    T const first = 2.0;
    T const exp_0p5x = std::exp(0.5 * x);
    T exp_lamt{};
    T sum{};
    T num{}, den{}, var{};
    T lambda_n{};
    T delta_n{};

    for (std::size_t i = 1; i <= n; ++i) {
      delta_n = i * PI;
      lambda_n = 0.25 + delta_n * delta_n;
      exp_lamt = std::exp(-1.0 * lambda_n * t);
      num = (1.0 - std::pow(-1.0, i)) * exp_0p5x * exp_lamt *
            (std::sin(delta_n * x) - 2.0 * delta_n * std::cos(delta_n * x));
      den = delta_n * (1.0 + 4.0 * delta_n * delta_n);
      var = num / den;
      sum += var;
    }
    return (zero + first * sum);
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
void testImplAdvDiffEquationRobinBCDoubleSweepCN() {
  using lss_fdm_double_sweep_solver::FDMDoubleSweepSolver;
  using lss_one_dim_general_heat_equation_solvers::implicit_solvers::
      Implicit1DGeneralHeatEquation;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Advection Diffusion equation: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Crank-Nicolson \n"
               " method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = exp(0.5*x), x in <0,1> \n\n";
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
  auto initialCondition = [](T x) { return std::exp(0.5 * x); };
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
  impl_solver.setBoundaryCondition(leftBoundary, rightBoundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // set convection term
  impl_solver.set1OrderCoefficient(-1.0);
  // get the solution:
  impl_solver.solve(solution);
  // get exact solution:
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T const zero = 2.0 * (1.0 - std::exp(-0.5)) / (1.0 - std::exp(-1.0));
    T const first = 2.0;
    T const exp_0p5x = std::exp(0.5 * x);
    T exp_lamt{};
    T sum{};
    T num{}, den{}, var{};
    T lambda_n{};
    T delta_n{};

    for (std::size_t i = 1; i <= n; ++i) {
      delta_n = i * PI;
      lambda_n = 0.25 + delta_n * delta_n;
      exp_lamt = std::exp(-1.0 * lambda_n * t);
      num = (1.0 - std::pow(-1.0, i)) * exp_0p5x * exp_lamt *
            (std::sin(delta_n * x) - 2.0 * delta_n * std::cos(delta_n * x));
      den = delta_n * (1.0 + 4.0 * delta_n * delta_n);
      var = num / den;
      sum += var;
    }
    return (zero + first * sum);
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

void testImplAdvDiffEquationRobinBCDoubleSweep() {
  std::cout << "============================================================\n";
  std::cout << "===== Implicit Advection Diffusion Equation (Robin BC) =====\n";
  std::cout << "============================================================\n";

  testImplAdvDiffEquationRobinBCDoubleSweepEuler<double>();
  testImplAdvDiffEquationRobinBCDoubleSweepEuler<float>();
  testImplAdvDiffEquationRobinBCDoubleSweepCN<double>();
  testImplAdvDiffEquationRobinBCDoubleSweepCN<float>();

  std::cout << "============================================================\n";
}

template <typename T>
void testImplAdvDiffEquationRobinBCThomasLUEuler() {
  using lss_fdm_thomas_lu_solver::FDMThomasLUSolver;
  using lss_one_dim_general_heat_equation_solvers::implicit_solvers::
      Implicit1DGeneralHeatEquation;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Advection Diffusion equation: \n\n";
  std::cout << " Using Thomas LU algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = exp(0.5*x), x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquation
  typedef Implicit1DGeneralHeatEquation<T, BoundaryConditionType::Robin,
                                        FDMThomasLUSolver, std::vector,
                                        std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 400;
  // number of time subdivisions:
  std::size_t const Td = 150;
  // initial condition:
  auto initialCondition = [](T x) { return std::exp(0.5 * x); };
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
  impl_solver.setBoundaryCondition(leftBoundary, rightBoundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // set convection term
  impl_solver.set1OrderCoefficient(-1.0);
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::Euler);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T const zero = 2.0 * (1.0 - std::exp(-0.5)) / (1.0 - std::exp(-1.0));
    T const first = 2.0;
    T const exp_0p5x = std::exp(0.5 * x);
    T exp_lamt{};
    T sum{};
    T num{}, den{}, var{};
    T lambda_n{};
    T delta_n{};

    for (std::size_t i = 1; i <= n; ++i) {
      delta_n = i * PI;
      lambda_n = 0.25 + delta_n * delta_n;
      exp_lamt = std::exp(-1.0 * lambda_n * t);
      num = (1.0 - std::pow(-1.0, i)) * exp_0p5x * exp_lamt *
            (std::sin(delta_n * x) - 2.0 * delta_n * std::cos(delta_n * x));
      den = delta_n * (1.0 + 4.0 * delta_n * delta_n);
      var = num / den;
      sum += var;
    }
    return (zero + first * sum);
  };

  T const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 30);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

template <typename T>
void testImplAdvDiffEquationRobinBCThomasLUCN() {
  using lss_fdm_thomas_lu_solver::FDMThomasLUSolver;
  using lss_one_dim_general_heat_equation_solvers::implicit_solvers::
      Implicit1DGeneralHeatEquation;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Advection Diffusion equation: \n\n";
  std::cout << " Using Thomas LU algorithm with \n";
  std::cout << " implicit Crank-Nicolson method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = exp(0.5*x), x in <0,1> \n\n";
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
  auto initialCondition = [](T x) { return std::exp(0.5 * x); };
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
  impl_solver.setBoundaryCondition(leftBoundary, rightBoundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // set convection term
  impl_solver.set1OrderCoefficient(-1.0);
  // get the solution:
  impl_solver.solve(solution);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T const zero = 2.0 * (1.0 - std::exp(-0.5)) / (1.0 - std::exp(-1.0));
    T const first = 2.0;
    T const exp_0p5x = std::exp(0.5 * x);
    T exp_lamt{};
    T sum{};
    T num{}, den{}, var{};
    T lambda_n{};
    T delta_n{};

    for (std::size_t i = 1; i <= n; ++i) {
      delta_n = i * PI;
      lambda_n = 0.25 + delta_n * delta_n;
      exp_lamt = std::exp(-1.0 * lambda_n * t);
      num = (1.0 - std::pow(-1.0, i)) * exp_0p5x * exp_lamt *
            (std::sin(delta_n * x) - 2.0 * delta_n * std::cos(delta_n * x));
      den = delta_n * (1.0 + 4.0 * delta_n * delta_n);
      var = num / den;
      sum += var;
    }
    return (zero + first * sum);
  };

  T const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 30);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplAdvDiffEquationRobinBCThomasLU() {
  std::cout << "============================================================\n";
  std::cout << "= Implicit Advection Diffusion Equation (ThomasLU,Robin BC) \n";
  std::cout << "============================================================\n";

  testImplAdvDiffEquationRobinBCThomasLUEuler<double>();
  testImplAdvDiffEquationRobinBCThomasLUEuler<float>();
  testImplAdvDiffEquationRobinBCThomasLUCN<double>();
  testImplAdvDiffEquationRobinBCThomasLUCN<float>();

  std::cout << "============================================================\n";
}

// ===========================================================================
// ========================== EXPLICIT SOLVERS ===============================
// ===========================================================================

// ===========================================================================
// ===== Advection Diffusion problem with homogeneous boundary conditions ====
// ===========================================================================

template <typename T>
void testExplAdvDiffEquationDirichletBCEuler() {
  using lss_one_dim_general_heat_equation_solvers::explicit_solvers::
      Explicit1DGeneralHeatEquation;
  using lss_types::BoundaryConditionType;
  using lss_types::ExplicitPDESchemes;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Advection Diffusion equation: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <-1,1> and t > 0,\n";
  std::cout << " U(-1,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = 1, x in <-1,1> \n\n";
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
  auto boundary = std::make_pair(0.0, 0.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  explicit_solver expl_solver(Range<T>(0.0, 1.0), 0.08, Sd, Td);
  // set boundary conditions:
  expl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  expl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set2OrderCoefficient(1.0);
  // set convection term
  expl_solver.set1OrderCoefficient(-1.0);
  // get the solution:
  expl_solver.solve(solution, ExplicitPDESchemes::Euler);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T const first = 2.0 / PI;
    T const exp_0p5x = std::exp(0.5 * x);
    T const exp_m0p5 = std::exp(-0.5);
    T np_sqr{};
    T sum{};
    T num{}, den{}, var{};
    T lambda{};
    for (std::size_t i = 1; i <= n; ++i) {
      np_sqr = (i * i * PI * PI);
      lambda = 0.25 + np_sqr;
      num = (1.0 - std::pow(-1.0, i) * exp_m0p5) * exp_0p5x *
            std::exp(-1.0 * lambda * t) * std::sin(i * PI * x);
      den = i * (1.0 + (0.25 / np_sqr));
      var = num / den;
      sum += var;
    }
    return (first * sum);
  };

  T const h = expl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.08, 30);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

template <typename T>
void testExplAdvDiffEquationDirichletBCADEBC() {
  using lss_one_dim_general_heat_equation_solvers::explicit_solvers::
      Explicit1DGeneralHeatEquation;
  using lss_types::BoundaryConditionType;
  using lss_types::ExplicitPDESchemes;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Advection Diffusion equation: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <-1,1> and t > 0,\n";
  std::cout << " U(-1,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = 1, x in <-1,1> \n\n";
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
  auto boundary = std::make_pair(0.0, 0.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  explicit_solver expl_solver(Range<T>(0.0, 1.0), 0.08, Sd, Td);
  // set boundary conditions:
  expl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  expl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set2OrderCoefficient(1.0);
  // set convection term
  expl_solver.set1OrderCoefficient(-1.0);
  // get the solution:
  expl_solver.solve(solution);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T const first = 2.0 / PI;
    T const exp_0p5x = std::exp(0.5 * x);
    T const exp_m0p5 = std::exp(-0.5);
    T np_sqr{};
    T sum{};
    T num{}, den{}, var{};
    T lambda{};
    for (std::size_t i = 1; i <= n; ++i) {
      np_sqr = (i * i * PI * PI);
      lambda = 0.25 + np_sqr;
      num = (1.0 - std::pow(-1.0, i) * exp_m0p5) * exp_0p5x *
            std::exp(-1.0 * lambda * t) * std::sin(i * PI * x);
      den = i * (1.0 + (0.25 / np_sqr));
      var = num / den;
      sum += var;
    }
    return (first * sum);
  };

  T const h = expl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.08, 30);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

template <typename T>
void testExplAdvDiffEquationDirichletBCADES() {
  using lss_one_dim_general_heat_equation_solvers::explicit_solvers::
      Explicit1DGeneralHeatEquation;
  using lss_types::BoundaryConditionType;
  using lss_types::ExplicitPDESchemes;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Advection Diffusion equation: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <-1,1> and t > 0,\n";
  std::cout << " U(-1,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = 1, x in <-1,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquation
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
  auto boundary = std::make_pair(0.0, 0.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  explicit_solver expl_solver(Range<T>(0.0, 1.0), 0.08, Sd, Td);
  // set boundary conditions:
  expl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  expl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set2OrderCoefficient(1.0);
  // set convection term
  expl_solver.set1OrderCoefficient(-1.0);
  // get the solution:
  expl_solver.solve(solution, ExplicitPDESchemes::ADESaulyev);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T const first = 2.0 / PI;
    T const exp_0p5x = std::exp(0.5 * x);
    T const exp_m0p5 = std::exp(-0.5);
    T np_sqr{};
    T sum{};
    T num{}, den{}, var{};
    T lambda{};
    for (std::size_t i = 1; i <= n; ++i) {
      np_sqr = (i * i * PI * PI);
      lambda = 0.25 + np_sqr;
      num = (1.0 - std::pow(-1.0, i) * exp_m0p5) * exp_0p5x *
            std::exp(-1.0 * lambda * t) * std::sin(i * PI * x);
      den = i * (1.0 + (0.25 / np_sqr));
      var = num / den;
      sum += var;
    }
    return (first * sum);
  };

  T const h = expl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.08, 30);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testExplAdvDiffEquationDirichletBC() {
  std::cout << "============================================================\n";
  std::cout << "=== Explicit Advection Diffusion Equation (Dirichlet BC) ===\n";
  std::cout << "============================================================\n";

  testExplAdvDiffEquationDirichletBCEuler<double>();
  testExplAdvDiffEquationDirichletBCEuler<float>();
  testExplAdvDiffEquationDirichletBCADEBC<double>();
  testExplAdvDiffEquationDirichletBCADEBC<float>();
  testExplAdvDiffEquationDirichletBCADES<double>();
  testExplAdvDiffEquationDirichletBCADES<float>();

  std::cout << "============================================================\n";
}

// ===========================================================================
// ========= Advection Diffusion problem with homogeneous boundary ===========
// =========================== conditions and source =========================
// ===========================================================================

template <typename T>
void testExplAdvDiffEquationSourceDirichletBCEuler() {
  using lss_one_dim_general_heat_equation_solvers::explicit_solvers::
      Explicit1DGeneralHeatEquation;
  using lss_types::BoundaryConditionType;
  using lss_types::ExplicitPDESchemes;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Advection Diffusion equation with \n"
               "source: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t) + x, \n\n";
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
  auto boundary = std::make_pair(0.0, 0.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  explicit_solver expl_solver(Range<T>(0.0, 1.0), 0.08, Sd, Td);
  // set boundary conditions:
  expl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  expl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set2OrderCoefficient(1.0);
  // set convection term
  expl_solver.set1OrderCoefficient(-1.0);
  // set heat source:
  expl_solver.setHeatSource([](T x, T t) { return x; });
  // get the solution:
  expl_solver.solve(solution, ExplicitPDESchemes::Euler);

  T const h = expl_solver.spaceStep();
  std::cout << "tp : FDM \n";
  for (std::size_t j = 0; j < solution.size(); ++j) {
    std::cout << "t_" << j << ": " << solution[j] << '\n';
  }
}

template <typename T>
void testExplAdvDiffEquationSourceDirichletBCADEBC() {
  using lss_one_dim_general_heat_equation_solvers::explicit_solvers::
      Explicit1DGeneralHeatEquation;
  using lss_types::BoundaryConditionType;
  using lss_types::ExplicitPDESchemes;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Advection Diffusion equation with \n"
               "source: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t) + x, \n\n";
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
  auto boundary = std::make_pair(0.0, 0.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  explicit_solver expl_solver(Range<T>(0.0, 1.0), 0.08, Sd, Td);
  // set boundary conditions:
  expl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  expl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set2OrderCoefficient(1.0);
  // set convection term
  expl_solver.set1OrderCoefficient(-1.0);
  // set heat source:
  expl_solver.setHeatSource([](T x, T t) { return x; });
  // get the solution:
  expl_solver.solve(solution);

  std::cout << "tp : FDM \n";
  for (std::size_t j = 0; j < solution.size(); ++j) {
    std::cout << "t_" << j << ": " << solution[j] << '\n';
  }
}

template <typename T>
void testExplAdvDiffEquationSourceDirichletBCADES() {
  using lss_one_dim_general_heat_equation_solvers::explicit_solvers::
      Explicit1DGeneralHeatEquation;
  using lss_types::BoundaryConditionType;
  using lss_types::ExplicitPDESchemes;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Advection Diffusion equation with \n"
               "source: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t) + x, \n\n";
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
  auto boundary = std::make_pair(0.0, 0.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  explicit_solver expl_solver(Range<T>(0.0, 1.0), 0.08, Sd, Td);
  // set boundary conditions:
  expl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  expl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set2OrderCoefficient(1.0);
  // set convection term
  expl_solver.set1OrderCoefficient(-1.0);
  // set heat source:
  expl_solver.setHeatSource([](T x, T t) { return x; });
  // get the solution:
  expl_solver.solve(solution, ExplicitPDESchemes::ADESaulyev);

  std::cout << "tp : FDM \n";
  for (std::size_t j = 0; j < solution.size(); ++j) {
    std::cout << "t_" << j << ": " << solution[j] << '\n';
  }
}

void testExplAdvDiffEquationSourceDirichletBC() {
  std::cout << "============================================================\n";
  std::cout << "======= Explicit Advection Diffusion Equation with source ==\n"
               "======================  Dirichlet BC =======================\n";
  std::cout << "============================================================\n";

  testExplAdvDiffEquationSourceDirichletBCEuler<double>();
  testExplAdvDiffEquationSourceDirichletBCEuler<float>();
  testExplAdvDiffEquationSourceDirichletBCADEBC<double>();
  testExplAdvDiffEquationSourceDirichletBCADEBC<float>();
  testExplAdvDiffEquationSourceDirichletBCADES<double>();
  testExplAdvDiffEquationSourceDirichletBCADES<float>();

  std::cout << "============================================================\n";
}

// ===========================================================================
// == Advection Diffusion problem with homogeneous Robin boundary conditions =
// ===========================================================================

template <typename T>
void testExplAdvDiffEquationRobinBCEuler() {
  using lss_one_dim_general_heat_equation_solvers::explicit_solvers::
      Explicit1DGeneralHeatEquation;
  using lss_types::BoundaryConditionType;
  using lss_types::ExplicitPDESchemes;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Advection Diffusion equation: \n\n";
  std::cout << " Using explicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = exp(0.5*x), x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Explicit1DGeneralHeatEquation
  typedef Explicit1DGeneralHeatEquation<T, BoundaryConditionType::Robin,
                                        std::vector, std::allocator<T>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 10000;
  // initial condition:
  auto initialCondition = [](T x) { return std::exp(0.5 * x); };
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  explicit_solver expl_solver(Range<T>(0.0, 1.0), 0.08, Sd, Td);
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
  expl_solver.setBoundaryCondition(leftBoundary, rightBoundary);
  // set initial condition:
  expl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set2OrderCoefficient(1.0);
  // set convection term
  expl_solver.set1OrderCoefficient(-1.0);
  // get the solution:
  expl_solver.solve(solution);
  // get exact solution:
  auto exact = [](T x, T t, std::size_t n) {
    T const zero = 2.0 * (1.0 - std::exp(-0.5)) / (1.0 - std::exp(-1.0));
    T const first = 2.0;
    T const exp_0p5x = std::exp(0.5 * x);
    T exp_lamt{};
    T sum{};
    T num{}, den{}, var{};
    T lambda_n{};
    T delta_n{};

    for (std::size_t i = 1; i <= n; ++i) {
      delta_n = i * PI;
      lambda_n = 0.25 + delta_n * delta_n;
      exp_lamt = std::exp(-1.0 * lambda_n * t);
      num = (1.0 - std::pow(-1.0, i)) * exp_0p5x * exp_lamt *
            (std::sin(delta_n * x) - 2.0 * delta_n * std::cos(delta_n * x));
      den = delta_n * (1.0 + 4.0 * delta_n * delta_n);
      var = num / den;
      sum += var;
    }
    return (zero + first * sum);
  };

  T const h = expl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.08, 30);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testExplAdvDiffEquationRobinBC() {
  std::cout << "============================================================\n";
  std::cout << "=== Explicit Advection Diffusion Equation (Robin BC) =======\n";
  std::cout << "============================================================\n";

  testExplAdvDiffEquationRobinBCEuler<double>();
  testExplAdvDiffEquationRobinBCEuler<float>();

  std::cout << "============================================================\n";
}

#endif  ///_LSS_ONE_DIM_ADVECTION_DIFFUSION_EQUATION_T
