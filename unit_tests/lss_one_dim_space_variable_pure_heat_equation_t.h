#pragma once
#if !defined(_LSS_ONE_DIM_SPACE_VARIABLE_PURE_HEAT_EQUATION_T)
#define _LSS_ONE_DIM_SPACE_VARIABLE_PURE_HEAT_EQUATION_T

#pragma warning(disable : 4305)

#include "common/lss_enumerations.h"
#include "common/lss_utility.h"
#include "pde_solvers/one_dim/lss_pde_boundary.h"
#include "pde_solvers/one_dim/variable_coefficients/lss_space_variable_general_heat_equation_solvers.h"
#include "sparse_solvers/lss_fdm_double_sweep_solver.h"
#include "sparse_solvers/lss_fdm_thomas_lu_solver.h"

#define PI 3.14159

namespace pure_heat_equation {

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
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_double_sweep_solver::fdm_double_sweep_solver;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::implicit_solvers::
      general_heat_equation;
  using lss_utility::range;

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

  // typedef the general_heat_equation
  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                fdm_double_sweep_solver, std::vector,
                                std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initial_condition = [](T x) { return x; };
  // boundary conditions:
  auto const &dirichlet = [](T t) { return 0.0; };
  auto boundary =
      std::make_shared<dirichlet_boundary_1d<T>>(dirichlet, dirichlet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(range<T>(0.0, 1.0), 0.10, Sd, Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection term in PDE
  impl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero-order term in PDE
  impl_solver.set_0_order_coefficient([](T x) { return 0.0; });
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::Euler);
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

  T const h = impl_solver.space_step();
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
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_double_sweep_solver::fdm_double_sweep_solver;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::implicit_solvers::
      general_heat_equation;
  using lss_utility::range;

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

  // typedef the general_heat_equation
  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                fdm_double_sweep_solver, std::vector,
                                std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initial_condition = [](T x) { return x; };
  // boundary conditions:
  auto const &dirichlet = [](T t) { return 0.0; };
  auto boundary =
      std::make_shared<dirichlet_boundary_1d<T>>(dirichlet, dirichlet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(range<T>(0.0, 1.0), 0.10, Sd, Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection term in PDE
  impl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero-order term in PDE
  impl_solver.set_0_order_coefficient([](T x) { return 0.0; });
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

  T const h = impl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.10, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

template <typename T>
void testImplPureHeatEquationDirichletBCThomasLUEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_thomas_lu_solver::fdm_thomas_lu_solver;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::implicit_solvers::
      general_heat_equation;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using Thomas LU algorithm with implicit Euler method\n\n ";
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
  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                fdm_thomas_lu_solver, std::vector,
                                std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initial_condition = [](T x) { return x; };
  // boundary conditions:
  auto const &dirichlet = [](T t) { return 0.0; };
  auto boundary =
      std::make_shared<dirichlet_boundary_1d<T>>(dirichlet, dirichlet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(range<T>(0.0, 1.0), 0.10, Sd, Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection term in PDE
  impl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero-order term in PDE
  impl_solver.set_0_order_coefficient([](T x) { return 0.0; });
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::Euler);
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

  T const h = impl_solver.space_step();
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
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_thomas_lu_solver::fdm_thomas_lu_solver;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::implicit_solvers::
      general_heat_equation;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using Thomas LU algorithm with \n ";
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
  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                fdm_thomas_lu_solver, std::vector,
                                std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 1000;
  // number of time subdivisions:
  std::size_t const Td = 1000;
  // initial condition:
  auto initial_condition = [](T x) { return x; };
  // boundary conditions:
  auto const &dirichlet = [](T t) { return 0.0; };
  auto boundary =
      std::make_shared<dirichlet_boundary_1d<T>>(dirichlet, dirichlet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(range<T>(0.0, 1.0), 0.10, Sd, Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection term in PDE
  impl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero-order term in PDE
  impl_solver.set_0_order_coefficient([](T x) { return 0.0; });
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

  T const h = impl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.10, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

template <typename T>
void testImplPureHeatEquationRobinBCDoubleSweepEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_double_sweep_solver::fdm_double_sweep_solver;
  using lss_one_dim_pde_utility::robin_boundary;
  using lss_one_dim_space_variable_pde_solvers::implicit_solvers::
      general_heat_equation;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
  std::cout << " Value type:" << typeid(T).name() << "\n\n";
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

  // typedef the general_heat_equation
  typedef general_heat_equation<T, boundary_condition_enum::Robin,
                                fdm_double_sweep_solver, std::vector,
                                std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 400;
  // number of time subdivisions:
  std::size_t const Td = 150;
  // initial condition:
  auto initial_condition = [](T x) { return x; };
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(range<T>(0.0, 1.0), 0.1, Sd, Td);
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
  auto left_boundary = std::make_pair(1.0, 0.0);
  auto right_boundary = std::make_pair(1.0, 0.0);
  impl_solver.set_boundary_condition(
      robin_boundary<T>(left_boundary, right_boundary));
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity  in PDE
  impl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection diffusivity in PDE
  impl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero-order term  in PDE
  impl_solver.set_0_order_coefficient([](T x) { return 0.0; });
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::Euler);
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

  T const h = impl_solver.space_step();
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
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_double_sweep_solver::fdm_double_sweep_solver;
  using lss_one_dim_pde_utility::robin_boundary;
  using lss_one_dim_space_variable_pde_solvers::implicit_solvers::
      general_heat_equation;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Cranc-Nicolson \n";
  std::cout << " method\n\n";
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

  // typedef the general_heat_equation
  typedef general_heat_equation<T, boundary_condition_enum::Robin,
                                fdm_double_sweep_solver, std::vector,
                                std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 1000;
  // number of time subdivisions:
  std::size_t const Td = 50;
  // initial condition:
  auto initial_condition = [](T x) { return x; };

  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(range<T>(0.0, 1.0), 0.1, Sd, Td);
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
  auto left_boundary = std::make_pair(1.0, 0.0);
  auto right_boundary = std::make_pair(1.0, 0.0);
  impl_solver.set_boundary_condition(
      robin_boundary<T>(left_boundary, right_boundary));
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity  in PDE
  impl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection diffusivity in PDE
  impl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero-order term  in PDE
  impl_solver.set_0_order_coefficient([](T x) { return 0.0; });
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

  T const h = impl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

template <typename T>
void testImplPureHeatEquationRobinBCThomasLUEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_thomas_lu_solver::fdm_thomas_lu_solver;
  using lss_one_dim_pde_utility::robin_boundary;
  using lss_one_dim_space_variable_pde_solvers::implicit_solvers::
      general_heat_equation;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using Thomas LU algorithm with implicit Euler method\n\n ";
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

  // typedef the general_heat_equation
  typedef general_heat_equation<T, boundary_condition_enum::Robin,
                                fdm_thomas_lu_solver, std::vector,
                                std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initial_condition = [](T x) { return x; };
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(range<T>(0.0, 1.0), 0.1, Sd, Td);
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
  auto left_boundary = std::make_pair(1.0, 0.0);
  auto right_boundary = std::make_pair(1.0, 0.0);
  impl_solver.set_boundary_condition(
      robin_boundary<T>(left_boundary, right_boundary));
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity  in PDE
  impl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection term in PDE
  impl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero-order term in PDE
  impl_solver.set_0_order_coefficient([](T x) { return 0.0; });
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::Euler);
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

  T const h = impl_solver.space_step();
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
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_thomas_lu_solver::fdm_thomas_lu_solver;
  using lss_one_dim_pde_utility::robin_boundary;
  using lss_one_dim_space_variable_pde_solvers::implicit_solvers::
      general_heat_equation;
  using lss_utility::range;

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

  // typedef the general_heat_equation
  typedef general_heat_equation<T, boundary_condition_enum::Robin,
                                fdm_thomas_lu_solver, std::vector,
                                std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 1000;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initial_condition = [](T x) { return x; };
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(range<T>(0.0, 1.0), 0.1, Sd, Td);
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
  auto left_boundary = std::make_pair(1.0, 0.0);
  auto right_boundary = std::make_pair(1.0, 0.0);
  impl_solver.set_boundary_condition(
      robin_boundary<T>(left_boundary, right_boundary));
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity  in PDE
  impl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection term in PDE
  impl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero-order term in PDE
  impl_solver.set_0_order_coefficient([](T x) { return 0.0; });
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

  T const h = impl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

//
//// ===========================================================================
//// ======= Heat problem with homogeneous boundary conditions and source ======
//// ===========================================================================
//
template <typename T>
void testImplPureHeatEquationSourceDirichletBCDoubleSweepEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_double_sweep_solver::fdm_double_sweep_solver;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::implicit_solvers::
      general_heat_equation;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with source: \n\n";
  std::cout << "Using Double Sweep algorithm with implicit Euler method\n\n";
  std::cout << "Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = 1, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the general_heat_equation
  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                fdm_double_sweep_solver, std::vector,
                                std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initial_condition = [](T x) { return 1.0; };
  // boundary conditions:
  auto const &dirichlet = [](T t) { return 0.0; };
  auto boundary =
      std::make_shared<dirichlet_boundary_1d<T>>(dirichlet, dirichlet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(range<T>(0.0, 1.0), 0.10, Sd, Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection term in PDE
  impl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero-order term in PDE
  impl_solver.set_0_order_coefficient([](T x) { return 0.0; });
  // set heat source:
  impl_solver.set_heat_source([](T x, T t) { return x; });
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::Euler);
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

  T const h = impl_solver.space_step();
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
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_double_sweep_solver::fdm_double_sweep_solver;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::implicit_solvers::
      general_heat_equation;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Crank-Nicolson \n";
  std::cout << " method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = 1, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the general_heat_equation
  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                fdm_double_sweep_solver, std::vector,
                                std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initial_condition = [](T x) { return 1.0; };
  // boundary conditions:
  auto const &dirichlet = [](T t) { return 0.0; };
  auto boundary =
      std::make_shared<dirichlet_boundary_1d<T>>(dirichlet, dirichlet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(range<T>(0.0, 1.0), 0.10, Sd, Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection term in PDE
  impl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero-order term in PDE
  impl_solver.set_0_order_coefficient([](T x) { return 0.0; });
  // set heat source:
  impl_solver.set_heat_source([](T x, T t) { return x; });
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

  T const h = impl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.10, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

template <typename T>
void testImplPureHeatEquationSourceDirichletBCThomasLUEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_thomas_lu_solver::fdm_thomas_lu_solver;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::implicit_solvers::
      general_heat_equation;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using Thomas LU algorithm with implicit Euler method\n\n ";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x,\n\n ";
  std::cout << " where\n\n ";
  std::cout << " x in<0, 1> and t > 0,\n ";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = 1, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the general_heat_equation
  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                fdm_thomas_lu_solver, std::vector,
                                std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initial_condition = [](T x) { return 1.0; };
  // boundary conditions:
  auto const &dirichlet = [](T t) { return 0.0; };
  auto boundary =
      std::make_shared<dirichlet_boundary_1d<T>>(dirichlet, dirichlet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(range<T>(0.0, 1.0), 0.10, Sd, Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection term in PDE
  impl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero -order term in PDE
  impl_solver.set_0_order_coefficient([](T x) { return 0.0; });
  // set heat source:
  impl_solver.set_heat_source([](T x, T t) { return x; });
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::Euler);
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

  T const h = impl_solver.space_step();
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
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_thomas_lu_solver::fdm_thomas_lu_solver;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::implicit_solvers::
      general_heat_equation;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with source: \n\n";
  std::cout << "Using Thomas LU algorithm with \n ";
  std::cout << "implicit Crank-Nicolson method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = 1, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the general_heat_equation
  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                fdm_thomas_lu_solver, std::vector,
                                std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 1000;
  // number of time subdivisions:
  std::size_t const Td = 1000;
  // initial condition:
  auto initial_condition = [](T x) { return 1.0; };
  // boundary conditions:
  auto const &dirichlet = [](T t) { return 0.0; };
  auto boundary =
      std::make_shared<dirichlet_boundary_1d<T>>(dirichlet, dirichlet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(range<T>(0.0, 1.0), 0.10, Sd, Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection term in PDE
  impl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero -order term in PDE
  impl_solver.set_0_order_coefficient([](T x) { return 0.0; });
  // set heat source:
  impl_solver.set_heat_source([](T x, T t) { return x; });
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

  T const h = impl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.10, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

//
//// ===========================================================================
//// ==== Heat problem with homogeneous Robin boundary conditions and source ===
//// ===========================================================================
//
template <typename T>
void testImplPureHeatEquationSourceRobinBCDoubleSweepEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_double_sweep_solver::fdm_double_sweep_solver;
  using lss_one_dim_pde_utility::robin_boundary;
  using lss_one_dim_space_variable_pde_solvers::implicit_solvers::
      general_heat_equation;
  using lss_utility::range;

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

  // typedef the general_heat_equation
  typedef general_heat_equation<T, boundary_condition_enum::Robin,
                                fdm_double_sweep_solver, std::vector,
                                std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initial_condition = [](T x) { return x; };
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(range<T>(0.0, 1.0), 0.1, Sd, Td);
  // boundary conditions:
  auto left_boundary = std::make_pair(1.0, 0.0);
  auto right_boundary = std::make_pair(1.0, 0.0);
  // set boundary conditions:
  impl_solver.set_boundary_condition(
      robin_boundary<T>(left_boundary, right_boundary));
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection term in PDE
  impl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero-order term in PDE
  impl_solver.set_0_order_coefficient([](T x) { return 0.0; });
  // set heat source:
  impl_solver.set_heat_source([](T x, T t) { return x; });
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::Euler);
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

  T const h = impl_solver.space_step();
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
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_double_sweep_solver::fdm_double_sweep_solver;
  using lss_one_dim_pde_utility::robin_boundary;
  using lss_one_dim_space_variable_pde_solvers::implicit_solvers::
      general_heat_equation;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with Robin boundaries \n";
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

  // typedef the general_heat_equation
  typedef general_heat_equation<T, boundary_condition_enum::Robin,
                                fdm_double_sweep_solver, std::vector,
                                std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initial_condition = [](T x) { return x; };
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(range<T>(0.0, 1.0), 0.1, Sd, Td);
  // boundary conditions:
  auto left_boundary = std::make_pair(1.0, 0.0);
  auto right_boundary = std::make_pair(1.0, 0.0);
  // set boundary conditions:
  impl_solver.set_boundary_condition(
      robin_boundary<T>(left_boundary, right_boundary));
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection term in PDE
  impl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero-order term in PDE
  impl_solver.set_0_order_coefficient([](T x) { return 0.0; });
  // set heat source:
  impl_solver.set_heat_source([](T x, T t) { return x; });
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

  T const h = impl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

template <typename T>
void testImplPureHeatEquationSourceRobinBCThomasLUEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_thomas_lu_solver::fdm_thomas_lu_solver;
  using lss_one_dim_pde_utility::robin_boundary;
  using lss_one_dim_space_variable_pde_solvers::implicit_solvers::
      general_heat_equation;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with Robin boundaries\n";
  std::cout << " and source: \n\n";
  std::cout << " Using Thomas LU algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the general_heat_equation
  typedef general_heat_equation<T, boundary_condition_enum::Robin,
                                fdm_thomas_lu_solver, std::vector,
                                std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initial_condition = [](T x) { return x; };
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(range<T>(0.0, 1.0), 0.1, Sd, Td);
  // boundary conditions:
  auto left_boundary = std::make_pair(1.0, 0.0);
  auto right_boundary = std::make_pair(1.0, 0.0);
  // set boundary conditions:
  impl_solver.set_boundary_condition(
      robin_boundary<T>(left_boundary, right_boundary));
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity in PDE
  impl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection trem in PDE
  impl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero-order term in PDE
  impl_solver.set_0_order_coefficient([](T x) { return 0.0; });
  // set heat source:
  impl_solver.set_heat_source([](T x, T t) { return x; });
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::Euler);
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

  T const h = impl_solver.space_step();
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
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_thomas_lu_solver::fdm_thomas_lu_solver;
  using lss_one_dim_pde_utility::robin_boundary;
  using lss_one_dim_space_variable_pde_solvers::implicit_solvers::
      general_heat_equation;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << " Solving Boundary-value Heat equation with Robin boundaries \n";
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
  // typedef the general_heat_equation
  typedef general_heat_equation<T, boundary_condition_enum::Robin,
                                fdm_thomas_lu_solver, std::vector,
                                std::allocator<T>>
      implicit_solver;
  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initial_condition = [](T x) { return x; };
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(range<T>(0.0, 1.0), 0.1, Sd, Td);
  // boundary conditions:
  auto left_boundary = std::make_pair(1.0, 0.0);
  auto right_boundary = std::make_pair(1.0, 0.0);
  // set boundary conditions:
  impl_solver.set_boundary_condition(
      robin_boundary<T>(left_boundary, right_boundary));
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity in PDE
  impl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection trem in PDE
  impl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero-order term in PDE
  impl_solver.set_0_order_coefficient([](T x) { return 0.0; });
  // set heat source:
  impl_solver.set_heat_source([](T x, T t) { return x; });
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

  T const h = impl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

//
//// ===========================================================================
//// ============ Heat problem with nonhomogeneous boundary conditions =========
//// ===========================================================================
//
template <typename T>
void testImplNonHomPureHeatEquationDirichletBCDoubleSweepEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_double_sweep_solver::fdm_double_sweep_solver;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::implicit_solvers::
      general_heat_equation;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with Non-hom. BC: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t> 0,\n";
  std::cout << " U(0,t) = 0, U(1,t) = 100, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = 100*x + (198/pi)*sum_0^infty{ "
               "(-1)^(n)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the general_heat_equation
  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                fdm_double_sweep_solver, std::vector,
                                std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initial_condition = [](T x) { return x; };
  // boundary conditions:
  auto const &dirichlet_left = [](T t) { return 0.0; };
  auto const &dirichlet_right = [](T t) { return 100.0; };
  auto boundary = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_left,
                                                             dirichlet_right);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(range<T>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection term in PDE
  impl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero-order term in PDE
  impl_solver.set_0_order_coefficient([](T x) { return 0.0; });
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::Euler);
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

  T const h = impl_solver.space_step();
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
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_double_sweep_solver::fdm_double_sweep_solver;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::implicit_solvers::
      general_heat_equation;
  using lss_utility::range;

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

  // typedef the general_heat_equation
  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                fdm_double_sweep_solver, std::vector,
                                std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initial_condition = [](T x) { return x; };
  // boundary conditions:
  auto const &dirichlet_left = [](T t) { return 0.0; };
  auto const &dirichlet_right = [](T t) { return 100.0; };
  auto boundary = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_left,
                                                             dirichlet_right);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(range<T>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection term in PDE
  impl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero-order term in PDE
  impl_solver.set_0_order_coefficient([](T x) { return 0.0; });
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

  T const h = impl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

template <typename T>
void testImplNonHomPureHeatEquationDirichletBCThomasLUEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_thomas_lu_solver::fdm_thomas_lu_solver;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::implicit_solvers::
      general_heat_equation;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with non-hom BC: \n\n";
  std::cout << " Using Thomas LU algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t> 0,\n";
  std::cout << " U(0,t) = U(0,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = 100.0*x + (198/pi)*sum_0^infty{ \n"
               "(-1)^(n)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the general_heat_equation
  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                fdm_thomas_lu_solver, std::vector,
                                std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initial_condition = [](T x) { return x; };
  // boundary conditions:
  auto const &dirichlet_left = [](T t) { return 0.0; };
  auto const &dirichlet_right = [](T t) { return 100.0; };
  auto boundary = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_left,
                                                             dirichlet_right);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(range<T>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection term in PDE
  impl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero-order term in PDE
  impl_solver.set_0_order_coefficient([](T x) { return 0.0; });
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::Euler);
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

  T const h = impl_solver.space_step();
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
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_thomas_lu_solver::fdm_thomas_lu_solver;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::implicit_solvers::
      general_heat_equation;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with non-hom BC: \n\n";
  std::cout << " Using Thomas LU algorithm with \n"
               "implicit Crank-Nicolson method\n\n";
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

  // typedef the general_heat_equation
  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                fdm_thomas_lu_solver, std::vector,
                                std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initial_condition = [](T x) { return x; };
  // boundary conditions:
  auto const &dirichlet_left = [](T t) { return 0.0; };
  auto const &dirichlet_right = [](T t) { return 100.0; };
  auto boundary = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_left,
                                                             dirichlet_right);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(range<T>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection term in PDE
  impl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero-order term in PDE
  impl_solver.set_0_order_coefficient([](T x) { return 0.0; });
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

  T const h = impl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

//
//============================================================================
//====================== EPLICIT SOLVERS =====================================
//============================================================================

//============================================================================
//=============== Heat problem with homogeneous boundary conditions ==========
//============================================================================

template <typename T>
void testExplPureHeatEquationDirichletBCEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::explicit_pde_schemes_enum;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::explicit_solvers::
      general_heat_equation;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using explicit Euler method\n\n ";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << "where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) =U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) \n"
               "*sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the general_heat_equation
  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                std::vector, std::allocator<T>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 10000;
  // initial condition:
  auto initial_condition = [](T x) { return x; };
  // boundary conditions:
  auto const &dirichlet = [](T t) { return 0.0; };
  auto boundary =
      std::make_shared<dirichlet_boundary_1d<T>>(dirichlet, dirichlet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  explicit_solver expl_solver(range<T>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  expl_solver.set_boundary_condition(boundary);
  // set initial condition:
  expl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection term in PDE
  expl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero-order term term in PDE
  expl_solver.set_0_order_coefficient([](T x) { return 0.0; });
  // get the solution:
  expl_solver.solve(solution, explicit_pde_schemes_enum::Euler);
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

  T const h = expl_solver.space_step();
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
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::explicit_pde_schemes_enum;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::explicit_solvers::
      general_heat_equation;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using explicit ADE Barakat Clark method\n\n";
  std::cout << " Value type : " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) "
               "*sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the general_heat_equation
  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                std::vector, std::allocator<T>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 10000;
  // initial condition:
  auto initial_condition = [](T x) { return x; };
  // boundary conditions:
  auto const &dirichlet = [](T t) { return 0.0; };
  auto boundary =
      std::make_shared<dirichlet_boundary_1d<T>>(dirichlet, dirichlet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  explicit_solver expl_solver(range<T>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  expl_solver.set_boundary_condition(boundary);
  // set initial condition:
  expl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection term in PDE
  expl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero-order term term in PDE
  expl_solver.set_0_order_coefficient([](T x) { return 0.0; });
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

  T const h = expl_solver.space_step();
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
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::explicit_pde_schemes_enum;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::explicit_solvers::
      general_heat_equation;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using explicit ADE Saulyev method\n\n ";
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

  // typedef the general_heat_equation
  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                std::vector, std::allocator<T>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 10000;
  // initial condition:
  auto initial_condition = [](T x) { return x; };
  // boundary conditions:
  auto const &dirichlet = [](T t) { return 0.0; };
  auto boundary =
      std::make_shared<dirichlet_boundary_1d<T>>(dirichlet, dirichlet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  explicit_solver expl_solver(range<T>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  expl_solver.set_boundary_condition(boundary);
  // set initial condition:
  expl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection term in PDE
  expl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero-order term term in PDE
  expl_solver.set_0_order_coefficient([](T x) { return 0.0; });
  // get the solution:
  expl_solver.solve(solution, explicit_pde_schemes_enum::ADESaulyev);
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

  T const h = expl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

///============================================================================
///====== Heat problem with homogeneous boundary conditions and source ========
///============================================================================

template <typename T>
void testExplPureHeatEquationSourceDirichletBCEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::explicit_pde_schemes_enum;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::explicit_solvers::
      general_heat_equation;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with source: \n\n";
  std::cout << " Using explicit Euler method\n\n ";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = 1, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the general_heat_equation
  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                std::vector, std::allocator<T>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 10000;
  // initial condition:
  auto initial_condition = [](T x) { return 1.0; };
  // boundary conditions:
  auto const &dirichlet = [](T t) { return 0.0; };
  auto boundary =
      std::make_shared<dirichlet_boundary_1d<T>>(dirichlet, dirichlet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  explicit_solver expl_solver(range<T>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  expl_solver.set_boundary_condition(boundary);
  // set initial condition:
  expl_solver.set_initial_condition(initial_condition);
  // set heat source:
  expl_solver.set_heat_source([](T x, T t) { return x; });
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection term in PDE
  expl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero-order term term in PDE
  expl_solver.set_0_order_coefficient([](T x) { return 0.0; });
  // get the solution:
  expl_solver.solve(solution, explicit_pde_schemes_enum::Euler);
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

  T const h = expl_solver.space_step();
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
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::explicit_pde_schemes_enum;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::explicit_solvers::
      general_heat_equation;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with source: \n\n";
  std::cout << "Using explicit ADE Barakat Clark method\n\n ";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x,\n\n ";
  std::cout << " where\n\n ";
  std::cout << " x in<0, 1> and t > 0,\n ";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = 1, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the general_heat_equation
  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                std::vector, std::allocator<T>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 10000;
  // initial condition:
  auto initial_condition = [](T x) { return 1.0; };
  // boundary conditions:
  auto const &dirichlet = [](T t) { return 0.0; };
  auto boundary =
      std::make_shared<dirichlet_boundary_1d<T>>(dirichlet, dirichlet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  explicit_solver expl_solver(range<T>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  expl_solver.set_boundary_condition(boundary);
  // set initial condition:
  expl_solver.set_initial_condition(initial_condition);
  // set heat source:
  expl_solver.set_heat_source([](T x, T t) { return x; });
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection term in PDE
  expl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero-order term term in PDE
  expl_solver.set_0_order_coefficient([](T x) { return 0.0; });
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

  T const h = expl_solver.space_step();
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
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::explicit_pde_schemes_enum;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::explicit_solvers::
      general_heat_equation;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with source: \n\n";
  std::cout << " Using explicit ADE Saulyev method\n\n";
  std::cout << " Value type : " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = 1.0, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the general_heat_equation
  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                std::vector, std::allocator<T>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 10000;
  // initial condition:
  auto initial_condition = [](T x) { return 1.0; };
  // boundary conditions:
  auto const &dirichlet = [](T t) { return 0.0; };
  auto boundary =
      std::make_shared<dirichlet_boundary_1d<T>>(dirichlet, dirichlet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  explicit_solver expl_solver(range<T>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  expl_solver.set_boundary_condition(boundary);
  // set initial condition:
  expl_solver.set_initial_condition(initial_condition);
  // set heat source:
  expl_solver.set_heat_source([](T x, T t) { return x; });
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection term in PDE
  expl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero-order term term in PDE
  expl_solver.set_0_order_coefficient([](T x) { return 0.0; });
  // get the solution:
  expl_solver.solve(solution, explicit_pde_schemes_enum::ADESaulyev);
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

  T const h = expl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

///============================================================================
///========== Heat problem with nonhomogeneous boundary conditions ============
///============================================================================
//
template <typename T>
void testExplNonHomPureHeatEquationDirichletBCEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::explicit_pde_schemes_enum;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::explicit_solvers::
      general_heat_equation;
  using lss_utility::range;

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

  // typedef the general_heat_equation
  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                std::vector, std::allocator<T>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 10000;
  // initial condition:
  auto initial_condition = [](T x) { return x; };
  // boundary conditions:
  auto const &dirichlet_left = [](T t) { return 0.0; };
  auto const &dirichlet_right = [](T t) { return 100.0; };
  auto boundary = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_left,
                                                             dirichlet_right);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  explicit_solver expl_solver(range<T>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  expl_solver.set_boundary_condition(boundary);
  // set initial condition:
  expl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection term in PDE
  expl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero-order term term in PDE
  expl_solver.set_0_order_coefficient([](T x) { return 0.0; });
  // get the solution:
  expl_solver.solve(solution, explicit_pde_schemes_enum::Euler);
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

  T const h = expl_solver.space_step();
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
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::explicit_pde_schemes_enum;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::explicit_solvers::
      general_heat_equation;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with non-hom BC: \n\n";
  std::cout << " Using explicit ADE Barakat Clark method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t),\n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = 0, U(1,t) = 100, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = 100*x +  (198/pi)*sum_0^infty{ "
               "(-1)^(n+1)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the general_heat_equation
  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                std::vector, std::allocator<T>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 10000;
  // initial condition:
  auto initial_condition = [](T x) { return x; };
  // boundary conditions:
  auto const &dirichlet_left = [](T t) { return 0.0; };
  auto const &dirichlet_right = [](T t) { return 100.0; };
  auto boundary = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_left,
                                                             dirichlet_right);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  explicit_solver expl_solver(range<T>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  expl_solver.set_boundary_condition(boundary);
  // set initial condition:
  expl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection term in PDE
  expl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero-order term term in PDE
  expl_solver.set_0_order_coefficient([](T x) { return 0.0; });
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

  T const h = expl_solver.space_step();
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
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::explicit_pde_schemes_enum;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::explicit_solvers::
      general_heat_equation;
  using lss_utility::range;

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

  // typedef the general_heat_equation
  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                std::vector, std::allocator<T>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 10000;
  // initial condition:
  auto initial_condition = [](T x) { return x; };
  // boundary conditions:
  auto const &dirichlet_left = [](T t) { return 0.0; };
  auto const &dirichlet_right = [](T t) { return 100.0; };
  auto boundary = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet_left,
                                                             dirichlet_right);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  explicit_solver expl_solver(range<T>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  expl_solver.set_boundary_condition(boundary);
  // set initial condition:
  expl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection term in PDE
  expl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero-order term term in PDE
  expl_solver.set_0_order_coefficient([](T x) { return 0.0; });
  // get the solution:
  expl_solver.solve(solution, explicit_pde_schemes_enum::ADESaulyev);
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

  T const h = expl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}
//
////
///============================================================================
///=========== Heat problem with homogeneous Robin boundary conditions ========
///============================================================================
//
template <typename T>
void testExplHomPureHeatEquationRobinBCEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_one_dim_pde_utility::robin_boundary;
  using lss_one_dim_space_variable_pde_solvers::explicit_solvers::
      general_heat_equation;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using explicit Euler method\n\n ";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << "where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout
      << " U(x,t) = 0.5 - (4/(pi*pi))*sum_1^infty{ exp(-((2n-1)*pi)^2*t) \n"
         "*cos((2n-1)*pi*x)/(2n-1)^2}\n\n";
  std::cout << "============================================================\n";

  // typedef the general_heat_equation
  typedef general_heat_equation<T, boundary_condition_enum::Robin, std::vector,
                                std::allocator<T>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 5000;
  // initial condition:
  auto initial_condition = [](T x) { return x; };
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, 0.0);
  // initialize solver
  explicit_solver expl_solver(range<T>(0.0, 1.0), 0.1, Sd, Td);
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
  auto const h = expl_solver.space_step();
  auto left_boundary = std::make_pair(1.0, 0.0);
  auto right_boundary = std::make_pair(1.0, 0.0);
  // set boundary conditions:
  expl_solver.set_boundary_condition(
      robin_boundary<T>(left_boundary, right_boundary));
  // set initial condition:
  expl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection term in PDE
  expl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero-order term term in PDE
  expl_solver.set_0_order_coefficient([](T x) { return 0.0; });
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

//
////
///============================================================================
///=== Heat problem with homogeneous Robin boundary conditions and source =====
///============================================================================
//
template <typename T>
void testExplHomPureHeatEquationSourceRobinBCEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_one_dim_pde_utility::robin_boundary;
  using lss_one_dim_space_variable_pde_solvers::explicit_solvers::
      general_heat_equation;
  using lss_utility::range;

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

  // typedef the general_heat_equation
  typedef general_heat_equation<T, boundary_condition_enum::Robin, std::vector,
                                std::allocator<T>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 5000;
  // initial condition:
  auto initial_condition = [](T x) { return x; };
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, 0.0);
  // initialize solver
  explicit_solver expl_solver(range<T>(0.0, 1.0), 0.1, Sd, Td);
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
  auto const h = expl_solver.space_step();
  auto left_boundary = std::make_pair(1.0, 0.0);
  auto right_boundary = std::make_pair(1.0, 0.0);
  // set boundary conditions:
  expl_solver.set_boundary_condition(
      robin_boundary<T>(left_boundary, right_boundary));
  // set initial condition:
  expl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection term in PDE
  expl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero-order term term in PDE
  expl_solver.set_0_order_coefficient([](T x) { return 0.0; });
  // set heat source:
  expl_solver.set_heat_source([](T x, T t) { return x; });
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

}  // namespace pure_heat_equation

namespace reaction_diffusion_equation {

template <typename T>
void testImplHeatEquationDirichletBCDoubleSweepEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_double_sweep_solver::fdm_double_sweep_solver;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::implicit_solvers::
      general_heat_equation;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) - x*x*U(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <-1,1> and t > 0,\n";
  std::cout << " U(-1,t) = U(1,t) = exp(-0.5 - t), t > 0 \n\n";
  std::cout << " U(x,0) = exp(-0.5*x*x), x in <-1,1> \n\n";
  std::cout << " U(x,t) = exp(-0.5*x*x - t), x in <-1,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the general_heat_equation
  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                fdm_double_sweep_solver, std::vector,
                                std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 200;
  // number of time subdivisions:
  std::size_t const Td = 200;
  // initial condition:
  auto initial_condition = [](T x) { return exp(-0.5 * x * x); };
  // boundary conditions:
  auto const &dirichet = [](T t) { return exp(-0.5 - t); };
  auto boundary =
      std::make_shared<dirichlet_boundary_1d<T>>(dirichet, dirichet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(range<T>(-1.0, 1.0), 0.20, Sd, Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection term in PDE
  impl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero-order term in PDE
  impl_solver.set_0_order_coefficient([](T x) { return -1.0 * x * x; });
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::Euler);
  // get exact solution:
  auto exact = [](T x, T t) { return (exp(-0.5 * x * x - t)); };

  T const h = impl_solver.space_step();
  T const start = -1.0;
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(start + j * h, 0.2);
    std::cout << "t_" << j << "(" << (start + j * h) << ") : " << solution[j]
              << " |  " << benchmark << " | " << (solution[j] - benchmark)
              << '\n';
  }
}

template <typename T>
void testImplHeatEquationDirichletBCDoubleSweepCN() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_double_sweep_solver::fdm_double_sweep_solver;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::implicit_solvers::
      general_heat_equation;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Crank-Nicolson\n"
               "method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) - x*x*U(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <-1,1> and t > 0,\n";
  std::cout << " U(-1,t) = U(1,t) = exp(-0.5 - t), t > 0 \n\n";
  std::cout << " U(x,0) = exp(-0.5*x*x), x in <-1,1> \n\n";
  std::cout << " U(x,t) = exp(-0.5*x*x - t), x in <-1,1> \n\n";
  std::cout << "============================================================\n";
  // typedef the general_heat_equation
  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                fdm_double_sweep_solver, std::vector,
                                std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 200;
  // number of time subdivisions:
  std::size_t const Td = 200;
  // initial condition:
  auto initial_condition = [](T x) { return exp(-0.5 * x * x); };
  // boundary conditions:
  auto const &dirichet = [](T t) { return exp(-0.5 - t); };
  auto boundary =
      std::make_shared<dirichlet_boundary_1d<T>>(dirichet, dirichet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(range<T>(-1.0, 1.0), 0.20, Sd, Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection term in PDE
  impl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero-order term in PDE
  impl_solver.set_0_order_coefficient([](T x) { return -1.0 * x * x; });
  // get the solution:
  impl_solver.solve(solution);
  // get exact solution:
  auto exact = [](T x, T t) { return (exp(-0.5 * x * x - t)); };

  T const h = impl_solver.space_step();
  T const start = -1.0;
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(start + j * h, 0.2);
    std::cout << "t_" << j << "(" << (start + j * h) << ") : " << solution[j]
              << " |  " << benchmark << " | " << (solution[j] - benchmark)
              << '\n';
  }
}
template <typename T>
void testImplHeatEquationDirichletBCThomasLUEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_thomas_lu_solver::fdm_thomas_lu_solver;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::implicit_solvers::
      general_heat_equation;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using Thomas LU algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) - x*x*U(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <-1,1> and t > 0,\n";
  std::cout << " U(-1,t) = U(1,t) = exp(-0.5 - t), t > 0 \n\n";
  std::cout << " U(x,0) = exp(-0.5*x*x), x in <-1,1> \n\n";
  std::cout << " U(x,t) = exp(-0.5*x*x - t), x in <-1,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the general_heat_equation
  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                fdm_thomas_lu_solver, std::vector,
                                std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 200;
  // number of time subdivisions:
  std::size_t const Td = 200;
  // initial condition:
  auto initial_condition = [](T x) { return exp(-0.5 * x * x); };
  // boundary conditions:
  auto const &dirichet = [](T t) { return exp(-0.5 - t); };
  auto boundary =
      std::make_shared<dirichlet_boundary_1d<T>>(dirichet, dirichet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(range<T>(-1.0, 1.0), 0.20, Sd, Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection term in PDE
  impl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero-order term in PDE
  impl_solver.set_0_order_coefficient([](T x) { return -1.0 * x * x; });
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::Euler);
  // get exact solution:
  auto exact = [](T x, T t) { return (exp(-0.5 * x * x - t)); };

  T const h = impl_solver.space_step();
  T const start = -1.0;
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(start + j * h, 0.2);
    std::cout << "t_" << j << "(" << (start + j * h) << ") : " << solution[j]
              << " |  " << benchmark << " | " << (solution[j] - benchmark)
              << '\n';
  }
}

template <typename T>
void testImplHeatEquationDirichletBCThomalLUCN() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_thomas_lu_solver::fdm_thomas_lu_solver;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::implicit_solvers::
      general_heat_equation;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using Thomas LU algorithm with implicit Crank-Nicolson\n"
               "method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) - x*x*U(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <-1,1> and t > 0,\n";
  std::cout << " U(-1,t) = U(1,t) = exp(-0.5 - t), t > 0 \n\n";
  std::cout << " U(x,0) = exp(-0.5*x*x), x in <-1,1> \n\n";
  std::cout << " U(x,t) = exp(-0.5*x*x - t), x in <-1,1> \n\n";
  std::cout << "============================================================\n";
  // typedef the general_heat_equation
  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                fdm_thomas_lu_solver, std::vector,
                                std::allocator<T>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 200;
  // number of time subdivisions:
  std::size_t const Td = 200;
  // initial condition:
  auto initial_condition = [](T x) { return exp(-0.5 * x * x); };
  // boundary conditions:
  auto const &dirichet = [](T t) { return exp(-0.5 - t); };
  auto boundary =
      std::make_shared<dirichlet_boundary_1d<T>>(dirichet, dirichet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(range<T>(-1.0, 1.0), 0.20, Sd, Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection term in PDE
  impl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero-order term in PDE
  impl_solver.set_0_order_coefficient([](T x) { return -1.0 * x * x; });
  // get the solution:
  impl_solver.solve(solution);
  // get exact solution:
  auto exact = [](T x, T t) { return (exp(-0.5 * x * x - t)); };

  T const h = impl_solver.space_step();
  T const start = -1.0;
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(start + j * h, 0.2);
    std::cout << "t_" << j << "(" << (start + j * h) << ") : " << solution[j]
              << " |  " << benchmark << " | " << (solution[j] - benchmark)
              << '\n';
  }
}

template <typename T>
void testExplHeatEquationDirichletBCADEBarakatClark() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::explicit_pde_schemes_enum;
  using lss_fdm_double_sweep_solver::fdm_double_sweep_solver;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::explicit_solvers::
      general_heat_equation;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using ADE Barakat-Clark  method\n\n";
  std::cout << " Value type:" << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) - x*x*U(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <-1,1> and t > 0,\n";
  std::cout << " U(-1,t) = U(1,t) = exp(-0.5 - t), t > 0 \n\n";
  std::cout << " U(x,0) = exp(-0.5*x*x), x in <-1,1> \n\n";
  std::cout << " U(x,t) = exp(-0.5*x*x - t), x in <-1,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the general_heat_equation
  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                std::vector, std::allocator<T>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 10000;
  // initial condition:
  auto initial_condition = [](T x) { return exp(-0.5 * x * x); };
  // boundary conditions:
  auto const &dirichet = [](T t) { return exp(-0.5 - t); };
  auto boundary =
      std::make_shared<dirichlet_boundary_1d<T>>(dirichet, dirichet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  explicit_solver expl_solver(range<T>(-1.0, 1.0), 0.20, Sd, Td);
  // set boundary conditions:
  expl_solver.set_boundary_condition(boundary);
  // set initial condition:
  expl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection term in PDE
  expl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero-order term in PDE
  expl_solver.set_0_order_coefficient([](T x) { return -1.0 * x * x; });
  // get the solution:
  expl_solver.solve(solution, explicit_pde_schemes_enum::ADEBarakatClark);
  // get exact solution:
  auto exact = [](T x, T t) { return (exp(-0.5 * x * x - t)); };

  T const h = expl_solver.space_step();
  T const start = -1.0;
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(start + j * h, 0.2);
    std::cout << "t_" << j << "(" << (start + j * h) << ") : " << solution[j]
              << " |  " << benchmark << " | " << (solution[j] - benchmark)
              << '\n';
  }
}

template <typename T>
void testExplHeatEquationDirichletBCADESaulyev() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::explicit_pde_schemes_enum;
  using lss_fdm_double_sweep_solver::fdm_double_sweep_solver;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::explicit_solvers::
      general_heat_equation;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using ADE Saulyev  method\n\n";
  std::cout << " Value type:" << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) - x*x*U(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <-1,1> and t > 0,\n";
  std::cout << " U(-1,t) = U(1,t) = exp(-0.5 - t), t > 0 \n\n";
  std::cout << " U(x,0) = exp(-0.5*x*x), x in <-1,1> \n\n";
  std::cout << " U(x,t) = exp(-0.5*x*x - t), x in <-1,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the general_heat_equation
  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                std::vector, std::allocator<T>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 10000;
  // initial condition:
  auto initial_condition = [](T x) { return exp(-0.5 * x * x); };
  // boundary conditions:
  auto const &dirichet = [](T t) { return exp(-0.5 - t); };
  auto boundary =
      std::make_shared<dirichlet_boundary_1d<T>>(dirichet, dirichet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  explicit_solver expl_solver(range<T>(-1.0, 1.0), 0.20, Sd, Td);
  // set boundary conditions:
  expl_solver.set_boundary_condition(boundary);
  // set initial condition:
  expl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection term in PDE
  expl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero-order term in PDE
  expl_solver.set_0_order_coefficient([](T x) { return -1.0 * x * x; });
  // get the solution:
  expl_solver.solve(solution, explicit_pde_schemes_enum::ADESaulyev);
  // get exact solution:
  auto exact = [](T x, T t) { return (exp(-0.5 * x * x - t)); };

  T const h = expl_solver.space_step();
  T const start = -1.0;
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(start + j * h, 0.2);
    std::cout << "t_" << j << "(" << (start + j * h) << ") : " << solution[j]
              << " |  " << benchmark << " | " << (solution[j] - benchmark)
              << '\n';
  }
}

template <typename T>
void testExplHeatEquationDirichletBCADEEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::explicit_pde_schemes_enum;
  using lss_fdm_double_sweep_solver::fdm_double_sweep_solver;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::explicit_solvers::
      general_heat_equation;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using ADE Euler method\n\n";
  std::cout << " Value type:" << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) - x*x*U(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <-1,1> and t > 0,\n";
  std::cout << " U(-1,t) = U(1,t) = exp(-0.5 - t), t > 0 \n\n";
  std::cout << " U(x,0) = exp(-0.5*x*x), x in <-1,1> \n\n";
  std::cout << " U(x,t) = exp(-0.5*x*x - t), x in <-1,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the general_heat_equation
  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                std::vector, std::allocator<T>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 10000;
  // initial condition:
  auto initial_condition = [](T x) { return exp(-0.5 * x * x); };
  // boundary conditions:
  auto const &dirichet = [](T t) { return exp(-0.5 - t); };
  auto boundary =
      std::make_shared<dirichlet_boundary_1d<T>>(dirichet, dirichet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  explicit_solver expl_solver(range<T>(-1.0, 1.0), 0.20, Sd, Td);
  // set boundary conditions:
  expl_solver.set_boundary_condition(boundary);
  // set initial condition:
  expl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set_2_order_coefficient([](T x) { return 1.0; });
  // set convection term in PDE
  expl_solver.set_1_order_coefficient([](T x) { return 0.0; });
  // set zero-order term in PDE
  expl_solver.set_0_order_coefficient([](T x) { return -1.0 * x * x; });
  // get the solution:
  expl_solver.solve(solution, explicit_pde_schemes_enum::Euler);
  // get exact solution:
  auto exact = [](T x, T t) { return (exp(-0.5 * x * x - t)); };

  T const h = expl_solver.space_step();
  T const start = -1.0;
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(start + j * h, 0.2);
    std::cout << "t_" << j << "(" << (start + j * h) << ") : " << solution[j]
              << " |  " << benchmark << " | " << (solution[j] - benchmark)
              << '\n';
  }
}

}  // namespace reaction_diffusion_equation

void testImplSpaceVarPureHeatEquationDirichletBCDoubleSweep() {
  std::cout << "============================================================\n";
  std::cout << "========= Implicit Pure Heat Equation (Dirichlet BC) =======\n";
  std::cout << "============================================================\n";

  pure_heat_equation::testImplPureHeatEquationDirichletBCDoubleSweepEuler<
      double>();
  pure_heat_equation::testImplPureHeatEquationDirichletBCDoubleSweepEuler<
      float>();
  pure_heat_equation::testImplPureHeatEquationDirichletBCDoubleSweepCN<
      double>();
  pure_heat_equation::testImplPureHeatEquationDirichletBCDoubleSweepCN<float>();

  std::cout << "============================================================\n";
}

void testImplSpaceVarPureHeatEquationDirichletBCThomasLU() {
  std::cout << "============================================================\n";
  std::cout << "======== Implicit Pure Heat Equation (Dirichlet BC) ========\n";
  std::cout << "============================================================\n";

  pure_heat_equation::testImplPureHeatEquationDirichletBCThomasLUEuler<
      double>();
  pure_heat_equation::testImplPureHeatEquationDirichletBCThomasLUEuler<float>();
  pure_heat_equation::testImplPureHeatEquationDirichletBCThomasLUCN<double>();
  pure_heat_equation::testImplPureHeatEquationDirichletBCThomasLUCN<float>();

  std::cout << "============================================================\n";
}

void testImplSpaceVarPureHeatEquationRobinBCDoubleSweep() {
  std::cout << "============================================================\n";
  std::cout << "======= Implicit Pure Heat Equation (Robin BC) =============\n";
  std::cout << "============================================================\n";

  pure_heat_equation::testImplPureHeatEquationRobinBCDoubleSweepEuler<double>();
  pure_heat_equation::testImplPureHeatEquationRobinBCDoubleSweepEuler<float>();
  pure_heat_equation::testImplPureHeatEquationRobinBCDoubleSweepCN<double>();
  pure_heat_equation::testImplPureHeatEquationRobinBCDoubleSweepCN<float>();

  std::cout << "============================================================\n";
}

void testImplSpaceVarPureHeatEquationRobinBCThomasLU() {
  std::cout << "============================================================\n";
  std::cout << "========== Implicit Pure Heat Equation (Robin BC) ==========\n";
  std::cout << "============================================================\n";

  pure_heat_equation::testImplPureHeatEquationRobinBCThomasLUEuler<double>();
  pure_heat_equation::testImplPureHeatEquationRobinBCThomasLUEuler<float>();
  pure_heat_equation::testImplPureHeatEquationRobinBCThomasLUCN<double>();
  pure_heat_equation::testImplPureHeatEquationRobinBCThomasLUCN<float>();

  std::cout << "============================================================\n";
}

void testImplSpaceVarPureHeatEquationSourceDirichletBCDoubleSweep() {
  std::cout << "============================================================\n";
  std::cout << "== Implicit Pure Heat Equation with source (Dirichlet BC) ==\n";
  std::cout << "============================================================\n";

  pure_heat_equation::testImplPureHeatEquationSourceDirichletBCDoubleSweepEuler<
      double>();
  pure_heat_equation::testImplPureHeatEquationSourceDirichletBCDoubleSweepEuler<
      float>();
  pure_heat_equation::testImplPureHeatEquationSourceDirichletBCDoubleSweepCN<
      double>();
  pure_heat_equation::testImplPureHeatEquationSourceDirichletBCDoubleSweepCN<
      float>();

  std::cout << "============================================================\n";
}

void testImplSpaceVarPureHeatEquationSourceDirichletBCThomasLU() {
  std::cout << "============================================================\n";
  std::cout << "=== Implicit Pure Heat Equation with source (Dirichlet BC) =\n";
  std::cout << "============================================================\n";

  pure_heat_equation::testImplPureHeatEquationSourceDirichletBCThomasLUEuler<
      double>();
  pure_heat_equation::testImplPureHeatEquationSourceDirichletBCThomasLUEuler<
      float>();
  pure_heat_equation::testImplPureHeatEquationSourceDirichletBCThomasLUCN<
      double>();
  pure_heat_equation::testImplPureHeatEquationSourceDirichletBCThomasLUCN<
      float>();

  std::cout << "============================================================\n";
}

void testImplSpaceVarPureHeatEquationSourceRobinBCDoubleSweep() {
  std::cout << "============================================================\n";
  std::cout << "=== Implicit Pure Heat Equation with source (Robin BC) =====\n";
  std::cout << "============================================================\n";

  pure_heat_equation::testImplPureHeatEquationSourceRobinBCDoubleSweepEuler<
      double>();
  pure_heat_equation::testImplPureHeatEquationSourceRobinBCDoubleSweepEuler<
      float>();
  pure_heat_equation::testImplPureHeatEquationSourceRobinBCDoubleSweepCN<
      double>();
  pure_heat_equation::testImplPureHeatEquationSourceRobinBCDoubleSweepCN<
      float>();

  std::cout << "============================================================\n";
}

void testImplSpaceVarPureHeatEquationSourceRobinBCThomasLU() {
  std::cout << "============================================================\n";
  std::cout << "==== Implicit Pure Heat Equation with source (Robin BC) ====\n";
  std::cout << "============================================================\n";

  pure_heat_equation::testImplPureHeatEquationSourceRobinBCThomasLUEuler<
      double>();
  pure_heat_equation::testImplPureHeatEquationSourceRobinBCThomasLUEuler<
      float>();
  pure_heat_equation::testImplPureHeatEquationSourceRobinBCThomasLUCN<double>();
  pure_heat_equation::testImplPureHeatEquationSourceRobinBCThomasLUCN<float>();

  std::cout << "============================================================\n";
}
//

//
void testImplSpaceVarNonHomPureHeatEquationDirichletBCDoubleSweep() {
  std::cout << "============================================================\n";
  std::cout << "= Implicit Pure Heat Equation (non-homogenous Dirichlet BC) \n";
  std::cout << "============================================================\n";

  pure_heat_equation::testImplNonHomPureHeatEquationDirichletBCDoubleSweepEuler<
      double>();
  pure_heat_equation::testImplNonHomPureHeatEquationDirichletBCDoubleSweepEuler<
      float>();
  pure_heat_equation::testImplNonHomPureHeatEquationDirichletBCDoubleSweepCN<
      double>();
  pure_heat_equation::testImplNonHomPureHeatEquationDirichletBCDoubleSweepCN<
      float>();

  std::cout << "============================================================\n";
}
//

void testImplSpaceVarNonHomPureHeatEquationDirichletBCThomasLU() {
  std::cout << "============================================================\n";
  std::cout << "===== Implicit Pure Heat Equation (with non-homogeneous \n"
               "Dirichlet BC) ==========\n";
  std::cout << "============================================================\n";

  pure_heat_equation::testImplNonHomPureHeatEquationDirichletBCThomasLUEuler<
      double>();
  pure_heat_equation::testImplNonHomPureHeatEquationDirichletBCThomasLUEuler<
      float>();
  pure_heat_equation::testImplNonHomPureHeatEquationDirichletBCThomasLUCN<
      double>();
  pure_heat_equation::testImplNonHomPureHeatEquationDirichletBCThomasLUCN<
      float>();

  std::cout << "============================================================\n";
}

void testExplSpaceVarPureHeatEquationDirichletBC() {
  std::cout << "============================================================\n";
  std::cout << "====== Explicit Pure Heat Equation (Dirichlet BC) ==========\n";
  std::cout << "============================================================\n";

  pure_heat_equation::testExplPureHeatEquationDirichletBCEuler<double>();
  pure_heat_equation::testExplPureHeatEquationDirichletBCEuler<float>();
  pure_heat_equation::testExplPureHeatEquationDirichletBCADEBC<double>();
  pure_heat_equation::testExplPureHeatEquationDirichletBCADEBC<float>();
  pure_heat_equation::testExplPureHeatEquationDirichletBCADES<double>();
  pure_heat_equation::testExplPureHeatEquationDirichletBCADES<float>();

  std::cout << "============================================================\n";
}
//

//
void testExplSpaceVarPureHeatEquationSourceDirichletBC() {
  std::cout << "============================================================\n";
  std::cout << "== Explicit Pure Heat Equation with source (Dirichlet BC) ==\n";
  std::cout << "============================================================\n";

  pure_heat_equation::testExplPureHeatEquationSourceDirichletBCEuler<double>();
  pure_heat_equation::testExplPureHeatEquationSourceDirichletBCEuler<float>();
  pure_heat_equation::testExplPureHeatEquationSourceDirichletBCADEBC<double>();
  pure_heat_equation::testExplPureHeatEquationSourceDirichletBCADEBC<float>();
  pure_heat_equation::testExplPureHeatEquationSourceDirichletBCADES<double>();
  pure_heat_equation::testExplPureHeatEquationSourceDirichletBCADES<float>();

  std::cout << "============================================================\n";
}
//

//
void testExplSpaceVarNonHomPureHeatEquationDirichletBC() {
  std::cout << "============================================================\n";
  std::cout << "========= Explicit Pure Heat Equation (with non-homogeneous \n"
               "Dirichlet BC) ======\n";
  std::cout << "============================================================\n";

  pure_heat_equation::testExplNonHomPureHeatEquationDirichletBCEuler<double>();
  pure_heat_equation::testExplNonHomPureHeatEquationDirichletBCEuler<float>();
  pure_heat_equation::testExplNonHomPureHeatEquationDirichletBCADEBC<double>();
  pure_heat_equation::testExplNonHomPureHeatEquationDirichletBCADEBC<float>();
  pure_heat_equation::testExplNonHomPureHeatEquationDirichletBCADES<double>();
  pure_heat_equation::testExplNonHomPureHeatEquationDirichletBCADES<float>();

  std::cout << "============================================================\n";
}

void testExplSpaceVarHomPureHeatEquationRobinBC() {
  std::cout << "============================================================\n";
  std::cout << "======= Explicit Pure Heat Equation (with homogeneous Robin \n"
               "BC) ==============\n";
  std::cout << "============================================================\n";

  testExplHomPureHeatEquationRobinBCEuler<double>();
  testExplHomPureHeatEquationRobinBCEuler<float>();

  std::cout << "============================================================\n";
}

void testExplSpaceVarHomPureHeatEquationSourceRobinBC() {
  std::cout << "============================================================\n";
  std::cout << "====== Explicit Pure Heat Equation with Source (with \n"
               "homogeneous Robin BC) =====\n";
  std::cout << "============================================================\n";

  testExplHomPureHeatEquationSourceRobinBCEuler<double>();
  testExplHomPureHeatEquationSourceRobinBCEuler<float>();

  std::cout << "============================================================\n";
}

void testImplSpaceVarHeatEquationDirichletBCDoubleSweep() {
  std::cout << "============================================================\n";
  std::cout << "============ Implicit Heat Equation (Dirichlet BC) =========\n";
  std::cout << "============================================================\n";
  reaction_diffusion_equation::testImplHeatEquationDirichletBCDoubleSweepEuler<
      float>();
  reaction_diffusion_equation::testImplHeatEquationDirichletBCDoubleSweepCN<
      float>();

  reaction_diffusion_equation::testImplHeatEquationDirichletBCDoubleSweepEuler<
      double>();
  reaction_diffusion_equation::testImplHeatEquationDirichletBCDoubleSweepCN<
      double>();

  std::cout << "============================================================\n";
}

void testImplSpaceVarHeatEquationDirichletBCThomasLU() {
  std::cout << "============================================================\n";
  std::cout << "============ Implicit Heat Equation (Dirichlet BC) =========\n";
  std::cout << "============================================================\n";
  reaction_diffusion_equation::testImplHeatEquationDirichletBCThomasLUEuler<
      float>();
  reaction_diffusion_equation::testImplHeatEquationDirichletBCThomalLUCN<
      float>();

  reaction_diffusion_equation::testImplHeatEquationDirichletBCThomasLUEuler<
      double>();
  reaction_diffusion_equation::testImplHeatEquationDirichletBCThomalLUCN<
      double>();

  std::cout << "============================================================\n";
}

void testExplSpaceVarHeatEquationDirichletBCADE() {
  std::cout << "============================================================\n";
  std::cout << "============ Explicit Heat Equation (Dirichlet BC) =========\n";
  std::cout << "============================================================\n";
  reaction_diffusion_equation::testExplHeatEquationDirichletBCADEEuler<float>();
  reaction_diffusion_equation::testExplHeatEquationDirichletBCADESaulyev<
      float>();
  reaction_diffusion_equation::testExplHeatEquationDirichletBCADEBarakatClark<
      float>();

  reaction_diffusion_equation::testExplHeatEquationDirichletBCADEEuler<
      double>();
  reaction_diffusion_equation::testExplHeatEquationDirichletBCADESaulyev<
      double>();
  reaction_diffusion_equation::testExplHeatEquationDirichletBCADEBarakatClark<
      double>();

  std::cout << "============================================================\n";
}

#endif  ///_LSS_ONE_DIM_SPACE_VARIABLE_PURE_HEAT_EQUATION_T
