#pragma once
#if !defined(_LSS_ONE_DIM_BLACK_SCHOLES_EQUATION_T)
#define _LSS_ONE_DIM_BLACK_SCHOLES_EQUATION_T

#pragma warning(disable : 4305)

#include "common/lss_enumerations.h"
#include "common/lss_utility.h"
#include "pde_solvers/one_dim/lss_pde_boundary.h"
#include "pde_solvers/one_dim/variable_coefficients/lss_black_scholes_equation_solvers.h"
#include "sparse_solvers/lss_fdm_double_sweep_solver.h"
#include "sparse_solvers/lss_fdm_thomas_lu_solver.h"

// ///////////////////////////////////////////////////////////////////////////
//							BLACK-SCHOLES PROBLEMS
// ///////////////////////////////////////////////////////////////////////////

// ===========================================================================
// ========================== IMPLICIT SOLVERS ===============================
// ===========================================================================

template <typename T>
void testImplEuropeanBlackScholesCallOptionBCDoubleSweepEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_double_sweep_solver::fdm_double_sweep_solver;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::implicit_solvers::
      black_scholes_equation;
  using lss_utility::black_scholes_exact;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Black-Scholes equation: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = 0.5*sig*sig*x*x*U_xx(x,t) + r*x*U_x(x,t) - "
               "r*U(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " 0 < x < 20 and 0 < t < 1,\n";
  std::cout << " U(0,t) = 0 and  U(20,t) = 20-K*exp(-r*(1-t)),0 < t < 1 \n\n";
  std::cout << " U(x,T) = max(0,x-K), x in <0,20> \n\n";
  std::cout << "============================================================\n";

  // typedef the black_scholes_equation
  typedef black_scholes_equation<T, boundary_condition_enum::Dirichlet,
                                 fdm_double_sweep_solver, std::vector,
                                 std::allocator<T>>
      implicit_solver;
  // parameters of the call option:
  auto const &strike = 10;
  auto const &maturity = 1.0;
  auto const &rate = 0.2;
  auto const &sig = 0.25;
  // number of space subdivisions:
  std::size_t const Sd = 160;
  // number of time subdivisions:
  std::size_t const Td = 160;
  // initial condition:
  auto terminal_condition = [&](T x) { return std::max<T>(0.0, x - strike); };
  // boundary conditions:
  auto const &dirichletLeft = [](T t) { return 0.0; };
  auto const &dirichletRight = [&](T t) {
    return (20.0 - strike * std::exp(-rate * (maturity - t)));
  };
  auto boundary =
      std::make_shared<dirichlet_boundary_1d<T>>(dirichletLeft, dirichletRight);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(range<T>(0.0, 20.0), maturity, Sd, Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_terminal_condition(terminal_condition);
  // set second order coefficient:
  impl_solver.set_2_order_coefficient(
      [&](T x) { return (0.5 * x * x * sig * sig); });
  // set first order coefficient:
  impl_solver.set_1_order_coefficient([&](T x) { return (rate * x); });
  // set zero order coefficient:
  impl_solver.set_0_order_coefficient([&](T x) { return (-1.0 * rate); });
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::Euler);
  // get exact solution:
  black_scholes_exact<T> bs_exact(0.0, strike, rate, sig, maturity);

  T const h = impl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = bs_exact.call(j * h);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

template <typename T>
void testImplEuropeanBlackScholesCallOptionBCDoubleSweepCN() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_double_sweep_solver::fdm_double_sweep_solver;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::implicit_solvers::
      black_scholes_equation;
  using lss_utility::black_scholes_exact;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Black-Scholes equation: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Crank-Nicolson \n"
               "method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = 0.5*sig*sig*x*x*U_xx(x,t) + r*x*U_x(x,t) - "
               "r*U(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " 0 < x < 20 and 0 < t < 1,\n";
  std::cout << " U(0,t) = 0 and  U(20,t) = 20-K*exp(-r*(1-t)),0 < t < 1 \n\n";
  std::cout << " U(x,T) = max(0,x-K), x in <0,20> \n\n";
  std::cout << "============================================================\n";

  // typedef the black_scholes_equation
  typedef black_scholes_equation<T, boundary_condition_enum::Dirichlet,
                                 fdm_double_sweep_solver, std::vector,
                                 std::allocator<T>>
      implicit_solver;

  // parameters of the call option:
  auto const &strike = 10;
  auto const &maturity = 1.0;
  auto const &rate = 0.2;
  auto const &sig = 0.25;
  // number of space subdivisions:
  std::size_t const Sd = 160;
  // number of time subdivisions:
  std::size_t const Td = 160;
  // initial condition:
  auto terminal_condition = [&](T x) { return std::max<T>(0.0, x - strike); };
  // boundary conditions:
  auto const &dirichletLeft = [](T t) { return 0.0; };
  auto const &dirichletRight = [&](T t) {
    return (20.0 - strike * std::exp(-rate * (maturity - t)));
  };
  auto boundary =
      std::make_shared<dirichlet_boundary_1d<T>>(dirichletLeft, dirichletRight);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(range<T>(0.0, 20.0), maturity, Sd, Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_terminal_condition(terminal_condition);
  // set second order coefficient:
  impl_solver.set_2_order_coefficient(
      [&](T x) { return (0.5 * x * x * sig * sig); });
  // set first order coefficient:
  impl_solver.set_1_order_coefficient([&](T x) { return (rate * x); });
  // set zero order coefficient:
  impl_solver.set_0_order_coefficient([&](T x) { return (-1.0 * rate); });
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::CrankNicolson);
  // get exact solution:
  black_scholes_exact<T> bs_exact(0.0, strike, rate, sig, maturity);

  T const h = impl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = bs_exact.call(j * h);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplEuropeanBlackScholesCallDirichletBCDoubleSweep() {
  std::cout << "============================================================\n";
  std::cout << "======== Implicit Balck-Scholes Equation (Dirichlet BC) ====\n";
  std::cout << "============================================================\n";

  testImplEuropeanBlackScholesCallOptionBCDoubleSweepEuler<double>();
  testImplEuropeanBlackScholesCallOptionBCDoubleSweepEuler<float>();
  testImplEuropeanBlackScholesCallOptionBCDoubleSweepCN<double>();
  testImplEuropeanBlackScholesCallOptionBCDoubleSweepCN<float>();

  std::cout << "============================================================\n";
}

template <typename T>
void testImplEuropeanBlackScholesCallOptionBCThomasLUEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_thomas_lu_solver::fdm_thomas_lu_solver;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::implicit_solvers::
      black_scholes_equation;
  using lss_utility::black_scholes_exact;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Black-Scholes equation: \n\n";
  std::cout << " Using Thomas LU algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = 0.5*sig*sig*x*x*U_xx(x,t) + r*x*U_x(x,t) - "
               "r*U(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " 0 < x < 20 and 0 < t < 1,\n";
  std::cout << " U(0,t) = 0 and  U(20,t) = 20-K*exp(-r*(1-t)),0 < t < 1 \n\n";
  std::cout << " U(x,T) = max(0,x-K), x in <0,20> \n\n";
  std::cout << "============================================================\n";

  // typedef the black_scholes_equation
  typedef black_scholes_equation<T, boundary_condition_enum::Dirichlet,
                                 fdm_thomas_lu_solver, std::vector,
                                 std::allocator<T>>
      implicit_solver;

  // parameters of the call option:
  auto const &strike = 10;
  auto const &maturity = 1.0;
  auto const &rate = 0.2;
  auto const &sig = 0.25;
  // number of space subdivisions:
  std::size_t const Sd = 160;
  // number of time subdivisions:
  std::size_t const Td = 160;
  // initial condition:
  auto terminal_condition = [&](T x) { return std::max<T>(0.0, x - strike); };
  // boundary conditions:
  auto const &dirichletLeft = [](T t) { return 0.0; };
  auto const &dirichletRight = [&](T t) {
    return (20.0 - strike * std::exp(-rate * (maturity - t)));
  };
  auto boundary =
      std::make_shared<dirichlet_boundary_1d<T>>(dirichletLeft, dirichletRight);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(range<T>(0.0, 20.0), maturity, Sd, Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_terminal_condition(terminal_condition);
  // set second order coefficient:
  impl_solver.set_2_order_coefficient(
      [&](T x) { return (0.5 * x * x * sig * sig); });
  // set first order coefficient:
  impl_solver.set_1_order_coefficient([&](T x) { return (rate * x); });
  // set zero order coefficient:
  impl_solver.set_0_order_coefficient([&](T x) { return (-1.0 * rate); });
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::Euler);
  // get exact solution:
  black_scholes_exact<T> bs_exact(0.0, strike, rate, sig, maturity);

  T const h = impl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = bs_exact.call(j * h);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

template <typename T>
void testImplEuropeanBlackScholesCallOptionBCThomasLUCN() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_thomas_lu_solver::fdm_thomas_lu_solver;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::implicit_solvers::
      black_scholes_equation;
  using lss_utility::black_scholes_exact;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Black-Scholes equation: \n\n";
  std::cout << " Using Thomas LU algorithm with Crank-Nicolson method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = 0.5*sig*sig*x*x*U_xx(x,t) + r*x*U_x(x,t) - "
               "r*U(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " 0 < x < 20 and 0 < t < 1,\n";
  std::cout << " U(0,t) = 0 and  U(20,t) = 20-K*exp(-r*(1-t)),0 < t < 1 \n\n";
  std::cout << " U(x,T) = max(0,x-K), x in <0,20> \n\n";
  std::cout << "============================================================\n";

  // typedef the black_scholes_equation
  typedef black_scholes_equation<T, boundary_condition_enum::Dirichlet,
                                 fdm_thomas_lu_solver, std::vector,
                                 std::allocator<T>>
      implicit_solver;

  // parameters of the call option:
  auto const &strike = 10;
  auto const &maturity = 1.0;
  auto const &rate = 0.2;
  auto const &sig = 0.25;
  // number of space subdivisions:
  std::size_t const Sd = 160;
  // number of time subdivisions:
  std::size_t const Td = 160;
  // initial condition:
  auto terminal_condition = [&](T x) { return std::max<T>(0.0, x - strike); };
  // boundary conditions:
  auto const &dirichletLeft = [](T t) { return 0.0; };
  auto const &dirichletRight = [&](T t) {
    return (20.0 - strike * std::exp(-rate * (maturity - t)));
  };
  auto boundary =
      std::make_shared<dirichlet_boundary_1d<T>>(dirichletLeft, dirichletRight);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(range<T>(0.0, 20.0), maturity, Sd, Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_terminal_condition(terminal_condition);
  // set second order coefficient:
  impl_solver.set_2_order_coefficient(
      [&](T x) { return (0.5 * x * x * sig * sig); });
  // set first order coefficient:
  impl_solver.set_1_order_coefficient([&](T x) { return (rate * x); });
  // set zero order coefficient:
  impl_solver.set_0_order_coefficient([&](T x) { return (-1.0 * rate); });
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::Euler);
  // get exact solution:
  black_scholes_exact<T> bs_exact(0.0, strike, rate, sig, maturity);

  T const h = impl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = bs_exact.call(j * h);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplEuropeanBlackScholesCallDirichletBCThomasLU() {
  std::cout << "============================================================\n";
  std::cout << "======= Implicit Black-Scholes Equation (Dirichlet BC) =====\n";
  std::cout << "============================================================\n";

  testImplEuropeanBlackScholesCallOptionBCThomasLUEuler<double>();
  testImplEuropeanBlackScholesCallOptionBCThomasLUEuler<float>();
  testImplEuropeanBlackScholesCallOptionBCThomasLUCN<double>();
  testImplEuropeanBlackScholesCallOptionBCThomasLUCN<float>();

  std::cout << "============================================================\n";
}

template <typename T>
void testImplEuropeanBlackScholesPutOptionBCDoubleSweepEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_double_sweep_solver::fdm_double_sweep_solver;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::implicit_solvers::
      black_scholes_equation;
  using lss_utility::black_scholes_exact;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Black-Scholes equation: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = 0.5*sig*sig*x*x*U_xx(x,t) + r*x*U_x(x,t) - "
               "r*U(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " 0 < x < 20 and 0 < t < 1,\n";
  std::cout << " U(0,t) =  K*exp(-r*(1-t)) and  U(20,t) = 0 ,0 < t < 1 \n\n";
  std::cout << " U(x,T) = max(0,K-x), x in <0,20> \n\n";
  std::cout << "============================================================\n";

  // typedef the black_scholes_equation
  typedef black_scholes_equation<T, boundary_condition_enum::Dirichlet,
                                 fdm_double_sweep_solver, std::vector,
                                 std::allocator<T>>
      implicit_solver;
  // parameters of the call option:
  auto const &strike = 10;
  auto const &maturity = 1.0;
  auto const &rate = 0.2;
  auto const &sig = 0.25;
  // number of space subdivisions:
  std::size_t const Sd = 160;
  // number of time subdivisions:
  std::size_t const Td = 160;
  // initial condition:
  auto terminal_condition = [&](T x) { return std::max<T>(0.0, strike - x); };
  // boundary conditions:
  auto const &dirichletRight = [](T t) { return 0.0; };
  auto const &dirichletLeft = [&](T t) {
    return (strike * std::exp(-rate * (maturity - t)));
  };
  auto boundary =
      std::make_shared<dirichlet_boundary_1d<T>>(dirichletLeft, dirichletRight);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(range<T>(0.0, 20.0), maturity, Sd, Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_terminal_condition(terminal_condition);
  // set second order coefficient:
  impl_solver.set_2_order_coefficient(
      [&](T x) { return (0.5 * x * x * sig * sig); });
  // set first order coefficient:
  impl_solver.set_1_order_coefficient([&](T x) { return (rate * x); });
  // set zero order coefficient:
  impl_solver.set_0_order_coefficient([&](T x) { return (-1.0 * rate); });
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::Euler);
  // get exact solution:
  black_scholes_exact<T> bs_exact(0.0, strike, rate, sig, maturity);

  T const h = impl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = bs_exact.put(j * h);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

template <typename T>
void testImplEuropeanBlackScholesPutOptionBCDoubleSweepCN() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_double_sweep_solver::fdm_double_sweep_solver;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::implicit_solvers::
      black_scholes_equation;
  using lss_utility::black_scholes_exact;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Black-Scholes equation: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Crank-Nicolson \n"
               "method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = 0.5*sig*sig*x*x*U_xx(x,t) + r*x*U_x(x,t) - "
               "r*U(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " 0 < x < 20 and 0 < t < 1,\n";
  std::cout << " U(0,t) = K*exp(-r*(1-t)) and  U(20,t) = 0,0 < t < 1 \n\n";
  std::cout << " U(x,T) = max(0,K - x), x in <0,20> \n\n";
  std::cout << "============================================================\n";

  // typedef the black_scholes_equation
  typedef black_scholes_equation<T, boundary_condition_enum::Dirichlet,
                                 fdm_double_sweep_solver, std::vector,
                                 std::allocator<T>>
      implicit_solver;

  // parameters of the call option:
  auto const &strike = 10;
  auto const &maturity = 1.0;
  auto const &rate = 0.2;
  auto const &sig = 0.25;
  // number of space subdivisions:
  std::size_t const Sd = 160;
  // number of time subdivisions:
  std::size_t const Td = 160;
  // initial condition:
  auto terminal_condition = [&](T x) { return std::max<T>(0.0, strike - x); };
  // boundary conditions:
  auto const &dirichletRight = [](T t) { return 0.0; };
  auto const &dirichletLeft = [&](T t) {
    return (strike * std::exp(-rate * (maturity - t)));
  };
  auto boundary =
      std::make_shared<dirichlet_boundary_1d<T>>(dirichletLeft, dirichletRight);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(range<T>(0.0, 20.0), maturity, Sd, Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_terminal_condition(terminal_condition);
  // set second order coefficient:
  impl_solver.set_2_order_coefficient(
      [&](T x) { return (0.5 * x * x * sig * sig); });
  // set first order coefficient:
  impl_solver.set_1_order_coefficient([&](T x) { return (rate * x); });
  // set zero order coefficient:
  impl_solver.set_0_order_coefficient([&](T x) { return (-1.0 * rate); });
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::CrankNicolson);
  // get exact solution:
  black_scholes_exact<T> bs_exact(0.0, strike, rate, sig, maturity);

  T const h = impl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = bs_exact.put(j * h);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplEuropeanBlackScholesPutDirichletBCDoubleSweep() {
  std::cout << "============================================================\n";
  std::cout << "======== Implicit Balck-Scholes Equation (Dirichlet BC) ====\n";
  std::cout << "============================================================\n";

  testImplEuropeanBlackScholesPutOptionBCDoubleSweepEuler<double>();
  testImplEuropeanBlackScholesPutOptionBCDoubleSweepEuler<float>();
  testImplEuropeanBlackScholesPutOptionBCDoubleSweepCN<double>();
  testImplEuropeanBlackScholesPutOptionBCDoubleSweepCN<float>();

  std::cout << "============================================================\n";
}

template <typename T>
void testImplEuropeanBlackScholesPutOptionBCThomasLUEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_thomas_lu_solver::fdm_thomas_lu_solver;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::implicit_solvers::
      black_scholes_equation;
  using lss_utility::black_scholes_exact;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Black-Scholes equation: \n\n";
  std::cout << " Using Thomas LU algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = 0.5*sig*sig*x*x*U_xx(x,t) + r*x*U_x(x,t) - "
               "r*U(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " 0 < x < 20 and 0 < t < 1,\n";
  std::cout << " U(0,t) = K*exp(-r*(1-t)) and  U(20,t) = 0,0 < t < 1 \n\n";
  std::cout << " U(x,T) = max(0,K - x), x in <0,20> \n\n";
  std::cout << "============================================================\n";

  // typedef the black_scholes_equation
  typedef black_scholes_equation<T, boundary_condition_enum::Dirichlet,
                                 fdm_thomas_lu_solver, std::vector,
                                 std::allocator<T>>
      implicit_solver;

  // parameters of the call option:
  auto const &strike = 10;
  auto const &maturity = 1.0;
  auto const &rate = 0.2;
  auto const &sig = 0.25;
  // number of space subdivisions:
  std::size_t const Sd = 160;
  // number of time subdivisions:
  std::size_t const Td = 160;
  // initial condition:
  auto terminal_condition = [&](T x) { return std::max<T>(0.0, strike - x); };
  // boundary conditions:
  auto const &dirichletRight = [](T t) { return 0.0; };
  auto const &dirichletLeft = [&](T t) {
    return (strike * std::exp(-rate * (maturity - t)));
  };
  auto boundary =
      std::make_shared<dirichlet_boundary_1d<T>>(dirichletLeft, dirichletRight);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(range<T>(0.0, 20.0), maturity, Sd, Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_terminal_condition(terminal_condition);
  // set second order coefficient:
  impl_solver.set_2_order_coefficient(
      [&](T x) { return (0.5 * x * x * sig * sig); });
  // set first order coefficient:
  impl_solver.set_1_order_coefficient([&](T x) { return (rate * x); });
  // set zero order coefficient:
  impl_solver.set_0_order_coefficient([&](T x) { return (-1.0 * rate); });
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::Euler);
  // get exact solution:
  black_scholes_exact<T> bs_exact(0.0, strike, rate, sig, maturity);

  T const h = impl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = bs_exact.put(j * h);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

template <typename T>
void testImplEuropeanBlackScholesPutOptionBCThomasLUCN() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_thomas_lu_solver::fdm_thomas_lu_solver;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::implicit_solvers::
      black_scholes_equation;
  using lss_utility::black_scholes_exact;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Black-Scholes equation: \n\n";
  std::cout << " Using Thomas LU algorithm with Crank-Nicolson method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = 0.5*sig*sig*x*x*U_xx(x,t) + r*x*U_x(x,t) - "
               "r*U(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " 0 < x < 20 and 0 < t < 1,\n";
  std::cout << " U(0,t) = K*exp(-r*(1-t)) and  U(20,t) = 0,0 < t < 1 \n\n";
  std::cout << " U(x,T) = max(0,K-x), x in <0,20> \n\n";
  std::cout << "============================================================\n";

  // typedef the black_scholes_equation
  typedef black_scholes_equation<T, boundary_condition_enum::Dirichlet,
                                 fdm_thomas_lu_solver, std::vector,
                                 std::allocator<T>>
      implicit_solver;

  // parameters of the call option:
  auto const &strike = 10;
  auto const &maturity = 1.0;
  auto const &rate = 0.2;
  auto const &sig = 0.25;
  // number of space subdivisions:
  std::size_t const Sd = 160;
  // number of time subdivisions:
  std::size_t const Td = 160;
  // initial condition:
  auto terminal_condition = [&](T x) { return std::max<T>(0.0, strike - x); };
  // boundary conditions:
  auto const &dirichletRight = [](T t) { return 0.0; };
  auto const &dirichletLeft = [&](T t) {
    return (strike * std::exp(-rate * (maturity - t)));
  };
  auto boundary =
      std::make_shared<dirichlet_boundary_1d<T>>(dirichletLeft, dirichletRight);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  implicit_solver impl_solver(range<T>(0.0, 20.0), maturity, Sd, Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_terminal_condition(terminal_condition);
  // set second order coefficient:
  impl_solver.set_2_order_coefficient(
      [&](T x) { return (0.5 * x * x * sig * sig); });
  // set first order coefficient:
  impl_solver.set_1_order_coefficient([&](T x) { return (rate * x); });
  // set zero order coefficient:
  impl_solver.set_0_order_coefficient([&](T x) { return (-1.0 * rate); });
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::Euler);
  // get exact solution:
  black_scholes_exact<T> bs_exact(0.0, strike, rate, sig, maturity);

  T const h = impl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = bs_exact.put(j * h);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplEuropeanBlackScholesPutDirichletBCThomasLU() {
  std::cout << "============================================================\n";
  std::cout << "======= Implicit Black-Scholes Equation (Dirichlet BC) =====\n";
  std::cout << "============================================================\n";

  testImplEuropeanBlackScholesPutOptionBCThomasLUEuler<double>();
  testImplEuropeanBlackScholesPutOptionBCThomasLUEuler<float>();
  testImplEuropeanBlackScholesPutOptionBCThomasLUCN<double>();
  testImplEuropeanBlackScholesPutOptionBCThomasLUCN<float>();

  std::cout << "============================================================\n";
}

// ===========================================================================
// ========================== EXPLICIT SOLVERS ===============================
// ===========================================================================

template <typename T>
void testExplEuropeanBlackScholesCallOptionDirichletBCEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::explicit_pde_schemes_enum;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::explicit_solvers::
      black_scholes_equation;
  using lss_utility::black_scholes_exact;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Black-Scholes equation: \n\n";
  std::cout << " Using explicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = 0.5*sig*sig*x*x*U_xx(x,t) + r*x*U_x(x,t) - "
               "r*U(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " 0 < x < 20 and 0 < t < 1,\n";
  std::cout << " U(0,t) = 0 and  U(20,t) = 20-K*exp(-r*(1-t)),0 < t < 1 \n\n";
  std::cout << " U(x,T) = max(0,x-K), x in <0,20> \n\n";
  std::cout << "============================================================\n";

  // typedef the black_scholes_equation
  typedef black_scholes_equation<T, boundary_condition_enum::Dirichlet,
                                 std::vector, std::allocator<T>>
      explicit_solver;

  // parameters of the call option:
  auto const &strike = 10;
  auto const &maturity = 1.0;
  auto const &rate = 0.2;
  auto const &sig = 0.25;
  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 10000;
  // terminal condition:
  auto terminal_condition = [&](T x) { return std::max<T>(0.0, x - strike); };
  // boundary conditions:
  auto const &dirichletLeft = [](T t) { return 0.0; };
  auto const &dirichletRight = [&](T t) {
    return (20.0 - strike * std::exp(-rate * (maturity - t)));
  };
  auto boundary =
      std::make_shared<dirichlet_boundary_1d<T>>(dirichletLeft, dirichletRight);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  explicit_solver expl_solver(range<T>(0.0, 20.0), maturity, Sd, Td);
  // set boundary conditions:
  expl_solver.set_boundary_condition(boundary);
  // set initial condition:
  expl_solver.set_terminal_condition(terminal_condition);
  // set second order coefficient:
  expl_solver.set_2_order_coefficient(
      [&](T x) { return (0.5 * x * x * sig * sig); });
  // set first order coefficient:
  expl_solver.set_1_order_coefficient([&](T x) { return (rate * x); });
  // set zero order coefficient:
  expl_solver.set_0_order_coefficient([&](T x) { return (-1.0 * rate); });
  // get the solution:
  expl_solver.solve(solution, explicit_pde_schemes_enum::Euler);
  // get exact solution:
  black_scholes_exact<T> bs_exact(0.0, strike, rate, sig, maturity);

  T const h = expl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = bs_exact.call(j * h);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

template <typename T>
void testExplEuropeanBlackScholesCallOptionDirichletBCADEBC() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::explicit_pde_schemes_enum;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::explicit_solvers::
      black_scholes_equation;
  using lss_utility::black_scholes_exact;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Black-Scholes equation: \n\n";
  std::cout << " Using explicit ADE Barakat Clark method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = 0.5*sig*sig*x*x*U_xx(x,t) + r*x*U_x(x,t) - "
               "r*U(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " 0 < x < 20 and 0 < t < 1,\n";
  std::cout << " U(0,t) = 0 and  U(20,t) = 20-K*exp(-r*(1-t)),0 < t < 1 \n\n";
  std::cout << " U(x,T) = max(0,x-K), x in <0,20> \n\n";
  std::cout << "============================================================\n";

  // typedef the black_scholes_equation
  typedef black_scholes_equation<T, boundary_condition_enum::Dirichlet,
                                 std::vector, std::allocator<T>>
      explicit_solver;

  // parameters of the call option:
  auto const &strike = 10;
  auto const &maturity = 1.0;
  auto const &rate = 0.2;
  auto const &sig = 0.25;
  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 10000;
  // terminal condition:
  auto terminal_condition = [&](T x) { return std::max<T>(0.0, x - strike); };
  // boundary conditions:
  auto const &dirichletLeft = [](T t) { return 0.0; };
  auto const &dirichletRight = [&](T t) {
    return (20.0 - strike * std::exp(-rate * (maturity - t)));
  };
  auto boundary =
      std::make_shared<dirichlet_boundary_1d<T>>(dirichletLeft, dirichletRight);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  explicit_solver expl_solver(range<T>(0.0, 20.0), maturity, Sd, Td);
  // set boundary conditions:
  expl_solver.set_boundary_condition(boundary);
  // set initial condition:
  expl_solver.set_terminal_condition(terminal_condition);
  // set second order coefficient:
  expl_solver.set_2_order_coefficient(
      [&](T x) { return (0.5 * x * x * sig * sig); });
  // set first order coefficient:
  expl_solver.set_1_order_coefficient([&](T x) { return (rate * x); });
  // set zero order coefficient:
  expl_solver.set_0_order_coefficient([&](T x) { return (-1.0 * rate); });
  // get the solution:
  expl_solver.solve(solution, explicit_pde_schemes_enum::ADEBarakatClark);
  // get exact solution:
  black_scholes_exact<T> bs_exact(0.0, strike, rate, sig, maturity);

  T const h = expl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = bs_exact.call(j * h);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

template <typename T>
void testExplEuropeanBlackScholesCallOptionDirichletBCADES() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::explicit_pde_schemes_enum;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::explicit_solvers::
      black_scholes_equation;
  using lss_utility::black_scholes_exact;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Black-Scholes equation: \n\n";
  std::cout << " Using explicit ADE Saulyev method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = 0.5*sig*sig*x*x*U_xx(x,t) + r*x*U_x(x,t) - "
               "r*U(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " 0 < x < 20 and 0 < t < 1,\n";
  std::cout << " U(0,t) = 0 and  U(20,t) = 20-K*exp(-r*(1-t)),0 < t < 1 \n\n";
  std::cout << " U(x,T) = max(0,x-K), x in <0,20> \n\n";
  std::cout << "============================================================\n";

  // typedef the black_scholes_equation
  typedef black_scholes_equation<T, boundary_condition_enum::Dirichlet,
                                 std::vector, std::allocator<T>>
      explicit_solver;

  // parameters of the call option:
  auto const &strike = 10;
  auto const &maturity = 1.0;
  auto const &rate = 0.2;
  auto const &sig = 0.25;
  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 10000;
  // terminal condition:
  auto terminal_condition = [&](T x) { return std::max<T>(0.0, x - strike); };
  // boundary conditions:
  auto const &dirichletLeft = [](T t) { return 0.0; };
  auto const &dirichletRight = [&](T t) {
    return (20.0 - strike * std::exp(-rate * (maturity - t)));
  };
  auto boundary =
      std::make_shared<dirichlet_boundary_1d<T>>(dirichletLeft, dirichletRight);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  explicit_solver expl_solver(range<T>(0.0, 20.0), maturity, Sd, Td);
  // set boundary conditions:
  expl_solver.set_boundary_condition(boundary);
  // set initial condition:
  expl_solver.set_terminal_condition(terminal_condition);
  // set second order coefficient:
  expl_solver.set_2_order_coefficient(
      [&](T x) { return (0.5 * x * x * sig * sig); });
  // set first order coefficient:
  expl_solver.set_1_order_coefficient([&](T x) { return (rate * x); });
  // set zero order coefficient:
  expl_solver.set_0_order_coefficient([&](T x) { return (-1.0 * rate); });
  // get the solution:
  expl_solver.solve(solution, explicit_pde_schemes_enum::ADESaulyev);
  // get exact solution:
  black_scholes_exact<T> bs_exact(0.0, strike, rate, sig, maturity);

  T const h = expl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = bs_exact.call(j * h);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testExplEuropeanBlackScholesCallOptionDirichletBC() {
  std::cout << "============================================================\n";
  std::cout << "====== Explicit Black-Scholes Equation (Dirichlet BC) ======\n";
  std::cout << "============================================================\n";

  testExplEuropeanBlackScholesCallOptionDirichletBCEuler<double>();
  testExplEuropeanBlackScholesCallOptionDirichletBCEuler<float>();
  testExplEuropeanBlackScholesCallOptionDirichletBCADEBC<double>();
  testExplEuropeanBlackScholesCallOptionDirichletBCADEBC<float>();
  testExplEuropeanBlackScholesCallOptionDirichletBCADES<double>();
  testExplEuropeanBlackScholesCallOptionDirichletBCADES<float>();

  std::cout << "============================================================\n";
}

template <typename T>
void testExplEuropeanBlackScholesPutOptionDirichletBCEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::explicit_pde_schemes_enum;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::explicit_solvers::
      black_scholes_equation;
  using lss_utility::black_scholes_exact;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Black-Scholes equation: \n\n";
  std::cout << " Using explicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = 0.5*sig*sig*x*x*U_xx(x,t) + r*x*U_x(x,t) - "
               "r*U(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " 0 < x < 20 and 0 < t < 1,\n";
  std::cout << " U(0,t) = K*exp(-r*(1-t)) and  U(20,t) = 0,0 < t < 1 \n\n";
  std::cout << " U(x,T) = max(0,K-x), x in <0,20> \n\n";
  std::cout << "============================================================\n";

  // typedef the black_scholes_equation
  typedef black_scholes_equation<T, boundary_condition_enum::Dirichlet,
                                 std::vector, std::allocator<T>>
      explicit_solver;

  // parameters of the call option:
  auto const &strike = 10;
  auto const &maturity = 1.0;
  auto const &rate = 0.2;
  auto const &sig = 0.25;
  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 10000;
  // terminal condition:
  auto terminal_condition = [&](T x) { return std::max<T>(0.0, strike - x); };
  // boundary conditions:
  auto const &dirichletRight = [](T t) { return 0.0; };
  auto const &dirichletLeft = [&](T t) {
    return (strike * std::exp(-rate * (maturity - t)));
  };
  auto boundary =
      std::make_shared<dirichlet_boundary_1d<T>>(dirichletLeft, dirichletRight);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  explicit_solver expl_solver(range<T>(0.0, 20.0), maturity, Sd, Td);
  // set boundary conditions:
  expl_solver.set_boundary_condition(boundary);
  // set initial condition:
  expl_solver.set_terminal_condition(terminal_condition);
  // set second order coefficient:
  expl_solver.set_2_order_coefficient(
      [&](T x) { return (0.5 * x * x * sig * sig); });
  // set first order coefficient:
  expl_solver.set_1_order_coefficient([&](T x) { return (rate * x); });
  // set zero order coefficient:
  expl_solver.set_0_order_coefficient([&](T x) { return (-1.0 * rate); });
  // get the solution:
  expl_solver.solve(solution, explicit_pde_schemes_enum::Euler);
  // get exact solution:
  black_scholes_exact<T> bs_exact(0.0, strike, rate, sig, maturity);

  T const h = expl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = bs_exact.put(j * h);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

template <typename T>
void testExplEuropeanBlackScholesPutOptionDirichletBCADEBC() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::explicit_pde_schemes_enum;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::explicit_solvers::
      black_scholes_equation;
  using lss_utility::black_scholes_exact;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Black-Scholes equation: \n\n";
  std::cout << " Using explicit ADE Barakat Clark method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = 0.5*sig*sig*x*x*U_xx(x,t) + r*x*U_x(x,t) - "
               "r*U(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " 0 < x < 20 and 0 < t < 1,\n";
  std::cout << " U(0,t) = K*exp(-r*(1-t)) and  U(20,t) = 0,0 < t < 1 \n\n";
  std::cout << " U(x,T) = max(0,K-x), x in <0,20> \n\n";
  std::cout << "============================================================\n";

  // typedef the black_scholes_equation
  typedef black_scholes_equation<T, boundary_condition_enum::Dirichlet,
                                 std::vector, std::allocator<T>>
      explicit_solver;

  // parameters of the call option:
  auto const &strike = 10;
  auto const &maturity = 1.0;
  auto const &rate = 0.2;
  auto const &sig = 0.25;
  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 10000;
  // terminal condition:
  auto terminal_condition = [&](T x) { return std::max<T>(0.0, strike - x); };
  // boundary conditions:
  auto const &dirichletRight = [](T t) { return 0.0; };
  auto const &dirichletLeft = [&](T t) {
    return (strike * std::exp(-rate * (maturity - t)));
  };
  auto boundary =
      std::make_shared<dirichlet_boundary_1d<T>>(dirichletLeft, dirichletRight);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  explicit_solver expl_solver(range<T>(0.0, 20.0), maturity, Sd, Td);
  // set boundary conditions:
  expl_solver.set_boundary_condition(boundary);
  // set initial condition:
  expl_solver.set_terminal_condition(terminal_condition);
  // set second order coefficient:
  expl_solver.set_2_order_coefficient(
      [&](T x) { return (0.5 * x * x * sig * sig); });
  // set first order coefficient:
  expl_solver.set_1_order_coefficient([&](T x) { return (rate * x); });
  // set zero order coefficient:
  expl_solver.set_0_order_coefficient([&](T x) { return (-1.0 * rate); });
  // get the solution:
  expl_solver.solve(solution, explicit_pde_schemes_enum::ADEBarakatClark);
  // get exact solution:
  black_scholes_exact<T> bs_exact(0.0, strike, rate, sig, maturity);

  T const h = expl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = bs_exact.put(j * h);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

template <typename T>
void testExplEuropeanBlackScholesPutOptionDirichletBCADES() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::explicit_pde_schemes_enum;
  using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
  using lss_one_dim_space_variable_pde_solvers::explicit_solvers::
      black_scholes_equation;
  using lss_utility::black_scholes_exact;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Black-Scholes equation: \n\n";
  std::cout << " Using explicit ADE Saulyev method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = 0.5*sig*sig*x*x*U_xx(x,t) + r*x*U_x(x,t) - "
               "r*U(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " 0 < x < 20 and 0 < t < 1,\n";
  std::cout << " U(0,t) = K*exp(-r*(1-t)) and  U(20,t) = 0,0 < t < 1 \n\n";
  std::cout << " U(x,T) = max(0,K-x), x in <0,20> \n\n";
  std::cout << "============================================================\n";

  // typedef the black_scholes_equation
  typedef black_scholes_equation<T, boundary_condition_enum::Dirichlet,
                                 std::vector, std::allocator<T>>
      explicit_solver;

  // parameters of the call option:
  auto const &strike = 10;
  auto const &maturity = 1.0;
  auto const &rate = 0.2;
  auto const &sig = 0.25;
  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 10000;
  // terminal condition:
  auto terminal_condition = [&](T x) { return std::max<T>(0.0, strike - x); };
  // boundary conditions:
  auto const &dirichletRight = [](T t) { return 0.0; };
  auto const &dirichletLeft = [&](T t) {
    return (strike * std::exp(-rate * (maturity - t)));
  };
  auto boundary =
      std::make_shared<dirichlet_boundary_1d<T>>(dirichletLeft, dirichletRight);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<T> solution(Sd + 1, T{});
  // initialize solver
  explicit_solver expl_solver(range<T>(0.0, 20.0), maturity, Sd, Td);
  // set boundary conditions:
  expl_solver.set_boundary_condition(boundary);
  // set initial condition:
  expl_solver.set_terminal_condition(terminal_condition);
  // set second order coefficient:
  expl_solver.set_2_order_coefficient(
      [&](double x) { return (0.5 * x * x * sig * sig); });
  // set first order coefficient:
  expl_solver.set_1_order_coefficient([&](double x) { return (rate * x); });
  // set zero order coefficient:
  expl_solver.set_0_order_coefficient([&](double x) { return (-1.0 * rate); });
  // get the solution:
  expl_solver.solve(solution, explicit_pde_schemes_enum::ADESaulyev);
  // get exact solution:
  black_scholes_exact<T> bs_exact(0.0, strike, rate, sig, maturity);

  T const h = expl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = bs_exact.put(j * h);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testExplEuropeanBlackScholesPutOptionDirichletBC() {
  std::cout << "============================================================\n";
  std::cout << "====== Explicit Black-Scholes Equation (Dirichlet BC) ======\n";
  std::cout << "============================================================\n";

  testExplEuropeanBlackScholesPutOptionDirichletBCEuler<double>();
  testExplEuropeanBlackScholesPutOptionDirichletBCEuler<float>();
  testExplEuropeanBlackScholesPutOptionDirichletBCADEBC<double>();
  testExplEuropeanBlackScholesPutOptionDirichletBCADEBC<float>();
  testExplEuropeanBlackScholesPutOptionDirichletBCADES<double>();
  testExplEuropeanBlackScholesPutOptionDirichletBCADES<float>();

  std::cout << "============================================================\n";
}

#endif  //_LSS_ONE_DIM_BLACK_SCHOLES_EQUATION_T
