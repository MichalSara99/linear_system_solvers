#pragma once
#if !defined(_LSS_ONE_DIM_BLACK_SCHOLES_EQUATION_CUDA_T)
#define _LSS_ONE_DIM_BLACK_SCHOLES_EQUATION_CUDA_T

#pragma warning(disable : 4305)

#include "common/lss_enumerations.h"
#include "common/lss_utility.h"
#include "pde_solvers/one_dim/variable_coefficients/lss_black_scholes_equation_solvers_cuda.h"
#include "sparse_solvers/lss_fdm_double_sweep_solver.h"
#include "sparse_solvers/lss_fdm_thomas_lu_solver.h"

// ///////////////////////////////////////////////////////////////////////////
//					BLACK-SCHOLES PROBLEMS ON CUDA
// ///////////////////////////////////////////////////////////////////////////

// ===========================================================================
// ========================== IMPLICIT SOLVERS ===============================
// ===========================================================================

void testImplEuropeanBlackScholesCallOptionBCDoubleDeviceEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_enumerations::memory_space_enum;
  using lss_one_dim_space_variable_pde_solvers_cuda::implicit_solvers::
      black_sholes_equation_cuda;
  using lss_sparse_solvers::real_sparse_solver_cuda;
  using lss_utility::black_scholes_exact;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Black-Scholes equation: \n\n";
  std::cout << " Using CUDA solvers algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(double).name() << "\n\n";
  std::cout << " U_t(x,t) = 0.5*sig*sig*x*x*U_xx(x,t) + r*x*U_x(x,t) - "
               "r*U(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " 0 < x < 20 and 0 < t < 1,\n";
  std::cout << " U(0,t) = 0 and  U(20,t) = 20-K*exp(-r*(1-t)),0 < t < 1 \n\n";
  std::cout << " U(x,T) = max(0,x-K), x in <0,20> \n\n";
  std::cout << "============================================================\n";

  // typedef the black_sholes_equation_cuda
  typedef black_sholes_equation_cuda<
      double, boundary_condition_enum::Dirichlet, memory_space_enum::Device,
      real_sparse_solver_cuda, std::vector, std::allocator<double>>
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
  auto terminal_condition = [&](double x) {
    return std::max<double>(0.0, x - strike);
  };
  // boundary conditions:
  auto const &dirichletLeft = [](double t) { return 0.0; };
  auto const &dirichletRight = [&](double t) {
    return (20.0 - strike * std::exp(-rate * (maturity - t)));
  };
  auto boundary = std::make_pair(dirichletLeft, dirichletRight);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<double> solution(Sd + 1, double{});
  // initialize solver
  implicit_solver impl_solver(range<double>(0.0, 20.0), maturity, Sd, Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_terminal_condition(terminal_condition);
  // set second order coefficient:
  impl_solver.set_2_order_coefficient(
      [&](double x) { return (0.5 * x * x * sig * sig); });
  // set first order coefficient:
  impl_solver.set_1_order_coefficient([&](double x) { return (rate * x); });
  // set zero order coefficient:
  impl_solver.set_0_order_coefficient([&](double x) { return (-1.0 * rate); });
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::Euler);
  // get exact solution:
  black_scholes_exact<double> bs_exact(0.0, strike, rate, sig, maturity);

  double const h = impl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  double benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = bs_exact.call(j * h);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplEuropeanBlackScholesCallOptionBCFloatDeviceEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_enumerations::memory_space_enum;
  using lss_one_dim_space_variable_pde_solvers_cuda::implicit_solvers::
      black_sholes_equation_cuda;
  using lss_sparse_solvers::real_sparse_solver_cuda;
  using lss_utility::black_scholes_exact;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Black-Scholes equation: \n\n";
  std::cout << " Using CUDA solvers algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(float).name() << "\n\n";
  std::cout << " U_t(x,t) = 0.5*sig*sig*x*x*U_xx(x,t) + r*x*U_x(x,t) - "
               "r*U(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " 0 < x < 20 and 0 < t < 1,\n";
  std::cout << " U(0,t) = 0 and  U(20,t) = 20-K*exp(-r*(1-t)),0 < t < 1 \n\n";
  std::cout << " U(x,T) = max(0,x-K), x in <0,20> \n\n";
  std::cout << "============================================================\n";

  // typedef the black_sholes_equation_cuda
  typedef black_sholes_equation_cuda<
      float, boundary_condition_enum::Dirichlet, memory_space_enum::Device,
      real_sparse_solver_cuda, std::vector, std::allocator<float>>
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
  auto terminal_condition = [&](float x) {
    return std::max<float>(0.0, x - strike);
  };
  // boundary conditions:
  auto const &dirichletLeft = [](float t) { return 0.0; };
  auto const &dirichletRight = [&](float t) {
    return (20.0 - strike * std::exp(-rate * (maturity - t)));
  };
  auto boundary = std::make_pair(dirichletLeft, dirichletRight);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<float> solution(Sd + 1, float{});
  // initialize solver
  implicit_solver impl_solver(range<float>(0.0, 20.0), maturity, Sd, Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_terminal_condition(terminal_condition);
  // set second order coefficient:
  impl_solver.set_2_order_coefficient(
      [&](float x) { return (0.5 * x * x * sig * sig); });
  // set first order coefficient:
  impl_solver.set_1_order_coefficient([&](float x) { return (rate * x); });
  // set zero order coefficient:
  impl_solver.set_0_order_coefficient([&](float x) { return (-1.0 * rate); });
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::Euler);
  // get exact solution:
  black_scholes_exact<float> bs_exact(0.0, strike, rate, sig, maturity);

  float const h = impl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  float benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = bs_exact.call(j * h);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplEuropeanBlackScholesCallOptionBCDoubleDeviceCN() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_enumerations::memory_space_enum;
  using lss_one_dim_space_variable_pde_solvers_cuda::implicit_solvers::
      black_sholes_equation_cuda;
  using lss_sparse_solvers::real_sparse_solver_cuda;
  using lss_utility::black_scholes_exact;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Black-Scholes equation: \n\n";
  std::cout << " Using CUDA solvers algorithm with implicit Crank-Nicolson \n"
               "method\n\n";
  std::cout << " Value type: " << typeid(double).name() << "\n\n";
  std::cout << " U_t(x,t) = 0.5*sig*sig*x*x*U_xx(x,t) + r*x*U_x(x,t) - "
               "r*U(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " 0 < x < 20 and 0 < t < 1,\n";
  std::cout << " U(0,t) = 0 and  U(20,t) = 20-K*exp(-r*(1-t)),0 < t < 1 \n\n";
  std::cout << " U(x,T) = max(0,x-K), x in <0,20> \n\n";
  std::cout << "============================================================\n";

  // typedef the black_sholes_equation_cuda
  typedef black_sholes_equation_cuda<
      double, boundary_condition_enum::Dirichlet, memory_space_enum::Device,
      real_sparse_solver_cuda, std::vector, std::allocator<double>>
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
  auto terminal_condition = [&](double x) {
    return std::max<double>(0.0, x - strike);
  };
  // boundary conditions:
  auto const &dirichletLeft = [](double t) { return 0.0; };
  auto const &dirichletRight = [&](double t) {
    return (20.0 - strike * std::exp(-rate * (maturity - t)));
  };
  auto boundary = std::make_pair(dirichletLeft, dirichletRight);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<double> solution(Sd + 1, double{});
  // initialize solver
  implicit_solver impl_solver(range<double>(0.0, 20.0), maturity, Sd, Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_terminal_condition(terminal_condition);
  // set second order coefficient:
  impl_solver.set_2_order_coefficient(
      [&](double x) { return (0.5 * x * x * sig * sig); });
  // set first order coefficient:
  impl_solver.set_1_order_coefficient([&](double x) { return (rate * x); });
  // set zero order coefficient:
  impl_solver.set_0_order_coefficient([&](double x) { return (-1.0 * rate); });
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::CrankNicolson);
  // get exact solution:
  black_scholes_exact<double> bs_exact(0.0, strike, rate, sig, maturity);

  double const h = impl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  double benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = bs_exact.call(j * h);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplEuropeanBlackScholesCallOptionBCFloatDeviceCN() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_enumerations::memory_space_enum;
  using lss_one_dim_space_variable_pde_solvers_cuda::implicit_solvers::
      black_sholes_equation_cuda;
  using lss_sparse_solvers::real_sparse_solver_cuda;
  using lss_utility::black_scholes_exact;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Black-Scholes equation: \n\n";
  std::cout << " Using CUDA solvers algorithm with implicit Crank-Nicolson \n"
               "method\n\n";
  std::cout << " Value type: " << typeid(float).name() << "\n\n";
  std::cout << " U_t(x,t) = 0.5*sig*sig*x*x*U_xx(x,t) + r*x*U_x(x,t) - "
               "r*U(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " 0 < x < 20 and 0 < t < 1,\n";
  std::cout << " U(0,t) = 0 and  U(20,t) = 20-K*exp(-r*(1-t)),0 < t < 1 \n\n";
  std::cout << " U(x,T) = max(0,x-K), x in <0,20> \n\n";
  std::cout << "============================================================\n";

  // typedef the black_sholes_equation_cuda
  typedef black_sholes_equation_cuda<
      float, boundary_condition_enum::Dirichlet, memory_space_enum::Device,
      real_sparse_solver_cuda, std::vector, std::allocator<float>>
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
  auto terminal_condition = [&](float x) {
    return std::max<float>(0.0, x - strike);
  };
  // boundary conditions:
  auto const &dirichletLeft = [](float t) { return 0.0; };
  auto const &dirichletRight = [&](float t) {
    return (20.0 - strike * std::exp(-rate * (maturity - t)));
  };
  auto boundary = std::make_pair(dirichletLeft, dirichletRight);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<float> solution(Sd + 1, float{});
  // initialize solver
  implicit_solver impl_solver(range<float>(0.0, 20.0), maturity, Sd, Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_terminal_condition(terminal_condition);
  // set second order coefficient:
  impl_solver.set_2_order_coefficient(
      [&](float x) { return (0.5 * x * x * sig * sig); });
  // set first order coefficient:
  impl_solver.set_1_order_coefficient([&](float x) { return (rate * x); });
  // set zero order coefficient:
  impl_solver.set_0_order_coefficient([&](float x) { return (-1.0 * rate); });
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::CrankNicolson);
  // get exact solution:
  black_scholes_exact<float> bs_exact(0.0, strike, rate, sig, maturity);

  float const h = impl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  float benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = bs_exact.call(j * h);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplEuropeanBlackScholesCallDirichletBCDeviceCUDA() {
  std::cout << "============================================================\n";
  std::cout << "======== Implicit Balck-Scholes Equation (Dirichlet BC) ====\n";
  std::cout << "============================================================\n";

  testImplEuropeanBlackScholesCallOptionBCFloatDeviceEuler();
  testImplEuropeanBlackScholesCallOptionBCDoubleDeviceEuler();
  testImplEuropeanBlackScholesCallOptionBCFloatDeviceCN();
  testImplEuropeanBlackScholesCallOptionBCDoubleDeviceCN();

  std::cout << "============================================================\n";
}

void testImplEuropeanBlackScholesPutOptionBCDoubleDeviceEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_enumerations::memory_space_enum;
  using lss_one_dim_space_variable_pde_solvers_cuda::implicit_solvers::
      black_sholes_equation_cuda;
  using lss_sparse_solvers::real_sparse_solver_cuda;
  using lss_utility::black_scholes_exact;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Black-Scholes equation: \n\n";
  std::cout << "Using CUDA algorithm with implicit Euler method\n\n";
  std::cout << " Value type : " << typeid(double).name() << "\n\n";
  std::cout << " U_t(x, t) = 0.5*sig*sig*x*x*U_xx(x, t) + r*x*U_x(x, t) -"
               "r*U(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " 0 < x < 20 and 0 < t < 1,\n";
  std::cout << " U(0,t) = K*exp(-r*(1-t)) and  U(20,t) = 0,0 < t < 1 \n\n";
  std::cout << " U(x,T) = max(0,K-x), x in <0,20> \n\n";
  std::cout << "============================================================\n";

  // typedef the black_sholes_equation_cuda
  typedef black_sholes_equation_cuda<
      double, boundary_condition_enum::Dirichlet, memory_space_enum::Device,
      real_sparse_solver_cuda, std::vector, std::allocator<double>>
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
  auto terminal_condition = [&](double x) {
    return std::max<double>(0.0, strike - x);
  };
  // boundary conditions:
  auto const &dirichletRight = [](double t) { return 0.0; };
  auto const &dirichletLeft = [&](double t) {
    return (strike * std::exp(-rate * (maturity - t)));
  };
  auto boundary = std::make_pair(dirichletLeft, dirichletRight);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<double> solution(Sd + 1, double{});
  // initialize solver
  implicit_solver impl_solver(range<double>(0.0, 20.0), maturity, Sd, Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_terminal_condition(terminal_condition);
  // set second order coefficient:
  impl_solver.set_2_order_coefficient(
      [&](double x) { return (0.5 * x * x * sig * sig); });
  // set first order coefficient:
  impl_solver.set_1_order_coefficient([&](double x) { return (rate * x); });
  // set zero order coefficient:
  impl_solver.set_0_order_coefficient([&](double x) { return (-1.0 * rate); });
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::Euler);
  // get exact solution:
  black_scholes_exact<double> bs_exact(0.0, strike, rate, sig, maturity);

  double const h = impl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  double benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = bs_exact.put(j * h);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplEuropeanBlackScholesPutOptionBCFloatDeviceEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_enumerations::memory_space_enum;
  using lss_one_dim_space_variable_pde_solvers_cuda::implicit_solvers::
      black_sholes_equation_cuda;
  using lss_sparse_solvers::real_sparse_solver_cuda;
  using lss_utility::black_scholes_exact;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Black-Scholes equation: \n\n";
  std::cout << "Using CUDA algorithm with implicit Euler method\n\n";
  std::cout << " Value type : " << typeid(float).name() << "\n\n";
  std::cout << " U_t(x, t) = 0.5*sig*sig*x*x*U_xx(x, t) + r*x*U_x(x, t) -"
               "r*U(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " 0 < x < 20 and 0 < t < 1,\n";
  std::cout << " U(0,t) = K*exp(-r*(1-t)) and  U(20,t) = 0,0 < t < 1 \n\n";
  std::cout << " U(x,T) = max(0,K-x), x in <0,20> \n\n";
  std::cout << "============================================================\n";

  // typedef the black_sholes_equation_cuda
  typedef black_sholes_equation_cuda<
      float, boundary_condition_enum::Dirichlet, memory_space_enum::Device,
      real_sparse_solver_cuda, std::vector, std::allocator<float>>
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
  auto terminal_condition = [&](float x) {
    return std::max<float>(0.0, strike - x);
  };
  // boundary conditions:
  auto const &dirichletRight = [](float t) { return 0.0; };
  auto const &dirichletLeft = [&](float t) {
    return (strike * std::exp(-rate * (maturity - t)));
  };
  auto boundary = std::make_pair(dirichletLeft, dirichletRight);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<float> solution(Sd + 1, float{});
  // initialize solver
  implicit_solver impl_solver(range<float>(0.0, 20.0), maturity, Sd, Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_terminal_condition(terminal_condition);
  // set second order coefficient:
  impl_solver.set_2_order_coefficient(
      [&](float x) { return (0.5 * x * x * sig * sig); });
  // set first order coefficient:
  impl_solver.set_1_order_coefficient([&](float x) { return (rate * x); });
  // set zero order coefficient:
  impl_solver.set_0_order_coefficient([&](float x) { return (-1.0 * rate); });
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::Euler);
  // get exact solution:
  black_scholes_exact<float> bs_exact(0.0, strike, rate, sig, maturity);

  float const h = impl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  float benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = bs_exact.put(j * h);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplEuropeanBlackScholesPutOptionBCDoubleCN() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_enumerations::memory_space_enum;
  using lss_one_dim_space_variable_pde_solvers_cuda::implicit_solvers::
      black_sholes_equation_cuda;
  using lss_sparse_solvers::real_sparse_solver_cuda;
  using lss_utility::black_scholes_exact;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Black-Scholes equation: \n\n";
  std::cout << "Using CUDA algorithm with Crank -Nicolson method\n\n ";
  std::cout << " Value type: " << typeid(double).name() << "\n\n ";
  std::cout << " U_t(x, t) =0.5*sig*sig*x*x*U_xx(x, t) + r*x*U_x(x, t) -"
               "r*U(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " 0 < x < 20 and 0 < t < 1,\n";
  std::cout << " U(0,t) = K*exp(-r*(1-t)) and  U(20,t) = 0, 0 < t < 1 \n\n";
  std::cout << " U(x,T) = max(0,K-x), x in <0,20> \n\n";
  std::cout << "============================================================\n";

  // typedef the black_sholes_equation_cuda
  typedef black_sholes_equation_cuda<
      double, boundary_condition_enum::Dirichlet, memory_space_enum::Device,
      real_sparse_solver_cuda, std::vector, std::allocator<double>>
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
  auto terminal_condition = [&](double x) {
    return std::max<double>(0.0, strike - x);
  };
  // boundary conditions:
  auto const &dirichletRight = [](double t) { return 0.0; };
  auto const &dirichletLeft = [&](double t) {
    return (strike * std::exp(-rate * (maturity - t)));
  };
  auto boundary = std::make_pair(dirichletLeft, dirichletRight);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<double> solution(Sd + 1, double{});
  // initialize solver
  implicit_solver impl_solver(range<double>(0.0, 20.0), maturity, Sd, Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_terminal_condition(terminal_condition);
  // set second order coefficient:
  impl_solver.set_2_order_coefficient(
      [&](double x) { return (0.5 * x * x * sig * sig); });
  // set first order coefficient:
  impl_solver.set_1_order_coefficient([&](double x) { return (rate * x); });
  // set zero order coefficient:
  impl_solver.set_0_order_coefficient([&](double x) { return (-1.0 * rate); });
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::Euler);
  // get exact solution:
  black_scholes_exact<double> bs_exact(0.0, strike, rate, sig, maturity);

  double const h = impl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  double benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = bs_exact.put(j * h);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplEuropeanBlackScholesPutOptionBCFloatCN() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_enumerations::memory_space_enum;
  using lss_one_dim_space_variable_pde_solvers_cuda::implicit_solvers::
      black_sholes_equation_cuda;
  using lss_sparse_solvers::real_sparse_solver_cuda;
  using lss_utility::black_scholes_exact;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Black-Scholes equation: \n\n";
  std::cout << "Using CUDA algorithm with Crank -Nicolson method\n\n ";
  std::cout << " Value type: " << typeid(float).name() << "\n\n ";
  std::cout << " U_t(x, t) =0.5*sig*sig*x*x*U_xx(x, t) + r*x*U_x(x, t) -"
               "r*U(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " 0 < x < 20 and 0 < t < 1,\n";
  std::cout << " U(0,t) = K*exp(-r*(1-t)) and  U(20,t) = 0, 0 < t < 1 \n\n";
  std::cout << " U(x,T) = max(0,K-x), x in <0,20> \n\n";
  std::cout << "============================================================\n";

  // typedef the black_sholes_equation_cuda
  typedef black_sholes_equation_cuda<
      float, boundary_condition_enum::Dirichlet, memory_space_enum::Device,
      real_sparse_solver_cuda, std::vector, std::allocator<float>>
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
  auto terminal_condition = [&](float x) {
    return std::max<float>(0.0, strike - x);
  };
  // boundary conditions:
  auto const &dirichletRight = [](float t) { return 0.0; };
  auto const &dirichletLeft = [&](float t) {
    return (strike * std::exp(-rate * (maturity - t)));
  };
  auto boundary = std::make_pair(dirichletLeft, dirichletRight);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<float> solution(Sd + 1, float{});
  // initialize solver
  implicit_solver impl_solver(range<float>(0.0, 20.0), maturity, Sd, Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_terminal_condition(terminal_condition);
  // set second order coefficient:
  impl_solver.set_2_order_coefficient(
      [&](float x) { return (0.5 * x * x * sig * sig); });
  // set first order coefficient:
  impl_solver.set_1_order_coefficient([&](float x) { return (rate * x); });
  // set zero order coefficient:
  impl_solver.set_0_order_coefficient([&](float x) { return (-1.0 * rate); });
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::Euler);
  // get exact solution:
  black_scholes_exact<float> bs_exact(0.0, strike, rate, sig, maturity);

  float const h = impl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  float benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = bs_exact.put(j * h);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplEuropeanBlackScholesPutDirichletBCDeviceCUDA() {
  std::cout << "============================================================\n";
  std::cout << "======= Implicit Black-Scholes Equation (Dirichlet BC) =====\n";
  std::cout << "============================================================\n";

  testImplEuropeanBlackScholesPutOptionBCFloatDeviceEuler();
  testImplEuropeanBlackScholesPutOptionBCDoubleDeviceEuler();
  testImplEuropeanBlackScholesPutOptionBCFloatCN();
  testImplEuropeanBlackScholesPutOptionBCDoubleCN();

  std::cout << "============================================================\n";
}

// ===========================================================================
// ========================== EXPLICIT SOLVERS ===============================
// ===========================================================================

void testExplEuropeanBlackScholesCallOptionBCDoubleEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_enumerations::memory_space_enum;
  using lss_one_dim_space_variable_pde_solvers_cuda::explicit_solvers::
      black_sholes_equation_cuda;
  using lss_utility::black_scholes_exact;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Black-Scholes equation: \n\n";
  std::cout << "Using CUDA algorithm with implicit Euler method\n\n";
  std::cout << " Value type : " << typeid(double).name() << "\n\n";
  std::cout << " U_t(x, t) =0.5*sig*sig*x*x*U_xx(x, t) + r*x*U_x(x, t) -"
               "r*U(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " 0 < x < 20 and 0 < t < 1,\n";
  std::cout << " U(0,t) = 0 and  U(20,t) = 20-K*exp(-r*(1-t)) ,0 < t < 1 \n\n";
  std::cout << " U(x,T) = max(0,x-K), x in <0,20> \n\n";
  std::cout << "============================================================\n";

  // typedef the black_sholes_equation_cuda
  typedef black_sholes_equation_cuda<double, boundary_condition_enum::Dirichlet,
                                     std::vector, std::allocator<double>>
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
  // initial condition:
  auto terminal_condition = [&](double x) {
    return std::max<double>(0.0, x - strike);
  };
  // boundary conditions:
  auto const &dirichletLeft = [](double t) { return 0.0; };
  auto const &dirichletRight = [&](double t) {
    return (20.0 - strike * std::exp(-rate * (maturity - t)));
  };
  auto boundary = std::make_pair(dirichletLeft, dirichletRight);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<double> solution(Sd + 1, double{});
  // initialize solver
  explicit_solver expl_solver(range<double>(0.0, 20.0), maturity, Sd, Td);
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
  expl_solver.solve(solution);
  // get exact solution:
  black_scholes_exact<double> bs_exact(0.0, strike, rate, sig, maturity);

  double const h = expl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  double benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = bs_exact.call(j * h);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testExplEuropeanBlackScholesCallOptionBCFloatEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_enumerations::memory_space_enum;
  using lss_one_dim_space_variable_pde_solvers_cuda::explicit_solvers::
      black_sholes_equation_cuda;
  using lss_utility::black_scholes_exact;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Black-Scholes equation: \n\n";
  std::cout << "Using CUDA algorithm with implicit Euler method\n\n";
  std::cout << " Value type : " << typeid(float).name() << "\n\n";
  std::cout << " U_t(x, t) =0.5*sig*sig*x*x*U_xx(x, t) + r*x*U_x(x, t) -"
               "r*U(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " 0 < x < 20 and 0 < t < 1,\n";
  std::cout << " U(0,t) = 0 and  U(20,t) = 20-K*exp(-r*(1-t)) ,0 < t < 1 \n\n";
  std::cout << " U(x,T) = max(0.x-K), x in <0,20> \n\n";
  std::cout << "============================================================\n";

  // typedef the black_sholes_equation_cuda
  typedef black_sholes_equation_cuda<float, boundary_condition_enum::Dirichlet,
                                     std::vector, std::allocator<float>>
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
  // initial condition:
  auto terminal_condition = [&](float x) {
    return std::max<float>(0.0, x - strike);
  };
  // boundary conditions:
  auto const &dirichletLeft = [](float t) { return 0.0; };
  auto const &dirichletRight = [&](float t) {
    return (20.0 - strike * std::exp(-rate * (maturity - t)));
  };
  auto boundary = std::make_pair(dirichletLeft, dirichletRight);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<float> solution(Sd + 1, float{});
  // initialize solver
  explicit_solver expl_solver(range<float>(0.0, 20.0), maturity, Sd, Td);
  // set boundary conditions:
  expl_solver.set_boundary_condition(boundary);
  // set initial condition:
  expl_solver.set_terminal_condition(terminal_condition);
  // set second order coefficient:
  expl_solver.set_2_order_coefficient(
      [&](float x) { return (0.5 * x * x * sig * sig); });
  // set first order coefficient:
  expl_solver.set_1_order_coefficient([&](float x) { return (rate * x); });
  // set zero order coefficient:
  expl_solver.set_0_order_coefficient([&](float x) { return (-1.0 * rate); });
  // get the solution:
  expl_solver.solve(solution);
  // get exact solution:
  black_scholes_exact<float> bs_exact(0.0, strike, rate, sig, maturity);

  float const h = expl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  float benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = bs_exact.call(j * h);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testExplEuropeanBlackScholesCallDirichletBCDeviceCUDA() {
  std::cout << "============================================================\n";
  std::cout << "======== Explicit Balck-Scholes Equation (Dirichlet BC) ====\n";
  std::cout << "============================================================\n";

  testExplEuropeanBlackScholesCallOptionBCDoubleEuler();
  testExplEuropeanBlackScholesCallOptionBCFloatEuler();

  std::cout << "============================================================\n";
}

void testExplEuropeanBlackScholesPutOptionBCDoubleEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_enumerations::memory_space_enum;
  using lss_one_dim_space_variable_pde_solvers_cuda::explicit_solvers::
      black_sholes_equation_cuda;
  using lss_utility::black_scholes_exact;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Black-Scholes equation: \n\n";
  std::cout << "Using CUDA algorithm with implicit Euler method\n\n";
  std::cout << " Value type : " << typeid(double).name() << "\n\n";
  std::cout << " U_t(x, t) =0.5*sig*sig*x*x*U_xx(x, t) + r*x*U_x(x, t) -"
               "r*U(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " 0 < x < 20 and 0 < t < 1,\n";
  std::cout << " U(0,t) = K*exp(-r*(1-t)) and  U(20,t) = 0 ,0 < t < 1 \n\n";
  std::cout << " U(x,T) = max(0,K-x), x in <0,20> \n\n";
  std::cout << "============================================================\n";

  // typedef the black_sholes_equation_cuda
  typedef black_sholes_equation_cuda<double, boundary_condition_enum::Dirichlet,
                                     std::vector, std::allocator<double>>
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
  // initial condition:
  auto terminal_condition = [&](double x) {
    return std::max<double>(0.0, strike - x);
  };
  // boundary conditions:
  auto const &dirichletRight = [](double t) { return 0.0; };
  auto const &dirichletLeft = [&](double t) {
    return (strike * std::exp(-rate * (maturity - t)));
  };
  auto boundary = std::make_pair(dirichletLeft, dirichletRight);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<double> solution(Sd + 1, double{});
  // initialize solver
  explicit_solver expl_solver(range<double>(0.0, 20.0), maturity, Sd, Td);
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
  expl_solver.solve(solution);
  // get exact solution:
  black_scholes_exact<double> bs_exact(0.0, strike, rate, sig, maturity);

  double const h = expl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  double benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = bs_exact.put(j * h);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testExplEuropeanBlackScholesPutOptionBCFloatEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_enumerations::memory_space_enum;
  using lss_one_dim_space_variable_pde_solvers_cuda::explicit_solvers::
      black_sholes_equation_cuda;
  using lss_utility::black_scholes_exact;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Black-Scholes equation: \n\n";
  std::cout << "Using CUDA algorithm with implicit Euler method\n\n";
  std::cout << " Value type : " << typeid(float).name() << "\n\n";
  std::cout << " U_t(x, t) =0.5*sig*sig*x*x*U_xx(x, t) + r*x*U_x(x, t) -"
               "r*U(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " 0 < x < 20 and 0 < t < 1,\n";
  std::cout << " U(0,t) = K*exp(-r*(1-t)) and  U(20,t) = 0 ,0 < t < 1 \n\n";
  std::cout << " U(x,T) = max(0,K-x), x in <0,20> \n\n";
  std::cout << "============================================================\n";

  // typedef the black_sholes_equation_cuda
  typedef black_sholes_equation_cuda<float, boundary_condition_enum::Dirichlet,
                                     std::vector, std::allocator<float>>
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
  // initial condition:
  auto terminal_condition = [&](float x) {
    return std::max<float>(0.0, strike - x);
  };
  // boundary conditions:
  auto const &dirichletRight = [](float t) { return 0.0; };
  auto const &dirichletLeft = [&](float t) {
    return (strike * std::exp(-rate * (maturity - t)));
  };
  auto boundary = std::make_pair(dirichletLeft, dirichletRight);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<float> solution(Sd + 1, float{});
  // initialize solver
  explicit_solver expl_solver(range<float>(0.0, 20.0), maturity, Sd, Td);
  // set boundary conditions:
  expl_solver.set_boundary_condition(boundary);
  // set initial condition:
  expl_solver.set_terminal_condition(terminal_condition);
  // set second order coefficient:
  expl_solver.set_2_order_coefficient(
      [&](float x) { return (0.5 * x * x * sig * sig); });
  // set first order coefficient:
  expl_solver.set_1_order_coefficient([&](float x) { return (rate * x); });
  // set zero order coefficient:
  expl_solver.set_0_order_coefficient([&](float x) { return (-1.0 * rate); });
  // get the solution:
  expl_solver.solve(solution);
  // get exact solution:
  black_scholes_exact<float> bs_exact(0.0, strike, rate, sig, maturity);

  float const h = expl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  float benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = bs_exact.put(j * h);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testExplEuropeanBlackScholesPutDirichletBCDeviceCUDA() {
  std::cout << "============================================================\n";
  std::cout << "======== Explicit Balck-Scholes Equation (Dirichlet BC) ====\n";
  std::cout << "============================================================\n";

  testExplEuropeanBlackScholesPutOptionBCDoubleEuler();
  testExplEuropeanBlackScholesPutOptionBCFloatEuler();

  std::cout << "============================================================\n";
}

#endif  //_LSS_ONE_DIM_BLACK_SCHOLES_EQUATION_CUDA_T
