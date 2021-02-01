#pragma once
#if !defined( \
    _LSS_ONE_DIM_SPACE_VARIABLE_ADVECTION_DIFFUSION_EQUATION_SOLVERS_CUDA_T)
#define _LSS_ONE_DIM_SPACE_VARIABLE_ADVECTION_DIFFUSION_EQUATION_SOLVERS_CUDA_T

#pragma warning(disable : 4305)

#include "common/lss_enumerations.h"
#include "common/lss_utility.h"
#include "pde_solvers/one_dim/lss_one_dim_pde_utility.h"
#include "pde_solvers/one_dim/variable_coefficients/lss_one_dim_space_variable_general_heat_equation_solvers_cuda.h"

#define PI 3.14159

namespace advection_equation {
// //////////////////////////////////////////////////////////////////////////////
//					 ADVECTION-DIFFUSION PROBLEMS ON CUDA
// //////////////////////////////////////////////////////////////////////////////

// =============================================================================
// =============================== IMPLICIT SOLVERS ============================
// =============================================================================

// =============================================================================
// ===== Advection-Diffusion problem with homogeneous boundary conditions ======
// =============================================================================

void testImplAdvDiffEquationDoubleDirichletBCDeviceEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_enumerations::memory_space_enum;
  using lss_one_dim_space_variable_pde_solvers_cuda::implicit_solvers::
      general_heat_equation_cuda;
  using lss_sparse_solvers::real_sparse_solver_cuda;
  using lss_utility::range;

  std::cout << "===========================================================\n";
  std::cout << "Solving Boundary-value Advection-Diffusion equation: \n\n";
  std::cout << " Using CUDA solvers algorithm with \n";
  std::cout << " implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(double).name() << "\n\n";
  std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = 1, x in <0,1> \n\n";
  std::cout << "===========================================================\n";

  // typedef the general_heat_equation_cuda
  typedef general_heat_equation_cuda<
      double, boundary_condition_enum::Dirichlet, memory_space_enum::Device,
      real_sparse_solver_cuda, std::vector, std::allocator<double>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initial_condition = [](double x) { return 1.0; };
  // boundary conditions:
  auto const &dirichlet = [](double x) { return 0.0; };
  auto boundary = std::make_pair(dirichlet, dirichlet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<double> solution(Sd + 1, 0.0);
  // initialize solver
  implicit_solver impl_solver(range<double>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set_2_order_coefficient([](double x) { return 1.0; });
  // set convection term:
  impl_solver.set_1_order_coefficient([](double x) { return -1.0; });
  // se zero-order term:
  impl_solver.set_0_order_coefficient([](double x) { return 0.0; });
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::Euler);
  // get exact solution:
  auto exact = [](double x, double t, std::size_t n) {
    double const first = 2.0 / PI;
    double const exp_0p5x = std::exp(0.5 * x);
    double const exp_m0p5 = std::exp(-0.5);
    double np_sqr{};
    double sum{};
    double num{}, den{}, var{};
    double lambda{};
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

  double const h = impl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  double benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 40);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplAdvDiffEquationFloatDirichletBCDeviceEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_enumerations::memory_space_enum;
  using lss_one_dim_space_variable_pde_solvers_cuda::implicit_solvers::
      general_heat_equation_cuda;
  using lss_sparse_solvers::real_sparse_solver_cuda;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using CUDA solvers algorithm with \n";
  std::cout << " implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(float).name() << "\n\n";
  std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = 1, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the general_heat_equation_cuda
  typedef general_heat_equation_cuda<
      float, boundary_condition_enum::Dirichlet, memory_space_enum::Device,
      real_sparse_solver_cuda, std::vector, std::allocator<float>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initial_condition = [](float x) { return 1.0; };
  // boundary conditions:
  auto const &dirichlet = [](float x) { return 0.0; };
  auto boundary = std::make_pair(dirichlet, dirichlet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<float> solution(Sd + 1, 0.0f);
  // initialize solver
  implicit_solver impl_solver(range<float>(0.0f, 1.0f), 0.1f, Sd, Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set_2_order_coefficient([](double x) { return 1.0; });
  // set convection term:
  impl_solver.set_1_order_coefficient([](double x) { return -1.0; });
  // se zero-order term:
  impl_solver.set_0_order_coefficient([](double x) { return 0.0; });
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::Euler);
  // get exact solution:
  auto exact = [](float x, float t, std::size_t n) {
    float const first = 2.0f / PI;
    float const exp_0p5x = std::exp(0.5f * x);
    float const exp_m0p5 = std::exp(-0.5f);
    float np_sqr{};
    float sum{};
    float num{}, den{}, var{};
    float lambda{};
    for (std::size_t i = 1; i <= n; ++i) {
      np_sqr = (i * i * PI * PI);
      lambda = 0.25f + np_sqr;
      num = (1.0f - std::pow(-1.0f, i) * exp_m0p5) * exp_0p5x *
            std::exp(-1.0f * lambda * t) * std::sin(i * PI * x);
      den = i * (1.0f + (0.25f / np_sqr));
      var = num / den;
      sum += var;
    }
    return (first * sum);
  };

  float const h = impl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  float benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1f, 40);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplAdvDiffEquationDoubleDirichletBCDeviceCN() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_enumerations::memory_space_enum;
  using lss_one_dim_space_variable_pde_solvers_cuda::implicit_solvers::
      general_heat_equation_cuda;
  using lss_sparse_solvers::real_sparse_solver_cuda;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << " Solving Boundary-value Advection-Diffusion equation: \n\n";
  std::cout << " Using CUDA solvers algorithm with \n";
  std::cout << " implicit Clark-Nicolson method\n\n";
  std::cout << " Value type: " << typeid(double).name() << "\n\n";
  std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = 1, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the general_heat_equation_cuda
  typedef general_heat_equation_cuda<
      double, boundary_condition_enum::Dirichlet, memory_space_enum::Device,
      real_sparse_solver_cuda, std::vector, std::allocator<double>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initial_condition = [](double x) { return 1.0; };
  // boundary conditions:
  auto const &dirichlet = [](double x) { return 0.0; };
  auto boundary = std::make_pair(dirichlet, dirichlet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<double> solution(Sd + 1, 0.0);
  // initialize solver
  implicit_solver impl_solver(range<double>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set_2_order_coefficient([](double x) { return 1.0; });
  // set convection term:
  impl_solver.set_1_order_coefficient([](double x) { return -1.0; });
  // se zero-order term:
  impl_solver.set_0_order_coefficient([](double x) { return 0.0; });
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::CrankNicolson);
  // get exact solution:
  auto exact = [](double x, double t, std::size_t n) {
    double const first = 2.0 / PI;
    double const exp_0p5x = std::exp(0.5 * x);
    double const exp_m0p5 = std::exp(-0.5);
    double np_sqr{};
    double sum{};
    double num{}, den{}, var{};
    double lambda{};
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

  double const h = impl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  double benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 40);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplAdvDiffEquationFloatDirichletBCDeviceCN() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_enumerations::memory_space_enum;
  using lss_one_dim_space_variable_pde_solvers_cuda::implicit_solvers::
      general_heat_equation_cuda;
  using lss_sparse_solvers::real_sparse_solver_cuda;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Advection-Diffusion equation: \n\n";
  std::cout << " Using CUDA solvers algorithm with implicit Clark-Nicolson \n"
               "method\n\n";
  std::cout << " Value type: " << typeid(float).name() << "\n\n";
  std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = 1, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the general_heat_equation_cuda
  typedef general_heat_equation_cuda<
      float, boundary_condition_enum::Dirichlet, memory_space_enum::Device,
      real_sparse_solver_cuda, std::vector, std::allocator<float>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 1000;
  // number of time subdivisions:
  std::size_t const Td = 1000;
  // initial condition:
  auto initial_condition = [](float x) { return 1.0f; };
  // boundary conditions:
  auto const &dirichlet = [](float x) { return 0.0; };
  auto boundary = std::make_pair(dirichlet, dirichlet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<float> solution(Sd + 1, 0.0f);
  // initialize solver
  implicit_solver impl_solver(range<float>(0.0f, 1.0f), 0.1f, Sd, Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set_2_order_coefficient([](float x) { return 1.0; });
  // set convection term:
  impl_solver.set_1_order_coefficient([](float x) { return -1.0; });
  // se zero-order term:
  impl_solver.set_0_order_coefficient([](float x) { return 0.0; });
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::CrankNicolson);
  // get exact solution:
  auto exact = [](float x, float t, std::size_t n) {
    float const first = 2.0f / PI;
    float const exp_0p5x = std::exp(0.5f * x);
    float const exp_m0p5 = std::exp(-0.5f);
    float np_sqr{};
    float sum{};
    float num{}, den{}, var{};
    float lambda{};
    for (std::size_t i = 1; i <= n; ++i) {
      np_sqr = (i * i * PI * PI);
      lambda = 0.25f + np_sqr;
      num = (1.0f - std::pow(-1.0f, i) * exp_m0p5) * exp_0p5x *
            std::exp(-1.0f * lambda * t) * std::sin(i * PI * x);
      den = i * (1.0f + (0.25f / np_sqr));
      var = num / den;
      sum += var;
    }
    return (first * sum);
  };

  float const h = impl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  float benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1f, 40);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

// ===========================================================================
// == Advection Diffusion problem with homogeneous Robin boundary conditions =
// ===========================================================================

void testImplAdvDiffEquationDoubleRobinBCDeviceEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_enumerations::memory_space_enum;
  using lss_one_dim_pde_utility::robin_boundary;
  using lss_one_dim_space_variable_pde_solvers_cuda::implicit_solvers::
      general_heat_equation_cuda;
  using lss_sparse_solvers::real_sparse_solver_cuda;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << " Solving Boundary-value Advection-Diffusion equation: \n\n";
  std::cout << " Using CUDA solvers algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(double).name() << "\n\n";
  std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = exp(0.5*x), x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the general_heat_equation_cuda
  typedef general_heat_equation_cuda<
      double, boundary_condition_enum::Robin, memory_space_enum::Device,
      real_sparse_solver_cuda, std::vector, std::allocator<double>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 150;
  // number of time subdivisions:
  std::size_t const Td = 150;
  // initial condition:
  auto initial_condition = [](double x) { return std::exp(0.5 * x); };
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<double> solution(Sd + 1, 0.0);
  // initialize solver
  implicit_solver impl_solver(range<double>(0.0, 1.0), 0.1, Sd, Td);
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
      robin_boundary<double>(left_boundary, right_boundary));
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set_2_order_coefficient([](double x) { return 1.0; });
  // set convection term:
  impl_solver.set_1_order_coefficient([](double x) { return -1.0; });
  // se zero-order term:
  impl_solver.set_0_order_coefficient([](double x) { return 0.0; });
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::Euler);
  // get exact solution:
  auto exact = [](double x, double t, std::size_t n) {
    double const zero = 2.0 * (1.0 - std::exp(-0.5)) / (1.0 - std::exp(-1.0));
    double const first = 2.0;
    double const exp_0p5x = std::exp(0.5 * x);
    double exp_lamt{};
    double sum{};
    double num{}, den{}, var{};
    double lambda_n{};
    double delta_n{};

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

  double const h = impl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  double benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplAdvDiffEquationFloatRobinBCDeviceEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_enumerations::memory_space_enum;
  using lss_one_dim_pde_utility::robin_boundary;
  using lss_one_dim_space_variable_pde_solvers_cuda::implicit_solvers::
      general_heat_equation_cuda;
  using lss_sparse_solvers::real_sparse_solver_cuda;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Advection-Diffusion equation: \n\n";
  std::cout << " Using CUDA solvers algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(float).name() << "\n\n";
  std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = exp(0.5*x), x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the general_heat_equation_cuda
  typedef general_heat_equation_cuda<
      float, boundary_condition_enum::Robin, memory_space_enum::Device,
      real_sparse_solver_cuda, std::vector, std::allocator<float>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 400;
  // number of time subdivisions:
  std::size_t const Td = 150;
  // initial condition:
  auto initial_condition = [](float x) { return std::exp(0.5 * x); };
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<float> solution(Sd + 1, 0.0);
  // initialize solver
  implicit_solver impl_solver(range<float>(0.0, 1.0), 0.1f, Sd, Td);
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
      robin_boundary<float>(left_boundary, right_boundary));
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set_2_order_coefficient([](float x) { return 1.0; });
  // set convection term:
  impl_solver.set_1_order_coefficient([](float x) { return -1.0; });
  // se zero-order term:
  impl_solver.set_0_order_coefficient([](float x) { return 0.0; });
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::Euler);
  // get exact solution:
  auto exact = [](float x, float t, std::size_t n) {
    float const zero =
        2.0f * (1.0f - std::exp(-0.5f)) / (1.0f - std::exp(-1.0f));
    float const first = 2.0f;
    float const exp_0p5x = std::exp(0.5f * x);
    float exp_lamt{};
    float sum{};
    float num{}, den{}, var{};
    float lambda_n{};
    float delta_n{};

    for (std::size_t i = 1; i <= n; ++i) {
      delta_n = i * PI;
      lambda_n = 0.25f + delta_n * delta_n;
      exp_lamt = std::exp(-1.0f * lambda_n * t);
      num = (1.0f - std::pow(-1.0f, i)) * exp_0p5x * exp_lamt *
            (std::sin(delta_n * x) - 2.0f * delta_n * std::cos(delta_n * x));
      den = delta_n * (1.0f + 4.0f * delta_n * delta_n);
      var = num / den;
      sum += var;
    }
    return (zero + first * sum);
  };

  float const h = impl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  float benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1f, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplAdvDiffEquationDoubleRobinBCDeviceCN() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_enumerations::memory_space_enum;
  using lss_one_dim_pde_utility::robin_boundary;
  using lss_one_dim_space_variable_pde_solvers_cuda::implicit_solvers::
      general_heat_equation_cuda;
  using lss_sparse_solvers::real_sparse_solver_cuda;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Advection Diffusion equation: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Crank-Nicolson "
               "method\n\n";
  std::cout << " Value type: " << typeid(double).name() << "\n\n";
  std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = exp(0.5*x), x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the general_heat_equation_cuda
  typedef general_heat_equation_cuda<
      double, boundary_condition_enum::Robin, memory_space_enum::Device,
      real_sparse_solver_cuda, std::vector, std::allocator<double>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initial_condition = [](double x) { return std::exp(0.5 * x); };
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<double> solution(Sd + 1, 0.0);
  // initialize solver
  implicit_solver impl_solver(range<double>(0.0, 1.0), 0.1, Sd, Td);
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
      robin_boundary<double>(left_boundary, right_boundary));
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set_2_order_coefficient([](double x) { return 1.0; });
  // set convection term:
  impl_solver.set_1_order_coefficient([](double x) { return -1.0; });
  // se zero-order term:
  impl_solver.set_0_order_coefficient([](double x) { return 0.0; });
  // get the solution:
  impl_solver.solve(solution);
  // get exact solution:
  // get exact solution:
  auto exact = [](double x, double t, std::size_t n) {
    double const zero = 2.0 * (1.0 - std::exp(-0.5)) / (1.0 - std::exp(-1.0));
    double const first = 2.0;
    double const exp_0p5x = std::exp(0.5 * x);
    double exp_lamt{};
    double sum{};
    double num{}, den{}, var{};
    double lambda_n{};
    double delta_n{};

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

  double const h = impl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  double benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplAdvDiffEquationFloatRobinBCDeviceCN() {
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_enumerations::memory_space_enum;
  using lss_one_dim_pde_utility::robin_boundary;
  using lss_one_dim_space_variable_pde_solvers_cuda::implicit_solvers::
      general_heat_equation_cuda;
  using lss_sparse_solvers::real_sparse_solver_cuda;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Advection Diffusion equation: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Crank-Nicolson \n"
               "method\n\n";
  std::cout << " Value type: " << typeid(float).name() << "\n\n";
  std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = exp(0.5*x), x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the general_heat_equation_cuda
  typedef general_heat_equation_cuda<
      float, boundary_condition_enum::Robin, memory_space_enum::Device,
      real_sparse_solver_cuda, std::vector, std::allocator<float>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initial_condition = [](float x) { return std::exp(0.5 * x); };
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<float> solution(Sd + 1, 0.0);
  // initialize solver
  implicit_solver impl_solver(range<float>(0.0, 1.0), 0.1f, Sd, Td);
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
      robin_boundary<float>(left_boundary, right_boundary));
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set_2_order_coefficient([](float x) { return 1.0; });
  // set convection term:
  impl_solver.set_1_order_coefficient([](float x) { return -1.0; });
  // se zero-order term:
  impl_solver.set_0_order_coefficient([](float x) { return 0.0; });
  // get the solution:
  impl_solver.solve(solution);
  // get exact solution:
  auto exact = [](float x, float t, std::size_t n) {
    float const zero =
        2.0f * (1.0f - std::exp(-0.5f)) / (1.0f - std::exp(-1.0f));
    float const first = 2.0f;
    float const exp_0p5x = std::exp(0.5f * x);
    float exp_lamt{};
    float sum{};
    float num{}, den{}, var{};
    float lambda_n{};
    float delta_n{};

    for (std::size_t i = 1; i <= n; ++i) {
      delta_n = i * PI;
      lambda_n = 0.25f + delta_n * delta_n;
      exp_lamt = std::exp(-1.0f * lambda_n * t);
      num = (1.0f - std::pow(-1.0f, i)) * exp_0p5x * exp_lamt *
            (std::sin(delta_n * x) - 2.0f * delta_n * std::cos(delta_n * x));
      den = delta_n * (1.0f + 4.0f * delta_n * delta_n);
      var = num / den;
      sum += var;
    }
    return (zero + first * sum);
  };

  float const h = impl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  float benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1f, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

// ===========================================================================
// ======================= EXPLICIT SOLVERS ==================================
// ===========================================================================

// ===========================================================================
// ===== Advection-Diffusion problem with homogeneous boundary conditions ====
// ===========================================================================

void testExplAdvDiffEquationDoubleDirichletBCEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_one_dim_space_variable_pde_solvers_cuda::explicit_solvers::
      general_heat_equation_cuda;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Advection-Diffusion equation: \n\n";
  std::cout << " Using CUDA solvers algorithm with explicit Euler method\n\n";
  std::cout << " Value type: " << typeid(double).name() << "\n\n";
  std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = 1, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the general_heat_equation_cuda
  typedef general_heat_equation_cuda<double, boundary_condition_enum::Dirichlet,
                                     std::vector, std::allocator<double>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 10000;
  // initial condition:
  auto initial_condition = [](double x) { return 1.0; };
  // boundary conditions:
  auto const &dirichlet = [](double x) { return 0.0; };
  auto boundary = std::make_pair(dirichlet, dirichlet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<double> solution(Sd + 1, 0.0);
  // initialize solver
  explicit_solver expl_solver(range<double>(0.0, 1.0), 0.2, Sd, Td);
  // set boundary conditions:
  expl_solver.set_boundary_condition(boundary);
  // set initial condition:
  expl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set_2_order_coefficient([](double x) { return 1.0; });
  // set convection term:
  expl_solver.set_1_order_coefficient([](double x) { return -1.0; });
  // set zero-order term:
  expl_solver.set_0_order_coefficient([](double x) { return 0.0; });
  // get the solution:
  expl_solver.solve(solution);
  // get exact solution:
  auto exact = [](double x, double t, std::size_t n) {
    double const first = 2.0 / PI;
    double const exp_0p5x = std::exp(0.5 * x);
    double const exp_m0p5 = std::exp(-0.5);
    double np_sqr{};
    double sum{};
    double num{}, den{}, var{};
    double lambda{};
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

  double const h = expl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  double benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.2, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testExplAdvDiffEquationFloatDirichletBCEuler() {
  using lss_enumerations::boundary_condition_enum;
  using lss_one_dim_space_variable_pde_solvers_cuda::explicit_solvers::
      general_heat_equation_cuda;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using CUDA solvers algorithm with explicit Euler method\n\n";
  std::cout << " Value type: " << typeid(float).name() << "\n\n";
  std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = 1, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the general_heat_equation_cuda
  typedef general_heat_equation_cuda<float, boundary_condition_enum::Dirichlet,
                                     std::vector, std::allocator<float>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 10000;
  // initial condition:
  auto initial_condition = [](float x) { return 1.0f; };
  // boundary conditions:
  auto const &dirichlet = [](float x) { return 0.0; };
  auto boundary = std::make_pair(dirichlet, dirichlet);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<float> solution(Sd + 1, 0.0f);
  // initialize solver
  explicit_solver expl_solver(range<float>(0.0f, 1.0f), 0.2f, Sd, Td);
  // set boundary conditions:
  expl_solver.set_boundary_condition(boundary);
  // set initial condition:
  expl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set_2_order_coefficient([](float x) { return 1.0; });
  // set convection term:
  expl_solver.set_1_order_coefficient([](float x) { return -1.0; });
  // set zero-order term:
  expl_solver.set_0_order_coefficient([](float x) { return 0.0; });
  // get the solution:
  expl_solver.solve(solution);
  // get exact solution:
  auto exact = [](float x, float t, std::size_t n) {
    float const first = 2.0f / PI;
    float const exp_0p5x = std::exp(0.5f * x);
    float const exp_m0p5 = std::exp(-0.5f);
    float np_sqr{};
    float sum{};
    float num{}, den{}, var{};
    float lambda{};
    for (std::size_t i = 1; i <= n; ++i) {
      np_sqr = (i * i * PI * PI);
      lambda = 0.25f + np_sqr;
      num = (1.0f - std::pow(-1.0f, i) * exp_m0p5) * exp_0p5x *
            std::exp(-1.0f * lambda * t) * std::sin(i * PI * x);
      den = i * (1.0f + (0.25f / np_sqr));
      var = num / den;
      sum += var;
    }
    return (first * sum);
  };

  float const h = expl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  float benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.2f, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

// ===========================================================================
// == Advection Diffusion problem with homogeneous Robin boundary conditions =
// ===========================================================================

void testExplAdvDiffEquationDoubleRobinBC() {
  using lss_enumerations::boundary_condition_enum;
  using lss_one_dim_pde_utility::robin_boundary;
  using lss_one_dim_space_variable_pde_solvers_cuda::explicit_solvers::
      general_heat_equation_cuda;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Advection Diffusion equation: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Crank-Nicolson \n"
               "method\n\n";
  std::cout << " Value type: " << typeid(double).name() << "\n\n";
  std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = exp(0.5*x), x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the general_heat_equation_cuda
  typedef general_heat_equation_cuda<double, boundary_condition_enum::Robin,
                                     std::vector, std::allocator<double>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 10000;
  // initial condition:
  auto initial_condition = [](double x) { return std::exp(0.5 * x); };
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<double> solution(Sd + 1, 0.0);
  // initialize solver
  explicit_solver expl_solver(range<double>(0.0, 1.0), 0.5, Sd, Td);
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
  expl_solver.set_boundary_condition(
      robin_boundary<double>(left_boundary, right_boundary));
  // set initial condition:
  expl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set_2_order_coefficient([](double x) { return 1.0; });
  // set convection term:
  expl_solver.set_1_order_coefficient([](double x) { return -1.0; });
  // set zero-order term:
  expl_solver.set_0_order_coefficient([](double x) { return 0.0; });
  // get the solution:
  expl_solver.solve(solution);
  // get exact solution:
  // get exact solution:
  auto exact = [](double x, double t, std::size_t n) {
    double const zero = 2.0 * (1.0 - std::exp(-0.5)) / (1.0 - std::exp(-1.0));
    double const first = 2.0;
    double const exp_0p5x = std::exp(0.5 * x);
    double exp_lamt{};
    double sum{};
    double num{}, den{}, var{};
    double lambda_n{};
    double delta_n{};

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

  double const h = expl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  double benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.5, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testExplAdvDiffEquationFloatRobinBC() {
  using lss_enumerations::boundary_condition_enum;
  using lss_one_dim_pde_utility::robin_boundary;
  using lss_one_dim_space_variable_pde_solvers_cuda::explicit_solvers::
      general_heat_equation_cuda;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Advection Diffusion equation: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Crank-Nicolson \n"
               "method\n\n";
  std::cout << " Value type: " << typeid(float).name() << "\n\n";
  std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = exp(0.5*x), x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the general_heat_equation_cuda
  typedef general_heat_equation_cuda<float, boundary_condition_enum::Robin,
                                     std::vector, std::allocator<float>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 10000;
  // initial condition:
  auto initial_condition = [](float x) { return std::exp(0.5 * x); };
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<float> solution(Sd + 1, 0.0);
  // initialize solver
  explicit_solver expl_solver(range<float>(0.0, 1.0), 0.5, Sd, Td);
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
  expl_solver.set_boundary_condition(
      robin_boundary<float>(left_boundary, right_boundary));
  // set initial condition:
  expl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set_2_order_coefficient([](float x) { return 1.0; });
  // set convection term:
  expl_solver.set_1_order_coefficient([](float x) { return -1.0; });
  // set zero-order term:
  expl_solver.set_0_order_coefficient([](float x) { return 0.0; });
  // get the solution:
  expl_solver.solve(solution);
  // get exact solution:
  auto exact = [](float x, float t, std::size_t n) {
    float const zero =
        2.0f * (1.0f - std::exp(-0.5f)) / (1.0f - std::exp(-1.0f));
    float const first = 2.0f;
    float const exp_0p5x = std::exp(0.5f * x);
    float exp_lamt{};
    float sum{};
    float num{}, den{}, var{};
    float lambda_n{};
    float delta_n{};

    for (std::size_t i = 1; i <= n; ++i) {
      delta_n = i * PI;
      lambda_n = 0.25f + delta_n * delta_n;
      exp_lamt = std::exp(-1.0f * lambda_n * t);
      num = (1.0f - std::pow(-1.0f, i)) * exp_0p5x * exp_lamt *
            (std::sin(delta_n * x) - 2.0f * delta_n * std::cos(delta_n * x));
      den = delta_n * (1.0f + 4.0f * delta_n * delta_n);
      var = num / den;
      sum += var;
    }
    return (zero + first * sum);
  };

  float const h = expl_solver.space_step();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  float benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.5f, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

}  // namespace advection_equation

void testImplSpaceVarAdvDiffEquationDirichletBCDeviceCUDA() {
  std::cout << "============================================================\n";
  std::cout << "==== Implicit Advection-Diffusion Equation (Dirichlet BC) ==\n";
  std::cout << "============================================================\n";

  advection_equation::testImplAdvDiffEquationDoubleDirichletBCDeviceEuler();
  advection_equation::testImplAdvDiffEquationFloatDirichletBCDeviceEuler();
  advection_equation::testImplAdvDiffEquationDoubleDirichletBCDeviceCN();
  advection_equation::testImplAdvDiffEquationFloatDirichletBCDeviceCN();

  std::cout << "============================================================\n";
}

void testImplSpaceVarAdvDiffEquationRobinBCDeviceCUDA() {
  std::cout << "============================================================\n";
  std::cout << "======= Implicit Advection Diffusion Equation (Robin BC) ===\n";
  std::cout << "============================================================\n";

  advection_equation::testImplAdvDiffEquationDoubleRobinBCDeviceEuler();
  advection_equation::testImplAdvDiffEquationFloatRobinBCDeviceEuler();
  advection_equation::testImplAdvDiffEquationDoubleRobinBCDeviceCN();
  advection_equation::testImplAdvDiffEquationFloatRobinBCDeviceCN();

  std::cout << "============================================================\n";
}

void testExplSpaceVarAdvDiffEquationDirichletBCCUDA() {
  std::cout << "============================================================\n";
  std::cout << "=== Explicit Advection Diffusion Equation (Dirichlet BC) ===\n";
  std::cout << "============================================================\n";

  advection_equation::testExplAdvDiffEquationDoubleDirichletBCEuler();
  advection_equation::testExplAdvDiffEquationFloatDirichletBCEuler();

  std::cout << "============================================================\n";
}

void testExplSpaceVarAdvDiffEquationRobinBCCUDA() {
  std::cout << "============================================================\n";
  std::cout << "===== Explicit Advection Diffusion Equation (Robin BC) =====\n";
  std::cout << "============================================================\n";

  advection_equation::testExplAdvDiffEquationDoubleRobinBC();
  advection_equation::testExplAdvDiffEquationFloatRobinBC();

  std::cout << "============================================================\n";
}

#endif  ///_LSS_ONE_DIM_SPACE_VARIABLE_ADVECTION_DIFFUSION_EQUATION_SOLVERS_CUDA_T
