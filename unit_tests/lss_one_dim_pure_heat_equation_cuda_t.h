#pragma once
#if !defined(_LSS_ONE_DIM_PURE_HEAT_EQUATION_CUDA_T)
#define _LSS_ONE_DIM_PURE_HEAT_EQUATION_CUDA_T

#pragma warning(disable : 4305)

#include "common/lss_types.h"
#include "common/lss_utility.h"
#include "pde_solvers/one_dim/classic/lss_one_dim_general_heat_equation_solvers_cuda.h"

#define PI 3.14159

// ////////////////////////////////////////////////////////////////////////////
//						PURE HEAT PROBLEMS ON CUDA
// ////////////////////////////////////////////////////////////////////////////

// ============================================================================
// ============================ IMPLICIT SOLVERS ==============================
// ============================================================================

// ============================================================================
// =============== Heat problem with homogeneous boundary conditions ==========
// ============================================================================

void testImplPureHeatEquationDoubleDirichletBCDeviceEuler() {
  using lss_one_dim_general_heat_equation_solvers_cuda::implicit_solvers::
      Implicit1DGeneralHeatEquationCUDA;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_types::MemorySpace;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using CUDA solvers algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(double).name() << "\n\n";
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
  typedef Implicit1DGeneralHeatEquationCUDA<
      double, BoundaryConditionType::Dirichlet, MemorySpace::Device,
      RealSparseSolverCUDA, std::vector, std::allocator<double>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](double x) { return x; };
  // boundary conditions:
  auto boundary = std::make_pair(0.0, 0.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<double> solution(Sd + 1, 0.0);
  // initialize solver
  implicit_solver impl_solver(Range<double>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::Euler);
  // get exact solution:
  auto exact = [](double x, double t, std::size_t n) {
    double const first = 2.0 / PI;
    double sum{};
    double var1{};
    double var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var1 = std::pow(-1.0, i + 1) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
      var2 = std::sin(i * PI * x) / i;
      sum += (var1 * var2);
    }
    return (first * sum);
  };

  double const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  double benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplPureHeatEquationFloatDirichletBCDeviceEuler() {
  using lss_one_dim_general_heat_equation_solvers_cuda::implicit_solvers::
      Implicit1DGeneralHeatEquationCUDA;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_types::MemorySpace;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using CUDA solvers algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(float).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) \n"
               "*sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquationCUDA
  typedef Implicit1DGeneralHeatEquationCUDA<
      float, BoundaryConditionType::Dirichlet, MemorySpace::Device,
      RealSparseSolverCUDA, std::vector, std::allocator<float>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](float x) { return x; };
  // boundary conditions:
  auto boundary = std::make_pair(0.0, 0.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<float> solution(Sd + 1, 0.0);
  // initialize solver
  implicit_solver impl_solver(Range<float>(0.0f, 1.0f), 0.1f, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::Euler);
  // get exact solution:
  auto exact = [](float x, float t, std::size_t n) {
    float const first = 2.0f / PI;
    float sum{};
    float var1{};
    float var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var1 = std::pow(-1.0f, i + 1) * std::exp(-1.0f * (i * PI) * (i * PI) * t);
      var2 = std::sin(i * PI * x) / i;
      sum += (var1 * var2);
    }
    return (first * sum);
  };

  float const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  float benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1f, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplPureHeatEquationDoubleDirichletBCDeviceCN() {
  using lss_one_dim_general_heat_equation_solvers_cuda::implicit_solvers::
      Implicit1DGeneralHeatEquationCUDA;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_types::MemorySpace;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using CUDA solvers algorithm with implicit Clark-Nicolson "
               "method\n\n";
  std::cout << " Value type: " << typeid(double).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) \n"
               "*sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquationCUDA
  typedef Implicit1DGeneralHeatEquationCUDA<
      double, BoundaryConditionType::Dirichlet, MemorySpace::Device,
      RealSparseSolverCUDA, std::vector, std::allocator<double>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](double x) { return x; };
  // boundary conditions:
  auto boundary = std::make_pair(0.0, 0.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<double> solution(Sd + 1, 0.0);
  // initialize solver
  implicit_solver impl_solver(Range<double>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::CrankNicolson);
  // get exact solution:
  auto exact = [](double x, double t, std::size_t n) {
    double const first = 2.0 / PI;
    double sum{};
    double var1{};
    double var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var1 = std::pow(-1.0, i + 1) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
      var2 = std::sin(i * PI * x) / i;
      sum += (var1 * var2);
    }
    return (first * sum);
  };

  double const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  double benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplPureHeatEquationFloatDirichletBCDeviceCN() {
  using lss_one_dim_general_heat_equation_solvers_cuda::implicit_solvers::
      Implicit1DGeneralHeatEquationCUDA;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_types::MemorySpace;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using CUDA solvers algorithm with implicit Clark-Nicolson "
               "method\n\n";
  std::cout << " Value type: " << typeid(float).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) \n"
               "*sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquationCUDA
  typedef Implicit1DGeneralHeatEquationCUDA<
      float, BoundaryConditionType::Dirichlet, MemorySpace::Device,
      RealSparseSolverCUDA, std::vector, std::allocator<float>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 1000;
  // number of time subdivisions:
  std::size_t const Td = 1000;
  // initial condition:
  auto initialCondition = [](float x) { return x; };
  // boundary conditions:
  auto boundary = std::make_pair(0.0, 0.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<float> solution(Sd + 1, 0.0);
  // initialize solver
  implicit_solver impl_solver(Range<float>(0.0f, 1.0f), 0.1f, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::CrankNicolson);
  // get exact solution:
  auto exact = [](float x, float t, std::size_t n) {
    float const first = 2.0f / PI;
    float sum{};
    float var1{};
    float var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var1 = std::pow(-1.0f, i + 1) * std::exp(-1.0f * (i * PI) * (i * PI) * t);
      var2 = std::sin(i * PI * x) / i;
      sum += (var1 * var2);
    }
    return (first * sum);
  };

  float const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  float benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1f, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplPureHeatEquationDirichletBCDeviceCUDA() {
  std::cout << "============================================================\n";
  std::cout << "===== Implicit Pure Heat Equation (Dirichlet BC) ===========\n";
  std::cout << "============================================================\n";

  testImplPureHeatEquationDoubleDirichletBCDeviceEuler();
  testImplPureHeatEquationFloatDirichletBCDeviceEuler();
  testImplPureHeatEquationDoubleDirichletBCDeviceCN();
  testImplPureHeatEquationFloatDirichletBCDeviceCN();

  std::cout << "============================================================\n";
}

void testImplPureHeatEquationDoubleDirichletBCHostEuler() {
  using lss_one_dim_general_heat_equation_solvers_cuda::implicit_solvers::
      Implicit1DGeneralHeatEquationCUDA;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_types::MemorySpace;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using CUDA solvers algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(double).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) \n"
               "*sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquationCUDA
  typedef Implicit1DGeneralHeatEquationCUDA<
      double, BoundaryConditionType::Dirichlet, MemorySpace::Host,
      RealSparseSolverCUDA, std::vector, std::allocator<double>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](double x) { return x; };
  // boundary conditions:
  auto boundary = std::make_pair(0.0, 0.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<double> solution(Sd + 1, 0.0);
  // initialize solver
  implicit_solver impl_solver(Range<double>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::Euler);
  // get exact solution:
  auto exact = [](double x, double t, std::size_t n) {
    double const first = 2.0 / PI;
    double sum{};
    double var1{};
    double var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var1 = std::pow(-1.0, i + 1) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
      var2 = std::sin(i * PI * x) / i;
      sum += (var1 * var2);
    }
    return (first * sum);
  };

  double const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  double benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplPureHeatEquationFloatDirichletBCHostEuler() {
  using lss_one_dim_general_heat_equation_solvers_cuda::implicit_solvers::
      Implicit1DGeneralHeatEquationCUDA;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_types::MemorySpace;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using CUDA solvers algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(float).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) \n"
               "*sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquationCUDA
  typedef Implicit1DGeneralHeatEquationCUDA<
      float, BoundaryConditionType::Dirichlet, MemorySpace::Host,
      RealSparseSolverCUDA, std::vector, std::allocator<float>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](float x) { return x; };
  // boundary conditions:
  auto boundary = std::make_pair(0.0, 0.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<float> solution(Sd + 1, 0.0);
  // initialize solver
  implicit_solver impl_solver(Range<float>(0.0f, 1.0f), 0.1f, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::Euler);
  // get exact solution:
  auto exact = [](float x, float t, std::size_t n) {
    float const first = 2.0f / PI;
    float sum{};
    float var1{};
    float var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var1 = std::pow(-1.0f, i + 1) * std::exp(-1.0f * (i * PI) * (i * PI) * t);
      var2 = std::sin(i * PI * x) / i;
      sum += (var1 * var2);
    }
    return (first * sum);
  };

  float const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  float benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1f, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplPureHeatEquationDoubleDirichletBCHostCN() {
  using lss_one_dim_general_heat_equation_solvers_cuda::implicit_solvers::
      Implicit1DGeneralHeatEquationCUDA;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_types::MemorySpace;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using CUDA solvers algorithm with implicit Clark-Nicolson "
               "method\n\n";
  std::cout << " Value type: " << typeid(double).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) \n"
               "*sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquationCUDA
  typedef Implicit1DGeneralHeatEquationCUDA<
      double, BoundaryConditionType::Dirichlet, MemorySpace::Host,
      RealSparseSolverCUDA, std::vector, std::allocator<double>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](double x) { return x; };
  // boundary conditions:
  auto boundary = std::make_pair(0.0, 0.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<double> solution(Sd + 1, 0.0);
  // initialize solver
  implicit_solver impl_solver(Range<double>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::CrankNicolson);
  // get exact solution:
  auto exact = [](double x, double t, std::size_t n) {
    double const first = 2.0 / PI;
    double sum{};
    double var1{};
    double var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var1 = std::pow(-1.0, i + 1) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
      var2 = std::sin(i * PI * x) / i;
      sum += (var1 * var2);
    }
    return (first * sum);
  };

  double const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  double benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplPureHeatEquationFloatDirichletBCHostCN() {
  using lss_one_dim_general_heat_equation_solvers_cuda::implicit_solvers::
      Implicit1DGeneralHeatEquationCUDA;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_types::MemorySpace;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using CUDA solvers algorithm with implicit Clark-Nicolson "
               "method\n\n";
  std::cout << " Value type: " << typeid(float).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) \n"
               "*sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquationCUDA
  typedef Implicit1DGeneralHeatEquationCUDA<
      float, BoundaryConditionType::Dirichlet, MemorySpace::Host,
      RealSparseSolverCUDA, std::vector, std::allocator<float>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 1000;
  // number of time subdivisions:
  std::size_t const Td = 1000;
  // initial condition:
  auto initialCondition = [](float x) { return x; };
  // boundary conditions:
  auto boundary = std::make_pair(0.0, 0.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<float> solution(Sd + 1, 0.0);
  // initialize solver
  implicit_solver impl_solver(Range<float>(0.0f, 1.0f), 0.1f, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::CrankNicolson);
  // get exact solution:
  auto exact = [](float x, float t, std::size_t n) {
    float const first = 2.0f / PI;
    float sum{};
    float var1{};
    float var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var1 = std::pow(-1.0f, i + 1) * std::exp(-1.0f * (i * PI) * (i * PI) * t);
      var2 = std::sin(i * PI * x) / i;
      sum += (var1 * var2);
    }
    return (first * sum);
  };

  float const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  float benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1f, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplPureHeatEquationDirichletBCHostCUDA() {
  std::cout << "============================================================\n";
  std::cout << "========= Implicit Pure Heat Equation (Dirichlet BC) =======\n";
  std::cout << "============================================================\n";

  testImplPureHeatEquationDoubleDirichletBCHostEuler();
  testImplPureHeatEquationFloatDirichletBCHostEuler();
  testImplPureHeatEquationDoubleDirichletBCHostCN();
  testImplPureHeatEquationFloatDirichletBCHostCN();

  std::cout << "============================================================\n";
}

void testImplPureHeatEquationDoubleRobinBCDeviceEuler() {
  using lss_one_dim_general_heat_equation_solvers_cuda::implicit_solvers::
      Implicit1DGeneralHeatEquationCUDA;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_types::MemorySpace;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using CUDA solvers algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(double).name() << "\n\n";
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

  // typedef the Implicit1DGeneralHeatEquationCUDA
  typedef Implicit1DGeneralHeatEquationCUDA<
      double, BoundaryConditionType::Robin, MemorySpace::Device,
      RealSparseSolverCUDA, std::vector, std::allocator<double>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](double x) { return x; };
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<double> solution(Sd + 1, 0.0);
  // initialize solver
  implicit_solver impl_solver(Range<double>(0.0, 1.0), 0.2, Sd, Td);
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
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::Euler);
  // get exact solution:
  auto exact = [](double x, double t, std::size_t n) {
    double const pipi = PI * PI;
    double const first = 4.0 / pipi;
    double sum{};
    double var0{};
    double var1{};
    double var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var0 = (2.0 * i - 1.0);
      var1 = std::exp(-1.0 * pipi * var0 * var0 * t);
      var2 = std::cos(var0 * PI * x) / (var0 * var0);
      sum += (var1 * var2);
    }
    return (0.5 - first * sum);
  };

  double const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  double benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.2, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplPureHeatEquationFloatRobinBCDeviceEuler() {
  using lss_one_dim_general_heat_equation_solvers_cuda::implicit_solvers::
      Implicit1DGeneralHeatEquationCUDA;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_types::MemorySpace;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using CUDA solvers algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(float).name() << "\n\n";
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

  // typedef the Implicit1DGeneralHeatEquationCUDA
  typedef Implicit1DGeneralHeatEquationCUDA<
      float, BoundaryConditionType::Robin, MemorySpace::Device,
      RealSparseSolverCUDA, std::vector, std::allocator<float>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](float x) { return x; };
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<float> solution(Sd + 1, 0.0);
  // initialize solver
  implicit_solver impl_solver(Range<float>(0.0f, 1.0f), 0.2f, Sd, Td);
  // set boundary conditions:
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
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::Euler);
  // get exact solution:
  auto exact = [](float x, float t, std::size_t n) {
    float const pipi = PI * PI;
    float const first = 4.0f / pipi;
    float sum{};
    float var0{};
    float var1{};
    float var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var0 = (2.0f * i - 1.0f);
      var1 = std::exp(-1.0f * pipi * var0 * var0 * t);
      var2 = std::cos(var0 * PI * x) / (var0 * var0);
      sum += (var1 * var2);
    }
    return (0.5f - first * sum);
  };

  float const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  float benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.2f, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplPureHeatEquationDoubleRobinBCDeviceCN() {
  using lss_one_dim_general_heat_equation_solvers_cuda::implicit_solvers::
      Implicit1DGeneralHeatEquationCUDA;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_types::MemorySpace;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using CUDA solvers algorithm with implicit Clark-Nicolson \n"
               "method\n\n";
  std::cout << " Value type: " << typeid(double).name() << "\n\n";
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

  // typedef the Implicit1DGeneralHeatEquationCUDA
  typedef Implicit1DGeneralHeatEquationCUDA<
      double, BoundaryConditionType::Robin, MemorySpace::Device,
      RealSparseSolverCUDA, std::vector, std::allocator<double>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](double x) { return x; };
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<double> solution(Sd + 1, 0.0);
  // initialize solver
  implicit_solver impl_solver(Range<double>(0.0, 1.0), 0.2, Sd, Td);
  // set boundary conditions:
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
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::CrankNicolson);
  // get exact solution:
  auto exact = [](double x, double t, std::size_t n) {
    double const pipi = PI * PI;
    double const first = 4.0 / pipi;
    double sum{};
    double var0{};
    double var1{};
    double var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var0 = (2.0 * i - 1.0);
      var1 = std::exp(-1.0 * pipi * var0 * var0 * t);
      var2 = std::cos(var0 * PI * x) / (var0 * var0);
      sum += (var1 * var2);
    }
    return (0.5 - first * sum);
  };

  double const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  double benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.2, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplPureHeatEquationFloatRobinBCDeviceCN() {
  using lss_one_dim_general_heat_equation_solvers_cuda::implicit_solvers::
      Implicit1DGeneralHeatEquationCUDA;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_types::MemorySpace;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using CUDA solvers algorithm with implicit Clark-Nicolson \n"
               "method\n\n";
  std::cout << " Value type: " << typeid(float).name() << "\n\n";
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

  // typedef the Implicit1DGeneralHeatEquationCUDA
  typedef Implicit1DGeneralHeatEquationCUDA<
      float, BoundaryConditionType::Robin, MemorySpace::Device,
      RealSparseSolverCUDA, std::vector, std::allocator<float>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 1000;
  // number of time subdivisions:
  std::size_t const Td = 1000;
  // initial condition:
  auto initialCondition = [](float x) { return x; };
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<float> solution(Sd + 1, 0.0);
  // initialize solver
  implicit_solver impl_solver(Range<float>(0.0f, 1.0f), 0.2f, Sd, Td);
  // set boundary conditions:
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
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::CrankNicolson);
  // get exact solution:
  auto exact = [](float x, float t, std::size_t n) {
    float const pipi = PI * PI;
    float const first = 4.0f / pipi;
    float sum{};
    float var0{};
    float var1{};
    float var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var0 = (2.0f * i - 1.0f);
      var1 = std::exp(-1.0f * pipi * var0 * var0 * t);
      var2 = std::cos(var0 * PI * x) / (var0 * var0);
      sum += (var1 * var2);
    }
    return (0.5f - first * sum);
  };

  float const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  float benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.2f, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplPureHeatEquationRobinBCDeviceCUDA() {
  std::cout << "============================================================\n";
  std::cout << "========== Implicit Pure Heat Equation (Robin BC) ==========\n";
  std::cout << "============================================================\n";

  testImplPureHeatEquationDoubleRobinBCDeviceEuler();
  testImplPureHeatEquationFloatRobinBCDeviceEuler();
  testImplPureHeatEquationDoubleRobinBCDeviceCN();
  testImplPureHeatEquationFloatRobinBCDeviceCN();

  std::cout << "============================================================\n";
}

// ============================================================================
// ======= Heat problem with homogeneous boundary conditions and source =======
// ============================================================================

void testImplPureHeatEquationSourceFloatDirichletBCDeviceEuler() {
  using lss_one_dim_general_heat_equation_solvers_cuda::implicit_solvers::
      Implicit1DGeneralHeatEquationCUDA;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_types::MemorySpace;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with source: \n\n";
  std::cout << " Using Euler algorithm on CUDA device \n"
               "with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(float).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = 1, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquationCUDA
  typedef Implicit1DGeneralHeatEquationCUDA<
      float, BoundaryConditionType::Dirichlet, MemorySpace::Device,
      RealSparseSolverCUDA, std::vector, std::allocator<float>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](float x) { return 1.0f; };
  // boundary conditions:
  auto boundary = std::make_pair(0.0f, 0.0f);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<float> solution(Sd + 1, 0.0f);
  // initialize solver
  implicit_solver impl_solver(Range<float>(0.0f, 1.0f), 0.1f, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // set heat source:
  impl_solver.setHeatSource([](float x, float t) { return x; });
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::Euler);
  // get exact solution:
  auto exact = [](float x, float t, std::size_t n) {
    float sum{};
    float q_n{};
    float f_n{};
    float lam_n{};
    float lam_2{};
    float var1{};
    for (std::size_t i = 1; i <= n; ++i) {
      q_n = (2.0f / (i * PI)) * std::pow(-1.0f, i + 1);
      f_n = (2.0f / (i * PI)) * (1.0f - std::pow(-1.0f, i));
      lam_n = i * PI;
      lam_2 = lam_n * lam_n;
      var1 =
          (q_n / lam_2 + (f_n - (q_n / lam_2)) * std::exp(-1.0f * lam_2 * t)) *
          std::sin(i * PI * x);
      sum += var1;
    }
    return sum;
  };

  float const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  float benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1f, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplPureHeatEquationSourceDoubleDirichletBCDeviceEuler() {
  using lss_one_dim_general_heat_equation_solvers_cuda::implicit_solvers::
      Implicit1DGeneralHeatEquationCUDA;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_types::MemorySpace;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with source: \n\n";
  std::cout << " Using Euler algorithm on CUDA device \n"
               "with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(double).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = 1, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquationCUDA
  typedef Implicit1DGeneralHeatEquationCUDA<
      double, BoundaryConditionType::Dirichlet, MemorySpace::Device,
      RealSparseSolverCUDA, std::vector, std::allocator<double>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](double x) { return 1.0; };
  // boundary conditions:
  auto boundary = std::make_pair(0.0, 0.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<double> solution(Sd + 1, 0.0);
  // initialize solver
  implicit_solver impl_solver(Range<double>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // set heat source:
  impl_solver.setHeatSource([](double x, double t) { return x; });
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::Euler);
  // get exact solution:
  auto exact = [](double x, double t, std::size_t n) {
    double sum{};
    double q_n{};
    double f_n{};
    double lam_n{};
    double lam_2{};
    double var1{};
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

  double const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  double benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplPureHeatEquationSourceFloatDirichletBCDeviceCN() {
  using lss_one_dim_general_heat_equation_solvers_cuda::implicit_solvers::
      Implicit1DGeneralHeatEquationCUDA;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_types::MemorySpace;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with source: \n\n";
  std::cout << " Using Euler algorithm on CUDA device with implicit \n"
               "Crank-Nicolson method\n\n";
  std::cout << " Value type: " << typeid(float).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = 1, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquationCUDA
  typedef Implicit1DGeneralHeatEquationCUDA<
      float, BoundaryConditionType::Dirichlet, MemorySpace::Device,
      RealSparseSolverCUDA, std::vector, std::allocator<float>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](float x) { return 1.0f; };
  // boundary conditions:
  auto boundary = std::make_pair(0.0f, 0.0f);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<float> solution(Sd + 1, 0.0f);
  // initialize solver
  implicit_solver impl_solver(Range<float>(0.0f, 1.0f), 0.1f, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0f);
  // set heat source:
  impl_solver.setHeatSource([](float x, float t) { return x; });
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::CrankNicolson);
  // get exact solution:
  auto exact = [](float x, float t, std::size_t n) {
    float sum{};
    float q_n{};
    float f_n{};
    float lam_n{};
    float lam_2{};
    float var1{};
    for (std::size_t i = 1; i <= n; ++i) {
      q_n = (2.0f / (i * PI)) * std::pow(-1.0f, i + 1);
      f_n = (2.0f / (i * PI)) * (1.0f - std::pow(-1.0f, i));
      lam_n = i * PI;
      lam_2 = lam_n * lam_n;
      var1 =
          (q_n / lam_2 + (f_n - (q_n / lam_2)) * std::exp(-1.0f * lam_2 * t)) *
          std::sin(i * PI * x);
      sum += var1;
    }
    return sum;
  };

  float const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  float benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1f, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplPureHeatEquationSourceDoubleDirichletBCDeviceCN() {
  using lss_one_dim_general_heat_equation_solvers_cuda::implicit_solvers::
      Implicit1DGeneralHeatEquationCUDA;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_types::MemorySpace;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with source: \n\n";
  std::cout << " Using Euler algorithm on CUDA device with implicit \n"
               "Crank-Nicolson method\n\n";
  std::cout << " Value type: " << typeid(double).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = 1, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquationCUDA
  typedef Implicit1DGeneralHeatEquationCUDA<
      double, BoundaryConditionType::Dirichlet, MemorySpace::Device,
      RealSparseSolverCUDA, std::vector, std::allocator<double>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](double x) { return 1.0; };
  // boundary conditions:
  auto boundary = std::make_pair(0.0f, 0.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<double> solution(Sd + 1, 0.0);
  // initialize solver
  implicit_solver impl_solver(Range<double>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // set heat source:
  impl_solver.setHeatSource([](double x, double t) { return x; });
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::CrankNicolson);
  // get exact solution:
  auto exact = [](double x, double t, std::size_t n) {
    double sum{};
    double q_n{};
    double f_n{};
    double lam_n{};
    double lam_2{};
    double var1{};
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

  double const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  double benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplPureHeatEquationSourceFloatDirichletBCHostEuler() {
  using lss_one_dim_general_heat_equation_solvers_cuda::implicit_solvers::
      Implicit1DGeneralHeatEquationCUDA;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_types::MemorySpace;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with source: \n\n";
  std::cout << " Using Euler algorithm on CUDA host \n"
               "with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(float).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = 1, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquationCUDA
  typedef Implicit1DGeneralHeatEquationCUDA<
      float, BoundaryConditionType::Dirichlet, MemorySpace::Host,
      RealSparseSolverCUDA, std::vector, std::allocator<float>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](float x) { return 1.0f; };
  // boundary conditions:
  auto boundary = std::make_pair(0.0f, 0.0f);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<float> solution(Sd + 1, 0.0f);
  // initialize solver
  implicit_solver impl_solver(Range<float>(0.0f, 1.0f), 0.1f, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0f);
  // set heat source:
  impl_solver.setHeatSource([](float x, float t) { return x; });
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::Euler);
  // get exact solution:
  auto exact = [](float x, float t, std::size_t n) {
    float sum{};
    float q_n{};
    float f_n{};
    float lam_n{};
    float lam_2{};
    float var1{};
    for (std::size_t i = 1; i <= n; ++i) {
      q_n = (2.0f / (i * PI)) * std::pow(-1.0f, i + 1);
      f_n = (2.0f / (i * PI)) * (1.0f - std::pow(-1.0f, i));
      lam_n = i * PI;
      lam_2 = lam_n * lam_n;
      var1 =
          (q_n / lam_2 + (f_n - (q_n / lam_2)) * std::exp(-1.0f * lam_2 * t)) *
          std::sin(i * PI * x);
      sum += var1;
    }
    return sum;
  };

  float const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  float benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1f, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplPureHeatEquationSourceDoubleDirichletBCHostEuler() {
  using lss_one_dim_general_heat_equation_solvers_cuda::implicit_solvers::
      Implicit1DGeneralHeatEquationCUDA;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_types::MemorySpace;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with source: \n\n";
  std::cout << " Using Euler algorithm on CUDA host with \n"
               "implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(double).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = 1, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquationCUDA
  typedef Implicit1DGeneralHeatEquationCUDA<
      double, BoundaryConditionType::Dirichlet, MemorySpace::Host,
      RealSparseSolverCUDA, std::vector, std::allocator<double>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 1000;
  // number of time subdivisions:
  std::size_t const Td = 1000;
  // initial condition:
  auto initialCondition = [](double x) { return 1.0; };
  // boundary conditions:
  auto boundary = std::make_pair(0.0, 0.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<double> solution(Sd + 1, 0.0);
  // initialize solver
  implicit_solver impl_solver(Range<double>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // set heat source:
  impl_solver.setHeatSource([](double x, double t) { return x; });
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::Euler);
  // get exact solution:
  auto exact = [](double x, double t, std::size_t n) {
    double sum{};
    double q_n{};
    double f_n{};
    double lam_n{};
    double lam_2{};
    double var1{};
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

  double const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  double benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplPureHeatEquationSourceFloatDirichletBCHostCN() {
  using lss_one_dim_general_heat_equation_solvers_cuda::implicit_solvers::
      Implicit1DGeneralHeatEquationCUDA;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_types::MemorySpace;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with source: \n\n";
  std::cout << " Using Euler algorithm on CUDA host with implicit "
               "Crank-Nicolson method\n\n";
  std::cout << " Value type: " << typeid(float).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = 1, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquationCUDA
  typedef Implicit1DGeneralHeatEquationCUDA<
      float, BoundaryConditionType::Dirichlet, MemorySpace::Host,
      RealSparseSolverCUDA, std::vector, std::allocator<float>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 1000;
  // number of time subdivisions:
  std::size_t const Td = 1000;
  // initial condition:
  auto initialCondition = [](float x) { return 1.0f; };
  // boundary conditions:
  auto boundary = std::make_pair(0.0f, 0.0f);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<float> solution(Sd + 1, 0.0f);
  // initialize solver
  implicit_solver impl_solver(Range<float>(0.0f, 1.0f), 0.1f, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0f);
  // set heat source:
  impl_solver.setHeatSource([](float x, float t) { return x; });
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::CrankNicolson);
  // get exact solution:
  auto exact = [](float x, float t, std::size_t n) {
    float sum{};
    float q_n{};
    float f_n{};
    float lam_n{};
    float lam_2{};
    float var1{};
    for (std::size_t i = 1; i <= n; ++i) {
      q_n = (2.0f / (i * PI)) * std::pow(-1.0f, i + 1);
      f_n = (2.0f / (i * PI)) * (1.0f - std::pow(-1.0f, i));
      lam_n = i * PI;
      lam_2 = lam_n * lam_n;
      var1 =
          (q_n / lam_2 + (f_n - (q_n / lam_2)) * std::exp(-1.0f * lam_2 * t)) *
          std::sin(i * PI * x);
      sum += var1;
    }
    return sum;
  };

  float const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  float benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1f, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplPureHeatEquationSourceDoubleDirichletBCHostCN() {
  using lss_one_dim_general_heat_equation_solvers_cuda::implicit_solvers::
      Implicit1DGeneralHeatEquationCUDA;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_types::MemorySpace;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with source: \n\n";
  std::cout << " Using Euler algorithm on CUDA host with implicit \n"
               "Crank-Nicolson method\n\n";
  std::cout << " Value type: " << typeid(double).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = 1, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquationCUDA
  typedef Implicit1DGeneralHeatEquationCUDA<
      double, BoundaryConditionType::Dirichlet, MemorySpace::Host,
      RealSparseSolverCUDA, std::vector, std::allocator<double>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 1000;
  // number of time subdivisions:
  std::size_t const Td = 1000;
  // initial condition:
  auto initialCondition = [](double x) { return 1.0; };
  // boundary conditions:
  auto boundary = std::make_pair(0.0, 0.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<double> solution(Sd + 1, 0.0);
  // initialize solver
  implicit_solver impl_solver(Range<double>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // set heat source:
  impl_solver.setHeatSource([](double x, double t) { return x; });
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::CrankNicolson);
  // get exact solution:
  auto exact = [](double x, double t, std::size_t n) {
    double sum{};
    double q_n{};
    double f_n{};
    double lam_n{};
    double lam_2{};
    double var1{};
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

  double const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  double benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplPureHeatEquationSourceDirichletBCCUDA() {
  std::cout << "============================================================\n";
  std::cout << "== Implicit Pure Heat Equation with source (Dirichlet BC) ==\n";
  std::cout << "============================================================\n";

  testImplPureHeatEquationSourceFloatDirichletBCDeviceEuler();
  testImplPureHeatEquationSourceDoubleDirichletBCDeviceEuler();
  testImplPureHeatEquationSourceFloatDirichletBCDeviceCN();
  testImplPureHeatEquationSourceDoubleDirichletBCDeviceCN();
  testImplPureHeatEquationSourceFloatDirichletBCHostEuler();
  testImplPureHeatEquationSourceDoubleDirichletBCHostEuler();
  testImplPureHeatEquationSourceFloatDirichletBCHostCN();
  testImplPureHeatEquationSourceDoubleDirichletBCHostCN();

  std::cout << "============================================================\n";
}

// ============================================================================
// ==== Heat problem with homogeneous Robin boundaryconditions and source =====
// ============================================================================

void testImplPureHeatEquationSourceFloatRobinBCDeviceEuler() {
  using lss_one_dim_general_heat_equation_solvers_cuda::implicit_solvers::
      Implicit1DGeneralHeatEquationCUDA;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_types::MemorySpace;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with source: \n\n";
  std::cout << " Using Euler algorithm on CUDA device with \n"
               "implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(float).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquationCUDA
  typedef Implicit1DGeneralHeatEquationCUDA<
      float, BoundaryConditionType::Robin, MemorySpace::Device,
      RealSparseSolverCUDA, std::vector, std::allocator<float>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](float x) { return x; };
  // boundary conditions:
  auto leftBoundary = std::make_pair(1.0f, 0.0f);
  auto rightBoundary = std::make_pair(1.0f, 0.0f);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<float> solution(Sd + 1, 0.0f);
  // initialize solver
  implicit_solver impl_solver(Range<float>(0.0f, 1.0f), 0.1f, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(leftBoundary, rightBoundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0f);
  // set heat source:
  impl_solver.setHeatSource([](float x, float t) { return x; });
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::Euler);
  // get exact solution:
  auto exact = [](float x, float t, std::size_t n) {
    float sum{};
    float q_n{};
    float f_n{};
    float lam_n{};
    float lam_2{};
    float var1{};
    for (std::size_t i = 1; i <= n; ++i) {
      lam_n = i * PI;
      lam_2 = lam_n * lam_n;
      q_n = (2.0f / (lam_2)) * (std::pow(-1.0f, i) - 1.0f);
      f_n = q_n;

      var1 = ((q_n / lam_2) +
              (f_n - (q_n / lam_2)) * std::exp(-1.0f * lam_2 * t)) *
             std::cos(lam_n * x);
      sum += var1;
    }
    return ((0.5f + 0.5f * t) + sum);
  };

  float const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  float benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1f, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplPureHeatEquationSourceDoubleRobinBCDeviceEuler() {
  using lss_one_dim_general_heat_equation_solvers_cuda::implicit_solvers::
      Implicit1DGeneralHeatEquationCUDA;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_types::MemorySpace;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with source: \n\n";
  std::cout << " Using Euler algorithm on CUDA device with \n"
               "implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(double).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquationCUDA
  typedef Implicit1DGeneralHeatEquationCUDA<
      double, BoundaryConditionType::Robin, MemorySpace::Device,
      RealSparseSolverCUDA, std::vector, std::allocator<double>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](double x) { return x; };
  // boundary conditions:
  auto leftBoundary = std::make_pair(1.0, 0.0);
  auto rightBoundary = std::make_pair(1.0, 0.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<double> solution(Sd + 1, 0.0);
  // initialize solver
  implicit_solver impl_solver(Range<double>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(leftBoundary, rightBoundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // set heat source:
  impl_solver.setHeatSource([](double x, double t) { return x; });
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::Euler);
  // get exact solution:
  auto exact = [](double x, double t, std::size_t n) {
    double sum{};
    double q_n{};
    double f_n{};
    double lam_n{};
    double lam_2{};
    double var1{};
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

  double const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  double benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplPureHeatEquationSourceFloatRobinBCHostEuler() {
  using lss_one_dim_general_heat_equation_solvers_cuda::implicit_solvers::
      Implicit1DGeneralHeatEquationCUDA;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_types::MemorySpace;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with source: \n\n";
  std::cout << " Using Euler algorithm on CUDA host with \n"
               "implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(float).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquationCUDA
  typedef Implicit1DGeneralHeatEquationCUDA<
      float, BoundaryConditionType::Robin, MemorySpace::Host,
      RealSparseSolverCUDA, std::vector, std::allocator<float>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](float x) { return x; };
  // boundary conditions:
  auto leftBoundary = std::make_pair(1.0f, 0.0f);
  auto rightBoundary = std::make_pair(1.0f, 0.0f);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<float> solution(Sd + 1, 0.0f);
  // initialize solver
  implicit_solver impl_solver(Range<float>(0.0f, 1.0f), 0.1f, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(leftBoundary, rightBoundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0f);
  // set heat source:
  impl_solver.setHeatSource([](float x, float t) { return x; });
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::Euler);
  // get exact solution:
  auto exact = [](float x, float t, std::size_t n) {
    float sum{};
    float q_n{};
    float f_n{};
    float lam_n{};
    float lam_2{};
    float var1{};
    for (std::size_t i = 1; i <= n; ++i) {
      lam_n = i * PI;
      lam_2 = lam_n * lam_n;
      q_n = (2.0f / (lam_2)) * (std::pow(-1.0f, i) - 1.0f);
      f_n = q_n;

      var1 = ((q_n / lam_2) +
              (f_n - (q_n / lam_2)) * std::exp(-1.0f * lam_2 * t)) *
             std::cos(lam_n * x);
      sum += var1;
    }
    return ((0.5f + 0.5f * t) + sum);
  };

  float const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  float benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1f, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplPureHeatEquationSourceDoubleRobinBCHostEuler() {
  using lss_one_dim_general_heat_equation_solvers_cuda::implicit_solvers::
      Implicit1DGeneralHeatEquationCUDA;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_types::MemorySpace;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with source: \n\n";
  std::cout << " Using Euler algorithm on CUDA host with \n"
               "implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(double).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquationCUDA
  typedef Implicit1DGeneralHeatEquationCUDA<
      double, BoundaryConditionType::Robin, MemorySpace::Host,
      RealSparseSolverCUDA, std::vector, std::allocator<double>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](double x) { return x; };
  // boundary conditions:
  auto leftBoundary = std::make_pair(1.0, 0.0);
  auto rightBoundary = std::make_pair(1.0, 0.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<double> solution(Sd + 1, 0.0);
  // initialize solver
  implicit_solver impl_solver(Range<double>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(leftBoundary, rightBoundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // set heat source:
  impl_solver.setHeatSource([](double x, double t) { return x; });
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::Euler);
  // get exact solution:
  auto exact = [](double x, double t, std::size_t n) {
    double sum{};
    double q_n{};
    double f_n{};
    double lam_n{};
    double lam_2{};
    double var1{};
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

  double const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  double benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplPureHeatEquationSourceFloatRobinBCDeviceCN() {
  using lss_one_dim_general_heat_equation_solvers_cuda::implicit_solvers::
      Implicit1DGeneralHeatEquationCUDA;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_types::MemorySpace;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with source: \n\n";
  std::cout << " Using Euler algorithm on CUDA device with implicit \n"
               "Crank-Nicoloson method\n\n";
  std::cout << " Value type: " << typeid(float).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquationCUDA
  typedef Implicit1DGeneralHeatEquationCUDA<
      float, BoundaryConditionType::Robin, MemorySpace::Device,
      RealSparseSolverCUDA, std::vector, std::allocator<float>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](float x) { return x; };
  // boundary conditions:
  auto leftBoundary = std::make_pair(1.0f, 0.0f);
  auto rightBoundary = std::make_pair(1.0f, 0.0f);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<float> solution(Sd + 1, 0.0f);
  // initialize solver
  implicit_solver impl_solver(Range<float>(0.0f, 1.0f), 0.1f, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(leftBoundary, rightBoundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0f);
  // set heat source:
  impl_solver.setHeatSource([](float x, float t) { return x; });
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::CrankNicolson);
  // get exact solution:
  auto exact = [](float x, float t, std::size_t n) {
    float sum{};
    float q_n{};
    float f_n{};
    float lam_n{};
    float lam_2{};
    float var1{};
    for (std::size_t i = 1; i <= n; ++i) {
      lam_n = i * PI;
      lam_2 = lam_n * lam_n;
      q_n = (2.0f / (lam_2)) * (std::pow(-1.0f, i) - 1.0f);
      f_n = q_n;

      var1 = ((q_n / lam_2) +
              (f_n - (q_n / lam_2)) * std::exp(-1.0f * lam_2 * t)) *
             std::cos(lam_n * x);
      sum += var1;
    }
    return ((0.5f + 0.5f * t) + sum);
  };

  float const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  float benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1f, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplPureHeatEquationSourceDoubleRobinBCDeviceCN() {
  using lss_one_dim_general_heat_equation_solvers_cuda::implicit_solvers::
      Implicit1DGeneralHeatEquationCUDA;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_types::MemorySpace;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with source: \n\n";
  std::cout << " Using Euler algorithm on CUDA device with implicit \n"
               "Crank-Nicolson method\n\n";
  std::cout << " Value type: " << typeid(double).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquationCUDA
  typedef Implicit1DGeneralHeatEquationCUDA<
      double, BoundaryConditionType::Robin, MemorySpace::Device,
      RealSparseSolverCUDA, std::vector, std::allocator<double>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](double x) { return x; };
  // boundary conditions:
  auto leftBoundary = std::make_pair(1.0, 0.0);
  auto rightBoundary = std::make_pair(1.0, 0.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<double> solution(Sd + 1, 0.0);
  // initialize solver
  implicit_solver impl_solver(Range<double>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(leftBoundary, rightBoundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // set heat source:
  impl_solver.setHeatSource([](double x, double t) { return x; });
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::CrankNicolson);
  // get exact solution:
  auto exact = [](double x, double t, std::size_t n) {
    double sum{};
    double q_n{};
    double f_n{};
    double lam_n{};
    double lam_2{};
    double var1{};
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

  double const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  double benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplPureHeatEquationSourceFloatRobinBCHostCN() {
  using lss_one_dim_general_heat_equation_solvers_cuda::implicit_solvers::
      Implicit1DGeneralHeatEquationCUDA;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_types::MemorySpace;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with source: \n\n";
  std::cout << " Using Euler algorithm on CUDA host with implicit \n"
               "Crank-Nicolson method\n\n";
  std::cout << " Value type: " << typeid(float).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquationCUDA
  typedef Implicit1DGeneralHeatEquationCUDA<
      float, BoundaryConditionType::Robin, MemorySpace::Host,
      RealSparseSolverCUDA, std::vector, std::allocator<float>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](float x) { return x; };
  // boundary conditions:
  auto leftBoundary = std::make_pair(1.0f, 0.0f);
  auto rightBoundary = std::make_pair(1.0f, 0.0f);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<float> solution(Sd + 1, 0.0f);
  // initialize solver
  implicit_solver impl_solver(Range<float>(0.0f, 1.0f), 0.1f, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(leftBoundary, rightBoundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0f);
  // set heat source:
  impl_solver.setHeatSource([](float x, float t) { return x; });
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::CrankNicolson);
  // get exact solution:
  auto exact = [](float x, float t, std::size_t n) {
    float sum{};
    float q_n{};
    float f_n{};
    float lam_n{};
    float lam_2{};
    float var1{};
    for (std::size_t i = 1; i <= n; ++i) {
      lam_n = i * PI;
      lam_2 = lam_n * lam_n;
      q_n = (2.0f / (lam_2)) * (std::pow(-1.0f, i) - 1.0f);
      f_n = q_n;

      var1 = ((q_n / lam_2) +
              (f_n - (q_n / lam_2)) * std::exp(-1.0f * lam_2 * t)) *
             std::cos(lam_n * x);
      sum += var1;
    }
    return ((0.5f + 0.5f * t) + sum);
  };

  float const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  float benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1f, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplPureHeatEquationSourceDoubleRobinBCHostCN() {
  using lss_one_dim_general_heat_equation_solvers_cuda::implicit_solvers::
      Implicit1DGeneralHeatEquationCUDA;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_types::MemorySpace;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with source: \n\n";
  std::cout << " Using Euler algorithm on CUDA host with implicit \n"
               "Crank-Nicolson method\n\n";
  std::cout << " Value type: " << typeid(double).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquationCUDA
  typedef Implicit1DGeneralHeatEquationCUDA<
      double, BoundaryConditionType::Robin, MemorySpace::Host,
      RealSparseSolverCUDA, std::vector, std::allocator<double>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](double x) { return x; };
  // boundary conditions:
  auto leftBoundary = std::make_pair(1.0, 0.0);
  auto rightBoundary = std::make_pair(1.0, 0.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<double> solution(Sd + 1, 0.0);
  // initialize solver
  implicit_solver impl_solver(Range<double>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(leftBoundary, rightBoundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // set heat source:
  impl_solver.setHeatSource([](double x, double t) { return x; });
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::CrankNicolson);
  // get exact solution:
  auto exact = [](double x, double t, std::size_t n) {
    double sum{};
    double q_n{};
    double f_n{};
    double lam_n{};
    double lam_2{};
    double var1{};
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

  double const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  double benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplPureHeatEquationSourceRobinBCCUDA() {
  std::cout << "============================================================\n";
  std::cout << "===== Implicit Pure Heat Equation with source (Robin BC) ===\n";
  std::cout << "============================================================\n";

  testImplPureHeatEquationSourceFloatRobinBCDeviceEuler();
  testImplPureHeatEquationSourceDoubleRobinBCDeviceEuler();
  testImplPureHeatEquationSourceFloatRobinBCHostEuler();
  testImplPureHeatEquationSourceDoubleRobinBCHostEuler();

  testImplPureHeatEquationSourceFloatRobinBCDeviceCN();
  testImplPureHeatEquationSourceDoubleRobinBCDeviceCN();
  testImplPureHeatEquationSourceFloatRobinBCHostCN();
  testImplPureHeatEquationSourceDoubleRobinBCHostCN();

  std::cout << "============================================================\n";
}

// ============================================================================
// ========= Heat problem with non-homogeneous boundary conditions ============
// ============================================================================

void testImplNonHomPureHeatEquationDoubleDirichletBCDeviceEuler() {
  using lss_one_dim_general_heat_equation_solvers_cuda::implicit_solvers::
      Implicit1DGeneralHeatEquationCUDA;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_types::MemorySpace;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with non-hom BC: \n\n";
  std::cout << " Using CUDA solvers algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(double).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = 0, U(1,t) = 100, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = 100*x + (198/pi)*sum_0^infty{ \n"
               "(-1)^(n)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquationCUDA
  typedef Implicit1DGeneralHeatEquationCUDA<
      double, BoundaryConditionType::Dirichlet, MemorySpace::Device,
      RealSparseSolverCUDA, std::vector, std::allocator<double>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 1000;
  // number of time subdivisions:
  std::size_t const Td = 1000;
  // initial condition:
  auto initialCondition = [](double x) { return x; };
  // boundary conditions:
  auto boundary = std::make_pair(0.0, 100.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<double> solution(Sd + 1, 0.0);
  // initialize solver
  implicit_solver impl_solver(Range<double>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::Euler);
  // get exact solution:
  auto exact = [](double x, double t, std::size_t n) {
    double const first = 198.0 / PI;
    double sum{};
    double var1{};
    double var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var1 = std::pow(-1.0, i) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
      var2 = std::sin(i * PI * x) / i;
      sum += (var1 * var2);
    }
    return (100 * x + first * sum);
  };

  double const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  double benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplNonHomPureHeatEquationFloatDirichletBCDeviceEuler() {
  using lss_one_dim_general_heat_equation_solvers_cuda::implicit_solvers::
      Implicit1DGeneralHeatEquationCUDA;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_types::MemorySpace;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with non-hom BC: \n\n";
  std::cout << " Using CUDA solvers algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(float).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = 0, U(1,t) = 100, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = 100*x +  (198/pi)*sum_0^infty{ \n"
               "(-1)^(n)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquationCUDA
  typedef Implicit1DGeneralHeatEquationCUDA<
      float, BoundaryConditionType::Dirichlet, MemorySpace::Device,
      RealSparseSolverCUDA, std::vector, std::allocator<float>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 1000;
  // number of time subdivisions:
  std::size_t const Td = 1000;
  // initial condition:
  auto initialCondition = [](float x) { return x; };
  // boundary conditions:
  auto boundary = std::make_pair(0.0, 100.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<float> solution(Sd + 1, 0.0);
  // initialize solver
  implicit_solver impl_solver(Range<float>(0.0f, 1.0f), 0.1f, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0f);
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::Euler);
  // get exact solution:
  auto exact = [](float x, float t, std::size_t n) {
    float const first = 198.0f / PI;
    float sum{};
    float var1{};
    float var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var1 = std::pow(-1.0f, i) * std::exp(-1.0f * (i * PI) * (i * PI) * t);
      var2 = std::sin(i * PI * x) / i;
      sum += (var1 * var2);
    }
    return (100.0f * x + first * sum);
  };

  float const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  float benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1f, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplNonHomPureHeatEquationDoubleDirichletBCDeviceCN() {
  using lss_one_dim_general_heat_equation_solvers_cuda::implicit_solvers::
      Implicit1DGeneralHeatEquationCUDA;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_types::MemorySpace;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with non-hom BC: \n\n";
  std::cout << " Using CUDA solvers algorithm with implicit Clark-Nicolson \n"
               "method\n\n";
  std::cout << " Value type: " << typeid(double).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = 0, U(1,t) = 100, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = 100*x +  (198/pi)*sum_0^infty{ \n"
               "(-1)^(n)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquationCUDA
  typedef Implicit1DGeneralHeatEquationCUDA<
      double, BoundaryConditionType::Dirichlet, MemorySpace::Device,
      RealSparseSolverCUDA, std::vector, std::allocator<double>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 1000;
  // number of time subdivisions:
  std::size_t const Td = 1000;
  // initial condition:
  auto initialCondition = [](double x) { return x; };
  // boundary conditions:
  auto boundary = std::make_pair(0.0, 100.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<double> solution(Sd + 1, 0.0);
  // initialize solver
  implicit_solver impl_solver(Range<double>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::CrankNicolson);
  // get exact solution:
  auto exact = [](double x, double t, std::size_t n) {
    double const first = 198.0 / PI;
    double sum{};
    double var1{};
    double var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var1 = std::pow(-1.0, i) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
      var2 = std::sin(i * PI * x) / i;
      sum += (var1 * var2);
    }
    return (100.0 * x + first * sum);
  };

  double const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  double benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplNonHomPureHeatEquationFloatDirichletBCDeviceCN() {
  using lss_one_dim_general_heat_equation_solvers_cuda::implicit_solvers::
      Implicit1DGeneralHeatEquationCUDA;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_types::MemorySpace;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with non-hom BC: \n\n";
  std::cout << " Using CUDA solvers algorithm with implicit Clark-Nicolson \n"
               "method\n\n";
  std::cout << " Value type: " << typeid(float).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = 0, U(1,t) = 100, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = 100*x + (198/pi)*sum_0^infty{ \n"
               "(-1)^(n)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquationCUDA
  typedef Implicit1DGeneralHeatEquationCUDA<
      float, BoundaryConditionType::Dirichlet, MemorySpace::Device,
      RealSparseSolverCUDA, std::vector, std::allocator<float>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 1000;
  // number of time subdivisions:
  std::size_t const Td = 1000;
  // initial condition:
  auto initialCondition = [](float x) { return x; };
  // boundary conditions:
  auto boundary = std::make_pair(0.0f, 100.0f);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<float> solution(Sd + 1, 0.0f);
  // initialize solver
  implicit_solver impl_solver(Range<float>(0.0f, 1.0f), 0.1f, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::CrankNicolson);
  // get exact solution:
  auto exact = [](float x, float t, std::size_t n) {
    float const first = 198.0f / PI;
    float sum{};
    float var1{};
    float var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var1 = std::pow(-1.0f, i) * std::exp(-1.0f * (i * PI) * (i * PI) * t);
      var2 = std::sin(i * PI * x) / i;
      sum += (var1 * var2);
    }
    return (100.0f * x + first * sum);
  };

  float const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  float benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1f, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplNonHomPureHeatEquationDirichletBCDeviceCUDA() {
  std::cout << "============================================================\n";
  std::cout << "======== Implicit Pure Heat Equation (with non-homogeneous \n"
               "Dirichlet BC) =======\n";
  std::cout << "============================================================\n";

  testImplNonHomPureHeatEquationDoubleDirichletBCDeviceEuler();
  testImplNonHomPureHeatEquationFloatDirichletBCDeviceEuler();
  testImplNonHomPureHeatEquationDoubleDirichletBCDeviceCN();
  testImplNonHomPureHeatEquationFloatDirichletBCDeviceCN();

  std::cout << "============================================================\n";
}

void testImplNonHomPureHeatEquationDoubleDirichletBCHostEuler() {
  using lss_one_dim_general_heat_equation_solvers_cuda::implicit_solvers::
      Implicit1DGeneralHeatEquationCUDA;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_types::MemorySpace;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with non-hom BC: \n\n";
  std::cout << " Using CUDA solvers algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(double).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = 0, U(1,t) = 100, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = 100*x + (198/pi)*sum_0^infty{ \n"
               "(-1)^(n)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquationCUDA
  typedef Implicit1DGeneralHeatEquationCUDA<
      double, BoundaryConditionType::Dirichlet, MemorySpace::Host,
      RealSparseSolverCUDA, std::vector, std::allocator<double>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](double x) { return x; };
  // boundary conditions:
  auto boundary = std::make_pair(0.0, 100.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<double> solution(Sd + 1, 0.0);
  // initialize solver
  implicit_solver impl_solver(Range<double>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::Euler);
  // get exact solution:
  auto exact = [](double x, double t, std::size_t n) {
    double const first = 198.0 / PI;
    double sum{};
    double var1{};
    double var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var1 = std::pow(-1.0, i) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
      var2 = std::sin(i * PI * x) / i;
      sum += (var1 * var2);
    }
    return (100.0 * x + first * sum);
  };

  double const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  double benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplNonHomPureHeatEquationFloatDirichletBCHostEuler() {
  using lss_one_dim_general_heat_equation_solvers_cuda::implicit_solvers::
      Implicit1DGeneralHeatEquationCUDA;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_types::MemorySpace;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with non-hom BC: \n\n";
  std::cout << " Using CUDA solvers algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(float).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = 0, U(1,t) = 100, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = 100*x + (198/pi)*sum_0^infty{ \n"
               "(-1)^(n+1)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquationCUDA
  typedef Implicit1DGeneralHeatEquationCUDA<
      float, BoundaryConditionType::Dirichlet, MemorySpace::Host,
      RealSparseSolverCUDA, std::vector, std::allocator<float>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](float x) { return x; };
  // boundary conditions:
  auto boundary = std::make_pair(0.0f, 100.0f);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<float> solution(Sd + 1, 0.0f);
  // initialize solver
  implicit_solver impl_solver(Range<float>(0.0f, 1.0f), 0.1f, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0f);
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::Euler);
  // get exact solution:
  auto exact = [](float x, float t, std::size_t n) {
    float const first = 198.0f / PI;
    float sum{};
    float var1{};
    float var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var1 = std::pow(-1.0f, i) * std::exp(-1.0f * (i * PI) * (i * PI) * t);
      var2 = std::sin(i * PI * x) / i;
      sum += (var1 * var2);
    }
    return (100.0f * x + first * sum);
  };

  float const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  float benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1f, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplNonHomPureHeatEquationDoubleDirichletBCHostCN() {
  using lss_one_dim_general_heat_equation_solvers_cuda::implicit_solvers::
      Implicit1DGeneralHeatEquationCUDA;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_types::MemorySpace;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with non-hom BC: \n\n";
  std::cout << " Using CUDA solvers algorithm with implicit Clark-Nicolson \n"
               "method\n\n";
  std::cout << " Value type: " << typeid(double).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = 0, U(1,t) = 100, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = 100.0*x + (198/pi)*sum_0^infty{ \n"
               "(-1)^(n+1)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquationCUDA
  typedef Implicit1DGeneralHeatEquationCUDA<
      double, BoundaryConditionType::Dirichlet, MemorySpace::Host,
      RealSparseSolverCUDA, std::vector, std::allocator<double>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](double x) { return x; };
  // boundary conditions:
  auto boundary = std::make_pair(0.0, 100.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<double> solution(Sd + 1, 0.0);
  // initialize solver
  implicit_solver impl_solver(Range<double>(0.0, 1.0), 0.1, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::CrankNicolson);
  // get exact solution:
  auto exact = [](double x, double t, std::size_t n) {
    double const first = 198.0 / PI;
    double sum{};
    double var1{};
    double var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var1 = std::pow(-1.0, i) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
      var2 = std::sin(i * PI * x) / i;
      sum += (var1 * var2);
    }
    return (100.0 * x + first * sum);
  };

  double const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  double benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplNonHomPureHeatEquationFloatDirichletBCHostCN() {
  using lss_one_dim_general_heat_equation_solvers_cuda::implicit_solvers::
      Implicit1DGeneralHeatEquationCUDA;
  using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ImplicitPDESchemes;
  using lss_types::MemorySpace;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with non-hom BC: \n\n";
  std::cout << " Using CUDA solvers algorithm with implicit Clark-Nicolson \n"
               "method\n\n";
  std::cout << " Value type: " << typeid(float).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = 0, U(1,t) = 100, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = 100.0*x + (198/pi)*sum_0^infty{ \n"
               "(-1)^(n)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DGeneralHeatEquationCUDA
  typedef Implicit1DGeneralHeatEquationCUDA<
      float, BoundaryConditionType::Dirichlet, MemorySpace::Host,
      RealSparseSolverCUDA, std::vector, std::allocator<float>>
      implicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initialCondition = [](float x) { return x; };
  // boundary conditions:
  auto boundary = std::make_pair(0.0f, 100.0f);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<float> solution(Sd + 1, 0.0f);
  // initialize solver
  implicit_solver impl_solver(Range<float>(0.0f, 1.0f), 0.1f, Sd, Td);
  // set boundary conditions:
  impl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  impl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  impl_solver.set2OrderCoefficient(1.0f);
  // get the solution:
  impl_solver.solve(solution, ImplicitPDESchemes::CrankNicolson);
  // get exact solution:
  auto exact = [](float x, float t, std::size_t n) {
    float const first = 198.0f / PI;
    float sum{};
    float var1{};
    float var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var1 = std::pow(-1.0f, i) * std::exp(-1.0f * (i * PI) * (i * PI) * t);
      var2 = std::sin(i * PI * x) / i;
      sum += (var1 * var2);
    }
    return (100.0f * x + first * sum);
  };

  float const h = impl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  float benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.1f, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testImplNonHomPureHeatEquationDirichletBCHostCUDA() {
  std::cout << "============================================================\n";
  std::cout << "========= Implicit Pure Heat Equation (with non-homogeneous \n"
               "Dirichlet BC) =====\n";
  std::cout << "============================================================\n";

  testImplNonHomPureHeatEquationDoubleDirichletBCHostEuler();
  testImplNonHomPureHeatEquationFloatDirichletBCHostEuler();
  testImplNonHomPureHeatEquationDoubleDirichletBCHostCN();
  testImplNonHomPureHeatEquationFloatDirichletBCHostCN();

  std::cout << "============================================================\n";
}

// ============================================================================
// ======================== EXPLICIT SOLVERS ==================================
// ============================================================================

// ============================================================================
// ====== Heat problem with homogeneous Dirichlet boundary conditions =========
// ============================================================================

void testExplPureHeatEquationDoubleDirichletBCDeviceEuler() {
  using lss_one_dim_general_heat_equation_solvers_cuda::explicit_solvers::
      Explicit1DGeneralHeatEquationCUDA;
  using lss_types::BoundaryConditionType;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using CUDA  explicit Euler method\n\n";
  std::cout << " Value type: " << typeid(double).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) \n"
               "*sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Explicit1DGeneralHeatEquationCUDA
  typedef Explicit1DGeneralHeatEquationCUDA<double,
                                            BoundaryConditionType::Dirichlet,
                                            std::vector, std::allocator<double>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 5000;
  // initial condition:
  auto initialCondition = [](double x) { return x; };
  // boundary conditions:
  auto boundary = std::make_pair(0.0, 0.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<double> solution(Sd + 1, 0.0);
  // initialize solver
  explicit_solver expl_solver(Range<double>(0.0, 1.0), 0.2, Sd, Td);
  // set boundary conditions:
  expl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  expl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  expl_solver.solve(solution);
  // get exact solution:
  auto exact = [](double x, double t, std::size_t n) {
    double const first = 2.0 / PI;
    double sum{};
    double var1{};
    double var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var1 = std::pow(-1.0, i + 1) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
      var2 = std::sin(i * PI * x) / i;
      sum += (var1 * var2);
    }
    return (first * sum);
  };

  double const h = expl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  double benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.2, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testExplPureHeatEquationFloatDirichletBCDeviceEuler() {
  using lss_one_dim_general_heat_equation_solvers_cuda::explicit_solvers::
      Explicit1DGeneralHeatEquationCUDA;
  using lss_types::BoundaryConditionType;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using CUDA  explicit Euler method\n\n";
  std::cout << " Value type: " << typeid(double).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) \n"
               "*sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Explicit1DGeneralHeatEquationCUDA
  typedef Explicit1DGeneralHeatEquationCUDA<float,
                                            BoundaryConditionType::Dirichlet,
                                            std::vector, std::allocator<float>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 5000;
  // initial condition:
  auto initialCondition = [](float x) { return x; };
  // boundary conditions:
  auto boundary = std::make_pair(0.0f, 0.0f);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<float> solution(Sd + 1, 0.0f);
  // initialize solver
  explicit_solver expl_solver(Range<float>(0.0f, 1.0f), 0.2f, Sd, Td);
  // set boundary conditions:
  expl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  expl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  expl_solver.solve(solution);
  // get exact solution:
  auto exact = [](float x, float t, std::size_t n) {
    float const first = 2.0f / PI;
    float sum{};
    float var1{};
    float var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var1 = std::pow(-1.0f, i + 1) * std::exp(-1.0f * (i * PI) * (i * PI) * t);
      var2 = std::sin(i * PI * x) / i;
      sum += (var1 * var2);
    }
    return (first * sum);
  };

  float const h = expl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  float benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.2f, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testExplPureHeatEquationDirichletBCDeviceCUDA() {
  std::cout << "============================================================\n";
  std::cout << "======== Explicit Pure Heat Equation (Dirichlet BC) ========\n";
  std::cout << "============================================================\n";

  testExplPureHeatEquationDoubleDirichletBCDeviceEuler();
  testExplPureHeatEquationFloatDirichletBCDeviceEuler();

  std::cout << "============================================================\n";
}

// ============================================================================
// ====== Heat problem with nonhomogeneous Dirichlet boundary conditions ======
// ============================================================================

void testExplNonHomPureHeatEquationDoubleDirichletBCDeviceEuler() {
  using lss_one_dim_general_heat_equation_solvers_cuda::explicit_solvers::
      Explicit1DGeneralHeatEquationCUDA;
  using lss_types::BoundaryConditionType;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with non-hom BC: \n\n";
  std::cout << " Using CUDA implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(double).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = 0, U(1,t) = 100, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = 100*x + (198/pi)*sum_0^infty{ \n"
               "(-1)^(n)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Explicit1DGeneralHeatEquationCUDA
  typedef Explicit1DGeneralHeatEquationCUDA<double,
                                            BoundaryConditionType::Dirichlet,
                                            std::vector, std::allocator<double>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 5000;
  // initial condition:
  auto initialCondition = [](double x) { return x; };
  // boundary conditions:
  auto boundary = std::make_pair(0.0, 100.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<double> solution(Sd + 1, 0.0);
  // initialize solver
  explicit_solver expl_solver(Range<double>(0.0, 1.0), 0.2, Sd, Td);
  // set boundary conditions:
  expl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  expl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  expl_solver.solve(solution);
  // get exact solution:
  auto exact = [](double x, double t, std::size_t n) {
    double const first = 198.0 / PI;
    double sum{};
    double var1{};
    double var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var1 = std::pow(-1.0, i) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
      var2 = std::sin(i * PI * x) / i;
      sum += (var1 * var2);
    }
    return (100.0 * x + first * sum);
  };

  double const h = expl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  double benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.2, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testExplNonHomPureHeatEquationFloatDirichletBCDeviceEuler() {
  using lss_one_dim_general_heat_equation_solvers_cuda::explicit_solvers::
      Explicit1DGeneralHeatEquationCUDA;
  using lss_types::BoundaryConditionType;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with non-hom BC: \n\n";
  std::cout << " Using CUDA solvers algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(float).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = 0, U(1,t) = 100, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = 100*x + (198/pi)*sum_0^infty{ \n"
               "(-1)^(n+1)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Explicit1DGeneralHeatEquationCUDA
  typedef Explicit1DGeneralHeatEquationCUDA<float,
                                            BoundaryConditionType::Dirichlet,
                                            std::vector, std::allocator<float>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 5000;
  // initial condition:
  auto initialCondition = [](float x) { return x; };
  // boundary conditions:
  auto boundary = std::make_pair(0.0, 100.0f);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<float> solution(Sd + 1, 0.0f);
  // initialize solver
  explicit_solver expl_solver(Range<float>(0.0f, 1.0f), 0.2f, Sd, Td);
  // set boundary conditions:
  expl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  expl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  expl_solver.solve(solution);
  // get exact solution:
  auto exact = [](float x, float t, std::size_t n) {
    float const first = 198.0f / PI;
    float sum{};
    float var1{};
    float var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var1 = std::pow(-1.0f, i) * std::exp(-1.0f * (i * PI) * (i * PI) * t);
      var2 = std::sin(i * PI * x) / i;
      sum += (var1 * var2);
    }
    return (100.0f * x + first * sum);
  };

  float const h = expl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  float benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.2f, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testExplNonHomPureHeatEquationDirichletBCDeviceCUDA() {
  std::cout << "============================================================\n";
  std::cout << "======= Explicit Pure Heat Equation (with non-homogeneous \n"
               "Dirichlet BC) ========\n";
  std::cout << "============================================================\n";

  testExplNonHomPureHeatEquationDoubleDirichletBCDeviceEuler();
  testExplNonHomPureHeatEquationFloatDirichletBCDeviceEuler();

  std::cout << "============================================================\n";
}

// ============================================================================
// ========= Heat problem with homogeneous boundary conditions and source =====
// ============================================================================

void testExplPureHeatEquationSourceFloatDirichletBCEuler() {
  using lss_one_dim_general_heat_equation_solvers_cuda::explicit_solvers::
      Explicit1DGeneralHeatEquationCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ExplicitPDESchemes;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with source: \n\n";
  std::cout << " Using explicit Euler method\n\n";
  std::cout << " Value type: " << typeid(float).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = 1, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Explicit1DGeneralHeatEquationCUDA
  typedef Explicit1DGeneralHeatEquationCUDA<float,
                                            BoundaryConditionType::Dirichlet,
                                            std::vector, std::allocator<float>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 10000;
  // initial condition:
  auto initialCondition = [](float x) { return 1.0; };
  // boundary conditions:
  auto boundary = std::make_pair(0.0, 0.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<float> solution(Sd + 1, 0.0f);
  // initialize solver
  explicit_solver expl_solver(Range<float>(0.0, 1.0), 0.5, Sd, Td);
  // set boundary conditions:
  expl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  expl_solver.setInitialCondition(initialCondition);
  // set heat source:
  expl_solver.setHeatSource([](float x, float t) { return x; });
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  expl_solver.solve(solution);
  // get exact solution:
  auto exact = [](float x, float t, std::size_t n) {
    float sum{};
    float q_n{};
    float f_n{};
    float lam_n{};
    float lam_2{};
    float var1{};
    for (std::size_t i = 1; i <= n; ++i) {
      q_n = (2.0f / (i * PI)) * std::pow(-1.0f, i + 1);
      f_n = (2.0f / (i * PI)) * (1.0f - std::pow(-1.0f, i));
      lam_n = i * PI;
      lam_2 = lam_n * lam_n;
      var1 =
          (q_n / lam_2 + (f_n - (q_n / lam_2)) * std::exp(-1.0f * lam_2 * t)) *
          std::sin(i * PI * x);
      sum += var1;
    }
    return sum;
  };

  float const h = expl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  float benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.5f, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testExplPureHeatEquationSourceDoubleDirichletBCEuler() {
  using lss_one_dim_general_heat_equation_solvers_cuda::explicit_solvers::
      Explicit1DGeneralHeatEquationCUDA;
  using lss_types::BoundaryConditionType;
  using lss_types::ExplicitPDESchemes;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with source: \n\n";
  std::cout << " Using explicit Euler method\n\n";
  std::cout << " Value type: " << typeid(double).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = 1, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Explicit1DGeneralHeatEquationCUDA
  typedef Explicit1DGeneralHeatEquationCUDA<double,
                                            BoundaryConditionType::Dirichlet,
                                            std::vector, std::allocator<double>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 10000;
  // initial condition:
  auto initialCondition = [](double x) { return 1.0; };
  // boundary conditions:
  auto boundary = std::make_pair(0.0, 0.0);
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<double> solution(Sd + 1, 0.0f);
  // initialize solver
  explicit_solver expl_solver(Range<double>(0.0, 1.0), 0.5, Sd, Td);
  // set boundary conditions:
  expl_solver.setBoundaryCondition(boundary);
  // set initial condition:
  expl_solver.setInitialCondition(initialCondition);
  // set heat source:
  expl_solver.setHeatSource([](double x, double t) { return x; });
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  expl_solver.solve(solution);
  // get exact solution:
  auto exact = [](double x, double t, std::size_t n) {
    double sum{};
    double q_n{};
    double f_n{};
    double lam_n{};
    double lam_2{};
    double var1{};
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

  double const h = expl_solver.spaceStep();
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  double benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.5, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testExplPureHeatEquationSourceDirichletBCEulerCUDA() {
  std::cout << "============================================================\n";
  std::cout << "== Explicit Pure Heat Equation with source (Dirichlet BC) ==\n";
  std::cout << "============================================================\n";

  testExplPureHeatEquationSourceFloatDirichletBCEuler();
  testExplPureHeatEquationSourceDoubleDirichletBCEuler();

  std::cout << "============================================================\n";
}

// ============================================================================
// ======== Heat problem with homogeneous Robin boundary conditions ===========
// ============================================================================

void testExplPureHeatEquationDoubleRobinBCDeviceEuler() {
  using lss_one_dim_general_heat_equation_solvers_cuda::explicit_solvers::
      Explicit1DGeneralHeatEquationCUDA;
  using lss_types::BoundaryConditionType;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using CUDA  explicit Euler method\n\n";
  std::cout << " Value type: " << typeid(double).name() << "\n\n";
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

  // typedef the Explicit1DGeneralHeatEquationCUDA
  typedef Explicit1DGeneralHeatEquationCUDA<
      double, BoundaryConditionType::Robin, std::vector, std::allocator<double>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 5000;
  // initial condition:
  auto initialCondition = [](double x) { return x; };
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<double> solution(Sd + 1, 0.0);
  // initialize solver
  explicit_solver expl_solver(Range<double>(0.0, 1.0), 0.2, Sd, Td);
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
  expl_solver.setBoundaryCondition(leftBoundary, rightBoundary);
  // set initial condition:
  expl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set2OrderCoefficient(1.0);
  // get the solution:
  expl_solver.solve(solution);
  // get exact solution:
  auto exact = [](double x, double t, std::size_t n) {
    double const pipi = PI * PI;
    double const first = 4.0 / pipi;
    double sum{};
    double var0{};
    double var1{};
    double var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var0 = (2.0f * i - 1.0f);
      var1 = std::exp(-1.0f * pipi * var0 * var0 * t);
      var2 = std::cos(var0 * PI * x) / (var0 * var0);
      sum += (var1 * var2);
    }
    return (0.5f - first * sum);
  };

  std::cout << "tp : FDM | Exact | Abs Diff\n";
  double benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.2, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testExplPureHeatEquationFloatRobinBCDeviceEuler() {
  using lss_one_dim_general_heat_equation_solvers_cuda::explicit_solvers::
      Explicit1DGeneralHeatEquationCUDA;
  using lss_types::BoundaryConditionType;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using CUDA  explicit Euler method\n\n";
  std::cout << " Value type: " << typeid(float).name() << "\n\n";
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

  // typedef the Explicit1DGeneralHeatEquationCUDA
  typedef Explicit1DGeneralHeatEquationCUDA<float, BoundaryConditionType::Robin,
                                            std::vector, std::allocator<float>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 5000;
  // initial condition:
  auto initialCondition = [](float x) { return x; };
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<float> solution(Sd + 1, 0.0f);
  // initialize solver
  explicit_solver expl_solver(Range<float>(0.0f, 1.0f), 0.2f, Sd, Td);
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
  auto leftBoundary = std::make_pair(1.0f, 0.0f);
  auto rightBoundary = std::make_pair(1.0f, 0.0f);
  // set boundary conditions:
  expl_solver.setBoundaryCondition(leftBoundary, rightBoundary);
  // set initial condition:
  expl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set2OrderCoefficient(1.0f);
  // get the solution:
  expl_solver.solve(solution);
  // get exact solution:
  auto exact = [](float x, float t, std::size_t n) {
    float const pipi = PI * PI;
    float const first = 4.0f / pipi;
    float sum{};
    float var0{};
    float var1{};
    float var2{};
    for (std::size_t i = 1; i <= n; ++i) {
      var0 = (2.0f * i - 1.0f);
      var1 = std::exp(-1.0f * pipi * var0 * var0 * t);
      var2 = std::cos(var0 * PI * x) / (var0 * var0);
      sum += (var1 * var2);
    }
    return (0.5f - first * sum);
  };

  std::cout << "tp : FDM | Exact | Abs Diff\n";
  float benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.2f, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testExplPureHeatEquationRobinBCDeviceCUDA() {
  std::cout << "============================================================\n";
  std::cout << "======== Explicit Pure Heat Equation (Robin BC) ============\n";
  std::cout << "============================================================\n";

  testExplPureHeatEquationDoubleRobinBCDeviceEuler();
  testExplPureHeatEquationFloatRobinBCDeviceEuler();

  std::cout << "============================================================\n";
}

// ============================================================================
// ==== Heat problem with homogeneous Robin boundary conditions and source ====
// ============================================================================

void testExplHomPureHeatEquationSourceFloatRobinBCEuler() {
  using lss_one_dim_general_heat_equation_solvers_cuda::explicit_solvers::
      Explicit1DGeneralHeatEquationCUDA;
  using lss_types::BoundaryConditionType;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with heat source: \n\n";
  std::cout << " Using explicit Euler method\n\n";
  std::cout << " Value type: " << typeid(float).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Explicit1DGeneralHeatEquationCUDA
  typedef Explicit1DGeneralHeatEquationCUDA<float, BoundaryConditionType::Robin,
                                            std::vector, std::allocator<float>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 5000;
  // initial condition:
  auto initialCondition = [](float x) { return x; };
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<float> solution(Sd + 1, 0.0f);
  // initialize solver
  explicit_solver expl_solver(Range<float>(0.0f, 1.0f), 0.2f, Sd, Td);
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
  auto leftBoundary = std::make_pair(1.0f, 0.0f);
  auto rightBoundary = std::make_pair(1.0f, 0.0f);
  // set boundary conditions:
  expl_solver.setBoundaryCondition(leftBoundary, rightBoundary);
  // set initial condition:
  expl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set2OrderCoefficient(1.0);
  // set heat source:
  expl_solver.setHeatSource([](float x, float t) { return x; });
  // get the solution:
  expl_solver.solve(solution);
  // get exact solution:
  auto exact = [](float x, float t, std::size_t n) {
    float sum{};
    float q_n{};
    float f_n{};
    float lam_n{};
    float lam_2{};
    float var1{};
    for (std::size_t i = 1; i <= n; ++i) {
      lam_n = i * PI;
      lam_2 = lam_n * lam_n;
      q_n = (2.0f / (lam_2)) * (std::pow(-1.0f, i) - 1.0f);
      f_n = q_n;

      var1 = ((q_n / lam_2) +
              (f_n - (q_n / lam_2)) * std::exp(-1.0f * lam_2 * t)) *
             std::cos(lam_n * x);
      sum += var1;
    }
    return ((0.5f + 0.5f * t) + sum);
  };

  std::cout << "tp : FDM | Exact | Abs Diff\n";
  float benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.2f, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testExplHomPureHeatEquationSourceDoubleRobinBCEuler() {
  using lss_one_dim_general_heat_equation_solvers_cuda::explicit_solvers::
      Explicit1DGeneralHeatEquationCUDA;
  using lss_types::BoundaryConditionType;
  using lss_utility::Range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation with heat source: \n\n";
  std::cout << " Using explicit Euler method\n\n";
  std::cout << " Value type: " << typeid(double).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Explicit1DGeneralHeatEquationCUDA
  typedef Explicit1DGeneralHeatEquationCUDA<
      double, BoundaryConditionType::Robin, std::vector, std::allocator<double>>
      explicit_solver;

  // number of space subdivisions:
  std::size_t const Sd = 100;
  // number of time subdivisions:
  std::size_t const Td = 5000;
  // initial condition:
  auto initialCondition = [](double x) { return x; };
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  std::vector<double> solution(Sd + 1, 0.0);
  // initialize solver
  explicit_solver expl_solver(Range<double>(0.0, 1.0), 0.2, Sd, Td);
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
  expl_solver.setBoundaryCondition(leftBoundary, rightBoundary);
  // set initial condition:
  expl_solver.setInitialCondition(initialCondition);
  // set thermal diffusivity (C^2 in PDE)
  expl_solver.set2OrderCoefficient(1.0);
  // set heat source:
  expl_solver.setHeatSource([](double x, double t) { return x; });
  // get the solution:
  expl_solver.solve(solution);
  // get exact solution:
  auto exact = [](double x, double t, std::size_t n) {
    double sum{};
    double q_n{};
    double f_n{};
    double lam_n{};
    double lam_2{};
    double var1{};
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
  double benchmark{};
  for (std::size_t j = 0; j < solution.size(); ++j) {
    benchmark = exact(j * h, 0.2, 20);
    std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
              << " | " << (solution[j] - benchmark) << '\n';
  }
}

void testExplHomPureHeatEquationSourceRobinBCCUDA() {
  std::cout << "============================================================\n";
  std::cout << "======= Explicit Pure Heat Equation with Source (with \n"
               "homogeneous Robin BC) ====\n";
  std::cout << "============================================================\n";

  testExplHomPureHeatEquationSourceFloatRobinBCEuler();
  testExplHomPureHeatEquationSourceDoubleRobinBCEuler();

  std::cout << "============================================================\n";
}

#endif  ///_LSS_ONE_DIM_PURE_HEAT_EQUATION_CUDA_T
