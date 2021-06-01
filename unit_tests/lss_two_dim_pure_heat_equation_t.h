#pragma once
#if !defined(_LSS_TWO_DIM_PURE_HEAT_EQUATION_T)
#define _LSS_TWO_DIM_PURE_HEAT_EQUATION_T

#pragma warning(disable : 4305)

#include "common/lss_containers.h"
#include "common/lss_enumerations.h"
#include "common/lss_utility.h"
#include "pde_solvers/two_dim/classic/lss_general_heat_equation_solvers.h"
#include "sparse_solvers/lss_fdm_double_sweep_solver.h"
#include "sparse_solvers/lss_fdm_thomas_lu_solver.h"

#define PI 3.14159

// ///////////////////////////////////////////////////////////////////////////
//							PURE 2D HEAT PROBLEMS
// ///////////////////////////////////////////////////////////////////////////

// ===========================================================================
// ========================== IMPLICIT SOLVERS ===============================
// ===========================================================================

// ===========================================================================
// =========== Heat problem with homogeneous boundary conditions =============
// ===========================================================================

template <typename T>
void test2DImplPureHeatEquationDirichletBCDoubleSweepEuler() {
  using lss_containers::container_2d;
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_double_sweep_solver::fdm_double_sweep_solver;
  using lss_two_dim_classic_pde_solvers::implicit_solvers::
      general_heat_equation;
  using lss_two_dim_pde_utility::dirichlet_boundary_2d;
  using lss_utility::range;
  using lss_utility::sptr_t;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,y,t) + U_yy(x,y,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1>, y in <0,1> and t > 0,\n";
  std::cout << " U(0,y,t) = U(1,y,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0,t) = U(x,1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,y,0) = 100, x,y in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) "
               "*sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DHeatEquation

  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                fdm_double_sweep_solver, std::vector,
                                std::allocator<T>>
      implicit_solver;

  typedef container_2d<std::vector, T, std::allocator<T>> matrix_t;

  // number of space X subdivisions:
  std::size_t const Sxd = 100;
  // number of space y subdivisions:
  std::size_t const Syd = 50;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initial_condition = [](T x, T y) { return 100.0; };
  // boundary conditions:
  auto const &dirichlet_x_low = [](T y, T t) { return 0.0; };
  auto const &dirichlet_x_high = [](T y, T t) { return 0.0; };
  auto const &dirichlet_y_low = [](T x, T t) { return 0.0; };
  auto const &dirichlet_y_high = [](T x, T t) { return 0.0; };

  const auto &boundary = std::make_shared<dirichlet_boundary_2d<T>>(
      std::make_pair(dirichlet_x_low, dirichlet_x_high),
      std::make_pair(dirichlet_y_low, dirichlet_y_high));
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  matrix_t solution(Sxd + 1, Syd + 1, T{});
  // ranges of spatial variables:
  auto const &space_ranges =
      std::make_pair(range<T>(0.0, 1.0), range<T>(0.0, 1.0));

  // initialize solver
  implicit_solver impl_solver(space_ranges, 0.10, std::make_pair(Sxd, Syd), Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  auto const thermal_diff = (1.0 / PI) * (1.0 / PI);
  impl_solver.set_2_order_coefficients(
      std::make_pair(thermal_diff, thermal_diff));
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::Euler);
  // get exact solution:
  auto exact = [](T x, T y, T t, std::size_t n_x, std::size_t n_y) {
    T const first = static_cast<T>(1600.0) / (PI * PI);
    T sum{};
    T var_k{};
    T sin_k{};
    T var_l{};
    T sin_l{};
    T lam{};
    T exp_lam{};
    for (std::size_t k = 0; k <= n_x; ++k) {
      var_k = (2.0 * k + 1.0);
      sin_k = std::sin(var_k * PI * x);
      for (std::size_t l = 0; l <= n_y; ++l) {
        var_l = (2.0 * l + 1.0);
        sin_l = std::sin(var_l * PI * y);
        lam = var_k * var_k + var_l * var_l;
        exp_lam = std::exp(-lam * t);
        sum += (sin_l * sin_k * exp_lam) / (var_l * var_k);
      }
    }
    return (first * sum);
  };

  auto const &h = impl_solver.space_step();
  auto const h_1 = h.first;
  auto const h_2 = h.second;
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  T sol{};
  for (std::size_t r = 0; r < solution.rows(); ++r) {
    for (std::size_t c = 0; c < solution.columns(); ++c) {
      benchmark = exact(r * h_1, c * h_2, 0.10, 20, 20);
      sol = solution(r, c);
      std::cout << "(" << r << "," << c << "): " << sol << " |  " << benchmark
                << " | " << (sol - benchmark) << '\n';
    }
    std::cout << "\n";
  }
}

template <typename T>
void test2DImplPureHeatEquationDirichletBCDoubleSweepCN() {
  using lss_containers::container_2d;
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_double_sweep_solver::fdm_double_sweep_solver;
  using lss_two_dim_classic_pde_solvers::implicit_solvers::
      general_heat_equation;
  using lss_two_dim_pde_utility::dirichlet_boundary_2d;
  using lss_utility::range;
  using lss_utility::sptr_t;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using Double Sweep algorithm with Crank-Nicolson method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,y,t) + U_yy(x,y,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1>, y in <0,1> and t > 0,\n";
  std::cout << " U(0,y,t) = U(1,y,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0,t) = U(x,1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,y,0) = 100, x,y in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) "
               "*sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DHeatEquation
  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                fdm_double_sweep_solver, std::vector,
                                std::allocator<T>>
      implicit_solver;

  typedef container_2d<std::vector, T, std::allocator<T>> matrix_t;

  // number of space X subdivisions:
  std::size_t const Sxd = 100;
  // number of space y subdivisions:
  std::size_t const Syd = 50;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initial_condition = [](T x, T y) { return 100.0; };
  // boundary conditions:
  auto const &dirichlet_x_low = [](T y, T t) { return 0.0; };
  auto const &dirichlet_x_high = [](T y, T t) { return 0.0; };
  auto const &dirichlet_y_low = [](T x, T t) { return 0.0; };
  auto const &dirichlet_y_high = [](T x, T t) { return 0.0; };

  const auto &boundary = std::make_shared<dirichlet_boundary_2d<T>>(
      std::make_pair(dirichlet_x_low, dirichlet_x_high),
      std::make_pair(dirichlet_y_low, dirichlet_y_high));
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  matrix_t solution(Sxd + 1, Syd + 1, T{});
  // ranges of spatial variables:
  auto const &space_ranges =
      std::make_pair(range<T>(0.0, 1.0), range<T>(0.0, 1.0));

  // initialize solver
  implicit_solver impl_solver(space_ranges, 0.10, std::make_pair(Sxd, Syd), Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  auto const thermal_diff = (1.0 / PI) * (1.0 / PI);
  impl_solver.set_2_order_coefficients(
      std::make_pair(thermal_diff, thermal_diff));
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::CrankNicolson);
  // get exact solution:
  auto exact = [](T x, T y, T t, std::size_t n_x, std::size_t n_y) {
    T const first = static_cast<T>(1600.0) / (PI * PI);
    T sum{};
    T var_k{};
    T sin_k{};
    T var_l{};
    T sin_l{};
    T lam{};
    T exp_lam{};
    for (std::size_t k = 0; k <= n_x; ++k) {
      var_k = (2.0 * k + 1.0);
      sin_k = std::sin(var_k * PI * x);
      for (std::size_t l = 0; l <= n_y; ++l) {
        var_l = (2.0 * l + 1.0);
        sin_l = std::sin(var_l * PI * y);
        lam = var_k * var_k + var_l * var_l;
        exp_lam = std::exp(-lam * t);
        sum += (sin_l * sin_k * exp_lam) / (var_l * var_k);
      }
    }
    return (first * sum);
  };

  auto const &h = impl_solver.space_step();
  auto const h_1 = h.first;
  auto const h_2 = h.second;
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  T sol{};
  for (std::size_t r = 0; r < solution.rows(); ++r) {
    for (std::size_t c = 0; c < solution.columns(); ++c) {
      benchmark = exact(r * h_1, c * h_2, 0.10, 20, 20);
      sol = solution(r, c);
      std::cout << "(" << r << "," << c << "): " << sol << " |  " << benchmark
                << " | " << (sol - benchmark) << '\n';
    }
    std::cout << "\n";
  }
}

void test2DImplPureHeatEquationDirichletBCDoubleSweep() {
  std::cout << "============================================================\n";
  std::cout << "======== Implicit Pure Heat Equation (Dirichlet BC) ========\n";
  std::cout << "============================================================\n";

  test2DImplPureHeatEquationDirichletBCDoubleSweepEuler<double>();
  test2DImplPureHeatEquationDirichletBCDoubleSweepEuler<float>();
  test2DImplPureHeatEquationDirichletBCDoubleSweepCN<double>();
  test2DImplPureHeatEquationDirichletBCDoubleSweepCN<float>();

  std::cout << "============================================================\n";
}

// ===========================================================================
// ===== Heat problem with non-homogeneous Dirichlet boundary conditions =====
// ===========================================================================

template <typename T>
void test2DImplPureHeatEquationNonHomDirichletBCDoubleSweepEuler() {
  using lss_containers::container_2d;
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_double_sweep_solver::fdm_double_sweep_solver;
  using lss_two_dim_classic_pde_solvers::implicit_solvers::
      general_heat_equation;
  using lss_two_dim_pde_utility::dirichlet_boundary_2d;
  using lss_utility::range;
  using lss_utility::sptr_t;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,y,t) + U_yy(x,y,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1>, y in <0,1> and t > 0,\n";
  std::cout << " U(0,y,t) = U(1,y,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0,t) = 0, U(x,1,t) = 100, t > 0 \n\n";
  std::cout << " U(x,y,0) = 0, x,y in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DHeatEquation
  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                fdm_double_sweep_solver, std::vector,
                                std::allocator<T>>
      implicit_solver;

  typedef container_2d<std::vector, T, std::allocator<T>> matrix_t;

  // number of space X subdivisions:
  std::size_t const Sxd = 100;
  // number of space y subdivisions:
  std::size_t const Syd = 50;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initial_condition = [](T x, T y) { return 0.0; };
  // boundary conditions:
  auto const &dirichlet_x_low = [](T y, T t) { return 0.0; };
  auto const &dirichlet_x_high = [](T y, T t) { return 0.0; };
  auto const &dirichlet_y_low = [](T x, T t) { return 0.0; };
  auto const &dirichlet_y_high = [](T x, T t) { return 100.0; };

  const auto &boundary = std::make_shared<dirichlet_boundary_2d<T>>(
      std::make_pair(dirichlet_x_low, dirichlet_x_high),
      std::make_pair(dirichlet_y_low, dirichlet_y_high));
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  matrix_t solution(Sxd + 1, Syd + 1, T{});
  // ranges of spatial variables:
  auto const &space_ranges =
      std::make_pair(range<T>(0.0, 1.0), range<T>(0.0, 1.0));

  // initialize solver
  implicit_solver impl_solver(space_ranges, 0.2, std::make_pair(Sxd, Syd), Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  auto const thermal_diff = 1.0;
  impl_solver.set_2_order_coefficients(
      std::make_pair(thermal_diff, thermal_diff));
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::Euler);
  // get exact solution:
  auto exact = [](T x, T y, T t, std::size_t n_x, std::size_t n_y) {
    T const first = static_cast<T>(400.0) / (PI);
    T const second = static_cast<T>(800.0) / (PI * PI);
    T var_k{};
    T sum_first{};
    T sinh_k_c{};
    T sin_k_x{};
    T sinh_k_y{};
    for (std::size_t k = 1; k <= n_x; ++k) {
      var_k = 2.0 * k - 1.0;
      sinh_k_c = std::sinh(var_k * PI);
      sinh_k_y = std::sinh(var_k * y * PI);
      sin_k_x = std::sin(var_k * x * PI);
      sum_first += ((sin_k_x * sinh_k_y) / (var_k * sinh_k_c));
    }

    T sum_second{};
    T arg_m{};
    T sin_m{};
    T arg_n{};
    T sin_n{};
    T m_2{};
    T n_2{};
    T lam{};
    T exp_lam{};
    T arg = {};
    for (std::size_t m = 1; m <= n_x; ++m) {
      arg_m = (m * PI * x);
      sin_m = std::sin(arg_m);
      m_2 = m * m;
      for (std::size_t n = 1; n <= n_y; ++n) {
        arg_n = (n * PI * x);
        sin_n = std::sin(arg_n);
        n_2 = n * n;
        lam = PI * PI * (m_2 + n_2);
        exp_lam = std::exp(-lam * t);
        arg = std::pow(-1, n) * n / (m * (m_2 + n_2));
        sum_second += (arg * sin_m * sin_n * exp_lam);
      }
    }
    return ((first * sum_first) + (second * sum_second));
  };

  auto const &h = impl_solver.space_step();
  auto const h_1 = h.first;
  auto const h_2 = h.second;
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  T sol{};
  for (std::size_t r = 0; r < solution.rows(); ++r) {
    for (std::size_t c = 0; c < solution.columns(); ++c) {
      benchmark = exact(r * h_1, c * h_2, 0.2, 20, 20);
      sol = solution(r, c);
      std::cout << "(" << r << "," << c << "): " << sol << " |  " << benchmark
                << " | " << (sol - benchmark) << '\n';
    }
    std::cout << "\n";
  }
}

template <typename T>
void test2DImplPureHeatEquationNonHomDirichletBCDoubleSweepCN() {
  using lss_containers::container_2d;
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_double_sweep_solver::fdm_double_sweep_solver;
  using lss_two_dim_classic_pde_solvers::implicit_solvers::
      general_heat_equation;
  using lss_two_dim_pde_utility::dirichlet_boundary_2d;
  using lss_utility::range;
  using lss_utility::sptr_t;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,y,t) + U_yy(x,y,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1>, y in <0,1> and t > 0,\n";
  std::cout << " U(0,y,t) = U(1,y,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0,t) = 0, U(x,1,t) = 100, t > 0 \n\n";
  std::cout << " U(x,y,0) = 0, x,y in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DHeatEquation
  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                fdm_double_sweep_solver, std::vector,
                                std::allocator<T>>
      implicit_solver;

  typedef container_2d<std::vector, T, std::allocator<T>> matrix_t;

  // number of space X subdivisions:
  std::size_t const Sxd = 100;
  // number of space y subdivisions:
  std::size_t const Syd = 50;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initial_condition = [](T x, T y) { return 0.0; };
  // boundary conditions:
  auto const &dirichlet_x_low = [](T y, T t) { return 0.0; };
  auto const &dirichlet_x_high = [](T y, T t) { return 0.0; };
  auto const &dirichlet_y_low = [](T x, T t) { return 0.0; };
  auto const &dirichlet_y_high = [](T x, T t) { return 100.0; };

  const auto &boundary = std::make_shared<dirichlet_boundary_2d<T>>(
      std::make_pair(dirichlet_x_low, dirichlet_x_high),
      std::make_pair(dirichlet_y_low, dirichlet_y_high));
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  matrix_t solution(Sxd + 1, Syd + 1, T{});
  // ranges of spatial variables:
  auto const &space_ranges =
      std::make_pair(range<T>(0.0, 1.0), range<T>(0.0, 1.0));

  // initialize solver
  implicit_solver impl_solver(space_ranges, 0.20, std::make_pair(Sxd, Syd), Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  auto const thermal_diff = 1.0;
  impl_solver.set_2_order_coefficients(
      std::make_pair(thermal_diff, thermal_diff));
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::CrankNicolson);
  // get exact solution:
  auto exact = [](T x, T y, T t, std::size_t n_x, std::size_t n_y) {
    T const first = static_cast<T>(400.0) / (PI);
    T const second = static_cast<T>(800.0) / (PI * PI);
    T var_k{};
    T sum_first{0.0};
    T sinh_k_c{};
    T sin_k_x{};
    T sinh_k_y{};
    for (std::size_t k = 1; k <= n_x; ++k) {
      var_k = 2.0 * k - 1.0;
      sinh_k_c = std::sinh(var_k * PI);
      sinh_k_y = std::sinh(var_k * y * PI);
      sin_k_x = std::sin(var_k * x * PI);
      sum_first += ((sin_k_x * sinh_k_y) / (var_k * sinh_k_c));
    }

    T sum_second{0.0};
    T arg_m{};
    T sin_m{};
    T arg_n{};
    T sin_n{};
    T m_2{};
    T n_2{};
    T lam{};
    T exp_lam{};
    T arg = {};
    for (std::size_t m = 1; m <= n_x; ++m) {
      arg_m = (m * PI * x);
      sin_m = std::sin(arg_m);
      m_2 = m * m;
      for (std::size_t n = 1; n <= n_y; ++n) {
        arg_n = (n * PI * x);
        sin_n = std::sin(arg_n);
        n_2 = n * n;
        lam = PI * PI * (m_2 + n_2);
        exp_lam = std::exp(-lam * t);
        arg = std::pow(-1, n) * n / (m * (m_2 + n_2));
        sum_second += (arg * sin_m * sin_n * exp_lam);
      }
    }
    return ((first * sum_first) + (second * sum_second));
  };

  auto const &h = impl_solver.space_step();
  auto const h_1 = h.first;
  auto const h_2 = h.second;
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  T sol{};
  for (std::size_t r = 0; r < solution.rows(); ++r) {
    for (std::size_t c = 0; c < solution.columns(); ++c) {
      benchmark = exact(r * h_1, c * h_2, 0.2, 20, 20);
      sol = solution(r, c);
      std::cout << "(" << r << "," << c << "): " << sol << " |  " << benchmark
                << " | " << (sol - benchmark) << '\n';
    }
    std::cout << "\n";
  }
}

void test2DImplPureHeatEquationNonHomDirichletBCDoubleSweep() {
  std::cout << "============================================================\n";
  std::cout << "==== Implicit Pure Heat Equation (Non Hom Dirichlet BC) ====\n";
  std::cout << "============================================================\n";

  test2DImplPureHeatEquationNonHomDirichletBCDoubleSweepEuler<double>();
  test2DImplPureHeatEquationNonHomDirichletBCDoubleSweepEuler<float>();
  test2DImplPureHeatEquationNonHomDirichletBCDoubleSweepCN<double>();
  test2DImplPureHeatEquationNonHomDirichletBCDoubleSweepCN<float>();

  std::cout << "============================================================\n";
}

// ===========================================================================
// ===== Non-homogeneous Heat problem with Dirichlet boundary conditions =====
// ===========================================================================

template <typename T>
void test2DImplNonHomPureHeatEquationDirichletBCDoubleSweepEuler() {
  using lss_containers::container_2d;
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_double_sweep_solver::fdm_double_sweep_solver;
  using lss_two_dim_classic_pde_solvers::implicit_solvers::
      general_heat_equation;
  using lss_two_dim_pde_utility::dirichlet_boundary_2d;
  using lss_utility::range;
  using lss_utility::sptr_t;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,y,t) + U_yy(x,y,t) + G(x,y,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1>, y in <0,1> and t > 0,\n";
  std::cout << " U(0,y,t) = U(1,y,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0,t) = U(x,1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,y,0) = 100, x,y in <0,1> \n\n";
  std::cout << " G(x,y,t) = 0\n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) "
               "*sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DHeatEquation

  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                fdm_double_sweep_solver, std::vector,
                                std::allocator<T>>
      implicit_solver;

  typedef container_2d<std::vector, T, std::allocator<T>> matrix_t;

  // number of space X subdivisions:
  std::size_t const Sxd = 100;
  // number of space y subdivisions:
  std::size_t const Syd = 50;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initial_condition = [](T x, T y) { return 100.0; };
  // boundary conditions:
  auto const &dirichlet_x_low = [](T y, T t) { return 0.0; };
  auto const &dirichlet_x_high = [](T y, T t) { return 0.0; };
  auto const &dirichlet_y_low = [](T x, T t) { return 0.0; };
  auto const &dirichlet_y_high = [](T x, T t) { return 0.0; };

  const auto &boundary = std::make_shared<dirichlet_boundary_2d<T>>(
      std::make_pair(dirichlet_x_low, dirichlet_x_high),
      std::make_pair(dirichlet_y_low, dirichlet_y_high));
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  matrix_t solution(Sxd + 1, Syd + 1, T{});
  // ranges of spatial variables:
  auto const &space_ranges =
      std::make_pair(range<T>(0.0, 1.0), range<T>(0.0, 1.0));

  // initialize solver
  implicit_solver impl_solver(space_ranges, 0.10, std::make_pair(Sxd, Syd), Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set source function:
  impl_solver.set_heat_source([](T x, T y, T t) { return 0.0; });
  // set thermal diffusivity (C^2 in PDE)
  auto const thermal_diff = (1.0 / PI) * (1.0 / PI);
  impl_solver.set_2_order_coefficients(
      std::make_pair(thermal_diff, thermal_diff));
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::Euler);
  // get exact solution:
  auto exact = [](T x, T y, T t, std::size_t n_x, std::size_t n_y) {
    T const first = static_cast<T>(1600.0) / (PI * PI);
    T sum{};
    T var_k{};
    T sin_k{};
    T var_l{};
    T sin_l{};
    T lam{};
    T exp_lam{};
    for (std::size_t k = 0; k <= n_x; ++k) {
      var_k = (2.0 * k + 1.0);
      sin_k = std::sin(var_k * PI * x);
      for (std::size_t l = 0; l <= n_y; ++l) {
        var_l = (2.0 * l + 1.0);
        sin_l = std::sin(var_l * PI * y);
        lam = var_k * var_k + var_l * var_l;
        exp_lam = std::exp(-lam * t);
        sum += (sin_l * sin_k * exp_lam) / (var_l * var_k);
      }
    }
    return (first * sum);
  };

  auto const &h = impl_solver.space_step();
  auto const h_1 = h.first;
  auto const h_2 = h.second;
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  T sol{};
  for (std::size_t r = 0; r < solution.rows(); ++r) {
    for (std::size_t c = 0; c < solution.columns(); ++c) {
      benchmark = exact(r * h_1, c * h_2, 0.10, 20, 20);
      sol = solution(r, c);
      std::cout << "(" << r << "," << c << "): " << sol << " |  " << benchmark
                << " | " << (sol - benchmark) << '\n';
    }
    std::cout << "\n";
  }
}

template <typename T>
void test2DImplNonHomPureHeatEquationDirichletBCDoubleSweepCN() {
  using lss_containers::container_2d;
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_double_sweep_solver::fdm_double_sweep_solver;
  using lss_two_dim_classic_pde_solvers::implicit_solvers::
      general_heat_equation;
  using lss_two_dim_pde_utility::dirichlet_boundary_2d;
  using lss_utility::range;
  using lss_utility::sptr_t;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using Double Sweep algorithm with Crank-Nicolson method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,y,t) + U_yy(x,y,t) + G(x,y,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1>, y in <0,1> and t > 0,\n";
  std::cout << " U(0,y,t) = U(1,y,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0,t) = U(x,1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,y,0) = 100, x,y in <0,1> \n\n";
  std::cout << " G(x,y,t) = 0\n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) "
               "*sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DHeatEquation
  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                fdm_double_sweep_solver, std::vector,
                                std::allocator<T>>
      implicit_solver;

  typedef container_2d<std::vector, T, std::allocator<T>> matrix_t;

  // number of space X subdivisions:
  std::size_t const Sxd = 100;
  // number of space y subdivisions:
  std::size_t const Syd = 50;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initial_condition = [](T x, T y) { return 100.0; };
  // boundary conditions:
  auto const &dirichlet_x_low = [](T y, T t) { return 0.0; };
  auto const &dirichlet_x_high = [](T y, T t) { return 0.0; };
  auto const &dirichlet_y_low = [](T x, T t) { return 0.0; };
  auto const &dirichlet_y_high = [](T x, T t) { return 0.0; };

  const auto &boundary = std::make_shared<dirichlet_boundary_2d<T>>(
      std::make_pair(dirichlet_x_low, dirichlet_x_high),
      std::make_pair(dirichlet_y_low, dirichlet_y_high));
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  matrix_t solution(Sxd + 1, Syd + 1, T{});
  // ranges of spatial variables:
  auto const &space_ranges =
      std::make_pair(range<T>(0.0, 1.0), range<T>(0.0, 1.0));

  // initialize solver
  implicit_solver impl_solver(space_ranges, 0.10, std::make_pair(Sxd, Syd), Td);
  // set boundary conditions:
  impl_solver.set_boundary_condition(boundary);
  // set initial condition:
  impl_solver.set_initial_condition(initial_condition);
  // set source function:
  impl_solver.set_heat_source([](T x, T y, T t) { return 0.0; });
  // set thermal diffusivity (C^2 in PDE)
  auto const thermal_diff = (1.0 / PI) * (1.0 / PI);
  impl_solver.set_2_order_coefficients(
      std::make_pair(thermal_diff, thermal_diff));
  // get the solution:
  impl_solver.solve(solution, implicit_pde_schemes_enum::CrankNicolson);
  // get exact solution:
  auto exact = [](T x, T y, T t, std::size_t n_x, std::size_t n_y) {
    T const first = static_cast<T>(1600.0) / (PI * PI);
    T sum{};
    T var_k{};
    T sin_k{};
    T var_l{};
    T sin_l{};
    T lam{};
    T exp_lam{};
    for (std::size_t k = 0; k <= n_x; ++k) {
      var_k = (2.0 * k + 1.0);
      sin_k = std::sin(var_k * PI * x);
      for (std::size_t l = 0; l <= n_y; ++l) {
        var_l = (2.0 * l + 1.0);
        sin_l = std::sin(var_l * PI * y);
        lam = var_k * var_k + var_l * var_l;
        exp_lam = std::exp(-lam * t);
        sum += (sin_l * sin_k * exp_lam) / (var_l * var_k);
      }
    }
    return (first * sum);
  };

  auto const &h = impl_solver.space_step();
  auto const h_1 = h.first;
  auto const h_2 = h.second;
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  T sol{};
  for (std::size_t r = 0; r < solution.rows(); ++r) {
    for (std::size_t c = 0; c < solution.columns(); ++c) {
      benchmark = exact(r * h_1, c * h_2, 0.10, 20, 20);
      sol = solution(r, c);
      std::cout << "(" << r << "," << c << "): " << sol << " |  " << benchmark
                << " | " << (sol - benchmark) << '\n';
    }
    std::cout << "\n";
  }
}

void test2DImplNonHomPureHeatEquationDirichletBCDoubleSweep() {
  std::cout << "============================================================\n";
  std::cout << "==== Implicit Non-Hom Pure Heat Equation (Dirichlet BC) ====\n";
  std::cout << "============================================================\n";

  test2DImplNonHomPureHeatEquationDirichletBCDoubleSweepEuler<double>();
  test2DImplNonHomPureHeatEquationDirichletBCDoubleSweepEuler<float>();
  test2DImplNonHomPureHeatEquationDirichletBCDoubleSweepCN<double>();
  test2DImplNonHomPureHeatEquationDirichletBCDoubleSweepCN<float>();

  std::cout << "============================================================\n";
}

// ===========================================================================
// ========================== EXPLICIT SOLVERS ===============================
// ===========================================================================

// ===========================================================================
// =========== Heat problem with homogeneous boundary conditions =============
// ===========================================================================

template <typename T>
void test2DExplPureHeatEquationDirichletBCEuler() {
  using lss_containers::container_2d;
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::explicit_pde_schemes_enum;
  using lss_two_dim_classic_pde_solvers::explicit_solvers::
      general_heat_equation;
  using lss_two_dim_pde_utility::dirichlet_boundary_2d;
  using lss_utility::range;
  using lss_utility::sptr_t;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using ADE Euler\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,y,t) + U_yy(x,y,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1>, y in <0,1> and t > 0,\n";
  std::cout << " U(0,y,t) = U(1,y,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0,t) = U(x,1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,y,0) = 100, x,y in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) "
               "*sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DHeatEquation

  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                std::vector, std::allocator<T>>
      explicit_solver;

  typedef container_2d<std::vector, T, std::allocator<T>> matrix_t;

  // number of space X subdivisions:
  std::size_t const Sxd = 100;
  // number of space y subdivisions:
  std::size_t const Syd = 70;
  // number of time subdivisions:
  std::size_t const Td = 500;
  // initial condition:
  auto initial_condition = [](T x, T y) { return 100.0; };
  // boundary conditions:
  auto const &dirichlet_x_low = [](T y, T t) { return 0.0; };
  auto const &dirichlet_x_high = [](T y, T t) { return 0.0; };
  auto const &dirichlet_y_low = [](T x, T t) { return 0.0; };
  auto const &dirichlet_y_high = [](T x, T t) { return 0.0; };

  const auto &boundary = std::make_shared<dirichlet_boundary_2d<T>>(
      std::make_pair(dirichlet_x_low, dirichlet_x_high),
      std::make_pair(dirichlet_y_low, dirichlet_y_high));
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  matrix_t solution(Sxd + 1, Syd + 1, T{});
  // ranges of spatial variables:
  auto const &space_ranges =
      std::make_pair(range<T>(0.0, 1.0), range<T>(0.0, 1.0));

  // initialize solver
  explicit_solver expl_solver(space_ranges, 0.10, std::make_pair(Sxd, Syd), Td);
  // set boundary conditions:
  expl_solver.set_boundary_condition(boundary);
  // set initial condition:
  expl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  auto const thermal_diff = (1.0 / PI) * (1.0 / PI);
  expl_solver.set_2_order_coefficients(
      std::make_pair(thermal_diff, thermal_diff));
  // get the solution:
  expl_solver.solve(solution, explicit_pde_schemes_enum::Euler);
  // get exact solution:
  auto exact = [](T x, T y, T t, std::size_t n_x, std::size_t n_y) {
    T const first = static_cast<T>(1600.0) / (PI * PI);
    T sum{};
    T var_k{};
    T sin_k{};
    T var_l{};
    T sin_l{};
    T lam{};
    T exp_lam{};
    for (std::size_t k = 0; k <= n_x; ++k) {
      var_k = (2.0 * k + 1.0);
      sin_k = std::sin(var_k * PI * x);
      for (std::size_t l = 0; l <= n_y; ++l) {
        var_l = (2.0 * l + 1.0);
        sin_l = std::sin(var_l * PI * y);
        lam = var_k * var_k + var_l * var_l;
        exp_lam = std::exp(-lam * t);
        sum += (sin_l * sin_k * exp_lam) / (var_l * var_k);
      }
    }
    return (first * sum);
  };

  auto const &h = expl_solver.space_step();
  auto const h_1 = h.first;
  auto const h_2 = h.second;
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  T sol{};
  for (std::size_t r = 0; r < solution.rows(); ++r) {
    for (std::size_t c = 0; c < solution.columns(); ++c) {
      benchmark = exact(r * h_1, c * h_2, 0.10, 20, 20);
      sol = solution(r, c);
      std::cout << "(" << r << "," << c << "): " << sol << " |  " << benchmark
                << " | " << (sol - benchmark) << '\n';
    }
    std::cout << "\n";
  }
}

template <typename T>
void test2DExplPureHeatEquationDirichletBCADEBarakatClark() {
  using lss_containers::container_2d;
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::explicit_pde_schemes_enum;
  using lss_two_dim_classic_pde_solvers::explicit_solvers::
      general_heat_equation;
  using lss_two_dim_pde_utility::dirichlet_boundary_2d;
  using lss_utility::range;
  using lss_utility::sptr_t;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using ADE Barakat Clark\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,y,t) + U_yy(x,y,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1>, y in <0,1> and t > 0,\n";
  std::cout << " U(0,y,t) = U(1,y,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0,t) = U(x,1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,y,0) = 100, x,y in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) "
               "*sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DHeatEquation

  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                std::vector, std::allocator<T>>
      explicit_solver;

  typedef container_2d<std::vector, T, std::allocator<T>> matrix_t;

  // number of space X subdivisions:
  std::size_t const Sxd = 100;
  // number of space y subdivisions:
  std::size_t const Syd = 70;
  // number of time subdivisions:
  std::size_t const Td = 1000;
  // initial condition:
  auto initial_condition = [](T x, T y) { return 100.0; };
  // boundary conditions:
  auto const &dirichlet_x_low = [](T y, T t) { return 0.0; };
  auto const &dirichlet_x_high = [](T y, T t) { return 0.0; };
  auto const &dirichlet_y_low = [](T x, T t) { return 0.0; };
  auto const &dirichlet_y_high = [](T x, T t) { return 0.0; };

  const auto &boundary = std::make_shared<dirichlet_boundary_2d<T>>(
      std::make_pair(dirichlet_x_low, dirichlet_x_high),
      std::make_pair(dirichlet_y_low, dirichlet_y_high));
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  matrix_t solution(Sxd + 1, Syd + 1, T{});
  // ranges of spatial variables:
  auto const &space_ranges =
      std::make_pair(range<T>(0.0, 1.0), range<T>(0.0, 1.0));

  // initialize solver
  explicit_solver expl_solver(space_ranges, 0.10, std::make_pair(Sxd, Syd), Td);
  // set boundary conditions:
  expl_solver.set_boundary_condition(boundary);
  // set initial condition:
  expl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  auto const thermal_diff = (1.0 / PI) * (1.0 / PI);
  expl_solver.set_2_order_coefficients(
      std::make_pair(thermal_diff, thermal_diff));
  // get the solution:
  expl_solver.solve(solution, explicit_pde_schemes_enum::ADEBarakatClark);
  // get exact solution:
  auto exact = [](T x, T y, T t, std::size_t n_x, std::size_t n_y) {
    T const first = static_cast<T>(1600.0) / (PI * PI);
    T sum{};
    T var_k{};
    T sin_k{};
    T var_l{};
    T sin_l{};
    T lam{};
    T exp_lam{};
    for (std::size_t k = 0; k <= n_x; ++k) {
      var_k = (2.0 * k + 1.0);
      sin_k = std::sin(var_k * PI * x);
      for (std::size_t l = 0; l <= n_y; ++l) {
        var_l = (2.0 * l + 1.0);
        sin_l = std::sin(var_l * PI * y);
        lam = var_k * var_k + var_l * var_l;
        exp_lam = std::exp(-lam * t);
        sum += (sin_l * sin_k * exp_lam) / (var_l * var_k);
      }
    }
    return (first * sum);
  };

  auto const &h = expl_solver.space_step();
  auto const h_1 = h.first;
  auto const h_2 = h.second;
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  T sol{};
  for (std::size_t r = 0; r < solution.rows(); ++r) {
    for (std::size_t c = 0; c < solution.columns(); ++c) {
      benchmark = exact(r * h_1, c * h_2, 0.10, 20, 20);
      sol = solution(r, c);
      std::cout << "(" << r << "," << c << "): " << sol << " |  " << benchmark
                << " | " << (sol - benchmark) << '\n';
    }
    std::cout << "\n";
  }
}

template <typename T>
void test2DExplPureHeatEquationDirichletBCADESaulyev() {
  using lss_containers::container_2d;
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::explicit_pde_schemes_enum;
  using lss_two_dim_classic_pde_solvers::explicit_solvers::
      general_heat_equation;
  using lss_two_dim_pde_utility::dirichlet_boundary_2d;
  using lss_utility::range;
  using lss_utility::sptr_t;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using ADE Saulyev scheme\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,y,t) + U_yy(x,y,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1>, y in <0,1> and t > 0,\n";
  std::cout << " U(0,y,t) = U(1,y,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0,t) = U(x,1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,y,0) = 100, x,y in <0,1> \n\n";
  std::cout << " Exact solution: \n";
  std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) "
               "*sin(n*pi*x)/n}\n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DHeatEquation
  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                std::vector, std::allocator<T>>
      explicit_solver;

  typedef container_2d<std::vector, T, std::allocator<T>> matrix_t;

  // number of space X subdivisions:
  std::size_t const Sxd = 100;
  // number of space y subdivisions:
  std::size_t const Syd = 70;
  // number of time subdivisions:
  std::size_t const Td = 1000;
  // initial condition:
  auto initial_condition = [](T x, T y) { return 100.0; };
  // boundary conditions:
  auto const &dirichlet_x_low = [](T y, T t) { return 0.0; };
  auto const &dirichlet_x_high = [](T y, T t) { return 0.0; };
  auto const &dirichlet_y_low = [](T x, T t) { return 0.0; };
  auto const &dirichlet_y_high = [](T x, T t) { return 0.0; };

  const auto &boundary = std::make_shared<dirichlet_boundary_2d<T>>(
      std::make_pair(dirichlet_x_low, dirichlet_x_high),
      std::make_pair(dirichlet_y_low, dirichlet_y_high));
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  matrix_t solution(Sxd + 1, Syd + 1, T{});
  // ranges of spatial variables:
  auto const &space_ranges =
      std::make_pair(range<T>(0.0, 1.0), range<T>(0.0, 1.0));

  // initialize solver
  explicit_solver expl_solver(space_ranges, 0.10, std::make_pair(Sxd, Syd), Td);
  // set boundary conditions:
  expl_solver.set_boundary_condition(boundary);
  // set initial condition:
  expl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  auto const thermal_diff = (1.0 / PI) * (1.0 / PI);
  expl_solver.set_2_order_coefficients(
      std::make_pair(thermal_diff, thermal_diff));
  // get the solution:
  expl_solver.solve(solution, explicit_pde_schemes_enum::ADESaulyev);
  // get exact solution:
  auto exact = [](T x, T y, T t, std::size_t n_x, std::size_t n_y) {
    T const first = static_cast<T>(1600.0) / (PI * PI);
    T sum{};
    T var_k{};
    T sin_k{};
    T var_l{};
    T sin_l{};
    T lam{};
    T exp_lam{};
    for (std::size_t k = 0; k <= n_x; ++k) {
      var_k = (2.0 * k + 1.0);
      sin_k = std::sin(var_k * PI * x);
      for (std::size_t l = 0; l <= n_y; ++l) {
        var_l = (2.0 * l + 1.0);
        sin_l = std::sin(var_l * PI * y);
        lam = var_k * var_k + var_l * var_l;
        exp_lam = std::exp(-lam * t);
        sum += (sin_l * sin_k * exp_lam) / (var_l * var_k);
      }
    }
    return (first * sum);
  };

  auto const &h = expl_solver.space_step();
  auto const h_1 = h.first;
  auto const h_2 = h.second;
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  T sol{};
  for (std::size_t r = 0; r < solution.rows(); ++r) {
    for (std::size_t c = 0; c < solution.columns(); ++c) {
      benchmark = exact(r * h_1, c * h_2, 0.10, 20, 20);
      sol = solution(r, c);
      std::cout << "(" << r << "," << c << "): " << sol << " |  " << benchmark
                << " | " << (sol - benchmark) << '\n';
    }
    std::cout << "\n";
  }
}

void test2DExplPureHeatEquationDirichletBCADE() {
  std::cout << "============================================================\n";
  std::cout << "======== Explicit Pure Heat Equation (Dirichlet BC) ========\n";
  std::cout << "============================================================\n";

  test2DExplPureHeatEquationDirichletBCEuler<double>();
  test2DExplPureHeatEquationDirichletBCEuler<float>();
  test2DExplPureHeatEquationDirichletBCADEBarakatClark<double>();
  test2DExplPureHeatEquationDirichletBCADEBarakatClark<float>();
  test2DExplPureHeatEquationDirichletBCADESaulyev<double>();
  test2DExplPureHeatEquationDirichletBCADESaulyev<float>();

  std::cout << "============================================================\n";
}

// ===========================================================================
// ===== Heat problem with non-homogeneous Dirichlet boundary conditions =====
// ===========================================================================

template <typename T>
void test2DExplPureHeatEquationNonHomDirichletBCEuler() {
  using lss_containers::container_2d;
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::explicit_pde_schemes_enum;
  using lss_two_dim_classic_pde_solvers::explicit_solvers::
      general_heat_equation;
  using lss_two_dim_pde_utility::dirichlet_boundary_2d;
  using lss_utility::range;
  using lss_utility::sptr_t;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,y,t) + U_yy(x,y,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1>, y in <0,1> and t > 0,\n";
  std::cout << " U(0,y,t) = U(1,y,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0,t) = 0, U(x,1,t) = 100, t > 0 \n\n";
  std::cout << " U(x,y,0) = 0, x,y in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DHeatEquation
  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                std::vector, std::allocator<T>>
      explicit_solver;

  typedef container_2d<std::vector, T, std::allocator<T>> matrix_t;

  // number of space X subdivisions:
  std::size_t const Sxd = 100;
  // number of space y subdivisions:
  std::size_t const Syd = 70;
  // number of time subdivisions:
  std::size_t const Td = 8100;
  // initial condition:
  auto initial_condition = [](T x, T y) { return 0.0; };
  // boundary conditions:
  auto const &dirichlet_x_low = [](T y, T t) { return 0.0; };
  auto const &dirichlet_x_high = [](T y, T t) { return 0.0; };
  auto const &dirichlet_y_low = [](T x, T t) { return 0.0; };
  auto const &dirichlet_y_high = [](T x, T t) { return 100.0; };

  const auto &boundary = std::make_shared<dirichlet_boundary_2d<T>>(
      std::make_pair(dirichlet_x_low, dirichlet_x_high),
      std::make_pair(dirichlet_y_low, dirichlet_y_high));
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  matrix_t solution(Sxd + 1, Syd + 1, T{});
  // ranges of spatial variables:
  auto const &space_ranges =
      std::make_pair(range<T>(0.0, 1.0), range<T>(0.0, 1.0));

  // initialize solver
  explicit_solver expl_solver(space_ranges, 0.2, std::make_pair(Sxd, Syd), Td);
  // set boundary conditions:
  expl_solver.set_boundary_condition(boundary);
  // set initial condition:
  expl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  auto const thermal_diff = 1.0;
  expl_solver.set_2_order_coefficients(
      std::make_pair(thermal_diff, thermal_diff));
  // get the solution:
  expl_solver.solve(solution, explicit_pde_schemes_enum::Euler);
  // get exact solution:
  auto exact = [](T x, T y, T t, std::size_t n_x, std::size_t n_y) {
    T const first = static_cast<T>(400.0) / (PI);
    T const second = static_cast<T>(800.0) / (PI * PI);
    T var_k{};
    T sum_first{};
    T sinh_k_c{};
    T sin_k_x{};
    T sinh_k_y{};
    for (std::size_t k = 1; k <= n_x; ++k) {
      var_k = 2.0 * k - 1.0;
      sinh_k_c = std::sinh(var_k * PI);
      sinh_k_y = std::sinh(var_k * y * PI);
      sin_k_x = std::sin(var_k * x * PI);
      sum_first += ((sin_k_x * sinh_k_y) / (var_k * sinh_k_c));
    }

    T sum_second{};
    T arg_m{};
    T sin_m{};
    T arg_n{};
    T sin_n{};
    T m_2{};
    T n_2{};
    T lam{};
    T exp_lam{};
    T arg = {};
    for (std::size_t m = 1; m <= n_x; ++m) {
      arg_m = (m * PI * x);
      sin_m = std::sin(arg_m);
      m_2 = m * m;
      for (std::size_t n = 1; n <= n_y; ++n) {
        arg_n = (n * PI * x);
        sin_n = std::sin(arg_n);
        n_2 = n * n;
        lam = PI * PI * (m_2 + n_2);
        exp_lam = std::exp(-lam * t);
        arg = std::pow(-1, n) * n / (m * (m_2 + n_2));
        sum_second += (arg * sin_m * sin_n * exp_lam);
      }
    }
    return ((first * sum_first) + (second * sum_second));
  };

  auto const &h = expl_solver.space_step();
  auto const h_1 = h.first;
  auto const h_2 = h.second;
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  T sol{};
  for (std::size_t r = 0; r < solution.rows(); ++r) {
    for (std::size_t c = 0; c < solution.columns(); ++c) {
      benchmark = exact(r * h_1, c * h_2, 0.2, 20, 20);
      sol = solution(r, c);
      std::cout << "(" << r << "," << c << "): " << sol << " |  " << benchmark
                << " | " << (sol - benchmark) << '\n';
    }
    std::cout << "\n";
  }
}

template <typename T>
void test2DExplPureHeatEquationNonHomDirichletBCADEBarakatClark() {
  using lss_containers::container_2d;
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::explicit_pde_schemes_enum;
  using lss_two_dim_classic_pde_solvers::explicit_solvers::
      general_heat_equation;
  using lss_two_dim_pde_utility::dirichlet_boundary_2d;
  using lss_utility::range;
  using lss_utility::sptr_t;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using ADE Barakat-Clark method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,y,t) + U_yy(x,y,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1>, y in <0,1> and t > 0,\n";
  std::cout << " U(0,y,t) = U(1,y,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0,t) = 0, U(x,1,t) = 100, t > 0 \n\n";
  std::cout << " U(x,y,0) = 0, x,y in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DHeatEquation
  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                std::vector, std::allocator<T>>
      explicit_solver;

  typedef container_2d<std::vector, T, std::allocator<T>> matrix_t;

  // number of space X subdivisions:
  std::size_t const Sxd = 100;
  // number of space y subdivisions:
  std::size_t const Syd = 50;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initial_condition = [](T x, T y) { return 0.0; };
  // boundary conditions:
  auto const &dirichlet_x_low = [](T y, T t) { return 0.0; };
  auto const &dirichlet_x_high = [](T y, T t) { return 0.0; };
  auto const &dirichlet_y_low = [](T x, T t) { return 0.0; };
  auto const &dirichlet_y_high = [](T x, T t) { return 100.0; };

  const auto &boundary = std::make_shared<dirichlet_boundary_2d<T>>(
      std::make_pair(dirichlet_x_low, dirichlet_x_high),
      std::make_pair(dirichlet_y_low, dirichlet_y_high));
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  matrix_t solution(Sxd + 1, Syd + 1, T{});
  // ranges of spatial variables:
  auto const &space_ranges =
      std::make_pair(range<T>(0.0, 1.0), range<T>(0.0, 1.0));

  // initialize solver
  explicit_solver expl_solver(space_ranges, 0.2, std::make_pair(Sxd, Syd), Td);
  // set boundary conditions:
  expl_solver.set_boundary_condition(boundary);
  // set initial condition:
  expl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  auto const thermal_diff = 1.0;
  expl_solver.set_2_order_coefficients(
      std::make_pair(thermal_diff, thermal_diff));
  // get the solution:
  expl_solver.solve(solution, explicit_pde_schemes_enum::ADEBarakatClark);
  // get exact solution:
  auto exact = [](T x, T y, T t, std::size_t n_x, std::size_t n_y) {
    T const first = static_cast<T>(400.0) / (PI);
    T const second = static_cast<T>(800.0) / (PI * PI);
    T var_k{};
    T sum_first{};
    T sinh_k_c{};
    T sin_k_x{};
    T sinh_k_y{};
    for (std::size_t k = 1; k <= n_x; ++k) {
      var_k = 2.0 * k - 1.0;
      sinh_k_c = std::sinh(var_k * PI);
      sinh_k_y = std::sinh(var_k * y * PI);
      sin_k_x = std::sin(var_k * x * PI);
      sum_first += ((sin_k_x * sinh_k_y) / (var_k * sinh_k_c));
    }

    T sum_second{};
    T arg_m{};
    T sin_m{};
    T arg_n{};
    T sin_n{};
    T m_2{};
    T n_2{};
    T lam{};
    T exp_lam{};
    T arg = {};
    for (std::size_t m = 1; m <= n_x; ++m) {
      arg_m = (m * PI * x);
      sin_m = std::sin(arg_m);
      m_2 = m * m;
      for (std::size_t n = 1; n <= n_y; ++n) {
        arg_n = (n * PI * x);
        sin_n = std::sin(arg_n);
        n_2 = n * n;
        lam = PI * PI * (m_2 + n_2);
        exp_lam = std::exp(-lam * t);
        arg = std::pow(-1, n) * n / (m * (m_2 + n_2));
        sum_second += (arg * sin_m * sin_n * exp_lam);
      }
    }
    return ((first * sum_first) + (second * sum_second));
  };

  auto const &h = expl_solver.space_step();
  auto const h_1 = h.first;
  auto const h_2 = h.second;
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  T sol{};
  for (std::size_t r = 0; r < solution.rows(); ++r) {
    for (std::size_t c = 0; c < solution.columns(); ++c) {
      benchmark = exact(r * h_1, c * h_2, 0.2, 20, 20);
      sol = solution(r, c);
      std::cout << "(" << r << "," << c << "): " << sol << " |  " << benchmark
                << " | " << (sol - benchmark) << '\n';
    }
    std::cout << "\n";
  }
}

template <typename T>
void test2DExplPureHeatEquationNonHomDirichletBCADESaulyev() {
  using lss_containers::container_2d;
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::explicit_pde_schemes_enum;
  using lss_two_dim_classic_pde_solvers::explicit_solvers::
      general_heat_equation;
  using lss_two_dim_pde_utility::dirichlet_boundary_2d;
  using lss_utility::range;
  using lss_utility::sptr_t;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using ADE Saulyev method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,y,t) + U_yy(x,y,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1>, y in <0,1> and t > 0,\n";
  std::cout << " U(0,y,t) = U(1,y,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0,t) = 0, U(x,1,t) = 100, t > 0 \n\n";
  std::cout << " U(x,y,0) = 0, x,y in <0,1> \n\n";
  std::cout << "============================================================\n";

  // typedef the Implicit1DHeatEquation
  typedef general_heat_equation<T, boundary_condition_enum::Dirichlet,
                                std::vector, std::allocator<T>>
      explicit_solver;

  typedef container_2d<std::vector, T, std::allocator<T>> matrix_t;

  // number of space X subdivisions:
  std::size_t const Sxd = 100;
  // number of space y subdivisions:
  std::size_t const Syd = 50;
  // number of time subdivisions:
  std::size_t const Td = 100;
  // initial condition:
  auto initial_condition = [](T x, T y) { return 0.0; };
  // boundary conditions:
  auto const &dirichlet_x_low = [](T y, T t) { return 0.0; };
  auto const &dirichlet_x_high = [](T y, T t) { return 0.0; };
  auto const &dirichlet_y_low = [](T x, T t) { return 0.0; };
  auto const &dirichlet_y_high = [](T x, T t) { return 100.0; };

  const auto &boundary = std::make_shared<dirichlet_boundary_2d<T>>(
      std::make_pair(dirichlet_x_low, dirichlet_x_high),
      std::make_pair(dirichlet_y_low, dirichlet_y_high));
  // prepare container for solution:
  // note: size is Sd+1 since we must include space point at x = 0
  matrix_t solution(Sxd + 1, Syd + 1, T{});
  // ranges of spatial variables:
  auto const &space_ranges =
      std::make_pair(range<T>(0.0, 1.0), range<T>(0.0, 1.0));

  // initialize solver
  explicit_solver expl_solver(space_ranges, 0.20, std::make_pair(Sxd, Syd), Td);
  // set boundary conditions:
  expl_solver.set_boundary_condition(boundary);
  // set initial condition:
  expl_solver.set_initial_condition(initial_condition);
  // set thermal diffusivity (C^2 in PDE)
  auto const thermal_diff = 1.0;
  expl_solver.set_2_order_coefficients(
      std::make_pair(thermal_diff, thermal_diff));
  // get the solution:
  expl_solver.solve(solution, explicit_pde_schemes_enum::ADESaulyev);
  // get exact solution:
  auto exact = [](T x, T y, T t, std::size_t n_x, std::size_t n_y) {
    T const first = static_cast<T>(400.0) / (PI);
    T const second = static_cast<T>(800.0) / (PI * PI);
    T var_k{};
    T sum_first{0.0};
    T sinh_k_c{};
    T sin_k_x{};
    T sinh_k_y{};
    for (std::size_t k = 1; k <= n_x; ++k) {
      var_k = 2.0 * k - 1.0;
      sinh_k_c = std::sinh(var_k * PI);
      sinh_k_y = std::sinh(var_k * y * PI);
      sin_k_x = std::sin(var_k * x * PI);
      sum_first += ((sin_k_x * sinh_k_y) / (var_k * sinh_k_c));
    }

    T sum_second{0.0};
    T arg_m{};
    T sin_m{};
    T arg_n{};
    T sin_n{};
    T m_2{};
    T n_2{};
    T lam{};
    T exp_lam{};
    T arg = {};
    for (std::size_t m = 1; m <= n_x; ++m) {
      arg_m = (m * PI * x);
      sin_m = std::sin(arg_m);
      m_2 = m * m;
      for (std::size_t n = 1; n <= n_y; ++n) {
        arg_n = (n * PI * x);
        sin_n = std::sin(arg_n);
        n_2 = n * n;
        lam = PI * PI * (m_2 + n_2);
        exp_lam = std::exp(-lam * t);
        arg = std::pow(-1, n) * n / (m * (m_2 + n_2));
        sum_second += (arg * sin_m * sin_n * exp_lam);
      }
    }
    return ((first * sum_first) + (second * sum_second));
  };

  auto const &h = expl_solver.space_step();
  auto const h_1 = h.first;
  auto const h_2 = h.second;
  std::cout << "tp : FDM | Exact | Abs Diff\n";
  T benchmark{};
  T sol{};
  for (std::size_t r = 0; r < solution.rows(); ++r) {
    for (std::size_t c = 0; c < solution.columns(); ++c) {
      benchmark = exact(r * h_1, c * h_2, 0.2, 20, 20);
      sol = solution(r, c);
      std::cout << "(" << r << "," << c << "): " << sol << " |  " << benchmark
                << " | " << (sol - benchmark) << '\n';
    }
    std::cout << "\n";
  }
}

void test2DExplPureHeatEquationNonHomDirichletBCADE() {
  std::cout << "============================================================\n";
  std::cout << "==== Explicit Pure Heat Equation (Non-Hom Dirichlet BC) ====\n";
  std::cout << "============================================================\n";

  test2DExplPureHeatEquationNonHomDirichletBCEuler<double>();
  test2DExplPureHeatEquationNonHomDirichletBCEuler<float>();
  test2DExplPureHeatEquationNonHomDirichletBCADEBarakatClark<double>();
  test2DExplPureHeatEquationNonHomDirichletBCADEBarakatClark<float>();
  test2DExplPureHeatEquationNonHomDirichletBCADESaulyev<double>();
  test2DExplPureHeatEquationNonHomDirichletBCADESaulyev<float>();

  std::cout << "============================================================\n";
}

#endif  ///_LSS_TWO_DIM_PURE_HEAT_EQUATION_T
