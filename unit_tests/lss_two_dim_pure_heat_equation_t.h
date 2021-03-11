#pragma once
#if !defined(_LSS_TWO_DIM_PURE_HEAT_EQUATION_T)
#define _LSS_TWO_DIM_PURE_HEAT_EQUATION_T

#pragma warning(disable : 4305)

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
  using lss_enumerations::boundary_condition_enum;
  using lss_enumerations::implicit_pde_schemes_enum;
  using lss_fdm_double_sweep_solver::fdm_double_sweep_solver;
  using lss_two_dim_classic_pde_solvers::implicit_solvers::
      general_heat_equation;
  using lss_two_dim_pde_utility::dirichlet_boundary_2d;
  using lss_utility::container_2d;
  using lss_utility::range;

  std::cout << "============================================================\n";
  std::cout << "Solving Boundary-value Heat equation: \n\n";
  std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
  std::cout << " Value type: " << typeid(T).name() << "\n\n";
  std::cout << " U_t(x,t) = U_xx(x,y,t) + U_yy(x,y,t), \n\n";
  std::cout << " where\n\n";
  std::cout << " x in <0,1> and t > 0,\n";
  std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
  std::cout << " U(x,0) = x, x in <0,1> \n\n";
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

  dirichlet_boundary_2d<T> boundary(
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
  // auto exact = [](T x, T t, std::size_t n) {
  //  T const first = static_cast<T>(2.0) / PI;
  //  T sum{};
  //  T var1{};
  //  T var2{};
  //  for (std::size_t i = 1; i <= n; ++i) {
  //    var1 = std::pow(-1.0, i + 1) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
  //    var2 = std::sin(i * PI * x) / i;
  //    sum += (var1 * var2);
  //  }
  //  return (first * sum);
  //};

  // T const h = impl_solver.space_step();
  // std::cout << "tp : FDM | Exact | Abs Diff\n";
  // T benchmark{};
  // for (std::size_t j = 0; j < solution.size(); ++j) {
  //  benchmark = exact(j * h, 0.10, 20);
  //  std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark
  //            << " | " << (solution[j] - benchmark) << '\n';
  //}
}

void test2DImplPureHeatEquationDirichletBCDoubleSweep() {
  std::cout << "============================================================\n";
  std::cout << "======== Implicit Pure Heat Equation (Dirichlet BC) ========\n";
  std::cout << "============================================================\n";

  test2DImplPureHeatEquationDirichletBCDoubleSweepEuler<double>();
  test2DImplPureHeatEquationDirichletBCDoubleSweepEuler<float>();

  std::cout << "============================================================\n";
}

#endif  ///_LSS_TWO_DIM_PURE_HEAT_EQUATION_T
