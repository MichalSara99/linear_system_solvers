#pragma once
#if !defined(_LSS_1D_SPACE_VARIABLE_GENERAL_HEAT_EQUATION_ROBIN_SOLVERS)
#define _LSS_1D_SPACE_VARIABLE_GENERAL_HEAT_EQUATION_ROBIN_SOLVERS

#include <functional>

#include "common/lss_enumerations.h"
#include "common/lss_macros.h"
#include "common/lss_utility.h"
#include "lss_space_variable_general_heat_equation_solvers_base.h"
#include "lss_space_variable_heat_explicit_schemes.h"
#include "lss_space_variable_heat_implicit_schemes.h"
#include "pde_solvers/one_dim/lss_pde_boundary.h"
#include "pde_solvers/one_dim/lss_pde_utility.h"

namespace lss_one_dim_space_variable_pde_solvers {

using lss_enumerations::boundary_condition_enum;
using lss_enumerations::explicit_pde_schemes_enum;
using lss_enumerations::implicit_pde_schemes_enum;
using lss_one_dim_pde_utility::discretization;
using lss_one_dim_pde_utility::heat_data;
using lss_one_dim_pde_utility::pde_coefficient_holder_fun_1_arg;
using lss_one_dim_pde_utility::robin_boundary;
using lss_utility::range;
using lss_utility::sptr_t;
using lss_utility::uptr_t;

namespace implicit_solvers {

// ============================================================================
// ============ general_heat_equation Robin Specialisation Template ===========
// ============================================================================
/*!
   ============================================================================
   Represents general spacial variable 1D heat equation Robin solver

   u_t = a(x)*u_xx + b(x)*u_x + c(x)*u + F(x,t),
   t > 0, x_1 < x < x_2

   with initial condition

   u(x,0) = f(x)

   and Robin boundaries:

   d_1*u_x(x_1,t) + f_1*u(x_1,t) = A
   d_2*u_x(x_2,t) + f_2*u(x_2,t) = B

   It is assumed the Robin boundaries are discretised in following way:

   d_1*(U_1 - U_0)/h + f_1*(U_0 + U_1)/2 = A
   d_2*(U_N - U_N-1)/h + f_2*(U_N-1 + U_N)/2 = B

   And therefore can be rewritten in form:

   U_0 = ((2*d_1 + f_1*h)/(2*d_1 - f_1*h)) * U_1 +
                        (2*h/(f_1*h - 2*d_1))*A
   U_N-1 = ((2*d_2 + f_2*h)/(2*d_2 - f_2*h)) * U_N +
                        (2*h/(f_2*h -2*d_2))*B

   or

   U_0 = alpha * U_1 + phi,
   U_N-1 = beta * U_N + psi,

  ============================================================================
 */
template <typename fp_type,
          template <typename, boundary_condition_enum,
                    template <typename, typename> typename cont, typename>
          typename FDMSolver,
          template <typename, typename> typename container, typename alloc>
class general_heat_equation<fp_type, boundary_condition_enum::Robin, FDMSolver,
                            container, alloc> {
 private:
  typedef FDMSolver<fp_type, boundary_condition_enum::Robin, container, alloc>
      fdm_solver_t;
  typedef heat_data<fp_type> heat_data_t;

  uptr_t<fdm_solver_t> solverPtr_;    // finite-difference solver
  uptr_t<heat_data_t> dataPtr_;       // one-dim heat data
  robin_boundary<fp_type> boundary_;  // Robin boundary
  pde_coefficient_holder_fun_1_arg<fp_type> coeffs_;  // coefficients of PDE

 public:
  typedef fp_type value_type;
  explicit general_heat_equation() = delete;
  explicit general_heat_equation(range<fp_type> const &space_range,
                                 fp_type terminal_time,
                                 std::size_t const &space_discretization,
                                 std::size_t const &time_discretization)
      : solverPtr_{std::make_unique<fdm_solver_t>(space_discretization + 1)},
        dataPtr_{std::make_unique<heat_data_t>(
            space_range, range<fp_type>(fp_type{}, terminal_time),
            space_discretization, time_discretization, nullptr, nullptr,
            nullptr, false)} {}

  ~general_heat_equation() {}

  general_heat_equation(general_heat_equation const &) = delete;
  general_heat_equation(general_heat_equation &&) = delete;
  general_heat_equation &operator=(general_heat_equation const &) = delete;
  general_heat_equation &operator=(general_heat_equation &&) = delete;

  inline fp_type space_step() const {
    return ((dataPtr_->space_range.spread()) /
            static_cast<fp_type>(dataPtr_->space_division));
  }
  inline fp_type time_step() const {
    return ((dataPtr_->time_range.upper()) /
            static_cast<fp_type>(dataPtr_->time_division));
  }

  inline void set_boundary_condition(
      robin_boundary<fp_type> const &robin_boundary) {
    boundary_ = robin_boundary;
    solverPtr_->set_boundary_condition(boundary_.left, boundary_.right);
  }

  inline void set_initial_condition(
      std::function<fp_type(fp_type)> const &initial_condition) {
    dataPtr_->initial_condition = initial_condition;
  }
  inline void set_heat_source(
      std::function<fp_type(fp_type, fp_type)> const &heat_source) {
    dataPtr_->is_source_function_set = true;
    dataPtr_->source_function = heat_source;
  }
  inline void set_2_order_coefficient(
      std::function<fp_type(fp_type)> const &a) {
    std::get<0>(coeffs_) = a;
  }
  inline void set_1_order_coefficient(
      std::function<fp_type(fp_type)> const &b) {
    std::get<1>(coeffs_) = b;
  }
  inline void set_0_order_coefficient(
      std::function<fp_type(fp_type)> const &c) {
    std::get<2>(coeffs_) = c;
  }

  void solve(container<fp_type, alloc> &solution,
             implicit_pde_schemes_enum scheme =
                 implicit_pde_schemes_enum::CrankNicolson);
};

}  // namespace implicit_solvers

namespace explicit_solvers {

// ============================================================================
// ============ general_heat_equation Robin Specialisation Template ===========
// ============================================================================
/*!
============================================================================
Represents general spacial variable 1D heat equation Robin solver

u_t = a(x)*u_xx + b(x)*u_x + c(x)*u + F(x,t),
t > 0, x_1 < x < x_2

with initial condition

u(x,0) = f(x)

and Robin boundaries:

d_1*u_x(x_1,t) + f_1*u(x_1,t) = A
d_2*u_x(x_2,t) + f_2*u(x_2,t) = B

It is assumed the Robin boundaries are discretised in following way:

d_1*(U_1 - U_0)/h + f_1*(U_0 + U_1)/2 = A
d_2*(U_N - U_N-1)/h + f_2*(U_N-1 + U_N)/2 = B

And therefore can be rewritten in form:

U_0 = ((2*d_1 + f_1*h)/(2*d_1 - f_1*h)) * U_1 +
(2*h/(f_1*h - 2*d_1))*A
U_N-1 = ((2*d_2 + f_2*h)/(2*d_2 - f_2*h)) * U_N +
(2*h/(f_2*h -2*d_2))*B

or

U_0 = alpha * U_1 + phi,
U_N-1 = beta * U_N + psi,

============================================================================
*/
template <typename fp_type, template <typename, typename> typename container,
          typename alloc>
class general_heat_equation<fp_type, boundary_condition_enum::Robin, container,
                            alloc> {
 private:
  typedef heat_data<fp_type> heat_data_t;

  uptr_t<heat_data_t> dataPtr_;                       // one-dim heat data
  robin_boundary<fp_type> boundary_;                  // Robin boundary
  pde_coefficient_holder_fun_1_arg<fp_type> coeffs_;  // coefficients of PDE

 public:
  typedef fp_type value_type;
  explicit general_heat_equation() = delete;
  explicit general_heat_equation(range<fp_type> const &space_range,
                                 fp_type terminal_time,
                                 std::size_t const &space_discretization,
                                 std::size_t const &time_discretization)
      : dataPtr_{std::make_unique<heat_data_t>(
            space_range, range<fp_type>(fp_type{}, terminal_time),
            space_discretization, time_discretization, nullptr, nullptr,
            nullptr, false)} {}

  ~general_heat_equation() {}

  general_heat_equation(general_heat_equation const &) = delete;
  general_heat_equation(general_heat_equation &&) = delete;
  general_heat_equation &operator=(general_heat_equation const &) = delete;
  general_heat_equation &operator=(general_heat_equation &&) = delete;

  inline fp_type space_step() const {
    return ((dataPtr_->space_range.spread()) /
            static_cast<fp_type>(dataPtr_->space_division));
  }
  inline fp_type time_step() const {
    return ((dataPtr_->time_range.upper()) /
            static_cast<fp_type>(dataPtr_->time_division));
  }

  inline void set_boundary_condition(
      robin_boundary<fp_type> const &robin_boundary) {
    boundary_ = robin_boundary;
  }
  inline void set_initial_condition(
      std::function<fp_type(fp_type)> const &initial_condition) {
    dataPtr_->initial_condition = initial_condition;
  }
  inline void set_heat_source(
      std::function<fp_type(fp_type, fp_type)> const &heat_source) {
    dataPtr_->is_source_function_set = true;
    dataPtr_->source_function = heat_source;
  }
  inline void set_2_order_coefficient(
      std::function<fp_type(fp_type)> const &a) {
    std::get<0>(coeffs_) = a;
  }
  inline void set_1_order_coefficient(
      std::function<fp_type(fp_type)> const &b) {
    std::get<1>(coeffs_) = b;
  }
  inline void set_0_order_coefficient(
      std::function<fp_type(fp_type)> const &c) {
    std::get<2>(coeffs_) = c;
  }

  void solve(container<fp_type, alloc> &solution);
};

}  // namespace explicit_solvers

// ============================================================================
// ========================= IMPLEMENTATIONS ==================================

// ============================================================================
// =============== general_heat_equation (Robin) implementation ===============
// ============================================================================

template <typename fp_type,
          template <typename, boundary_condition_enum,
                    template <typename, typename> typename cont, typename>
          typename FDMSolver,
          template <typename, typename> typename container, typename alloc>
void implicit_solvers::general_heat_equation<
    fp_type, boundary_condition_enum::Robin, FDMSolver, container,
    alloc>::solve(container<fp_type, alloc> &solution,
                  implicit_pde_schemes_enum scheme) {
  LSS_VERIFY(dataPtr_->initial_condition, "Initial condition must be set.");
  LSS_VERIFY(std::get<0>(coeffs_), "2.order coefficient needs to be set.");
  LSS_VERIFY(std::get<1>(coeffs_), "1.order coefficient needs to be set.");
  LSS_VERIFY(std::get<2>(coeffs_), "0.order coefficient needs to be set.");
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");
  typedef discretization<fp_type, container, alloc> d_1d;
  typedef container<fp_type, alloc> container_t;
  // get correct theta according to the scheme:
  fp_type const theta =
      lss_one_dim_space_variable_heat_implicit_schemes::heat_equation_schemes<
          container, fp_type, alloc>::get_theta(scheme);
  // get space step:
  fp_type const h = space_step();
  // get time step:
  fp_type const k = time_step();
  // get space range:
  auto const &space_range = dataPtr_->space_range;
  // space divisions:
  std::size_t const &space_size = dataPtr_->space_division;
  // get source heat function:
  auto const &heat_source = dataPtr_->source_function;
  // calculate scheme const coefficients:
  fp_type const lambda = k / (h * h);
  fp_type const gamma = k / (2.0 * h);
  fp_type const delta = 0.5 * k;
  // save scheme variable coefficients:
  auto const &a = std::get<0>(coeffs_);
  auto const &b = std::get<1>(coeffs_);
  auto const &c = std::get<2>(coeffs_);
  // create container to carry mesh in space and then previous solution:
  container_t prev_sol(space_size + 1, fp_type{});
  // populate the container with mesh in space
  d_1d::discretize_space(h, space_range.lower(), prev_sol);
  // use the mesh in space to get values of initial condition
  d_1d::discretize_initial_condition(dataPtr_->initial_condition, prev_sol);
  // since coefficients are different in space :
  container_t low(space_size + 1, fp_type{});
  container_t diag(space_size + 1, fp_type{});
  container_t up(space_size + 1, fp_type{});
  // prepare space variable coefficients:
  auto const &A = [&](fp_type x) { return (lambda * a(x) - gamma * b(x)); };
  auto const &B = [&](fp_type x) { return (lambda * a(x) - delta * c(x)); };
  auto const &D = [&](fp_type x) { return (lambda * a(x) + gamma * b(x)); };
  for (std::size_t t = 0; t < low.size(); ++t) {
    low[t] = -1.0 * A(t * h) * theta;
    diag[t] = (1.0 + 2.0 * B(t * h) * theta);
    up[t] = -1.0 * D(t * h) * theta;
  }
  container_t rhs(space_size + 1, fp_type{});
  // create container to carry new solution:
  container_t next_sol(space_size + 1, fp_type{});
  // create first time point:
  fp_type time = k;
  // store terminal time:
  fp_type const last_time = dataPtr_->time_range.upper();
  // set properties of FDMSolver:
  solverPtr_->set_diagonals(std::move(low), std::move(diag), std::move(up));
  // differentiate between inhomogeneous and homogeneous PDE:
  if ((dataPtr_->is_source_function_set)) {
    // wrap the scheme coefficients:
    const auto scheme_coeffs = std::make_tuple(A, B, D, h, k);
    // get the correct scheme:
    auto scheme_fun =
        lss_one_dim_space_variable_heat_implicit_schemes::heat_equation_schemes<
            container, fp_type, alloc>::get_inhom_scheme(scheme);
    // create a container to carry discretized source heat
    container_t source_curr(space_size + 1, fp_type{});
    container_t source_next(space_size + 1, fp_type{});
    d_1d::discretize_in_space(h, space_range.lower(), 0.0, heat_source,
                              source_curr);
    d_1d::discretize_in_space(h, space_range.lower(), time, heat_source,
                              source_next);
    // loop for stepping in time:
    while (time <= last_time) {
      scheme_fun(scheme_coeffs, prev_sol, source_curr, source_next, rhs);
      solverPtr_->set_rhs(rhs);
      solverPtr_->solve(next_sol);
      prev_sol = next_sol;
      d_1d::discretize_in_space(h, space_range.lower(), time, heat_source,
                                source_curr);
      d_1d::discretize_in_space(h, space_range.lower(), 2.0 * time, heat_source,
                                source_next);
      time += k;
    }
  } else {
    // wrap the scheme coefficients:
    const auto scheme_coeffs = std::make_tuple(A, B, D, h, fp_type{});
    // get the correct scheme:
    auto scheme_fun =
        lss_one_dim_space_variable_heat_implicit_schemes::heat_equation_schemes<
            container, fp_type, alloc>::get_scheme(scheme);
    // loop for stepping in time:
    while (time <= last_time) {
      scheme_fun(scheme_coeffs, prev_sol, container_t(), container_t(), rhs);
      solverPtr_->set_rhs(rhs);
      solverPtr_->solve(next_sol);
      prev_sol = next_sol;
      time += k;
    }
  }
  // copy into solution vector
  std::copy(prev_sol.begin(), prev_sol.end(), solution.begin());
}

// ============================================================================
// ================= general_heat_equation (Robin) implementation =============
// ============================================================================

template <typename fp_type, template <typename, typename> typename container,
          typename alloc>
void explicit_solvers::general_heat_equation<
    fp_type, boundary_condition_enum::Robin, container,
    alloc>::solve(container<fp_type, alloc> &solution) {
  LSS_VERIFY(dataPtr_->initial_condition, "Initial condition must be set.");
  LSS_VERIFY(std::get<0>(coeffs_), "2.order coefficient needs to be set.");
  LSS_VERIFY(std::get<1>(coeffs_), "1.order coefficient needs to be set.");
  LSS_VERIFY(std::get<2>(coeffs_), "0.order coefficient needs to be set.");
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");
  typedef discretization<fp_type, container, alloc> d_1d;
  typedef container<fp_type, alloc> container_t;
  // get space step:
  fp_type const h = space_step();
  // get time step:
  fp_type const k = time_step();
  // get space range:
  auto const &space_range = dataPtr_->space_range;
  // space divisions:
  std::size_t const &space_size = dataPtr_->space_division;
  // get source heat function:
  auto const &heat_source = dataPtr_->source_function;
  // calculate scheme const coefficients:
  fp_type const lambda = k / (h * h);
  fp_type const gamma = k / (2.0 * h);
  fp_type const delta = 0.5 * k;
  // save scheme variable coefficients:
  auto const &a = std::get<0>(coeffs_);
  auto const &b = std::get<1>(coeffs_);
  auto const &c = std::get<2>(coeffs_);
  // prepare space variable coefficients:
  auto const &A = [&](fp_type x) { return (lambda * a(x) - gamma * b(x)); };
  auto const &B = [&](fp_type x) { return (lambda * a(x) - delta * c(x)); };
  auto const &D = [&](fp_type x) { return (lambda * a(x) + gamma * b(x)); };
  // wrap up the scheme coefficients:
  auto scheme_coeffs = std::make_tuple(A, B, D);
  // create container to carry mesh in space and then previous solution:
  container_t init_condition(space_size + 1, fp_type{});
  // populate the container with mesh in space
  d_1d::discretize_space(h, space_range.lower(), init_condition);
  // use the mesh in space to get values of initial condition
  d_1d::discretize_initial_condition(dataPtr_->initial_condition,
                                     init_condition);
  // get the correct scheme:
  // Here we have only ExplicitEulerScheme available
  lss_one_dim_space_variable_heat_explicit_schemes::heat_euler_scheme<
      container, fp_type, alloc>
      euler{space_range.lower(),  dataPtr_->time_range.upper(),
            std::make_pair(k, h), coeffs_,
            scheme_coeffs,        init_condition,
            heat_source,          dataPtr_->is_source_function_set};
  euler(boundary_, solution);
}

}  // namespace lss_one_dim_space_variable_pde_solvers

#endif  //_LSS_1D_SPACE_VARIABLE_GENERAL_HEAT_EQUATION_ROBIN_SOLVERS
