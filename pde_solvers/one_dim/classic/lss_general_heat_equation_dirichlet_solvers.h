#pragma once
#if !defined(_LSS_1D_GENERAL_HEAT_EQUATION_DIRICHLET_SOLVERS)
#define _LSS_1D_GENERAL_HEAT_EQUATION_DIRICHLET_SOLVERS

#include <functional>

#include "common/lss_enumerations.h"
#include "common/lss_macros.h"
#include "common/lss_utility.h"
#include "lss_general_heat_equation_solvers_base.h"
#include "lss_heat_explicit_schemes.h"
#include "lss_heat_implicit_schemes.h"
#include "pde_solvers/one_dim/lss_pde_utility.h"

namespace lss_one_dim_classic_pde_solvers {

using lss_enumerations::boundary_condition_enum;
using lss_enumerations::explicit_pde_schemes_enum;
using lss_enumerations::implicit_pde_schemes_enum;
using lss_one_dim_pde_utility::dirichlet_boundary;
using lss_one_dim_pde_utility::discretization;
using lss_one_dim_pde_utility::heat_data;
using lss_one_dim_pde_utility::pde_coefficient_holder_const;
using lss_utility::range;
using lss_utility::uptr_t;

namespace implicit_solvers {

// ============================================================================
// ======= general_heat_equation Dirichlet Specialisation Template ============
// ============================================================================

/*!
   ============================================================================
    Represents implicit solver for general Dirichlet 1D heat equation

    u_t = a*u_xx + b*u_x + c*u + F(x,t),
    t > 0, x_1 < x < x_2

    with initial condition:

    u(x,0) = f(x)

    and Dirichlet boundaries:

    u(x_1,t) = A(t)
    u(x_2,t) = B(t)

   ============================================================================
*/
template <typename fp_type,
          template <typename, boundary_condition_enum,
                    template <typename, typename> typename cont, typename>
          typename fdm_solver,
          template <typename, typename> typename container, typename alloc>
class general_heat_equation<fp_type, boundary_condition_enum::Dirichlet,
                            fdm_solver, container, alloc> {
 private:
  typedef fdm_solver<fp_type, boundary_condition_enum::Dirichlet, container,
                     alloc>
      fdm_solver_t;
  typedef heat_data<fp_type> heat_data_t;

  uptr_t<fdm_solver_t> solverPtr_;                // finite-difference solver
  uptr_t<heat_data_t> dataPtr_;                   // heat data
  dirichlet_boundary<fp_type> boundary_;          // boundaries
  pde_coefficient_holder_const<fp_type> coeffs_;  // coefficients a, b, c in PDE

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
      dirichlet_boundary<fp_type> const &dirichlet_boundary) {
    boundary_ = dirichlet_boundary;
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
  inline void set_2_order_coefficient(fp_type value) {
    std::get<0>(coeffs_) = value;
  }
  inline void set_1_order_coefficient(fp_type value) {
    std::get<1>(coeffs_) = value;
  }
  inline void set_0_order_coefficient(fp_type value) {
    std::get<2>(coeffs_) = value;
  }

  void solve(container<fp_type, alloc> &solution,
             implicit_pde_schemes_enum scheme =
                 implicit_pde_schemes_enum::CrankNicolson);
};

}  // namespace implicit_solvers

namespace explicit_solvers {
// ============================================================================
// ======= general_heat_equation Dirichlet Specialisation Template ============
// ============================================================================

/*!
============================================================================
Represents implicit solver for general Dirichlet 1D heat equation

u_t = a*u_xx + b*u_x + c*u + F(x,t),
t > 0, x_1 < x < x_2

with initial condition:

u(x,0) = f(x)

and Dirichlet boundaries:

u(x_1,t) = A(t)
u(x_2,t) = B(t)

============================================================================
*/
template <typename fp_type, template <typename, typename> typename container,
          typename alloc>
class general_heat_equation<fp_type, boundary_condition_enum::Dirichlet,
                            container, alloc> {
 private:
  typedef heat_data<fp_type> heat_data_t;

  uptr_t<heat_data_t> dataPtr_;                   // one-dim heat data
  dirichlet_boundary<fp_type> boundary_;          // boundaries
  pde_coefficient_holder_const<fp_type> coeffs_;  // coefficients a, b, c in PDE

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
      dirichlet_boundary<fp_type> const &dirichlet_boundary) {
    boundary_ = dirichlet_boundary;
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
  inline void set_2_order_coefficient(fp_type value) {
    std::get<0>(coeffs_) = value;
  }
  inline void set_1_order_coefficient(fp_type value) {
    std::get<1>(coeffs_) = value;
  }
  inline void set_0_order_coefficient(fp_type value) {
    std::get<2>(coeffs_) = value;
  }

  void solve(container<fp_type, alloc> &solution,
             explicit_pde_schemes_enum scheme =
                 explicit_pde_schemes_enum::ADEBarakatClark);
};

}  // namespace explicit_solvers

// ========================= IMPLEMENTATIONS ==================================

// ============================================================================
// ============== general_heat_equation (Dirichlet) implementation ============
// ============================================================================

template <typename fp_type,
          template <typename, boundary_condition_enum,
                    template <typename, typename> typename cont, typename>
          typename fdm_solver,
          template <typename, typename> typename container, typename alloc>
void implicit_solvers::general_heat_equation<
    fp_type, boundary_condition_enum::Dirichlet, fdm_solver, container,
    alloc>::solve(container<fp_type, alloc> &solution,
                  implicit_pde_schemes_enum scheme) {
  LSS_VERIFY(dataPtr_->initial_condition, "Initial condition must be set.");
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");

  // get correct theta according to the scheme:
  fp_type const theta =
      lss_one_dim_heat_implicit_schemes::heat_equation_schemes<
          fp_type>::get_theta(scheme);
  // get space step:
  fp_type const h = space_step();
  // get time step:
  fp_type const k = time_step();
  // get space range:
  auto const &space_range = dataPtr_->space_range;
  // get source heat function:
  auto const &heat_source = dataPtr_->source_function;
  // space divisions:
  std::size_t const space_size = dataPtr_->space_division;
  // calculate scheme const coefficients:
  fp_type const lambda = (std::get<0>(coeffs_) * k) / (h * h);
  fp_type const gamma = (std::get<1>(coeffs_) * k) / (2.0 * h);
  fp_type const delta = (std::get<2>(coeffs_) * k);
  // create container to carry mesh in space and then previous solution:
  container<fp_type, alloc> prev_sol(space_size + 1, fp_type{});
  // populate the container with mesh in space
  discretization<fp_type, container, alloc>::discretize_space(
      h, space_range.lower(), prev_sol);
  // use the mesh in space to get values of initial condition
  discretization<fp_type, container, alloc>::discretize_initial_condition(
      dataPtr_->initial_condition, prev_sol);
  // prepare containers for diagonal vectors for fdm_solver:
  container<fp_type, alloc> low(space_size + 1,
                                -1.0 * (lambda - gamma) * theta);
  container<fp_type, alloc> diag(space_size + 1,
                                 (1.0 + (2.0 * lambda - delta) * theta));
  container<fp_type, alloc> up(space_size + 1, -1.0 * (lambda + gamma) * theta);
  container<fp_type, alloc> rhs(space_size + 1, fp_type{});
  // create container to carry new solution:
  container<fp_type, alloc> next_sol(space_size + 1, fp_type{});
  // create first time point:
  fp_type time = k;
  // store terminal time:
  fp_type const last_time = dataPtr_->time_range.upper();
  // set properties of fdm_solver:
  solverPtr_->set_diagonals(std::move(low), std::move(diag), std::move(up));
  // differentiate between inhomogeneous and homogeneous PDE:
  if ((dataPtr_->is_source_function_set)) {
    // wrap the scheme coefficients:
    const auto scheme_coeffs = std::make_tuple(lambda, gamma, delta, k);
    // get the correct scheme:
    auto scheme_fun = lss_one_dim_heat_implicit_schemes::heat_equation_schemes<
        fp_type>::get_inhom_scheme(scheme);
    // create a container to carry discretized source heat
    container<fp_type, alloc> source_curr(space_size + 1, fp_type{});
    container<fp_type, alloc> source_next(space_size + 1, fp_type{});
    discretization<fp_type, container, alloc>::discretize_in_space(
        h, space_range.lower(), 0.0, heat_source, source_curr);
    discretization<fp_type, container, alloc>::discretize_in_space(
        h, space_range.lower(), time, heat_source, source_next);
    // loop for stepping in time:
    while (time <= last_time) {
      scheme_fun(scheme_coeffs, prev_sol, source_curr, source_next, rhs);
      solverPtr_->set_boundary_condition(
          std::make_pair(boundary_.first(time), boundary_.second(time)));
      solverPtr_->set_rhs(rhs);
      solverPtr_->solve(next_sol);
      prev_sol = next_sol;
      discretization<fp_type, container, alloc>::discretize_in_space(
          h, space_range.lower(), time, heat_source, source_curr);
      discretization<fp_type, container, alloc>::discretize_in_space(
          h, space_range.lower(), 2.0 * time, heat_source, source_next);
      time += k;
    }
  } else {
    // wrap the scheme coefficients:
    const auto scheme_coeffs = std::make_tuple(lambda, gamma, delta, fp_type{});
    // get the correct scheme:
    auto scheme_fun = lss_one_dim_heat_implicit_schemes::heat_equation_schemes<
        fp_type>::get_scheme(scheme);
    // loop for stepping in time:
    while (time <= last_time) {
      scheme_fun(scheme_coeffs, prev_sol, container<fp_type, alloc>(),
                 container<fp_type, alloc>(), rhs);
      solverPtr_->set_boundary_condition(
          std::make_pair(boundary_.first(time), boundary_.second(time)));
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
// =============== general_heat_equation (Dirichlet) implementation ===========
// ============================================================================

template <typename fp_type, template <typename, typename> typename container,
          typename alloc>
void explicit_solvers::general_heat_equation<
    fp_type, boundary_condition_enum::Dirichlet, container,
    alloc>::solve(container<fp_type, alloc> &solution,
                  explicit_pde_schemes_enum scheme) {
  LSS_VERIFY(dataPtr_->initial_condition, "Initial condition must be set.");
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");
  // get space step:
  fp_type const h = space_step();
  // get time step:
  fp_type const k = time_step();
  // get space range:
  auto const &space_range = dataPtr_->space_range;
  // get time range:
  auto const &time_range = dataPtr_->time_range;
  // get source heat function:
  auto const &heat_source = dataPtr_->source_function;
  // space divisions:
  std::size_t const &space_size = dataPtr_->space_division;
  // flag set:
  bool const &is_source_set = dataPtr_->is_source_function_set;
  // create container to carry mesh in space and then previous solution:
  container<fp_type, alloc> init_condition(space_size + 1, fp_type{});
  // populate the container with mesh in space
  discretization<fp_type, container, alloc>::discretize_space(
      h, space_range.lower(), init_condition);
  // use the mesh in space to get values of initial condition
  discretization<fp_type, container, alloc>::discretize_initial_condition(
      dataPtr_->initial_condition, init_condition);
  // get the correct scheme:
  if (scheme == explicit_pde_schemes_enum::Euler) {
    lss_one_dim_heat_explicit_schemes::heat_euler_scheme<fp_type> euler{
        space_range.lower(), time_range.upper(), std::make_pair(k, h), coeffs_,
        init_condition,      heat_source,        is_source_set};
    euler(boundary_, solution);
  } else if (scheme == explicit_pde_schemes_enum::ADEBarakatClark) {
    lss_one_dim_heat_explicit_schemes::ade_heat_bakarat_clark_scheme<fp_type>
        adebc{space_range.lower(),  time_range.upper(),
              std::make_pair(k, h), coeffs_,
              init_condition,       heat_source,
              is_source_set};
    adebc(boundary_, solution);
  } else {
    lss_one_dim_heat_explicit_schemes::ade_heat_saulyev_scheme<fp_type> ades{
        space_range.lower(), time_range.upper(), std::make_pair(k, h), coeffs_,
        init_condition,      heat_source,        is_source_set};
    ades(boundary_, solution);
  }
}

}  // namespace lss_one_dim_classic_pde_solvers

#endif  //_LSS_1D_GENERAL_HEAT_EQUATION_DIRICHLET_SOLVERS
