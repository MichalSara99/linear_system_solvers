#pragma once
#if !defined(_LSS_1D_GENERAL_HEAT_EQUATION_ROBIN_SOLVERS_CUDA)
#define _LSS_1D_GENERAL_HEAT_EQUATION_ROBIN_SOLVERS_CUDA

#include "common/lss_containers.h"
#include "common/lss_enumerations.h"
#include "common/lss_utility.h"
#include "lss_general_heat_equation_solvers_base.h"
#include "lss_heat_explicit_schemes_cuda.h"
#include "lss_heat_implicit_schemes_cuda.h"
#include "pde_solvers/one_dim/lss_pde_utility.h"
#include "sparse_solvers/lss_sparse_solvers_cuda.h"

namespace lss_one_dim_classic_pde_solvers {

using lss_containers::flat_matrix;
using lss_enumerations::boundary_condition_enum;
using lss_enumerations::implicit_pde_schemes_enum;
using lss_enumerations::memory_space_enum;
using lss_one_dim_pde_utility::discretization;
using lss_one_dim_pde_utility::heat_data;
using lss_one_dim_pde_utility::pde_coefficient_holder_const;
using lss_one_dim_pde_utility::robin_boundary;
using lss_sparse_solvers::real_sparse_solver_cuda;
using lss_utility::range;
using lss_utility::uptr_t;

namespace implicit_solvers {

// ============================================================================
// ======== general_heat_equation_cuda Robin Specialisation Template ==========
// ============================================================================
/*!
   ============================================================================

    u_t = a*u_xx + b*u_x + c*u + F(x,t),
    t > 0, x_1 < x < x_2

    with initial condition:

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
                        (2*h/(f_2*h - 2*d_2))*B

    or

    U_0 = alpha * U_1 + phi,
    U_N-1 = beta * U_N + psi,

  ============================================================================
*/
template <typename fp_type, memory_space_enum memory_space,
          template <memory_space_enum, typename>
          typename real_sparse_policy_cuda,
          template <typename, typename> typename container, typename alloc>
class general_heat_equation_cuda<fp_type, boundary_condition_enum::Robin,
                                 memory_space, real_sparse_policy_cuda,
                                 container, alloc> {
 private:
  typedef real_sparse_policy_cuda<memory_space, fp_type> cuda_solver_t;
  typedef heat_data<fp_type> heat_data_t;

  uptr_t<cuda_solver_t> solverPtr_;               // finite-difference solver
  uptr_t<heat_data_t> dataPtr_;                   // one-dim heat data
  robin_boundary<fp_type> boundary_;              // boundaries
  pde_coefficient_holder_const<fp_type> coeffs_;  // coefficients a, b, c in PDE
  void transform_robin_bc(robin_boundary<fp_type> const &boundary);

 public:
  typedef fp_type value_type;
  explicit general_heat_equation_cuda() = delete;
  explicit general_heat_equation_cuda(range<fp_type> const &space_range,
                                      fp_type terminal_time,
                                      std::size_t const &space_discretization,
                                      std::size_t const &time_discretization)
      : solverPtr_{std::make_unique<cuda_solver_t>(space_discretization + 1)},
        dataPtr_{std::make_unique<heat_data_t>(
            space_range, range<fp_type>(fp_type{}, terminal_time),
            space_discretization, time_discretization, nullptr, nullptr,
            nullptr, false)} {}

  ~general_heat_equation_cuda() {}

  general_heat_equation_cuda(general_heat_equation_cuda const &) = delete;
  general_heat_equation_cuda(general_heat_equation_cuda &&) = delete;
  general_heat_equation_cuda &operator=(general_heat_equation_cuda const &) =
      delete;
  general_heat_equation_cuda &operator=(general_heat_equation_cuda &&) = delete;

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
    transform_robin_bc(robin_boundary);
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
// ========= general_heat_equation_cuda Robin Specialisation Template =========
// ============================================================================
/*!
============================================================================

u_t = a*u_xx + b*u_x + c*u + F(x,t),
t > 0, x_1 < x < x_2

with initial condition:

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
(2*h/(f_2*h - 2*d_2))*B

or

U_0 = alpha * U_1 + phi,
U_N-1 = beta * U_N + psi,

============================================================================
*/
template <typename fp_type, template <typename, typename> typename container,
          typename alloc>
class general_heat_equation_cuda<fp_type, boundary_condition_enum::Robin,
                                 container, alloc> {
 private:
  typedef heat_data<fp_type> heat_data_t;

  uptr_t<heat_data_t> dataPtr_;                   // one-dim heat data
  robin_boundary<fp_type> boundary_;              // boundaries
  pde_coefficient_holder_const<fp_type> coeffs_;  // coefficients a, b, c in PDE
  void transform_robin_bc(robin_boundary<fp_type> const &boundary);

 public:
  typedef fp_type value_type;
  explicit general_heat_equation_cuda() = delete;
  explicit general_heat_equation_cuda(range<fp_type> const &space_range,
                                      fp_type terminal_time,
                                      std::size_t const &space_discretization,
                                      std::size_t const &time_discretization)
      : dataPtr_{std::make_unique<heat_data_t>(
            space_range, range<fp_type>(fp_type{}, terminal_time),
            space_discretization, time_discretization, nullptr, nullptr,
            nullptr, false)} {}

  ~general_heat_equation_cuda() {}

  general_heat_equation_cuda(general_heat_equation_cuda const &) = delete;
  general_heat_equation_cuda(general_heat_equation_cuda &&) = delete;
  general_heat_equation_cuda &operator=(general_heat_equation_cuda const &) =
      delete;
  general_heat_equation_cuda &operator=(general_heat_equation_cuda &&) = delete;

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
    transform_robin_bc(robin_boundary);
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

  // stability check:
  bool is_stable() const;

  void solve(container<fp_type, alloc> &solution);
};
}  // namespace explicit_solvers

// ============================= IMPLEMENTATIONS ==============================

// ============================================================================
// =========== general_heat_equation_cuda (Robin BC) implementation ===========
// ============================================================================

template <typename fp_type, memory_space_enum memory_space,
          template <memory_space_enum, typename>
          typename real_sparse_policy_cuda,
          template <typename, typename> typename container, typename alloc>
void implicit_solvers::general_heat_equation_cuda<
    fp_type, boundary_condition_enum::Robin, memory_space,
    real_sparse_policy_cuda, container,
    alloc>::transform_robin_bc(robin_boundary<fp_type> const &boundary) {
  auto const &beta_ = static_cast<fp_type>(1.0) / boundary.right.first;
  auto const &psi_ =
      static_cast<fp_type>(-1.0) * boundary.right.second / boundary.right.first;
  boundary_.left = boundary.left;
  boundary_.right = std::make_pair(beta_, psi_);
}

template <typename fp_type, memory_space_enum memory_space,
          template <memory_space_enum, typename>
          typename real_sparse_policy_cuda,
          template <typename, typename> typename container, typename alloc>
void implicit_solvers::general_heat_equation_cuda<
    fp_type, boundary_condition_enum::Robin, memory_space,
    real_sparse_policy_cuda, container,
    alloc>::solve(container<fp_type, alloc> &solution,
                  implicit_pde_schemes_enum scheme) {
  LSS_VERIFY(dataPtr_->initial_condition, "Initial condition must be set.");
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");
  typedef container<fp_type, alloc> container_t;
  typedef discretization<fp_type, container, alloc> d1d_t;
  // get correct theta according to the scheme:
  fp_type const theta =
      lss_one_dim_heat_implicit_schemes_cuda::heat_equation_schemes<
          container, fp_type, alloc>::get_theta(scheme);
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
  // store size of matrix:
  std::size_t const m = space_size + 1;
  // create container to carry mesh in space and then previous solution:
  container_t prev_sol(m, fp_type{});
  // populate the container with mesh in space
  d1d_t::discretize_space(h, space_range.lower(), prev_sol);
  // use the mesh in space to get values of initial condition
  d1d_t::discretize_initial_condition(dataPtr_->initial_condition, prev_sol);
  // first create and populate the sparse matrix:
  flat_matrix<fp_type> fsm(m, m);
  // populate the matrix:
  fsm.emplace_back(0, 0, (1.0 + (2.0 * lambda - delta) * theta));
  fsm.emplace_back(
      0, 1,
      -1.0 * ((lambda + gamma) + (lambda - gamma) * boundary_.left.first) *
          theta);
  for (std::size_t t = 1; t < m - 1; ++t) {
    fsm.emplace_back(t, t - 1, -1.0 * (lambda - gamma) * theta);
    fsm.emplace_back(t, t, (1.0 + (2.0 * lambda - delta) * theta));
    fsm.emplace_back(t, t + 1, -1.0 * (lambda + gamma) * theta);
  }
  fsm.emplace_back(
      m - 1, m - 2,
      -1.0 * ((lambda - gamma) + (lambda + gamma) * boundary_.right.first) *
          theta);
  fsm.emplace_back(m - 1, m - 1, (1.0 + (2.0 * lambda - delta) * theta));
  container_t rhs(m, fp_type{});
  // create container to carry new solution:
  container_t next_sol(m, fp_type{});
  // create first time point:
  fp_type time = k;
  // store terminal time:
  fp_type const last_time = dataPtr_->time_range.upper();
  // initialise the solver:
  solverPtr_->initialize(m);
  // insert sparse matrix A and vector b:
  solverPtr_->set_flat_sparse_matrix(std::move(fsm));
  // differentiate between inhomogeneous and homogeneous PDE:
  if (dataPtr_->is_source_function_set) {
    // wrap the scheme coefficients:
    const auto scheme_coeffs = std::make_tuple(lambda, gamma, delta, k);
    // get the correct scheme:
    auto scheme_fun = lss_one_dim_heat_implicit_schemes_cuda::
        heat_equation_schemes<container, fp_type, alloc>::get_inhom_scheme(
            boundary_condition_enum::Robin, scheme);
    // create a container to carry discretized source heat
    container_t source_curr(m, fp_type{});
    container_t source_next(m, fp_type{});
    d1d_t::discretize_in_space(h, (space_range.lower() + h), 0.0, heat_source,
                               source_curr);
    d1d_t::discretize_in_space(h, (space_range.lower() + h), time, heat_source,
                               source_next);
    // loop for stepping in time:
    while (time <= last_time) {
      scheme_fun(scheme_coeffs, prev_sol, source_curr, source_next, rhs,
                 boundary_.left, boundary_.right);
      solverPtr_->set_rhs(rhs);
      solverPtr_->solve(next_sol);
      prev_sol = next_sol;
      d1d_t::discretize_in_space(h, (space_range.lower() + h), time,
                                 heat_source, source_curr);
      d1d_t::discretize_in_space(h, (space_range.lower() + h), 2.0 * time,
                                 heat_source, source_next);
      time += k;
    }
  } else {
    // wrap the scheme coefficients:
    const auto scheme_coeffs = std::make_tuple(lambda, gamma, delta, fp_type{});
    // get the correct scheme:
    auto scheme_fun = lss_one_dim_heat_implicit_schemes_cuda::
        heat_equation_schemes<container, fp_type, alloc>::get_scheme(
            boundary_condition_enum::Robin, scheme);
    // loop for stepping in time:
    while (time <= last_time) {
      scheme_fun(scheme_coeffs, prev_sol, container_t(), container_t(), rhs,
                 boundary_.left, boundary_.right);
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
// ============== general_heat_equation_cuda (Robin BC) implementation ========
// ============================================================================

template <typename fp_type, template <typename, typename> typename container,
          typename alloc>
void explicit_solvers::general_heat_equation_cuda<
    fp_type, boundary_condition_enum::Robin, container,
    alloc>::transform_robin_bc(robin_boundary<fp_type> const &boundary) {
  auto const &beta_ = static_cast<fp_type>(1.0) / boundary.right.first;
  auto const &psi_ =
      static_cast<fp_type>(-1.0) * boundary.right.second / boundary.right.first;
  boundary_.left = boundary.left;
  boundary_.right = std::make_pair(beta_, psi_);
}

template <typename fp_type, template <typename, typename> typename container,
          typename alloc>
bool explicit_solvers::general_heat_equation_cuda<
    fp_type, boundary_condition_enum::Robin, container, alloc>::is_stable()
    const {
  fp_type const k = time_step();
  fp_type const h = space_step();
  fp_type const A = std::get<0>(coeffs_);
  fp_type const B = std::get<1>(coeffs_);

  return (((2.0 * A * k / (h * h)) <= 1.0) && (B * (k / h) <= 1.0));
}

template <typename fp_type, template <typename, typename> typename container,
          typename alloc>
void explicit_solvers::general_heat_equation_cuda<
    fp_type, boundary_condition_enum::Robin, container,
    alloc>::solve(container<fp_type, alloc> &solution) {
  LSS_VERIFY(dataPtr_->initial_condition, "Initial condition must be set.");
  LSS_ASSERT(is_stable() == true, "This discretization is not stable.");
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");
  typedef container<fp_type, alloc> container_t;
  typedef discretization<fp_type, container, alloc> d1d_t;
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
  std::size_t const space_size = dataPtr_->space_division;
  // flag set:
  bool const &is_source_set = dataPtr_->is_source_function_set;
  // create container to carry mesh in space and then previous solution:
  container_t prev_sol(space_size + 1, fp_type{});
  // populate the container with mesh in space
  d1d_t::discretize_space(h, space_range.lower(), prev_sol);
  // use the mesh in space to get values of initial condition
  d1d_t::discretize_initial_condition(dataPtr_->initial_condition, prev_sol);
  lss_one_dim_heat_explicit_schemes_cuda::euler_heat_equation_scheme<
      fp_type, container, alloc>
      euler_scheme(space_range.lower(), time_range.upper(),
                   std::make_pair(k, h), coeffs_, prev_sol, heat_source,
                   is_source_set);
  euler_scheme(boundary_, solution);
}

}  // namespace lss_one_dim_classic_pde_solvers

#endif  ///_LSS_1D_GENERAL_HEAT_EQUATION_ROBIN_SOLVERS_CUDA
