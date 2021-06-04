#pragma once
#if !defined(_LSS_BLACK_SCHOLES_EQUATION_SOLVERS_CUDA)
#define _LSS_BLACK_SCHOLES_EQUATION_SOLVERS_CUDA

#include "common/lss_containers.h"
#include "common/lss_enumerations.h"
#include "common/lss_utility.h"
#include "lss_space_variable_heat_explicit_schemes_cuda.h"
#include "lss_space_variable_heat_explicit_schemes_cuda_policy.h"
#include "lss_space_variable_heat_implicit_schemes_cuda.h"
#include "pde_solvers/one_dim/lss_pde_boundary.h"
#include "pde_solvers/one_dim/lss_pde_utility.h"
#include "sparse_solvers/lss_sparse_solvers_cuda.h"

namespace lss_one_dim_space_variable_pde_solvers_cuda {

using lss_containers::flat_matrix;
using lss_enumerations::boundary_condition_enum;
using lss_enumerations::implicit_pde_schemes_enum;
using lss_enumerations::memory_space_enum;
using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
using lss_one_dim_pde_utility::dirichlet_boundary;
using lss_one_dim_pde_utility::discretization;
using lss_one_dim_pde_utility::heat_data;
using lss_one_dim_pde_utility::pde_coefficient_holder_fun_1_arg;
using lss_one_dim_pde_utility::robin_boundary;
using lss_one_dim_space_variable_heat_explicit_schemes_cuda_policy::
    heat_euler_scheme_backward_policy;
using lss_sparse_solvers::real_sparse_solver_cuda;
using lss_utility::range;
using lss_utility::sptr_t;
using lss_utility::uptr_t;

namespace implicit_solvers {

// ============================================================================
// ================= black_sholes_equation_cuda General Template ==============
// ============================================================================

template <typename fp_type, boundary_condition_enum b_type,
          memory_space_enum memory_space,
          template <memory_space_enum, typename>
          typename real_sparse_policy_cuda,
          template <typename, typename> typename container, typename alloc>
class black_sholes_equation_cuda {};

// ============================================================================
// ====== black_sholes_equation_cuda Dirichlet Specialisation Template ========
// ============================================================================
//
//	u_t = a(x)*u_xx + b(x)*u_x + c(x)*u + F(x,t),
//	x_1 < x < x_2, 0 < t < T
//
//	with terminal condition
//  u(x,T) = f(x)
//
//	and Dirichlet boundaries:
//  u(x_1,t) = A(t)
//	u(x_2,t) = B(t)
//
// ============================================================================

template <typename fp_type, memory_space_enum memory_space,
          template <memory_space_enum, typename>
          typename real_sparse_policy_cuda,
          template <typename, typename> typename container, typename alloc>
class black_sholes_equation_cuda<fp_type, boundary_condition_enum::Dirichlet,
                                 memory_space, real_sparse_policy_cuda,
                                 container, alloc> {
 private:
  typedef real_sparse_policy_cuda<memory_space, fp_type> cuda_solver_t;
  typedef heat_data<fp_type> heat_data_t;

  uptr_t<cuda_solver_t> solverPtr_;  // finite-difference solver
  uptr_t<heat_data_t> dataPtr_;      // one-dim heat data
  sptr_t<dirichlet_boundary_1d<fp_type>> boundaryPtr_;  // boundaries
  pde_coefficient_holder_fun_1_arg<fp_type> coeffs_;    // coefficients of PDE

 public:
  typedef fp_type value_type;
  explicit black_sholes_equation_cuda() = delete;
  explicit black_sholes_equation_cuda(range<fp_type> const &space_range,
                                      fp_type terminal_time,
                                      std::size_t const &space_discretization,
                                      std::size_t const &time_discretization)
      : solverPtr_{std::make_unique<cuda_solver_t>(space_discretization - 1)},
        dataPtr_{std::make_unique<heat_data_t>(
            space_range, range<fp_type>(fp_type{}, terminal_time),
            space_discretization, time_discretization, nullptr, nullptr,
            nullptr, false)} {}

  ~black_sholes_equation_cuda() {}

  black_sholes_equation_cuda(black_sholes_equation_cuda const &) = delete;
  black_sholes_equation_cuda(black_sholes_equation_cuda &&) = delete;
  black_sholes_equation_cuda &operator=(black_sholes_equation_cuda const &) =
      delete;
  black_sholes_equation_cuda &operator=(black_sholes_equation_cuda &&) = delete;

  inline fp_type space_step() const {
    return ((dataPtr_->space_range.spread()) /
            static_cast<fp_type>(dataPtr_->space_division));
  }
  inline fp_type time_step() const {
    return ((dataPtr_->time_range.upper()) /
            static_cast<fp_type>(dataPtr_->time_division));
  }

  inline void set_boundary_condition(
      sptr_t<dirichlet_boundary_1d<fp_type>> const &dirichlet_boundary) {
    boundaryPtr_ = dirichlet_boundary;
  }

  inline void set_terminal_condition(
      std::function<fp_type(fp_type)> const &terminal_condition) {
    dataPtr_->terminal_condition = terminal_condition;
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

// ============================================================================
// ===== black_sholes_equation_cuda Robin Specialisation Template =============
// ============================================================================
//
//	u_t = a(x)*u_xx + b(x)*u_x + c(x)*u + F(x,t),
//	x_1 < x < x_2, 0 < t < T
//
//	with terminal condition
//  u(x,T) = f(x)
//
//	and Robin boundaries:
//  d_1*u_x(x_1,t) + f_1*u(x_1,t) = A
//	d_2*u_x(x_2,t) + f_2*u(x_2,t) = B
//
//	It is assumed the Robin boundaries are discretised in following way:
//	d_1*(U_1 - U_0)/h + f_1*(U_0 + U_1)/2 = A
//	d_2*(U_N - U_N-1)/h + f_2*(U_N-1 + U_N)/2 = B
//
//	And therefore can be rewritten in form:
//
//	U_0 = ((2*d_1 + f_1*h)/(2*d_1 - f_1*h)) * U_1 +
//			(2*h/(f_1*h - 2*d_1))*A
//	U_N-1 = ((2*d_2 + f_2*h)/(2*d_2 - f_2*h)) * U_N +
//			(2*h/(f_2*h - 2*d_2))*B
//
//	or
//
//	U_0 = alpha * U_1 + phi,
//	U_N-1 = beta * U_N + psi,
//
// ============================================================================

template <typename fp_type, memory_space_enum memory_space,
          template <memory_space_enum, typename>
          typename real_sparse_policy_cuda,
          template <typename, typename> typename container, typename alloc>
class black_sholes_equation_cuda<fp_type, boundary_condition_enum::Robin,
                                 memory_space, real_sparse_policy_cuda,
                                 container, alloc> {
 private:
  typedef real_sparse_policy_cuda<memory_space, fp_type> cuda_solver_t;
  typedef heat_data<fp_type> heat_data_t;

  uptr_t<cuda_solver_t> solverPtr_;   // finite-difference solver
  uptr_t<heat_data_t> dataPtr_;       // one-dim heat data
  robin_boundary<fp_type> boundary_;  // Robin boundary
  pde_coefficient_holder_fun_1_arg<fp_type> coeffs_;  // coefficients of PDE
  void transform_robin_bc(robin_boundary<fp_type> const &robin_boundary);

 public:
  typedef fp_type value_type;
  explicit black_sholes_equation_cuda() = delete;
  explicit black_sholes_equation_cuda(range<fp_type> const &space_range,
                                      fp_type terminal_time,
                                      std::size_t const &space_discretization,
                                      std::size_t const &time_discretization)
      : solverPtr_{std::make_unique<cuda_solver_t>(space_discretization + 1)},
        dataPtr_{std::make_unique<heat_data_t>(
            space_range, range<fp_type>(fp_type{}, terminal_time),
            space_discretization, time_discretization, nullptr, nullptr,
            nullptr, false)} {}

  ~black_sholes_equation_cuda() {}

  black_sholes_equation_cuda(black_sholes_equation_cuda const &) = delete;
  black_sholes_equation_cuda(black_sholes_equation_cuda &&) = delete;
  black_sholes_equation_cuda &operator=(black_sholes_equation_cuda const &) =
      delete;
  black_sholes_equation_cuda &operator=(black_sholes_equation_cuda &&) = delete;

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

  inline void set_terminal_condition(
      std::function<fp_type(fp_type)> const &terminal_condition) {
    dataPtr_->terminal_condition = terminal_condition;
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
// ================= black_sholes_equation_cuda General Template ==============
// ============================================================================

template <typename fp_type, boundary_condition_enum b_type,
          template <typename, typename> typename container, typename alloc>
class black_sholes_equation_cuda {};

// ============================================================================
// ========= black_sholes_equation_cuda Dirichlet Specialisation Template =====
// ============================================================================
//
//	u_t = a(x)*u_xx + b(x)*u_x + c(x)*u + F(x,t),
//	x_1 < x < x_2, 0 < t < T
//
//	with terminal condition
//  u(x,T) = f(x)
//
//	and Dirichlet boundaries:
//  u(x_1,t) = A(t)
//	u(x_2,t) = B(t)
//
// ============================================================================

template <typename fp_type, template <typename, typename> typename container,
          typename alloc>
class black_sholes_equation_cuda<fp_type, boundary_condition_enum::Dirichlet,
                                 container, alloc> {
 private:
  typedef heat_data<fp_type> heat_data_t;

  uptr_t<heat_data_t> dataPtr_;                         // one-dim heat data
  sptr_t<dirichlet_boundary_1d<fp_type>> boundaryPtr_;  // boundaries
  pde_coefficient_holder_fun_1_arg<fp_type> coeffs_;    // coefficients of PDE

 public:
  typedef fp_type value_type;
  explicit black_sholes_equation_cuda() = delete;
  explicit black_sholes_equation_cuda(range<fp_type> const &space_range,
                                      fp_type terminal_time,
                                      std::size_t const &space_discretization,
                                      std::size_t const &time_discretization)
      : dataPtr_{std::make_unique<heat_data_t>(
            space_range, range<fp_type>(fp_type{}, terminal_time),
            space_discretization, time_discretization, nullptr, nullptr,
            nullptr, false)} {}

  ~black_sholes_equation_cuda() {}

  black_sholes_equation_cuda(black_sholes_equation_cuda const &) = delete;
  black_sholes_equation_cuda(black_sholes_equation_cuda &&) = delete;
  black_sholes_equation_cuda &operator=(black_sholes_equation_cuda const &) =
      delete;
  black_sholes_equation_cuda &operator=(black_sholes_equation_cuda &&) = delete;

  inline fp_type space_step() const {
    return ((dataPtr_->space_range.spread()) /
            static_cast<fp_type>(dataPtr_->space_division));
  }
  inline fp_type time_step() const {
    return ((dataPtr_->time_range.upper()) /
            static_cast<fp_type>(dataPtr_->time_division));
  }

  inline void set_boundary_condition(
      sptr_t<dirichlet_boundary_1d<fp_type>> const &dirichlet_boundary) {
    boundaryPtr_ = dirichlet_boundary;
  }

  inline void set_terminal_condition(
      std::function<fp_type(fp_type)> const &terminal_condition) {
    dataPtr_->terminal_condition = terminal_condition;
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
  // stability check:
  bool is_stable() const;

  void solve(container<fp_type, alloc> &solution);
};

// ============================================================================
// ========= black_sholes_equation_cuda Robin Specialisation Template =========
// ============================================================================
//
//	u_t = a(x)*u_xx + b(x)*u_x + c(x)*u + F(x,t),
//	x_1 < x < x_2, 0 < t < T
//
//	with terminal condition
//  u(x,T) = f(x)
//
//	and Robin boundaries:
//  d_1*u_x(x_1,t) + f_1*u(x_1,t) = A
//	d_2*u_x(x_2,t) + f_2*u(x_2,t) = B
//
//	It is assumed the Robin boundaries are discretised in following way:
//	d_1*(U_1 - U_0)/h + f_1*(U_0 + U_1)/2 = A
//	d_2*(U_N - U_N-1)/h + f_2*(U_N-1 + U_N)/2 = B
//
//	And therefore can be rewritten in form:
//
//	U_0 = ((2*d_1 + f_1*h)/(2*d_1 - f_1*h)) * U_1 +
//			(2*h/(f_1*h - 2*d_1))*A
//	U_N-1 = ((2*d_2 + f_2*h)/(2*d_2 - f_2*h)) * U_N +
//			(2*h/(f_2*h - 2*d_2))*B
//
//	or
//
//	U_0 = alpha * U_1 + phi,
//	U_N-1 = beta * U_N + psi,
//
// ============================================================================

template <typename fp_type, template <typename, typename> typename container,
          typename alloc>
class black_sholes_equation_cuda<fp_type, boundary_condition_enum::Robin,
                                 container, alloc> {
 private:
  typedef heat_data<fp_type> heat_data_t;

  uptr_t<heat_data_t> dataPtr_;                       // one-dim heat data
  robin_boundary<fp_type> boundary_;                  // Robin boundary
  pde_coefficient_holder_fun_1_arg<fp_type> coeffs_;  // coefficients of PDE
  void transform_robin_bc(robin_boundary<fp_type> const &robin_boundary);

 public:
  typedef fp_type value_type;
  explicit black_sholes_equation_cuda() = delete;
  explicit black_sholes_equation_cuda(range<fp_type> const &space_range,
                                      fp_type terminal_time,
                                      std::size_t const &space_discretization,
                                      std::size_t const &time_discretization)
      : dataPtr_{std::make_unique<heat_data_t>(
            space_range, range<fp_type>(fp_type{}, terminal_time),
            space_discretization, time_discretization, nullptr, nullptr,
            nullptr, false)} {}

  ~black_sholes_equation_cuda() {}

  black_sholes_equation_cuda(black_sholes_equation_cuda const &) = delete;
  black_sholes_equation_cuda(black_sholes_equation_cuda &&) = delete;
  black_sholes_equation_cuda &operator=(black_sholes_equation_cuda const &) =
      delete;
  black_sholes_equation_cuda &operator=(black_sholes_equation_cuda &&) = delete;

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

  inline void set_terminal_condition(
      std::function<fp_type(fp_type)> const &terminal_condition) {
    dataPtr_->terminal_condition = terminal_condition;
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

  // stability check:
  bool is_stable() const;

  void solve(container<fp_type, alloc> &solution);
};
}  // namespace explicit_solvers

// ============================================================================
// ============================= IMPLEMENTATIONS ==============================

// ============================================================================
// ========= black_sholes_equation_cuda (Dirichlet) implementation ============
// ============================================================================

template <typename fp_type, memory_space_enum memory_space,
          template <memory_space_enum, typename>
          typename real_sparse_policy_cuda,
          template <typename, typename> typename container, typename alloc>
void implicit_solvers::black_sholes_equation_cuda<
    fp_type, boundary_condition_enum::Dirichlet, memory_space,
    real_sparse_policy_cuda, container,
    alloc>::solve(container<fp_type, alloc> &solution,
                  implicit_pde_schemes_enum scheme) {
  LSS_VERIFY(dataPtr_->terminal_condition, "Terminal condition must be set.");
  LSS_VERIFY(std::get<0>(coeffs_), "2.order coefficient needs to be set.");
  LSS_VERIFY(std::get<1>(coeffs_), "1.order coefficient needs to be set.");
  LSS_VERIFY(std::get<2>(coeffs_), "0.order coefficient needs to be set.");
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");
  typedef discretization<fp_type, container, alloc> d_1d;
  typedef container<fp_type, alloc> container_t;
  // get correct theta according to the scheme:
  fp_type const theta = lss_one_dim_space_variable_heat_implicit_schemes_cuda::
      heat_equation_schemes<container, fp_type, alloc>::get_theta(scheme);
  // get space step:
  fp_type const h = space_step();
  // get time step:
  fp_type const k = time_step();
  // get space range:
  auto const &space_range = dataPtr_->space_range;
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
  // store size of matrix:
  std::size_t const m = dataPtr_->space_division - 1;
  // create container to carry mesh in space and then previous solution:
  container_t prev_sol(m, fp_type{});
  // populate the container with mesh in space
  d_1d::discretize_space(h, (space_range.lower() + h), prev_sol);
  // use the mesh in space to get values of initial condition
  d_1d::discretize_initial_condition(dataPtr_->terminal_condition, prev_sol);
  // first create and populate the sparse matrix:
  flat_matrix<fp_type> fsm(m, m);
  // prepare space variable coefficients:
  auto const &A = [&](fp_type x) { return (lambda * a(x) - gamma * b(x)); };
  auto const &B = [&](fp_type x) { return (lambda * a(x) - delta * c(x)); };
  auto const &D = [&](fp_type x) { return (lambda * a(x) + gamma * b(x)); };
  // populate the matrix:
  fsm.emplace_back(0, 0, (1.0 + 2.0 * B(1 * h) * theta));
  fsm.emplace_back(0, 1, (-1.0 * D(1 * h) * theta));
  for (std::size_t t = 1; t < m - 1; ++t) {
    fsm.emplace_back(t, t - 1, (-1.0 * A((t + 1) * h) * theta));
    fsm.emplace_back(t, t, (1.0 + 2.0 * B((t + 1) * h) * theta));
    fsm.emplace_back(t, t + 1, (-1.0 * D((t + 1) * h) * theta));
  }
  fsm.emplace_back(m - 1, m - 2, (-1.0 * A(m * h) * theta));
  fsm.emplace_back(m - 1, m - 1, (1.0 + 2.0 * B(m * h) * theta));
  container_t rhs(m, fp_type{});
  // create container to carry new solution:
  container_t next_sol(m, fp_type{});
  // store terminal time:
  fp_type const last_time = dataPtr_->time_range.upper();
  // create first time point:
  fp_type time = last_time - k;
  // initialise the solver:
  solverPtr_->initialize(m);
  // insert sparse matrix A and vector b:
  solverPtr_->set_flat_sparse_matrix(std::move(fsm));
  if ((dataPtr_->is_source_function_set)) {
    // wrap the scheme coefficients:
    const auto scheme_coeffs = std::make_tuple(A, B, D, h, k);
    // get the correct scheme:
    auto scheme_fun = lss_one_dim_space_variable_heat_implicit_schemes_cuda::
        heat_equation_schemes<container, fp_type, alloc>::get_inhom_scheme(
            boundary_condition_enum::Dirichlet, scheme);
    // create a container to carry discretized source heat
    container_t source_curr(m, fp_type{});
    container_t source_next(m, fp_type{});
    d_1d::discretize_in_space(h, (space_range.lower() + h), 0.0, heat_source,
                              source_curr);
    d_1d::discretize_in_space(h, (space_range.lower() + h), time, heat_source,
                              source_next);
    // loop for stepping in time:
    while (time >= 0.0) {
      scheme_fun(
          scheme_coeffs, prev_sol, source_curr, source_next, rhs,
          std::make_pair(boundaryPtr_->first(time), boundaryPtr_->second(time)),
          std::pair<fp_type, fp_type>());
      solverPtr_->set_rhs(rhs);
      solverPtr_->solve(next_sol);
      prev_sol = next_sol;
      d_1d::discretize_in_space(h, (space_range.lower() + h), time, heat_source,
                                source_curr);
      d_1d::discretize_in_space(h, (space_range.lower() + h), 2.0 * time,
                                heat_source, source_next);
      time -= k;
    }
  } else {
    // wrap the scheme coefficients:
    const auto scheme_coeffs = std::make_tuple(A, B, D, h, fp_type{});
    // get the correct scheme:
    auto schemeFun = lss_one_dim_space_variable_heat_implicit_schemes_cuda::
        heat_equation_schemes<container, fp_type, alloc>::get_scheme(
            boundary_condition_enum::Dirichlet, scheme);
    // loop for stepping in time:
    while (time >= 0.0) {
      schemeFun(
          scheme_coeffs, prev_sol, container_t(), container_t(), rhs,
          std::make_pair(boundaryPtr_->first(time), boundaryPtr_->second(time)),
          std::pair<fp_type, fp_type>());
      solverPtr_->set_rhs(rhs);
      solverPtr_->solve(next_sol);
      prev_sol = next_sol;
      time -= k;
    }
  }

  // copy into solution vector
  solution[0] = boundaryPtr_->first(0.0);
  std::copy(prev_sol.begin(), prev_sol.end(), std::next(solution.begin()));
  solution[solution.size() - 1] = boundaryPtr_->second(0.0);
}

// ============================================================================
// =========== black_sholes_equation_cuda (Robin BC) implementation ===========
// ============================================================================

template <typename fp_type, memory_space_enum memory_space,
          template <memory_space_enum, typename>
          typename real_sparse_policy_cuda,
          template <typename, typename> typename container, typename alloc>
void implicit_solvers::black_sholes_equation_cuda<
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
void implicit_solvers::black_sholes_equation_cuda<
    fp_type, boundary_condition_enum::Robin, memory_space,
    real_sparse_policy_cuda, container,
    alloc>::solve(container<fp_type, alloc> &solution,
                  implicit_pde_schemes_enum scheme) {
  LSS_VERIFY(dataPtr_->terminal_condition, "Terminal condition must be set.");
  LSS_VERIFY(std::get<0>(coeffs_), "2.order coefficient needs to be set.");
  LSS_VERIFY(std::get<1>(coeffs_), "1.order coefficient needs to be set.");
  LSS_VERIFY(std::get<2>(coeffs_), "0.order coefficient needs to be set.");
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");
  typedef discretization<fp_type, container, alloc> d_1d;
  typedef container<fp_type, alloc> container_t;
  // get correct theta according to the scheme:
  fp_type const theta = lss_one_dim_space_variable_heat_implicit_schemes_cuda::
      heat_equation_schemes<container, fp_type, alloc>::get_theta(scheme);
  // get space step:
  fp_type const h = space_step();
  // get time step:
  fp_type const k = time_step();
  // get space range:
  auto const &space_range = dataPtr_->space_range;
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
  // store size of matrix:
  std::size_t const m = dataPtr_->space_division + 1;
  // create container to carry mesh in space and then previous solution:
  container_t prev_sol(m, fp_type{});
  // populate the container with mesh in space
  d_1d::discretize_space(h, space_range.lower(), prev_sol);
  // use the mesh in space to get values of initial condition
  d_1d::discretize_initial_condition(dataPtr_->terminal_condition, prev_sol);
  // first create and populate the sparse matrix:
  flat_matrix<fp_type> fsm(m, m);
  // prepare space variable coefficients:
  auto const &A = [&](fp_type x) { return (lambda * a(x) - gamma * b(x)); };
  auto const &B = [&](fp_type x) { return (lambda * a(x) - delta * c(x)); };
  auto const &D = [&](fp_type x) { return (lambda * a(x) + gamma * b(x)); };
  // populate the matrix:
  fsm.emplace_back(0, 0, (1.0 + 2.0 * B(0 * h) * theta));
  fsm.emplace_back(
      0, 1, (-1.0 * (boundary_.left.first * A(0 * h) + D(0 * h)) * theta));
  for (std::size_t t = 1; t < m - 1; ++t) {
    fsm.emplace_back(t, t - 1, (-1.0 * A(t * h) * theta));
    fsm.emplace_back(t, t, (1.0 + 2.0 * B(t * h) * theta));
    fsm.emplace_back(t, t + 1, (-1.0 * D(t * h) * theta));
  }
  fsm.emplace_back(
      m - 1, m - 2,
      (-1.0 * (A((m - 1) * h) + boundary_.right.first * D((m - 1) * h)) *
       theta));
  fsm.emplace_back(m - 1, m - 1, (1.0 + 2.0 * B((m - 1) * h) * theta));

  container_t rhs(m, fp_type{});
  // create container to carry new solution:
  container_t next_sol(m, fp_type{});
  // store terminal time:
  fp_type const last_time = dataPtr_->time_range.upper();
  // create first time point:
  fp_type time = last_time - k;
  // initialise the solver:
  solverPtr_->initialize(m);
  // insert sparse matrix A and vector b:
  solverPtr_->set_flat_sparse_matrix(std::move(fsm));
  // differentiate between inhomogeneous and homogeneous PDE:
  if ((dataPtr_->is_source_function_set)) {
    // wrap the scheme coefficients:
    const auto scheme_coeffs = std::make_tuple(A, B, D, h, k);
    // get the correct scheme:
    auto scheme_fun = lss_one_dim_space_variable_heat_implicit_schemes_cuda::
        heat_equation_schemes<container, fp_type, alloc>::get_inhom_scheme(
            boundary_condition_enum::Robin, scheme);
    // create a container to carry discretized source heat
    container_t source_curr(m, fp_type{});
    container_t source_next(m, fp_type{});
    d_1d::discretize_in_space(h, (space_range.lower() + h), 0.0, heat_source,
                              source_curr);
    d_1d::discretize_in_space(h, (space_range.lower() + h), time, heat_source,
                              source_next);
    // loop for stepping in time:
    while (time >= 0.0) {
      scheme_fun(scheme_coeffs, prev_sol, source_curr, source_next, rhs,
                 boundary_.left, boundary_.right);
      solverPtr_->set_rhs(rhs);
      solverPtr_->solve(next_sol);
      prev_sol = next_sol;
      d_1d::discretize_in_space(h, (space_range.lower() + h), time, heat_source,
                                source_curr);
      d_1d::discretize_in_space(h, (space_range.lower() + h), 2.0 * time,
                                heat_source, source_next);
      time -= k;
    }
  } else {
    // wrap the scheme coefficients:
    const auto scheme_coeffs = std::make_tuple(A, B, D, h, fp_type{});
    // get the correct scheme:
    auto scheme_fun = lss_one_dim_space_variable_heat_implicit_schemes_cuda::
        heat_equation_schemes<container, fp_type, alloc>::get_scheme(
            boundary_condition_enum::Robin, scheme);
    // loop for stepping in time:
    while (time >= 0.0) {
      scheme_fun(scheme_coeffs, prev_sol, container_t(), container_t(), rhs,
                 boundary_.left, boundary_.right);
      solverPtr_->set_rhs(rhs);
      solverPtr_->solve(next_sol);
      prev_sol = next_sol;
      time -= k;
    }
  }
  // copy into solution vector
  std::copy(prev_sol.begin(), prev_sol.end(), solution.begin());
}

// ============================================================================
// ========= black_sholes_equation_cuda (Dirichlet BC) implementation =========
// ============================================================================

template <typename fp_type, template <typename, typename> typename container,
          typename alloc>
bool explicit_solvers::black_sholes_equation_cuda<
    fp_type, boundary_condition_enum::Dirichlet, container, alloc>::is_stable()
    const {
  auto const &a = std::get<0>(coeffs_);
  auto const &b = std::get<1>(coeffs_);
  auto const &c = std::get<2>(coeffs_);
  fp_type const k = time_step();
  fp_type const h = space_step();
  fp_type const lambda = k / (h * h);
  fp_type const gamma = k / h;

  const std::size_t space_size = dataPtr_->space_division + 1;
  for (std::size_t i = 0; i < space_size; ++i) {
    if (c(i * h) > 0.0) return false;
    if ((2.0 * lambda * a(i * h) - k * c(i * h)) > 1.0) return false;
    if (((gamma * std::abs(b(i * h))) * (gamma * std::abs(b(i * h)))) >
        (2.0 * lambda * a(i * h)))
      return false;
  }
  return true;
}

template <typename fp_type, template <typename, typename> typename container,
          typename alloc>
void explicit_solvers::black_sholes_equation_cuda<
    fp_type, boundary_condition_enum::Dirichlet, container,
    alloc>::solve(container<fp_type, alloc> &solution) {
  LSS_VERIFY(dataPtr_->terminal_condition, "Initial condition must be set.");
  LSS_VERIFY(std::get<0>(coeffs_), "2.order coefficient needs to be set.");
  LSS_VERIFY(std::get<1>(coeffs_), "1.order coefficient needs to be set.");
  LSS_VERIFY(std::get<2>(coeffs_), "0.order coefficient needs to be set.");

  LSS_ASSERT(is_stable() == true, "This discretization is not stable.");
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
  d_1d::discretize_space(h, space_range.lower(), prev_sol);
  // use the mesh in space to get values of initial condition
  d_1d::discretize_initial_condition(dataPtr_->terminal_condition, prev_sol);

  lss_one_dim_space_variable_heat_explicit_schemes_cuda::
      euler_heat_equation_scheme<
          fp_type, container, alloc,
          heat_euler_scheme_backward_policy<
              fp_type, pde_coefficient_holder_fun_1_arg<fp_type>>>
          euler_scheme(space_range.lower(), time_range.upper(),
                       std::make_pair(k, h), coeffs_, prev_sol, heat_source,
                       is_source_set);
  euler_scheme(boundaryPtr_, solution);
}

// ============================================================================
// ========= black_sholes_equation_cuda (Robin BC) implementation =============
// ============================================================================

template <typename fp_type, template <typename, typename> typename container,
          typename alloc>
void explicit_solvers::black_sholes_equation_cuda<
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
bool explicit_solvers::black_sholes_equation_cuda<
    fp_type, boundary_condition_enum::Robin, container, alloc>::is_stable()
    const {
  auto const &a = std::get<0>(coeffs_);
  auto const &b = std::get<1>(coeffs_);
  auto const &c = std::get<2>(coeffs_);
  fp_type const k = time_step();
  fp_type const h = space_step();
  fp_type const lambda = k / (h * h);
  fp_type const gamma = k / h;

  const std::size_t space_size = dataPtr_->space_division + 1;
  for (std::size_t i = 0; i < space_size; ++i) {
    if (c(i * h) > 0.0) return false;
    if ((2.0 * lambda * a(i * h) - k * c(i * h)) > 1.0) return false;
    if (((gamma * std::abs(b(i * h))) * (gamma * std::abs(b(i * h)))) >
        (2.0 * lambda * a(i * h)))
      return false;
  }
  return true;
}

template <typename fp_type, template <typename, typename> typename container,
          typename alloc>
void explicit_solvers::black_sholes_equation_cuda<
    fp_type, boundary_condition_enum::Robin, container,
    alloc>::solve(container<fp_type, alloc> &solution) {
  LSS_VERIFY(dataPtr_->terminal_condition, "Terminal condition must be set.");
  LSS_VERIFY(std::get<0>(coeffs_), "2.order coefficient needs to be set.");
  LSS_VERIFY(std::get<1>(coeffs_), "1.order coefficient needs to be set.");
  LSS_VERIFY(std::get<2>(coeffs_), "0.order coefficient needs to be set.");
  LSS_ASSERT(is_stable() == true, "This discretization is not stable.");
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
  d_1d::discretize_space(h, space_range.lower(), prev_sol);
  // use the mesh in space to get values of initial condition
  d_1d::discretize_initial_condition(dataPtr_->terminal_condition, prev_sol);
  lss_one_dim_space_variable_heat_explicit_schemes_cuda::
      euler_heat_equation_scheme<
          fp_type, container, alloc,
          heat_euler_scheme_backward_policy<
              fp_type, pde_coefficient_holder_fun_1_arg<fp_type>>>
          euler_scheme(space_range.lower(), time_range.upper(),
                       std::make_pair(k, h), coeffs_, prev_sol, heat_source,
                       is_source_set);
  euler_scheme(boundary_, solution);
}

}  // namespace lss_one_dim_space_variable_pde_solvers_cuda

#endif  ///_LSS_BLACK_SCHOLES_EQUATION_SOLVERS_CUDA
