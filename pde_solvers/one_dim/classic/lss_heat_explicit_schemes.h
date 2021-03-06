#pragma once
#if !defined(_LSS_1D_HEAT_EXPLICIT_SCHEMES)
#define _LSS_1D_HEAT_EXPLICIT_SCHEMES

#pragma warning(disable : 4244)

#include <thread>

#include "common/lss_enumerations.h"
#include "common/lss_utility.h"
#include "pde_solvers/one_dim/lss_base_explicit_schemes.h"
#include "pde_solvers/one_dim/lss_pde_boundary.h"
#include "pde_solvers/one_dim/lss_pde_utility.h"

namespace lss_one_dim_heat_explicit_schemes {

using lss_enumerations::boundary_condition_enum;
using lss_one_dim_base_explicit_schemes::heat_scheme_base;
using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
using lss_one_dim_pde_utility::discretization;
using lss_one_dim_pde_utility::pde_coefficient_holder_const;
using lss_one_dim_pde_utility::robin_boundary;
using lss_utility::sptr_t;

// ============================================================================
// ======================= heat_euler_scheme ==================================
// ============================================================================

template <template <typename, typename> typename container, typename fp_type,
          typename alloc>
class heat_euler_scheme
    : public heat_scheme_base<container, fp_type, alloc,
                              pde_coefficient_holder_const<fp_type>> {
 private:
  typedef container<fp_type, alloc> container_t;
  explicit heat_euler_scheme() = default;

 public:
  explicit heat_euler_scheme(
      fp_type space_start, fp_type terminal_time,
      std::pair<fp_type, fp_type> const &deltas,
      pde_coefficient_holder_const<fp_type> const &coeffs,
      container_t const &initial_condition,
      std::function<fp_type(fp_type, fp_type)> const &source = nullptr,
      bool is_source_set = false)
      : heat_scheme_base<container, fp_type, alloc,
                         pde_coefficient_holder_const<fp_type>>(
            space_start, terminal_time, deltas, coeffs, initial_condition,
            source, is_source_set) {}

  ~heat_euler_scheme() {}

  heat_euler_scheme(heat_euler_scheme const &) = delete;
  heat_euler_scheme(heat_euler_scheme &&) = delete;
  heat_euler_scheme &operator=(heat_euler_scheme const &) = delete;
  heat_euler_scheme &operator=(heat_euler_scheme &&) = delete;

  // stability check:
  bool is_stable() const override;

  // for Dirichlet BC
  void operator()(
      sptr_t<dirichlet_boundary_1d<fp_type>> const &dirichlet_boundary,
      container_t &solution) const override;
  // for Robin BC
  void operator()(robin_boundary<fp_type> const &robin_boundary,
                  container_t &solution) const override;
};

// ============================================================================
// ================== ade_heat_bakarat_clark_scheme  ==========================
// ============================================================================

template <template <typename, typename> typename container, typename fp_type,
          typename alloc>
class ade_heat_bakarat_clark_scheme
    : public heat_scheme_base<container, fp_type, alloc,
                              pde_coefficient_holder_const<fp_type>> {
 private:
  typedef container<fp_type, alloc> container_t;
  explicit ade_heat_bakarat_clark_scheme() = default;

 public:
  explicit ade_heat_bakarat_clark_scheme(
      fp_type space_start, fp_type terminal_time,
      std::pair<fp_type, fp_type> const &deltas,
      pde_coefficient_holder_const<fp_type> const &coeffs,
      container_t const &initial_condition,
      std::function<fp_type(fp_type, fp_type)> const &source = nullptr,
      bool is_source_set = false)
      : heat_scheme_base<container, fp_type, alloc,
                         pde_coefficient_holder_const<fp_type>>(
            space_start, terminal_time, deltas, coeffs, initial_condition,
            source, is_source_set) {}

  ~ade_heat_bakarat_clark_scheme() {}

  ade_heat_bakarat_clark_scheme(ade_heat_bakarat_clark_scheme const &) = delete;
  ade_heat_bakarat_clark_scheme(ade_heat_bakarat_clark_scheme &&) = delete;
  ade_heat_bakarat_clark_scheme &operator=(
      ade_heat_bakarat_clark_scheme const &) = delete;
  ade_heat_bakarat_clark_scheme &operator=(ade_heat_bakarat_clark_scheme &&) =
      delete;

  // stability check:
  bool is_stable() const override { return true; };

  // for Dirichlet BC
  void operator()(
      sptr_t<dirichlet_boundary_1d<fp_type>> const &dirichlet_boundary,
      container_t &solution) const override;
  // for Robin BC
  void operator()(robin_boundary<fp_type> const &robin_boundary,
                  container_t &solution) const override;
};

// ============================================================================
// ==================== ade_heat_saulyev_scheme ===============================
// ============================================================================

template <template <typename, typename> typename container, typename fp_type,
          typename alloc>
class ade_heat_saulyev_scheme
    : public heat_scheme_base<container, fp_type, alloc,
                              pde_coefficient_holder_const<fp_type>> {
 private:
  typedef container<fp_type, alloc> container_t;
  explicit ade_heat_saulyev_scheme() = default;

 public:
  explicit ade_heat_saulyev_scheme(
      fp_type space_start, fp_type terminal_time,
      std::pair<fp_type, fp_type> const &deltas,
      pde_coefficient_holder_const<fp_type> const &coeffs,
      container_t const &initial_condition,
      std::function<fp_type(fp_type, fp_type)> const &source = nullptr,
      bool is_source_set = false)
      : heat_scheme_base<container, fp_type, alloc,
                         pde_coefficient_holder_const<fp_type>>(
            space_start, terminal_time, deltas, coeffs, initial_condition,
            source, is_source_set) {}

  ~ade_heat_saulyev_scheme() {}

  ade_heat_saulyev_scheme(ade_heat_saulyev_scheme const &) = delete;
  ade_heat_saulyev_scheme(ade_heat_saulyev_scheme &&) = delete;
  ade_heat_saulyev_scheme &operator=(ade_heat_saulyev_scheme const &) = delete;
  ade_heat_saulyev_scheme &operator=(ade_heat_saulyev_scheme &&) = delete;

  // stability check:
  bool is_stable() const override { return true; };

  // for Dirichlet BC
  void operator()(
      sptr_t<dirichlet_boundary_1d<fp_type>> const &dirichlet_boundary,
      container_t &solution) const override;
  // for Robin BC
  void operator()(robin_boundary<fp_type> const &robin_boundary,
                  container_t &solution) const override;
};

}  // namespace lss_one_dim_heat_explicit_schemes

// ============================================================================
// =========================== IMPLEMENTATIONS ================================

template <template <typename, typename> typename container, typename fp_type,
          typename alloc>
bool lss_one_dim_heat_explicit_schemes::heat_euler_scheme<
    container, fp_type, alloc>::is_stable() const {
  fp_type const A = std::get<0>(coeffs_);
  fp_type const B = std::get<1>(coeffs_);
  fp_type const k = std::get<0>(deltas_);
  fp_type const h = std::get<1>(deltas_);

  return (((2.0 * A * k / (h * h)) <= 1.0) && (B * (k / h) <= 1.0));
}

template <template <typename, typename> typename container, typename fp_type,
          typename alloc>
void lss_one_dim_heat_explicit_schemes::
    heat_euler_scheme<container, fp_type, alloc>::operator()(
        sptr_t<dirichlet_boundary_1d<fp_type>> const &dirichlet_boundary,
        container_t &solution) const {
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");
  LSS_ASSERT(
      solution.size() == initial_condition_.size(),
      "Entered solution vector size differs from initialCondition vector.");
  LSS_ASSERT(is_stable() == true, "This discretization is not stable.");
  typedef discretization<fp_type, container, alloc> d_1d;
  // get delta time:
  fp_type const k = std::get<0>(deltas_);
  // get delta space:
  fp_type const h = std::get<1>(deltas_);
  // create first time point:
  fp_type time = k;
  // get coefficients:
  fp_type const A = std::get<0>(coeffs_);
  fp_type const B = std::get<1>(coeffs_);
  fp_type const C = std::get<2>(coeffs_);
  // calculate scheme coefficients:
  fp_type const lambda = (A * k) / (h * h);
  fp_type const gamma = (B * k) / (2.0 * h);
  fp_type const delta = C * k;
  // set up coefficients:
  fp_type const a = 1.0 - (2.0 * lambda - delta);
  fp_type const b = lambda + gamma;
  fp_type const c = lambda - gamma;
  // previous solution:
  container_t prev_sol = initial_condition_;
  // size of the space vector:
  std::size_t const space_size = solution.size();
  if (!is_source_set_) {
    // loop for stepping in time:
    while (time <= terminal_time_) {
      dirichlet_boundary->fill(time, solution);
      for (std::size_t t = 1; t < space_size - 1; ++t) {
        solution[t] =
            a * prev_sol[t] + b * prev_sol[t + 1] + c * prev_sol[t - 1];
      }
      prev_sol = solution;
      time += k;
    }
  } else {
    // create a container to carry discretized source heat
    container_t source_curr(solution.size(), fp_type{});
    d_1d::discretize_in_space(h, space_start_, 0.0, source_, source_curr);
    // loop for stepping in time:
    while (time <= terminal_time_) {
      dirichlet_boundary->fill(time, solution);
      for (std::size_t t = 1; t < space_size - 1; ++t) {
        solution[t] = a * prev_sol[t] + b * prev_sol[t + 1] +
                      c * prev_sol[t - 1] + k * source_curr[t];
      }
      d_1d::discretize_in_space(h, space_start_, time, source_, source_curr);
      prev_sol = solution;
      time += k;
    }
  }
}

template <template <typename, typename> typename container, typename fp_type,
          typename alloc>
void lss_one_dim_heat_explicit_schemes::heat_euler_scheme<
    container, fp_type, alloc>::operator()(robin_boundary<fp_type> const
                                               &robin_boundary,
                                           container_t &solution) const {
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");
  LSS_ASSERT(
      solution.size() == initial_condition_.size(),
      "Entered solution vector size differs from initialCondition vector.");
  LSS_ASSERT(is_stable() == true, "This discretization is not stable.");
  typedef discretization<fp_type, container, alloc> d_1d;
  // get delta time:
  fp_type const k = std::get<0>(deltas_);
  // get delta space:
  fp_type const h = std::get<1>(deltas_);
  // create first time point:
  fp_type time = k;
  // get coefficients:
  fp_type const A = std::get<0>(coeffs_);
  fp_type const B = std::get<1>(coeffs_);
  fp_type const C = std::get<2>(coeffs_);
  // calculate scheme coefficients:
  fp_type const lambda = (A * k) / (h * h);
  fp_type const gamma = (B * k) / (2.0 * h);
  fp_type const delta = C * k;
  // left space boundary:
  fp_type const left_lin = robin_boundary.left.first;
  fp_type const left_const = robin_boundary.left.second;
  // right space boundary:
  fp_type const right_lin_ = robin_boundary.right.first;
  fp_type const right_const_ = robin_boundary.right.second;
  // conversion of right hand boundaries:
  fp_type const right_lin = 1.0 / right_lin_;
  fp_type const right_const = -1.0 * (right_const_ / right_lin_);
  // set up coefficients:
  fp_type const a = 1.0 - (2.0 * lambda - delta);
  fp_type const b = lambda + gamma;
  fp_type const c = lambda - gamma;
  // previous solution:
  container_t prev_sol = initial_condition_;
  // size of the space vector:
  std::size_t const space_size = solution.size();
  if (!is_source_set_) {
    // loop for stepping in time:
    while (time <= terminal_time_) {
      solution[0] =
          (b + (c * left_lin)) * prev_sol[1] + a * prev_sol[0] + c * left_const;
      solution[solution.size() - 1] =
          (c + (b * right_lin)) * prev_sol[solution.size() - 2] +
          a * prev_sol[solution.size() - 1] + b * right_const;
      for (std::size_t t = 1; t < space_size - 1; ++t) {
        solution[t] =
            a * prev_sol[t] + b * prev_sol[t + 1] + c * prev_sol[t - 1];
      }
      prev_sol = solution;
      time += k;
    }
  } else {
    // create a container to carry discretized source heat
    container_t source_curr(solution.size(), fp_type{});
    d_1d::discretize_in_space(h, space_start_, 0.0, source_, source_curr);
    // loop for stepping in time:
    while (time <= terminal_time_) {
      solution[0] =
          (b + (c * left_lin)) * prev_sol[1] + a * prev_sol[0] + c * left_const;
      solution[solution.size() - 1] =
          (c + (b * right_lin)) * prev_sol[solution.size() - 2] +
          a * prev_sol[solution.size() - 1] + b * right_const;
      for (std::size_t t = 1; t < space_size - 1; ++t) {
        solution[t] = a * prev_sol[t] + b * prev_sol[t + 1] +
                      c * prev_sol[t - 1] + k * source_curr[t];
      }
      d_1d::discretize_in_space(h, space_start_, time, source_, source_curr);
      prev_sol = solution;
      time += k;
    }
  }
}

template <template <typename, typename> typename container, typename fp_type,
          typename alloc>
void lss_one_dim_heat_explicit_schemes::
    ade_heat_bakarat_clark_scheme<container, fp_type, alloc>::operator()(
        sptr_t<dirichlet_boundary_1d<fp_type>> const &dirichlet_boundary,
        container_t &solution) const {
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");
  LSS_ASSERT(
      solution.size() == initial_condition_.size(),
      "Entered solution vector size differs from initialCondition vector.");
  typedef discretization<fp_type, container, alloc> d_1d;
  // get delta time:
  fp_type const k = std::get<0>(deltas_);
  // get delta space:
  fp_type const h = std::get<1>(deltas_);
  // create first time point:
  fp_type time = k;
  // get coefficients:
  fp_type const A = std::get<0>(coeffs_);
  fp_type const B = std::get<1>(coeffs_);
  fp_type const C = std::get<2>(coeffs_);
  // calculate scheme coefficients:
  fp_type const lambda = (A * k) / (h * h);
  fp_type const gamma = (B * k) / (2.0 * h);
  fp_type const delta = C * k / 2.0;
  // set up coefficients:
  fp_type const divisor = 1.0 + lambda - delta;
  fp_type const a = (1.0 - lambda + delta) / divisor;
  fp_type const b = (lambda + gamma) / divisor;
  fp_type const c = (lambda - gamma) / divisor;
  fp_type const d = k / divisor;
  // left space boundary:
  auto const &left = dirichlet_boundary->first;
  // right space boundary:
  auto const &right = dirichlet_boundary->second;
  // conmponents of the solution:
  container_t com_1(initial_condition_);
  container_t com_2(initial_condition_);
  // size of the space vector:
  std::size_t const space_size = solution.size();
  // create a container to carry discretized source heat
  container_t source_curr(space_size, fp_type{});
  container_t source_next(space_size, fp_type{});
  // create upsweep anonymous function:
  auto up_sweep = [=](container_t &up_component, container_t const &rhs,
                      fp_type rhs_coeff) {
    for (std::size_t t = 1; t < space_size - 1; ++t) {
      up_component[t] = a * up_component[t] + b * up_component[t + 1] +
                        c * up_component[t - 1] + d * rhs_coeff * rhs[t];
    }
  };
  // create downsweep anonymous function:
  auto down_sweep = [=](container_t &down_component, container_t const &rhs,
                        fp_type rhs_coeff) {
    for (std::size_t t = space_size - 2; t >= 1; --t) {
      down_component[t] = a * down_component[t] + b * down_component[t + 1] +
                          c * down_component[t - 1] + d * rhs_coeff * rhs[t];
    }
  };

  if (!is_source_set_) {
    // loop for stepping in time:
    while (time <= terminal_time_) {
      dirichlet_boundary->fill(time, com_1);
      dirichlet_boundary->fill(time, com_2);
      std::thread up_sweep_tr(std::move(up_sweep), std::ref(com_1), source_curr,
                              0.0);
      std::thread down_sweep_tr(std::move(down_sweep), std::ref(com_2),
                                source_curr, 0.0);
      up_sweep_tr.join();
      down_sweep_tr.join();
      for (std::size_t t = 0; t < space_size; ++t) {
        solution[t] = 0.5 * (com_1[t] + com_2[t]);
      }
      time += k;
    }
  } else {
    d_1d::discretize_in_space(h, space_start_, 0.0, source_, source_curr);
    d_1d::discretize_in_space(h, space_start_, time, source_, source_next);
    // loop for stepping in time:
    while (time <= terminal_time_) {
      dirichlet_boundary->fill(time, com_1);
      dirichlet_boundary->fill(time, com_2);
      std::thread up_sweep_tr(std::move(up_sweep), std::ref(com_1), source_next,
                              1.0);
      std::thread down_sweep_tr(std::move(down_sweep), std::ref(com_2),
                                source_curr, 1.0);
      up_sweep_tr.join();
      down_sweep_tr.join();
      for (std::size_t t = 0; t < space_size; ++t) {
        solution[t] = 0.5 * (com_1[t] + com_2[t]);
      }
      d_1d::discretize_in_space(h, space_start_, time, source_, source_curr);
      d_1d::discretize_in_space(h, space_start_, 2.0 * time, source_,
                                source_next);
      time += k;
    }
  }
}

template <template <typename, typename> typename container, typename fp_type,
          typename alloc>
void lss_one_dim_heat_explicit_schemes::ade_heat_bakarat_clark_scheme<
    container, fp_type, alloc>::operator()(robin_boundary<fp_type> const
                                               &robin_boundary,
                                           container_t &solution) const {
  throw new std::exception("Not available.");
}

template <template <typename, typename> typename container, typename fp_type,
          typename alloc>
void lss_one_dim_heat_explicit_schemes::
    ade_heat_saulyev_scheme<container, fp_type, alloc>::operator()(
        sptr_t<dirichlet_boundary_1d<fp_type>> const &dirichlet_boundary,
        container_t &solution) const {
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");
  LSS_ASSERT(
      solution.size() == initial_condition_.size(),
      "Entered solution vector size differs from initialCondition vector.");
  typedef discretization<fp_type, container, alloc> d_1d;
  // get delta time:
  fp_type const k = std::get<0>(deltas_);
  // get delta space:
  fp_type const h = std::get<1>(deltas_);
  // create first time point:
  fp_type time = k;
  // get coefficients:
  fp_type const A = std::get<0>(coeffs_);
  fp_type const B = std::get<1>(coeffs_);
  fp_type const C = std::get<2>(coeffs_);
  // calculate scheme coefficients:
  fp_type const lambda = (A * k) / (h * h);
  fp_type const gamma = (B * k) / (2.0 * h);
  fp_type const delta = C * k / 2.0;
  // set up coefficients:
  fp_type const divisor = 1.0 + lambda - delta;
  fp_type const a = (1.0 - lambda + delta) / divisor;
  fp_type const b = (lambda + gamma) / divisor;
  fp_type const c = (lambda - gamma) / divisor;
  fp_type const d = k / divisor;

  // get the initial condition :
  solution = initial_condition_;
  // size of the space vector:
  std::size_t const space_size = solution.size();
  // create a container to carry discretized source heat
  container_t source_curr(space_size, fp_type{});
  container_t source_next(space_size, fp_type{});
  // create upsweep anonymous function:
  auto up_sweep = [=](container_t &up_component, container_t const &rhs,
                      fp_type rhs_coeff) {
    for (std::size_t t = 1; t < space_size - 1; ++t) {
      up_component[t] = a * up_component[t] + b * up_component[t + 1] +
                        c * up_component[t - 1] + d * rhs_coeff * rhs[t];
    }
  };
  // create downsweep anonymous function:
  auto down_sweep = [=](container_t &down_component, container_t const &rhs,
                        fp_type rhs_coeff) {
    for (std::size_t t = space_size - 2; t >= 1; --t) {
      down_component[t] = a * down_component[t] + b * down_component[t + 1] +
                          c * down_component[t - 1] + d * rhs_coeff * rhs[t];
    }
  };

  if (!is_source_set_) {
    // loop for stepping in time:
    std::size_t t = 1;
    while (time <= terminal_time_) {
      dirichlet_boundary->fill(time, solution);
      if (t % 2 == 0)
        down_sweep(solution, source_curr, 0.0);
      else
        up_sweep(solution, source_curr, 0.0);
      ++t;
      time += k;
    }
  } else {
    d_1d::discretize_in_space(h, space_start_, 0.0, source_, source_curr);
    d_1d::discretize_in_space(h, space_start_, time, source_, source_next);
    // loop for stepping in time:
    std::size_t t = 1;
    while (time <= terminal_time_) {
      dirichlet_boundary->fill(time, solution);
      if (t % 2 == 0)
        down_sweep(solution, source_curr, 1.0);
      else
        up_sweep(solution, source_next, 1.0);
      ++t;
      d_1d::discretize_in_space(h, space_start_, time, source_, source_curr);
      d_1d::discretize_in_space(h, space_start_, 2.0 * time, source_,
                                source_next);
      time += k;
    }
  }
}

template <template <typename, typename> typename container, typename fp_type,
          typename alloc>
void lss_one_dim_heat_explicit_schemes::ade_heat_saulyev_scheme<
    container, fp_type, alloc>::operator()(robin_boundary<fp_type> const
                                               &robin_boundary,
                                           container_t &solution) const {
  throw new std::exception("Not available.");
}

#endif  ///_LSS_1D_HEAT_EXPLICIT_SCHEMES
