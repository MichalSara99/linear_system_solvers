#pragma once
#if !defined(_LSS_2D_HEAT_EXPLICIT_SCHEMES)
#define _LSS_2D_HEAT_EXPLICIT_SCHEMES

#pragma warning(disable : 4244)

#include <thread>

#include "common/lss_enumerations.h"
#include "common/lss_utility.h"
#include "pde_solvers/two_dim/lss_base_explicit_schemes.h"
#include "pde_solvers/two_dim/lss_pde_utility.h"

namespace lss_two_dim_heat_explicit_schemes {

using lss_enumerations::boundary_condition_enum;
using lss_two_dim_base_explicit_schemes::heat_scheme_base;
using lss_two_dim_pde_utility::dirichlet_boundary_2d;
using lss_two_dim_pde_utility::pde_coefficient_holder_const;
using lss_two_dim_pde_utility::robin_boundary_2d;
using lss_two_dim_pde_utility::v_discretization_2d;

// ============================================================================
// ======================= heat_euler_scheme ==================================
// ============================================================================

template <template <typename, typename> typename container, typename fp_type,
          typename alloc>
class heat_euler_scheme
    : public heat_scheme_base<container, fp_type, alloc,
                              pde_coefficient_holder_const<fp_type>> {
 protected:
  typedef container_2d<container, fp_type, alloc> matrix_t;
  heat_euler_scheme() = default;

 public:
  explicit heat_euler_scheme(
      fp_type time, fp_type time_delta,
      std::pair<fp_type, fp_type> const &spatial_inits,
      std::pair<fp_type, fp_type> const &spatial_deltas,
      sptr_t<pde_coefficient_holder_const<fp_type>> const &coeffs,
      sptr_t<matrix_t> const &initial_condition,
      std::function<fp_type(fp_type, fp_type, fp_type)> const &source = nullptr,
      bool is_source_set = false)
      : heat_scheme_base<container, fp_type, alloc,
                         pde_coefficient_holder_const<fp_type>>(
            time, time_delta, spatial_inits, spatial_deltas, coeffs,
            initial_condition, source, is_source_set) {}

  ~heat_euler_scheme() {}

  heat_euler_scheme(heat_euler_scheme const &) = delete;
  heat_euler_scheme(heat_euler_scheme &&) = delete;
  heat_euler_scheme &operator=(heat_euler_scheme const &) = delete;
  heat_euler_scheme &operator=(heat_euler_scheme &&) = delete;

  // stability check:
  bool is_stable() const override;

  // for Dirichlet BC
  void operator()(
      sptr_t<dirichlet_boundary_2d<fp_type>> const &dirichlet_boundary,
      matrix_t &solution) const override;
  // for Robin BC
  void operator()(sptr_t<robin_boundary_2d<fp_type>> const &robin_boundary,
                  matrix_t &solution) const override;
};

// ============================================================================
// ================== ade_heat_bakarat_clark_scheme  ==========================
// ============================================================================

template <template <typename, typename> typename container, typename fp_type,
          typename alloc>
class ade_heat_bakarat_clark_scheme
    : public heat_scheme_base<container, fp_type, alloc,
                              pde_coefficient_holder_const<fp_type>> {
 protected:
  typedef container_2d<container, fp_type, alloc> matrix_t;
  ade_heat_bakarat_clark_scheme() = default;

 public:
  explicit ade_heat_bakarat_clark_scheme(
      fp_type time, fp_type time_delta,
      std::pair<fp_type, fp_type> const &spatial_inits,
      std::pair<fp_type, fp_type> const &spatial_deltas,
      sptr_t<pde_coefficient_holder_const<fp_type>> const &coeffs,
      sptr_t<matrix_t> const &initial_condition,
      std::function<fp_type(fp_type, fp_type, fp_type)> const &source = nullptr,
      bool is_source_set = false)
      : heat_scheme_base<container, fp_type, alloc,
                         pde_coefficient_holder_const<fp_type>>(
            time, time_delta, spatial_inits, spatial_deltas, coeffs,
            initial_condition, source, is_source_set) {}

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
      sptr_t<dirichlet_boundary_2d<fp_type>> const &dirichlet_boundary,
      matrix_t &solution) const override;
  // for Robin BC
  void operator()(sptr_t<robin_boundary_2d<fp_type>> const &robin_boundary,
                  matrix_t &solution) const override;
};

// ============================================================================
// ==================== ade_heat_saulyev_scheme ===============================
// ============================================================================

template <template <typename, typename> typename container, typename fp_type,
          typename alloc>
class ade_heat_saulyev_scheme
    : public heat_scheme_base<container, fp_type, alloc,
                              pde_coefficient_holder_const<fp_type>> {
 protected:
  typedef container_2d<container, fp_type, alloc> matrix_t;
  ade_heat_saulyev_scheme() = default;

 public:
  explicit ade_heat_saulyev_scheme(
      fp_type time, fp_type time_delta,
      std::pair<fp_type, fp_type> const &spatial_inits,
      std::pair<fp_type, fp_type> const &spatial_deltas,
      sptr_t<pde_coefficient_holder_const<fp_type>> const &coeffs,
      sptr_t<matrix_t> const &initial_condition,
      std::function<fp_type(fp_type, fp_type, fp_type)> const &source = nullptr,
      bool is_source_set = false)
      : heat_scheme_base<container, fp_type, alloc,
                         pde_coefficient_holder_const<fp_type>>(
            time, time_delta, spatial_inits, spatial_deltas, coeffs,
            initial_condition, source, is_source_set) {}

  ~ade_heat_saulyev_scheme() {}

  ade_heat_saulyev_scheme(ade_heat_saulyev_scheme const &) = delete;
  ade_heat_saulyev_scheme(ade_heat_saulyev_scheme &&) = delete;
  ade_heat_saulyev_scheme &operator=(ade_heat_saulyev_scheme const &) = delete;
  ade_heat_saulyev_scheme &operator=(ade_heat_saulyev_scheme &&) = delete;

  // stability check:
  bool is_stable() const override { return true; };

  // for Dirichlet BC
  void operator()(
      sptr_t<dirichlet_boundary_2d<fp_type>> const &dirichlet_boundary,
      matrix_t &solution) const override;
  // for Robin BC
  void operator()(sptr_t<robin_boundary_2d<fp_type>> const &robin_boundary,
                  matrix_t &solution) const override;
};

}  // namespace lss_two_dim_heat_explicit_schemes

// ============================================================================
// =========================== IMPLEMENTATIONS ================================

template <template <typename, typename> typename container, typename fp_type,
          typename alloc>
bool lss_two_dim_heat_explicit_schemes::heat_euler_scheme<
    container, fp_type, alloc>::is_stable() const {
  throw std::exception("Not yet implemented. TODO");
}

template <template <typename, typename> typename container, typename fp_type,
          typename alloc>
void lss_two_dim_heat_explicit_schemes::
    heat_euler_scheme<container, fp_type, alloc>::operator()(
        sptr_t<dirichlet_boundary_2d<fp_type>> const &dirichlet_boundary,
        matrix_t &solution) const {
  throw std::exception("Not yet implemented. TODO");
}

template <template <typename, typename> typename container, typename fp_type,
          typename alloc>
void lss_two_dim_heat_explicit_schemes::heat_euler_scheme<
    container, fp_type,
    alloc>::operator()(sptr_t<robin_boundary_2d<fp_type>> const &robin_boundary,
                       matrix_t &solution) const {
  throw std::exception("Not yet implemented. TODO");
}

template <template <typename, typename> typename container, typename fp_type,
          typename alloc>
void lss_two_dim_heat_explicit_schemes::
    ade_heat_bakarat_clark_scheme<container, fp_type, alloc>::operator()(
        sptr_t<dirichlet_boundary_2d<fp_type>> const &dirichlet_boundary,
        matrix_t &solution) const {
  LSS_ASSERT(((solution.rows() > 0) && (solution.columns())),
             "The input solution container must be initialized.");
  LSS_ASSERT(
      ((solution.rows() == initial_condition_.rows()) &&
       (solution.columns() == initial_condition_.columns())),
      "Entered solution vector size differs from initialCondition vector.");
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
  auto const &left = dirichlet_boundary.first;
  // right space boundary:
  auto const &right = dirichlet_boundary.second;
  // conmponents of the solution:
  std::vector<fp_type> com_1(initial_condition_);
  std::vector<fp_type> com_2(initial_condition_);
  // size of the space vector:
  std::size_t const space_size = solution.size();
  // create a container to carry discretized source heat
  std::vector<fp_type> source_curr(space_size, fp_type{});
  std::vector<fp_type> source_next(space_size, fp_type{});
  // create upsweep anonymous function:
  auto up_sweep = [=](std::vector<fp_type> &up_component,
                      std::vector<fp_type> const &rhs, fp_type rhs_coeff) {
    for (std::size_t t = 1; t < space_size - 1; ++t) {
      up_component[t] = a * up_component[t] + b * up_component[t + 1] +
                        c * up_component[t - 1] + d * rhs_coeff * rhs[t];
    }
  };
  // create downsweep anonymous function:
  auto down_sweep = [=](std::vector<fp_type> &down_component,
                        std::vector<fp_type> const &rhs, fp_type rhs_coeff) {
    for (std::size_t t = space_size - 2; t >= 1; --t) {
      down_component[t] = a * down_component[t] + b * down_component[t + 1] +
                          c * down_component[t - 1] + d * rhs_coeff * rhs[t];
    }
  };

  if (!is_source_set_) {
    // loop for stepping in time:
    while (time <= terminal_time_) {
      com_1[0] = com_2[0] = left(time);
      com_1[solution.size() - 1] = com_2[solution.size() - 1] = right(time);
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
    v_discretization<fp_type>::discretize_in_space(h, space_start_, 0.0,
                                                   source_, source_curr);
    v_discretization<fp_type>::discretize_in_space(h, space_start_, time,
                                                   source_, source_next);
    // loop for stepping in time:
    while (time <= terminal_time_) {
      com_1[0] = com_2[0] = left(time);
      com_1[solution.size() - 1] = com_2[solution.size() - 1] = right(time);
      std::thread up_sweep_tr(std::move(up_sweep), std::ref(com_1), source_next,
                              1.0);
      std::thread down_sweep_tr(std::move(down_sweep), std::ref(com_2),
                                source_curr, 1.0);
      up_sweep_tr.join();
      down_sweep_tr.join();
      for (std::size_t t = 0; t < space_size; ++t) {
        solution[t] = 0.5 * (com_1[t] + com_2[t]);
      }
      v_discretization<fp_type>::discretize_in_space(h, space_start_, time,
                                                     source_, source_curr);
      v_discretization<fp_type>::discretize_in_space(
          h, space_start_, 2.0 * time, source_, source_next);
      time += k;
    }
  }
}

template <template <typename, typename> typename container, typename fp_type,
          typename alloc>
void lss_two_dim_heat_explicit_schemes::ade_heat_bakarat_clark_scheme<
    container, fp_type,
    alloc>::operator()(sptr_t<robin_boundary_2d<fp_type>> const &robin_boundary,
                       matrix_t &solution) const {
  throw new std::exception("Not available.");
}

template <template <typename, typename> typename container, typename fp_type,
          typename alloc>
void lss_two_dim_heat_explicit_schemes::
    ade_heat_saulyev_scheme<container, fp_type, alloc>::operator()(
        sptr_t<dirichlet_boundary_2d<fp_type>> const &dirichlet_boundary,
        matrix_t &solution) const {
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");
  LSS_ASSERT(
      solution.size() == initial_condition_.size(),
      "Entered solution vector size differs from initialCondition vector.");
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
  auto const &left = dirichlet_boundary.first;
  // right space boundary:
  auto const &right = dirichlet_boundary.second;
  // get the initial condition :
  solution = initial_condition_;
  // size of the space vector:
  std::size_t const space_size = solution.size();
  // create a container to carry discretized source heat
  std::vector<fp_type> source_curr(space_size, fp_type{});
  std::vector<fp_type> source_next(space_size, fp_type{});
  // create upsweep anonymous function:
  auto up_sweep = [=](std::vector<fp_type> &up_component,
                      std::vector<fp_type> const &rhs, fp_type rhs_coeff) {
    for (std::size_t t = 1; t < space_size - 1; ++t) {
      up_component[t] = a * up_component[t] + b * up_component[t + 1] +
                        c * up_component[t - 1] + d * rhs_coeff * rhs[t];
    }
  };
  // create downsweep anonymous function:
  auto down_sweep = [=](std::vector<fp_type> &down_component,
                        std::vector<fp_type> const &rhs, fp_type rhs_coeff) {
    for (std::size_t t = space_size - 2; t >= 1; --t) {
      down_component[t] = a * down_component[t] + b * down_component[t + 1] +
                          c * down_component[t - 1] + d * rhs_coeff * rhs[t];
    }
  };

  if (!is_source_set_) {
    // loop for stepping in time:
    std::size_t t = 1;
    while (time <= terminal_time_) {
      solution[0] = left(time);
      solution[solution.size() - 1] = right(time);
      if (t % 2 == 0)
        down_sweep(solution, source_curr, 0.0);
      else
        up_sweep(solution, source_curr, 0.0);
      ++t;
      time += k;
    }
  } else {
    v_discretization<fp_type>::discretize_in_space(h, space_start_, 0.0,
                                                   source_, source_curr);
    v_discretization<fp_type>::discretize_in_space(h, space_start_, time,
                                                   source_, source_next);
    // loop for stepping in time:
    std::size_t t = 1;
    while (time <= terminal_time_) {
      solution[0] = left(time);
      solution[solution.size() - 1] = right(time);
      if (t % 2 == 0)
        down_sweep(solution, source_curr, 1.0);
      else
        up_sweep(solution, source_next, 1.0);
      ++t;
      v_discretization<fp_type>::discretize_in_space(h, space_start_, time,
                                                     source_, source_curr);
      v_discretization<fp_type>::discretize_in_space(
          h, space_start_, 2.0 * time, source_, source_next);
      time += k;
    }
  }
}

template <template <typename, typename> typename container, typename fp_type,
          typename alloc>
void lss_two_dim_heat_explicit_schemes::ade_heat_saulyev_scheme<
    container, fp_type,
    alloc>::operator()(sptr_t<robin_boundary_2d<fp_type>> const &robin_boundary,
                       matrix_t &solution) const {
  throw new std::exception("Not available.");
}

#endif  //_LSS_2D_HEAT_EXPLICIT_SCHEMES
