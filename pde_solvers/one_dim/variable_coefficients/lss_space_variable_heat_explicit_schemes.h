#pragma once
#if !defined(_LSS_SPACE_VARIABLE_HEAT_EXPLICIT_SCHEMES)
#define _LSS_SPACE_VARIABLE_HEAT_EXPLICIT_SCHEMES

#pragma warning(disable : 4244)

#include <thread>

#include "common/lss_enumerations.h"
#include "pde_solvers/one_dim/lss_base_explicit_schemes.h"
#include "pde_solvers/one_dim/lss_pde_utility.h"

namespace lss_one_dim_space_variable_heat_explicit_schemes {

using lss_enumerations::boundary_condition_enum;
using lss_one_dim_base_explicit_schemes::heat_scheme_base;
using lss_one_dim_pde_utility::dirichlet_boundary;
using lss_one_dim_pde_utility::pde_coefficient_holder_fun_1_arg;
using lss_one_dim_pde_utility::robin_boundary;

// ============================================================================
// ============================= heat_euler_scheme ============================
// ============================================================================

template <typename fp_type>
class heat_euler_scheme
    : public heat_scheme_base<fp_type,
                              pde_coefficient_holder_fun_1_arg<fp_type>> {
 private:
  pde_coefficient_holder_fun_1_arg<fp_type> pde_coeffs_;

 public:
  explicit heat_euler_scheme() = delete;
  explicit heat_euler_scheme(
      fp_type space_start, fp_type terminal_time,
      std::pair<fp_type, fp_type> const &deltas,
      pde_coefficient_holder_fun_1_arg<fp_type> const &pde_coeffs,
      pde_coefficient_holder_fun_1_arg<fp_type> const &coeffs,
      std::vector<fp_type> const &initial_condition,
      std::function<fp_type(fp_type, fp_type)> const &source = nullptr,
      bool is_source_set = false)
      : heat_scheme_base<fp_type, pde_coefficient_holder_fun_1_arg<fp_type>>(
            space_start, terminal_time, deltas, coeffs, initial_condition,
            source, is_source_set),
        pde_coeffs_{pde_coeffs} {}

  ~heat_euler_scheme() {}

  heat_euler_scheme(heat_euler_scheme const &) = delete;
  heat_euler_scheme(heat_euler_scheme &&) = delete;
  heat_euler_scheme &operator=(heat_euler_scheme const &) = delete;
  heat_euler_scheme &operator=(heat_euler_scheme &&) = delete;

  // stability check:
  bool is_stable() const override;

  // for Dirichlet BC
  void operator()(dirichlet_boundary<fp_type> const &dirichlet_boundary,
                  std::vector<fp_type> &solution) const override;
  // for Robin BC
  void operator()(robin_boundary<fp_type> const &robin_boundary,
                  std::vector<fp_type> &solution) const override;
};

// ============================================================================
// ================== ade_heat_bakarat_clark_scheme ===========================
// ============================================================================

template <typename fp_type>
class ade_heat_bakarat_clark_scheme
    : public heat_scheme_base<fp_type,
                              pde_coefficient_holder_fun_1_arg<fp_type>> {
 public:
  explicit ade_heat_bakarat_clark_scheme() = delete;
  explicit ade_heat_bakarat_clark_scheme(
      fp_type space_start, fp_type terminal_time,
      std::pair<fp_type, fp_type> const &deltas,
      pde_coefficient_holder_fun_1_arg<fp_type> const &coeffs,
      std::vector<fp_type> const &initial_condition,
      std::function<fp_type(fp_type, fp_type)> const &source = nullptr,
      bool is_source_set = false)
      : heat_scheme_base<fp_type, pde_coefficient_holder_fun_1_arg<fp_type>>(
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
  void operator()(dirichlet_boundary<fp_type> const &dirichlet_boundary,
                  std::vector<fp_type> &solution) const override;
  // for Robin BC
  void operator()(robin_boundary<fp_type> const &robin_boundary,
                  std::vector<fp_type> &solution) const override;
};

// ============================================================================
// ======================= ade_heat_saulyev_scheme ============================
// ============================================================================

template <typename fp_type>
class ade_heat_saulyev_scheme
    : public heat_scheme_base<fp_type,
                              pde_coefficient_holder_fun_1_arg<fp_type>> {
 public:
  explicit ade_heat_saulyev_scheme() = delete;
  explicit ade_heat_saulyev_scheme(
      fp_type space_start, fp_type terminal_time,
      std::pair<fp_type, fp_type> const &deltas,
      pde_coefficient_holder_fun_1_arg<fp_type> const &coeffs,
      std::vector<fp_type> const &initial_condition,
      std::function<fp_type(fp_type, fp_type)> const &source = nullptr,
      bool is_source_set = false)
      : heat_scheme_base<fp_type, pde_coefficient_holder_fun_1_arg<fp_type>>(
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
  void operator()(dirichlet_boundary<fp_type> const &dirichlet_boundary,
                  std::vector<fp_type> &solution) const override;
  // for Robin BC
  void operator()(robin_boundary<fp_type> const &robin_boundary,
                  std::vector<fp_type> &solution) const override;
};

}  // namespace lss_one_dim_space_variable_heat_explicit_schemes

// ============================================================================
// ======================== IMPLEMENTATIONS ===================================

template <typename fp_type>
bool lss_one_dim_space_variable_heat_explicit_schemes::heat_euler_scheme<
    fp_type>::is_stable() const {
  auto const &a = std::get<0>(pde_coeffs_);
  auto const &b = std::get<1>(pde_coeffs_);
  auto const &c = std::get<2>(pde_coeffs_);
  fp_type const k = std::get<0>(deltas_);
  fp_type const h = std::get<1>(deltas_);
  fp_type const lambda = k / (h * h);
  fp_type const gamma = k / h;

  const std::size_t space_size = initial_condition_.size();
  for (std::size_t i = 0; i < space_size; ++i) {
    if (c(i * h) > 0.0) return false;
    if ((2.0 * lambda * a(i * h) - k * c(i * h)) > 1.0) return false;
    if (((gamma * std::abs(b(i * h))) * (gamma * std::abs(b(i * h)))) >
        (2.0 * lambda * a(i * h)))
      return false;
  }
  return true;
}

template <typename fp_type>
void lss_one_dim_space_variable_heat_explicit_schemes::heat_euler_scheme<
    fp_type>::operator()(dirichlet_boundary<fp_type> const &dirichlet_boundary,
                         std::vector<fp_type> &solution) const {
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");
  LSS_ASSERT(
      solution.size() == initial_condition_.size(),
      "Entered solution vector size differs from initialCondition vector.");
  LSS_ASSERT(is_stable() == true, "This discretization is not stable.");
  // get delta time:
  fp_type const k = std::get<0>(deltas_);
  // get delta space:
  fp_type const h = std::get<1>(deltas_);
  // create first time point:
  fp_type time = k;
  // get coefficients:
  auto const &A = std::get<0>(coeffs_);
  auto const &B = std::get<1>(coeffs_);
  auto const &D = std::get<2>(coeffs_);
  // previous solution:
  std::vector<fp_type> prev_sol = initial_condition_;
  // left space boundary:
  auto const &left = dirichlet_boundary.first;
  // right space boundary:
  auto const &right = dirichlet_boundary.second;
  // size of the space vector:
  std::size_t const space_size = solution.size();
  if (!is_source_set_) {
    // loop for stepping in time:
    while (time <= terminal_time_) {
      solution[0] = left(time);
      solution[solution.size() - 1] = right(time);
      for (std::size_t t = 1; t < space_size - 1; ++t) {
        solution[t] = (1.0 - 2.0 * B(t * h)) * prev_sol[t] +
                      D(t * h) * prev_sol[t + 1] + A(t * h) * prev_sol[t - 1];
      }
      prev_sol = solution;
      time += k;
    }
  } else {
    // create a container to carry discretized source heat
    std::vector<fp_type> source_curr(solution.size(), fp_type{});
    discretize_in_space(h, space_start_, 0.0, source_, source_curr);
    // loop for stepping in time:
    while (time <= terminal_time_) {
      solution[0] = left(time);
      solution[solution.size() - 1] = right(time);
      for (std::size_t t = 1; t < space_size - 1; ++t) {
        solution[t] = solution[t] =
            (1.0 - 2.0 * B(t * h)) * prev_sol[t] + D(t * h) * prev_sol[t + 1] +
            A(t * h) * prev_sol[t - 1] + k * source_curr[t];
      }
      discretize_in_space(h, space_start_, time, source_, source_curr);
      prev_sol = solution;
      time += k;
    }
  }
}

template <typename fp_type>
void lss_one_dim_space_variable_heat_explicit_schemes::heat_euler_scheme<
    fp_type>::operator()(robin_boundary<fp_type> const &robin_boundary,
                         std::vector<fp_type> &solution) const {
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");
  LSS_ASSERT(
      solution.size() == initial_condition_.size(),
      "Entered solution vector size differs from initialCondition vector.");
  LSS_ASSERT(is_stable() == true, "This discretization is not stable.");
  // get delta time:
  fp_type const k = std::get<0>(deltas_);
  // get delta space:
  fp_type const h = std::get<1>(deltas_);
  // create first time point:
  fp_type time = k;
  // get coefficients:
  auto const &A = std::get<0>(coeffs_);
  auto const &B = std::get<1>(coeffs_);
  auto const &D = std::get<2>(coeffs_);
  // left space boundary:
  fp_type const left_lin = robin_boundary.left.first;
  fp_type const left_const = robin_boundary.left.second;
  // right space boundary:
  fp_type const right_lin_ = robin_boundary.right.first;
  fp_type const right_const_ = robin_boundary.right.second;
  // conversion of right hand boundaries:
  fp_type const right_lin = 1.0 / right_lin_;
  fp_type const right_const = -1.0 * (right_const_ / right_lin_);
  // previous solution:
  std::vector<fp_type> prev_sol = initial_condition_;
  // size of the space vector:
  std::size_t const space_size = solution.size();
  if (!is_source_set_) {
    // loop for stepping in time:
    while (time <= terminal_time_) {
      solution[0] = (D(0) + (A(0) * left_lin)) * prev_sol[1] +
                    (1.0 - 2.0 * B(0)) * prev_sol[0] + A(0) * left_const;
      solution[space_size - 1] =
          (A((space_size - 1) * h) + (D((space_size - 1) * h) * right_lin)) *
              prev_sol[space_size - 2] +
          (1.0 - 2.0 * B((space_size - 1) * h)) * prev_sol[space_size - 1] +
          D((space_size - 1) * h) * right_const;
      for (std::size_t t = 1; t < space_size - 1; ++t) {
        solution[t] = (1.0 - 2.0 * B(t * h)) * prev_sol[t] +
                      D(t * h) * prev_sol[t + 1] + A(t * h) * prev_sol[t - 1];
      }
      prev_sol = solution;
      time += k;
    }
  } else {
    // create a container to carry discretized source heat
    std::vector<fp_type> source_curr(solution.size(), fp_type{});
    discretize_in_space(h, space_start_, 0.0, source_, source_curr);
    // loop for stepping in time:
    while (time <= terminal_time_) {
      solution[0] = (D(0) + (A(0) * left_lin)) * prev_sol[1] +
                    (1.0 - 2.0 * B(0)) * prev_sol[0] + A(0) * left_const +
                    k * source_curr[0];
      solution[space_size - 1] =
          (A((space_size - 1) * h) + (D((space_size - 1) * h) * right_lin)) *
              prev_sol[space_size - 2] +
          (1.0 - 2.0 * B((space_size - 1) * h)) * prev_sol[space_size - 1] +
          D((space_size - 1) * h) * right_const +
          k * source_curr[space_size - 1];
      for (std::size_t t = 1; t < space_size - 1; ++t) {
        solution[t] = (1.0 - 2.0 * B(t * h)) * prev_sol[t] +
                      D(t * h) * prev_sol[t + 1] + A(t * h) * prev_sol[t - 1] +
                      k * source_curr[t];
      }
      discretize_in_space(h, space_start_, time, source_, source_curr);
      prev_sol = solution;
      time += k;
    }
  }
}

template <typename fp_type>
void lss_one_dim_space_variable_heat_explicit_schemes::
    ade_heat_bakarat_clark_scheme<fp_type>::operator()(
        dirichlet_boundary<fp_type> const &dirichlet_boundary,
        std::vector<fp_type> &solution) const {
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
  auto const &A = std::get<0>(coeffs_);
  auto const &B = std::get<1>(coeffs_);
  auto const &D = std::get<2>(coeffs_);
  // calculate scheme coefficients:
  auto const &a = [&](fp_type x) { return (A(x) / (1.0 + B(x))); };
  auto const &b = [&](fp_type x) { return ((1.0 - B(x)) / (1.0 + B(x))); };
  auto const &d = [&](fp_type x) { return (D(x) / (1.0 + B(x))); };
  auto const &f = [&](fp_type x) { return (k / (1.0 + B(x))); };
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
      up_component[t] =
          b(t * h) * up_component[t] + d(t * h) * up_component[t + 1] +
          a(t * h) * up_component[t - 1] + f(t * h) * rhs_coeff * rhs[t];
    }
  };
  // create downsweep anonymous function:
  auto down_sweep = [=](std::vector<fp_type> &down_component,
                        std::vector<fp_type> const &rhs, fp_type rhs_coeff) {
    for (std::size_t t = space_size - 2; t >= 1; --t) {
      down_component[t] =
          b(t * h) * down_component[t] + d(t * h) * down_component[t + 1] +
          a(t * h) * down_component[t - 1] + f(t * h) * rhs_coeff * rhs[t];
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
    discretize_in_space(h, space_start_, 0.0, source_, source_curr);
    discretize_in_space(h, space_start_, time, source_, source_next);
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
      discretize_in_space(h, space_start_, time, source_, source_curr);
      discretize_in_space(h, space_start_, 2.0 * time, source_, source_next);
      time += k;
    }
  }
}

template <typename fp_type>
void lss_one_dim_space_variable_heat_explicit_schemes::
    ade_heat_bakarat_clark_scheme<fp_type>::operator()(
        robin_boundary<fp_type> const &robin_boundary,
        std::vector<fp_type> &solution) const {
  throw new std::exception("Not available.");
}

template <typename fp_type>
void lss_one_dim_space_variable_heat_explicit_schemes::ade_heat_saulyev_scheme<
    fp_type>::operator()(dirichlet_boundary<fp_type> const &dirichlet_boundary,
                         std::vector<fp_type> &solution) const {
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
  auto const &A = std::get<0>(coeffs_);
  auto const &B = std::get<1>(coeffs_);
  auto const &D = std::get<2>(coeffs_);
  // calculate scheme coefficients:
  auto const &a = [&](fp_type x) { return (A(x) / (1.0 + B(x))); };
  auto const &b = [&](fp_type x) { return ((1.0 - B(x)) / (1.0 + B(x))); };
  auto const &d = [&](fp_type x) { return (D(x) / (1.0 + B(x))); };
  auto const &f = [&](fp_type x) { return (k / (1.0 + B(x))); };
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
      up_component[t] =
          b(t * h) * up_component[t] + d(t * h) * up_component[t + 1] +
          a(t * h) * up_component[t - 1] + f(t * h) * rhs_coeff * rhs[t];
    }
  };
  // create downsweep anonymous function:
  auto down_sweep = [=](std::vector<fp_type> &down_component,
                        std::vector<fp_type> const &rhs, fp_type rhs_coeff) {
    for (std::size_t t = space_size - 2; t >= 1; --t) {
      down_component[t] =
          b(t * h) * down_component[t] + d(t * h) * down_component[t + 1] +
          a(t * h) * down_component[t - 1] + f(t * h) * rhs_coeff * rhs[t];
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
    discretize_in_space(h, space_start_, 0.0, source_, source_curr);
    discretize_in_space(h, space_start_, time, source_, source_next);
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
      discretize_in_space(h, space_start_, time, source_, source_curr);
      discretize_in_space(h, space_start_, 2.0 * time, source_, source_next);
      time += k;
    }
  }
}

template <typename fp_type>
void lss_one_dim_space_variable_heat_explicit_schemes::ade_heat_saulyev_scheme<
    fp_type>::operator()(robin_boundary<fp_type> const &robin_boundary,
                         std::vector<fp_type> &solution) const {
  throw new std::exception("Not available.");
}

#endif  //_LSS_SPACE_VARIABLE_HEAT_EXPLICIT_SCHEMES
