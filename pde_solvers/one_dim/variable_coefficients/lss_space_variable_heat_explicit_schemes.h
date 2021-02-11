#pragma once
#if !defined(_LSS_SPACE_VARIABLE_HEAT_EXPLICIT_SCHEMES)
#define _LSS_SPACE_VARIABLE_HEAT_EXPLICIT_SCHEMES

#pragma warning(disable : 4244)

#include <thread>

#include "common/lss_enumerations.h"
#include "lss_space_variable_heat_explicit_schemes_policy.h"
#include "pde_solvers/one_dim/lss_base_explicit_schemes.h"
#include "pde_solvers/one_dim/lss_pde_utility.h"

namespace lss_one_dim_space_variable_heat_explicit_schemes {

using lss_enumerations::boundary_condition_enum;
using lss_one_dim_base_explicit_schemes::heat_scheme_base;
using lss_one_dim_pde_utility::dirichlet_boundary;
using lss_one_dim_pde_utility::pde_coefficient_holder_fun_1_arg;
using lss_one_dim_pde_utility::robin_boundary;
using lss_one_dim_pde_utility::v_discretization;
using lss_one_dim_space_variable_heat_explicit_schemes_policy::
    ade_heat_bakarat_clark_scheme_forward_policy;
using lss_one_dim_space_variable_heat_explicit_schemes_policy::
    ade_heat_saulyev_scheme_forward_policy;
using lss_one_dim_space_variable_heat_explicit_schemes_policy::
    heat_euler_scheme_forward_policy;

// ============================================================================
// ============================= heat_euler_scheme ============================
// ============================================================================

template <typename fp_type,
          typename scheme_policy = heat_euler_scheme_forward_policy<
              fp_type, pde_coefficient_holder_fun_1_arg<fp_type>, std::vector,
              std::allocator<fp_type>>>
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

template <typename fp_type,
          typename scheme_policy = ade_heat_bakarat_clark_scheme_forward_policy<
              fp_type, pde_coefficient_holder_fun_1_arg<fp_type>, std::vector,
              std::allocator<fp_type>>>
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

template <typename fp_type,
          typename scheme_policy = ade_heat_saulyev_scheme_forward_policy<
              fp_type, pde_coefficient_holder_fun_1_arg<fp_type>, std::vector,
              std::allocator<fp_type>>>
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

template <typename fp_type, typename scheme_policy>
bool lss_one_dim_space_variable_heat_explicit_schemes::heat_euler_scheme<
    fp_type, scheme_policy>::is_stable() const {
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

template <typename fp_type, typename scheme_policy>
void lss_one_dim_space_variable_heat_explicit_schemes::heat_euler_scheme<
    fp_type, scheme_policy>::operator()(dirichlet_boundary<fp_type> const
                                            &dirichlet_boundary,
                                        std::vector<fp_type> &solution) const {
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");
  LSS_ASSERT(
      solution.size() == initial_condition_.size(),
      "Entered solution vector size differs from initialCondition vector.");
  LSS_ASSERT(is_stable() == true, "This discretization is not stable.");

  if (!is_source_set_) {
    scheme_policy::traverse(solution, initial_condition_, dirichlet_boundary,
                            deltas_, coeffs_, terminal_time_);
  } else {
    scheme_policy::traverse(solution, initial_condition_, dirichlet_boundary,
                            deltas_, coeffs_, terminal_time_, space_start_,
                            source_);
  }
}

template <typename fp_type, typename scheme_policy>
void lss_one_dim_space_variable_heat_explicit_schemes::heat_euler_scheme<
    fp_type, scheme_policy>::operator()(robin_boundary<fp_type> const
                                            &robin_boundary,
                                        std::vector<fp_type> &solution) const {
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");
  LSS_ASSERT(
      solution.size() == initial_condition_.size(),
      "Entered solution vector size differs from initialCondition vector.");
  LSS_ASSERT(is_stable() == true, "This discretization is not stable.");

  if (!is_source_set_) {
    scheme_policy::traverse(solution, initial_condition_, robin_boundary,
                            deltas_, coeffs_, terminal_time_);
  } else {
    scheme_policy::traverse(solution, initial_condition_, robin_boundary,
                            deltas_, coeffs_, terminal_time_, space_start_,
                            source_);
  }
}

template <typename fp_type, typename scheme_policy>
void lss_one_dim_space_variable_heat_explicit_schemes::
    ade_heat_bakarat_clark_scheme<fp_type, scheme_policy>::operator()(
        dirichlet_boundary<fp_type> const &dirichlet_boundary,
        std::vector<fp_type> &solution) const {
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");
  LSS_ASSERT(
      solution.size() == initial_condition_.size(),
      "Entered solution vector size differs from initialCondition vector.");

  if (!is_source_set_) {
    scheme_policy::traverse(solution, initial_condition_, dirichlet_boundary,
                            deltas_, coeffs_, terminal_time_);
  } else {
    scheme_policy::traverse(solution, initial_condition_, dirichlet_boundary,
                            deltas_, coeffs_, terminal_time_, space_start_,
                            source_);
  }
}

template <typename fp_type, typename scheme_policy>
void lss_one_dim_space_variable_heat_explicit_schemes::
    ade_heat_bakarat_clark_scheme<fp_type, scheme_policy>::operator()(
        robin_boundary<fp_type> const &robin_boundary,
        std::vector<fp_type> &solution) const {
  throw new std::exception("Not available.");
}

template <typename fp_type, typename scheme_policy>
void lss_one_dim_space_variable_heat_explicit_schemes::ade_heat_saulyev_scheme<
    fp_type, scheme_policy>::operator()(dirichlet_boundary<fp_type> const
                                            &dirichlet_boundary,
                                        std::vector<fp_type> &solution) const {
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");
  LSS_ASSERT(
      solution.size() == initial_condition_.size(),
      "Entered solution vector size differs from initialCondition vector.");

  if (!is_source_set_) {
    scheme_policy::traverse(solution, initial_condition_, dirichlet_boundary,
                            deltas_, coeffs_, terminal_time_);
  } else {
    scheme_policy::traverse(solution, initial_condition_, dirichlet_boundary,
                            deltas_, coeffs_, terminal_time_, space_start_,
                            source_);
  }
}

template <typename fp_type, typename scheme_policy>
void lss_one_dim_space_variable_heat_explicit_schemes::ade_heat_saulyev_scheme<
    fp_type, scheme_policy>::operator()(robin_boundary<fp_type> const
                                            &robin_boundary,
                                        std::vector<fp_type> &solution) const {
  throw new std::exception("Not available.");
}

#endif  //_LSS_SPACE_VARIABLE_HEAT_EXPLICIT_SCHEMES
