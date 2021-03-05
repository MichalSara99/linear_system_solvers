#pragma once
#if !defined(_LSS_1D_HEAT_IMPLICIT_SCHEMES_CUDA)
#define _LSS_1D_HEAT_IMPLICIT_SCHEMES_CUDA

#pragma warning(disable : 4244)

#include <thread>

#include "common/lss_enumerations.h"
#include "pde_solvers/one_dim/lss_pde_utility.h"

namespace lss_one_dim_heat_implicit_schemes_cuda {

using lss_enumerations::boundary_condition_enum;
using lss_enumerations::implicit_pde_schemes_enum;

// Alias for Scheme coefficients (A(x),B(x),D(x),h,k)
template <typename type>
using scheme_coefficient_holder = std::tuple<type, type, type, type>;

template <typename type>
using scheme_function = std::function<void(
    scheme_coefficient_holder<type> const &, std::vector<type> const &,
    std::vector<type> const &, std::vector<type> const &, std::vector<type> &,
    std::pair<type, type> const &, std::pair<type, type> const &)>;

// ============================================================================
// ============================ heat_equation_schemes  ========================
// ============================================================================

template <typename fp_type>
class heat_equation_schemes {
 public:
  static fp_type const get_theta(implicit_pde_schemes_enum scheme) {
    fp_type theta{};
    if (scheme == implicit_pde_schemes_enum::Euler)
      theta = 1.0;
    else
      theta = 0.5;
    return theta;
  }

  static scheme_function<fp_type> const get_scheme(
      boundary_condition_enum bc_type, implicit_pde_schemes_enum scheme) {
    fp_type theta{};
    if (scheme == implicit_pde_schemes_enum::Euler)
      theta = 1.0;
    else
      theta = 0.5;

    auto scheme_fun_dirichlet =
        [=](scheme_coefficient_holder<fp_type> const &coeffs,
            std::vector<fp_type> const &input,
            std::vector<fp_type> const &inhom_input,
            std::vector<fp_type> const &inhom_input_next,
            std::vector<fp_type> &solution,
            std::pair<fp_type, fp_type> const &boundary_pair_0,
            std::pair<fp_type, fp_type> const &boundary_pair_1) {
          // inhom_input			= not used here
          // inhom_input_next		= not uesd here
          // boundary_pair_1		= not used here

          fp_type const left = boundary_pair_0.first;
          fp_type const right = boundary_pair_0.second;
          std::size_t const last_idx = solution.size() - 1;

          fp_type const lambda = std::get<0>(coeffs);
          fp_type const gamma = std::get<1>(coeffs);
          fp_type const delta = std::get<2>(coeffs);

          solution[0] =
              ((lambda + gamma) * (1.0 - theta) * input[1]) +
              (1.0 - (2.0 * lambda - delta) * (1.0 - theta)) * input[0] +
              ((lambda - gamma) * left);

          for (std::size_t t = 1; t < last_idx; ++t) {
            solution[t] =
                ((lambda + gamma) * (1.0 - theta) * input[t + 1]) +
                (1.0 - (2.0 * lambda - delta) * (1.0 - theta)) * input[t] +
                ((lambda - gamma) * (1.0 - theta) * input[t - 1]);
          }

          solution[last_idx] =
              ((lambda + gamma) * right) +
              (1.0 - (2.0 * lambda - delta) * (1.0 - theta)) * input[last_idx] +
              ((lambda - gamma) * (1.0 - theta) * input[last_idx - 1]);
        };

    auto scheme_fun_robin =
        [=](scheme_coefficient_holder<fp_type> const &coeffs,
            std::vector<fp_type> const &input,
            std::vector<fp_type> const &inhom_input,
            std::vector<fp_type> const &inhom_input_next,
            std::vector<fp_type> &solution,
            std::pair<fp_type, fp_type> const &boundary_pair_0,
            std::pair<fp_type, fp_type> const &boundary_pair_1) {
          // inhom_input			= not used here
          // inhom_input_next		= not uesd here

          fp_type const left_linear = boundary_pair_0.first;
          fp_type const left_const = boundary_pair_0.second;
          fp_type const right_linear = boundary_pair_1.first;
          fp_type const right_const = boundary_pair_1.second;
          std::size_t const last_idx = solution.size() - 1;

          fp_type const lambda = std::get<0>(coeffs);
          fp_type const gamma = std::get<1>(coeffs);
          fp_type const delta = std::get<2>(coeffs);

          solution[0] =
              (((lambda + gamma) + (lambda - gamma) * left_linear) *
               (1.0 - theta) * input[1]) +
              (1.0 - (2.0 * lambda - delta) * (1.0 - theta)) * input[0] +
              ((lambda - gamma) * left_const);

          for (std::size_t t = 1; t < last_idx; ++t) {
            solution[t] =
                ((lambda + gamma) * (1.0 - theta) * input[t + 1]) +
                (1.0 - (2.0 * lambda - delta) * (1.0 - theta)) * input[t] +
                ((lambda - gamma) * (1.0 - theta) * input[t - 1]);
          }

          solution[last_idx] =
              (((lambda - gamma) + (lambda + gamma) * right_linear) *
               (1.0 - theta) * input[last_idx - 1]) +
              (1.0 - (2.0 * lambda - delta) * (1.0 - theta)) * input[last_idx] +
              ((lambda + gamma) * right_const);
        };

    if (bc_type == boundary_condition_enum::Dirichlet)
      return scheme_fun_dirichlet;
    else
      return scheme_fun_robin;
  }

  static scheme_function<fp_type> const get_inhom_scheme(
      boundary_condition_enum bc_type, implicit_pde_schemes_enum scheme) {
    fp_type theta{};
    if (scheme == implicit_pde_schemes_enum::Euler)
      theta = 1.0;
    else
      theta = 0.5;

    auto scheme_fun_dirichlet =
        [=](scheme_coefficient_holder<fp_type> const &coeffs,
            std::vector<fp_type> const &input,
            std::vector<fp_type> const &inhom_input,
            std::vector<fp_type> const &inhom_input_next,
            std::vector<fp_type> &solution,
            std::pair<fp_type, fp_type> const &boundary_pair_0,
            std::pair<fp_type, fp_type> const &boundary_pair_1) {
          // boundary_pair_1		= not used here

          fp_type const left = boundary_pair_0.first;
          fp_type const right = boundary_pair_0.second;
          std::size_t const last_idx = solution.size() - 1;

          fp_type const lambda = std::get<0>(coeffs);
          fp_type const gamma = std::get<1>(coeffs);
          fp_type const delta = std::get<2>(coeffs);
          fp_type const k = std::get<3>(coeffs);

          solution[0] =
              ((lambda + gamma) * (1.0 - theta) * input[1]) +
              (1.0 - (2.0 * lambda - delta) * (1.0 - theta)) * input[0] +
              ((lambda - gamma) * left) +
              k * (theta * inhom_input_next[0] +
                   (1.0 - theta) * inhom_input[0]);

          for (std::size_t t = 1; t < last_idx; ++t) {
            solution[t] =
                ((lambda + gamma) * (1.0 - theta) * input[t + 1]) +
                (1.0 - (2.0 * lambda - delta) * (1.0 - theta)) * input[t] +
                ((lambda - gamma) * (1.0 - theta) * input[t - 1]) +
                k * (theta * inhom_input_next[t] +
                     (1.0 - theta) * inhom_input[t]);
          }

          solution[last_idx] =
              ((lambda + gamma) * right) +
              (1.0 - (2.0 * lambda - delta) * (1.0 - theta)) * input[last_idx] +
              ((lambda - gamma) * (1.0 - theta) * input[last_idx - 1]) +
              k * (theta * inhom_input_next[last_idx] +
                   (1.0 - theta) * inhom_input[last_idx]);
        };

    auto scheme_fun_robin =
        [=](scheme_coefficient_holder<fp_type> const &coeffs,
            std::vector<fp_type> const &input,
            std::vector<fp_type> const &inhom_input,
            std::vector<fp_type> const &inhom_input_next,
            std::vector<fp_type> &solution,
            std::pair<fp_type, fp_type> const &boundary_pair_0,
            std::pair<fp_type, fp_type> const &boundary_pair_1) {
          fp_type const left_linear = boundary_pair_0.first;
          fp_type const left_const = boundary_pair_0.second;
          fp_type const right_linear = boundary_pair_1.first;
          fp_type const right_const = boundary_pair_1.second;
          std::size_t const last_idx = solution.size() - 1;

          fp_type const lambda = std::get<0>(coeffs);
          fp_type const gamma = std::get<1>(coeffs);
          fp_type const delta = std::get<2>(coeffs);
          fp_type const k = std::get<3>(coeffs);

          solution[0] =
              (((lambda + gamma) + (lambda - gamma) * left_linear) *
               (1.0 - theta) * input[1]) +
              (1.0 - (2.0 * lambda - delta) * (1.0 - theta)) * input[0] +
              ((lambda - gamma) * left_const) +
              k * (theta * inhom_input_next[0] +
                   (1.0 - theta) * inhom_input[0]);

          for (std::size_t t = 1; t < last_idx; ++t) {
            solution[t] =
                ((lambda + gamma) * (1.0 - theta) * input[t + 1]) +
                (1.0 - (2.0 * lambda - delta) * (1.0 - theta)) * input[t] +
                ((lambda - gamma) * (1.0 - theta) * input[t - 1]) +
                k * (theta * inhom_input_next[t] +
                     (1.0 - theta) * inhom_input[t]);
          }

          solution[last_idx] =
              (((lambda - gamma) + (lambda + gamma) * right_linear) *
               (1.0 - theta) * input[last_idx - 1]) +
              (1.0 - (2.0 * lambda - delta) * (1.0 - theta)) * input[last_idx] +
              ((lambda + gamma) * right_const) +
              k * (theta * inhom_input_next[last_idx] +
                   (1.0 - theta) * inhom_input[last_idx]);
        };

    if (bc_type == boundary_condition_enum::Dirichlet)
      return scheme_fun_dirichlet;
    else
      return scheme_fun_robin;
  }
};

}  // namespace lss_one_dim_heat_implicit_schemes_cuda

#endif  //_LSS_1D_HEAT_IMPLICIT_SCHEMES_CUDA
