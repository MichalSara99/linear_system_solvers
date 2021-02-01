#pragma once
#if !defined(_LSS_ONE_DIM_SPACE_VARIABLE_HEAT_IMPLICIT_SCHEMES_CUDA)
#define _LSS_ONE_DIM_SPACE_VARIABLE_HEAT_IMPLICIT_SCHEMES_CUDA

#pragma warning(disable : 4244)

#include <thread>

#include "common/lss_enumerations.h"
#include "pde_solvers/one_dim/lss_one_dim_pde_utility.h"

namespace lss_one_dim_space_variable_heat_implicit_schemes_cuda {

using lss_enumerations::boundary_condition_enum;
using lss_enumerations::implicit_pde_schemes_enum;

// Alias for Scheme coefficients (A(x),B(x),D(x),h,k)
template <typename type>
using scheme_coefficient_holder =
    std::tuple<std::function<type(type)>, std::function<type(type)>,
               std::function<type(type)>, type, type>;

template <typename type>
using scheme_function = std::function<void(
    scheme_coefficient_holder<type> const &, std::vector<type> const &,
    std::vector<type> const &, std::vector<type> const &, std::vector<type> &,
    std::pair<type, type> const &, std::pair<type, type> const &)>;

// ============================================================================
// =========================== heat_equation_schemes  =========================
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

          auto const &A = std::get<0>(coeffs);
          auto const &B = std::get<1>(coeffs);
          auto const &D = std::get<2>(coeffs);
          auto const h = std::get<3>(coeffs);

          solution[0] = ((D(1 * h) * (1.0 - theta)) * input[1]) +
                        ((1.0 - 2.0 * B(1 * h) * (1.0 - theta)) * input[0]) +
                        (A(1 * h) * left);

          for (std::size_t t = 1; t < last_idx; ++t) {
            solution[t] =
                ((D((t + 1) * h) * (1.0 - theta)) * input[t + 1]) +
                ((1.0 - 2.0 * B((t + 1) * h) * (1.0 - theta)) * input[t]) +
                ((A((t + 1) * h) * (1.0 - theta)) * input[t - 1]);
          }

          solution[last_idx] =
              (D((last_idx + 1) * h) * right) +
              ((1.0 - 2.0 * B((last_idx + 1) * h) * (1.0 - theta)) *
               input[last_idx]) +
              ((A((last_idx + 1) * h) * (1.0 - theta)) * input[last_idx - 1]);
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

          auto const &A = std::get<0>(coeffs);
          auto const &B = std::get<1>(coeffs);
          auto const &D = std::get<2>(coeffs);
          auto const h = std::get<3>(coeffs);

          solution[0] = (((D(0 * h) + A(0 * h) * left_linear) * (1.0 - theta)) *
                         input[1]) +
                        (((1.0 - 2.0 * B(0 * h) * (1.0 - theta))) * input[0]) +
                        (A(0 * h) * left_const);

          for (std::size_t t = 1; t < last_idx; ++t) {
            solution[t] = ((D(t * h) * (1.0 - theta)) * input[t + 1]) +
                          ((1.0 - 2.0 * B(t * h) * (1.0 - theta)) * input[t]) +
                          ((A(t * h) * (1.0 - theta)) * input[t - 1]);
          }

          solution[last_idx] =
              (((A(last_idx * h) + D(last_idx * h) * right_linear) *
                (1.0 - theta)) *
               input[last_idx - 1]) +
              ((1.0 - 2.0 * B(last_idx * h) * (1.0 - theta)) *
               input[last_idx]) +
              (D(last_idx * h) * right_const);
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

          auto const &A = std::get<0>(coeffs);
          auto const &B = std::get<1>(coeffs);
          auto const &D = std::get<2>(coeffs);
          auto const h = std::get<3>(coeffs);
          auto const k = std::get<4>(coeffs);

          solution[0] = ((D(1 * h) * (1.0 - theta)) * input[1]) +
                        ((1.0 - 2.0 * B(1 * h) * (1.0 - theta)) * input[0]) +
                        (A(1 * h) * left) +
                        k * (theta * inhom_input_next[0] +
                             (1.0 - theta) * inhom_input[0]);

          for (std::size_t t = 1; t < last_idx; ++t) {
            solution[t] =
                ((D((t + 1) * h) * (1.0 - theta)) * input[t + 1]) +
                ((1.0 - 2.0 * B((t + 1) * h) * (1.0 - theta)) * input[t]) +
                ((A((t + 1) * h) * (1.0 - theta)) * input[t - 1]) +
                k * (theta * inhom_input_next[t] +
                     (1.0 - theta) * inhom_input[t]);
          }

          solution[last_idx] =
              (D((last_idx + 1) * h) * right) +
              ((1.0 - 2.0 * B((last_idx + 1) * h) * (1.0 - theta)) *
               input[last_idx]) +
              ((A((last_idx + 1) * h) * (1.0 - theta)) * input[last_idx - 1]) +
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

          auto const &A = std::get<0>(coeffs);
          auto const &B = std::get<1>(coeffs);
          auto const &D = std::get<2>(coeffs);
          auto const h = std::get<3>(coeffs);
          auto const k = std::get<4>(coeffs);

          solution[0] = (((D(0 * h) + A(0 * h) * left_linear) * (1.0 - theta)) *
                         input[1]) +
                        (((1.0 - 2.0 * B(0 * h) * (1.0 - theta))) * input[0]) +
                        (A(0 * h) * left_const) +
                        k * (theta * inhom_input_next[0] +
                             (1.0 - theta) * inhom_input[0]);

          for (std::size_t t = 1; t < last_idx; ++t) {
            solution[t] = ((D(t * h) * (1.0 - theta)) * input[t + 1]) +
                          ((1.0 - 2.0 * B(t * h) * (1.0 - theta)) * input[t]) +
                          ((A(t * h) * (1.0 - theta)) * input[t - 1]) +
                          k * (theta * inhom_input_next[t] +
                               (1.0 - theta) * inhom_input[t]);
          }

          solution[last_idx] =
              (((A(last_idx * h) + D(last_idx * h) * right_linear) *
                (1.0 - theta)) *
               input[last_idx - 1]) +
              ((1.0 - 2.0 * B(last_idx * h) * (1.0 - theta)) *
               input[last_idx]) +
              (D(last_idx * h) * right_const) +
              k * (theta * inhom_input_next[last_idx] +
                   (1.0 - theta) * inhom_input[last_idx]);
        };

    if (bc_type == boundary_condition_enum::Dirichlet)
      return scheme_fun_dirichlet;
    else
      return scheme_fun_robin;
  }
};

}  // namespace lss_one_dim_space_variable_heat_implicit_schemes_cuda

#endif  //_LSS_ONE_DIM_SPACE_VARIABLE_HEAT_IMPLICIT_SCHEMES_CUDA
