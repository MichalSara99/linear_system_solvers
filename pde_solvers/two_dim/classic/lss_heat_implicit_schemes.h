#pragma once
#if !defined(_LSS_2D_HEAT_IMPLICIT_SCHEMES)
#define _LSS_2D_HEAT_IMPLICIT_SCHEMES

#pragma warning(disable : 4244)

#include <thread>

#include "common/lss_enumerations.h"
#include "common/lss_utility.h"
#include "pde_solvers/two_dim/lss_pde_utility.h"

namespace lss_two_dim_heat_implicit_schemes {

using lss_enumerations::boundary_condition_enum;
using lss_enumerations::implicit_pde_schemes_enum;
using lss_utility::container_2d;

// Alias for Scheme coefficients (A(x),B(x),D(x),h,k)
template <typename type>
using scheme_coefficient_holder =
    std::tuple<type, type, type, type, type, type, type>;

template <typename fp_type, template <typename, typename> typename container,
          typename alloc>
using scheme_function =
    std::function<void(scheme_coefficient_holder<fp_type> const&,
                       container_2d<container, fp_type, alloc> const&,
                       container_2d<container, fp_type, alloc> const&,
                       container_2d<container, fp_type, alloc> const&,
                       container<fp_type, alloc>&, std::size_t const&)>;

// ============================================================================
// ======================== heat_equation_schemes  ============================
// ============================================================================

template <typename fp_type, template <typename, typename> typename container,
          typename alloc>
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

  static std::pair<scheme_function<fp_type, container, alloc>,
                   scheme_function<fp_type, container, alloc>> const
  get_scheme(implicit_pde_schemes_enum scheme) {
    fp_type theta{};
    if (scheme == implicit_pde_schemes_enum::Euler)
      theta = 1.0;
    else
      theta = 0.5;
    auto scheme_fun_0 =
        [=](scheme_coefficient_holder<fp_type> const& coeffs,
            container_2d<container, fp_type, alloc> const& input,
            container_2d<container, fp_type, alloc> const& inhom_input,
            container_2d<container, fp_type, alloc> const& inhom_input_next,
            container<fp_type, alloc>& solution,
            std::size_t const& solution_idx) {
          // inhom_input not used
          // inhom_input_next not used

          fp_type const alpha = std::get<0>(coeffs);
          fp_type const beta = std::get<1>(coeffs);
          fp_type const gamma = std::get<2>(coeffs);
          fp_type const delta = std::get<3>(coeffs);
          fp_type const ni = std::get<4>(coeffs);
          fp_type const rho = std::get<5>(coeffs);

          for (std::size_t t = 1; t < solution.size() - 1; ++t) {
            solution[t] =
                gamma * input(t - 1, solution_idx - 1) +
                (1.0 - theta) * (alpha - delta) * input(t - 1, solution_idx) -
                gamma * input(t - 1, solution_idx + 1) +
                (beta - ni) * input(t, solution_idx - 1) +
                (1.0 - (2.0 * beta - 0.5 * rho) -
                 (1.0 - theta) * (2.0 * alpha - 0.5 * rho)) *
                    input(t, solution_idx) +
                (beta + ni) * input(t, solution_idx + 1) -
                gamma * input(t + 1, solution_idx - 1) +
                (1.0 - theta) * (alpha + delta) * input(t + 1, solution_idx) +
                gamma * input(t + 1, solution_idx + 1);
          }
        };

    auto scheme_fun_1 =
        [=](scheme_coefficient_holder<fp_type> const& coeffs,
            container_2d<container, fp_type, alloc> const& input,
            container_2d<container, fp_type, alloc> const& intermed_sol,
            container_2d<container, fp_type, alloc> const& not_used,
            container<fp_type, alloc>& solution,
            std::size_t const& solution_idx) {
          // inhom_input_next not used

          fp_type const alpha = std::get<0>(coeffs);
          fp_type const beta = std::get<1>(coeffs);
          fp_type const gamma = std::get<2>(coeffs);
          fp_type const delta = std::get<3>(coeffs);
          fp_type const ni = std::get<4>(coeffs);
          fp_type const rho = std::get<5>(coeffs);

          for (std::size_t t = 1; t < solution.size() - 1; ++t) {
            solution[t] =
                -1.0 * (beta - ni) * theta * input(solution_idx, t - 1) +
                theta * (2.0 * beta - 0.5 * rho) * input(solution_idx, t) -
                (beta + ni) * theta * input(solution_idx, t + 1) +
                intermed_sol(t, solution_idx);
          }
        };
    return std::make_pair(scheme_fun_0, scheme_fun_1);
  }

  static std::pair<scheme_function<fp_type, container, alloc>,
                   scheme_function<fp_type, container, alloc>> const
  get_inhom_scheme(implicit_pde_schemes_enum scheme) {
    fp_type theta{};
    if (scheme == implicit_pde_schemes_enum::Euler)
      theta = 1.0;
    else
      theta = 0.5;
    auto scheme_fun_0 =
        [=](scheme_coefficient_holder<fp_type> const& coeffs,
            container_2d<container, fp_type, alloc> const& input,
            container_2d<container, fp_type, alloc> const& inhom_input,
            container_2d<container, fp_type, alloc> const& inhom_input_next,
            container<fp_type, alloc>& solution,
            std::size_t const& solution_idx) {
          fp_type const alpha = std::get<0>(coeffs);
          fp_type const beta = std::get<1>(coeffs);
          fp_type const gamma = std::get<2>(coeffs);
          fp_type const delta = std::get<3>(coeffs);
          fp_type const ni = std::get<4>(coeffs);
          fp_type const rho = std::get<5>(coeffs);
          fp_type const k = std::get<6>(coeffs);

          for (std::size_t t = 1; t < solution.size() - 1; ++t) {
            solution[t] =
                gamma * input(t - 1, solution_idx - 1) +
                (1.0 - theta) * (alpha - delta) * input(t - 1, solution_idx) -
                gamma * input(t - 1, solution_idx + 1) +
                (beta - ni) * input(t, solution_idx - 1) +
                (1.0 - (2.0 * beta - 0.5 * rho) -
                 (1.0 - theta) * (2.0 * alpha - 0.5 * rho)) *
                    input(t, solution_idx) +
                (beta + ni) * input(t, solution_idx + 1) -
                gamma * input(t + 1, solution_idx - 1) +
                (1.0 - theta) * (alpha + delta) * input(t + 1, solution_idx) +
                gamma * input(t + 1, solution_idx + 1) +
                (1.0 - theta) * k * inhom_input(t, solution_idx) +
                theta * k * inhom_input_next(t, solution_idx);
          }
        };

    auto scheme_fun_1 =
        [=](scheme_coefficient_holder<fp_type> const& coeffs,
            container_2d<container, fp_type, alloc> const& input,
            container_2d<container, fp_type, alloc> const& intermed_sol,
            container_2d<container, fp_type, alloc> const& not_used,
            container<fp_type, alloc>& solution,
            std::size_t const& solution_idx) {
          fp_type const alpha = std::get<0>(coeffs);
          fp_type const beta = std::get<1>(coeffs);
          fp_type const gamma = std::get<2>(coeffs);
          fp_type const delta = std::get<3>(coeffs);
          fp_type const ni = std::get<4>(coeffs);
          fp_type const rho = std::get<5>(coeffs);
          fp_type const k = std::get<6>(coeffs);

          for (std::size_t t = 1; t < solution.size() - 1; ++t) {
            solution[t] =
                -1.0 * (beta - ni) * theta * input(solution_idx, t - 1) +
                theta * (2.0 * beta - 0.5 * rho) * input(solution_idx, t) -
                (beta + ni) * theta * input(solution_idx, t + 1) +
                intermed_sol(t, solution_idx);
          }
        };
    return std::make_pair(scheme_fun_0, scheme_fun_1);
  }
};

}  // namespace lss_two_dim_heat_implicit_schemes

#endif  //_LSS_2D_HEAT_IMPLICIT_SCHEMES
