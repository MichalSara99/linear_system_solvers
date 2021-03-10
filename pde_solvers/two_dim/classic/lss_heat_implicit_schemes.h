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

template <typename type>
using scheme_function = std::function<void(
    scheme_coefficient_holder<type> const&, std::vector<type> const&,
    std::vector<type> const&, std::vector<type> const&, std::vector<type>&,
    std::size_t const&)>;

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

  static scheme_function<fp_type> const get_scheme(
      implicit_pde_schemes_enum scheme) {
    fp_type theta{};
    if (scheme == implicit_pde_schemes_enum::Euler)
      theta = 1.0;
    else
      theta = 0.5;
    auto scheme_fun =
        [=](scheme_coefficient_holder<fp_type> const& coeffs,
            container_2d<container, fp_type, alloc> const& input,
            container_2d<container, fp_type, alloc> const& inhom_input,
            container_2d<container, fp_type, alloc> const& inhom_input_next,
            container<fp_type, alloc>& solution,
            std::size_t const& solution_idx) {
          // inhom_input not used
          // inhom_input_next not used

          fp_type const lambda = std::get<0>(coeffs);
          fp_type const gamma = std::get<1>(coeffs);
          fp_type const delta = std::get<2>(coeffs);

          for (std::size_t t = 1; t < solution.size() - 1; ++t) {
            solution[t] =
                ((lambda + gamma) * (1.0 - theta) * input[t + 1]) +
                ((1.0 - (2.0 * lambda - delta) * (1.0 - theta)) * input[t]) +
                ((lambda - gamma) * (1.0 - theta) * input[t - 1]);
          }
        };
    return scheme_fun;
  }

  static scheme_function<fp_type> const get_inhom_scheme(
      implicit_pde_schemes_enum scheme) {
    fp_type theta{};
    if (scheme == implicit_pde_schemes_enum::Euler)
      theta = 1.0;
    else
      theta = 0.5;
    auto scheme_fun = [=](scheme_coefficient_holder<fp_type> const& coeffs,
                          std::vector<fp_type> const& input,
                          std::vector<fp_type> const& inhom_input,
                          std::vector<fp_type> const& inhom_input_next,
                          std::vector<fp_type>& solution) {
      fp_type const lambda = std::get<0>(coeffs);
      fp_type const gamma = std::get<1>(coeffs);
      fp_type const delta = std::get<2>(coeffs);
      fp_type const k = std::get<3>(coeffs);

      for (std::size_t t = 1; t < solution.size() - 1; ++t) {
        solution[t] =
            ((lambda + gamma) * (1.0 - theta) * input[t + 1]) +
            (1.0 - ((2.0 * lambda - delta) * (1.0 - theta))) * input[t] +
            ((lambda - gamma) * (1.0 - theta) * input[t - 1]) +
            k * (theta * inhom_input_next[t] + (1.0 - theta) * inhom_input[t]);
      }
    };
    return scheme_fun;
  }
};

}  // namespace lss_two_dim_heat_implicit_schemes

#endif  //_LSS_2D_HEAT_IMPLICIT_SCHEMES
