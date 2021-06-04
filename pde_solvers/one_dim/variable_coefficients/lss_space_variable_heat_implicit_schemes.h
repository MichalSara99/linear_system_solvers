#pragma once
#if !defined(_LSS_1D_SPACE_VARIABLE_HEAT_IMPLICIT_SCHEMES)
#define _LSS_1D_SPACE_VARIABLE_HEAT_IMPLICIT_SCHEMES

#pragma warning(disable : 4244)

#include "common/lss_enumerations.h"
#include "pde_solvers/one_dim/lss_pde_utility.h"

namespace lss_one_dim_space_variable_heat_implicit_schemes {

using lss_enumerations::boundary_condition_enum;
using lss_enumerations::implicit_pde_schemes_enum;

// Alias for Scheme coefficients (A(x),B(x),D(x),h,k)
template <typename type>
using scheme_coefficient_holder =
    std::tuple<std::function<type(type)>, std::function<type(type)>,
               std::function<type(type)>, type, type>;

template <template <typename, typename> typename container, typename fp_type,
          typename alloc>
using scheme_function = std::function<void(
    scheme_coefficient_holder<fp_type> const &,
    container<fp_type, alloc> const &, container<fp_type, alloc> const &,
    container<fp_type, alloc> const &, container<fp_type, alloc> &)>;

// ============================================================================
// ========================= heat_equation_schemes ============================
// ============================================================================

template <template <typename, typename> typename container, typename fp_type,
          typename alloc>
class heat_equation_schemes {
  typedef container<fp_type, alloc> container_t;
  typedef scheme_function<container, fp_type, alloc> scheme_function_t;

 public:
  static fp_type const get_theta(implicit_pde_schemes_enum scheme) {
    fp_type theta{};
    if (scheme == implicit_pde_schemes_enum::Euler)
      theta = 1.0;
    else
      theta = 0.5;
    return theta;
  }

  static scheme_function_t const get_scheme(implicit_pde_schemes_enum scheme) {
    fp_type theta{};
    if (scheme == implicit_pde_schemes_enum::Euler)
      theta = 1.0;
    else
      theta = 0.5;
    auto scheme_fun =
        [=](scheme_coefficient_holder<fp_type> const &coeffs,
            container_t const &input, container_t const &inhom_input,
            container_t const &inhom_input_next, container_t &solution) {
          // inhom_input  = not used
          // inhom_input_next  = not used

          auto const &A = std::get<0>(coeffs);
          auto const &B = std::get<1>(coeffs);
          auto const &D = std::get<2>(coeffs);
          auto const h = std::get<3>(coeffs);

          for (std::size_t t = 1; t < solution.size() - 1; ++t) {
            solution[t] = (D(t * h) * (1.0 - theta) * input[t + 1]) +
                          ((1.0 - 2.0 * B(t * h) * (1.0 - theta)) * input[t]) +
                          (A(t * h) * (1.0 - theta) * input[t - 1]);
          }
        };
    return scheme_fun;
  }

  static scheme_function_t const get_inhom_scheme(
      implicit_pde_schemes_enum scheme) {
    fp_type theta{};
    if (scheme == implicit_pde_schemes_enum::Euler)
      theta = 1.0;
    else
      theta = 0.5;
    auto scheme_fun = [=](scheme_coefficient_holder<fp_type> const &coeffs,
                          container_t const &input,
                          container_t const &inhom_input,
                          container_t const &inhom_input_next,
                          container_t &solution) {
      auto const &A = std::get<0>(coeffs);
      auto const &B = std::get<1>(coeffs);
      auto const &D = std::get<2>(coeffs);
      auto const h = std::get<3>(coeffs);
      auto const k = std::get<4>(coeffs);

      for (std::size_t t = 1; t < solution.size() - 1; ++t) {
        solution[t] =
            (D(t * h) * (1.0 - theta) * input[t + 1]) +
            ((1.0 - 2.0 * B(t * h) * (1.0 - theta)) * input[t]) +
            (A(t * h) * (1.0 - theta) * input[t - 1]) +
            k * (theta * inhom_input_next[t] + (1.0 - theta) * inhom_input[t]);
      }
    };
    return scheme_fun;
  }
};

}  // namespace lss_one_dim_space_variable_heat_implicit_schemes

#endif  //_LSS_1D_SPACE_VARIABLE_HEAT_IMPLICIT_SCHEMES
