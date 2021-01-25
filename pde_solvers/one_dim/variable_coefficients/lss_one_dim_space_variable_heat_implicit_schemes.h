#pragma once
#if !defined(_LSS_ONE_DIM_SPACE_VARIABLE_HEAT_IMPLICIT_SCHEMES)
#define _LSS_ONE_DIM_SPACE_VARIABLE_HEAT_IMPLICIT_SCHEMES

#pragma warning(disable : 4244)

#include <thread>

#include "common/lss_enumerations.h"
#include "pde_solvers/one_dim/lss_one_dim_pde_utility.h"

namespace lss_one_dim_space_variable_heat_implicit_schemes {

using lss_enumerations::BoundaryConditionType;
using lss_enumerations::ExplicitPDESchemes;
using lss_enumerations::ImplicitPDESchemes;
using lss_one_dim_pde_utility::Discretization;

// Alias for Scheme coefficients (A(x),B(x),D(x),h,k)
template <typename T>
using SchemeCoefficientHolder =
    std::tuple<std::function<T(T)>, std::function<T(T)>, std::function<T(T)>, T,
               T>;

template <typename T>
using SchemeFunction = std::function<void(
    SchemeCoefficientHolder<T> const &, std::vector<T> const &,
    std::vector<T> const &, std::vector<T> const &, std::vector<T> &)>;

// ============================================================================
// ================ ImplicitSpaceVariableHeatEquationSchemes ==================
// ============================================================================

template <typename T>
class ImplicitSpaceVariableHeatEquationSchemes {
 public:
  static T const getTheta(ImplicitPDESchemes scheme) {
    double theta{};
    if (scheme == ImplicitPDESchemes::Euler)
      theta = 1.0;
    else
      theta = 0.5;
    return theta;
  }

  static SchemeFunction<T> const getScheme(ImplicitPDESchemes scheme) {
    double theta{};
    if (scheme == ImplicitPDESchemes::Euler)
      theta = 1.0;
    else
      theta = 0.5;
    auto schemeFun =
        [=](SchemeCoefficientHolder<T> const &coeffs,
            std::vector<T> const &input, std::vector<T> const &inhomInput,
            std::vector<T> const &inhomInputNext, std::vector<T> &solution) {
          // inhomInput  = not used
          // inhomInputNext  = not used

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
    return schemeFun;
  }

  static SchemeFunction<T> const getInhomScheme(ImplicitPDESchemes scheme) {
    double theta{};
    if (scheme == ImplicitPDESchemes::Euler)
      theta = 1.0;
    else
      theta = 0.5;
    auto schemeFun =
        [=](SchemeCoefficientHolder<T> const &coeffs,
            std::vector<T> const &input, std::vector<T> const &inhomInput,
            std::vector<T> const &inhomInputNext, std::vector<T> &solution) {
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
                k * (theta * inhomInputNext[t] + (1.0 - theta) * inhomInput[t]);
          }
        };
    return schemeFun;
  }
};

}  // namespace lss_one_dim_space_variable_heat_implicit_schemes

#endif  //_LSS_ONE_DIM_SPACE_VARIABLE_HEAT_IMPLICIT_SCHEMES
