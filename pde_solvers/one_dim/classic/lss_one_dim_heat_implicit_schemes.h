#pragma once
#if !defined(_LSS_ONE_DIM_HEAT_IMPLICIT_SCHEMES)
#define _LSS_ONE_DIM_HEAT_IMPLICIT_SCHEMES

#pragma warning(disable : 4244)

#include <thread>

#include "common/lss_types.h"
#include "pde_solvers/one_dim/lss_one_dim_pde_utility.h"

namespace lss_one_dim_heat_implicit_schemes {

using lss_one_dim_pde_utility::Discretization;
using lss_types::BoundaryConditionType;
using lss_types::ExplicitPDESchemes;
using lss_types::ImplicitPDESchemes;

// Alias for Scheme coefficients (A(x),B(x),D(x),h,k)
template <typename T>
using SchemeCoefficientHolder = std::tuple<T, T, T, T>;

template <typename T>
using SchemeFunction = std::function<void(
    SchemeCoefficientHolder<T> const&, std::vector<T> const&,
    std::vector<T> const&, std::vector<T> const&, std::vector<T>&)>;

// ============================================================================
// ================== ImplicitHeatEquationSchemes  ============================
// ============================================================================

template <typename T>
class ImplicitHeatEquationSchemes {
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
        [=](SchemeCoefficientHolder<T> const& coeffs,
            std::vector<T> const& input, std::vector<T> const& inhomInput,
            std::vector<T> const& inhomInputNext, std::vector<T>& solution) {
          // inhomInput not used
          // inhomInputNext not used

          T const lambda = std::get<0>(coeffs);
          T const gamma = std::get<1>(coeffs);
          T const delta = std::get<2>(coeffs);

          for (std::size_t t = 1; t < solution.size() - 1; ++t) {
            solution[t] =
                ((lambda + gamma) * (1.0 - theta) * input[t + 1]) +
                ((1.0 - (2.0 * lambda - delta) * (1.0 - theta)) * input[t]) +
                ((lambda - gamma) * (1.0 - theta) * input[t - 1]);
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
        [=](SchemeCoefficientHolder<T> const& coeffs,
            std::vector<T> const& input, std::vector<T> const& inhomInput,
            std::vector<T> const& inhomInputNext, std::vector<T>& solution) {
          T const lambda = std::get<0>(coeffs);
          T const gamma = std::get<1>(coeffs);
          T const delta = std::get<2>(coeffs);
          T const k = std::get<3>(coeffs);

          for (std::size_t t = 1; t < solution.size() - 1; ++t) {
            solution[t] =
                ((lambda + gamma) * (1.0 - theta) * input[t + 1]) +
                (1.0 - ((2.0 * lambda - delta) * (1.0 - theta))) * input[t] +
                ((lambda - gamma) * (1.0 - theta) * input[t - 1]) +
                k * (theta * inhomInputNext[t] + (1.0 - theta) * inhomInput[t]);
          }
        };
    return schemeFun;
  }
};

}  // namespace lss_one_dim_heat_implicit_schemes

#endif  //_LSS_ONE_DIM_HEAT_IMPLICIT_SCHEMES
