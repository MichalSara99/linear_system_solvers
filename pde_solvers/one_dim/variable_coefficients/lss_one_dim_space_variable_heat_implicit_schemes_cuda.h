#pragma once
#if !defined(_LSS_ONE_DIM_SPACE_VARIABLE_HEAT_IMPLICIT_SCHEMES_CUDA)
#define _LSS_ONE_DIM_SPACE_VARIABLE_HEAT_IMPLICIT_SCHEMES_CUDA

#pragma warning(disable : 4244)

#include <thread>

#include "common/lss_enumerations.h"
#include "pde_solvers/one_dim/lss_one_dim_pde_utility.h"

namespace lss_one_dim_space_variable_heat_implicit_schemes_cuda {

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
using SchemeFunctionCUDA = std::function<void(
    SchemeCoefficientHolder<T> const &, std::vector<T> const &,
    std::vector<T> const &, std::vector<T> const &, std::vector<T> &,
    std::pair<T, T> const &, std::pair<T, T> const &)>;

// ============================================================================
// =============== ImplicitSpaceVariableHeatEquationSchemesCUDA  ==============
// ============================================================================

template <typename T>
class ImplicitSpaceVariableHeatEquationSchemesCUDA {
 public:
  static T const getTheta(ImplicitPDESchemes scheme) {
    double theta{};
    if (scheme == ImplicitPDESchemes::Euler)
      theta = 1.0;
    else
      theta = 0.5;
    return theta;
  }

  static SchemeFunctionCUDA<T> const getScheme(BoundaryConditionType bcType,
                                               ImplicitPDESchemes scheme) {
    double theta{};
    if (scheme == ImplicitPDESchemes::Euler)
      theta = 1.0;
    else
      theta = 0.5;

    auto schemeFunDirichlet =
        [=](SchemeCoefficientHolder<T> const &coeffs,
            std::vector<T> const &input, std::vector<T> const &inhomInput,
            std::vector<T> const &inhomInputNext, std::vector<T> &solution,
            std::pair<T, T> const &boundaryPair0,
            std::pair<T, T> const &boundaryPair1) {
          // inhomInput			= not used here
          // inhomInputNext		= not uesd here
          // boundaryPair1		= not used here

          T const left = boundaryPair0.first;
          T const right = boundaryPair0.second;
          std::size_t const lastIdx = solution.size() - 1;

          auto const &A = std::get<0>(coeffs);
          auto const &B = std::get<1>(coeffs);
          auto const &D = std::get<2>(coeffs);
          auto const h = std::get<3>(coeffs);

          solution[0] = ((D(1 * h) * (1.0 - theta)) * input[1]) +
                        ((1.0 - 2.0 * B(1 * h) * (1.0 - theta)) * input[0]) +
                        (A(1 * h) * left);

          for (std::size_t t = 1; t < lastIdx; ++t) {
            solution[t] =
                ((D((t + 1) * h) * (1.0 - theta)) * input[t + 1]) +
                ((1.0 - 2.0 * B((t + 1) * h) * (1.0 - theta)) * input[t]) +
                ((A((t + 1) * h) * (1.0 - theta)) * input[t - 1]);
          }

          solution[lastIdx] =
              (D((lastIdx + 1) * h) * right) +
              ((1.0 - 2.0 * B((lastIdx + 1) * h) * (1.0 - theta)) *
               input[lastIdx]) +
              ((A((lastIdx + 1) * h) * (1.0 - theta)) * input[lastIdx - 1]);
        };

    auto schemeFunRobin = [=](SchemeCoefficientHolder<T> const &coeffs,
                              std::vector<T> const &input,
                              std::vector<T> const &inhomInput,
                              std::vector<T> const &inhomInputNext,
                              std::vector<T> &solution,
                              std::pair<T, T> const &boundaryPair0,
                              std::pair<T, T> const &boundaryPair1) {
      // inhomInput			= not used here
      // inhomInputNext		= not uesd here

      T const leftLinear = boundaryPair0.first;
      T const leftConst = boundaryPair0.second;
      T const rightLinear = boundaryPair1.first;
      T const rightConst = boundaryPair1.second;
      std::size_t const lastIdx = solution.size() - 1;

      auto const &A = std::get<0>(coeffs);
      auto const &B = std::get<1>(coeffs);
      auto const &D = std::get<2>(coeffs);
      auto const h = std::get<3>(coeffs);

      solution[0] =
          (((D(0 * h) + A(0 * h) * leftLinear) * (1.0 - theta)) * input[1]) +
          (((1.0 - 2.0 * B(0 * h) * (1.0 - theta))) * input[0]) +
          (A(0 * h) * leftConst);

      for (std::size_t t = 1; t < lastIdx; ++t) {
        solution[t] = ((D(t * h) * (1.0 - theta)) * input[t + 1]) +
                      ((1.0 - 2.0 * B(t * h) * (1.0 - theta)) * input[t]) +
                      ((A(t * h) * (1.0 - theta)) * input[t - 1]);
      }

      solution[lastIdx] =
          (((A(lastIdx * h) + D(lastIdx * h) * rightLinear) * (1.0 - theta)) *
           input[lastIdx - 1]) +
          ((1.0 - 2.0 * B(lastIdx * h) * (1.0 - theta)) * input[lastIdx]) +
          (D(lastIdx * h) * rightConst);
    };

    if (bcType == BoundaryConditionType::Dirichlet)
      return schemeFunDirichlet;
    else
      return schemeFunRobin;
  }

  static SchemeFunctionCUDA<T> const getInhomScheme(
      BoundaryConditionType bcType, ImplicitPDESchemes scheme) {
    double theta{};
    if (scheme == ImplicitPDESchemes::Euler)
      theta = 1.0;
    else
      theta = 0.5;

    auto schemeFunDirichlet =
        [=](SchemeCoefficientHolder<T> const &coeffs,
            std::vector<T> const &input, std::vector<T> const &inhomInput,
            std::vector<T> const &inhomInputNext, std::vector<T> &solution,
            std::pair<T, T> const &boundaryPair0,
            std::pair<T, T> const &boundaryPair1) {
          // boundaryPair1		= not used here

          T const left = boundaryPair0.first;
          T const right = boundaryPair0.second;
          std::size_t const lastIdx = solution.size() - 1;

          auto const &A = std::get<0>(coeffs);
          auto const &B = std::get<1>(coeffs);
          auto const &D = std::get<2>(coeffs);
          auto const h = std::get<3>(coeffs);
          auto const k = std::get<4>(coeffs);

          solution[0] =
              ((D(1 * h) * (1.0 - theta)) * input[1]) +
              ((1.0 - 2.0 * B(1 * h) * (1.0 - theta)) * input[0]) +
              (A(1 * h) * left) +
              k * (theta * inhomInputNext[0] + (1.0 - theta) * inhomInput[0]);

          for (std::size_t t = 1; t < lastIdx; ++t) {
            solution[t] =
                ((D((t + 1) * h) * (1.0 - theta)) * input[t + 1]) +
                ((1.0 - 2.0 * B((t + 1) * h) * (1.0 - theta)) * input[t]) +
                ((A((t + 1) * h) * (1.0 - theta)) * input[t - 1]) +
                k * (theta * inhomInputNext[t] + (1.0 - theta) * inhomInput[t]);
          }

          solution[lastIdx] =
              (D((lastIdx + 1) * h) * right) +
              ((1.0 - 2.0 * B((lastIdx + 1) * h) * (1.0 - theta)) *
               input[lastIdx]) +
              ((A((lastIdx + 1) * h) * (1.0 - theta)) * input[lastIdx - 1]) +
              k * (theta * inhomInputNext[lastIdx] +
                   (1.0 - theta) * inhomInput[lastIdx]);
        };

    auto schemeFunRobin = [=](SchemeCoefficientHolder<T> const &coeffs,
                              std::vector<T> const &input,
                              std::vector<T> const &inhomInput,
                              std::vector<T> const &inhomInputNext,
                              std::vector<T> &solution,
                              std::pair<T, T> const &boundaryPair0,
                              std::pair<T, T> const &boundaryPair1) {
      T const leftLinear = boundaryPair0.first;
      T const leftConst = boundaryPair0.second;
      T const rightLinear = boundaryPair1.first;
      T const rightConst = boundaryPair1.second;
      std::size_t const lastIdx = solution.size() - 1;

      auto const &A = std::get<0>(coeffs);
      auto const &B = std::get<1>(coeffs);
      auto const &D = std::get<2>(coeffs);
      auto const h = std::get<3>(coeffs);
      auto const k = std::get<4>(coeffs);

      solution[0] =
          (((D(0 * h) + A(0 * h) * leftLinear) * (1.0 - theta)) * input[1]) +
          (((1.0 - 2.0 * B(0 * h) * (1.0 - theta))) * input[0]) +
          (A(0 * h) * leftConst) +
          k * (theta * inhomInputNext[0] + (1.0 - theta) * inhomInput[0]);

      for (std::size_t t = 1; t < lastIdx; ++t) {
        solution[t] =
            ((D(t * h) * (1.0 - theta)) * input[t + 1]) +
            ((1.0 - 2.0 * B(t * h) * (1.0 - theta)) * input[t]) +
            ((A(t * h) * (1.0 - theta)) * input[t - 1]) +
            k * (theta * inhomInputNext[t] + (1.0 - theta) * inhomInput[t]);
      }

      solution[lastIdx] =
          (((A(lastIdx * h) + D(lastIdx * h) * rightLinear) * (1.0 - theta)) *
           input[lastIdx - 1]) +
          ((1.0 - 2.0 * B(lastIdx * h) * (1.0 - theta)) * input[lastIdx]) +
          (D(lastIdx * h) * rightConst) +
          k * (theta * inhomInputNext[lastIdx] +
               (1.0 - theta) * inhomInput[lastIdx]);
    };

    if (bcType == BoundaryConditionType::Dirichlet)
      return schemeFunDirichlet;
    else
      return schemeFunRobin;
  }
};

}  // namespace lss_one_dim_space_variable_heat_implicit_schemes_cuda

#endif  //_LSS_ONE_DIM_SPACE_VARIABLE_HEAT_IMPLICIT_SCHEMES_CUDA
