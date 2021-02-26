#pragma once
#if !defined(_LSS_SOR_SOLVER_TRAITS)
#define _LSS_SOR_SOLVER_TRAITS

#include <typeinfo>

#include "common/lss_enumerations.h"
#include "common/lss_utility.h"

namespace lss_sor_solver_traits {

template <typename fp_type>
struct sor_solver_traits {};

template <>
struct sor_solver_traits<double> {
  static double tolerance() { return 1.e-18; }
  static std::size_t iteration_limit() { return 10'000; }
};

template <>
struct sor_solver_traits<float> {
  static float tolerance() { return 1.e-11f; }
  static std::size_t iteration_limit() { return 10'000; }
};

template <typename fp_type>
struct sor_solver_cuda_traits {};

template <>
struct sor_solver_cuda_traits<double> {
  static double tolerance() { return 1.e-18; }
  static std::size_t iteration_limit() { return 100'000; }
};

template <>
struct sor_solver_cuda_traits<float> {
  static float tolerance() { return 1.e-11f; }
  static std::size_t iteration_limit() { return 100'000; }
};

}  // namespace lss_sor_solver_traits

#endif  ///_LSS_SOR_SOLVER_TRAITS
