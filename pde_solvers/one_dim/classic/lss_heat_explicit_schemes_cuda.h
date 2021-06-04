#pragma once
#if !defined(_LSS_1D_HEAT_EXPLICIT_SCHEMES_CUDA)
#define _LSS_1D_HEAT_EXPLICIT_SCHEMES_CUDA

#include <tuple>

#include "common/lss_macros.h"
#include "common/lss_utility.h"
#include "pde_solvers/one_dim/lss_pde_boundary.h"
#include "pde_solvers/one_dim/lss_pde_utility.h"

namespace lss_one_dim_heat_explicit_schemes_cuda {

using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
using lss_one_dim_pde_utility::robin_boundary;
using lss_utility::sptr_t;

class euler_loop_sp {
 private:
  float space_start_;
  float terminal_t_;
  std::pair<float, float> deltas_;  // first = delta time, second = delta space;
  std::tuple<float, float, float> coeffs_;  // coefficients of PDE
  std::function<float(float, float)> source_;
  bool is_source_set_;

 public:
  ~euler_loop_sp() {}
  explicit euler_loop_sp() = delete;
  explicit euler_loop_sp(float space_start, float terminal_time,
                         std::pair<float, float> const &deltas,
                         std::tuple<float, float, float> const &coeffs,
                         std::function<float(float, float)> const &source,
                         bool is_source_set = false)
      : space_start_{space_start},
        terminal_t_{terminal_time},
        deltas_{deltas},
        coeffs_{coeffs},
        source_{source},
        is_source_set_{is_source_set} {}

  void operator()(
      float const *input,
      sptr_t<dirichlet_boundary_1d<float>> const &dirichlet_boundary,
      unsigned long long const size, float *solution) const;
  void operator()(float const *input,
                  robin_boundary<float> const &robin_boundary,
                  unsigned long long const size, float *solution) const;
};

class euler_loop_dp {
 private:
  double space_start_;
  double terminal_t_;
  std::pair<double, double>
      deltas_;  // first = delta time, second = delta space;
  std::tuple<double, double, double> coeffs_;  // coefficients of PDE
  std::function<double(double, double)> source_;
  bool is_source_set_;

 public:
  ~euler_loop_dp() {}
  explicit euler_loop_dp() = delete;
  explicit euler_loop_dp(double space_start, double terminal_time,
                         std::pair<double, double> const &deltas,
                         std::tuple<double, double, double> const &coeffs,
                         std::function<double(double, double)> const &source,
                         bool is_source_set = false)
      : space_start_{space_start},
        terminal_t_{terminal_time},
        deltas_{deltas},
        coeffs_{coeffs},
        source_{source},
        is_source_set_{is_source_set} {}

  void operator()(
      double const *input,
      sptr_t<dirichlet_boundary_1d<double>> const &dirichlet_boundary,
      unsigned long long const size, double *solution) const;
  void operator()(double const *input,
                  robin_boundary<double> const &robin_boundary,
                  unsigned long long const size, double *solution) const;
};

// ============================================================================
// ================= euler_heat_equation_scheme general template ==============
// ============================================================================

template <typename fp_type, template <typename, typename> typename container,
          typename alloc>
class euler_heat_equation_scheme {};

// ============================================================================
// ========= Single-Precision Floating-Point euler_heat_equation_scheme =======
// ============================================================================

template <template <typename, typename> typename container, typename alloc>
class euler_heat_equation_scheme<float, container, alloc> {
 private:
  float space_start_;
  float terminal_t_;
  std::pair<float, float> deltas_;  // first = delta time, second = delta space;
  std::tuple<float, float, float> coeffs_;  // coefficients of PDE
  container<float, alloc> init_;
  std::function<float(float, float)> source_;
  bool is_source_set_;

 public:
  typedef float value_type;
  explicit euler_heat_equation_scheme() = delete;
  explicit euler_heat_equation_scheme(
      float space_start, float terminal_time,
      std::pair<float, float> const &deltas,
      std::tuple<float, float, float> const &coeffs,
      container<float, alloc> const &init,
      std::function<float(float, float)> const &source = nullptr,
      bool is_source_set = false)
      : space_start_{space_start},
        terminal_t_{terminal_time},
        deltas_{deltas},
        coeffs_{coeffs},
        init_{init},
        source_{source},
        is_source_set_{is_source_set} {}

  ~euler_heat_equation_scheme() {}

  euler_heat_equation_scheme(euler_heat_equation_scheme const &) = delete;
  euler_heat_equation_scheme(euler_heat_equation_scheme &&) = delete;
  euler_heat_equation_scheme &operator=(euler_heat_equation_scheme const &) =
      delete;
  euler_heat_equation_scheme &operator=(euler_heat_equation_scheme &&) = delete;

  void operator()(
      sptr_t<dirichlet_boundary_1d<float>> const &dirichlet_boundary,
      container<float, alloc> &solution) const;
  void operator()(robin_boundary<float> const &robin_boundary,
                  container<float, alloc> &solution) const;
};

// ============================================================================
// ====== Double-Precision Floating-Point euler_heat_equation_scheme ==========
// ============================================================================

template <template <typename, typename> typename container, typename alloc>
class euler_heat_equation_scheme<double, container, alloc> {
 private:
  double space_start_;
  double terminal_t_;
  std::pair<double, double>
      deltas_;  // first = delta time, second = delta space;
  std::tuple<double, double, double> coeffs_;  // coefficients of PDE
  container<double, alloc> init_;
  std::function<double(double, double)> source_;
  bool is_source_set_;

 public:
  typedef double value_type;
  explicit euler_heat_equation_scheme() = delete;
  explicit euler_heat_equation_scheme(
      double space_start, double terminal_time,
      std::pair<double, double> const &deltas,
      std::tuple<double, double, double> const &coeffs,
      container<double, alloc> const &init,
      std::function<double(double, double)> const &source = nullptr,
      bool is_source_set = false)
      : space_start_{space_start},
        terminal_t_{terminal_time},
        deltas_{deltas},
        coeffs_{coeffs},
        init_{init},
        source_{source},
        is_source_set_{is_source_set} {}

  ~euler_heat_equation_scheme() {}

  euler_heat_equation_scheme(euler_heat_equation_scheme const &) = delete;
  euler_heat_equation_scheme(euler_heat_equation_scheme &&) = delete;
  euler_heat_equation_scheme &operator=(euler_heat_equation_scheme const &) =
      delete;
  euler_heat_equation_scheme &operator=(euler_heat_equation_scheme &&) = delete;

  void operator()(
      sptr_t<dirichlet_boundary_1d<double>> const &dirichlet_boundary,
      container<double, alloc> &solution) const;
  void operator()(robin_boundary<double> const &robin_boundary,
                  container<double, alloc> &solution) const;
};

// ============================================================================
// ========================= IMPLEMENTATIONS  =================================

// ============================================================================
// ===================== euler_heat_equation_scheme implementation ============
// ============================================================================

template <template <typename, typename> typename container, typename alloc>
void euler_heat_equation_scheme<float, container, alloc>::operator()(
    sptr_t<dirichlet_boundary_1d<float>> const &dirichlet_boundary,
    container<float, alloc> &solution) const {
  LSS_ASSERT(init_.size() == solution.size(),
             "Initial and final solution must have the same size");
  // get the size of the vector:
  std::size_t const size = solution.size();
  // create prev pointer:
  float *prev = (float *)malloc(size * sizeof(float));
  std::copy(init_.begin(), init_.end(), prev);
  // create next pointer:
  float *next = (float *)malloc(size * sizeof(float));
  // launch the Euler loop:
  euler_loop_sp loop{space_start_, terminal_t_, deltas_,
                     coeffs_,      source_,     is_source_set_};
  loop(prev, dirichlet_boundary, size, next);
  // next point to the solution
  std::copy(next, next + size, solution.begin());
  free(prev);
  free(next);
}

template <template <typename, typename> typename container, typename alloc>
void euler_heat_equation_scheme<double, container, alloc>::operator()(
    sptr_t<dirichlet_boundary_1d<double>> const &dirichlet_boundary,
    container<double, alloc> &solution) const {
  LSS_ASSERT(init_.size() == solution.size(),
             "Initial and final solution must have the same size");
  // get the size of the vector:
  std::size_t const size = solution.size();
  // create prev pointer:
  double *prev = (double *)malloc(size * sizeof(double));
  std::copy(init_.begin(), init_.end(), prev);
  // create next pointer:
  double *next = (double *)malloc(size * sizeof(double));
  // launch the Euler loop:
  euler_loop_dp loop{space_start_, terminal_t_, deltas_,
                     coeffs_,      source_,     is_source_set_};
  loop(prev, dirichlet_boundary, size, next);
  // next point to the solution
  std::copy(next, next + size, solution.begin());
  free(prev);
  free(next);
}

template <template <typename, typename> typename container, typename alloc>
void euler_heat_equation_scheme<float, container, alloc>::operator()(
    robin_boundary<float> const &robin_boundary,
    container<float, alloc> &solution) const {
  LSS_ASSERT(init_.size() == solution.size(),
             "Initial and final solution must have the same size");
  // get the size of the vector:
  std::size_t const size = solution.size();
  // create prev pointer:
  float *prev = (float *)malloc(size * sizeof(float));
  std::copy(init_.begin(), init_.end(), prev);
  // create next pointer:
  float *next = (float *)malloc(size * sizeof(float));
  // launch the Euler loop:
  euler_loop_sp loop{space_start_, terminal_t_, deltas_,
                     coeffs_,      source_,     is_source_set_};
  loop(prev, robin_boundary, size, next);
  // next point to the solution
  std::copy(next, next + size, solution.begin());
  free(prev);
  free(next);
}

template <template <typename, typename> typename container, typename alloc>
void euler_heat_equation_scheme<double, container, alloc>::operator()(
    robin_boundary<double> const &robin_boundary,
    container<double, alloc> &solution) const {
  LSS_ASSERT(init_.size() == solution.size(),
             "Initial and final solution must have the same size");
  // get the size of the vector:
  std::size_t const size = solution.size();
  // create prev pointer:
  double *prev = (double *)malloc(size * sizeof(double));
  std::copy(init_.begin(), init_.end(), prev);
  // create next pointer:
  double *next = (double *)malloc(size * sizeof(double));
  // launch the Euler loop:
  euler_loop_dp loop{space_start_, terminal_t_, deltas_,
                     coeffs_,      source_,     is_source_set_};
  loop(prev, robin_boundary, size, next);
  // next point to the solution
  std::copy(next, next + size, solution.begin());
  free(prev);
  free(next);
}

}  // namespace lss_one_dim_heat_explicit_schemes_cuda

#endif  ///_LSS_1D_HEAT_EXPLICIT_SCHEMES_CUDA
