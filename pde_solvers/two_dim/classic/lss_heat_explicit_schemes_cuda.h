#pragma once
#if !defined(_LSS_2D_HEAT_EXPLICIT_SCHEMES_CUDA)
#define _LSS_2D_HEAT_EXPLICIT_SCHEMES_CUDA

#include <tuple>

#include "common/lss_containers.h"
#include "common/lss_macros.h"
#include "common/lss_utility.h"
#include "pde_solvers/two_dim/lss_pde_utility.h"

namespace lss_two_dim_heat_explicit_schemes_cuda {

using lss_containers::container_2d;
using lss_two_dim_pde_utility::dirichlet_boundary_2d;
using lss_two_dim_pde_utility::pde_coefficient_holder_const;
using lss_two_dim_pde_utility::robin_boundary_2d;
using lss_utility::sptr_t;

template <typename fp_type>
struct dirichlet_device {
  // left Dirichlet boundary = function of x
  fp_type *left_y1;
  // right Dirichlet boundary = function of x
  fp_type *right_y2;
  // upper Dirichlet boundary = function of y
  fp_type *up_x1;
  // bottom Dirichlet boundary = function of y
  fp_type *bottom_x2;
};

template <typename fp_type>
struct euler_scheme_coeffs_device {
  fp_type A;
  fp_type B_1;
  fp_type B_2;
  fp_type C_1;
  fp_type C_2;
  fp_type gamma;
  fp_type k;
};

class euler_2d_loop_sp {
 private:
  float time_;
  float time_delta_;
  std::pair<float, float> spatial_inits_;
  std::pair<float, float> spatial_deltas_;
  pde_coefficient_holder_const<float> coeffs_;
  std::function<float(float, float, float)> source_;
  bool is_source_set_;

 public:
  ~euler_2d_loop_sp() {}
  explicit euler_2d_loop_sp() = delete;
  explicit euler_2d_loop_sp(
      float time, float time_delta,
      std::pair<float, float> const &spatial_inits,
      std::pair<float, float> const &spatial_deltas,
      pde_coefficient_holder_const<float> const &coeffs,
      std::function<float(float, float, float)> const &source,
      bool is_source_set = false)
      : time_{time},
        time_delta_{time_delta},
        spatial_inits_{spatial_inits},
        spatial_deltas_{spatial_deltas},
        coeffs_{coeffs},
        source_{source},
        is_source_set_{is_source_set} {}

  void operator()(
      float const *input,
      sptr_t<dirichlet_boundary_2d<float>> const &dirichlet_boundary,
      unsigned long long const rows, unsigned long long const columns,
      unsigned long long const size, float *solution) const;
  // TODO: missing implementation
  void operator()(float const *input,
                  sptr_t<robin_boundary_2d<float>> const &robin_boundary,
                  unsigned long long const rows,
                  unsigned long long const columns,
                  unsigned long long const size, float *solution) const;
};

class euler_2d_loop_dp {
 private:
  double time_;
  double time_delta_;
  std::pair<double, double> spatial_inits_;
  std::pair<double, double> spatial_deltas_;
  pde_coefficient_holder_const<double> coeffs_;
  std::function<double(double, double, double)> source_;
  bool is_source_set_;

 public:
  ~euler_2d_loop_dp() {}
  explicit euler_2d_loop_dp() = delete;
  explicit euler_2d_loop_dp(
      double time, double time_delta,
      std::pair<double, double> const &spatial_inits,
      std::pair<double, double> const &spatial_deltas,
      pde_coefficient_holder_const<double> const &coeffs,
      std::function<double(double, double, double)> const &source,
      bool is_source_set = false)
      : time_{time},
        time_delta_{time_delta},
        spatial_inits_{spatial_inits},
        spatial_deltas_{spatial_deltas},
        coeffs_{coeffs},
        source_{source},
        is_source_set_{is_source_set} {}

  void operator()(
      double const *input,
      sptr_t<dirichlet_boundary_2d<double>> const &dirichlet_boundary,
      unsigned long long const rows, unsigned long long const columns,
      unsigned long long const size, double *solution) const;
  // TODO: missing implementation
  void operator()(double const *input,
                  sptr_t<robin_boundary_2d<double>> const &robin_boundary,
                  unsigned long long const rows,
                  unsigned long long const columns,
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
  typedef container_2d<container, float, alloc> matrix_t;

  float time_;
  float time_delta_;
  std::pair<float, float> spatial_inits_;
  std::pair<float, float> spatial_deltas_;
  pde_coefficient_holder_const<float> coeffs_;
  sptr_t<matrix_t> initial_condition_;
  std::function<float(float, float, float)> source_;
  bool is_source_set_;

 public:
  typedef float value_type;
  explicit euler_heat_equation_scheme() = delete;
  explicit euler_heat_equation_scheme(
      float time, float time_delta,
      std::pair<float, float> const &spatial_inits,
      std::pair<float, float> const &spatial_deltas,
      pde_coefficient_holder_const<float> const &coeffs,
      sptr_t<matrix_t> const &initial_condition,
      std::function<float(float, float, float)> const &source = nullptr,
      bool is_source_set = false)
      : time_{time},
        time_delta_{time_delta},
        spatial_inits_{spatial_inits},
        spatial_deltas_{spatial_deltas},
        coeffs_{coeffs},
        initial_condition_{initial_condition},
        source_{source},
        is_source_set_{is_source_set} {}

  ~euler_heat_equation_scheme() {}

  euler_heat_equation_scheme(euler_heat_equation_scheme const &) = delete;
  euler_heat_equation_scheme(euler_heat_equation_scheme &&) = delete;
  euler_heat_equation_scheme &operator=(euler_heat_equation_scheme const &) =
      delete;
  euler_heat_equation_scheme &operator=(euler_heat_equation_scheme &&) = delete;

  void operator()(
      sptr_t<dirichlet_boundary_2d<float>> const &dirichlet_boundary,
      container_2d<container, float, alloc> &solution) const;
  // TODO:
  // void operator()(robin_boundary_2d<float> const &robin_boundary,
  //                container<float, alloc> &solution) const;
};

// ============================================================================
// ====== Double-Precision Floating-Point euler_heat_equation_scheme ==========
// ============================================================================

template <template <typename, typename> typename container, typename alloc>
class euler_heat_equation_scheme<double, container, alloc> {
 private:
  typedef container_2d<container, double, alloc> matrix_t;

  double time_;
  double time_delta_;
  std::pair<double, double> spatial_inits_;
  std::pair<double, double> spatial_deltas_;
  pde_coefficient_holder_const<double> coeffs_;
  sptr_t<matrix_t> initial_condition_;
  std::function<double(double, double, double)> source_;
  bool is_source_set_;

 public:
  typedef double value_type;
  explicit euler_heat_equation_scheme() = delete;
  explicit euler_heat_equation_scheme(
      double time, double time_delta,
      std::pair<double, double> const &spatial_inits,
      std::pair<double, double> const &spatial_deltas,
      pde_coefficient_holder_const<double> const &coeffs,
      sptr_t<matrix_t> const &initial_condition,
      std::function<double(double, double, double)> const &source = nullptr,
      bool is_source_set = false)
      : time_{time},
        time_delta_{time_delta},
        spatial_inits_{spatial_inits},
        spatial_deltas_{spatial_deltas},
        coeffs_{coeffs},
        initial_condition_{initial_condition},
        source_{source},
        is_source_set_{is_source_set} {}

  ~euler_heat_equation_scheme() {}

  euler_heat_equation_scheme(euler_heat_equation_scheme const &) = delete;
  euler_heat_equation_scheme(euler_heat_equation_scheme &&) = delete;
  euler_heat_equation_scheme &operator=(euler_heat_equation_scheme const &) =
      delete;
  euler_heat_equation_scheme &operator=(euler_heat_equation_scheme &&) = delete;

  void operator()(
      sptr_t<dirichlet_boundary_2d<double>> const &dirichlet_boundary,
      container_2d<container, double, alloc> &solution) const;
  // TODO:
  // void operator()(robin_boundary<double> const &robin_boundary,
  //                container<double, alloc> &solution) const;
};

// ============================================================================
// ========================= IMPLEMENTATIONS  =================================

// ============================================================================
// ===================== euler_heat_equation_scheme implementation ============
// ============================================================================

template <template <typename, typename> typename container, typename alloc>
void euler_heat_equation_scheme<float, container, alloc>::operator()(
    sptr_t<dirichlet_boundary_2d<float>> const &dirichlet_boundary,
    container_2d<container, float, alloc> &solution) const {
  LSS_ASSERT(initial_condition_->total_size() == solution.total_size(),
             "Initial and final solution must have the same size");
  // get the size of the vector:
  std::size_t const size = solution.total_size();
  std::size_t const rows = solution.rows();
  std::size_t const cols = solution.columns();
  // get flat data:
  const std::vector<float, alloc> data = initial_condition_->data();
  // create prev pointer:
  float *prev = (float *)malloc(size * sizeof(float));
  std::copy(data.begin(), data.end(), prev);
  // create next pointer:
  float *next = (float *)malloc(size * sizeof(float));
  // launch the Euler loop:
  euler_2d_loop_sp loop(time_, time_delta_, spatial_inits_, spatial_deltas_,
                        coeffs_, source_, is_source_set_);
  loop(prev, dirichlet_boundary, rows, cols, size, next);
  // next point to the solution
  solution.set_data(next, next + size);
  free(prev);
  free(next);
}

template <template <typename, typename> typename container, typename alloc>
void euler_heat_equation_scheme<double, container, alloc>::operator()(
    sptr_t<dirichlet_boundary_2d<double>> const &dirichlet_boundary,
    container_2d<container, double, alloc> &solution) const {
  LSS_ASSERT(initial_condition_->total_size() == solution.total_size(),
             "Initial and final solution must have the same size");
  // get the size of the vector:
  std::size_t const size = solution.total_size();
  std::size_t const rows = solution.rows();
  std::size_t const cols = solution.columns();
  // get flat data:
  const std::vector<double, alloc> data = initial_condition_->data();
  // create prev pointer:
  double *prev = (double *)malloc(size * sizeof(double));
  std::copy(data.begin(), data.end(), prev);
  // create next pointer:
  double *next = (double *)malloc(size * sizeof(double));
  // launch the Euler loop:
  euler_2d_loop_dp loop(time_, time_delta_, spatial_inits_, spatial_deltas_,
                        coeffs_, source_, is_source_set_);
  loop(prev, dirichlet_boundary, rows, cols, size, next);
  // next point to the solution
  solution.set_data(next, next + size);
  free(prev);
  free(next);
}

}  // namespace lss_two_dim_heat_explicit_schemes_cuda

#endif  ///_LSS_1D_HEAT_EXPLICIT_SCHEMES_CUDA
