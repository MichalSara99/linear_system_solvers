#pragma once
#if !defined(_LSS_1D_SPACE_VARIABLE_HEAT_EXPLICIT_SCHEMES_CUDA)
#define _LSS_1D_SPACE_VARIABLE_HEAT_EXPLICIT_SCHEMES_CUDA

#include <tuple>

#include "common/lss_macros.h"
#include "common/lss_utility.h"
#include "pde_solvers/one_dim/lss_pde_utility.h"
#include "pde_solvers/one_dim/variable_coefficients/lss_space_variable_heat_explicit_schemes_cuda_policy.h"

namespace lss_one_dim_space_variable_heat_explicit_schemes_cuda {

using lss_one_dim_pde_utility::dirichlet_boundary;
using lss_one_dim_pde_utility::pde_coefficient_holder_fun_1_arg;
using lss_one_dim_pde_utility::robin_boundary;
using lss_one_dim_space_variable_heat_explicit_schemes_cuda_policy::
    heat_euler_scheme_forward_policy;

template <typename fp_type,
          typename scheme_policy = heat_euler_scheme_forward_policy<
              fp_type, pde_coefficient_holder_fun_1_arg<fp_type>>>
class euler_loop {
 private:
  fp_type space_start_;
  fp_type terminal_t_;
  std::pair<fp_type, fp_type>
      deltas_;  // first = delta time, second = delta space;
  pde_coefficient_holder_fun_1_arg<fp_type> coeffs_;  // coefficients of PDE
  std::function<fp_type(fp_type, fp_type)> source_;
  bool is_source_set_;

 public:
  ~euler_loop() {}
  explicit euler_loop() = delete;
  explicit euler_loop(fp_type space_start, fp_type terminal_time,
                      std::pair<fp_type, fp_type> const &deltas,
                      pde_coefficient_holder_fun_1_arg<fp_type> const &coeffs,
                      std::function<fp_type(fp_type, fp_type)> const &source,
                      bool is_source_set = false)
      : space_start_{space_start},
        terminal_t_{terminal_time},
        deltas_{deltas},
        coeffs_{coeffs},
        source_{source},
        is_source_set_{is_source_set} {}

  void operator()(fp_type const *input,
                  dirichlet_boundary<fp_type> const &dirichlet_boundary,
                  unsigned long long const size, fp_type *solution) const {
    if (!is_source_set_) {
      scheme_policy::traverse(solution, input, size, dirichlet_boundary,
                              deltas_, coeffs_, terminal_t_, space_start_);
    } else {
      scheme_policy::traverse(solution, input, size, dirichlet_boundary,
                              deltas_, coeffs_, terminal_t_, space_start_,
                              source_);
    }
  }
  void operator()(fp_type const *input,
                  robin_boundary<fp_type> const &robin_boundary,
                  unsigned long long const size, fp_type *solution) const {
    if (!is_source_set_) {
      scheme_policy::traverse(solution, input, size, robin_boundary, deltas_,
                              coeffs_, terminal_t_, space_start_);
    } else {
      scheme_policy::traverse(solution, input, size, robin_boundary, deltas_,
                              coeffs_, terminal_t_, space_start_, source_);
    }
  }
};

// ============================================================================
// =============== euler_heat_equation_scheme general template ================
// ============================================================================

template <typename fp_type, template <typename, typename> typename container,
          typename alloc,
          typename scheme_policy = heat_euler_scheme_forward_policy<
              fp_type, pde_coefficient_holder_fun_1_arg<fp_type>>>
class euler_heat_equation_scheme {
 private:
  fp_type space_start_;
  fp_type terminal_t_;
  std::pair<fp_type, fp_type>
      deltas_;  // first = delta time, second = delta space;
  pde_coefficient_holder_fun_1_arg<fp_type> coeffs_;  // coefficients of PDE
  container<fp_type, alloc> init_;
  std::function<fp_type(fp_type, fp_type)> source_;
  bool is_source_set_;

 public:
  typedef fp_type value_type;
  explicit euler_heat_equation_scheme() = delete;
  explicit euler_heat_equation_scheme(
      fp_type space_start, fp_type terminal_time,
      std::pair<fp_type, fp_type> const &deltas,
      pde_coefficient_holder_fun_1_arg<fp_type> const &coeffs,
      container<fp_type, alloc> const &init,
      std::function<fp_type(fp_type, fp_type)> const &source = nullptr,
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

  void operator()(dirichlet_boundary<fp_type> const &dirichlet_boundary,
                  container<fp_type, alloc> &solution) const;
  void operator()(robin_boundary<fp_type> const &robin_boundary,
                  container<fp_type, alloc> &solution) const;
};

// ============================================================================
// ========================= IMPLEMENTATIONS  =================================

// ============================================================================
// ===================== euler_heat_equation_scheme implementation ============
// ============================================================================

template <typename fp_type, template <typename, typename> typename container,
          typename alloc, typename scheme_policy>
void euler_heat_equation_scheme<fp_type, container, alloc, scheme_policy>::
operator()(dirichlet_boundary<fp_type> const &dirichlet_boundary,
           container<fp_type, alloc> &solution) const {
  LSS_ASSERT(init_.size() == solution.size(),
             "Initial and final solution must have the same size");
  // get the size of the vector:
  std::size_t const size = solution.size();
  // create prev pointer:
  fp_type *prev = (fp_type *)malloc(size * sizeof(fp_type));
  std::copy(init_.begin(), init_.end(), prev);
  // create next pointer:
  fp_type *next = (fp_type *)malloc(size * sizeof(fp_type));
  // launch the Euler loop:
  euler_loop<fp_type, scheme_policy> loop{
      space_start_, terminal_t_, deltas_, coeffs_, source_, is_source_set_};
  loop(prev, dirichlet_boundary, size, next);
  // next point to the solution
  std::copy(next, next + size, solution.begin());
  free(prev);
  free(next);
}

template <typename fp_type, template <typename, typename> typename container,
          typename alloc, typename scheme_policy>
void euler_heat_equation_scheme<fp_type, container, alloc, scheme_policy>::
operator()(robin_boundary<fp_type> const &robin_boundary,
           container<fp_type, alloc> &solution) const {
  LSS_ASSERT(init_.size() == solution.size(),
             "Initial and final solution must have the same size");
  // get the size of the vector:
  std::size_t const size = solution.size();
  // create prev pointer:
  fp_type *prev = (fp_type *)malloc(size * sizeof(fp_type));
  std::copy(init_.begin(), init_.end(), prev);
  // create next pointer:
  fp_type *next = (fp_type *)malloc(size * sizeof(fp_type));
  // launch the Euler loop:
  euler_loop<fp_type, scheme_policy> loop{
      space_start_, terminal_t_, deltas_, coeffs_, source_, is_source_set_};
  loop(prev, robin_boundary, size, next);
  // next point to the solution
  std::copy(next, next + size, solution.begin());
  free(prev);
  free(next);
}

}  // namespace lss_one_dim_space_variable_heat_explicit_schemes_cuda

#endif  ///_LSS_1D_SPACE_VARIABLE_HEAT_EXPLICIT_SCHEMES_CUDA
