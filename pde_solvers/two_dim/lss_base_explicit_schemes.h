#pragma once
#if !defined(_LSS_2D_BASE_EXPLICIT_SCHEMES)
#define _LSS_2D_BASE_EXPLICIT_SCHEMES

#pragma warning(disable : 4244)

#include <thread>

#include "common/lss_enumerations.h"
#include "common/lss_utility.h"
#include "pde_solvers/two_dim/lss_pde_utility.h"

namespace lss_two_dim_base_explicit_schemes {

using lss_two_dim_pde_utility::dirichlet_boundary_2d;
using lss_two_dim_pde_utility::discretization_2d;
using lss_two_dim_pde_utility::robin_boundary_2d;
using lss_utility::container_2d;
using lss_utility::sptr_t;

// ============================================================================
// ================================ heat_scheme_base  =========================
// ============================================================================

template <template <typename, typename> typename container, typename fp_type,
          typename alloc, typename scheme_coefficient_holder>
class heat_scheme_base {
  typedef container_2d<container, fp_type, alloc> matrix_t;

 protected:
  fp_type time_;
  fp_type time_delta_;
  std::pair<fp_type, fp_type> spatial_inits_;
  std::pair<fp_type, fp_type> spatial_deltas_;
  sptr_t<scheme_coefficient_holder> coeffs_;  // scheme coefficients of PDE
  sptr_t<matrix_t> initial_condition_;
  std::function<fp_type(fp_type, fp_type, fp_type)> source_;
  bool is_source_set_;

  heat_scheme_base() = delete;

 public:
  explicit heat_scheme_base(
      fp_type time, fp_type time_delta,
      std::pair<fp_type, fp_type> const &spatial_inits,
      std::pair<fp_type, fp_type> const &spatial_deltas,
      sptr_t<scheme_coefficient_holder> const &coeffs,
      sptr_t<matrix_t> const &initial_condition,
      std::function<fp_type(fp_type, fp_type, fp_type)> const &source = nullptr,
      bool is_source_set = false)
      : time_{time},
        time_delta_{time_delta},
        spatial_inits_{spatial_inits},
        spatial_deltas_{spatial_deltas},
        coeffs_{coeffs},
        initial_condition_{initial_condition},
        source_{source},
        is_source_set_{is_source_set} {}

  virtual ~heat_scheme_base() = default;

  // stability check:
  virtual bool is_stable() const = 0;

  // for Dirichlet BC
  virtual void operator()(
      sptr_t<dirichlet_boundary_2d<fp_type>> const &dirichlet_boundary,
      matrix_t &solution) const = 0;
  // for Robin BC
  virtual void operator()(
      sptr_t<robin_boundary_2d<fp_type>> const &robin_boundary,
      matrix_t &solution) const = 0;
};

}  // namespace lss_two_dim_base_explicit_schemes

#endif  //_LSS_2D_BASE_EXPLICIT_SCHEMES
