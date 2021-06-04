#pragma once
#if !defined(_LSS_1D_BASE_EXPLICIT_SCHEMES)
#define _LSS_1D_BASE_EXPLICIT_SCHEMES

#pragma warning(disable : 4244)

#include <thread>

#include "common/lss_enumerations.h"
#include "pde_solvers/one_dim/lss_pde_boundary.h"
#include "pde_solvers/one_dim/lss_pde_utility.h"

namespace lss_one_dim_base_explicit_schemes {

using lss_one_dim_pde_boundary::dirichlet_boundary_1d;
using lss_one_dim_pde_utility::dirichlet_boundary;
using lss_one_dim_pde_utility::discretization;
using lss_one_dim_pde_utility::robin_boundary;
using lss_utility::sptr_t;

// ===================================================================
// ======================== heat_scheme_base =========================
// ===================================================================

template <template <typename, typename> typename container, typename fp_type,
          typename alloc, typename scheme_coefficient_holder>
class heat_scheme_base {
  typedef container<fp_type, alloc> container_t;

 protected:
  fp_type space_start_;
  fp_type terminal_time_;

  std::pair<fp_type, fp_type>
      deltas_;  // first = delta time, second = delta space
  scheme_coefficient_holder coeffs_;  // scheme coefficients of PDE
  container_t initial_condition_;
  std::function<fp_type(fp_type, fp_type)> source_;
  bool is_source_set_;

 public:
  explicit heat_scheme_base() = delete;

  explicit heat_scheme_base(
      fp_type space_start, fp_type terminal_time,
      std::pair<fp_type, fp_type> const &deltas,
      scheme_coefficient_holder const &coeffs,
      container_t const &initial_condition,
      std::function<fp_type(fp_type, fp_type)> const &source = nullptr,
      bool is_source_set = false)
      : space_start_{space_start},
        terminal_time_{terminal_time},
        deltas_{deltas},
        coeffs_{coeffs},
        initial_condition_{initial_condition},
        source_{source},
        is_source_set_{is_source_set} {}

  virtual ~heat_scheme_base() = default;

  // stability check:
  virtual bool is_stable() const = 0;

  // for Dirichlet BC
  virtual void operator()(
      sptr_t<dirichlet_boundary_1d<fp_type>> const &dirichlet_boundary,
      container_t &solution) const = 0;
  // for Robin BC
  virtual void operator()(robin_boundary<fp_type> const &robin_boundary,
                          container_t &solution) const = 0;
};

}  // namespace lss_one_dim_base_explicit_schemes

#endif  //_LSS_1D_BASE_EXPLICIT_SCHEMES
