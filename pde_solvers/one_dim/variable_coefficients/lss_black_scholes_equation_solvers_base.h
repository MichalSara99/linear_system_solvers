#pragma once
#if !defined(_LSS_BLACK_SCHOLES_EQUATION_SOLVERS_BASE)
#define _LSS_BLACK_SCHOLES_EQUATION_SOLVERS_BASE

#include <functional>

#include "common/lss_enumerations.h"
#include "common/lss_macros.h"
#include "common/lss_utility.h"
#include "lss_space_variable_heat_explicit_schemes.h"
#include "lss_space_variable_heat_explicit_schemes_policy.h"
#include "lss_space_variable_heat_implicit_schemes.h"
#include "pde_solvers/one_dim/lss_pde_boundary.h"
#include "pde_solvers/one_dim/lss_pde_utility.h"

namespace lss_one_dim_space_variable_pde_solvers {

using lss_enumerations::boundary_condition_enum;
using lss_enumerations::memory_space_enum;

namespace implicit_solvers {

// ============================================================================
// =================== black_scholes_equation General Template ================
// ============================================================================

template <typename fp_type, boundary_condition_enum b_type,
          template <typename, boundary_condition_enum,
                    template <typename, typename> typename cont, typename>
          typename FDMSolver,
          template <typename, typename> typename container, typename alloc>
class black_scholes_equation {};

// ============================================================================
// ================= black_scholes_equation_cuda General Template ==============
// ============================================================================

template <typename fp_type, boundary_condition_enum b_type,
          memory_space_enum memory_space,
          template <memory_space_enum, typename>
          typename real_sparse_policy_cuda,
          template <typename, typename> typename container, typename alloc>
class black_scholes_equation_cuda {};

}  // namespace implicit_solvers

namespace explicit_solvers {

// ============================================================================
// ================ black_scholes_equation General Template ===================
// ============================================================================

template <typename fp_type, boundary_condition_enum b_type,
          template <typename, typename> typename container, typename alloc>
class black_scholes_equation {};

// ============================================================================
// ================= black_scholes_equation_cuda General Template ==============
// ============================================================================

template <typename fp_type, boundary_condition_enum b_type,
          template <typename, typename> typename container, typename alloc>
class black_scholes_equation_cuda {};

}  // namespace explicit_solvers

}  // namespace lss_one_dim_space_variable_pde_solvers

#endif  //_LSS_BLACK_SCHOLES_EQUATION_SOLVERS_BASE
