#pragma once
#if !defined(_LSS_1D_GENERAL_HEAT_EQUATION_SOLVERS_BASE)
#define _LSS_1D_GENERAL_HEAT_EQUATION_SOLVERS_BASE

#include <functional>

#include "common/lss_enumerations.h"

namespace lss_one_dim_classic_pde_solvers {
using lss_enumerations::boundary_condition_enum;
using lss_enumerations::memory_space_enum;

namespace implicit_solvers {
// ============================================================================
// ================== general_heat_equation General Template ==================
// ============================================================================

template <typename fp_type, boundary_condition_enum b_type,
          template <typename, boundary_condition_enum,
                    template <typename, typename> typename cont, typename>
          typename fdm_solver,
          template <typename, typename> typename container, typename alloc>
class general_heat_equation {};

// ============================================================================
// ============= general_heat_equation_cuda General Template ==================
// ============================================================================

template <typename fp_type, boundary_condition_enum b_type,
          memory_space_enum memory_space,
          template <memory_space_enum, typename>
          typename real_sparse_policy_cuda,
          template <typename, typename> typename container, typename alloc>
class general_heat_equation_cuda {};

}  // namespace implicit_solvers

namespace explicit_solvers {

// ============================================================================
// ================== general_heat_equation General Template ==================
// ============================================================================

template <typename fp_type, boundary_condition_enum b_type,
          template <typename, typename> typename container, typename alloc>
class general_heat_equation {};

// ============================================================================
// ================= general_heat_equation_cuda General Template ==============
// ============================================================================

template <typename fp_type, boundary_condition_enum b_type,
          template <typename, typename> typename container, typename alloc>
class general_heat_equation_cuda {};

}  // namespace explicit_solvers

}  // namespace lss_one_dim_classic_pde_solvers

#endif  //_LSS_1D_GENERAL_HEAT_EQUATION_SOLVERS_BASE
