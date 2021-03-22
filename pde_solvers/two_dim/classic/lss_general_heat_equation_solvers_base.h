#pragma once
#if !defined(_LSS_2D_GENERAL_HEAT_EQUATION_SOLVERS_BASE)
#define _LSS_2D_GENERAL_HEAT_EQUATION_SOLVERS_BASE

#include "common/lss_enumerations.h"

namespace lss_two_dim_classic_pde_solvers {

using lss_enumerations::boundary_condition_enum;

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

}  // namespace implicit_solvers

namespace explicit_solvers {
// ============================================================================
// ================== general_heat_equation General Template ==================
// ============================================================================

template <typename fp_type, boundary_condition_enum b_type,
          template <typename, typename> typename container, typename alloc>
class general_heat_equation {};

}  // namespace explicit_solvers

}  // namespace lss_two_dim_classic_pde_solvers

#endif  //_LSS_2D_GENERAL_HEAT_EQUATION_SOLVERS_BASE
