#if !defined(_LSS_WAVE_SOLVER_CONFIG_BUILDER_T_HPP_)
#define _LSS_WAVE_SOLVER_CONFIG_BUILDER_T_HPP_

#include <map>

#include "builders/lss_wave_solver_config_builder.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"

using lss_enumerations::dimension_enum;
using lss_enumerations::factorization_enum;
using lss_enumerations::memory_space_enum;
using lss_enumerations::traverse_direction_enum;
using lss_enumerations::tridiagonal_method_enum;
using lss_pde_solvers::wave_explicit_solver_config_builder;
using lss_pde_solvers::wave_implicit_solver_config_builder;
using lss_utility::range;
using lss_utility::sptr_t;

void test_wave_solver_config_implicit_builder()
{
    auto const &solver = wave_implicit_solver_config_builder()
                             .memory_space(memory_space_enum::Host)
                             .traverse_direction(traverse_direction_enum::Forward)
                             .tridiagonal_method(tridiagonal_method_enum::SORSolver)
                             .tridiagonal_factorization(factorization_enum::None)
                             .build();
    LSS_ASSERT(solver != nullptr, "Must not be null pointer");
}

void test_wave_solver_config_explicit_builder()
{
    auto const &solver = wave_explicit_solver_config_builder()
                             .memory_space(memory_space_enum::Host)
                             .traverse_direction(traverse_direction_enum::Forward)
                             .build();
    LSS_ASSERT(solver != nullptr, "Must not be null pointer");
}

#endif ///_LSS_WAVE_SOLVER_CONFIG_BUILDER_T_HPP_
