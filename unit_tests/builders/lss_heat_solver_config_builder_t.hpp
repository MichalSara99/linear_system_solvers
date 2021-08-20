#if !defined(_LSS_HEAT_SOLVER_CONFIG_BUILDER_T_HPP_)
#define _LSS_HEAT_SOLVER_CONFIG_BUILDER_T_HPP_

#include <map>

#include "builders/lss_heat_solver_config_builder.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"

using lss_enumerations::dimension_enum;
using lss_enumerations::explicit_pde_schemes_enum;
using lss_enumerations::factorization_enum;
using lss_enumerations::implicit_pde_schemes_enum;
using lss_enumerations::memory_space_enum;
using lss_enumerations::traverse_direction_enum;
using lss_enumerations::tridiagonal_method_enum;
using lss_pde_solvers::heat_explicit_solver_config_builder;
using lss_pde_solvers::heat_implicit_solver_config_builder;
using lss_utility::range;
using lss_utility::sptr_t;

void test_heat_solver_config_implicit_builder()
{
    auto const &solver = heat_implicit_solver_config_builder()
                             .memory_space(memory_space_enum::Host)
                             .traverse_direction(traverse_direction_enum::Forward)
                             .tridiagonal_method(tridiagonal_method_enum::SORSolver)
                             .tridiagonal_factorization(factorization_enum::None)
                             .implicit_pde_scheme(implicit_pde_schemes_enum::CrankNicolson)
                             .build();
    LSS_ASSERT(solver != nullptr, "Must not be null pointer");
}

void test_heat_solver_config_explicit_builder()
{
    auto const &solver = heat_explicit_solver_config_builder()
                             .memory_space(memory_space_enum::Host)
                             .traverse_direction(traverse_direction_enum::Forward)
                             .explicit_pde_scheme(explicit_pde_schemes_enum::Euler)
                             .build();
    LSS_ASSERT(solver != nullptr, "Must not be null pointer");
}

#endif ///_LSS_HEAT_SOLVER_CONFIG_BUILDER_T_HPP_
