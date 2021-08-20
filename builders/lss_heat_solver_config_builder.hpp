#if !defined(_LSS_HEAT_SOLVER_CONFIG_BUILDER_HPP_)
#define _LSS_HEAT_SOLVER_CONFIG_BUILDER_HPP_

#include <map>

#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include "pde_solvers/lss_heat_solver_config.hpp"

namespace lss_pde_solvers
{

using lss_enumerations::dimension_enum;
using lss_enumerations::explicit_pde_schemes_enum;
using lss_enumerations::factorization_enum;
using lss_enumerations::implicit_pde_schemes_enum;
using lss_enumerations::memory_space_enum;
using lss_enumerations::traverse_direction_enum;
using lss_enumerations::tridiagonal_method_enum;
using lss_utility::range;
using lss_utility::sptr_t;

/**
    heat_implicit_solver_config_builder structure
 */
struct heat_implicit_solver_config_builder
{
  private:
    memory_space_enum memory_space_;
    traverse_direction_enum traverse_direction_;
    tridiagonal_method_enum tridiagonal_method_;
    factorization_enum tridiagonal_factorization_;
    implicit_pde_schemes_enum implicit_pde_scheme_;

  public:
    heat_implicit_solver_config_builder &memory_space(memory_space_enum memory_space)
    {
        memory_space_ = memory_space;
        return *this;
    }

    heat_implicit_solver_config_builder &traverse_direction(traverse_direction_enum traverse_direction)
    {
        traverse_direction_ = traverse_direction;
        return *this;
    }

    heat_implicit_solver_config_builder &tridiagonal_method(tridiagonal_method_enum tridiagonal_method)
    {
        tridiagonal_method_ = tridiagonal_method;
        return *this;
    }

    heat_implicit_solver_config_builder &tridiagonal_factorization(factorization_enum tridiagonal_factorization)
    {
        tridiagonal_factorization_ = tridiagonal_factorization;
        return *this;
    }

    heat_implicit_solver_config_builder &implicit_pde_scheme(implicit_pde_schemes_enum implicit_pde_scheme)
    {
        implicit_pde_scheme_ = implicit_pde_scheme;
        return *this;
    }

    heat_implicit_solver_config_ptr build()
    {
        return std::make_shared<heat_implicit_solver_config>(memory_space_, traverse_direction_, tridiagonal_method_,
                                                             tridiagonal_factorization_, implicit_pde_scheme_);
    }
};

/**
    heat_explicit_solver_config_builder structure
 */
struct heat_explicit_solver_config_builder
{
  private:
    memory_space_enum memory_space_;
    traverse_direction_enum traverse_direction_;
    explicit_pde_schemes_enum explicit_pde_scheme_;

  public:
    heat_explicit_solver_config_builder &memory_space(memory_space_enum memory_space)
    {
        memory_space_ = memory_space;
        return *this;
    }

    heat_explicit_solver_config_builder &traverse_direction(traverse_direction_enum traverse_direction)
    {
        traverse_direction_ = traverse_direction;
        return *this;
    }

    heat_explicit_solver_config_builder &explicit_pde_scheme(explicit_pde_schemes_enum explicit_pde_scheme)
    {
        explicit_pde_scheme_ = explicit_pde_scheme;
        return *this;
    }

    heat_explicit_solver_config_ptr build()
    {
        return std::make_shared<heat_explicit_solver_config>(memory_space_, traverse_direction_, explicit_pde_scheme_);
    }
};

using heat_implicit_solver_config_builder_ptr = sptr_t<heat_implicit_solver_config_builder>;

using heat_explicit_solver_config_builder_ptr = sptr_t<heat_explicit_solver_config_builder>;

} // namespace lss_pde_solvers

#endif ///_LSS_HEAT_SOLVER_CONFIG_BUILDER_HPP_
