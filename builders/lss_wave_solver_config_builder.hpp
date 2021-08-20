#if !defined(_LSS_WAVE_SOLVER_CONFIG_BUILDER_HPP_)
#define _LSS_WAVE_SOLVER_CONFIG_BUILDER_HPP_

#include <map>

#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include "pde_solvers/lss_wave_solver_config.hpp"

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
    wave_implicit_solver_config_builder structure
 */
struct wave_implicit_solver_config_builder
{
  private:
    memory_space_enum memory_space_;
    traverse_direction_enum traverse_direction_;
    tridiagonal_method_enum tridiagonal_method_;
    factorization_enum tridiagonal_factorization_;

  public:
    wave_implicit_solver_config_builder &memory_space(memory_space_enum memory_space)
    {
        memory_space_ = memory_space;
        return *this;
    }

    wave_implicit_solver_config_builder &traverse_direction(traverse_direction_enum traverse_direction)
    {
        traverse_direction_ = traverse_direction;
        return *this;
    }

    wave_implicit_solver_config_builder &tridiagonal_method(tridiagonal_method_enum tridiagonal_method)
    {
        tridiagonal_method_ = tridiagonal_method;
        return *this;
    }

    wave_implicit_solver_config_builder &tridiagonal_factorization(factorization_enum tridiagonal_factorization)
    {
        tridiagonal_factorization_ = tridiagonal_factorization;
        return *this;
    }

    wave_implicit_solver_config_ptr build()
    {
        return std::make_shared<wave_implicit_solver_config>(memory_space_, traverse_direction_, tridiagonal_method_,
                                                             tridiagonal_factorization_);
    }
};

/**
    wave_explicit_solver_config_builder structure
 */
struct wave_explicit_solver_config_builder
{
  private:
    memory_space_enum memory_space_;
    traverse_direction_enum traverse_direction_;

  public:
    wave_explicit_solver_config_builder &memory_space(memory_space_enum memory_space)
    {
        memory_space_ = memory_space;
        return *this;
    }

    wave_explicit_solver_config_builder &traverse_direction(traverse_direction_enum traverse_direction)
    {
        traverse_direction_ = traverse_direction;
        return *this;
    }

    wave_explicit_solver_config_ptr build()
    {
        return std::make_shared<wave_explicit_solver_config>(memory_space_, traverse_direction_);
    }
};

using wave_implicit_solver_config_builder_ptr = sptr_t<wave_implicit_solver_config_builder>;

using wave_explicit_solver_config_builder_ptr = sptr_t<wave_explicit_solver_config_builder>;

} // namespace lss_pde_solvers

#endif ///_LSS_WAVE_SOLVER_CONFIG_BUILDER_HPP_
