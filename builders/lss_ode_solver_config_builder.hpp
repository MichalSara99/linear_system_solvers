#if !defined(_LSS_ODE_SOLVER_CONFIG_BUILDER_HPP_)
#define _LSS_ODE_SOLVER_CONFIG_BUILDER_HPP_

#include <map>

#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include "ode_solvers/lss_ode_solver_config.hpp"

namespace lss_ode_solvers
{

using lss_enumerations::factorization_enum;
using lss_enumerations::memory_space_enum;
using lss_enumerations::tridiagonal_method_enum;
using lss_utility::range;
using lss_utility::sptr_t;

/**
    ode_implicit_solver_config_builder structure
 */
struct ode_implicit_solver_config_builder
{
  private:
    memory_space_enum memory_space_;
    tridiagonal_method_enum tridiagonal_method_;
    factorization_enum tridiagonal_factorization_;

  public:
    ode_implicit_solver_config_builder &memory_space(memory_space_enum memory_space)
    {
        memory_space_ = memory_space;
        return *this;
    }

    ode_implicit_solver_config_builder &tridiagonal_method(tridiagonal_method_enum tridiagonal_method)
    {
        tridiagonal_method_ = tridiagonal_method;
        return *this;
    }

    ode_implicit_solver_config_builder &tridiagonal_factorization(factorization_enum tridiagonal_factorization)
    {
        tridiagonal_factorization_ = tridiagonal_factorization;
        return *this;
    }

    ode_implicit_solver_config_ptr build()
    {
        return std::make_shared<ode_implicit_solver_config>(memory_space_, tridiagonal_method_,
                                                            tridiagonal_factorization_);
    }
};

using ode_implicit_solver_config_builder_ptr = sptr_t<ode_implicit_solver_config_builder>;

} // namespace lss_ode_solvers

#endif ///_LSS_ODE_SOLVER_CONFIG_BUILDER_HPP_
