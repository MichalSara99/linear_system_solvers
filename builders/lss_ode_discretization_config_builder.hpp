#if !defined(_LSS_ODE_DISCRETIZATION_CONFIG_BUILDER_HPP_)
#define _LSS_ODE_DISCRETIZATION_CONFIG_BUILDER_HPP_

#include "common/lss_utility.hpp"
#include "ode_solvers/lss_ode_discretization_config.hpp"

namespace lss_ode_solvers
{

using lss_utility::range;
using lss_utility::sptr_t;

/**
    ode_discretization_config_builder structure
 */
template <typename fp_type> struct ode_discretization_config_builder
{
  private:
    range<fp_type> space_range_;
    std::size_t number_of_space_points_;

  public:
    ode_discretization_config_builder &space_range(range<fp_type> const &space_range)
    {
        space_range_ = space_range;
        return *this;
    }

    ode_discretization_config_builder &number_of_space_points(std::size_t const &number_of_space_points)
    {
        number_of_space_points_ = number_of_space_points;
        return *this;
    }

    ode_discretization_config_ptr<fp_type> build()
    {
        return std::make_shared<ode_discretization_config<fp_type>>(space_range_, number_of_space_points_);
    }
};

template <typename fp_type>
using ode_discretization_config_builder_ptr = sptr_t<ode_discretization_config_builder<fp_type>>;

} // namespace lss_ode_solvers

#endif ///_LSS_ODE_DISCRETIZATION_CONFIG_BUILDER_HPP_
