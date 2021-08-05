#if !defined(_LSS_ODE_DISCRETIZATION_CONFIG_HPP_)
#define _LSS_ODE_DISCRETIZATION_CONFIG_HPP_

#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"

namespace lss_ode_solvers
{

using lss_enumerations::dimension_enum;
using lss_utility::range;
using lss_utility::sptr_t;

/**
    ode_discretization_config structure
 */
template <typename fp_type> struct ode_discretization_config
{
  private:
    range<fp_type> space_range_;
    std::size_t number_of_space_points_;

    explicit ode_discretization_config() = delete;

  public:
    explicit ode_discretization_config(range<fp_type> const &space_range, std::size_t const &number_of_space_points)
        : space_range_{space_range}, number_of_space_points_{number_of_space_points}
    {
    }
    ~ode_discretization_config()
    {
    }

    inline range<fp_type> const &space_range() const
    {
        return space_range_;
    }

    inline std::size_t number_of_space_points() const
    {
        return number_of_space_points_;
    }

    inline fp_type space_step() const
    {
        return ((space_range_.spread()) / static_cast<fp_type>(number_of_space_points_ - 1));
    }
};

template <typename fp_type> using ode_discretization_config_ptr = sptr_t<ode_discretization_config<fp_type>>;

} // namespace lss_ode_solvers

#endif ///_LSS_ODE_DISCRETIZATION_CONFIG_HPP_
