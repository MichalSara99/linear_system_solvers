#if !defined(_LSS_GRID_CONFIG_HPP_)
#define _LSS_GRID_CONFIG_HPP_

#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include "lss_discretization.hpp"
#include "ode_solvers/lss_ode_discretization_config.hpp"
#include "pde_solvers/lss_pde_discretization_config.hpp"
#include <cmath>
#include <functional>
#include <map>
#include <vector>

namespace lss_grids
{
using lss_enumerations::dimension_enum;
using lss_ode_solvers::ode_discretization_config_ptr;
using lss_pde_solvers::pde_discretization_config_1d_ptr;
using lss_pde_solvers::pde_discretization_config_2d_ptr;
using lss_utility::range;
using lss_utility::sptr_t;

/**
    grid_config object
 */
template <dimension_enum dimension, typename fp_type> struct grid_config
{
};

/**
    1D grid_config structure
 */
template <typename fp_type> struct grid_config<dimension_enum::One, fp_type>
{
  private:
    fp_type step_;

  public:
    explicit grid_config(pde_discretization_config_1d_ptr<fp_type> const &discretization_config)
    {
        auto const one = static_cast<fp_type>(1.0);
        step_ = one / (discretization_config->number_of_space_points() - 1);
    }

    explicit grid_config(ode_discretization_config_ptr<fp_type> const &discretization_config)
    {
        auto const one = static_cast<fp_type>(1.0);
        step_ = one / (discretization_config->number_of_space_points() - 1);
    }

    inline fp_type step() const
    {
        return step_;
    }

    inline std::size_t index_of(fp_type zeta)
    {
        return static_cast<std::size_t>(zeta / step_);
    }

    inline fp_type value_for(std::size_t idx)
    {
        return (step_ * idx);
    }
};

/**
    2D grid_config structure
 */
template <typename fp_type> struct grid_config<dimension_enum::Two, fp_type>
{
  private:
    fp_type step_1_;
    fp_type step_2_;
    sptr_t<grid_config<dimension_enum::One, fp_type>> grid_1_;
    sptr_t<grid_config<dimension_enum::One, fp_type>> grid_2_;

  public:
    explicit grid_config(pde_discretization_config_2d_ptr<fp_type> const &discretization_config)
    {
        auto const one = static_cast<fp_type>(1.0);
        step_1_ = one / (discretization_config->number_of_space_points().first - 1);
        step_2_ = one / (discretization_config->number_of_space_points().second - 1);
        grid_1_ =
            std::make_shared<grid_config<dimension_enum::One, fp_type>>(discretization_config->pde_discretization_1());
        grid_2_ =
            std::make_shared<grid_config<dimension_enum::One, fp_type>>(discretization_config->pde_discretization_2());
    }

    inline sptr_t<grid_config<dimension_enum::One, fp_type>> const &grid_1() const
    {
        return grid_1_;
    };

    inline sptr_t<grid_config<dimension_enum::One, fp_type>> const &grid_2() const
    {
        return grid_2_;
    }

    inline fp_type step_1() const
    {
        return step_1_;
    }

    inline fp_type step_2() const
    {
        return step_2_;
    }

    inline std::size_t index_of_1(fp_type zeta)
    {
        return static_cast<std::size_t>(zeta / step_1_);
    }

    inline std::size_t index_of_2(fp_type eta)
    {
        return static_cast<std::size_t>(eta / step_2_);
    }

    inline fp_type value_for_1(std::size_t idx)
    {
        return (step_1_ * idx);
    }

    inline fp_type value_for_2(std::size_t idx)
    {
        return (step_2_ * idx);
    }
};

template <typename fp_type> using grid_config_1d = grid_config<dimension_enum::One, fp_type>;
template <typename fp_type> using grid_config_2d = grid_config<dimension_enum::Two, fp_type>;
template <typename fp_type> using grid_config_1d_ptr = sptr_t<grid_config_1d<fp_type>>;
template <typename fp_type> using grid_config_2d_ptr = sptr_t<grid_config_2d<fp_type>>;

///////////////////////////////////////////////////////////////////////////////////////

} // namespace lss_grids
#endif ///_LSS_GRID_CONFIG_HPP_
