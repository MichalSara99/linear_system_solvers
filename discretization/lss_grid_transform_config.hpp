#if !defined(_LSS_GRID_TRANSFORM_CONFIG_HPP_)
#define _LSS_GRID_TRANSFORM_CONFIG_HPP_

#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include "lss_discretization.hpp"
#include "lss_grid_config_hints.hpp"
#include "ode_solvers/lss_ode_discretization_config.hpp"
#include "pde_solvers/lss_pde_discretization_config.hpp"
#include <cmath>
#include <functional>
#include <map>
#include <vector>

namespace lss_grids
{
using lss_enumerations::dimension_enum;
using lss_enumerations::grid_enum;
using lss_ode_solvers::ode_discretization_config_ptr;
using lss_pde_solvers::pde_discretization_config_1d_ptr;
using lss_pde_solvers::pde_discretization_config_2d_ptr;
using lss_utility::range;
using lss_utility::sptr_t;

/**
    grid_config object
 */
template <dimension_enum dimension, typename fp_type> struct grid_transform_config
{
};

/**
    1D grid_transform_config structure
 */
template <typename fp_type> struct grid_transform_config<dimension_enum::One, fp_type>
{
  private:
    fp_type init_;
    fp_type alpha_;
    fp_type c_[2];
    std::function<fp_type(fp_type)> a_der_;
    std::function<fp_type(fp_type)> b_der_;

    void initialize(fp_type low, fp_type high, grid_config_hints_1d_ptr<fp_type> const &grid_hints)
    {
        auto const point = grid_hints->accumulation_point();
        alpha_ = high - low;
        // in case non-uniform spacing is requested alpha and beta are overriden
        if (grid_hints->grid() == grid_enum::Nonuniform)
        {
            alpha_ = (point - low) / grid_hints->alpha_scale();
        }
        init_ = point;
        c_[0] = std::asinh((high - point) / alpha_);
        c_[1] = std::asinh((low - point) / alpha_);
        auto const c_diff = (c_[0] - c_[1]);

        // initialize derivatives:
        auto const one = static_cast<fp_type>(1.0);
        a_der_ = [=](fp_type zeta) { return (alpha_ * c_diff * std::cosh(c_[0] * zeta + c_[1] * (one - zeta))); };
        b_der_ = [=](fp_type zeta) {
            return (alpha_ * c_diff * c_diff * std::sinh(c_[0] * zeta + c_[1] * (one - zeta)));
        };
    }

  public:
    explicit grid_transform_config(pde_discretization_config_1d_ptr<fp_type> const &discretization_config,
                                   grid_config_hints_1d_ptr<fp_type> const &grid_hints)
    {
        auto const low = discretization_config->space_range().lower();
        auto const high = discretization_config->space_range().upper();
        initialize(low, high, grid_hints);
    }

    explicit grid_transform_config(ode_discretization_config_ptr<fp_type> const &discretization_config,
                                   grid_config_hints_1d_ptr<fp_type> const &grid_hints)
    {
        auto const low = discretization_config->space_range().lower();
        auto const high = discretization_config->space_range().upper();
        initialize(low, high, grid_hints);
    }

    inline std::function<fp_type(fp_type)> const &a_derivative() const
    {
        return a_der_;
    }

    inline std::function<fp_type(fp_type)> const &b_derivative() const
    {
        return b_der_;
    }

    inline fp_type value_for(fp_type zeta)
    {
        auto const one = static_cast<fp_type>(1.0);
        return (init_ + alpha_ * std::sinh(c_[0] * zeta + c_[1] * (one - zeta)));
    }
};

/**
    2D grid_config structure
 */
template <typename fp_type> struct grid_transform_config<dimension_enum::Two, fp_type>
{
  private:
    fp_type init_1_;
    fp_type init_2_;
    fp_type alpha_;
    fp_type beta_;
    fp_type c_[2];
    fp_type d_;

    std::function<fp_type(fp_type)> a_der_;
    std::function<fp_type(fp_type)> b_der_;
    std::function<fp_type(fp_type)> c_der_;
    std::function<fp_type(fp_type)> d_der_;

    void initialize(pde_discretization_config_2d_ptr<fp_type> const &discretization_config,
                    grid_config_hints_2d_ptr<fp_type> const &grid_hints)
    {
        auto const low_1 = discretization_config->space_range().first.lower();
        auto const high_1 = discretization_config->space_range().first.upper();
        auto const low_2 = discretization_config->space_range().second.lower();
        auto const high_2 = discretization_config->space_range().second.upper();
        auto const point = grid_hints->accumulation_point();
        alpha_ = high_1 - low_1;
        beta_ = high_2 - low_2;
        // in case non-uniform spacing is requested alpha and beta are overriden
        if (grid_hints->grid() == grid_enum::Nonuniform)
        {
            alpha_ = (point - low_1) / grid_hints->alpha_scale();
            beta_ = (high_2 - low_2) / grid_hints->beta_scale();
        }
        init_1_ = point;
        init_2_ = low_2;
        c_[0] = std::asinh((high_1 - point) / alpha_);
        c_[1] = std::asinh((low_1 - point) / alpha_);
        d_ = std::asinh((high_2 - low_2) / beta_);
        auto const c_diff = (c_[0] - c_[1]);

        // initialize derivatives:
        auto const one = static_cast<fp_type>(1.0);
        a_der_ = [=](fp_type zeta) { return (alpha_ * c_diff * std::cosh(c_[0] * zeta + c_[1] * (one - zeta))); };
        b_der_ = [=](fp_type eta) { return (beta_ * d_ * std::cosh(d_ * eta)); };
        c_der_ = [=](fp_type zeta) {
            return (alpha_ * c_diff * c_diff * std::sinh(c_[0] * zeta + c_[1] * (one - zeta)));
        };
        d_der_ = [=](fp_type eta) { return (beta_ * d_ * d_ * std::sinh(d_ * eta)); };
    }

  public:
    explicit grid_transform_config(pde_discretization_config_2d_ptr<fp_type> const &discretization_config,
                                   grid_config_hints_2d_ptr<fp_type> const &grid_hints)
    {
        initialize(discretization_config, grid_hints);
    }

    inline std::function<fp_type(fp_type)> const &a_derivative() const
    {
        return a_der_;
    }

    inline std::function<fp_type(fp_type)> const &b_derivative() const
    {
        return b_der_;
    }

    inline std::function<fp_type(fp_type)> const &c_derivative() const
    {
        return c_der_;
    }

    inline std::function<fp_type(fp_type)> const &d_derivative() const
    {
        return d_der_;
    }

    inline fp_type value_for_1(fp_type zeta)
    {
        auto const one = static_cast<fp_type>(1.0);
        return (init_1_ + alpha_ * std::sinh(c_[0] * zeta + c_[1] * (one - zeta)));
    }

    inline fp_type value_for_2(fp_type eta)
    {
        return (init_2_ + beta_ * std::sinh(d_ * eta));
    }
};

template <typename fp_type> using grid_transform_config_1d = grid_transform_config<dimension_enum::One, fp_type>;
template <typename fp_type> using grid_transform_config_2d = grid_transform_config<dimension_enum::Two, fp_type>;
template <typename fp_type> using grid_transform_config_1d_ptr = sptr_t<grid_transform_config_1d<fp_type>>;
template <typename fp_type> using grid_transform_config_2d_ptr = sptr_t<grid_transform_config_2d<fp_type>>;

///////////////////////////////////////////////////////////////////////////////////////

} // namespace lss_grids
#endif ///_LSS_GRID_TRANSFORM_CONFIG_HPP_
