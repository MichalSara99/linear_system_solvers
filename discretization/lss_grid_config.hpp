#if !defined(_LSS_GRID_CONFIG_HPP_)
#define _LSS_GRID_CONFIG_HPP_

#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include "lss_discretization.hpp"
#include "ode_solvers/lss_ode_discretization_config.hpp"
#include "pde_solvers/lss_pde_discretization_config.hpp"
#include <cmath>
#include <functional>

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
  protected:
    fp_type init_;
    // fp_type step_;

    virtual void initialize(pde_discretization_config_1d_ptr<fp_type> const &discretization_config) = 0;

  public:
    explicit grid_config(pde_discretization_config_1d_ptr<fp_type> const &discretization_config)
    {
    }

    explicit grid_config(ode_discretization_config_ptr<fp_type> const &discretization_config)
    {
    }

    inline fp_type init() const
    {
        return init_;
    }

    // inline fp_type step() const
    //{
    //     return step_;
    // }

    virtual inline fp_type gen_function(std::size_t t) = 0;
};

/**
    2D grid_config structure
 */
template <typename fp_type> struct grid_config<dimension_enum::Two, fp_type>
{
  protected:
    fp_type init_1_;
    fp_type init_2_;
    // fp_type step_1_;
    // fp_type step_2_;
    sptr_t<grid_config<dimension_enum::One, fp_type>> grid_1_;
    sptr_t<grid_config<dimension_enum::One, fp_type>> grid_2_;

  private:
    virtual void initialize(pde_discretization_config_2d_ptr<fp_type> const &discretization_config) = 0;

  public:
    explicit grid_config(pde_discretization_config_2d_ptr<fp_type> const &discretization_config)
    {
    }

    inline fp_type const &init_1() const
    {
        return init_1_;
    }

    inline fp_type const &init_2() const
    {
        return init_2_;
    }

    // inline fp_type const &step_1() const
    //{
    //     return step_1_;
    // }

    // inline fp_type const &step_2() const
    //{
    //     return step_2_;
    // }

    inline sptr_t<grid_config<dimension_enum::One, fp_type>> const &grid_1() const
    {
        return grid_1_;
    };

    inline sptr_t<grid_config<dimension_enum::One, fp_type>> const &grid_2() const
    {
        return grid_2_;
    }

    virtual inline fp_type gen_function_1(std::size_t t) = 0;

    virtual inline fp_type gen_function_2(std::size_t t) = 0;
};

template <typename fp_type> using grid_config_1d = grid_config<dimension_enum::One, fp_type>;
template <typename fp_type> using grid_config_2d = grid_config<dimension_enum::Two, fp_type>;
template <typename fp_type> using grid_config_1d_ptr = sptr_t<grid_config_1d<fp_type>>;
template <typename fp_type> using grid_config_2d_ptr = sptr_t<grid_config_2d<fp_type>>;

///////////////////////////////////////////////////////////////////////////////////////

/**
    uniform_grid_config object
 */
template <dimension_enum dimension, typename fp_type> struct uniform_grid_config
{
};

/**
    1D uniform_grid_config structure
 */
template <typename fp_type>
struct uniform_grid_config<dimension_enum::One, fp_type> : grid_config<dimension_enum::One, fp_type>
{
  private:
    fp_type step_;

    void initialize(pde_discretization_config_1d_ptr<fp_type> const &discretization_config) override
    {
        init_ = discretization_config->space_range().lower();
        step_ = discretization_config->space_step();
    }

  public:
    uniform_grid_config() = delete;

    explicit uniform_grid_config(pde_discretization_config_1d_ptr<fp_type> const &discretization_config)
        : grid_config<dimension_enum::One, fp_type>(discretization_config)
    {
        initialize(discretization_config);
    }

    explicit uniform_grid_config(ode_discretization_config_ptr<fp_type> const &discretization_config)
        : grid_config<dimension_enum::One, fp_type>(discretization_config)
    {
        init_ = discretization_config->space_range().lower();
        step_ = discretization_config->space_step();
    }

    inline fp_type gen_function(std::size_t t) override
    {
        return (static_cast<fp_type>(t) * step_);
    }
};

/**
    2D uniform_grid_config structure
 */
template <typename fp_type>
struct uniform_grid_config<dimension_enum::Two, fp_type> : grid_config<dimension_enum::Two, fp_type>
{
  private:
    fp_type step_1_;
    fp_type step_2_;

    void initialize(pde_discretization_config_2d_ptr<fp_type> const &discretization_config) override
    {
        init_1_ = discretization_config->space_range().first.lower();
        init_2_ = discretization_config->space_range().second.lower();
        step_1_ = discretization_config->space_step().first;
        step_2_ = discretization_config->space_step().second;
        grid_1_ = std::make_shared<uniform_grid_config<dimension_enum::One, fp_type>>(
            discretization_config->pde_discretization_1());
        grid_2_ = std::make_shared<uniform_grid_config<dimension_enum::One, fp_type>>(
            discretization_config->pde_discretization_2());
    }

  public:
    uniform_grid_config() = delete;

    explicit uniform_grid_config(pde_discretization_config_2d_ptr<fp_type> const &discretization_config)
        : grid_config<dimension_enum::Two, fp_type>(discretization_config)
    {
        initialize(discretization_config);
    }

    inline fp_type gen_function_1(std::size_t t) override
    {
        return (static_cast<fp_type>(t) * step_1_);
    }

    inline fp_type gen_function_2(std::size_t t) override
    {
        return (static_cast<fp_type>(t) * step_2_);
    }
};

template <typename fp_type> using uniform_grid_config_1d = uniform_grid_config<dimension_enum::One, fp_type>;
template <typename fp_type> using uniform_grid_config_2d = uniform_grid_config<dimension_enum::Two, fp_type>;
template <typename fp_type> using uniform_grid_config_1d_ptr = sptr_t<uniform_grid_config_1d<fp_type>>;
template <typename fp_type> using uniform_grid_config_2d_ptr = sptr_t<uniform_grid_config_2d<fp_type>>;
////////////////////////////////////////////////////////////////////////////////////

/**
    nonuniform_grid_config object
 */
template <dimension_enum dimension, typename fp_type> struct nonuniform_grid_config
{
};

/**
    1D nonuniform_grid_config structure
 */
template <typename fp_type>
struct nonuniform_grid_config<dimension_enum::One, fp_type> : grid_config<dimension_enum::One, fp_type>
{
  private:
    fp_type c_[2];
    fp_type step_;
    fp_type strike_;
    fp_type p_scale_;
    fp_type c_scale_;

    void initialize(pde_discretization_config_1d_ptr<fp_type> const &discretization_config) override
    {
        auto const one = static_cast<fp_type>(1.0);
        auto const alpha = c_scale_ / p_scale_;
        auto const lower = discretization_config->space_range().lower();
        auto const upper = discretization_config->space_range().upper();
        step_ = one / discretization_config->number_of_space_points();
        init_ = strike_;
        c_[0] = std::asinh((upper - strike_) / alpha);
        c_[1] = std::asinh((lower - strike_) / alpha);
    }

  public:
    nonuniform_grid_config() = delete;

    explicit nonuniform_grid_config(pde_discretization_config_1d_ptr<fp_type> const &discretization_config,
                                    fp_type strike, fp_type p_scale, fp_type c_scale)
        : grid_config<dimension_enum::One, fp_type>(discretization_config), strike_{strike}, p_scale_{p_scale},
          c_scale_{c_scale}
    {
        initialize(discretization_config);
    }

    inline fp_type gen_function(std::size_t t) override
    {
        auto const alpha = c_scale_ / p_scale_;
        auto const one = static_cast<fp_type>(1.0);
        auto const zeta = static_cast<fp_type>(t) * step_;
        return (alpha * std::sinh(zeta * c_[0] + (one - zeta) * c_[1]));
    }
};

/**
    2D nonuniform_grid_config structure
 */
template <typename fp_type>
struct nonuniform_grid_config<dimension_enum::Two, fp_type> : grid_config<dimension_enum::Two, fp_type>
{
  private:
    fp_type c_[3];
    fp_type step_1_;
    fp_type step_2_;
    fp_type strike_;
    fp_type p_scale_;
    fp_type c_scale_;
    fp_type d_scale_;

    void initialize(pde_discretization_config_2d_ptr<fp_type> const &discretization_config) override
    {
        auto const one = static_cast<fp_type>(1.0);
        auto const alpha = c_scale_ / p_scale_;
        auto const lower_1 = discretization_config->space_range().first.lower();
        auto const upper_1 = discretization_config->space_range().first.upper();
        auto const upper_2 = discretization_config->space_range().second.upper();
        c_[0] = std::asinh((upper_1 - strike_) / alpha);
        c_[1] = std::asinh((lower_1 - strike_) / alpha);
        init_1_ = strike_;
        init_2_ = static_cast<fp_type>(0.0);
        step_1_ = one / discretization_config->number_of_space_points().first;
        step_2_ = one / discretization_config->number_of_space_points().second;
        c_[2] = std::asinh(upper_2 / d_scale_);

        grid_1_ = std::make_shared<nonuniform_grid_config<dimension_enum::One, fp_type>>(
            discretization_config->pde_discretization_1(), strike_, p_scale_, d_scale_);
        grid_2_ = std::make_shared<nonuniform_grid_config<dimension_enum::One, fp_type>>(
            discretization_config->pde_discretization_2(), strike_, p_scale_, d_scale_);
    }

  public:
    nonuniform_grid_config() = delete;

    explicit nonuniform_grid_config(pde_discretization_config_2d_ptr<fp_type> const &discretization_config,
                                    fp_type strike, fp_type p_scale, fp_type c_scale, fp_type d_scale)
        : grid_config<dimension_enum::Two, fp_type>(discretization_config), strike_{strike}, p_scale_{p_scale},
          c_scale_{c_scale}, d_scale_{d_scale}
    {
        initialize(discretization_config);
    }

    inline fp_type gen_function_1(std::size_t t) override
    {
        auto const alpha = c_scale_ / p_scale_;
        auto const one = static_cast<fp_type>(1.0);
        auto const zeta = static_cast<fp_type>(t) * step_1_;
        return (alpha * std::sinh(zeta * c_[0] + (one - zeta) * c_[1]));
    }

    inline fp_type gen_function_2(std::size_t t) override
    {
        auto const zeta = static_cast<fp_type>(t) * step_2_;
        return (d_scale_ * std::sinh(zeta * c_[2]));
    }
};

template <typename fp_type> using nonuniform_grid_config_1d = nonuniform_grid_config<dimension_enum::One, fp_type>;
template <typename fp_type> using nonuniform_grid_config_2d = nonuniform_grid_config<dimension_enum::Two, fp_type>;
template <typename fp_type> using nonuniform_grid_config_2d_ptr = sptr_t<nonuniform_grid_config_1d<fp_type>>;
template <typename fp_type> using nonuniform_grid_config_1d_ptr = sptr_t<nonuniform_grid_config_2d<fp_type>>;

} // namespace lss_grids
#endif ///_LSS_GRID_CONFIG_HPP_
