#if !defined(_LSS_GRID_CONFIG_HPP_)
#define _LSS_GRID_CONFIG_HPP_

#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include "lss_discretization.hpp"
#include <cmath>
#include <functional>

namespace lss_pde_solvers
{
using lss_enumerations::dimension_enum;
using lss_utility::range;

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
    fp_type step_;

    virtual void initialize(pde_discretization_config_1d_ptr<fp_type> const &discretization_config) = 0;

  public:
    explicit grid_config(pde_discretization_config_1d_ptr<fp_type> const &discretization_config)
    {
        initialize(discretization_config);
    }

    inline fp_type init() const
    {
        return init_;
    }

    inline fp_type step() const
    {
        return step_;
    }
};

/**
    2D grid_config structure
 */
template <typename fp_type> struct grid_config<dimension_enum::Two, fp_type>
{
  protected:
    fp_type init_1_;
    fp_type init_2_;
    fp_type step_1_;
    fp_type step_2_;

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

    inline fp_type const &step_1() const
    {
        return step_1_;
    }

    inline fp_type const &step_2() const
    {
        return step_2_;
    }
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
};

/**
    2D uniform_grid_config structure
 */
template <typename fp_type>
struct uniform_grid_config<dimension_enum::Two, fp_type> : grid_config<dimension_enum::Two, fp_type>
{
  private:
    void initialize(pde_discretization_config_2d_ptr<fp_type> const &discretization_config) override
    {
        init_1_ = discretization_config->space_range().first.lower();
        init_2_ = discretization_config->space_range().second.lower();
        step_1_ = discretization_config->space_step().first;
        step_2_ = discretization_config->space_step().second;
    }

  public:
    uniform_grid_config() = delete;

    explicit uniform_grid_config(pde_discretization_config_2d_ptr<fp_type> const &discretization_config)
        : grid_config<dimension_enum::Two, fp_type>(discretization_config)
    {
        initialize(discretization_config);
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
    fp_type upper_;
    fp_type strike_;
    fp_type p_scale_;
    fp_type c_scale_;

    void initialize(pde_discretization_config_1d_ptr<fp_type> const &discretization_config) override
    {
        const fp_type one = static_cast<fp_type>(1.0);
        init_ = discretization_config->space_range().lower();
        upper_ = discretization_config->space_range().upper();
        step_ = (one / static_cast<fp_type>(discretization_config->number_of_space_points() - 1)) *
                (std::asinh((upper_ - strike_) * (p_scale_ / c_scale_)) - std::asinh(-strike_ * (p_scale_ / c_scale_)));
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
};

/**
    2D nonuniform_grid_config structure
 */
template <typename fp_type>
struct nonuniform_grid_config<dimension_enum::Two, fp_type> : grid_config<dimension_enum::Two, fp_type>
{
  private:
    fp_type upper_1_;
    fp_type upper_2_;
    fp_type strike_;
    fp_type p_scale_;
    fp_type c_scale_;
    fp_type d_scale_;

    void initialize(pde_discretization_config_2d_ptr<fp_type> const &discretization_config) override
    {
        const fp_type one = static_cast<fp_type>(1.0);
        init_1_ = discretization_config->space_range().first.lower();
        upper_1_ = discretization_config->space_range().first.upper();
        init_2_ = discretization_config->space_range().second.lower();
        upper_2_ = discretization_config->space_range().second.upper();
        step_1_ =
            (one / static_cast<fp_type>(discretization_config->number_of_space_points().first - 1)) *
            (std::asinh((upper_1_ - strike_) * (p_scale_ / c_scale_)) - std::asinh(-strike_ * (p_scale_ / c_scale_)));
        step_2_ = (one / static_cast<fp_type>(discretization_config->number_of_space_points().second - 1)) *
                  (std::asinh(upper_2_ / d_scale_));
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
};

template <typename fp_type> using nonuniform_grid_config_1d = nonuniform_grid_config<dimension_enum::One, fp_type>;
template <typename fp_type> using nonuniform_grid_config_2d = nonuniform_grid_config<dimension_enum::Two, fp_type>;
template <typename fp_type> using nonuniform_grid_config_2d_ptr = sptr_t<nonuniform_grid_config_1d<fp_type>>;
template <typename fp_type> using nonuniform_grid_config_1d_ptr = sptr_t<nonuniform_grid_config_2d<fp_type>>;

} // namespace lss_pde_solvers
#endif ///_LSS_GRID_CONFIG_HPP_
