#if !defined(_LSS_GRID_CONFIG_HINTS_BUILDER_HPP_)
#define _LSS_GRID_CONFIG_HINTS_BUILDER_HPP_

#include <functional>

#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "common/lss_utility.hpp"
#include "discretization/lss_grid_config_hints.hpp"

namespace lss_grids
{

using lss_enumerations::dimension_enum;
using lss_enumerations::grid_enum;
using lss_utility::range;
using lss_utility::sptr_t;

// ============================================================================
// ======================== grid_config_hints_builder =========================
// ============================================================================
template <dimension_enum dimension, typename fp_type> struct grid_config_hints_builder
{
};

/**
    1D grid_config_hints_builder structure
 */
template <typename fp_type> struct grid_config_hints_builder<dimension_enum::One, fp_type>
{
  private:
    fp_type accumulation_point_;
    fp_type alpha_scale_;
    grid_enum grid_;

  public:
    grid_config_hints_builder &accumulation_point(fp_type accumulation_point)
    {
        accumulation_point_ = accumulation_point;
        return *this;
    }

    grid_config_hints_builder &alpha_scale(fp_type alpha_scale)
    {
        alpha_scale_ = alpha_scale;
        return *this;
    }

    grid_config_hints_builder &grid(grid_enum grid)
    {
        grid_ = grid;
        return *this;
    }

    grid_config_hints_1d_ptr<fp_type> build()
    {
        return std::make_shared<grid_config_hints_1d<fp_type>>(accumulation_point_, alpha_scale_, grid_);
    }
};

/**
    2D grid_config_hints_builder structure
 */
template <typename fp_type> struct grid_config_hints_builder<dimension_enum::Two, fp_type>
{
  private:
    fp_type accumulation_point_;
    fp_type alpha_scale_;
    fp_type beta_scale_;
    grid_enum grid_;

  public:
    grid_config_hints_builder &accumulation_point(fp_type accumulation_point)
    {
        accumulation_point_ = accumulation_point;
        return *this;
    }

    grid_config_hints_builder &alpha_scale(fp_type alpha_scale)
    {
        alpha_scale_ = alpha_scale;
        return *this;
    }

    grid_config_hints_builder &beta_scale(fp_type beta_scale)
    {
        beta_scale_ = beta_scale;
        return *this;
    }

    grid_config_hints_builder &grid(grid_enum grid)
    {
        grid_ = grid;
        return *this;
    }

    grid_config_hints_2d_ptr<fp_type> build()
    {
        return std::make_shared<grid_config_hints_2d<fp_type>>(accumulation_point_, alpha_scale_, beta_scale_, grid_);
    }
};

template <typename fp_type>
using grid_config_hints_1d_builder = grid_config_hints_builder<dimension_enum::One, fp_type>;

template <typename fp_type>
using grid_config_hints_2d_builder = grid_config_hints_builder<dimension_enum::Two, fp_type>;

template <typename fp_type> using grid_config_hints_1d_builder_ptr = sptr_t<grid_config_hints_1d_builder<fp_type>>;

template <typename fp_type> using grid_config_hints_2d_builder_ptr = sptr_t<grid_config_hints_2d_builder<fp_type>>;

} // namespace lss_grids

#endif ///_LSS_GRID_CONFIG_HINTS_BUILDER_HPP_
