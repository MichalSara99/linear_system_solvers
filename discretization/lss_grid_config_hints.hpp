#if !defined(_LSS_GRID_CONFIG_HINTS_HPP_)
#define _LSS_GRID_CONFIG_HINTS_HPP_

#include "common/lss_enumerations.hpp"

namespace lss_grids
{
using lss_enumerations::dimension_enum;
using lss_enumerations::grid_enum;

/**
    grid_config_hints object
 */
template <dimension_enum dimension, typename fp_type> struct grid_config_hints
{
};

/**
    1D grid_config_hints structure
 */
template <typename fp_type> struct grid_config_hints<dimension_enum::One, fp_type>
{
  private:
    fp_type accumulation_point_;
    fp_type alpha_scale_;
    grid_enum grid_;

  public:
    explicit grid_config_hints(fp_type accumulation_point = fp_type(0.0), fp_type alpha_scale = fp_type(3.0),
                               grid_enum grid_type = grid_enum::Uniform)
        : accumulation_point_{accumulation_point}, alpha_scale_{alpha_scale}, grid_{grid_type}
    {
    }

    inline fp_type accumulation_point() const
    {
        return accumulation_point_;
    }

    inline fp_type alpha_scale() const
    {
        return alpha_scale_;
    }

    inline grid_enum grid() const
    {
        return grid_;
    }
};

/**
    2D grid_config_hints structure
 */
template <typename fp_type> struct grid_config_hints<dimension_enum::Two, fp_type>
{
  private:
    fp_type accumulation_point_;
    fp_type alpha_scale_;
    fp_type beta_scale_;
    grid_enum grid_;

  public:
    explicit grid_config_hints(fp_type accumulation_point = fp_type(0.0), fp_type alpha_scale = fp_type(3.0),
                               fp_type beta_scale = fp_type(50.0), grid_enum grid_type = grid_enum::Uniform)
        : accumulation_point_{accumulation_point}, alpha_scale_{alpha_scale}, beta_scale_{beta_scale}, grid_{grid_type}
    {
    }

    inline fp_type accumulation_point() const
    {
        return accumulation_point_;
    }

    inline fp_type alpha_scale() const
    {
        return alpha_scale_;
    }

    inline fp_type beta_scale() const
    {
        return beta_scale_;
    }

    inline grid_enum grid() const
    {
        return grid_;
    }
};

template <typename fp_type> using grid_config_hints_1d = grid_config_hints<dimension_enum::One, fp_type>;
template <typename fp_type> using grid_config_hints_2d = grid_config_hints<dimension_enum::Two, fp_type>;
template <typename fp_type> using grid_config_hints_1d_ptr = sptr_t<grid_config_hints_1d<fp_type>>;
template <typename fp_type> using grid_config_hints_2d_ptr = sptr_t<grid_config_hints_2d<fp_type>>;

///////////////////////////////////////////////////////////////////////////////////////

} // namespace lss_grids
#endif ///_LSS_GRID_CONFIG_HINTS_HPP_
