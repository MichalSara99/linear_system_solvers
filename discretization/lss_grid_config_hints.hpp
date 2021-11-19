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
    fp_type strike_;
    fp_type p_scale_;
    fp_type c_scale_;
    grid_enum grid_;

  public:
    explicit grid_config_hints(fp_type strike = fp_type(0.0), fp_type p_scale = fp_type(8.4216),
                               fp_type c_scale = fp_type(0.1), grid_enum grid_type = grid_enum::Uniform)
        : strike_{strike}, p_scale_{p_scale}, c_scale_{c_scale}, grid_{grid_type}
    {
    }

    inline fp_type strike() const
    {
        return strike_;
    }

    inline fp_type p_scale() const
    {
        return p_scale_;
    }

    inline fp_type c_scale() const
    {
        return c_scale_;
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
    fp_type strike_;
    fp_type p_scale_;
    fp_type c_scale_;
    fp_type d_scale_;
    grid_enum grid_;

  public:
    explicit grid_config_hints(fp_type strike = fp_type(0.0), fp_type p_scale = fp_type(8.4216),
                               fp_type c_scale = fp_type(0.1), fp_type d_scale = fp_type(2.0),
                               grid_enum grid_type = grid_enum::Uniform)
        : strike_{strike}, p_scale_{p_scale}, c_scale_{c_scale}, d_scale_{d_scale}, grid_{grid_type}
    {
    }

    inline fp_type strike() const
    {
        return strike_;
    }

    inline fp_type p_scale() const
    {
        return p_scale_;
    }

    inline fp_type c_scale() const
    {
        return c_scale_;
    }

    inline fp_type d_scale() const
    {
        return d_scale_;
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
