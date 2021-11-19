#if !defined(_LSS_GRID_HPP_)
#define _LSS_GRID_HPP_

#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include "lss_grid_config.hpp"
#include <functional>

namespace lss_grids
{
using lss_enumerations::dimension_enum;

///////////////////////////////////////////////////////////////////////////////////////

/**
    grid object
 */
template <dimension_enum dimension, typename fp_type> struct grid
{
};

/**
    1D uniform_grid structure
 */
template <typename fp_type> struct grid<dimension_enum::One, fp_type>
{
  public:
    static fp_type value(grid_config_1d_ptr<fp_type> const &grid_config, std::size_t const &idx)
    {
        return (grid_config->init() + grid_config->gen_function(idx));
    }
};

/**
    2D grid structure
 */
template <typename fp_type> struct grid<dimension_enum::Two, fp_type>
{
  public:
    static fp_type value_1(grid_config_2d_ptr<fp_type> const &grid_config, std::size_t const &idx)
    {
        return (grid_config->init_1() + grid_config->gen_function_1(idx));
    }

    static fp_type value_2(grid_config_2d_ptr<fp_type> const &grid_config, std::size_t const &idx)
    {
        return (grid_config->init_2() + grid_config->gen_function_2(idx));
    }
};

template <typename fp_type> using grid_1d = grid<dimension_enum::One, fp_type>;
template <typename fp_type> using grid_2d = grid<dimension_enum::Two, fp_type>;
template <typename fp_type> using grid_1d_ptr = sptr_t<grid_1d<fp_type>>;
template <typename fp_type> using grid_2d_ptr = sptr_t<grid_2d<fp_type>>;

} // namespace lss_grids
#endif ///_LSS_GRID_HPP_
