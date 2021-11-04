#if !defined(_LSS_GRID_HPP_)
#define _LSS_GRID_HPP_

#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include "lss_grid_config.hpp"
#include <functional>

namespace lss_pde_solvers
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
        return (grid_config->init() + static_cast<fp_type>(idx) * grid_config->step());
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
        return (grid_config->init_1() + static_cast<fp_type>(idx) * grid_config->step_1());
    }

    static fp_type value_2(grid_config_2d_ptr<fp_type> const &grid_config, std::size_t const &idx)
    {
        return (grid_config->init_2() + static_cast<fp_type>(idx) * grid_config->step_2());
    }
};

template <typename fp_type> using grid_1d = grid<dimension_enum::One, fp_type>;
template <typename fp_type> using grid_2d = grid<dimension_enum::Two, fp_type>;
template <typename fp_type> using grid_1d_ptr = sptr_t<grid_1d<fp_type>>;
template <typename fp_type> using grid_2d_ptr = sptr_t<grid_2d<fp_type>>;

} // namespace lss_pde_solvers
#endif ///_LSS_GRID_HPP_
