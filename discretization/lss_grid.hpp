#if !defined(_LSS_GRID_HPP_)
#define _LSS_GRID_HPP_

#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include "lss_grid_config.hpp"
#include "lss_grid_transform_config.hpp"
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
        return grid_config->value_for(idx);
    }

    static fp_type step(grid_config_1d_ptr<fp_type> const &grid_config)
    {
        return grid_config->step();
    }

    static std::size_t index_of(grid_config_1d_ptr<fp_type> const &grid_config, fp_type zeta)
    {
        return grid_config->index_of(zeta);
    }

    static fp_type transformed_value(grid_transform_config_1d_ptr<fp_type> const &grid_trans_config, fp_type zeta)
    {
        return grid_trans_config->value_for(zeta);
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
        return grid_config->value_for_1(idx);
    }

    static fp_type value_2(grid_config_2d_ptr<fp_type> const &grid_config, std::size_t const &idx)
    {
        return grid_config->value_for_2(idx);
    }

    static fp_type step_1(grid_config_2d_ptr<fp_type> const &grid_config)
    {
        return grid_config->step_1();
    }

    static fp_type step_2(grid_config_2d_ptr<fp_type> const &grid_config)
    {
        return grid_config->step_2();
    }

    static std::size_t index_of_1(grid_config_2d_ptr<fp_type> const &grid_config, fp_type zeta)
    {
        return grid_config->index_of_1(zeta);
    }

    static std::size_t index_of_2(grid_config_2d_ptr<fp_type> const &grid_config, fp_type eta)
    {
        return grid_config->index_of_2(eta);
    }

    static fp_type transformed_value_1(grid_transform_config_2d_ptr<fp_type> const &grid_trans_config, fp_type zeta)
    {
        return grid_trans_config->value_for_1(zeta);
    }

    static fp_type transformed_value_2(grid_transform_config_2d_ptr<fp_type> const &grid_trans_config, fp_type eta)
    {
        return grid_trans_config->value_for_2(eta);
    }
};

template <typename fp_type> using grid_1d = grid<dimension_enum::One, fp_type>;
template <typename fp_type> using grid_2d = grid<dimension_enum::Two, fp_type>;
template <typename fp_type> using grid_1d_ptr = sptr_t<grid_1d<fp_type>>;
template <typename fp_type> using grid_2d_ptr = sptr_t<grid_2d<fp_type>>;

} // namespace lss_grids
#endif ///_LSS_GRID_HPP_
