#if !defined(_LSS_DISCRETIZATION_CONFIG_HPP_)
#define _LSS_DISCRETIZATION_CONFIG_HPP_

#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"

namespace lss_pde_solvers
{

using lss_enumerations::dimension_enum;
using lss_utility::range;
using lss_utility::sptr_t;

template <dimension_enum dimension, typename fp_type> struct pde_discretization_config
{
};

/**
    1D pde_discretization_config structure
 */
template <typename fp_type> struct pde_discretization_config<dimension_enum::One, fp_type>
{
  private:
    range<fp_type> space_range_;
    range<fp_type> time_range_;
    std::size_t number_of_space_points_;
    std::size_t number_of_time_points_;

    explicit pde_discretization_config() = delete;

  public:
    explicit pde_discretization_config(range<fp_type> const &space_range, std::size_t const &number_of_space_points,
                                       range<fp_type> const &time_range, std::size_t const &number_of_time_points)
        : space_range_{space_range}, number_of_space_points_{number_of_space_points}, time_range_{time_range},
          number_of_time_points_{number_of_time_points}
    {
    }
    ~pde_discretization_config()
    {
    }

    inline range<fp_type> const &space_range() const
    {
        return space_range_;
    }

    inline range<fp_type> const &time_range() const
    {
        return time_range_;
    }

    inline std::size_t number_of_space_points() const
    {
        return number_of_space_points_;
    }

    inline std::size_t number_of_time_points() const
    {
        return number_of_time_points_;
    }

    inline fp_type space_step() const
    {
        return ((space_range_.spread()) / static_cast<fp_type>(number_of_space_points_ - 1));
    }

    inline fp_type time_step() const
    {
        return ((time_range_.spread()) / static_cast<fp_type>(number_of_time_points_ - 1));
    }
};

/**
    2D pde_discretization_config structure
 */
template <typename fp_type> struct pde_discretization_config<dimension_enum::Two, fp_type>
{
  private:
    range<fp_type> space_range_1_;
    range<fp_type> space_range_2_;
    range<fp_type> time_range_;
    std::size_t number_of_space_points_1_;
    std::size_t number_of_space_points_2_;
    std::size_t number_of_time_points_;

    explicit pde_discretization_config() = delete;

  public:
    explicit pde_discretization_config(range<fp_type> const &space_range_1, range<fp_type> const &space_range_2,
                                       std::size_t const &number_of_space_points_1,
                                       std::size_t const &number_of_space_points_2, range<fp_type> const &time_range,
                                       std::size_t const &number_of_time_points)
        : space_range_1_{space_range_1}, space_range_2_{space_range_2},
          number_of_space_points_1_{number_of_space_points_1}, number_of_space_points_2_{number_of_space_points_2},
          time_range_{time_range}, number_of_time_points_{number_of_time_points}
    {
    }
    ~pde_discretization_config()
    {
    }

    inline std::pair<range<fp_type>, range<fp_type>> const &space_range() const
    {
        return std::make_pair(space_range_1_, space_range_2_);
    }

    inline range<fp_type> const &time_range() const
    {
        return time_range_;
    }

    inline std::pair<std::size_t, std::size_t> const number_of_space_points() const
    {
        return std::make_pair(number_of_space_points_1_, number_of_space_points_2_);
    }

    inline std::size_t number_of_time_points() const
    {
        return number_of_time_points_;
    }

    inline std::pair<fp_type, fp_type> space_step() const
    {
        return std::make_pair(((space_range_1_.spread()) / static_cast<fp_type>(number_of_space_points_1_ - 1)),
                              ((space_range_2_.spread()) / static_cast<fp_type>(number_of_space_points_2_ - 1)));
    }

    inline fp_type time_step() const
    {
        return ((time_range_.spread()) / static_cast<fp_type>(number_of_time_points_ - 1));
    }
};

template <typename fp_type>
using pde_discretization_config_1d = pde_discretization_config<dimension_enum::One, fp_type>;

template <typename fp_type>
using pde_discretization_config_2d = pde_discretization_config<dimension_enum::Two, fp_type>;

template <typename fp_type>
using pde_discretization_config_1d_ptr = sptr_t<pde_discretization_config<dimension_enum::One, fp_type>>;

template <typename fp_type>
using pde_discretization_config_2d_ptr = sptr_t<pde_discretization_config<dimension_enum::Two, fp_type>>;

} // namespace lss_pde_solvers

#endif ///_LSS_DISCRETIZATION_CONFIG_HPP_
