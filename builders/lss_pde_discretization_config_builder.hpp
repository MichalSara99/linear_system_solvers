#if !defined(_LSS_PDE_DISCRETIZATION_CONFIG_BUILDER_HPP_)
#define _LSS_PDE_DISCRETIZATION_CONFIG_BUILDER_HPP_

#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include "pde_solvers/lss_pde_discretization_config.hpp"

namespace lss_pde_solvers
{

using lss_enumerations::dimension_enum;
using lss_utility::range;
using lss_utility::sptr_t;

template <dimension_enum dimension, typename fp_type> struct pde_discretization_config_builder
{
};

/**
    1D pde_discretization_config_builder structure
 */
template <typename fp_type> struct pde_discretization_config_builder<dimension_enum::One, fp_type>
{
  private:
    range<fp_type> space_range_;
    std::size_t number_of_space_points_;
    range<fp_type> time_range_;
    std::size_t number_of_time_points_;

  public:
    pde_discretization_config_builder &space_range(range<fp_type> const &space_range)
    {
        space_range_ = space_range;
        return *this;
    }

    pde_discretization_config_builder &number_of_space_points(std::size_t const &number_of_space_points)
    {
        number_of_space_points_ = number_of_space_points;
        return *this;
    }

    pde_discretization_config_builder &time_range(range<fp_type> const &time_range)
    {
        time_range_ = time_range;
        return *this;
    }

    pde_discretization_config_builder &number_of_time_points(std::size_t const &number_of_time_points)
    {
        number_of_time_points_ = number_of_time_points;
        return *this;
    }

    pde_discretization_config_1d_ptr<fp_type> build()
    {
        return std::make_shared<pde_discretization_config<dimension_enum::One, fp_type>>(
            space_range_, number_of_space_points_, time_range_, number_of_time_points_);
    }
};

/**
    2D pde_discretization_config_builder structure
 */
template <typename fp_type> struct pde_discretization_config_builder<dimension_enum::Two, fp_type>
{
  private:
    range<fp_type> space_range_1_;
    range<fp_type> space_range_2_;
    std::size_t number_of_space_points_1_;
    std::size_t number_of_space_points_2_;
    range<fp_type> time_range_;
    std::size_t number_of_time_points_;

  public:
    pde_discretization_config_builder &space_range_1(range<fp_type> const &space_range_1)
    {
        space_range_1_ = space_range_1;
        return *this;
    }

    pde_discretization_config_builder &space_range_2(range<fp_type> const &space_range_2)
    {
        space_range_2_ = space_range_2;
        return *this;
    }

    pde_discretization_config_builder &number_of_space_points_1(std::size_t const &number_of_space_points_1)
    {
        number_of_space_points_1_ = number_of_space_points_1;
        return *this;
    }

    pde_discretization_config_builder &number_of_space_points_2(std::size_t const &number_of_space_points_2)
    {
        number_of_space_points_2_ = number_of_space_points_2;
        return *this;
    }

    pde_discretization_config_builder &time_range(range<fp_type> const &time_range)
    {
        time_range_ = time_range;
        return *this;
    }

    pde_discretization_config_builder &number_of_time_points(std::size_t const &number_of_time_points)
    {
        number_of_time_points_ = number_of_time_points;
        return *this;
    }

    pde_discretization_config_2d_ptr<fp_type> build()
    {
        return std::make_shared<pde_discretization_config<dimension_enum::Two, fp_type>>(
            space_range_1_, space_range_2_, number_of_space_points_1_, number_of_space_points_2_, time_range_,
            number_of_time_points_);
    }
};

template <typename fp_type>
using pde_discretization_config_1d_builder = pde_discretization_config_builder<dimension_enum::One, fp_type>;

template <typename fp_type>
using pde_discretization_config_2d_builder = pde_discretization_config_builder<dimension_enum::Two, fp_type>;

template <typename fp_type>
using pde_discretization_config_1d_builder_ptr =
    sptr_t<pde_discretization_config_builder<dimension_enum::One, fp_type>>;

template <typename fp_type>
using pde_discretization_config_2d_builder_ptr =
    sptr_t<pde_discretization_config_builder<dimension_enum::Two, fp_type>>;

} // namespace lss_pde_solvers

#endif ///_LSS_PDE_DISCRETIZATION_CONFIG_BUILDER_HPP_
