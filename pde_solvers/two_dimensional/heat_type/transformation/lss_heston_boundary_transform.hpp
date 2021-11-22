#if !defined(_LSS_HESTON_BOUNDARY_TRANSFORM_HPP_)
#define _LSS_HESTON_BOUNDARY_TRANSFORM_HPP_

#include <functional>

#include "boundaries/lss_boundary.hpp"
#include "boundaries/lss_dirichlet_boundary.hpp"
#include "boundaries/lss_neumann_boundary.hpp"
#include "common/lss_macros.hpp"
#include "common/lss_utility.hpp"
#include "discretization/lss_grid.hpp"
#include "discretization/lss_grid_transform_config.hpp"
#include "pde_solvers/lss_heat_data_config.hpp"

namespace lss_pde_solvers
{

using lss_boundary::boundary_2d_pair;
using lss_boundary::boundary_2d_ptr;
using lss_boundary::dirichlet_boundary_2d;
using lss_boundary::neumann_boundary_2d;
using lss_grids::grid_1d;
using lss_grids::grid_2d;
using lss_grids::grid_transform_config_2d_ptr;
using lss_utility::sptr_t;

/**
    heston_boundary_transform structure
 */
template <typename fp_type> struct heston_boundary_transform
{
  private:
    boundary_2d_ptr<fp_type> v_upper_ptr_;
    boundary_2d_pair<fp_type> h_pair_ptr_;

    void initialize(boundary_2d_ptr<fp_type> const &vertical_upper_boundary_ptr,
                    boundary_2d_pair<fp_type> const &horizontal_boundary_pair,
                    grid_transform_config_2d_ptr<fp_type> const grid_transform_config)
    {
        LSS_VERIFY(grid_transform_config, "grid_transform_config must not be null");
        auto const one = static_cast<fp_type>(1.0);
        auto const &v_upper_orig = vertical_upper_boundary_ptr;
        auto const &h_lower_orig = std::get<0>(horizontal_boundary_pair);
        auto const &h_upper_orig = std::get<1>(horizontal_boundary_pair);
        auto const &a_der = grid_transform_config->a_derivative();

        // transform vertical upper:
        auto const v_upper_trans = [=](fp_type t, fp_type zeta) -> fp_type {
            auto const x = grid_2d<fp_type>::transformed_value_1(grid_transform_config, zeta);
            return v_upper_orig->value(t, x);
        };
        // transform both horizontal:
        // horizontal lower:
        auto const h_lower_trans = [=](fp_type t, fp_type eta) -> fp_type {
            auto const y = grid_2d<fp_type>::transformed_value_2(grid_transform_config, eta);
            return h_lower_orig->value(t, y);
        };
        // horizontal upper:
        auto const h_upper_trans = [=](fp_type t, fp_type eta) -> fp_type {
            auto const y = grid_2d<fp_type>::transformed_value_2(grid_transform_config, eta);
            return (h_upper_orig->value(t, y) * a_der(one));
        };
        v_upper_ptr_ = std::make_shared<dirichlet_boundary_2d<fp_type>>(v_upper_trans);
        h_pair_ptr_ = std::make_pair(std::make_shared<dirichlet_boundary_2d<fp_type>>(h_lower_trans),
                                     std::make_shared<neumann_boundary_2d<fp_type>>(h_upper_trans));
    }

    explicit heston_boundary_transform() = delete;

  public:
    explicit heston_boundary_transform(boundary_2d_ptr<fp_type> const &vertical_upper_boundary_ptr,
                                       boundary_2d_pair<fp_type> const &horizontal_boundary_pair,
                                       grid_transform_config_2d_ptr<fp_type> const grid_transform_config)
    {
        initialize(vertical_upper_boundary_ptr, horizontal_boundary_pair, grid_transform_config);
    }

    ~heston_boundary_transform()
    {
    }

    inline boundary_2d_ptr<fp_type> const &vertical_upper() const
    {
        return v_upper_ptr_;
    }

    inline boundary_2d_pair<fp_type> const &horizontal_pair() const
    {
        return h_pair_ptr_;
    }
};

template <typename fp_type> using heston_boundary_transform_ptr = sptr_t<heston_boundary_transform<fp_type>>;

} // namespace lss_pde_solvers

#endif ///_LSS_HESTON_BOUNDARY_TRANSFORM_HPP_
