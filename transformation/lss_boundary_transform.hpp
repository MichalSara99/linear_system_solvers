#if !defined(_LSS_BOUNDARY_TRANSFORM_HPP_)
#define _LSS_BOUNDARY_TRANSFORM_HPP_

#include <functional>

#include "boundaries/lss_boundary.hpp"
#include "boundaries/lss_dirichlet_boundary.hpp"
#include "boundaries/lss_neumann_boundary.hpp"
#include "boundaries/lss_robin_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "common/lss_utility.hpp"
#include "discretization/lss_grid.hpp"
#include "discretization/lss_grid_transform_config.hpp"

namespace lss_transformation
{

using lss_boundary::boundary_1d_pair;
using lss_boundary::boundary_1d_ptr;
using lss_boundary::dirichlet_boundary_1d;
using lss_boundary::neumann_boundary_1d;
using lss_boundary::robin_boundary_1d;
using lss_enumerations::dimension_enum;
using lss_grids::grid_transform_config_1d_ptr;
using lss_utility::sptr_t;

// ============================================================================
// ============================= boundary_transform ===========================
// ============================================================================
template <dimension_enum dimension, typename fp_type> struct boundary_transform
{
};

/**
   1D boundary_transform structure
 */
template <typename fp_type> struct boundary_transform<dimension_enum::One, fp_type>
{
  private:
    boundary_1d_pair<fp_type> pair_ptr_;

    void initialize(boundary_1d_pair<fp_type> const &boundary_pair,
                    grid_transform_config_1d_ptr<fp_type> const grid_transform_config)
    {
        LSS_VERIFY(grid_transform_config, "grid_transform_config must not be null");
        auto const zero = static_cast<fp_type>(0.0);
        auto const one = static_cast<fp_type>(1.0);
        auto const &lower_orig = std::get<0>(boundary_pair);
        auto const &upper_orig = std::get<1>(boundary_pair);
        auto const &a_der = grid_transform_config->a_derivative();

        // first transform lower boundary:
        boundary_1d_ptr<fp_type> lower_ptr;
        if (auto const &ptr = std::dynamic_pointer_cast<dirichlet_boundary_1d<fp_type>>(lower_orig))
        {
            // for Dirichlet no transform is necessary:
            lower_ptr = ptr;
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(lower_orig))
        {
            auto const &lower_trans = [=](fp_type t) -> fp_type { return ptr->value(t) * a_der(zero); };
            lower_ptr = std::make_shared<neumann_boundary_1d<fp_type>>(lower_trans);
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(lower_orig))
        {
            auto const &lower_trans_lin = [=](fp_type t) -> fp_type { return ptr->linear_value(t) * a_der(zero); };
            auto const &lower_trans = [=](fp_type t) -> fp_type { return ptr->value(t) * a_der(zero); };
            lower_ptr = std::make_shared<robin_boundary_1d<fp_type>>(lower_trans_lin, lower_trans);
        }
        else
        {
            std::exception("Unreachable");
        }

        // second transform upper boundary:
        boundary_1d_ptr<fp_type> upper_ptr;
        if (auto const &ptr = std::dynamic_pointer_cast<dirichlet_boundary_1d<fp_type>>(upper_orig))
        {
            // for Dirichlet no transform is necessary:
            upper_ptr = ptr;
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(upper_orig))
        {
            auto const &upper_trans = [=](fp_type t) -> fp_type { return ptr->value(t) * a_der(one); };
            upper_ptr = std::make_shared<neumann_boundary_1d<fp_type>>(upper_trans);
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(upper_orig))
        {
            auto const &upper_trans_lin = [=](fp_type t) -> fp_type { return ptr->linear_value(t) * a_der(one); };
            auto const &upper_trans = [=](fp_type t) -> fp_type { return ptr->value(t) * a_der(one); };
            upper_ptr = std::make_shared<robin_boundary_1d<fp_type>>(upper_trans_lin, upper_trans);
        }
        else
        {
            std::exception("Unreachable");
        }
        pair_ptr_ = std::make_pair(lower_ptr, upper_ptr);
    }

    explicit boundary_transform() = delete;

  public:
    explicit boundary_transform(boundary_1d_pair<fp_type> const &boundary_pair,
                                grid_transform_config_1d_ptr<fp_type> const grid_transform_config)
    {
        initialize(boundary_pair, grid_transform_config);
    }

    ~boundary_transform()
    {
    }

    inline boundary_1d_pair<fp_type> const &boundary_pair() const
    {
        return pair_ptr_;
    }
};

template <typename fp_type> using boundary_transform_1d = boundary_transform<dimension_enum::One, fp_type>;

template <typename fp_type> using boundary_transform_1d_ptr = sptr_t<boundary_transform_1d<fp_type>>;

} // namespace lss_transformation

#endif ///_LSS_BOUNDARY_TRANSFORM_HPP_
