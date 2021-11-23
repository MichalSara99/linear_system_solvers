#if !defined(_LSS_ODE_DATA_TRANSFORM_HPP_)
#define _LSS_ODE_DATA_TRANSFORM_HPP_

#include <functional>

#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "common/lss_utility.hpp"
#include "discretization/lss_grid.hpp"
#include "discretization/lss_grid_transform_config.hpp"
#include "ode_solvers/lss_ode_data_config.hpp"

namespace lss_ode_solvers
{

using lss_grids::grid_1d;
using lss_grids::grid_2d;
using lss_grids::grid_transform_config_1d_ptr;
using lss_grids::grid_transform_config_2d_ptr;
using lss_utility::range;
using lss_utility::sptr_t;

/**
    1D heat_data_transform structure
 */
template <typename fp_type> struct ode_data_transform
{
  private:
    bool is_nonhom_data_set_{false};
    std::function<fp_type(fp_type)> a_coeff_{nullptr};
    std::function<fp_type(fp_type)> b_coeff_{nullptr};
    std::function<fp_type(fp_type)> nonhom_coeff_{nullptr};

    void initialize(ode_data_config_ptr<fp_type> const &ode_data_config,
                    grid_transform_config_1d_ptr<fp_type> const grid_transform_config)
    {
        auto const A = ode_data_config->a_coefficient();
        auto const B = ode_data_config->b_coefficient();
        auto const a = grid_transform_config->a_derivative();
        auto const b = grid_transform_config->b_derivative();
        auto const &nonhom_cfg = ode_data_config->nonhom_data_config();
        std::function<fp_type(fp_type)> nonhom = nullptr;
        if (nonhom_cfg)
        {
            nonhom = nonhom_cfg->nonhom_function();
            is_nonhom_data_set_ = true;
            nonhom_coeff_ = [=](fp_type zeta) {
                auto const x = grid_1d<fp_type>::transformed_value(grid_transform_config, zeta);
                auto const a_val = a(zeta);
                return (a_val * a_val * nonhom(x));
            };
        }
        a_coeff_ = [=](fp_type zeta) {
            auto const x = grid_1d<fp_type>::transformed_value(grid_transform_config, zeta);
            auto const first = a(zeta) * A(x);
            auto const second = b(zeta) / a(zeta);
            return (first - second);
        };

        b_coeff_ = [=](fp_type zeta) {
            auto const x = grid_1d<fp_type>::transformed_value(grid_transform_config, zeta);
            auto const a_val = a(zeta);
            return (a_val * a_val * B(x));
        };
    }

    explicit ode_data_transform() = delete;

  public:
    explicit ode_data_transform(ode_data_config_ptr<fp_type> const &ode_data_config,
                                grid_transform_config_1d_ptr<fp_type> const grid_transform_config)
    {
        initialize(ode_data_config, grid_transform_config);
    }

    ~ode_data_transform()
    {
    }

    bool const &is_nonhom_data_set() const
    {
        return is_nonhom_data_set_;
    }

    std::function<fp_type(fp_type)> nonhom_function() const
    {
        return (is_nonhom_data_set() == true) ? nonhom_coeff_ : nullptr;
    }

    std::function<fp_type(fp_type)> const &a_coefficient() const
    {
        return a_coeff_;
    }

    std::function<fp_type(fp_type)> const &b_coefficient() const
    {
        return b_coeff_;
    }
};

template <typename fp_type> using ode_data_transform_ptr = sptr_t<ode_data_transform<fp_type>>;

} // namespace lss_ode_solvers

#endif ///_LSS_ODE_DATA_TRANSFORM_HPP_
