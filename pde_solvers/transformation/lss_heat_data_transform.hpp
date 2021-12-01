#if !defined(_LSS_HEAT_DATA_TRANSFORM_HPP_)
#define _LSS_HEAT_DATA_TRANSFORM_HPP_

#include <functional>

#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "common/lss_utility.hpp"
#include "discretization/lss_grid.hpp"
#include "discretization/lss_grid_transform_config.hpp"
#include "pde_solvers/lss_heat_data_config.hpp"

namespace lss_pde_solvers
{

using lss_enumerations::dimension_enum;
using lss_grids::grid_1d;
using lss_grids::grid_2d;
using lss_grids::grid_transform_config_1d_ptr;
using lss_grids::grid_transform_config_2d_ptr;
using lss_utility::range;
using lss_utility::sptr_t;

// ============================================================================
// ============================= heat_data_transform ==========================
// ============================================================================
template <dimension_enum dimension, typename fp_type> struct heat_data_transform
{
};

/**
    1D heat_data_transform structure
 */
template <typename fp_type> struct heat_data_transform<dimension_enum::One, fp_type>
{
  private:
    bool is_heat_source_set_{false};
    std::function<fp_type(fp_type, fp_type)> a_coeff_{nullptr};
    std::function<fp_type(fp_type, fp_type)> b_coeff_{nullptr};
    std::function<fp_type(fp_type, fp_type)> c_coeff_{nullptr};
    std::function<fp_type(fp_type)> init_coeff_{nullptr};
    std::function<fp_type(fp_type, fp_type)> src_coeff_{nullptr};

    void initialize(heat_data_config_1d_ptr<fp_type> const &heat_data_config,
                    grid_transform_config_1d_ptr<fp_type> const grid_transform_config)
    {
        auto const A = heat_data_config->a_coefficient();
        auto const B = heat_data_config->b_coefficient();
        auto const C = heat_data_config->c_coefficient();
        auto const a = grid_transform_config->a_derivative();
        auto const b = grid_transform_config->b_derivative();
        auto const init = heat_data_config->initial_condition();
        auto const &src_cfg = heat_data_config->source_data_config();
        std::function<fp_type(fp_type, fp_type)> src = nullptr;
        if (src_cfg)
        {
            src = src_cfg->heat_source();
            is_heat_source_set_ = true;
            src_coeff_ = [=](fp_type t, fp_type zeta) {
                auto const x = grid_1d<fp_type>::transformed_value(grid_transform_config, zeta);
                return src(t, x);
            };
        }
        a_coeff_ = [=](fp_type t, fp_type zeta) {
            auto const x = grid_1d<fp_type>::transformed_value(grid_transform_config, zeta);
            return (A(t, x) / (a(zeta) * a(zeta)));
        };

        b_coeff_ = [=](fp_type t, fp_type zeta) {
            auto const x = grid_1d<fp_type>::transformed_value(grid_transform_config, zeta);
            auto const a_val = a(zeta);
            auto const first = B(t, x) / a_val;
            auto const second = (A(t, x) * b(zeta)) / (a_val * a_val * a_val);
            return (first - second);
        };

        c_coeff_ = [=](fp_type t, fp_type zeta) {
            auto const x = grid_1d<fp_type>::transformed_value(grid_transform_config, zeta);
            return C(t, x);
        };

        init_coeff_ = [=](fp_type zeta) {
            auto const x = grid_1d<fp_type>::transformed_value(grid_transform_config, zeta);
            return init(x);
        };
    }

    explicit heat_data_transform() = delete;

  public:
    explicit heat_data_transform(heat_data_config_1d_ptr<fp_type> const &heat_data_config,
                                 grid_transform_config_1d_ptr<fp_type> const grid_transform_config)
    {
        initialize(heat_data_config, grid_transform_config);
    }

    ~heat_data_transform()
    {
    }

    bool const &is_heat_source_set() const
    {
        return is_heat_source_set_;
    }

    std::function<fp_type(fp_type, fp_type)> heat_source() const
    {
        return (is_heat_source_set() == true) ? src_coeff_ : nullptr;
    }

    std::function<fp_type(fp_type)> const &initial_condition() const
    {
        return init_coeff_;
    }

    std::function<fp_type(fp_type, fp_type)> const &a_coefficient() const
    {
        return a_coeff_;
    }

    std::function<fp_type(fp_type, fp_type)> const &b_coefficient() const
    {
        return b_coeff_;
    }

    std::function<fp_type(fp_type, fp_type)> const &c_coefficient() const
    {
        return c_coeff_;
    }
};

/**
    2D heat_data_transform structure
 */
template <typename fp_type> struct heat_data_transform<dimension_enum::Two, fp_type>
{
  private:
    bool is_heat_source_set_{false};
    std::function<fp_type(fp_type, fp_type, fp_type)> a_coeff_{nullptr};
    std::function<fp_type(fp_type, fp_type, fp_type)> b_coeff_{nullptr};
    std::function<fp_type(fp_type, fp_type, fp_type)> c_coeff_{nullptr};
    std::function<fp_type(fp_type, fp_type, fp_type)> d_coeff_{nullptr};
    std::function<fp_type(fp_type, fp_type, fp_type)> e_coeff_{nullptr};
    std::function<fp_type(fp_type, fp_type, fp_type)> f_coeff_{nullptr};
    std::function<fp_type(fp_type, fp_type)> init_coeff_{nullptr};
    std::function<fp_type(fp_type, fp_type, fp_type)> src_coeff_{nullptr};

    void initialize(heat_data_config_2d_ptr<fp_type> const &heat_data_config,
                    grid_transform_config_2d_ptr<fp_type> const &grid_transform_config)
    {
        auto const A = heat_data_config->a_coefficient();
        auto const B = heat_data_config->b_coefficient();
        auto const C = heat_data_config->c_coefficient();
        auto const D = heat_data_config->d_coefficient();
        auto const E = heat_data_config->e_coefficient();
        auto const F = heat_data_config->f_coefficient();
        auto const a = grid_transform_config->a_derivative();
        auto const b = grid_transform_config->b_derivative();
        auto const c = grid_transform_config->c_derivative();
        auto const d = grid_transform_config->d_derivative();
        auto const init = heat_data_config->initial_condition();
        auto const &src_cfg = heat_data_config->source_data_config();
        std::function<fp_type(fp_type, fp_type, fp_type)> src = nullptr;
        if (src_cfg)
        {
            src = src_cfg->heat_source();
            is_heat_source_set_ = true;
            src_coeff_ = [=](fp_type t, fp_type zeta, fp_type eta) {
                auto const x = grid_2d<fp_type>::transformed_value_1(grid_transform_config, zeta);
                auto const y = grid_2d<fp_type>::transformed_value_2(grid_transform_config, eta);
                return src(t, x, y);
            };
        }
        a_coeff_ = [=](fp_type t, fp_type zeta, fp_type eta) {
            auto const x = grid_2d<fp_type>::transformed_value_1(grid_transform_config, zeta);
            auto const y = grid_2d<fp_type>::transformed_value_2(grid_transform_config, eta);
            return (A(t, x, y) / (a(zeta) * a(zeta)));
        };

        b_coeff_ = [=](fp_type t, fp_type zeta, fp_type eta) {
            auto const x = grid_2d<fp_type>::transformed_value_1(grid_transform_config, zeta);
            auto const y = grid_2d<fp_type>::transformed_value_2(grid_transform_config, eta);
            return (B(t, x, y) / (b(eta) * b(eta)));
        };

        c_coeff_ = [=](fp_type t, fp_type zeta, fp_type eta) {
            auto const x = grid_2d<fp_type>::transformed_value_1(grid_transform_config, zeta);
            auto const y = grid_2d<fp_type>::transformed_value_2(grid_transform_config, eta);
            return (C(t, x, y) / (a(zeta) * b(eta)));
        };

        d_coeff_ = [=](fp_type t, fp_type zeta, fp_type eta) {
            auto const x = grid_2d<fp_type>::transformed_value_1(grid_transform_config, zeta);
            auto const y = grid_2d<fp_type>::transformed_value_2(grid_transform_config, eta);
            auto const a_val = a(zeta);
            auto const first = D(t, x, y) / a_val;
            auto const second = (A(t, x, y) * c(zeta)) / (a_val * a_val * a_val);
            return (first - second);
        };

        e_coeff_ = [=](fp_type t, fp_type zeta, fp_type eta) {
            auto const x = grid_2d<fp_type>::transformed_value_1(grid_transform_config, zeta);
            auto const y = grid_2d<fp_type>::transformed_value_2(grid_transform_config, eta);
            auto const b_val = b(eta);
            auto const first = E(t, x, y) / b_val;
            auto const second = (B(t, x, y) * d(eta)) / (b_val * b_val * b_val);
            return (first - second);
        };

        f_coeff_ = [=](fp_type t, fp_type zeta, fp_type eta) {
            auto const x = grid_2d<fp_type>::transformed_value_1(grid_transform_config, zeta);
            auto const y = grid_2d<fp_type>::transformed_value_2(grid_transform_config, eta);
            return F(t, x, y);
        };

        init_coeff_ = [=](fp_type zeta, fp_type eta) {
            auto const x = grid_2d<fp_type>::transformed_value_1(grid_transform_config, zeta);
            auto const y = grid_2d<fp_type>::transformed_value_2(grid_transform_config, eta);
            return init(x, y);
        };
    }

    explicit heat_data_transform() = delete;

  public:
    explicit heat_data_transform(heat_data_config_2d_ptr<fp_type> const &heat_data_config,
                                 grid_transform_config_2d_ptr<fp_type> const &grid_transform_config)
    {
        initialize(heat_data_config, grid_transform_config);
    }

    ~heat_data_transform()
    {
    }

    bool const &is_heat_source_set() const
    {
        return is_heat_source_set_;
    }

    std::function<fp_type(fp_type, fp_type, fp_type)> heat_source() const
    {
        return (is_heat_source_set() == true) ? src_coeff_ : nullptr;
    }

    std::function<fp_type(fp_type, fp_type)> const &initial_condition() const
    {
        return init_coeff_;
    }

    std::function<fp_type(fp_type, fp_type, fp_type)> const &a_coefficient() const
    {
        return a_coeff_;
    }

    std::function<fp_type(fp_type, fp_type, fp_type)> const &b_coefficient() const
    {
        return b_coeff_;
    }

    std::function<fp_type(fp_type, fp_type, fp_type)> const &c_coefficient() const
    {
        return c_coeff_;
    }

    std::function<fp_type(fp_type, fp_type, fp_type)> const &d_coefficient() const
    {
        return d_coeff_;
    }

    std::function<fp_type(fp_type, fp_type, fp_type)> const &e_coefficient() const
    {
        return e_coeff_;
    }

    std::function<fp_type(fp_type, fp_type, fp_type)> const &f_coefficient() const
    {
        return f_coeff_;
    }
};

template <typename fp_type> using heat_data_transform_1d = heat_data_transform<dimension_enum::One, fp_type>;

template <typename fp_type> using heat_data_transform_2d = heat_data_transform<dimension_enum::Two, fp_type>;

template <typename fp_type>
using heat_data_transform_1d_ptr = sptr_t<heat_data_transform<dimension_enum::One, fp_type>>;

template <typename fp_type>
using heat_data_transform_2d_ptr = sptr_t<heat_data_transform<dimension_enum::Two, fp_type>>;

} // namespace lss_pde_solvers

#endif ///_LSS_HEAT_DATA_TRANSFORM_HPP_
