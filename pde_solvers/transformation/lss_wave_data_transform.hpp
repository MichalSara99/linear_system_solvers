#if !defined(_LSS_WAVE_DATA_TRANSFORM_HPP_)
#define _LSS_WAVE_DATA_TRANSFORM_HPP_

#include <functional>

#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "common/lss_utility.hpp"
#include "discretization/lss_grid.hpp"
#include "discretization/lss_grid_transform_config.hpp"
#include "pde_solvers/lss_wave_data_config.hpp"

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
// ============================= wave_data_transform ==========================
// ============================================================================
template <dimension_enum dimension, typename fp_type> struct wave_data_transform
{
};

/**
    1D wave_data_transform structure
 */
template <typename fp_type> struct wave_data_transform<dimension_enum::One, fp_type>
{
  private:
    bool is_wave_source_set_{false};
    std::function<fp_type(fp_type)> a_coeff_{nullptr};
    std::function<fp_type(fp_type)> b_coeff_{nullptr};
    std::function<fp_type(fp_type)> c_coeff_{nullptr};
    std::function<fp_type(fp_type)> d_coeff_{nullptr};
    std::function<fp_type(fp_type)> init_first_coeff_{nullptr};
    std::function<fp_type(fp_type)> init_second_coeff_{nullptr};
    std::function<fp_type(fp_type, fp_type)> src_coeff_{nullptr};

    void initialize(wave_data_config_1d_ptr<fp_type> const &wave_data_config,
                    grid_transform_config_1d_ptr<fp_type> const grid_transform_config)
    {
        auto const A = wave_data_config->a_coefficient();
        auto const B = wave_data_config->b_coefficient();
        auto const C = wave_data_config->c_coefficient();
        auto const D = wave_data_config->d_coefficient();
        auto const a = grid_transform_config->a_derivative();
        auto const b = grid_transform_config->b_derivative();
        auto const init_first = wave_data_config->first_initial_condition();
        auto const init_second = wave_data_config->second_initial_condition();
        auto const &src_cfg = wave_data_config->source_data_config();
        std::function<fp_type(fp_type, fp_type)> src = nullptr;
        if (src_cfg)
        {
            src = src_cfg->wave_source();
            is_wave_source_set_ = true;
            src_coeff_ = [=](fp_type t, fp_type zeta) {
                auto const x = grid_1d<fp_type>::transformed_value(grid_transform_config, zeta);
                return src(t, x);
            };
        }

        a_coeff_ = [=](fp_type zeta) {
            auto const x = grid_1d<fp_type>::transformed_value(grid_transform_config, zeta);
            return A(x);
        };

        b_coeff_ = [=](fp_type zeta) {
            auto const x = grid_1d<fp_type>::transformed_value(grid_transform_config, zeta);
            return (B(x) / (a(zeta) * a(zeta)));
        };

        c_coeff_ = [=](fp_type zeta) {
            auto const x = grid_1d<fp_type>::transformed_value(grid_transform_config, zeta);
            auto const a_val = a(zeta);
            auto const first = C(x) / a_val;
            auto const second = (B(x) * b(zeta)) / (a_val * a_val * a_val);
            return (first - second);
        };

        d_coeff_ = [=](fp_type zeta) {
            auto const x = grid_1d<fp_type>::transformed_value(grid_transform_config, zeta);
            return D(x);
        };

        init_first_coeff_ = [=](fp_type zeta) {
            auto const x = grid_1d<fp_type>::transformed_value(grid_transform_config, zeta);
            return init_first(x);
        };

        init_second_coeff_ = [=](fp_type zeta) {
            auto const x = grid_1d<fp_type>::transformed_value(grid_transform_config, zeta);
            return init_second(x);
        };
    }

    explicit wave_data_transform() = delete;

  public:
    explicit wave_data_transform(wave_data_config_1d_ptr<fp_type> const &wave_data_config,
                                 grid_transform_config_1d_ptr<fp_type> const grid_transform_config)
    {
        initialize(wave_data_config, grid_transform_config);
    }

    ~wave_data_transform()
    {
    }

    bool const &is_wave_source_set() const
    {
        return is_wave_source_set_;
    }

    std::function<fp_type(fp_type, fp_type)> wave_source() const
    {
        return (is_wave_source_set() == true) ? src_coeff_ : nullptr;
    }

    std::function<fp_type(fp_type)> const &first_initial_condition() const
    {
        return init_first_coeff_;
    }

    std::function<fp_type(fp_type)> const &second_initial_condition() const
    {
        return init_second_coeff_;
    }

    std::function<fp_type(fp_type)> const &a_coefficient() const
    {
        return a_coeff_;
    }

    std::function<fp_type(fp_type)> const &b_coefficient() const
    {
        return b_coeff_;
    }

    std::function<fp_type(fp_type)> const &c_coefficient() const
    {
        return c_coeff_;
    }

    std::function<fp_type(fp_type)> const &d_coefficient() const
    {
        return d_coeff_;
    }
};

template <typename fp_type> using wave_data_transform_1d = wave_data_transform<dimension_enum::One, fp_type>;

template <typename fp_type> using wave_data_transform_1d_ptr = sptr_t<wave_data_transform_1d<fp_type>>;

} // namespace lss_pde_solvers

#endif ///_LSS_WAVE_DATA_TRANSFORM_HPP_
