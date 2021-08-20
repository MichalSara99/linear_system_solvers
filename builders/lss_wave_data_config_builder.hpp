#if !defined(_LSS_WAVE_DATA_CONFIG_BUILDER_HPP_)
#define _LSS_WAVE_DATA_CONFIG_BUILDER_HPP_

#include <functional>

#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "common/lss_utility.hpp"
#include "pde_solvers/lss_wave_data_config.hpp"

namespace lss_pde_solvers
{

using lss_enumerations::dimension_enum;
using lss_utility::range;
using lss_utility::sptr_t;

// ============================================================================
// ============= wave_coefficient_data_config_builder =========================
// ============================================================================
template <dimension_enum dimension, typename fp_type> struct wave_coefficient_data_config_builder
{
};

/**
    1D wave_coefficient_data_config_builder structure
 */
template <typename fp_type> struct wave_coefficient_data_config_builder<dimension_enum::One, fp_type>
{
  public:
    std::function<fp_type(fp_type)> a_coefficient_;
    std::function<fp_type(fp_type)> b_coefficient_;
    std::function<fp_type(fp_type)> c_coefficient_;
    std::function<fp_type(fp_type)> d_coefficient_;

  public:
    wave_coefficient_data_config_builder &a_coefficient(std::function<fp_type(fp_type)> const &a_coefficient)
    {
        a_coefficient_ = a_coefficient;
        return *this;
    }

    wave_coefficient_data_config_builder &b_coefficient(std::function<fp_type(fp_type)> const &b_coefficient)
    {
        b_coefficient_ = b_coefficient;
        return *this;
    }

    wave_coefficient_data_config_builder &c_coefficient(std::function<fp_type(fp_type)> const &c_coefficient)
    {
        c_coefficient_ = c_coefficient;
        return *this;
    }

    wave_coefficient_data_config_builder &d_coefficient(std::function<fp_type(fp_type)> const &d_coefficient)
    {
        d_coefficient_ = d_coefficient;
        return *this;
    }

    wave_coefficient_data_config_1d_ptr<fp_type> build()
    {
        return std::make_shared<wave_coefficient_data_config<dimension_enum::One, fp_type>>(
            a_coefficient_, b_coefficient_, c_coefficient_, d_coefficient_);
    }
};

/**
    2D wave_coefficient_data_config_builder structure
 */
template <typename fp_type> struct wave_coefficient_data_config_builder<dimension_enum::Two, fp_type>
{
    // TODO:implementation to be done
};

template <typename fp_type>
using wave_coefficient_data_config_1d_builder = wave_coefficient_data_config_builder<dimension_enum::One, fp_type>;

template <typename fp_type>
using wave_coefficient_data_config_2d_builder = wave_coefficient_data_config_builder<dimension_enum::Two, fp_type>;

template <typename fp_type>
using wave_coefficient_data_config_1d_builder_ptr =
    sptr_t<wave_coefficient_data_config_builder<dimension_enum::One, fp_type>>;

template <typename fp_type>
using wave_coefficient_data_config_2d_builder_ptr =
    sptr_t<wave_coefficient_data_config_builder<dimension_enum::Two, fp_type>>;

// ============================================================================
// ================== wave_initial_data_config_builder ========================
// ============================================================================
template <dimension_enum dimension, typename fp_type> struct wave_initial_data_config_builder
{
};

/**
    1D wave_initial_data_config_builder structure
 */
template <typename fp_type> struct wave_initial_data_config_builder<dimension_enum::One, fp_type>
{
  private:
    std::function<fp_type(fp_type)> first_initial_condition_;
    std::function<fp_type(fp_type)> second_initial_condition_;

  public:
    wave_initial_data_config_builder &first_initial_condition(
        std::function<fp_type(fp_type)> const &first_initial_condition)
    {
        first_initial_condition_ = first_initial_condition;
        return *this;
    }

    wave_initial_data_config_builder &second_initial_condition(
        std::function<fp_type(fp_type)> const &second_initial_condition)
    {
        second_initial_condition_ = second_initial_condition;
        return *this;
    }

    wave_initial_data_config_1d_ptr<fp_type> build()
    {
        return std::make_shared<wave_initial_data_config<dimension_enum::One, fp_type>>(first_initial_condition_,
                                                                                        second_initial_condition_);
    }
};

/**
    2D wave_initial_data_config_builder structure
 */
template <typename fp_type> struct wave_initial_data_config_builder<dimension_enum::Two, fp_type>
{
    // TODO:implementation to be done
};

template <typename fp_type>
using wave_initial_data_config_1d_builder = wave_initial_data_config_builder<dimension_enum::One, fp_type>;

template <typename fp_type>
using wave_initial_data_config_2d_builder = wave_initial_data_config_builder<dimension_enum::Two, fp_type>;

template <typename fp_type>
using wave_initial_data_config_1d_builder_ptr = sptr_t<wave_initial_data_config_builder<dimension_enum::One, fp_type>>;

template <typename fp_type>
using wave_initial_data_config_2d_builder_ptr = sptr_t<wave_initial_data_config_builder<dimension_enum::Two, fp_type>>;

// ============================================================================
// ================== wave_source_data_config_builder =========================
// ============================================================================
template <dimension_enum dimension, typename fp_type> struct wave_source_data_config_builder
{
};

/**
    1D wave_source_data_config_builder structure
 */
template <typename fp_type> struct wave_source_data_config_builder<dimension_enum::One, fp_type>
{
  private:
    std::function<fp_type(fp_type, fp_type)> wave_source_;

  public:
    wave_source_data_config_builder &wave_source(std::function<fp_type(fp_type, fp_type)> const &wave_source)
    {
        wave_source_ = wave_source;
        return *this;
    }

    wave_source_data_config_1d_ptr build()
    {
        return std::make_shared<wave_source_data_config<dimension_enum::One, fp_type>>(wave_source_);
    }
};

/**
    2D wave_source_data_config_builder structure
 */
template <typename fp_type> struct wave_source_data_config_builder<dimension_enum::Two, fp_type>
{
    // TODO:implementation to be done
};

template <typename fp_type>
using wave_source_data_config_1d_builder = wave_source_data_config_builder<dimension_enum::One, fp_type>;

template <typename fp_type>
using wave_source_data_config_2d_builder = wave_source_data_config_builder<dimension_enum::Two, fp_type>;

template <typename fp_type>
using wave_source_data_config_1d_builder_ptr = sptr_t<wave_source_data_config_builder<dimension_enum::One, fp_type>>;

template <typename fp_type>
using wave_source_data_config_2d_builder_ptr = sptr_t<wave_source_data_config_builder<dimension_enum::Two, fp_type>>;

// ============================================================================
// ======================== wave_data_config_builder ==========================
// ============================================================================
template <dimension_enum dimension, typename fp_type> struct wave_data_config_builder
{
};

/**
    1D wave_data_config_builder structure
 */
template <typename fp_type> struct wave_data_config_builder<dimension_enum::One, fp_type>
{
  private:
    wave_coefficient_data_config_1d_ptr<fp_type> coefficient_data_config_;
    wave_initial_data_config_1d_ptr<fp_type> initial_data_config_;
    wave_source_data_config_1d_ptr<fp_type> source_data_config_;

  public:
    wave_data_config_builder &coefficient_data_config(
        wave_coefficient_data_config_1d_ptr<fp_type> const &coefficient_data_config)
    {
        coefficient_data_config_ = coefficient_data_config;
        return *this;
    }

    wave_data_config_builder &initial_data_config(wave_initial_data_config_1d_ptr<fp_type> const &initial_data_config)
    {
        initial_data_config_ = initial_data_config;
        return *this;
    }

    wave_data_config_builder &source_data_config(wave_source_data_config_1d_ptr<fp_type> const &source_data_config)
    {
        source_data_config_ = source_data_config;
        return *this;
    }

    wave_data_config_1d_ptr<fp_type> build()
    {
        return std::make_shared<wave_data_config<dimension_enum::One, fp_type>>(
            coefficient_data_config_, initial_data_config_, source_data_config_);
    }
};

/**
    2D wave_data_config_builder structure
 */
template <typename fp_type> struct wave_data_config_builder<dimension_enum::Two, fp_type>
{
    // TODO:implementation to be done
};

template <typename fp_type> using wave_data_config_1d_builder = wave_data_config_builder<dimension_enum::One, fp_type>;

template <typename fp_type> using wave_data_config_2d_builder = wave_data_config_builder<dimension_enum::Two, fp_type>;

template <typename fp_type>
using wave_data_config_1d_builder_ptr = sptr_t<wave_data_config_builder<dimension_enum::One, fp_type>>;

template <typename fp_type>
using wave_data_config_2d_builder_ptr = sptr_t<wave_data_config_builder<dimension_enum::Two, fp_type>>;

} // namespace lss_pde_solvers

#endif ///_LSS_WAVE_DATA_CONFIG_BUILDER_HPP_
