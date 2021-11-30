#if !defined(_LSS_HEAT_DATA_CONFIG_BUILDER_HPP_)
#define _LSS_HEAT_DATA_CONFIG_BUILDER_HPP_

#include <functional>

#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "common/lss_utility.hpp"
#include "pde_solvers/lss_heat_data_config.hpp"

namespace lss_pde_solvers
{

using lss_enumerations::dimension_enum;
using lss_utility::range;
using lss_utility::sptr_t;

// ============================================================================
// ============== heat_coefficient_data_config_builder ========================
// ============================================================================
template <dimension_enum dimension, typename fp_type> struct heat_coefficient_data_config_builder
{
};

/**
    1D heat_coefficient_data_config_builder structure
 */
template <typename fp_type> struct heat_coefficient_data_config_builder<dimension_enum::One, fp_type>
{
  private:
    std::function<fp_type(fp_type)> a_coefficient_;
    std::function<fp_type(fp_type)> b_coefficient_;
    std::function<fp_type(fp_type)> c_coefficient_;

  public:
    heat_coefficient_data_config_builder &a_coefficient(std::function<fp_type(fp_type)> const &a_coefficient)
    {
        a_coefficient_ = a_coefficient;
        return *this;
    }

    heat_coefficient_data_config_builder &b_coefficient(std::function<fp_type(fp_type)> const &b_coefficient)
    {
        b_coefficient_ = b_coefficient;
        return *this;
    }

    heat_coefficient_data_config_builder &c_coefficient(std::function<fp_type(fp_type)> const &c_coefficient)
    {
        c_coefficient_ = c_coefficient;
        return *this;
    }

    heat_coefficient_data_config_1d_ptr<fp_type> build()
    {
        return std::make_shared<heat_coefficient_data_config<dimension_enum::One, fp_type>>(
            a_coefficient_, b_coefficient_, c_coefficient_);
    }
};

/**
    2D heat_coefficient_data_config_builder structure
 */
template <typename fp_type> struct heat_coefficient_data_config_builder<dimension_enum::Two, fp_type>
{
  private:
    std::function<fp_type(fp_type, fp_type, fp_type)> a_coefficient_;
    std::function<fp_type(fp_type, fp_type, fp_type)> b_coefficient_;
    std::function<fp_type(fp_type, fp_type, fp_type)> c_coefficient_;
    std::function<fp_type(fp_type, fp_type, fp_type)> d_coefficient_;
    std::function<fp_type(fp_type, fp_type, fp_type)> e_coefficient_;
    std::function<fp_type(fp_type, fp_type, fp_type)> f_coefficient_;

  public:
    heat_coefficient_data_config_builder &a_coefficient(
        std::function<fp_type(fp_type, fp_type, fp_type)> const &a_coefficient)
    {
        a_coefficient_ = a_coefficient;
        return *this;
    }

    heat_coefficient_data_config_builder &b_coefficient(
        std::function<fp_type(fp_type, fp_type, fp_type)> const &b_coefficient)
    {
        b_coefficient_ = b_coefficient;
        return *this;
    }

    heat_coefficient_data_config_builder &c_coefficient(
        std::function<fp_type(fp_type, fp_type, fp_type)> const &c_coefficient)
    {
        c_coefficient_ = c_coefficient;
        return *this;
    }

    heat_coefficient_data_config_builder &d_coefficient(
        std::function<fp_type(fp_type, fp_type, fp_type)> const &d_coefficient)
    {
        d_coefficient_ = d_coefficient;
        return *this;
    }

    heat_coefficient_data_config_builder &e_coefficient(
        std::function<fp_type(fp_type, fp_type, fp_type)> const &e_coefficient)
    {
        e_coefficient_ = e_coefficient;
        return *this;
    }

    heat_coefficient_data_config_builder &f_coefficient(
        std::function<fp_type(fp_type, fp_type, fp_type)> const &f_coefficient)
    {
        f_coefficient_ = f_coefficient;
        return *this;
    }

    heat_coefficient_data_config_2d_ptr<fp_type> build()
    {
        return std::make_shared<heat_coefficient_data_config<dimension_enum::Two, fp_type>>(
            a_coefficient_, b_coefficient_, c_coefficient_, d_coefficient_, e_coefficient_, f_coefficient_);
    }
};

template <typename fp_type>
using heat_coefficient_data_config_1d_builder = heat_coefficient_data_config_builder<dimension_enum::One, fp_type>;

template <typename fp_type>
using heat_coefficient_data_config_2d_builder = heat_coefficient_data_config_builder<dimension_enum::Two, fp_type>;

template <typename fp_type>
using heat_coefficient_data_config_1d_builder_ptr =
    sptr_t<heat_coefficient_data_config_builder<dimension_enum::One, fp_type>>;

template <typename fp_type>
using heat_coefficient_data_config_2d_builder_ptr =
    sptr_t<heat_coefficient_data_config_builder<dimension_enum::Two, fp_type>>;

// ============================================================================
// ================= heat_initial_data_config_builder =========================
// ============================================================================
template <dimension_enum dimension, typename fp_type> struct heat_initial_data_config_builder
{
};

/**
    1D heat_initial_data_config_builder structure
 */
template <typename fp_type> struct heat_initial_data_config_builder<dimension_enum::One, fp_type>
{
  private:
    std::function<fp_type(fp_type)> initial_condition_;

  public:
    heat_initial_data_config_builder &initial_condition(std::function<fp_type(fp_type)> const &initial_condition)
    {
        initial_condition_ = initial_condition;
        return *this;
    }

    heat_initial_data_config_1d_ptr<fp_type> build()
    {
        return std::make_shared<heat_initial_data_config<dimension_enum::One, fp_type>>(initial_condition_);
    }
};

/**
    2D heat_initial_data_config_builder structure
 */
template <typename fp_type> struct heat_initial_data_config_builder<dimension_enum::Two, fp_type>
{
  private:
    std::function<fp_type(fp_type, fp_type)> initial_condition_;

  public:
    heat_initial_data_config_builder &initial_condition(
        std::function<fp_type(fp_type, fp_type)> const &initial_condition)
    {
        initial_condition_ = initial_condition;
        return *this;
    }

    heat_initial_data_config_2d_ptr<fp_type> build()
    {
        return std::make_shared<heat_initial_data_config<dimension_enum::Two, fp_type>>(initial_condition_);
    }
};

template <typename fp_type>
using heat_initial_data_config_1d_builder = heat_initial_data_config_builder<dimension_enum::One, fp_type>;

template <typename fp_type>
using heat_initial_data_config_2d_builder = heat_initial_data_config_builder<dimension_enum::Two, fp_type>;

template <typename fp_type>
using heat_initial_data_config_1d_builder_ptr = sptr_t<heat_initial_data_config_builder<dimension_enum::One, fp_type>>;

template <typename fp_type>
using heat_initial_data_config_2d_builder_ptr = sptr_t<heat_initial_data_config_builder<dimension_enum::Two, fp_type>>;

// ============================================================================
// =================== heat_source_data_config_builder ========================
// ============================================================================
template <dimension_enum dimension, typename fp_type> struct heat_source_data_config_builder
{
};

/**
    1D heat_source_data_config_builder structure
 */
template <typename fp_type> struct heat_source_data_config_builder<dimension_enum::One, fp_type>
{
  private:
    std::function<fp_type(fp_type, fp_type)> heat_source_;

  public:
    heat_source_data_config_builder &heat_source(std::function<fp_type(fp_type, fp_type)> const &heat_source)
    {
        heat_source_ = heat_source;
        return *this;
    }

    heat_source_data_config_1d_ptr<fp_type> build()
    {
        return std::make_shared<heat_source_data_config<dimension_enum::One, fp_type>>(heat_source_);
    }
};

/**
    2D heat_source_data_config_builder structure
 */
template <typename fp_type> struct heat_source_data_config_builder<dimension_enum::Two, fp_type>
{
  private:
    std::function<fp_type(fp_type, fp_type, fp_type)> heat_source_;

  public:
    heat_source_data_config_builder &heat_source(std::function<fp_type(fp_type, fp_type, fp_type)> const &heat_source)
    {
        heat_source_ = heat_source;
        return *this;
    }

    heat_source_data_config_2d_ptr<fp_type> build()
    {
        return std::make_shared<heat_source_data_config<dimension_enum::Two, fp_type>>(heat_source_);
    }
};

template <typename fp_type>
using heat_source_data_config_1d_builder = heat_source_data_config_builder<dimension_enum::One, fp_type>;

template <typename fp_type>
using heat_source_data_config_2d_builder = heat_source_data_config_builder<dimension_enum::Two, fp_type>;

template <typename fp_type>
using heat_source_data_config_1d_builder_ptr = sptr_t<heat_source_data_config_builder<dimension_enum::One, fp_type>>;

template <typename fp_type>
using heat_source_data_config_2d_builder_ptr = sptr_t<heat_source_data_config_builder<dimension_enum::Two, fp_type>>;

// ============================================================================
// ========================== heat_data_config_builder ========================
// ============================================================================
template <dimension_enum dimension, typename fp_type> struct heat_data_config_builder
{
};

/**
    1D heat_data_config_builder structure
 */
template <typename fp_type> struct heat_data_config_builder<dimension_enum::One, fp_type>
{
  private:
    heat_coefficient_data_config_1d_ptr<fp_type> coefficient_data_config_;
    heat_initial_data_config_1d_ptr<fp_type> initial_data_config_;
    heat_source_data_config_1d_ptr<fp_type> source_data_config_;

  public:
    heat_data_config_builder &coefficient_data_config(
        heat_coefficient_data_config_1d_ptr<fp_type> const &coefficient_data_config)
    {
        coefficient_data_config_ = coefficient_data_config;
        return *this;
    }

    heat_data_config_builder &initial_data_config(heat_initial_data_config_1d_ptr<fp_type> const &initial_data_config)
    {
        initial_data_config_ = initial_data_config;
        return *this;
    }

    heat_data_config_builder &source_data_config(heat_source_data_config_1d_ptr<fp_type> const &source_data_config)
    {
        source_data_config_ = source_data_config;
        return *this;
    }

    heat_data_config_1d_ptr<fp_type> build()
    {
        return std::make_shared<heat_data_config<dimension_enum::One, fp_type>>(
            coefficient_data_config_, initial_data_config_, source_data_config_);
    }
};

/**
    2D heat_data_config_builder structure
 */
template <typename fp_type> struct heat_data_config_builder<dimension_enum::Two, fp_type>
{
  private:
    heat_coefficient_data_config_2d_ptr<fp_type> coefficient_data_config_;
    heat_initial_data_config_2d_ptr<fp_type> initial_data_config_;
    heat_source_data_config_2d_ptr<fp_type> source_data_config_;

  public:
    heat_data_config_builder &coefficient_data_config(
        heat_coefficient_data_config_2d_ptr<fp_type> const &coefficient_data_config)
    {
        coefficient_data_config_ = coefficient_data_config;
        return *this;
    }

    heat_data_config_builder &initial_data_config(heat_initial_data_config_2d_ptr<fp_type> const &initial_data_config)
    {
        initial_data_config_ = initial_data_config;
        return *this;
    }

    heat_data_config_builder &initial_data_config(heat_source_data_config_2d_ptr<fp_type> const &source_data_config)
    {
        source_data_config_ = source_data_config;
        return *this;
    }

    heat_data_config_2d_ptr<fp_type> build()
    {
        return std::make_shared<heat_data_config<dimension_enum::Two, fp_type>>(
            coefficient_data_config_, initial_data_config_, source_data_config_);
    }
};

template <typename fp_type> using heat_data_config_1d_builder = heat_data_config_builder<dimension_enum::One, fp_type>;

template <typename fp_type> using heat_data_config_2d_builder = heat_data_config_builder<dimension_enum::Two, fp_type>;

template <typename fp_type>
using heat_data_config_1d_builder_ptr = sptr_t<heat_data_config_builder<dimension_enum::One, fp_type>>;

template <typename fp_type>
using heat_data_config_2d_builder_ptr = sptr_t<heat_data_config_builder<dimension_enum::Two, fp_type>>;

} // namespace lss_pde_solvers

#endif ///_LSS_HEAT_DATA_CONFIG_BUILDER_HPP_
