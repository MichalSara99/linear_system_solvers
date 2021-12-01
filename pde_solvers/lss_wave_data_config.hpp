#if !defined(_LSS_WAVE_DATA_CONFIG_HPP_)
#define _LSS_WAVE_DATA_CONFIG_HPP_

#include <functional>

#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "common/lss_utility.hpp"

namespace lss_pde_solvers
{

using lss_enumerations::dimension_enum;
using lss_utility::range;
using lss_utility::sptr_t;

// ============================================================================
// ================== wave_coefficient_data_config ============================
// ============================================================================
template <dimension_enum dimension, typename fp_type> struct wave_coefficient_data_config
{
};

/**
    1D wave_coefficient_data_config structure
 */
template <typename fp_type> struct wave_coefficient_data_config<dimension_enum::One, fp_type>
{
  private:
    std::function<fp_type(fp_type, fp_type)> a_coeff_;
    std::function<fp_type(fp_type, fp_type)> b_coeff_;
    std::function<fp_type(fp_type, fp_type)> c_coeff_;
    std::function<fp_type(fp_type, fp_type)> d_coeff_;

    explicit wave_coefficient_data_config() = delete;

    void initialize()
    {
        LSS_VERIFY(a_coeff_, "a_coefficient must not be null");
        LSS_VERIFY(b_coeff_, "b_coefficient must not be null");
        LSS_VERIFY(c_coeff_, "c_coefficient must not be null");
        LSS_VERIFY(d_coeff_, "c_coefficient must not be null");
    }

  public:
    explicit wave_coefficient_data_config(std::function<fp_type(fp_type, fp_type)> const &a_coefficient,
                                          std::function<fp_type(fp_type, fp_type)> const &b_coefficient,
                                          std::function<fp_type(fp_type, fp_type)> const &c_coefficient,
                                          std::function<fp_type(fp_type, fp_type)> const &d_coefficient)
        : a_coeff_{a_coefficient}, b_coeff_{b_coefficient}, c_coeff_{c_coefficient}, d_coeff_{d_coefficient}
    {
        initialize();
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

    std::function<fp_type(fp_type, fp_type)> const &d_coefficient() const
    {
        return d_coeff_;
    }
};

/**
    2D wave_coefficient_data_config structure
 */
template <typename fp_type> struct wave_coefficient_data_config<dimension_enum::Two, fp_type>
{
};

template <typename fp_type>
using wave_coefficient_data_config_1d = wave_coefficient_data_config<dimension_enum::One, fp_type>;

template <typename fp_type>
using wave_coefficient_data_config_2d = wave_coefficient_data_config<dimension_enum::Two, fp_type>;

template <typename fp_type>
using wave_coefficient_data_config_1d_ptr = sptr_t<wave_coefficient_data_config<dimension_enum::One, fp_type>>;

template <typename fp_type>
using wave_coefficient_data_config_2d_ptr = sptr_t<wave_coefficient_data_config<dimension_enum::Two, fp_type>>;

// ============================================================================
// ====================== wave_initial_data_config ============================
// ============================================================================
template <dimension_enum dimension, typename fp_type> struct wave_initial_data_config
{
};

/**
    1D wave_initial_data_config structure
 */
template <typename fp_type> struct wave_initial_data_config<dimension_enum::One, fp_type>
{
  private:
    std::function<fp_type(fp_type)> first_initial_condition_;
    std::function<fp_type(fp_type)> second_initial_condition_;

    explicit wave_initial_data_config() = delete;

  public:
    explicit wave_initial_data_config(std::function<fp_type(fp_type)> const &first_initial_condition,
                                      std::function<fp_type(fp_type)> const &second_initial_condition)
        : first_initial_condition_{first_initial_condition}, second_initial_condition_{second_initial_condition}
    {
        LSS_VERIFY(first_initial_condition_, "first_initial_condition must not be null");
        LSS_VERIFY(second_initial_condition_, "second_initial_condition must not be null");
    }

    std::function<fp_type(fp_type)> const &first_initial_condition() const
    {
        return first_initial_condition_;
    }

    std::function<fp_type(fp_type)> const &second_initial_condition() const
    {
        return second_initial_condition_;
    }
};

/**
    2D wave_initial_data_config structure
 */
template <typename fp_type> struct wave_initial_data_config<dimension_enum::Two, fp_type>
{
};

template <typename fp_type> using wave_initial_data_config_1d = wave_initial_data_config<dimension_enum::One, fp_type>;

template <typename fp_type> using wave_initial_data_config_2d = wave_initial_data_config<dimension_enum::Two, fp_type>;

template <typename fp_type>
using wave_initial_data_config_1d_ptr = sptr_t<wave_initial_data_config<dimension_enum::One, fp_type>>;

template <typename fp_type>
using wave_initial_data_config_2d_ptr = sptr_t<wave_initial_data_config<dimension_enum::Two, fp_type>>;

// ============================================================================
// ====================== wave_source_data_config =============================
// ============================================================================
template <dimension_enum dimension, typename fp_type> struct wave_source_data_config
{
};

/**
    1D wave_source_data_config structure
 */
template <typename fp_type> struct wave_source_data_config<dimension_enum::One, fp_type>
{
  private:
    std::function<fp_type(fp_type, fp_type)> wave_source_;

    explicit wave_source_data_config() = delete;

  public:
    explicit wave_source_data_config(std::function<fp_type(fp_type, fp_type)> const &wave_source)
        : wave_source_{wave_source}
    {
        LSS_VERIFY(wave_source_, "wave_source must not be null");
    }

    std::function<fp_type(fp_type, fp_type)> const &wave_source() const
    {
        return wave_source_;
    }
};

/**
    2D wave_source_data_config structure
 */
template <typename fp_type> struct wave_source_data_config<dimension_enum::Two, fp_type>
{
};

template <typename fp_type> using wave_source_data_config_1d = wave_source_data_config<dimension_enum::One, fp_type>;

template <typename fp_type> using wave_source_data_config_2d = wave_source_data_config<dimension_enum::Two, fp_type>;

template <typename fp_type>
using wave_source_data_config_1d_ptr = sptr_t<wave_source_data_config<dimension_enum::One, fp_type>>;

template <typename fp_type>
using wave_source_data_config_2d_ptr = sptr_t<wave_source_data_config<dimension_enum::Two, fp_type>>;

// ============================================================================
// ============================= wave_data_config =============================
// ============================================================================
template <dimension_enum dimension, typename fp_type> struct wave_data_config
{
};

/**
    1D wave_data_config structure
 */
template <typename fp_type> struct wave_data_config<dimension_enum::One, fp_type>
{
  private:
    wave_coefficient_data_config_1d_ptr<fp_type> coefficient_data_cfg_;
    wave_initial_data_config_1d_ptr<fp_type> initial_data_cfg_;
    wave_source_data_config_1d_ptr<fp_type> source_data_cfg_;

    void initialize()
    {
        LSS_VERIFY(coefficient_data_cfg_, "coefficient_data_config must not be null");
        LSS_VERIFY(initial_data_cfg_, "initial_data_config must not be null");
    }

    explicit wave_data_config() = delete;

  public:
    explicit wave_data_config(wave_coefficient_data_config_1d_ptr<fp_type> const &coefficient_data_config,
                              wave_initial_data_config_1d_ptr<fp_type> const &initial_data_config,
                              wave_source_data_config_1d_ptr<fp_type> const &source_data_config = nullptr)
        : coefficient_data_cfg_{coefficient_data_config}, initial_data_cfg_{initial_data_config},
          source_data_cfg_{source_data_config}
    {
        initialize();
    }

    ~wave_data_config()
    {
    }

    wave_source_data_config_1d_ptr<fp_type> const &source_data_config() const
    {
        return source_data_cfg_;
    }

    std::function<fp_type(fp_type)> const &first_initial_condition() const
    {
        return initial_data_cfg_->first_initial_condition();
    }

    std::function<fp_type(fp_type)> const &second_initial_condition() const
    {
        return initial_data_cfg_->second_initial_condition();
    }

    std::function<fp_type(fp_type, fp_type)> const &a_coefficient() const
    {
        return coefficient_data_cfg_->a_coefficient();
    }

    std::function<fp_type(fp_type, fp_type)> const &b_coefficient() const
    {
        return coefficient_data_cfg_->b_coefficient();
    }

    std::function<fp_type(fp_type, fp_type)> const &c_coefficient() const
    {
        return coefficient_data_cfg_->c_coefficient();
    }

    std::function<fp_type(fp_type, fp_type)> const &d_coefficient() const
    {
        return coefficient_data_cfg_->d_coefficient();
    }
};

/**
    2D heat_data_config structure
 */
template <typename fp_type> struct wave_data_config<dimension_enum::Two, fp_type>
{
};

template <typename fp_type> using wave_data_config_1d = wave_data_config<dimension_enum::One, fp_type>;

template <typename fp_type> using wave_data_config_2d = wave_data_config<dimension_enum::Two, fp_type>;

template <typename fp_type> using wave_data_config_1d_ptr = sptr_t<wave_data_config<dimension_enum::One, fp_type>>;

template <typename fp_type> using wave_data_config_2d_ptr = sptr_t<wave_data_config<dimension_enum::Two, fp_type>>;

} // namespace lss_pde_solvers

#endif ///_LSS_WAVE_DATA_CONFIG_HPP_
