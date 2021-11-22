#if !defined(_LSS_HEAT_DATA_CONFIG_HPP_)
#define _LSS_HEAT_DATA_CONFIG_HPP_

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
// ================== heat_coefficient_data_config ============================
// ============================================================================
template <dimension_enum dimension, typename fp_type> struct heat_coefficient_data_config
{
};

/**
    1D heat_initial_data_config structure
 */
template <typename fp_type> struct heat_coefficient_data_config<dimension_enum::One, fp_type>
{
  private:
    std::function<fp_type(fp_type)> a_coeff_;
    std::function<fp_type(fp_type)> b_coeff_;
    std::function<fp_type(fp_type)> c_coeff_;

    explicit heat_coefficient_data_config() = delete;

    void initialize()
    {
        LSS_VERIFY(a_coeff_, "a_coefficient must not be null");
        LSS_VERIFY(b_coeff_, "b_coefficient must not be null");
        LSS_VERIFY(c_coeff_, "c_coefficient must not be null");
    }

  public:
    explicit heat_coefficient_data_config(std::function<fp_type(fp_type)> const &a_coefficient,
                                          std::function<fp_type(fp_type)> const &b_coefficient,
                                          std::function<fp_type(fp_type)> const &c_coefficient)
        : a_coeff_{a_coefficient}, b_coeff_{b_coefficient}, c_coeff_{c_coefficient}
    {
        initialize();
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
};

/**
    2D heat_initial_data_config structure
 */
template <typename fp_type> struct heat_coefficient_data_config<dimension_enum::Two, fp_type>
{
  private:
    std::function<fp_type(fp_type, fp_type)> a_coeff_;
    std::function<fp_type(fp_type, fp_type)> b_coeff_;
    std::function<fp_type(fp_type, fp_type)> c_coeff_;
    std::function<fp_type(fp_type, fp_type)> d_coeff_;
    std::function<fp_type(fp_type, fp_type)> e_coeff_;
    std::function<fp_type(fp_type, fp_type)> f_coeff_;

    explicit heat_coefficient_data_config() = delete;

    void initialize()
    {
        LSS_VERIFY(a_coeff_, "a_coefficient must not be null");
        LSS_VERIFY(b_coeff_, "b_coefficient must not be null");
        LSS_VERIFY(c_coeff_, "c_coefficient must not be null");
        LSS_VERIFY(d_coeff_, "d_coefficient must not be null");
        LSS_VERIFY(e_coeff_, "e_coefficient must not be null");
        LSS_VERIFY(f_coeff_, "f_coefficient must not be null");
    }

  public:
    explicit heat_coefficient_data_config(std::function<fp_type(fp_type, fp_type)> const &a_coefficient,
                                          std::function<fp_type(fp_type, fp_type)> const &b_coefficient,
                                          std::function<fp_type(fp_type, fp_type)> const &c_coefficient,
                                          std::function<fp_type(fp_type, fp_type)> const &d_coefficient,
                                          std::function<fp_type(fp_type, fp_type)> const &e_coefficient,
                                          std::function<fp_type(fp_type, fp_type)> const &f_coefficient)
        : a_coeff_{a_coefficient}, b_coeff_{b_coefficient}, c_coeff_{c_coefficient}, d_coeff_{d_coefficient},
          e_coeff_{e_coefficient}, f_coeff_{f_coefficient}
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

    std::function<fp_type(fp_type, fp_type)> const &e_coefficient() const
    {
        return e_coeff_;
    }

    std::function<fp_type(fp_type, fp_type)> const &f_coefficient() const
    {
        return f_coeff_;
    }
};

template <typename fp_type>
using heat_coefficient_data_config_1d = heat_coefficient_data_config<dimension_enum::One, fp_type>;

template <typename fp_type>
using heat_coefficient_data_config_2d = heat_coefficient_data_config<dimension_enum::Two, fp_type>;

template <typename fp_type>
using heat_coefficient_data_config_1d_ptr = sptr_t<heat_coefficient_data_config<dimension_enum::One, fp_type>>;

template <typename fp_type>
using heat_coefficient_data_config_2d_ptr = sptr_t<heat_coefficient_data_config<dimension_enum::Two, fp_type>>;

// ============================================================================
// ====================== heat_initial_data_config ============================
// ============================================================================
template <dimension_enum dimension, typename fp_type> struct heat_initial_data_config
{
};

/**
    1D heat_initial_data_config structure
 */
template <typename fp_type> struct heat_initial_data_config<dimension_enum::One, fp_type>
{
  private:
    std::function<fp_type(fp_type)> initial_condition_;

    explicit heat_initial_data_config() = delete;

  public:
    explicit heat_initial_data_config(std::function<fp_type(fp_type)> const &initial_condition)
        : initial_condition_{initial_condition}
    {
        LSS_VERIFY(initial_condition_, "initial_condition must not be null");
    }

    std::function<fp_type(fp_type)> const &initial_condition() const
    {
        return initial_condition_;
    }
};

/**
    2D heat_initial_data_config structure
 */
template <typename fp_type> struct heat_initial_data_config<dimension_enum::Two, fp_type>
{
  private:
    std::function<fp_type(fp_type, fp_type)> initial_condition_;

    explicit heat_initial_data_config() = delete;

  public:
    explicit heat_initial_data_config(std::function<fp_type(fp_type, fp_type)> const &initial_condition)
        : initial_condition_{initial_condition}
    {
        LSS_VERIFY(initial_condition_, "initial_condition must not be null");
    }

    std::function<fp_type(fp_type, fp_type)> const &initial_condition() const
    {
        return initial_condition_;
    }
};

template <typename fp_type> using heat_initial_data_config_1d = heat_initial_data_config<dimension_enum::One, fp_type>;

template <typename fp_type> using heat_initial_data_config_2d = heat_initial_data_config<dimension_enum::Two, fp_type>;

template <typename fp_type>
using heat_initial_data_config_1d_ptr = sptr_t<heat_initial_data_config<dimension_enum::One, fp_type>>;

template <typename fp_type>
using heat_initial_data_config_2d_ptr = sptr_t<heat_initial_data_config<dimension_enum::Two, fp_type>>;

// ============================================================================
// ====================== heat_source_data_config =============================
// ============================================================================
template <dimension_enum dimension, typename fp_type> struct heat_source_data_config
{
};

/**
    1D heat_source_data_config structure
 */
template <typename fp_type> struct heat_source_data_config<dimension_enum::One, fp_type>
{
  private:
    std::function<fp_type(fp_type, fp_type)> heat_source_;

    explicit heat_source_data_config() = delete;

  public:
    explicit heat_source_data_config(std::function<fp_type(fp_type, fp_type)> const &heat_source)
        : heat_source_{heat_source}
    {
        LSS_VERIFY(heat_source_, "heat_source must not be null");
    }

    std::function<fp_type(fp_type, fp_type)> const &heat_source() const
    {
        return heat_source_;
    }
};

/**
    2D heat_source_data_config structure
 */
template <typename fp_type> struct heat_source_data_config<dimension_enum::Two, fp_type>
{
  private:
    std::function<fp_type(fp_type, fp_type, fp_type)> heat_source_;

    explicit heat_source_data_config() = delete;

  public:
    explicit heat_source_data_config(std::function<fp_type(fp_type, fp_type, fp_type)> const &heat_source)
        : heat_source_{heat_source}
    {
        LSS_VERIFY(heat_source_, "heat_source must not be null");
    }

    std::function<fp_type(fp_type, fp_type, fp_type)> const &heat_source() const
    {
        return heat_source_;
    }
};

template <typename fp_type> using heat_source_data_config_1d = heat_source_data_config<dimension_enum::One, fp_type>;

template <typename fp_type> using heat_source_data_config_2d = heat_source_data_config<dimension_enum::Two, fp_type>;

template <typename fp_type>
using heat_source_data_config_1d_ptr = sptr_t<heat_source_data_config<dimension_enum::One, fp_type>>;

template <typename fp_type>
using heat_source_data_config_2d_ptr = sptr_t<heat_source_data_config<dimension_enum::Two, fp_type>>;

// ============================================================================
// ============================= heat_data_config =============================
// ============================================================================
template <dimension_enum dimension, typename fp_type> struct heat_data_config
{
};

/**
    1D heat_data_config structure
 */
template <typename fp_type> struct heat_data_config<dimension_enum::One, fp_type>
{
  private:
    heat_coefficient_data_config_1d_ptr<fp_type> coefficient_data_cfg_;
    heat_initial_data_config_1d_ptr<fp_type> initial_data_cfg_;
    heat_source_data_config_1d_ptr<fp_type> source_data_cfg_;

    void initialize()
    {
        LSS_VERIFY(coefficient_data_cfg_, "coefficient_data_config must not be null");
        LSS_VERIFY(initial_data_cfg_, "initial_data_config must not be null");
    }

    explicit heat_data_config() = delete;

  public:
    explicit heat_data_config(heat_coefficient_data_config_1d_ptr<fp_type> const &coefficient_data_config,
                              heat_initial_data_config_1d_ptr<fp_type> const &initial_data_config,
                              heat_source_data_config_1d_ptr<fp_type> const &source_data_config = nullptr)
        : coefficient_data_cfg_{coefficient_data_config}, initial_data_cfg_{initial_data_config},
          source_data_cfg_{source_data_config}
    {
        initialize();
    }

    ~heat_data_config()
    {
    }

    heat_source_data_config_1d_ptr<fp_type> const &source_data_config() const
    {
        return source_data_cfg_;
    }

    std::function<fp_type(fp_type)> const &initial_condition() const
    {
        return initial_data_cfg_->initial_condition();
    }

    std::function<fp_type(fp_type)> const &a_coefficient() const
    {
        return coefficient_data_cfg_->a_coefficient();
    }

    std::function<fp_type(fp_type)> const &b_coefficient() const
    {
        return coefficient_data_cfg_->b_coefficient();
    }

    std::function<fp_type(fp_type)> const &c_coefficient() const
    {
        return coefficient_data_cfg_->c_coefficient();
    }
};

/**
    2D heat_data_config structure
 */
template <typename fp_type> struct heat_data_config<dimension_enum::Two, fp_type>
{
  private:
    heat_coefficient_data_config_2d_ptr<fp_type> coefficient_data_cfg_;
    heat_initial_data_config_2d_ptr<fp_type> initial_data_cfg_;
    heat_source_data_config_2d_ptr<fp_type> source_data_cfg_;

    void initialize()
    {
        LSS_VERIFY(coefficient_data_cfg_, "coefficient_data_config must not be null");
        LSS_VERIFY(initial_data_cfg_, "initial_data_config must not be null");
    }

    explicit heat_data_config() = delete;

  public:
    explicit heat_data_config(heat_coefficient_data_config_2d_ptr<fp_type> const &coefficient_data_config,
                              heat_initial_data_config_2d_ptr<fp_type> const &initial_data_config,
                              heat_source_data_config_2d_ptr<fp_type> const &source_data_config = nullptr)
        : coefficient_data_cfg_{coefficient_data_config}, initial_data_cfg_{initial_data_config},
          source_data_cfg_{source_data_config}
    {
        initialize();
    }

    ~heat_data_config()
    {
    }

    heat_source_data_config_2d_ptr<fp_type> const &source_data_config() const
    {
        return source_data_cfg_;
    }

    std::function<fp_type(fp_type, fp_type)> const &initial_condition() const
    {
        return initial_data_cfg_->initial_condition();
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

    std::function<fp_type(fp_type, fp_type)> const &e_coefficient() const
    {
        return coefficient_data_cfg_->e_coefficient();
    }

    std::function<fp_type(fp_type, fp_type)> const &f_coefficient() const
    {
        return coefficient_data_cfg_->f_coefficient();
    }
};

template <typename fp_type> using heat_data_config_1d = heat_data_config<dimension_enum::One, fp_type>;

template <typename fp_type> using heat_data_config_2d = heat_data_config<dimension_enum::Two, fp_type>;

template <typename fp_type> using heat_data_config_1d_ptr = sptr_t<heat_data_config<dimension_enum::One, fp_type>>;

template <typename fp_type> using heat_data_config_2d_ptr = sptr_t<heat_data_config<dimension_enum::Two, fp_type>>;

} // namespace lss_pde_solvers

#endif ///_LSS_HEAT_DATA_CONFIG_HPP_
