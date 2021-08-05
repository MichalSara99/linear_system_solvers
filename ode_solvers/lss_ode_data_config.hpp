#if !defined(_LSS_ODE_DATA_CONFIG_HPP_)
#define _LSS_ODE_DATA_CONFIG_HPP_

#include <functional>

#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "common/lss_utility.hpp"

namespace lss_ode_solvers
{

using lss_enumerations::dimension_enum;
using lss_utility::range;
using lss_utility::sptr_t;

/**
    ode_coefficient_data_config structure
 */
template <typename fp_type> struct ode_coefficient_data_config
{
  private:
    std::function<fp_type(fp_type)> a_coeff_;
    std::function<fp_type(fp_type)> b_coeff_;

    explicit ode_coefficient_data_config() = delete;

    void initialize()
    {
        LSS_VERIFY(a_coeff_, "a_coefficient must not be null");
        LSS_VERIFY(b_coeff_, "b_coefficient must not be null");
    }

  public:
    explicit ode_coefficient_data_config(std::function<fp_type(fp_type)> const &a_coefficient,
                                         std::function<fp_type(fp_type)> const &b_coefficient)
        : a_coeff_{a_coefficient}, b_coeff_{b_coefficient}
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
};

template <typename fp_type> using ode_coefficient_data_config_ptr = sptr_t<ode_coefficient_data_config<fp_type>>;

// ============================================================================
// ====================== ode_nonhom_data_config =============================
// ============================================================================

/**
    ode_nonhom_data_config structure
 */
template <typename fp_type> struct ode_nonhom_data_config
{
  private:
    std::function<fp_type(fp_type)> nonhom_fun_;

    explicit ode_nonhom_data_config() = delete;

  public:
    explicit ode_nonhom_data_config(std::function<fp_type(fp_type)> const &nonhom_fun) : nonhom_fun_{nonhom_fun}
    {
        LSS_VERIFY(nonhom_fun_, "nonhom_fun must not be null");
    }

    std::function<fp_type(fp_type)> const &nonhom_function() const
    {
        return nonhom_fun_;
    }
};

template <typename fp_type> using ode_nonhom_data_config_ptr = sptr_t<ode_nonhom_data_config<fp_type>>;

// ============================================================================
// ============================= ode_data_config ==============================
// ============================================================================

/**
    ode_data_config structure
 */
template <typename fp_type> struct ode_data_config
{
  private:
    ode_coefficient_data_config_ptr<fp_type> coefficient_data_cfg_;
    ode_nonhom_data_config_ptr<fp_type> nonhom_data_cfg_;

    void initialize()
    {
        LSS_VERIFY(coefficient_data_cfg_, "coefficient_data_config must not be null");
    }

    explicit ode_data_config() = delete;

  public:
    explicit ode_data_config(ode_coefficient_data_config_ptr<fp_type> const &coefficient_data_config,
                             ode_nonhom_data_config_ptr<fp_type> const &nonhom_data_config = nullptr)
        : coefficient_data_cfg_{coefficient_data_config}, nonhom_data_cfg_{nonhom_data_config}
    {
        initialize();
    }

    ~ode_data_config()
    {
    }

    bool is_nonhom_data_set() const
    {
        return (nonhom_data_cfg_ != nullptr) ? true : false;
    }

    std::function<fp_type(fp_type)> const &nonhom_function() const
    {
        return nonhom_data_cfg_->nonhom_function();
    }

    std::function<fp_type(fp_type)> const &a_coefficient() const
    {
        return coefficient_data_cfg_->a_coefficient();
    }

    std::function<fp_type(fp_type)> const &b_coefficient() const
    {
        return coefficient_data_cfg_->b_coefficient();
    }
};

template <typename fp_type> using ode_data_config_ptr = sptr_t<ode_data_config<fp_type>>;

} // namespace lss_ode_solvers

#endif ///_LSS_ODE_DATA_CONFIG_HPP_
