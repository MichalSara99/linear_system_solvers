#if !defined(_LSS_ODE_DATA_CONFIG_BUILDER_HPP_)
#define _LSS_ODE_DATA_CONFIG_BUILDER_HPP_

#include <functional>

#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "common/lss_utility.hpp"
#include "ode_solvers/lss_ode_data_config.hpp"

namespace lss_ode_solvers
{

using lss_enumerations::dimension_enum;
using lss_utility::range;
using lss_utility::sptr_t;

/**
    ode_coefficient_data_config_builder structure
 */
template <typename fp_type> struct ode_coefficient_data_config_builder
{
  private:
    std::function<fp_type(fp_type)> a_coefficient_;
    std::function<fp_type(fp_type)> b_coefficient_;

  public:
    ode_coefficient_data_config_builder &a_coefficient(std::function<fp_type(fp_type)> const &a_coefficient)
    {
        a_coefficient_ = a_coefficient;
        return *this;
    }

    ode_coefficient_data_config_builder &b_coefficient(std::function<fp_type(fp_type)> const &b_coefficient)
    {
        b_coefficient_ = b_coefficient;
        return *this;
    }

    ode_coefficient_data_config_ptr<fp_type> build()
    {
        return std::make_shared<ode_coefficient_data_config<fp_type>>(a_coefficient_, b_coefficient_);
    }
};

template <typename fp_type>
using ode_coefficient_data_config_builder_ptr = sptr_t<ode_coefficient_data_config_builder<fp_type>>;

/**
    ode_nonhom_data_config_builder structure
 */
template <typename fp_type> struct ode_nonhom_data_config_builder
{
  private:
    std::function<fp_type(fp_type)> nonhom_fun_;

  public:
    ode_nonhom_data_config_builder &nonhom_function(std::function<fp_type(fp_type)> const &nonhom_function)
    {
        nonhom_fun_ = nonhom_function;
        return *this;
    }

    ode_nonhom_data_config_ptr<fp_type> build()
    {
        return std::make_shared<ode_nonhom_data_config<fp_type>>(nonhom_fun_);
    }
};

template <typename fp_type> using ode_nonhom_data_config_builder_ptr = sptr_t<ode_nonhom_data_config_builder<fp_type>>;

/**
    ode_data_config_builder structure
 */
template <typename fp_type> struct ode_data_config_builder
{
  private:
    ode_coefficient_data_config_ptr<fp_type> coefficient_data_cfg_;
    ode_nonhom_data_config_ptr<fp_type> nonhom_data_cfg_;

  public:
    ode_data_config_builder &coefficient_data_config(
        ode_coefficient_data_config_ptr<fp_type> const &coefficient_data_config)
    {
        coefficient_data_cfg_ = coefficient_data_config;
        return *this;
    }

    ode_data_config_builder &nonhom_data_config(ode_nonhom_data_config_ptr<fp_type> const &nonhom_data_config)
    {
        nonhom_data_cfg_ = nonhom_data_config;
        return *this;
    }

    ode_data_config_ptr<fp_type> build()
    {
        return std::make_shared<ode_data_config<fp_type>>(coefficient_data_cfg_, nonhom_data_cfg_);
    }
};

template <typename fp_type> using ode_data_config_builder_ptr = sptr_t<ode_data_config_builder<fp_type>>;

} // namespace lss_ode_solvers

#endif ///_LSS_ODE_DATA_CONFIG_BUILDER_HPP_
