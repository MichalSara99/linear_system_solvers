#if !defined(_LSS_GENERAL_ODE_IMPLICIT_COEFFICIENTS_HPP_)
#define _LSS_GENERAL_ODE_IMPLICIT_COEFFICIENTS_HPP_

#include <functional>

#include "common/lss_macros.hpp"
#include "containers/lss_container_2d.hpp"
#include "discretization/lss_discretization.hpp"
#include "ode_solvers/lss_ode_data_config.hpp"
#include "ode_solvers/lss_ode_discretization_config.hpp"

namespace lss_ode_solvers
{

using lss_utility::range;
using lss_utility::sptr_t;

template <typename fp_type> struct general_ode_implicit_coefficients
{
  public:
    // scheme coefficients:
    fp_type lambda_, gamma_, h_;
    std::size_t space_size_;
    range<fp_type> range_;
    // functional coefficients:
    std::function<fp_type(fp_type)> A_;
    std::function<fp_type(fp_type)> B_;
    std::function<fp_type(fp_type)> C_;

  private:
    void initialize(ode_discretization_config_ptr<fp_type> const &discretization_config)
    {
        // get space range:
        range_ = discretization_config->space_range();
        // get space step:
        h_ = discretization_config->space_step();
        // size of spaces discretization:
        space_size_ = discretization_config->number_of_space_points();
        const fp_type one = static_cast<fp_type>(1.0);
        const fp_type two = static_cast<fp_type>(2.0);
        // calculate scheme coefficients:
        lambda_ = one / (h_ * h_);
        gamma_ = one / (two * h_);
    }

    void initialize_coefficients(ode_data_config_ptr<fp_type> const &ode_data_config)
    {
        // save coefficients locally:
        auto const a = ode_data_config->a_coefficient();
        auto const b = ode_data_config->b_coefficient();
        const fp_type two = static_cast<fp_type>(2.0);
        A_ = [=](fp_type x) { return (lambda_ - gamma_ * a(x)); };
        B_ = [=](fp_type x) { return (lambda_ + gamma_ * a(x)); };
        C_ = [=](fp_type x) { return (b(x) - two * lambda_); };
    }

  public:
    general_ode_implicit_coefficients() = delete;

    explicit general_ode_implicit_coefficients(ode_data_config_ptr<fp_type> const &ode_data_config,
                                               ode_discretization_config_ptr<fp_type> const &discretization_config)
    {
        initialize(discretization_config);
        initialize_coefficients(ode_data_config);
    }
};

template <typename fp_type>
using general_ode_implicit_coefficients_ptr = sptr_t<general_ode_implicit_coefficients<fp_type>>;

} // namespace lss_ode_solvers

#endif ///_LSS_GENERAL_ODE_IMPLICIT_COEFFICIENTS_HPP_
