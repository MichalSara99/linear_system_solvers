#if !defined(_LSS_1D_GENERAL_HEAT_EQUATION_COEFFICIENTS_HPP_)
#define _LSS_1D_GENERAL_HEAT_EQUATION_COEFFICIENTS_HPP_

#include <functional>

#include "common/lss_macros.hpp"
#include "containers/lss_container_2d.hpp"
#include "discretization/lss_discretization.hpp"
#include "pde_solvers/lss_pde_discretization_config.hpp"
#include "pde_solvers/transformation/lss_heat_data_transform.hpp"

namespace lss_pde_solvers
{

namespace one_dimensional
{
using lss_utility::range;
using lss_utility::sptr_t;

template <typename fp_type> struct general_heat_equation_coefficients
{
  public:
    // scheme coefficients:
    fp_type lambda_, gamma_, delta_, k_;
    std::size_t space_size_;
    range<fp_type> range_;
    // theta variable:
    fp_type theta_;
    // functional coefficients:
    std::function<fp_type(fp_type, fp_type)> A_;
    std::function<fp_type(fp_type, fp_type)> B_;
    std::function<fp_type(fp_type, fp_type)> D_;

  private:
    void initialize(pde_discretization_config_1d_ptr<fp_type> const &discretization_config)
    {
        // get space range:
        range_ = discretization_config->space_range();
        // get time step:
        k_ = discretization_config->time_step();
        // size of spaces discretization:
        space_size_ = discretization_config->number_of_space_points();
        const fp_type one = static_cast<fp_type>(1.0);
        const fp_type two = static_cast<fp_type>(2.0);
        const fp_type half = static_cast<fp_type>(0.5);
        auto const h = one / (space_size_ - 1);
        // calculate scheme coefficients:
        lambda_ = k_ / (h * h);
        gamma_ = k_ / (two * h);
        delta_ = half * k_;
    }

    void initialize_coefficients(heat_data_transform_1d_ptr<fp_type> const &heat_data_config)
    {
        // save coefficients locally:
        auto const a = heat_data_config->a_coefficient();
        auto const b = heat_data_config->b_coefficient();
        auto const c = heat_data_config->c_coefficient();

        A_ = [=](fp_type t, fp_type x) { return (lambda_ * a(t, x) - gamma_ * b(t, x)); };
        B_ = [=](fp_type t, fp_type x) { return (lambda_ * a(t, x) - delta_ * c(t, x)); };
        D_ = [=](fp_type t, fp_type x) { return (lambda_ * a(t, x) + gamma_ * b(t, x)); };
    }

  public:
    general_heat_equation_coefficients() = delete;

    explicit general_heat_equation_coefficients(heat_data_transform_1d_ptr<fp_type> const &heat_data_config,
                                                pde_discretization_config_1d_ptr<fp_type> const &discretization_config,
                                                fp_type const &theta)
        : theta_{theta}
    {
        initialize(discretization_config);
        initialize_coefficients(heat_data_config);
    }
};

template <typename fp_type>
using general_heat_equation_coefficients_ptr = sptr_t<general_heat_equation_coefficients<fp_type>>;

} // namespace one_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_1D_GENERAL_HEAT_EQUATION_COEFFICIENTS_HPP_
