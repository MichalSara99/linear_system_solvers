#if !defined(_LSS_2D_GENERAL_HESTON_EQUATION_COEFFICIENTS_HPP_)
#define _LSS_2D_GENERAL_HESTON_EQUATION_COEFFICIENTS_HPP_

#include "common/lss_utility.hpp"
#include "discretization/lss_discretization.hpp"
#include "pde_solvers/lss_pde_discretization_config.hpp"
#include "pde_solvers/lss_splitting_method_config.hpp"
#include "pde_solvers/transformation/lss_heat_data_transform.hpp"

namespace lss_pde_solvers
{
namespace two_dimensional
{

using lss_utility::range;
using lss_utility::sptr_t;

/**
    general_heston_equation_coefficients object
 */
template <typename fp_type> struct general_heston_equation_coefficients
{
  public:
    // scheme constant coefficients:
    fp_type alpha_, beta_, gamma_, delta_, ni_, rho_, zeta_, k_;
    std::size_t space_size_x_, space_size_y_;
    range<fp_type> rangex_, rangey_;
    // theta variable:
    fp_type theta_;
    // functional coefficients:
    std::function<fp_type(fp_type, fp_type, fp_type)> M_;
    std::function<fp_type(fp_type, fp_type, fp_type)> M_tilde_;
    std::function<fp_type(fp_type, fp_type, fp_type)> P_;
    std::function<fp_type(fp_type, fp_type, fp_type)> P_tilde_;
    std::function<fp_type(fp_type, fp_type, fp_type)> Z_;
    std::function<fp_type(fp_type, fp_type, fp_type)> W_;
    std::function<fp_type(fp_type, fp_type, fp_type)> C_;
    std::function<fp_type(fp_type, fp_type, fp_type)> D_;
    std::function<fp_type(fp_type, fp_type, fp_type)> E_;
    std::function<fp_type(fp_type, fp_type, fp_type)> F_;

  private:
    void initialize(pde_discretization_config_2d_ptr<fp_type> const &discretization_config,
                    splitting_method_config_ptr<fp_type> const &splitting_config)
    {
        // get space ranges:
        const auto &spaces = discretization_config->space_range();
        // across X:
        rangex_ = std::get<0>(spaces);
        // across Y:
        rangey_ = std::get<1>(spaces);
        // time step:
        const fp_type k = discretization_config->time_step();
        // size of spaces discretization:
        const auto &space_sizes = discretization_config->number_of_space_points();
        space_size_x_ = std::get<0>(space_sizes);
        space_size_y_ = std::get<1>(space_sizes);
        // calculate scheme coefficients:
        const fp_type one = static_cast<fp_type>(1.0);
        const fp_type half = static_cast<fp_type>(0.5);
        const fp_type quarter = static_cast<fp_type>(0.25);
        auto const &h_1 = one / (space_size_x_ - 1);
        auto const &h_2 = one / (space_size_y_ - 1);

        k_ = k;
        alpha_ = k_ / (h_1 * h_1);
        beta_ = k_ / (h_2 * h_2);
        gamma_ = quarter * k_ / (h_1 * h_2);
        delta_ = half * k_ / h_1;
        ni_ = half * k_ / h_2;
        rho_ = k;
        if (splitting_config != nullptr)
            zeta_ = splitting_config->weighting_value();
    }

    void initialize_coefficients(heat_data_transform_2d_ptr<fp_type> const &heat_data_config)
    {
        // save coefficients locally:
        auto const a = heat_data_config->a_coefficient();
        auto const b = heat_data_config->b_coefficient();
        auto const c = heat_data_config->c_coefficient();
        auto const d = heat_data_config->d_coefficient();
        auto const e = heat_data_config->e_coefficient();
        auto const f = heat_data_config->f_coefficient();

        const fp_type two = static_cast<fp_type>(2.0);
        const fp_type half = static_cast<fp_type>(0.5);

        M_ = [=](fp_type t, fp_type x, fp_type y) { return (alpha_ * a(t, x, y) - delta_ * d(t, x, y)); };
        M_tilde_ = [=](fp_type t, fp_type x, fp_type y) { return (beta_ * b(t, x, y) - ni_ * e(t, x, y)); };
        P_ = [=](fp_type t, fp_type x, fp_type y) { return (alpha_ * a(t, x, y) + delta_ * d(t, x, y)); };
        P_tilde_ = [=](fp_type t, fp_type x, fp_type y) { return (beta_ * b(t, x, y) + ni_ * e(t, x, y)); };
        Z_ = [=](fp_type t, fp_type x, fp_type y) { return (two * alpha_ * a(t, x, y) - half * rho_ * f(t, x, y)); };
        W_ = [=](fp_type t, fp_type x, fp_type y) { return (two * beta_ * b(t, x, y) - half * rho_ * f(t, x, y)); };
        C_ = [=](fp_type t, fp_type x, fp_type y) { return c(t, x, y); };
        D_ = [=](fp_type t, fp_type x, fp_type y) { return d(t, x, y); };
        E_ = [=](fp_type t, fp_type x, fp_type y) { return e(t, x, y); };
        F_ = [=](fp_type t, fp_type x, fp_type y) { return f(t, x, y); };
    }

  public:
    general_heston_equation_coefficients() = delete;

    general_heston_equation_coefficients(heat_data_transform_2d_ptr<fp_type> const &heat_data_config,
                                         pde_discretization_config_2d_ptr<fp_type> const &discretization_config,
                                         splitting_method_config_ptr<fp_type> const splitting_config,
                                         fp_type const &theta)
        : theta_{theta}
    {
        initialize(discretization_config, splitting_config);
        initialize_coefficients(heat_data_config);
    }
};

template <typename fp_type>
using general_heston_equation_coefficients_ptr = sptr_t<general_heston_equation_coefficients<fp_type>>;

} // namespace two_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_2D_GENERAL_HESTON_EQUATION_COEFFICIENTS_HPP_
