#if !defined(_LSS_WAVE_IMPLICIT_COEFFICIENTS_HPP_)
#define _LSS_WAVE_IMPLICIT_COEFFICIENTS_HPP_

#include <functional>

#include "common/lss_macros.hpp"
#include "containers/lss_container_2d.hpp"
#include "discretization/lss_discretization.hpp"
#include "pde_solvers/lss_pde_discretization_config.hpp"
#include "pde_solvers/transformation/lss_wave_data_transform.hpp"

namespace lss_pde_solvers
{

namespace one_dimensional
{
using lss_utility::range;
using lss_utility::sptr_t;

template <typename fp_type> struct wave_implicit_coefficients
{
  public:
    // scheme coefficients:
    fp_type lambda_, gamma_, delta_, rho_, k_;
    std::size_t space_size_;
    range<fp_type> range_;
    // functional coefficients:
    std::function<fp_type(fp_type, fp_type)> A_;
    std::function<fp_type(fp_type, fp_type)> B_;
    std::function<fp_type(fp_type, fp_type)> C_;
    std::function<fp_type(fp_type, fp_type)> D_;
    std::function<fp_type(fp_type, fp_type)> E_;

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
        const fp_type h = one / (space_size_ - 1);
        // calculate scheme coefficients:
        lambda_ = one / (k_ * k_);
        gamma_ = one / (two * k_);
        delta_ = one / (h * h);
        rho_ = one / (two * h);
    }

    void initialize_coefficients(wave_data_transform_1d_ptr<fp_type> const &wave_data_config)
    {
        // save coefficients locally:
        auto const a = wave_data_config->a_coefficient();
        auto const b = wave_data_config->b_coefficient();
        auto const c = wave_data_config->c_coefficient();
        auto const d = wave_data_config->d_coefficient();

        const fp_type half = static_cast<fp_type>(0.5);
        const fp_type quater = static_cast<fp_type>(0.25);

        A_ = [=](fp_type t, fp_type x) { return quater * (delta_ * b(t, x) - rho_ * c(t, x)); };
        B_ = [=](fp_type t, fp_type x) { return quater * (delta_ * b(t, x) + rho_ * c(t, x)); };
        C_ = [=](fp_type t, fp_type x) { return half * (delta_ * b(t, x) - half * d(t, x)); };
        D_ = [=](fp_type t, fp_type x) { return (lambda_ - gamma_ * a(t, x)); };
        E_ = [=](fp_type t, fp_type x) { return (lambda_ + gamma_ * a(t, x)); };
    }

  public:
    wave_implicit_coefficients() = delete;

    explicit wave_implicit_coefficients(wave_data_transform_1d_ptr<fp_type> const &wave_data_config,
                                        pde_discretization_config_1d_ptr<fp_type> const &discretization_config)
    {
        initialize(discretization_config);
        initialize_coefficients(wave_data_config);
    }
};

template <typename fp_type> using wave_implicit_coefficients_ptr = sptr_t<wave_implicit_coefficients<fp_type>>;

} // namespace one_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_WAVE_IMPLICIT_COEFFICIENTS_HPP_
