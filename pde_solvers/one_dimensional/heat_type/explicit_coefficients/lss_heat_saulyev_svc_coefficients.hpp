#if !defined(_LSS_HEAT_SAULYEV_SVC_COEFFICIENTS_HPP_)
#define _LSS_HEAT_SAULYEV_SVC_COEFFICIENTS_HPP_

#include <functional>

#include "common/lss_macros.hpp"
#include "pde_solvers/one_dimensional/heat_type/implicit_coefficients/lss_1d_general_svc_heat_equation_implicit_coefficients.hpp"

namespace lss_pde_solvers
{

namespace one_dimensional
{
using lss_utility::sptr_t;

template <typename fp_type> struct heat_saulyev_svc_coefficients
{
  public:
    // scheme coefficients:
    fp_type h_, k_;
    std::size_t space_size_;
    // functional coefficients:
    std::function<fp_type(fp_type)> A_;
    std::function<fp_type(fp_type)> B_;
    std::function<fp_type(fp_type)> D_;
    std::function<fp_type(fp_type)> K_;

  private:
    void initialize_coefficients(general_svc_heat_equation_implicit_coefficients_ptr<fp_type> const &coefficients)
    {
        const fp_type one = static_cast<fp_type>(1.0);
        // save coefficients locally:
        auto const a = coefficients->A_;
        auto const b = coefficients->B_;
        auto const d = coefficients->D_;
        h_ = coefficients->h_;
        k_ = coefficients->k_;

        A_ = [=](fp_type x) { return (a(x) / (one + b(x))); };
        B_ = [=](fp_type x) { return ((one - b(x)) / (one + b(x))); };
        D_ = [=](fp_type x) { return (d(x) / (one + b(x))); };
        K_ = [=](fp_type x) { return (k_ / (one + b(x))); };

        space_size_ = coefficients->space_size_;
    }

  public:
    heat_saulyev_svc_coefficients() = delete;

    explicit heat_saulyev_svc_coefficients(
        general_svc_heat_equation_implicit_coefficients_ptr<fp_type> const &coefficients)
    {
        initialize_coefficients(coefficients);
    }
};

template <typename fp_type> using heat_saulyev_svc_coefficients_ptr = sptr_t<heat_saulyev_svc_coefficients<fp_type>>;

} // namespace one_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_HEAT_SAULYEV_SVC_COEFFICIENTS_HPP_
