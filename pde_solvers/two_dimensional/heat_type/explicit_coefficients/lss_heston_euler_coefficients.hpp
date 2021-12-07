#if !defined(_LSS_HESTON_EULER_COEFFICIENTS_HPP_)
#define _LSS_HESTON_EULER_COEFFICIENTS_HPP_

#include "common/lss_utility.hpp"
#include "pde_solvers/two_dimensional/heat_type/implicit_coefficients/lss_2d_general_heston_equation_coefficients.hpp"

namespace lss_pde_solvers
{
namespace two_dimensional
{

using lss_utility::range;
using lss_utility::sptr_t;

/**
    heston_euler_coefficients object
 */
template <typename fp_type> struct heston_euler_coefficients
{
  public:
    // scheme constant coefficients:
    fp_type rho_, k_;
    std::size_t space_size_x_, space_size_y_;
    // functional coefficients:
    std::function<fp_type(fp_type, fp_type, fp_type)> M_;
    std::function<fp_type(fp_type, fp_type, fp_type)> M_tilde_;
    std::function<fp_type(fp_type, fp_type, fp_type)> P_;
    std::function<fp_type(fp_type, fp_type, fp_type)> P_tilde_;
    std::function<fp_type(fp_type, fp_type, fp_type)> Z_;
    std::function<fp_type(fp_type, fp_type, fp_type)> W_;
    std::function<fp_type(fp_type, fp_type, fp_type)> C_;

  private:
    void initialize_coefficients(general_heston_equation_coefficients_ptr<fp_type> const &coefficients)
    {
        // time step:
        k_ = coefficients->k_;
        // size of spaces discretization:
        space_size_x_ = coefficients->space_size_x_;
        space_size_y_ = coefficients->space_size_y_;
        // calculate scheme coefficients:
        rho_ = coefficients->rho_;
        // save coefficients locally:
        M_ = coefficients->M_;
        M_tilde_ = coefficients->M_tilde_;
        P_ = coefficients->P_;
        P_tilde_ = coefficients->P_tilde_;
        Z_ = coefficients->Z_;
        W_ = coefficients->W_;
        auto const &c = coefficients->C_;
        auto const gamma = coefficients->gamma_;
        C_ = [=](fp_type t, fp_type x, fp_type y) { return (gamma * c(t, x, y)); };
    }

  public:
    heston_euler_coefficients() = delete;

    heston_euler_coefficients(general_heston_equation_coefficients_ptr<fp_type> const &coefficients)
    {
        initialize_coefficients(coefficients);
    }
};

template <typename fp_type> using heston_euler_coefficients_ptr = sptr_t<heston_euler_coefficients<fp_type>>;

} // namespace two_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_HESTON_EULER_COEFFICIENTS_HPP_
