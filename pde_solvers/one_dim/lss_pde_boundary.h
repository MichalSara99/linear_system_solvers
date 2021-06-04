#pragma once
#if !defined(_LSS_1D_PDE_BOUNDARY)
#define _LSS_1D_PDE_BOUNDARY

#include <functional>
#include <tuple>
#include <vector>

#include "common/lss_macros.h"
#include "common/lss_utility.h"

namespace lss_one_dim_pde_boundary {

/*!
Represents 1D Dirichlet boundary
*/
template <typename fp_type>
struct dirichlet_boundary_1d {
 private:
  explicit dirichlet_boundary_1d() {}

 public:
  typedef std::function<fp_type(fp_type)> fun_1d;
  fun_1d first;
  fun_1d second;
  /*!
  first_fun:

  u(x_1,t) = A(t)

  second_fun:

  u(x_2,t) = B(t)
  */
  explicit dirichlet_boundary_1d(fun_1d const &first_fun,
                                 fun_1d const &second_fun)
      : first{first_fun}, second{second_fun} {}

  /*!
  Populate solution with Dirichlet boundary

  \param time
  \param solution
  */
  template <template <typename, typename> typename container, typename alloc>
  void fill(fp_type const &time, container<fp_type, alloc> &solution) {
    LSS_ASSERT(!solution.empty(), "solution must not be empty.");
    solution[0] = first(time);
    solution[solution.size() - 1] = second(time);
  }
};

/*!
  Constant coefficients used in Robin boundaries
  l_p_g = lambda + gamma
  l_m_g = lambda - gamma
  o_m_2l_m_d = 1 - (2 * lambda - delta)
  t_coeff = k
 */
template <typename fp_type>
struct robin_boundary_coeffs {
  fp_type l_p_g;
  fp_type l_m_g;
  fp_type o_m_2l_m_d;
  fp_type t_coeff;

  explicit robin_boundary_coeffs(
      std::tuple<fp_type, fp_type, fp_type, fp_type> const &coeffs)
      : l_p_g{std::get<0>(coeffs)},
        l_m_g{std::get<1>(coeffs)},
        o_m_2l_m_d{std::get<2>(coeffs)},
        t_coeff{std::get<3>(coeffs)} {}
};

/*!
Represents 1D Robin boundary
*/
template <typename fp_type>
struct robin_boundary_1d {
 private:
  explicit robin_boundary_1d() {}
  std::pair<fp_type, fp_type> const converted_right() const {
    fp_type const beta_ = static_cast<fp_type>(1.0) / right.first;
    fp_type const psi_ = static_cast<fp_type>(-1.0) * right.second * beta_;
    return std::make_pair(beta_, psi_);
  }

 public:
  std::pair<fp_type, fp_type> left;
  std::pair<fp_type, fp_type> right;

  /*!
  left_boundary:

  U_{0} = alpha * U_{1} + phi

  right_boundary:

  U_{N-1} = beta * U_{N} + psi
  */
  explicit robin_boundary_1d(std::pair<fp_type, fp_type> const &left_boundary,
                             std::pair<fp_type, fp_type> const &right_boundary)
      : left{left_boundary}, right{right_boundary} {}

  /*!
  Populate solution with Robin boundary

  \param robin_coeffs
  \param time
  \param heat_source
  \param prev_solution
  \param next_solution
  */
  template <template <typename, typename> typename container, typename alloc>
  void fill(robin_boundary_coeffs<fp_type> const &robin_coeffs,
            fp_type const &time, container<fp_type, alloc> const &heat_source,
            container<fp_type, alloc> const &prev_solution,
            container<fp_type, alloc> &next_solution) {
    LSS_ASSERT(!heat_source.empty(), "heat_source must not be empty.");
    LSS_ASSERT(!prev_solution.empty(), "prev_solution must not be empty.");
    LSS_ASSERT(!next_solution.empty(), "next_solution must not be empty.");
    auto const right_ = converted_right();

    next_solution[0] =
        (robin_coeffs.l_p_g + (robin_coeffs.l_m_g * left.first)) *
            prev_solution[1] +
        robin_coeffs.o_m_2l_m_d * prev_solution[0] +
        robin_coeffs.l_m_g * left.second +
        robin_coeffs.t_coeff * heat_source[0];

    next_solution[next_solution.size() - 1] =
        (robin_coeffs.l_m_g + (robin_coeffs.l_p_g * right_.first)) *
            prev_solution[next_solution.size() - 2] +
        robin_coeffs.o_m_2l_m_d * prev_solution[next_solution.size() - 1] +
        robin_coeffs.l_p_g * right_.second +
        robin_coeffs.t_coeff * heat_source[next_solution.size() - 1];
  }
};

}  // namespace lss_one_dim_pde_boundary

#endif  ///_LSS_1D_PDE_BOUNDARY
