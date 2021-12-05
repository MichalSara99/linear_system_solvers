#pragma once
#if !defined(_LSS_THOMAS_LU_BOUNDARY_HPP_)
#define _LSS_THOMAS_LU_BOUNDARY_HPP_

#include <type_traits>
#include <vector>

#include "boundaries/lss_boundary.hpp"
#include "boundaries/lss_dirichlet_boundary.hpp"
#include "boundaries/lss_neumann_boundary.hpp"
#include "boundaries/lss_robin_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "common/lss_utility.hpp"

namespace lss_thomas_lu_solver
{

using lss_boundary::boundary_pair;
using lss_boundary::boundary_ptr;
using lss_boundary::dirichlet_boundary;
using lss_boundary::neumann_boundary;
using lss_boundary::robin_boundary;
using lss_utility::sptr_t;

template <typename fp_type> class thomas_lu_solver_boundary
{
  private:
    std::tuple<fp_type, fp_type, fp_type, fp_type> lowest_quad_;
    std::tuple<fp_type, fp_type, fp_type, fp_type> lower_quad_;
    std::tuple<fp_type, fp_type, fp_type, fp_type> higher_quad_;
    std::tuple<fp_type, fp_type, fp_type, fp_type> highest_quad_;
    fp_type space_step_;
    fp_type beta_, gamma_, r_, z_;
    fp_type alpha_n_, beta_n_, r_n_, upper_;
    std::size_t start_index_, end_index_;
    std::size_t discretization_size_;

    explicit thomas_lu_solver_boundary() = delete;
    template <typename... fp_space_types>
    void initialise(boundary_pair<fp_type, fp_space_types...> const &boundary, fp_type time,
                    fp_space_types... space_args);
    template <typename... fp_space_types>
    void finalise(boundary_pair<fp_type, fp_space_types...> const &boundary, fp_type time,
                  fp_space_types... space_args);

  public:
    typedef fp_type value_type;
    explicit thomas_lu_solver_boundary(const std::size_t discretization_size, const fp_type &space_step)
        : discretization_size_{discretization_size}, space_step_{space_step}
    {
    }

    ~thomas_lu_solver_boundary()
    {
    }

    inline void set_lowest_quad(const std::tuple<fp_type, fp_type, fp_type, fp_type> &lowest_quad)
    {
        lowest_quad_ = lowest_quad;
    }

    inline void set_lower_quad(const std::tuple<fp_type, fp_type, fp_type, fp_type> &lower_quad)
    {
        lower_quad_ = lower_quad;
    }

    inline void set_highest_quad(const std::tuple<fp_type, fp_type, fp_type, fp_type> &highest_quad)
    {
        highest_quad_ = highest_quad;
    }

    inline void set_higher_quad(const std::tuple<fp_type, fp_type, fp_type, fp_type> &higher_quad)
    {
        higher_quad_ = higher_quad;
    }

    template <typename... fp_space_types>
    const std::tuple<fp_type, fp_type, fp_type, fp_type> init_coefficients(
        boundary_pair<fp_type, fp_space_types...> const &boundary, fp_type time, fp_space_types... space_args)
    {
        initialise(boundary, time, space_args...);
        return std::make_tuple(beta_, gamma_, r_, z_);
    }

    template <typename... fp_space_types>
    const std::tuple<fp_type, fp_type, fp_type> final_coefficients(
        boundary_pair<fp_type, fp_space_types...> const &boundary, fp_type time, fp_space_types... space_args)
    {
        finalise(boundary, time, space_args...);
        return std::make_tuple(alpha_n_, beta_n_, r_n_);
    }

    std::size_t start_index() const
    {
        return start_index_;
    }

    std::size_t end_index() const
    {
        return end_index_;
    }

    template <typename... fp_space_types>
    const fp_type upper_boundary(boundary_pair<fp_type, fp_space_types...> const &boundary, fp_type time,
                                 fp_space_types... space_args);
    template <typename... fp_space_types>
    const fp_type lower_boundary(boundary_pair<fp_type, fp_space_types...> const &boundary, fp_type time,
                                 fp_space_types... space_args);
};

template <typename fp_type>
template <typename... fp_space_types>
void thomas_lu_solver_boundary<fp_type>::initialise(boundary_pair<fp_type, fp_space_types...> const &boundary,
                                                    fp_type time, fp_space_types... space_args)
{
    const auto a_0 = std::get<0>(lowest_quad_);
    const auto b_0 = std::get<1>(lowest_quad_);
    const auto c_0 = std::get<2>(lowest_quad_);
    const auto f_0 = std::get<3>(lowest_quad_);
    const auto a_1 = std::get<0>(lower_quad_);
    const auto b_1 = std::get<1>(lower_quad_);
    const auto c_1 = std::get<2>(lower_quad_);
    const auto f_1 = std::get<3>(lower_quad_);
    const fp_type two = static_cast<fp_type>(2.0);
    auto const first_bnd = boundary.first;
    if (auto ptr = std::dynamic_pointer_cast<dirichlet_boundary<fp_type, fp_space_types...>>(first_bnd))
    {
        const auto cst_val = ptr->value(time, space_args...);
        start_index_ = 1;
        beta_ = b_1;
        gamma_ = c_1 / beta_;
        r_ = f_1 - a_1 * cst_val;
        z_ = r_ / beta_;
    }
    else if (auto ptr = std::dynamic_pointer_cast<neumann_boundary<fp_type, fp_space_types...>>(first_bnd))
    {
        const auto cst_val = two * space_step_ * ptr->value(time, space_args...);
        start_index_ = 0;
        beta_ = b_0;
        gamma_ = (a_0 + c_0) / beta_;
        r_ = f_0 - a_0 * cst_val;
        z_ = r_ / beta_;
    }
    else if (auto ptr = std::dynamic_pointer_cast<robin_boundary<fp_type, fp_space_types...>>(first_bnd))
    {
        const auto lin_val = two * space_step_ * ptr->linear_value(time, space_args...);
        const auto cst_val = two * space_step_ * ptr->value(time, space_args...);
        start_index_ = 0;
        beta_ = b_0 + a_0 * lin_val;
        gamma_ = (a_0 + c_0) / beta_;
        r_ = f_0 - a_0 * cst_val;
        z_ = r_ / beta_;
    }
    else
    {
        // throw here unrecognized lower boundary
    }
}

template <typename fp_type>
template <typename... fp_space_types>
void thomas_lu_solver_boundary<fp_type>::finalise(boundary_pair<fp_type, fp_space_types...> const &boundary,
                                                  fp_type time, fp_space_types... space_args)
{
    const auto a = std::get<0>(higher_quad_);
    const auto b = std::get<1>(higher_quad_);
    const auto c = std::get<2>(higher_quad_);
    const auto f = std::get<3>(higher_quad_);
    const auto a_end = std::get<0>(highest_quad_);
    const auto b_end = std::get<1>(highest_quad_);
    const auto c_end = std::get<2>(highest_quad_);
    const auto f_end = std::get<3>(highest_quad_);
    const fp_type two = static_cast<fp_type>(2.0);
    auto const second_bnd = boundary.second;
    if (auto ptr = std::dynamic_pointer_cast<dirichlet_boundary<fp_type, fp_space_types...>>(second_bnd))
    {
        const auto cst_val = ptr->value(time, space_args...);
        end_index_ = discretization_size_ - 2;
        alpha_n_ = a;
        beta_n_ = b;
        r_n_ = f - c * cst_val;
    }
    else if (auto ptr = std::dynamic_pointer_cast<neumann_boundary<fp_type, fp_space_types...>>(second_bnd))
    {
        const auto cst_val = two * space_step_ * ptr->value(time, space_args...);
        end_index_ = discretization_size_ - 1;
        alpha_n_ = a_end + c_end;
        beta_n_ = b_end;
        r_n_ = f_end + c_end * cst_val;
    }
    else if (auto ptr = std::dynamic_pointer_cast<robin_boundary<fp_type, fp_space_types...>>(second_bnd))
    {
        const auto lin_val = two * space_step_ * ptr->linear_value(time, space_args...);
        const auto cst_val = two * space_step_ * ptr->value(time, space_args...);
        end_index_ = discretization_size_ - 1;
        alpha_n_ = a_end + c_end;
        beta_n_ = b_end - c_end * lin_val;
        r_n_ = f_end + c_end * cst_val;
    }
    else
    {
        // throw here unrecognized upper boundary
    }
}

template <typename fp_type>
template <typename... fp_space_types>
const fp_type thomas_lu_solver_boundary<fp_type>::upper_boundary(
    boundary_pair<fp_type, fp_space_types...> const &boundary, fp_type time, fp_space_types... space_args)
{
    fp_type ret{};
    auto const second_bnd = boundary.second;
    if (auto ptr = std::dynamic_pointer_cast<dirichlet_boundary<fp_type, fp_space_types...>>(second_bnd))
    {
        ret = ptr->value(time, space_args...);
    }

    return ret;
}

template <typename fp_type>
template <typename... fp_space_types>
const fp_type thomas_lu_solver_boundary<fp_type>::lower_boundary(
    boundary_pair<fp_type, fp_space_types...> const &boundary, fp_type time, fp_space_types... space_args)
{
    fp_type ret{};
    if (auto ptr = std::dynamic_pointer_cast<dirichlet_boundary<fp_type, fp_space_types...>>(boundary.first))
    {
        ret = ptr->value(time, space_args...);
    }
    return ret;
}

template <typename fp_type> using thomas_lu_solver_boundary_ptr = sptr_t<thomas_lu_solver_boundary<fp_type>>;

} // namespace lss_thomas_lu_solver

#endif ///_LSS_THOMAS_LU_BOUNDARY_HPP_
