#pragma once
#if !defined(_LSS_SOR_BOUNDARY_HPP_)
#define _LSS_SOR_BOUNDARY_HPP_

#pragma warning(disable : 4244)

#include "boundaries/lss_boundary.hpp"
#include "boundaries/lss_dirichlet_boundary.hpp"
#include "boundaries/lss_neumann_boundary.hpp"
#include "boundaries/lss_robin_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "common/lss_utility.hpp"

namespace lss_sor_solver
{

using lss_boundary::boundary_pair;
using lss_boundary::dirichlet_boundary;
using lss_boundary::neumann_boundary;
using lss_boundary::robin_boundary;

template <typename fp_type> class sor_boundary
{
  private:
    std::tuple<fp_type, fp_type, fp_type, fp_type> lowest_quad_;
    std::tuple<fp_type, fp_type, fp_type, fp_type> lower_quad_;
    std::tuple<fp_type, fp_type, fp_type, fp_type> higher_quad_;
    std::tuple<fp_type, fp_type, fp_type, fp_type> highest_quad_;
    fp_type space_step_;
    fp_type b_init_, c_init_, f_init_;
    fp_type a_end_, b_end_, f_end_;
    std::size_t start_index_, end_index_;
    std::size_t discretization_size_;

    explicit sor_boundary() = delete;

    template <typename... fp_space_types>
    void initialise(boundary_pair<fp_type, fp_space_types...> const &boundary, fp_type time,
                    fp_space_types... space_args);

    template <typename... fp_space_types>
    void finalise(boundary_pair<fp_type, fp_space_types...> const &boundary, fp_type time,
                  fp_space_types... space_args);

  public:
    typedef fp_type value_type;
    explicit sor_boundary(const std::tuple<fp_type, fp_type, fp_type, fp_type> &lowest_quad,
                          const std::tuple<fp_type, fp_type, fp_type, fp_type> &lower_quad,
                          const std::tuple<fp_type, fp_type, fp_type, fp_type> &higher_quad,
                          const std::tuple<fp_type, fp_type, fp_type, fp_type> &highest_quad,
                          const std::size_t discretization_size, const fp_type &space_step)
        : lowest_quad_{lowest_quad}, lower_quad_{lower_quad}, higher_quad_{higher_quad}, highest_quad_{highest_quad},
          discretization_size_{discretization_size}, space_step_{space_step}
    {
    }

    ~sor_boundary()
    {
    }

    template <typename... fp_space_types>
    const std::tuple<fp_type, fp_type, fp_type> init_coefficients(
        boundary_pair<fp_type, fp_space_types...> const &boundary, fp_type time, fp_space_types... space_args)
    {
        initialise(boundary, time, space_args...);
        return std::make_tuple(b_init_, c_init_, f_init_);
    }

    template <typename... fp_space_types>
    const std::tuple<fp_type, fp_type, fp_type> final_coefficients(
        boundary_pair<fp_type, fp_space_types...> const &boundary, fp_type time, fp_space_types... space_args)
    {
        finalise(boundary, time, space_args...);
        return std::make_tuple(a_end_, b_end_, f_end_);
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
void sor_boundary<fp_type>::initialise(boundary_pair<fp_type, fp_space_types...> const &boundary, fp_type time,
                                       fp_space_types... space_args)
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
    auto const &first_bnd = boundary.first;
    if (auto ptr = std::dynamic_pointer_cast<dirichlet_boundary<fp_type, fp_space_types...>>(first_bnd))
    {
        const auto cst_val = ptr->value(time, space_args...);
        start_index_ = 1;
        b_init_ = b_1;
        c_init_ = c_1;
        f_init_ = f_1 - a_1 * cst_val;
    }
    else if (auto ptr = std::dynamic_pointer_cast<neumann_boundary<fp_type, fp_space_types...>>(first_bnd))
    {
        const auto cst_val = two * space_step_ * ptr->value(time, space_args...);
        start_index_ = 0;
        b_init_ = b_0;
        c_init_ = a_0 + c_0;
        f_init_ = f_0 - a_0 * cst_val;
    }
    else if (auto ptr = std::dynamic_pointer_cast<robin_boundary<fp_type, fp_space_types...>>(first_bnd))
    {
        const auto lin_val = two * space_step_ * ptr->linear_value(time, space_args...);
        const auto cst_val = two * space_step_ * ptr->value(time, space_args...);
        start_index_ = 0;
        b_init_ = b_0 + a_0 * cst_val;
        c_init_ = a_0 + c_0;
        f_init_ = f_0 - a_0 * cst_val;
    }
    else
    {
        // throw here unrecognized lower boundary
    }
}

template <typename fp_type>
template <typename... fp_space_types>
void sor_boundary<fp_type>::finalise(boundary_pair<fp_type, fp_space_types...> const &boundary, fp_type time,
                                     fp_space_types... space_args)
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
    auto const &second_bnd = boundary.second;
    if (auto ptr = std::dynamic_pointer_cast<dirichlet_boundary<fp_type, fp_space_types...>>(second_bnd))
    {
        const auto cst_val = ptr->value(time, space_args...);
        end_index_ = discretization_size_ - 2;
        a_end_ = a;
        b_end_ = b;
        f_end_ = f - c * cst_val;
    }
    else if (auto ptr = std::dynamic_pointer_cast<neumann_boundary<fp_type, fp_space_types...>>(second_bnd))
    {
        const auto cst_val = two * space_step_ * ptr->value(time, space_args...);
        end_index_ = discretization_size_ - 1;
        a_end_ = a_end + c_end;
        b_end_ = b_end;
        f_end_ = f_end + c_end * cst_val;
    }
    else if (auto ptr = std::dynamic_pointer_cast<robin_boundary<fp_type, fp_space_types...>>(second_bnd))
    {
        const auto lin_val = two * space_step_ * ptr->linear_value(time, space_args...);
        const auto cst_val = two * space_step_ * ptr->value(time, space_args...);
        end_index_ = discretization_size_ - 1;
        a_end_ = a_end + c_end;
        b_end_ = b_end - c_end * lin_val;
        f_end_ = f_end + c_end * cst_val;
    }
    else
    {
        // throw here unrecognized upper boundary
    }
}

template <typename fp_type>
template <typename... fp_space_types>
const fp_type sor_boundary<fp_type>::upper_boundary(boundary_pair<fp_type, fp_space_types...> const &boundary,
                                                    fp_type time, fp_space_types... space_args)
{
    fp_type ret{};
    if (auto ptr = std::dynamic_pointer_cast<dirichlet_boundary<fp_type, fp_space_types...>>(boundary.second))
    {
        ret = ptr->value(time, space_args...);
    }

    return ret;
}

template <typename fp_type>
template <typename... fp_space_types>
const fp_type sor_boundary<fp_type>::lower_boundary(boundary_pair<fp_type, fp_space_types...> const &boundary,
                                                    fp_type time, fp_space_types... space_args)
{
    fp_type ret{};
    if (auto ptr = std::dynamic_pointer_cast<dirichlet_boundary<fp_type, fp_space_types...>>(boundary.first))
    {
        ret = ptr->value(time, space_args...);
    }
    return ret;
}

} // namespace lss_sor_solver

#endif ///_LSS_SOR_BOUNDARY_HPP_
