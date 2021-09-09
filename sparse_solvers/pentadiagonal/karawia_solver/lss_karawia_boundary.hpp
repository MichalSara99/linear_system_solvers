#pragma once
#if !defined(_LSS_KARAWIA_BOUNDARY_HPP_)
#define _LSS_KARAWIA_BOUNDARY_HPP_

#include <type_traits>
#include <vector>

#include "boundaries/lss_boundary.hpp"
#include "boundaries/lss_dirichlet_boundary.hpp"
#include "boundaries/lss_neumann_boundary.hpp"
#include "boundaries/lss_robin_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "common/lss_utility.hpp"

namespace lss_karawia_solver
{

using lss_boundary::boundary_pair;
using lss_boundary::boundary_ptr;
using lss_boundary::dirichlet_boundary;
using lss_boundary::neumann_boundary;
using lss_boundary::robin_boundary;

template <typename fp_type> using sixtuple = std::tuple<fp_type, fp_type, fp_type, fp_type, fp_type, fp_type>;

template <typename fp_type> class karawia_solver_boundary
{
  private:
    sixtuple<fp_type> lowest_sexta_;
    sixtuple<fp_type> lower_sexta_;
    sixtuple<fp_type> higher_sexta_;
    sixtuple<fp_type> highest_sexta_;
    fp_type space_step_;
    fp_type r0_, r1_, r_, rend_;
    std::size_t start_index_, end_index_;
    std::size_t discretization_size_;

    explicit karawia_solver_boundary() = delete;
    template <typename... fp_space_types>
    void initialise(boundary_pair<fp_type, fp_space_types...> const &lowest_boundary,
                    boundary_pair<fp_type, fp_space_types...> const &lower_boundary, fp_type time,
                    fp_space_types... space_args);
    template <typename... fp_space_types>
    void finalise(boundary_pair<fp_type, fp_space_types...> const &uppest_boundary,
                  boundary_pair<fp_type, fp_space_types...> const &upper_boundary, fp_type time,
                  fp_space_types... space_args);

  public:
    typedef fp_type value_type;
    explicit karawia_solver_boundary(const sixtuple<fp_type> &lowest_sexta, const sixtuple<fp_type> &lower_sexta,
                                     const sixtuple<fp_type> &higher_sexta, const sixtuple<fp_type> &highest_sexta,
                                     const std::size_t discretization_size, const fp_type &space_step)
        : lowest_sexta_{lowest_sexta}, lower_sexta_{lower_sexta}, higher_sexta_{higher_sexta},
          highest_sexta_{highest_sexta}, discretization_size_{discretization_size}, space_step_{space_step}
    {
    }

    ~karawia_solver_boundary()
    {
    }

    template <typename... fp_space_types>
    const std::tuple<fp_type, fp_type> init_coefficients(
        boundary_pair<fp_type, fp_space_types...> const &lowest_boundary,
        boundary_pair<fp_type, fp_space_types...> const &lower_boundary, fp_type time, fp_space_types... space_args)
    {
        initialise(lowest_boundary, lower_boundary, time, space_args...);
        return std::make_tuple(r0_, r1_);
    }

    template <typename... fp_space_types>
    const std::tuple<fp_type, fp_type> final_coefficients(
        boundary_pair<fp_type, fp_space_types...> const &uppest_boundary,
        boundary_pair<fp_type, fp_space_types...> const &upper_boundary, fp_type time, fp_space_types... space_args)
    {
        finalise(uppest_boundary, upper_boundary, time, space_args...);
        return std::make_tuple(r_, rend_);
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
void karawia_solver_boundary<fp_type>::initialise(boundary_pair<fp_type, fp_space_types...> const &lowest_boundary,
                                                  boundary_pair<fp_type, fp_space_types...> const &lower_boundary,
                                                  fp_type time, fp_space_types... space_args)
{
    const auto a_2 = std::get<0>(lowest_sexta_);
    const auto b_2 = std::get<1>(lowest_sexta_);
    const auto f_2 = std::get<5>(lowest_sexta_);
    const auto a_3 = std::get<0>(lower_sexta_);
    const auto f_3 = std::get<5>(lower_sexta_);
    auto const lower_bnd = lower_boundary.first;
    auto const lowest_bnd = lowest_boundary.first;
    if (auto ptr_0 = std::dynamic_pointer_cast<dirichlet_boundary<fp_type, fp_space_types...>>(lowest_bnd))
    {
        if (auto ptr_1 = std::dynamic_pointer_cast<dirichlet_boundary<fp_type, fp_space_types...>>(lower_bnd))
        {
            const auto cst_val_0 = ptr_0->value(time, space_args...);
            const auto cst_val_1 = ptr_1->value(time, space_args...);
            start_index_ = 2;
            r0_ = f_2 - b_2 * cst_val_1 - a_2 * cst_val_0;
            r1_ = f_3 - a_3 * cst_val_1;
        }
        else
        {
            throw std::exception("Any other boundary type is not supported");
        }
    }
    else
    {
        throw std::exception("Any other boundary type is not supported");
    }
}

template <typename fp_type>
template <typename... fp_space_types>
void karawia_solver_boundary<fp_type>::finalise(boundary_pair<fp_type, fp_space_types...> const &uppest_boundary,
                                                boundary_pair<fp_type, fp_space_types...> const &upper_boundary,
                                                fp_type time, fp_space_types... space_args)
{
    const auto e = std::get<4>(higher_sexta_);
    const auto f = std::get<5>(higher_sexta_);
    const auto d_end = std::get<3>(highest_sexta_);
    const auto e_end = std::get<4>(highest_sexta_);
    const auto f_end = std::get<5>(highest_sexta_);
    auto const upper_bnd = upper_boundary.second;
    auto const uppest_bnd = uppest_boundary.second;
    if (auto ptr_end = std::dynamic_pointer_cast<dirichlet_boundary<fp_type, fp_space_types...>>(uppest_bnd))
    {
        if (auto ptr = std::dynamic_pointer_cast<dirichlet_boundary<fp_type, fp_space_types...>>(upper_bnd))
        {
            const auto cst_val = ptr->value(time, space_args...);
            const auto cst_val_end = ptr_end->value(time, space_args...);
            end_index_ = discretization_size_ - 3;
            r_ = f - e * cst_val;
            rend_ = f_end - d_end * cst_val - e_end * cst_val_end;
        }
        else
        {
            throw std::exception("Any other boundary type is not supported");
        }
    }
    else
    {
        throw std::exception("Any other boundary type is not supported");
    }
}

template <typename fp_type>
template <typename... fp_space_types>
const fp_type karawia_solver_boundary<fp_type>::upper_boundary(
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
const fp_type karawia_solver_boundary<fp_type>::lower_boundary(
    boundary_pair<fp_type, fp_space_types...> const &boundary, fp_type time, fp_space_types... space_args)
{
    fp_type ret{};
    if (auto ptr = std::dynamic_pointer_cast<dirichlet_boundary<fp_type, fp_space_types...>>(boundary.first))
    {
        ret = ptr->value(time, space_args...);
    }
    return ret;
}

} // namespace lss_karawia_solver

#endif ///_LSS_KARAWIA_BOUNDARY_HPP_
