#pragma once
#if !defined(_LSS_DOUBLE_SWEEP_BOUNDARY_HPP_)
#define _LSS_DOUBLE_SWEEP_BOUNDARY_HPP_

#pragma warning(disable : 4244)

#include <type_traits>
#include <vector>

#include "boundaries/lss_boundary.hpp"
#include "boundaries/lss_dirichlet_boundary.hpp"
#include "boundaries/lss_neumann_boundary.hpp"
#include "boundaries/lss_robin_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "common/lss_utility.hpp"

namespace lss_double_sweep_solver
{
using lss_boundary::boundary_pair;
using lss_boundary::dirichlet_boundary;
using lss_boundary::neumann_boundary;
using lss_boundary::robin_boundary;

template <typename fp_type> class double_sweep_boundary
{
  private:
    std::tuple<fp_type, fp_type, fp_type, fp_type> low_quad_;
    fp_type space_step_, l_, k_, upper_;
    std::size_t start_index_;
    std::size_t discretization_size_;

    explicit double_sweep_boundary() = delete;

    template <typename... fp_space_types>
    void initialise(boundary_pair<fp_type, fp_space_types...> const &boundary, fp_type time,
                    fp_space_types... space_args);

    template <typename... fp_space_types>
    void finalise(boundary_pair<fp_type, fp_space_types...> const &boundary, const fp_type &k_nm1, const fp_type &k_n,
                  const fp_type &l_nm1, const fp_type &l_n, fp_type time, fp_space_types... space_args);

  public:
    typedef fp_type value_type;
    explicit double_sweep_boundary(const std::tuple<fp_type, fp_type, fp_type, fp_type> &low_quad,
                                   const std::size_t &discretization_size, const fp_type &space_step)
        : low_quad_{low_quad}, discretization_size_{discretization_size}, space_step_{space_step}
    {
    }
    ~double_sweep_boundary()
    {
    }

    inline std::size_t start_index() const
    {
        return start_index_;
    }

    template <typename... fp_space_types>
    inline std::size_t end_index(boundary_pair<fp_type, fp_space_types...> const &boundary) const
    {
        if (auto ptr = std::dynamic_pointer_cast<dirichlet_boundary<fp_type, fp_space_types...>>(boundary.second))
        {
            return (discretization_size_ - 2);
        }
        return (discretization_size_ - 1);
    }

    template <typename... fp_space_types>
    const std::pair<fp_type, fp_type> coefficients(boundary_pair<fp_type, fp_space_types...> const &boundary,
                                                   fp_type time, fp_space_types... space_args);

    template <typename... fp_space_types>
    const fp_type upper_boundary(boundary_pair<fp_type, fp_space_types...> const &boundary, const fp_type &k_nm1,
                                 const fp_type &k_n, const fp_type &l_nm1, const fp_type &l_n, fp_type time,
                                 fp_space_types... space_args);

    template <typename... fp_space_types>
    const fp_type lower_boundary(boundary_pair<fp_type, fp_space_types...> const &boundary, fp_type time,
                                 fp_space_types... space_args);
};

template <typename fp_type>
template <typename... fp_space_types>
void double_sweep_boundary<fp_type>::initialise(boundary_pair<fp_type, fp_space_types...> const &boundary, fp_type time,
                                                fp_space_types... space_args)
{
    const auto a = std::get<0>(low_quad_);
    const auto b = std::get<1>(low_quad_);
    const auto c = std::get<2>(low_quad_);
    const auto f = std::get<3>(low_quad_);
    const fp_type two = static_cast<fp_type>(2.0);
    const fp_type mone = static_cast<fp_type>(-1.0);
    auto const &first_bnd = boundary.first;
    if (auto ptr = std::dynamic_pointer_cast<dirichlet_boundary<fp_type, fp_space_types...>>(first_bnd))
    {
        const auto cst_val = ptr->value(time, space_args...);
        start_index_ = 1;
        k_ = cst_val;
        l_ = fp_type{};
    }
    else if (auto ptr = std::dynamic_pointer_cast<neumann_boundary<fp_type, fp_space_types...>>(first_bnd))
    {
        const auto cst_val = ptr->value(time, space_args...);
        start_index_ = 0;
        k_ = (f - a * space_step_ * two * cst_val) / b;
        l_ = mone * (a + c) / b;
    }
    else if (auto ptr = std::dynamic_pointer_cast<robin_boundary<fp_type, fp_space_types...>>(first_bnd))
    {
        const auto lin_val = ptr->linear_value(time, space_args...);
        const auto cst_val = ptr->value(time, space_args...);
        start_index_ = 0;
        const auto tmp = b + a * space_step_ * two * lin_val;
        k_ = (f - a * space_step_ * two * cst_val) / tmp;
        l_ = mone * (a + c) / tmp;
    }
    else
    {
        // throw here unrecognized lower boundary
    }
}

template <typename fp_type>
template <typename... fp_space_types>
void double_sweep_boundary<fp_type>::finalise(boundary_pair<fp_type, fp_space_types...> const &boundary,
                                              const fp_type &k_nm1, const fp_type &k_n, const fp_type &l_nm1,
                                              const fp_type &l_n, fp_type time, fp_space_types... space_args)
{
    const fp_type two = static_cast<fp_type>(2.0);
    const fp_type one = static_cast<fp_type>(1.0);
    auto const &second_bnd = boundary.second;
    if (auto ptr = std::dynamic_pointer_cast<dirichlet_boundary<fp_type, fp_space_types...>>(second_bnd))
    {
        upper_ = ptr->value(time, space_args...);
    }
    else if (auto ptr = std::dynamic_pointer_cast<neumann_boundary<fp_type, fp_space_types...>>(second_bnd))
    {
        const auto cst_val = two * space_step_ * ptr->value(time, space_args...);
        upper_ = (l_n * (k_nm1 - cst_val) + k_n) / (one - l_n * l_nm1);
    }
    else if (auto ptr = std::dynamic_pointer_cast<robin_boundary<fp_type, fp_space_types...>>(second_bnd))
    {
        const auto lin_val = two * space_step_ * ptr->linear_value(time, space_args...);
        const auto cst_val = two * space_step_ * ptr->value(time, space_args...);
        upper_ = (l_n * (k_nm1 - cst_val) + k_n) / (one - l_n * (l_nm1 - lin_val));
    }
    else
    {
        // throw here unrecognized upper boundary
    }
}

template <typename fp_type>
template <typename... fp_space_types>
const std::pair<fp_type, fp_type> double_sweep_boundary<fp_type>::coefficients(
    boundary_pair<fp_type, fp_space_types...> const &boundary, fp_type time, fp_space_types... space_args)
{
    initialise(boundary, time, space_args...);
    return std::make_pair(k_, l_);
}

template <typename fp_type>
template <typename... fp_space_types>
const fp_type double_sweep_boundary<fp_type>::upper_boundary(boundary_pair<fp_type, fp_space_types...> const &boundary,
                                                             const fp_type &k_nm1, const fp_type &k_n,
                                                             const fp_type &l_nm1, const fp_type &l_n, fp_type time,
                                                             fp_space_types... space_args)
{
    finalise(boundary, k_nm1, k_n, l_nm1, l_n, time, space_args...);
    return upper_;
}

template <typename fp_type>
template <typename... fp_space_types>
const fp_type double_sweep_boundary<fp_type>::lower_boundary(boundary_pair<fp_type, fp_space_types...> const &boundary,
                                                             fp_type time, fp_space_types... space_args)
{
    fp_type ret{};
    if (auto ptr = std::dynamic_pointer_cast<dirichlet_boundary<fp_type, fp_space_types...>>(boundary.first))
    {
        ret = ptr->value(time, space_args...);
    }
    return ret;
}

} // namespace lss_double_sweep_solver

#endif ///_LSS_DOUBLE_SWEEP_BOUNDARY_HPP_
