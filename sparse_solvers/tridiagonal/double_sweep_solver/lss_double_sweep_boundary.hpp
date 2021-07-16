#pragma once
#if !defined(_LSS_DOUBLE_SWEEP_BOUNDARY_HPP_)
#define _LSS_DOUBLE_SWEEP_BOUNDARY_HPP_

#pragma warning(disable : 4244)

#include <type_traits>
#include <vector>

#include "boundaries/lss_boundary_1d.hpp"
#include "boundaries/lss_dirichlet_boundary_1d.hpp"
#include "boundaries/lss_neumann_boundary_1d.hpp"
#include "boundaries/lss_robin_boundary_1d.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "common/lss_utility.hpp"

namespace lss_double_sweep_solver
{
using lss_boundary_1d::boundary_1d_ptr;
using lss_boundary_1d::dirichlet_boundary_1d;
using lss_boundary_1d::dirichlet_boundary_1d_ptr;
using lss_boundary_1d::neumann_boundary_1d;
using lss_boundary_1d::neumann_boundary_1d_ptr;
using lss_boundary_1d::robin_boundary_1d;
using lss_boundary_1d::robin_boundary_1d_ptr;

template <typename fp_type> class double_sweep_boundary
{
  private:
    boundary_1d_ptr<fp_type> first_;
    boundary_1d_ptr<fp_type> second_;
    std::tuple<fp_type, fp_type, fp_type, fp_type> low_quad_;
    fp_type space_step_, l_, k_, upper_;
    std::size_t start_index_;
    std::size_t discretization_size_;

    explicit double_sweep_boundary() = delete;
    void initialise(fp_type time);
    void finalise(const fp_type &k_nm1, const fp_type &k_n, const fp_type &l_nm1, const fp_type &l_n, fp_type time);

  public:
    typedef fp_type value_type;
    explicit double_sweep_boundary(const boundary_1d_ptr<fp_type> &first, const boundary_1d_ptr<fp_type> &second,
                                   const std::tuple<fp_type, fp_type, fp_type, fp_type> &low_quad,
                                   const std::size_t &discretization_size, const fp_type &space_step)
        : first_{first}, second_{second}, low_quad_{low_quad}, discretization_size_{discretization_size},
          space_step_{space_step}
    {
    }
    ~double_sweep_boundary()
    {
    }

    inline std::size_t start_index() const
    {
        return start_index_;
    }

    inline std::size_t end_index() const
    {
        if (auto ptr = std::dynamic_pointer_cast<dirichlet_boundary_1d<fp_type>>(second_))
        {
            return (discretization_size_ - 2);
        }
        return (discretization_size_ - 1);
    }

    const std::pair<fp_type, fp_type> coefficients(fp_type time);
    const fp_type upper_boundary(const fp_type &k_nm1, const fp_type &k_n, const fp_type &l_nm1, const fp_type &l_n,
                                 fp_type time);

    const fp_type lower_boundary(fp_type time);
};

template <typename fp_type> void double_sweep_boundary<fp_type>::initialise(fp_type time)
{
    const auto a = std::get<0>(low_quad_);
    const auto b = std::get<1>(low_quad_);
    const auto c = std::get<2>(low_quad_);
    const auto f = std::get<3>(low_quad_);
    const fp_type two = static_cast<fp_type>(2.0);
    const fp_type mone = static_cast<fp_type>(-1.0);
    if (auto ptr = std::dynamic_pointer_cast<dirichlet_boundary_1d<fp_type>>(first_))
    {
        const auto cst_val = ptr->value(time);
        start_index_ = 1;
        k_ = cst_val;
        l_ = fp_type{};
    }
    else if (auto ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(first_))
    {
        const auto cst_val = ptr->value(time);
        start_index_ = 0;
        k_ = (f - a * space_step_ * two * cst_val) / b;
        l_ = mone * (a + c) / b;
    }
    else if (auto ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_))
    {
        const auto lin_val = ptr->linear_value(time);
        const auto cst_val = ptr->value(time);
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
void double_sweep_boundary<fp_type>::finalise(const fp_type &k_nm1, const fp_type &k_n, const fp_type &l_nm1,
                                              const fp_type &l_n, fp_type time)
{
    const fp_type two = static_cast<fp_type>(2.0);
    const fp_type one = static_cast<fp_type>(1.0);
    if (auto ptr = std::dynamic_pointer_cast<dirichlet_boundary_1d<fp_type>>(second_))
    {
        upper_ = ptr->value(time);
    }
    else if (auto ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(second_))
    {
        const auto cst_val = two * space_step_ * ptr->value(time);
        upper_ = (l_n * (k_nm1 - cst_val) + k_n) / (one - l_n * l_nm1);
    }
    else if (auto ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_))
    {
        const auto lin_val = two * space_step_ * ptr->linear_value(time);
        const auto cst_val = two * space_step_ * ptr->value(time);
        upper_ = (l_n * (k_nm1 - cst_val) + k_n) / (one - l_n * (l_nm1 - lin_val));
    }
    else
    {
        // throw here unrecognized upper boundary
    }
}

template <typename fp_type> const std::pair<fp_type, fp_type> double_sweep_boundary<fp_type>::coefficients(fp_type time)
{
    initialise(time);
    return std::make_pair(k_, l_);
}
template <typename fp_type>
const fp_type double_sweep_boundary<fp_type>::upper_boundary(const fp_type &k_nm1, const fp_type &k_n,
                                                             const fp_type &l_nm1, const fp_type &l_n, fp_type time)
{
    finalise(k_nm1, k_n, l_nm1, l_n, time);
    return upper_;
}

template <typename fp_type> const fp_type double_sweep_boundary<fp_type>::lower_boundary(fp_type time)
{
    fp_type ret{};
    if (auto ptr = std::dynamic_pointer_cast<dirichlet_boundary_1d<fp_type>>(first_))
    {
        ret = ptr->value(time);
    }
    return ret;
}

} // namespace lss_double_sweep_solver

#endif ///_LSS_DOUBLE_SWEEP_BOUNDARY_HPP_
