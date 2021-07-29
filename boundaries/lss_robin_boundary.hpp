#pragma once
#if !defined(_LSS_ROBIN_BOUNDARY_HPP_)
#define _LSS_ROBIN_BOUNDARY_HPP_

#include <algorithm>
#include <iterator>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

#include "common/lss_utility.hpp"
#include "lss_boundary.hpp"

namespace lss_boundary
{

using lss_utility::sptr_t;

template <typename fp_type, typename... fp_space_types>
class robin_boundary : public boundary<fp_type, fp_space_types...>
{
  protected:
    robin_boundary() = delete;

  public:
    robin_boundary(const std::function<fp_type(fp_type, fp_space_types...)> &linear_value,
                   const std::function<fp_type(fp_type, fp_space_types...)> &value)
        : boundary<fp_type, fp_space_types...>(linear_value, value)
    {
    }

    const fp_type linear_value(fp_type time, fp_space_types... space_args) const
    {
        return this->linear_(time, space_args...);
    }
    const fp_type value(fp_type time, fp_space_types... space_args) const override
    {
        return this->const_(time, space_args...);
    }
};

template <typename fp_type> using robin_boundary_1d = robin_boundary<fp_type>;
template <typename fp_type> using robin_boundary_2d = robin_boundary<fp_type, fp_type>;
template <typename fp_type> using robin_boundary_1d_ptr = sptr_t<robin_boundary_1d<fp_type>>;
template <typename fp_type> using robin_boundary_2d_ptr = sptr_t<robin_boundary_2d<fp_type>>;
} // namespace lss_boundary

#endif ///_LSS_ROBIN_BOUNDARY_1D_HPP_
