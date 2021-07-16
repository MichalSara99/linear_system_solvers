#pragma once
#if !defined(_LSS_ROBIN_BOUNDARY_1D_HPP_)
#define _LSS_ROBIN_BOUNDARY_1D_HPP_

#include <algorithm>
#include <iterator>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

#include "common/lss_utility.hpp"
#include "lss_boundary_1d.hpp"

namespace lss_boundary_1d
{

using lss_utility::sptr_t;

template <typename fp_type> class robin_boundary_1d : public boundary_1d<fp_type>
{
  protected:
    robin_boundary_1d() = delete;

  public:
    robin_boundary_1d(const std::function<fp_type(fp_type)> &linear_value, const std::function<fp_type(fp_type)> &value)
        : boundary_1d<fp_type>(linear_value, value)
    {
    }

    const fp_type linear_value(fp_type time) const
    {
        return this->linear_(time);
    }
    const fp_type value(fp_type time) const override
    {
        return this->const_(time);
    }
};

template <typename fp_type> using robin_boundary_1d_ptr = sptr_t<robin_boundary_1d<fp_type>>;
} // namespace lss_boundary_1d

#endif ///_LSS_ROBIN_BOUNDARY_1D_HPP_
