#pragma once
#if !defined(_LSS_NEUMANN_BOUNDARY_1D_HPP_)
#define _LSS_NEUMANN_BOUNDARY_1D_HPP_

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

template <typename fp_type> class neumann_boundary_1d : public boundary_1d<fp_type>
{
  protected:
    neumann_boundary_1d() = delete;

  public:
    neumann_boundary_1d(const std::function<fp_type(fp_type)> &value) : boundary_1d<fp_type>(nullptr, value)
    {
    }

    const fp_type value(fp_type time) const override
    {
        return this->const_(time);
    }
};

template <typename fp_type> using neumann_boundary_1d_ptr = sptr_t<neumann_boundary_1d<fp_type>>;

} // namespace lss_boundary_1d

#endif ///_LSS_NEUMANN_BOUNDARY_1D_HPP_
