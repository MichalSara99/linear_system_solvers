#pragma once
#if !defined(_LSS_NEUMANN_BOUNDARY_HPP_)
#define _LSS_NEUMANN_BOUNDARY_HPP_

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
class neumann_boundary : public boundary<fp_type, fp_space_types...>
{
  protected:
    neumann_boundary() = delete;

  public:
    neumann_boundary(const std::function<fp_type(fp_type, fp_space_types...)> &value)
        : boundary<fp_type, fp_space_types...>(nullptr, value)
    {
    }

    const fp_type value(fp_type time, fp_space_types... space_args) const override
    {
        return this->const_(time, space_args...);
    }
};

template <typename fp_type> using neumann_boundary_1d = neumann_boundary<fp_type>;
template <typename fp_type> using neumann_boundary_2d = neumann_boundary<fp_type, fp_type>;
template <typename fp_type> using neumann_boundary_1d_ptr = sptr_t<neumann_boundary_1d<fp_type>>;
template <typename fp_type> using neumann_boundary_2d_ptr = sptr_t<neumann_boundary_2d<fp_type>>;

} // namespace lss_boundary

#endif ///_LSS_NEUMANN_BOUNDARY_HPP_
