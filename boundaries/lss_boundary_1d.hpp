#pragma once
#if !defined(_LSS_BOUNDARY_1D_HPP_)
#define _LSS_BOUNDARY_1D_HPP_

#include "common/lss_utility.hpp"
#include <functional>

namespace lss_boundary_1d
{
using lss_utility::sptr_t;
template <typename fp_type> class boundary_1d
{
  protected:
    std::function<fp_type(fp_type)> linear_;
    std::function<fp_type(fp_type)> const_;

  public:
    typedef fp_type value_type;
    explicit boundary_1d() = delete;
    explicit boundary_1d(const std::function<fp_type(fp_type)> &linear, const std::function<fp_type(fp_type)> &constant)
        : linear_{linear}, const_{constant}
    {
    }

    virtual ~boundary_1d()
    {
    }

    virtual const fp_type value(fp_type time) const = 0;
};

template <typename fp_type> using boundary_1d_ptr = sptr_t<boundary_1d<fp_type>>;
template <typename fp_type> using boundary_1d_pair = std::pair<boundary_1d_ptr<fp_type>, boundary_1d_ptr<fp_type>>;

} // namespace lss_boundary_1d

#endif ///_LSS_BOUNDARY_1D_HPP_
