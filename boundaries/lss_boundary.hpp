#pragma once
#if !defined(_LSS_BOUNDARY_HPP_)
#define _LSS_BOUNDARY_HPP_

#include "common/lss_utility.hpp"
#include <functional>

namespace lss_boundary
{
using lss_utility::sptr_t;
template <typename fp_type, typename... fp_space_types> class boundary
{
  protected:
    std::function<fp_type(fp_type, fp_space_types...)> linear_;
    std::function<fp_type(fp_type, fp_space_types...)> const_;

  public:
    typedef fp_type value_type;
    explicit boundary() = delete;
    explicit boundary(const std::function<fp_type(fp_type, fp_space_types...)> &linear,
                      const std::function<fp_type(fp_type, fp_space_types...)> &constant)
        : linear_{linear}, const_{constant}
    {
    }

    virtual ~boundary()
    {
    }

    virtual const fp_type value(fp_type time, fp_space_types... space_args) const = 0;
};

template <typename fp_type, typename... fp_space_types>
using boundary_ptr = sptr_t<boundary<fp_type, fp_space_types...>>;
template <typename fp_type> using boundary_1d_ptr = sptr_t<boundary<fp_type>>;
template <typename fp_type> using boundary_2d_ptr = sptr_t<boundary<fp_type, fp_type>>;
template <typename fp_type, typename... fp_space_types>
using boundary_pair = std::pair<boundary_ptr<fp_type, fp_space_types...>, boundary_ptr<fp_type, fp_space_types...>>;
template <typename fp_type> using boundary_1d_pair = std::pair<boundary_1d_ptr<fp_type>, boundary_1d_ptr<fp_type>>;
template <typename fp_type> using boundary_2d_pair = std::pair<boundary_2d_ptr<fp_type>, boundary_2d_ptr<fp_type>>;

} // namespace lss_boundary

#endif ///_LSS_BOUNDARY_HPP_
