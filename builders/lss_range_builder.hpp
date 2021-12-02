#if !defined(_LSS_RANGE_BUILDER_HPP_)
#define _LSS_RANGE_BUILDER_HPP_

#include "common/lss_utility.hpp"

namespace lss_utility
{

template <typename fp_type> struct range_builder
{
  private:
    fp_type l_, u_;

  public:
    range_builder &lower(fp_type value)
    {
        l_ = value;
        return *this;
    }

    range_builder &upper(fp_type value)
    {
        u_ = value;
        return *this;
    }

    range_ptr<fp_type> build()
    {
        return std::make_shared<range<fp_type>>(l_, u_);
    }
};

} // namespace lss_utility

#endif ///_LSS_RANGE_BUILDER_HPP_
