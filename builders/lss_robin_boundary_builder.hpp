#if !defined(_LSS_ROBIN_BOUNDARY_BUILDER_HPP_)
#define _LSS_ROBIN_BOUNDARY_BUILDER_HPP_

#include "boundaries/lss_boundary.hpp"
#include "boundaries/lss_robin_boundary.hpp"

namespace lss_boundary
{

/**
    robin_boundary_1d_builder object
 */
template <typename fp_type> struct robin_boundary_1d_builder
{
  private:
    std::function<fp_type(fp_type)> linear_value_;
    std::function<fp_type(fp_type)> value_;

  public:
    robin_boundary_1d_builder &linear_value(const std::function<fp_type(fp_type)> &linear_value)
    {
        linear_value_ = linear_value;
        return *this;
    }

    robin_boundary_1d_builder &value(const std::function<fp_type(fp_type)> &value)
    {
        value_ = value;
        return *this;
    }

    robin_boundary_1d_ptr<fp_type> build()
    {
        return std::make_shared<robin_boundary_1d<fp_type>>(linear_value_, value_);
    }
};

/**
    robin_boundary_2d_builder object
 */
template <typename fp_type> struct robin_boundary_2d_builder
{
  private:
    std::function<fp_type(fp_type, fp_type)> linear_value_;
    std::function<fp_type(fp_type, fp_type)> value_;

  public:
    robin_boundary_2d_builder &linear_value(const std::function<fp_type(fp_type, fp_type)> &linear_value)
    {
        linear_value_ = linear_value;
        return *this;
    }

    robin_boundary_2d_builder &value(const std::function<fp_type(fp_type, fp_type)> &value)
    {
        value_ = value;
        return *this;
    }

    robin_boundary_2d_ptr<fp_type> build()
    {
        return std::make_shared<robin_boundary_2d<fp_type>>(linear_value_, value_);
    }
};

} // namespace lss_boundary

#endif ///_LSS_ROBIN_BOUNDARY_BUILDER_HPP_
