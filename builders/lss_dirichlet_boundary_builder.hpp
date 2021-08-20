#if !defined(_LSS_DIRICHLET_BOUNDARY_BUILDER_HPP_)
#define _LSS_DIRICHLET_BOUNDARY_BUILDER_HPP_

#include "boundaries/lss_boundary.hpp"
#include "boundaries/lss_dirichlet_boundary.hpp"

namespace lss_boundary
{

template <typename fp_type> struct dirichlet_boundary_1d_builder
{
  private:
    std::function<fp_type(fp_type)> value_;

  public:
    dirichlet_boundary_1d_builder &value(const std::function<fp_type(fp_type)> &value)
    {
        value_ = value;
        return *this;
    }

    dirichlet_boundary_1d_ptr<fp_type> build()
    {
        return std::make_shared<dirichlet_boundary_1d<fp_type>>(value_);
    }
};

template <typename fp_type> struct dirichlet_boundary_2d_builder
{
  private:
    std::function<fp_type(fp_type, fp_type)> value_;

  public:
    dirichlet_boundary_2d_builder &value(const std::function<fp_type(fp_type, fp_type)> &value)
    {
        value_ = value;
        return *this;
    }

    dirichlet_boundary_2d_ptr<fp_type> build()
    {
        return std::make_shared<dirichlet_boundary_2d<fp_type>>(value_);
    }
};

} // namespace lss_boundary

#endif ///_LSS_DIRICHLET_BOUNDARY_BUILDER_HPP_
