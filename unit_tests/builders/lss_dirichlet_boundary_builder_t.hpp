#if !defined(_LSS_DIRICHLET_BOUNDARY_BUILDER_T_HPP_)
#define _LSS_DIRICHLET_BOUNDARY_BUILDER_T_HPP_

#include <functional>

#include "builders/lss_dirichlet_boundary_builder.hpp"
#include "common/lss_macros.hpp"

using lss_boundary::dirichlet_boundary_1d_builder;

template <typename T> void test_dirichlet_boundary_builder_t()
{
    auto const &boundary_ptr = dirichlet_boundary_1d_builder<T>().value([](T x) { return 0.0; }).build();

    LSS_ASSERT(boundary_ptr != nullptr, "Must not be null pointer");
}

void test_dirichlet_boundary_builder()
{
    test_dirichlet_boundary_builder_t<float>();
    test_dirichlet_boundary_builder_t<double>();
}

#endif ///_DIRICHLET_BOUNDARY_BUILDER_T_HPP_
