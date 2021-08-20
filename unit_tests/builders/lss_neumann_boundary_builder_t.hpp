#if !defined(_LSS_NEUMANN_BOUNDARY_BUILDER_T_HPP_)
#define _LSS_NEUMANN_BOUNDARY_BUILDER_T_HPP_

#include "builders/lss_neumann_boundary_builder.hpp"
#include "common/lss_macros.hpp"
#include <functional>

using lss_boundary::neumann_boundary_1d_builder;

template <typename T> void test_neumann_boundary_builder_t()
{
    auto const &boundary_ptr = neumann_boundary_1d_builder<T>().value([](T x) { return 0.0; }).build();

    LSS_ASSERT(boundary_ptr != nullptr, "Must not be null pointer");
}

void test_neumann_boundary_builder()
{
    test_neumann_boundary_builder_t<float>();
    test_neumann_boundary_builder_t<double>();
}

#endif ///_LSS_NEUMANN_BOUNDARY_BUILDER_T_HPP_
