#if !defined(_LSS_ROBIN_BOUNDARY_BUILDER_T_HPP_)
#define _LSS_ROBIN_BOUNDARY_BUILDER_T_HPP_

#include "builders/lss_robin_boundary_builder.hpp"
#include "common/lss_macros.hpp"
#include <functional>

using lss_boundary::robin_boundary_1d_builder;

template <typename T> void test_robin_boundary_builder_t()
{
    auto const &boundary_ptr =
        robin_boundary_1d_builder<T>().linear_value([](T x) { return x; }).value([](T x) { return 0.0; }).build();

    LSS_ASSERT(boundary_ptr != nullptr, "Must not be null pointer");
}

void test_robin_boundary_builder()
{
    test_robin_boundary_builder_t<float>();
    test_robin_boundary_builder_t<double>();
}

#endif ///_LSS_ROBIN_BOUNDARY_BUILDER_T_HPP_
