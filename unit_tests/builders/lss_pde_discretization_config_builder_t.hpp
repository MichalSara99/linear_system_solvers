#if !defined(_LSS_PDE_DISCRETIZATION_CONFIG_BUILDER_T_HPP_)
#define _LSS_PDE_DISCRETIZATION_CONFIG_BUILDER_T_HPP_

#include <map>

#include "builders/lss_pde_discretization_config_builder.hpp"
#include "common/lss_utility.hpp"

using lss_pde_solvers::pde_discretization_config_1d_builder;
using lss_utility::range;

template <typename T> void test_pde_discretization_config_builder_t()
{
    auto const &solver = pde_discretization_config_1d_builder<T>()
                             .space_range(range<T>(0.0, 1.0))
                             .number_of_space_points(100)
                             .time_range(range<T>(0.0, 1.0))
                             .number_of_time_points(100)
                             .build();

    LSS_ASSERT(solver != nullptr, "Must not be null pointer");
}

void test_pde_discretization_config_builder()
{
    test_pde_discretization_config_builder_t<float>();
    test_pde_discretization_config_builder_t<double>();
}

#endif ///_LSS_PDE_DISCRETIZATION_CONFIG_BUILDER_T_HPP_
