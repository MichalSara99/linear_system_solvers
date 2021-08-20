#if !defined(_LSS_HEAT_DATA_CONFIG_BUILDER_T_HPP_)
#define _LSS_HEAT_DATA_CONFIG_BUILDER_T_HPP_

#include "builders/lss_heat_data_config_builder.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"

using lss_pde_solvers::heat_coefficient_data_config_1d_builder;
using lss_pde_solvers::heat_data_config_1d_builder;
using lss_pde_solvers::heat_initial_data_config_1d_builder;
using lss_pde_solvers::heat_source_data_config_1d_builder;
using lss_utility::range;
using lss_utility::sptr_t;

template <typename T> void test_heat_data_config_builder_t()
{

    auto const &coeff_data_ptr = heat_coefficient_data_config_1d_builder<T>()
                                     .a_coefficient([](T x) { return 1.0; })
                                     .b_coefficient([](T x) { return 0.0; })
                                     .c_coefficient([](T x) { return 1.0; })
                                     .build();

    auto const &init_data_ptr =
        heat_initial_data_config_1d_builder<T>().initial_condition([](T x) { return std::sin(x); }).build();

    auto const &data_ptr = heat_data_config_1d_builder<T>()
                               .coefficient_data_config(coeff_data_ptr)
                               .initial_data_config(init_data_ptr)
                               .source_data_config(nullptr)
                               .build();
    LSS_ASSERT(data_ptr != nullptr, "Must not be null pointer");
}

void test_heat_data_config_builder()
{
    test_heat_data_config_builder_t<float>();
    test_heat_data_config_builder_t<double>();
}

#endif ///_LSS_HEAT_DATA_CONFIG_BUILDER_T_HPP_
