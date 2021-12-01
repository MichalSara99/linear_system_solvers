#if !defined(_LSS_WAVE_DATA_CONFIG_BUILDER_T_HPP_)
#define _LSS_WAVE_DATA_CONFIG_BUILDER_T_HPP_

#include "builders/lss_wave_data_config_builder.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"

using lss_pde_solvers::wave_coefficient_data_config_1d_builder;
using lss_pde_solvers::wave_data_config_1d_builder;
using lss_pde_solvers::wave_initial_data_config_1d_builder;
using lss_pde_solvers::wave_source_data_config_1d_builder;
using lss_utility::range;
using lss_utility::sptr_t;

template <typename T> void test_wave_data_config_builder_t()
{
    auto const &coeff_data_ptr = wave_coefficient_data_config_1d_builder<T>()
                                     .a_coefficient([](T t, T x) { return 1.0; })
                                     .b_coefficient([](T t, T x) { return 0.0; })
                                     .c_coefficient([](T t, T x) { return 1.0; })
                                     .d_coefficient([](T t, T x) { return 0.0; })
                                     .build();

    auto const &init_data_ptr = wave_initial_data_config_1d_builder<T>()
                                    .first_initial_condition([](T x) { return std::sin(x); })
                                    .second_initial_condition([](T x) { return 0.0; })
                                    .build();

    auto const &data_ptr = wave_data_config_1d_builder<T>()
                               .coefficient_data_config(coeff_data_ptr)
                               .initial_data_config(init_data_ptr)
                               .source_data_config(nullptr)
                               .build();
    LSS_ASSERT(data_ptr != nullptr, "Must not be null pointer");
}

void test_wave_data_config_builder()
{
    test_wave_data_config_builder_t<float>();
    test_wave_data_config_builder_t<double>();
}

#endif ///_LSS_WAVE_DATA_CONFIG_BUILDER_T_HPP_
