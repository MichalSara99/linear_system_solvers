#include "lss_euler_svc_cuda_scheme.hpp"

namespace lss_pde_solvers
{

namespace one_dimensional
{

using lss_boundary_1d::boundary_1d_pair;
using lss_boundary_1d::boundary_1d_ptr;
using lss_boundary_1d::dirichlet_boundary_1d;
using lss_boundary_1d::neumann_boundary_1d;
using lss_boundary_1d::robin_boundary_1d;
using lss_containers::container_2d;
using lss_enumerations::traverse_direction_enum;
using lss_utility::function_triplet_t;
using lss_utility::NaN;
using lss_utility::pair_t;
using lss_utility::range;

template <>
void euler_svc_cuda_kernel<float>::launch(thrust::host_vector<float> const &input, thrust::host_vector<float> &solution)
{
    const std::size_t N = solution.size() - 1;
    for (std::size_t t = 1; t < N; ++t)
    {
        solution[t] = (d_[t] * input[t + 1]) + (b_[t] * input[t]) + (a_[t] * input[t - 1]);
    }
}

template <>
void euler_svc_cuda_kernel<double>::launch(thrust::host_vector<double> const &input,
                                           thrust::host_vector<double> &solution)
{
    const std::size_t N = solution.size() - 1;
    for (std::size_t t = 1; t < N; ++t)
    {
        solution[t] = (d_[t] * input[t + 1]) + (b_[t] * input[t]) + (a_[t] * input[t - 1]);
    }
}

template <>
void euler_svc_cuda_kernel<float>::launch(thrust::host_vector<float> const &input,
                                          thrust::host_vector<float> const &source,
                                          thrust::host_vector<float> &solution)
{
    const std::size_t N = solution.size() - 1;
    const auto &k = steps_.first;
    for (std::size_t t = 1; t < N; ++t)
    {
        solution[t] = (d_[t] * input[t + 1]) + (b_[t] * input[t]) + (a_[t] * input[t - 1]) + (k * source[t]);
    }
}

template <>
void euler_svc_cuda_kernel<double>::launch(thrust::host_vector<double> const &input,
                                           thrust::host_vector<double> const &source,
                                           thrust::host_vector<double> &solution)
{
    const std::size_t N = solution.size() - 1;
    const auto &k = steps_.first;
    for (std::size_t t = 1; t < N; ++t)
    {
        solution[t] = (d_[t] * input[t + 1]) + (b_[t] * input[t]) + (a_[t] * input[t - 1]) + (k * source[t]);
    }
}

} // namespace one_dimensional

} // namespace lss_pde_solvers
