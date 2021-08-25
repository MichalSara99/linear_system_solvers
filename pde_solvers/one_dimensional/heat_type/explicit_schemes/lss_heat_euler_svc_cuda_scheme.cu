#include "lss_heat_euler_svc_cuda_scheme.hpp"

#define THREADS_PER_BLOCK 256

namespace lss_pde_solvers
{

namespace one_dimensional
{

using lss_boundary::boundary_1d_pair;
using lss_boundary::boundary_1d_ptr;
using lss_boundary::dirichlet_boundary_1d;
using lss_boundary::neumann_boundary_1d;
using lss_boundary::robin_boundary_1d;
using lss_containers::container_2d;
using lss_enumerations::traverse_direction_enum;
using lss_utility::function_triplet_t;
using lss_utility::NaN;
using lss_utility::pair_t;
using lss_utility::range;

template <>
void heat_euler_svc_cuda_kernel<float>::launch(thrust::device_vector<float> const &input,
                                               thrust::device_vector<float> &solution)
{
    const unsigned int threads_per_block = THREADS_PER_BLOCK;
    const unsigned int blocks_per_grid =
        static_cast<unsigned int>(solution.size() + threads_per_block - 1) / threads_per_block;
    float *raw_a = thrust::raw_pointer_cast(d_a_.data());
    float *raw_b = thrust::raw_pointer_cast(d_b_.data());
    float *raw_d = thrust::raw_pointer_cast(d_d_.data());
    const float *raw_input = thrust::raw_pointer_cast(input.data());
    float *raw_solution = thrust::raw_pointer_cast(solution.data());
    heat_core_kernel<float>
        <<<threads_per_block, blocks_per_grid>>>(raw_a, raw_b, raw_d, raw_input, raw_solution, solution.size());
}

template <>
void heat_euler_svc_cuda_kernel<double>::launch(thrust::device_vector<double> const &input,
                                                thrust::device_vector<double> &solution)
{
    const unsigned int threads_per_block = THREADS_PER_BLOCK;
    const unsigned int blocks_per_grid =
        static_cast<unsigned int>(solution.size() + threads_per_block - 1) / threads_per_block;
    double *raw_a = thrust::raw_pointer_cast(d_a_.data());
    double *raw_b = thrust::raw_pointer_cast(d_b_.data());
    double *raw_d = thrust::raw_pointer_cast(d_d_.data());
    const double *raw_input = thrust::raw_pointer_cast(input.data());
    double *raw_solution = thrust::raw_pointer_cast(solution.data());
    heat_core_kernel<double>
        <<<threads_per_block, blocks_per_grid>>>(raw_a, raw_b, raw_d, raw_input, raw_solution, solution.size());
}

template <>
void heat_euler_svc_cuda_kernel<float>::launch(thrust::device_vector<float> const &input,
                                               thrust::device_vector<float> const &source,
                                               thrust::device_vector<float> &solution)
{
    const float k = steps_.first;
    const unsigned int threads_per_block = THREADS_PER_BLOCK;
    const unsigned int blocks_per_grid =
        static_cast<unsigned int>(solution.size() + threads_per_block - 1) / threads_per_block;
    float *raw_a = thrust::raw_pointer_cast(d_a_.data());
    float *raw_b = thrust::raw_pointer_cast(d_b_.data());
    float *raw_d = thrust::raw_pointer_cast(d_d_.data());
    const float *raw_source = thrust::raw_pointer_cast(source.data());
    const float *raw_input = thrust::raw_pointer_cast(input.data());
    float *raw_solution = thrust::raw_pointer_cast(solution.data());
    heat_core_kernel<float><<<threads_per_block, blocks_per_grid>>>(raw_a, raw_b, raw_d, k, raw_input, raw_source,
                                                                    raw_solution, solution.size());
}

template <>
void heat_euler_svc_cuda_kernel<double>::launch(thrust::device_vector<double> const &input,
                                                thrust::device_vector<double> const &source,
                                                thrust::device_vector<double> &solution)
{
    const double k = steps_.first;
    const unsigned int threads_per_block = THREADS_PER_BLOCK;
    const unsigned int blocks_per_grid =
        static_cast<unsigned int>(solution.size() + threads_per_block - 1) / threads_per_block;
    double *raw_a = thrust::raw_pointer_cast(d_a_.data());
    double *raw_b = thrust::raw_pointer_cast(d_b_.data());
    double *raw_d = thrust::raw_pointer_cast(d_d_.data());
    const double *raw_source = thrust::raw_pointer_cast(source.data());
    const double *raw_input = thrust::raw_pointer_cast(input.data());
    double *raw_solution = thrust::raw_pointer_cast(solution.data());
    heat_core_kernel<double><<<threads_per_block, blocks_per_grid>>>(raw_a, raw_b, raw_d, k, raw_input, raw_source,
                                                                     raw_solution, solution.size());
}

} // namespace one_dimensional

} // namespace lss_pde_solvers
