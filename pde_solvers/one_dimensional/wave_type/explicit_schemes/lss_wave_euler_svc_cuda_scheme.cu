#include "lss_wave_euler_svc_cuda_scheme.hpp"

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
void wave_euler_svc_cuda_kernel<float>::launch(thrust::device_vector<float> const &input_0,
                                               thrust::device_vector<float> const &input_1,
                                               thrust::device_vector<float> &solution)
{
    const unsigned int threads_per_block = THREADS_PER_BLOCK;
    const unsigned int blocks_per_grid =
        static_cast<unsigned int>(solution.size() + threads_per_block - 1) / threads_per_block;
    float *raw_a = thrust::raw_pointer_cast(d_a_.data());
    float *raw_b = thrust::raw_pointer_cast(d_b_.data());
    float *raw_c = thrust::raw_pointer_cast(d_c_.data());
    float *raw_d = thrust::raw_pointer_cast(d_d_.data());
    const float *raw_input_0 = thrust::raw_pointer_cast(input_0.data());
    const float *raw_input_1 = thrust::raw_pointer_cast(input_1.data());
    float *raw_solution = thrust::raw_pointer_cast(solution.data());
    wave_core_kernel<float><<<threads_per_block, blocks_per_grid>>>(raw_a, raw_b, raw_c, raw_d, raw_input_0,
                                                                    raw_input_1, raw_solution, solution.size());
}

template <>
void wave_euler_svc_cuda_kernel<double>::launch(thrust::device_vector<double> const &input_0,
                                                thrust::device_vector<double> const &input_1,
                                                thrust::device_vector<double> &solution)
{
    const unsigned int threads_per_block = THREADS_PER_BLOCK;
    const unsigned int blocks_per_grid =
        static_cast<unsigned int>(solution.size() + threads_per_block - 1) / threads_per_block;
    double *raw_a = thrust::raw_pointer_cast(d_a_.data());
    double *raw_b = thrust::raw_pointer_cast(d_b_.data());
    double *raw_c = thrust::raw_pointer_cast(d_c_.data());
    double *raw_d = thrust::raw_pointer_cast(d_d_.data());
    const double *raw_input_0 = thrust::raw_pointer_cast(input_0.data());
    const double *raw_input_1 = thrust::raw_pointer_cast(input_1.data());
    double *raw_solution = thrust::raw_pointer_cast(solution.data());
    wave_core_kernel<double><<<threads_per_block, blocks_per_grid>>>(raw_a, raw_b, raw_c, raw_d, raw_input_0,
                                                                     raw_input_1, raw_solution, solution.size());
}

template <>
void wave_euler_svc_cuda_kernel<float>::launch(thrust::device_vector<float> const &input_0,
                                               thrust::device_vector<float> const &input_1,
                                               thrust::device_vector<float> const &source,
                                               thrust::device_vector<float> &solution)
{
    const unsigned int threads_per_block = THREADS_PER_BLOCK;
    const unsigned int blocks_per_grid =
        static_cast<unsigned int>(solution.size() + threads_per_block - 1) / threads_per_block;
    float *raw_a = thrust::raw_pointer_cast(d_a_.data());
    float *raw_b = thrust::raw_pointer_cast(d_b_.data());
    float *raw_c = thrust::raw_pointer_cast(d_c_.data());
    float *raw_d = thrust::raw_pointer_cast(d_d_.data());
    const float *raw_source = thrust::raw_pointer_cast(source.data());
    const float *raw_input_0 = thrust::raw_pointer_cast(input_0.data());
    const float *raw_input_1 = thrust::raw_pointer_cast(input_1.data());
    float *raw_solution = thrust::raw_pointer_cast(solution.data());
    wave_core_kernel<float><<<threads_per_block, blocks_per_grid>>>(
        raw_a, raw_b, raw_c, raw_d, raw_input_0, raw_input_1, raw_source, raw_solution, solution.size());
}

template <>
void wave_euler_svc_cuda_kernel<double>::launch(thrust::device_vector<double> const &input_0,
                                                thrust::device_vector<double> const &input_1,
                                                thrust::device_vector<double> const &source,
                                                thrust::device_vector<double> &solution)
{
    const unsigned int threads_per_block = THREADS_PER_BLOCK;
    const unsigned int blocks_per_grid =
        static_cast<unsigned int>(solution.size() + threads_per_block - 1) / threads_per_block;
    double *raw_a = thrust::raw_pointer_cast(d_a_.data());
    double *raw_b = thrust::raw_pointer_cast(d_b_.data());
    double *raw_c = thrust::raw_pointer_cast(d_c_.data());
    double *raw_d = thrust::raw_pointer_cast(d_d_.data());
    const double *raw_source = thrust::raw_pointer_cast(source.data());
    const double *raw_input_0 = thrust::raw_pointer_cast(input_0.data());
    const double *raw_input_1 = thrust::raw_pointer_cast(input_1.data());
    double *raw_solution = thrust::raw_pointer_cast(solution.data());
    wave_core_kernel<double><<<threads_per_block, blocks_per_grid>>>(
        raw_a, raw_b, raw_c, raw_d, raw_input_0, raw_input_1, raw_source, raw_solution, solution.size());
}

} // namespace one_dimensional

} // namespace lss_pde_solvers
