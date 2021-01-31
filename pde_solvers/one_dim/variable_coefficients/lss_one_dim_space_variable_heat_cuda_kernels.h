#pragma once
#if !defined(_LSS_ONE_DIM_SPACE_VARIABLE_HEAT_CUDA_KERNELS)
#define _LSS_ONE_DIM_SPACE_VARIABLE_HEAT_CUDA_KERNELS

#include <device_launch_parameters.h>

#define THREADS_PER_BLOCK 256

namespace lss_one_dim_space_variable_heat_cuda_kernels {

template <typename fp_type>
__global__ void fill_dirichlet_bc_1d(fp_type* solution, fp_type left,
                                     fp_type right, unsigned long long size) {
  unsigned long long const tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= size) return;
  if (tid == 0) solution[tid] = left;
  if (tid == (size - 1)) solution[tid] = right;
}

template <typename fp_type>
__global__ void fill_robin_bc_1d(fp_type* solution, fp_type* a, fp_type* b,
                                 fp_type* d, fp_type left_linear,
                                 fp_type left_const, fp_type right_linear,
                                 fp_type right_const, unsigned long long size) {
  unsigned long long const tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= size) return;
  if (tid == 0)
    solution[tid] = (left_linear * a[tid] + d[tid]) * solution[tid + 1] +
                    (1.0 - (2.0 * b[tid])) * solution[tid] +
                    a[tid] * left_const;
  if (tid == (size - 1))
    solution[tid] = (right_linear * d[tid] + a[tid]) * solution[tid - 1] +
                    (1.0 - (2.0 * b[tid])) * solution[tid] +
                    d[tid] * right_const;
}

template <typename fp_type>
__global__ void fill_robin_bc_1d(fp_type* solution, fp_type source_left,
                                 fp_type source_right, fp_type* a, fp_type* b,
                                 fp_type* d, fp_type time_step,
                                 fp_type left_linear, fp_type left_const,
                                 fp_type right_linear, fp_type right_const,
                                 unsigned long long size) {
  unsigned long long const tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= size) return;
  if (tid == 0)
    solution[tid] = (left_linear * a[tid] + d[tid]) * solution[tid + 1] +
                    (1.0 - (2.0 * b[tid])) * solution[tid] +
                    a[tid] * left_const + time_step * source_left;
  if (tid == (size - 1))
    solution[tid] = (right_linear * d[tid] + a[tid]) * solution[tid - 1] +
                    (1.0 - (2.0 * b[tid])) * solution[tid] +
                    d[tid] * right_const + time_step * source_right;
}

// Euler 1D kernel without source:
template <typename fp_type>
__global__ void explicit_euler_iterate_1d(fp_type* prev, fp_type* next,
                                          fp_type* a, fp_type* b, fp_type* d,
                                          unsigned long long size) {
  unsigned long long const tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= size) return;
  if (tid == 0) return;
  if (tid == (size - 1)) return;
  next[tid] = d[tid] * prev[tid + 1] + (1.0 - (2.0 * b[tid])) * prev[tid] +
              a[tid] * prev[tid - 1];
}

// Euler 1D kernel with source and timeStep:
template <typename fp_type>
__global__ void explicit_euler_iterate_1d(fp_type* prev, fp_type* next,
                                          fp_type* source, fp_type* a,
                                          fp_type* b, fp_type* d,
                                          fp_type time_step,
                                          unsigned long long size) {
  unsigned long long const tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= size) return;
  if (tid == 0) return;
  if (tid == (size - 1)) return;
  next[tid] = d[tid] * prev[tid + 1] + (1.0 - (2.0 * b[tid])) * prev[tid] +
              a[tid] * prev[tid - 1] + time_step * source[tid];
}

}  // namespace lss_one_dim_space_variable_heat_cuda_kernels

#endif  ///_LSS_ONE_DIM_SPACE_VARIABLE_HEAT_CUDA_KERNELS
