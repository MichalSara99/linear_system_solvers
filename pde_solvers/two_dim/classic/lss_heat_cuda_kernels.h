#pragma once
#if !defined(_LSS_2D_HEAT_CUDA_KERNELS)
#define _LSS_2D_HEAT_CUDA_KERNELS

#include <device_launch_parameters.h>

#include "lss_heat_explicit_schemes_cuda.h"

#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16

namespace lss_two_dim_heat_cuda_kernels {

using lss_two_dim_heat_explicit_schemes_cuda::euler_scheme_coeffs_device;

template <typename fp_type>
__global__ void fill_dirichlet_bc_1d(fp_type* solution,
                                     fp_type const* dirichlet_up,
                                     fp_type const* dirichlet_bottom,
                                     fp_type const* dirichlet_left,
                                     fp_type const* dirichlet_right,
                                     unsigned long long rows,
                                     unsigned long long columns) {
  unsigned long long const row_idx = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned long long const col_idx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned long long const idx = col_idx + row_idx * columns;
  // we are operating on the boundary:
  if (row_idx >= rows) return;
  if (col_idx >= columns) return;

  if (col_idx == 0) solution[idx] = dirichlet_left[row_idx];
  if (col_idx == (columns - 1)) solution[idx] = dirichlet_right[row_idx];
  if (row_idx == 0) solution[idx] = dirichlet_up[col_idx];
  if (row_idx == (rows - 1)) solution[idx] = dirichlet_bottom[col_idx];
}

// Euler 1D kernel without source:
template <typename fp_type>
__global__ void explicit_euler_iterate_1d(
    fp_type const* prev, fp_type* next,
    euler_scheme_coeffs_device<fp_type> const* coeffs, unsigned long long rows,
    unsigned long long columns) {
  unsigned long long const row_idx = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned long long const col_idx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned long long const idx = col_idx + row_idx * columns;
  // we are operationg inside the region, not on the boundary:
  if (col_idx == 0) return;
  if (row_idx == 0) return;
  if (row_idx >= (rows - 1)) return;
  if (col_idx >= (columns - 1)) return;
  // get the linearized indices in the neighbourhood:
  long long left = idx - 1;
  long long right = idx + 1;
  long long up = idx - columns;
  long long bottom = idx + columns;
  long long up_left = idx - columns - 1;
  long long bottom_right = idx + columns + 1;

  next[idx] = coeffs->A * prev[idx] + coeffs->B_1 * prev[bottom] +
              coeffs->B_2 * prev[up] + coeffs->C_1 * prev[right] +
              coeffs->C_2 * prev[left] +
              coeffs->gamma * (prev[up_left] + prev[bottom_right]);
}

// Euler 1D kernel with source and timeStep:
template <typename fp_type>
__global__ void explicit_euler_iterate_1d(
    fp_type const* prev, fp_type* next, fp_type* source,
    euler_scheme_coeffs_device<fp_type> const* coeffs, unsigned long long rows,
    unsigned long long columns) {
  unsigned long long const row_idx = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned long long const col_idx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned long long const idx = col_idx + row_idx * columns;
  // we are operationg inside the region, not on the boundary:
  if (col_idx == 0) return;
  if (row_idx == 0) return;
  if (row_idx >= (rows - 1)) return;
  if (col_idx >= (columns - 1)) return;
  // get the linearized indices in the neighbourhood:
  long long left = idx - 1;
  long long right = idx + 1;
  long long up = idx - columns;
  long long bottom = idx + columns;
  long long up_left = idx - columns - 1;
  long long bottom_right = idx + columns + 1;

  next[idx] = coeffs->A * prev[idx] + coeffs->B_1 * prev[bottom] +
              coeffs->B_2 * prev[up] + coeffs->C_1 * prev[right] +
              coeffs->C_2 * prev[left] +
              coeffs->gamma * (prev[up_left] + prev[bottom_right]) +
              coeffs->k * source[idx];
}

}  // namespace lss_two_dim_heat_cuda_kernels

#endif  ///_LSS_2D_HEAT_CUDA_KERNELS
