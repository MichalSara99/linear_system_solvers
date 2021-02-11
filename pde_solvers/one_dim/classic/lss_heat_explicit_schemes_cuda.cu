#include <device_launch_parameters.h>

#include "lss_heat_cuda_kernels.h"
#include "lss_heat_explicit_schemes_cuda.h"

namespace lss_one_dim_heat_explicit_schemes_cuda {

using lss_one_dim_heat_cuda_kernels::explicit_euler_iterate_1d;
using lss_one_dim_heat_cuda_kernels::fill_dirichlet_bc_1d;
using lss_one_dim_heat_cuda_kernels::fill_robin_bc_1d;
using lss_one_dim_pde_utility::dirichlet_boundary;
using lss_one_dim_pde_utility::robin_boundary;
using lss_one_dim_pde_utility::v_discretization;
using lss_utility::NaN;
using lss_utility::swap;

void euler_loop_sp::operator()(
    float const *input, dirichlet_boundary<float> const &dirichlet_boundary,
    unsigned long long const size, float *solution) const {
  // prepare pointers on device:
  float *d_prev = NULL;
  float *d_next = NULL;
  // allocate block of memory on device:
  cudaMalloc((void **)&d_prev, size * sizeof(float));
  cudaMalloc((void **)&d_next, size * sizeof(float));
  // copy contents of input to d_prev:
  cudaMemcpy(d_prev, input, size * sizeof(float),
             cudaMemcpyKind::cudaMemcpyHostToDevice);

  unsigned int const threads_per_block = THREADS_PER_BLOCK;
  unsigned int const blocks_per_grid =
      (size + threads_per_block - 1) / threads_per_block;
  // unpack the deltas and PDE coefficients:
  float const k = std::get<0>(deltas_);
  float const h = std::get<1>(deltas_);
  float const A = std::get<0>(coeffs_);
  float const B = std::get<1>(coeffs_);
  float const C = std::get<2>(coeffs_);
  // calculate scheme coefficients:
  float const lambda = (A * k) / (h * h);
  float const gamma = (B * k) / (2.0f * h);
  float const delta = (C * k);
  // store bc:
  auto const &left = dirichlet_boundary.first;
  auto const &right = dirichlet_boundary.second;

  float time = k;

  if (is_source_set_) {
    // prepare a pointer for source on device:
    float *d_source = NULL;
    // allocate block memory on device:
    cudaMalloc((void **)&d_source, size * sizeof(float));
    // create vector on host:
    std::vector<float> h_source(size, NaN<float>());
    // source is zero:
    while (time <= terminal_t_) {
      // discretize source function on host:
      v_discretization<float>::discretize_in_space(h, space_start_, time,
                                                   source_, h_source);
      // copy h_source contents to d_source (host => device ):
      cudaMemcpy(d_source, h_source.data(), size * sizeof(float),
                 cudaMemcpyKind::cudaMemcpyHostToDevice);
      // populate new solution in d_next:
      explicit_euler_iterate_1d<float><<<threads_per_block, blocks_per_grid>>>(
          d_prev, d_next, d_source, lambda, gamma, delta, k, size);
      // fill in the dirichlet boundaries in d_next:
      fill_dirichlet_bc_1d<float><<<threads_per_block, blocks_per_grid>>>(
          d_next, left(time), right(time), size);
      // swap the two pointers:
      swap(d_prev, d_next);
      time += k;
    }
    // free allocated memory blocks on device:
    cudaFree(d_source);
  } else {
    // source is zero:
    while (time <= terminal_t_) {
      // populate new solution in d_next:
      explicit_euler_iterate_1d<float><<<threads_per_block, blocks_per_grid>>>(
          d_prev, d_next, lambda, gamma, delta, size);
      // fill in the dirichlet boundaries in d_next:
      fill_dirichlet_bc_1d<float><<<threads_per_block, blocks_per_grid>>>(
          d_next, left(time), right(time), size);
      // swap the two pointers:
      swap(d_prev, d_next);
      time += k;
    }
  }
  // copy the contents of d_next to the solution pointer:
  cudaMemcpy(solution, d_prev, size * sizeof(float),
             cudaMemcpyKind::cudaMemcpyDeviceToHost);
  // free allocated memory blocks on device:
  cudaFree(d_prev);
  cudaFree(d_next);
}

void euler_loop_dp::operator()(
    double const *input, dirichlet_boundary<double> const &dirichlet_boundary,
    unsigned long long const size, double *solution) const {
  // prepare pointers on device:
  double *d_prev = NULL;
  double *d_next = NULL;
  // allocate block of memory on device:
  cudaMalloc((void **)&d_prev, size * sizeof(double));
  cudaMalloc((void **)&d_next, size * sizeof(double));
  // copy contents of input to d_prev:
  cudaMemcpy(d_prev, input, size * sizeof(double),
             cudaMemcpyKind::cudaMemcpyHostToDevice);

  unsigned int const threads_per_block = THREADS_PER_BLOCK;
  unsigned int const blocks_per_grid =
      (size + threads_per_block - 1) / threads_per_block;
  // unpack the deltas and PDE coefficients:
  double const k = std::get<0>(deltas_);
  double const h = std::get<1>(deltas_);
  double const A = std::get<0>(coeffs_);
  double const B = std::get<1>(coeffs_);
  double const C = std::get<2>(coeffs_);
  // calculate scheme coefficients:
  double const lambda = (A * k) / (h * h);
  double const gamma = (B * k) / (2.0f * h);
  double const delta = (C * k);
  // store bc:
  auto const &left = dirichlet_boundary.first;
  auto const &right = dirichlet_boundary.second;

  double time = k;

  if (is_source_set_) {
    // prepare a pointer for source on device:
    double *d_source = NULL;
    // allocate block memory on device:
    cudaMalloc((void **)&d_source, size * sizeof(double));
    // create vector on host:
    std::vector<double> h_source(size, NaN<double>());
    // source is zero:
    while (time <= terminal_t_) {
      // discretize source function on host:
      v_discretization<double>::discretize_in_space(h, space_start_, time,
                                                    source_, h_source);
      // copy h_source contents to d_source (host => device ):
      cudaMemcpy(d_source, h_source.data(), size * sizeof(double),
                 cudaMemcpyKind::cudaMemcpyHostToDevice);
      // populate new solution in d_next:
      explicit_euler_iterate_1d<double><<<threads_per_block, blocks_per_grid>>>(
          d_prev, d_next, d_source, lambda, gamma, delta, k, size);
      // fill in the dirichlet boundaries in d_next:
      fill_dirichlet_bc_1d<double><<<threads_per_block, blocks_per_grid>>>(
          d_next, left(time), right(time), size);
      // swap the two pointers:
      swap(d_prev, d_next);
      time += k;
    }
    // free allocated memory blocks on device:
    cudaFree(d_source);
  } else {
    while (time <= terminal_t_) {
      // populate new solution in d_next:
      explicit_euler_iterate_1d<double><<<threads_per_block, blocks_per_grid>>>(
          d_prev, d_next, lambda, gamma, delta, size);
      // fill in the dirichlet boundaries in d_next:
      fill_dirichlet_bc_1d<double><<<threads_per_block, blocks_per_grid>>>(
          d_next, left(time), right(time), size);
      // swap the two pointers:
      swap(d_prev, d_next);
      time += k;
    }
  }

  // copy the contents of d_next to the solution pointer:
  cudaMemcpy(solution, d_prev, size * sizeof(double),
             cudaMemcpyKind::cudaMemcpyDeviceToHost);
  cudaFree(d_prev);
  cudaFree(d_next);
}

void euler_loop_sp::operator()(float const *input,
                               robin_boundary<float> const &robin_boundary,
                               unsigned long long const size,
                               float *solution) const {
  // prepare pointers on device:
  float *d_prev = NULL;
  float *d_next = NULL;
  // allocate block of memory on device:
  cudaMalloc((void **)&d_prev, size * sizeof(float));
  cudaMalloc((void **)&d_next, size * sizeof(float));
  // copy contents of input to d_prev:
  cudaMemcpy(d_prev, input, size * sizeof(float),
             cudaMemcpyKind::cudaMemcpyHostToDevice);

  unsigned int const threads_per_block = THREADS_PER_BLOCK;
  unsigned int const blocks_per_grid =
      (size + threads_per_block - 1) / threads_per_block;
  // unpack the deltas and PDE coefficients:
  float const k = std::get<0>(deltas_);
  float const h = std::get<1>(deltas_);
  float const A = std::get<0>(coeffs_);
  float const B = std::get<1>(coeffs_);
  float const C = std::get<2>(coeffs_);
  // calculate scheme coefficients:
  float const lambda = (A * k) / (h * h);
  float const gamma = (B * k) / (2.0f * h);
  float const delta = (C * k);
  // store bc:
  float const left_linear = robin_boundary.left.first;
  float const left_const = robin_boundary.left.second;
  float const right_linear = robin_boundary.right.first;
  float const right_const = robin_boundary.right.second;

  float time = k;

  if (is_source_set_) {
    // prepare a pointer for source on device:
    float *d_source = NULL;
    // allocate block memory on device:
    cudaMalloc((void **)&d_source, size * sizeof(float));
    // create vector on host:
    std::vector<float> h_source(size, NaN<float>());
    // source is zero:
    while (time <= terminal_t_) {
      // discretize source function on host:
      v_discretization<float>::discretize_in_space(h, space_start_, time,
                                                   source_, h_source);
      // copy h_source contents to d_source (host => device ):
      cudaMemcpy(d_source, h_source.data(), size * sizeof(float),
                 cudaMemcpyKind::cudaMemcpyHostToDevice);
      // populate new solution in d_next:
      explicit_euler_iterate_1d<float><<<threads_per_block, blocks_per_grid>>>(
          d_prev, d_next, d_source, lambda, gamma, delta, k, size);
      // fill in the dirichlet boundaries in d_next:
      fill_robin_bc_1d<float><<<threads_per_block, blocks_per_grid>>>(
          d_next, h_source.front(), h_source.back(), lambda, gamma, delta, k,
          left_linear, left_const, right_linear, right_const, size);
      // swap the two pointers:
      swap(d_prev, d_next);
      time += k;
    }
    // free allocated memory blocks on device:
    cudaFree(d_source);
  } else {
    while (time <= terminal_t_) {
      // populate new solution in d_next:
      explicit_euler_iterate_1d<float><<<threads_per_block, blocks_per_grid>>>(
          d_prev, d_next, lambda, gamma, delta, size);
      // fill in the dirichlet boundaries in d_next:
      fill_robin_bc_1d<float><<<threads_per_block, blocks_per_grid>>>(
          d_next, lambda, gamma, delta, left_linear, left_const, right_linear,
          right_const, size);
      // swap the two pointers:
      swap(d_prev, d_next);
      time += k;
    }
  }

  // copy the contents of d_next to the solution pointer:
  cudaMemcpy(solution, d_prev, size * sizeof(float),
             cudaMemcpyKind::cudaMemcpyDeviceToHost);
  cudaFree(d_prev);
  cudaFree(d_next);
}

void euler_loop_dp::operator()(double const *input,
                               robin_boundary<double> const &robin_boundary,
                               unsigned long long const size,
                               double *solution) const {
  // prepare pointers on device:
  double *d_prev = NULL;
  double *d_next = NULL;
  // allocate block of memory on device:
  cudaMalloc((void **)&d_prev, size * sizeof(double));
  cudaMalloc((void **)&d_next, size * sizeof(double));
  // copy contents of input to d_prev:
  cudaMemcpy(d_prev, input, size * sizeof(double),
             cudaMemcpyKind::cudaMemcpyHostToDevice);

  unsigned int const threads_per_block = THREADS_PER_BLOCK;
  unsigned int const blocks_per_grid =
      (size + threads_per_block - 1) / threads_per_block;
  // unpack the deltas and PDE coefficients:
  double const k = std::get<0>(deltas_);
  double const h = std::get<1>(deltas_);
  double const A = std::get<0>(coeffs_);
  double const B = std::get<1>(coeffs_);
  double const C = std::get<2>(coeffs_);
  // calculate scheme coefficients:
  double const lambda = (A * k) / (h * h);
  double const gamma = (B * k) / (2.0f * h);
  double const delta = (C * k);
  // store bc:
  double const left_linear = robin_boundary.left.first;
  double const left_const = robin_boundary.left.second;
  double const right_linear = robin_boundary.right.first;
  double const right_const = robin_boundary.right.second;

  double time = k;

  if (is_source_set_) {
    // prepare a pointer for source on device:
    double *d_source = NULL;
    // allocate block memory on device:
    cudaMalloc((void **)&d_source, size * sizeof(double));
    // create vector on host:
    std::vector<double> h_source(size, NaN<double>());
    // source is zero:
    while (time <= terminal_t_) {
      // discretize source function on host:
      v_discretization<double>::discretize_in_space(h, space_start_, time,
                                                    source_, h_source);
      // copy h_source contents to d_source (host => device ):
      cudaMemcpy(d_source, h_source.data(), size * sizeof(double),
                 cudaMemcpyKind::cudaMemcpyHostToDevice);
      // populate new solution in d_next:
      explicit_euler_iterate_1d<double><<<threads_per_block, blocks_per_grid>>>(
          d_prev, d_next, d_source, lambda, gamma, delta, k, size);
      // fill in the dirichlet boundaries in d_next:
      fill_robin_bc_1d<double><<<threads_per_block, blocks_per_grid>>>(
          d_next, h_source.front(), h_source.back(), lambda, gamma, delta, k,
          left_linear, left_const, right_linear, right_const, size);
      // swap the two pointers:
      swap(d_prev, d_next);
      time += k;
    }
    // free allocated memory blocks on device:
    cudaFree(d_source);
  } else {
    while (time <= terminal_t_) {
      // populate new solution in d_next:
      explicit_euler_iterate_1d<double><<<threads_per_block, blocks_per_grid>>>(
          d_prev, d_next, lambda, gamma, delta, size);
      // fill in the dirichlet boundaries in d_next:
      fill_robin_bc_1d<double><<<threads_per_block, blocks_per_grid>>>(
          d_next, lambda, gamma, delta, left_linear, left_const, right_linear,
          right_const, size);
      // swap the two pointers:
      swap(d_prev, d_next);
      time += k;
    }
  }
  // copy the contents of d_next to the solution pointer:
  cudaMemcpy(solution, d_prev, size * sizeof(double),
             cudaMemcpyKind::cudaMemcpyDeviceToHost);
  cudaFree(d_prev);
  cudaFree(d_next);
}

}  // namespace lss_one_dim_heat_explicit_schemes_cuda
