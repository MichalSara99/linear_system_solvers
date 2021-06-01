#include <device_launch_parameters.h>

#include "lss_heat_cuda_kernels.h"
#include "lss_heat_explicit_schemes_cuda.h"
#include "pde_solvers/two_dim/lss_pde_utility.h"

namespace lss_two_dim_heat_explicit_schemes_cuda {

using lss_containers::matrix_double;
using lss_containers::matrix_float;
using lss_enumerations::dirichlet_side_enum;
using lss_two_dim_heat_cuda_kernels::explicit_euler_iterate_1d;
using lss_two_dim_heat_cuda_kernels::fill_dirichlet_bc_1d;
using lss_two_dim_pde_utility::dirichlet_boundary_2d;
using lss_two_dim_pde_utility::robin_boundary_2d;
using lss_two_dim_pde_utility::v_discretization_2d;
using lss_utility::NaN;
using lss_utility::sptr_t;
using lss_utility::swap;

void euler_2d_loop_sp::operator()(
    float const *input,
    sptr_t<dirichlet_boundary_2d<float>> const &dirichlet_boundary,
    unsigned long long const rows, unsigned long long const columns,
    unsigned long long const size, float *solution) const {
  typedef euler_scheme_coeffs_device<float> euler_coeffs_t;
  // prepare vectors for dirichlet boundary on host:
  std::vector<float> x1_dirichlet(columns);  // up = function of y
  std::vector<float> x2_dirichlet(columns);  // bottom = function of y
  std::vector<float> y1_dirichlet(rows);     // left = function of x
  std::vector<float> y2_dirichlet(rows);     // right = function of x
  // prepare pointers on device:
  float *d_prev = NULL;
  float *d_next = NULL;
  // prepare dirichlet struct for device
  dirichlet_device<float> d_dirichlet;
  // allocate block of memory on device:
  cudaMalloc((void **)&d_prev, size * sizeof(float));
  cudaMalloc((void **)&d_next, size * sizeof(float));
  // allocate block memory on device for Dirichlet:
  cudaMalloc((void **)&(d_dirichlet.up_x1), columns * sizeof(float));
  cudaMalloc((void **)&(d_dirichlet.bottom_x2), columns * sizeof(float));
  cudaMalloc((void **)&(d_dirichlet.left_y1), rows * sizeof(float));
  cudaMalloc((void **)&(d_dirichlet.right_y2), rows * sizeof(float));

  // copy contents of input to d_prev:
  cudaMemcpy(d_prev, input, size * sizeof(float),
             cudaMemcpyKind::cudaMemcpyHostToDevice);

  unsigned long long const threads = THREADS_PER_BLOCK_Y;
  const dim3 threads_per_block(threads, threads);
  const dim3 blocks_per_grid((columns + threads - 1) / threads,
                             (rows + threads - 1) / threads);
  // unpack the deltas and PDE coefficients:
  float const alpha = std::get<0>(coeffs_);
  float const beta = std::get<1>(coeffs_);
  float const gamma = std::get<2>(coeffs_);
  float const delta = std::get<3>(coeffs_);
  float const ni = std::get<4>(coeffs_);
  float const rho = std::get<5>(coeffs_);
  // calculate scheme coefficients:
  float const k = time_delta_;
  float const h_1 = std::get<0>(spatial_deltas_);
  float const h_2 = std::get<1>(spatial_deltas_);
  float const x_init = std::get<0>(spatial_inits_);
  float const y_init = std::get<1>(spatial_inits_);
  euler_coeffs_t *coeffs = (euler_coeffs_t *)malloc(sizeof(euler_coeffs_t));
  coeffs->A = (1.0f - 2.0f * alpha - 2.0f * beta + 2.0f * gamma + rho);
  coeffs->B_1 = (alpha - gamma + delta);
  coeffs->B_2 = (alpha - gamma - delta);
  coeffs->C_1 = (beta - gamma + ni);
  coeffs->C_2 = (beta - gamma - ni);
  coeffs->gamma = gamma;
  coeffs->k = k;
  // allocate euler coeffs on device and copy them there
  euler_coeffs_t *d_coeffs = NULL;
  cudaMalloc((void **)&d_coeffs, sizeof(euler_coeffs_t));
  cudaMemcpy(d_coeffs, coeffs, sizeof(euler_coeffs_t),
             cudaMemcpyKind::cudaMemcpyHostToDevice);
  free(coeffs);

  float time = k;

  if (is_source_set_) {
    // prepare a pointer for source on device:
    float *d_source = NULL;
    // allocate block memory on device:
    cudaMalloc((void **)&d_source, size * sizeof(float));
    // create vector on host:
    matrix_float h_source(rows, columns, NaN<float>());
    // source is zero:
    while (time <= time_) {
      // discretize source function on host:
      v_discretization_2d<float>::discretize_in_space(
          spatial_inits_, spatial_deltas_, time, source_, h_source);
      // discretize Dirichlet boundaries:
      dirichlet_boundary->fill(y_init, h_2, time, x1_dirichlet,
                               dirichlet_side_enum::Up);
      dirichlet_boundary->fill(y_init, h_2, time, x2_dirichlet,
                               dirichlet_side_enum::Bottom);
      dirichlet_boundary->fill(x_init, h_1, time, y1_dirichlet,
                               dirichlet_side_enum::Left);
      dirichlet_boundary->fill(x_init, h_1, time, y2_dirichlet,
                               dirichlet_side_enum::Right);

      // copy h_source contents to d_source (host => device ):
      cudaMemcpy(d_source, h_source.data().data(), size * sizeof(float),
                 cudaMemcpyKind::cudaMemcpyHostToDevice);
      // copy dirichlet boundaries to device:
      cudaMemcpy(d_dirichlet.up_x1, x1_dirichlet.data(),
                 columns * sizeof(float),
                 cudaMemcpyKind::cudaMemcpyHostToDevice);
      cudaMemcpy(d_dirichlet.bottom_x2, x2_dirichlet.data(),
                 columns * sizeof(float),
                 cudaMemcpyKind::cudaMemcpyHostToDevice);
      cudaMemcpy(d_dirichlet.left_y1, y1_dirichlet.data(), rows * sizeof(float),
                 cudaMemcpyKind::cudaMemcpyHostToDevice);
      cudaMemcpy(d_dirichlet.right_y2, y2_dirichlet.data(),
                 rows * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
      // populate new solution in d_next:
      explicit_euler_iterate_1d<float><<<blocks_per_grid, threads_per_block>>>(
          d_prev, d_next, d_source, d_coeffs, rows, columns);
      // fill in the dirichlet boundaries in d_next:
      fill_dirichlet_bc_1d<float><<<blocks_per_grid, threads_per_block>>>(
          d_next, d_dirichlet.up_x1, d_dirichlet.bottom_x2, d_dirichlet.left_y1,
          d_dirichlet.right_y2, rows, columns);
      // swap the two pointers:
      swap(d_prev, d_next);
      time += k;
    }
    // free allocated memory blocks on device:
    cudaFree(d_source);
  } else {
    // source is zero:
    while (time <= time_) {
      // discretize Dirichlet boundaries:
      dirichlet_boundary->fill(y_init, h_2, time, x1_dirichlet,
                               dirichlet_side_enum::Up);
      dirichlet_boundary->fill(y_init, h_2, time, x2_dirichlet,
                               dirichlet_side_enum::Bottom);
      dirichlet_boundary->fill(x_init, h_1, time, y1_dirichlet,
                               dirichlet_side_enum::Left);
      dirichlet_boundary->fill(x_init, h_1, time, y2_dirichlet,
                               dirichlet_side_enum::Right);
      // copy dirichlet boundaries to device:
      cudaMemcpy(d_dirichlet.up_x1, x1_dirichlet.data(),
                 columns * sizeof(float),
                 cudaMemcpyKind::cudaMemcpyHostToDevice);
      cudaMemcpy(d_dirichlet.bottom_x2, x2_dirichlet.data(),
                 columns * sizeof(float),
                 cudaMemcpyKind::cudaMemcpyHostToDevice);
      cudaMemcpy(d_dirichlet.left_y1, y1_dirichlet.data(), rows * sizeof(float),
                 cudaMemcpyKind::cudaMemcpyHostToDevice);
      cudaMemcpy(d_dirichlet.right_y2, y2_dirichlet.data(),
                 rows * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
      // populate new solution in d_next:
      explicit_euler_iterate_1d<float><<<blocks_per_grid, threads_per_block>>>(
          d_prev, d_next, d_coeffs, rows, columns);
      // fill in the dirichlet boundaries in d_next:
      fill_dirichlet_bc_1d<float><<<blocks_per_grid, threads_per_block>>>(
          d_next, d_dirichlet.up_x1, d_dirichlet.bottom_x2, d_dirichlet.left_y1,
          d_dirichlet.right_y2, rows, columns);
      // swap the two pointers:
      swap(d_prev, d_next);
      time += k;
    }
  }
  // copy the contents of d_next to the solution pointer:
  cudaMemcpy(solution, d_prev, size * sizeof(float),
             cudaMemcpyKind::cudaMemcpyDeviceToHost);
  // free allocated memory blocks on device:
  cudaFree(d_coeffs);
  cudaFree(d_prev);
  cudaFree(d_next);
  cudaFree(d_dirichlet.up_x1);
  cudaFree(d_dirichlet.bottom_x2);
  cudaFree(d_dirichlet.left_y1);
  cudaFree(d_dirichlet.right_y2);
}

void euler_2d_loop_dp::operator()(
    double const *input,
    sptr_t<dirichlet_boundary_2d<double>> const &dirichlet_boundary,
    unsigned long long const rows, unsigned long long const columns,
    unsigned long long const size, double *solution) const {
  typedef euler_scheme_coeffs_device<double> euler_coeffs_t;
  // prepare vectors for dirichlet boundary on host:
  std::vector<double> x1_dirichlet(columns);  // up = function of y
  std::vector<double> x2_dirichlet(columns);  // bottom = function of y
  std::vector<double> y1_dirichlet(rows);     // left = function of x
  std::vector<double> y2_dirichlet(rows);     // right = function of x
                                              // prepare pointers on device:
  double *d_prev = NULL;
  double *d_next = NULL;
  // prepare dirichlet struct for device
  dirichlet_device<double> d_dirichlet;
  // allocate block of memory on device:
  cudaMalloc((void **)&d_prev, size * sizeof(double));
  cudaMalloc((void **)&d_next, size * sizeof(double));
  // allocate block memory on device for Dirichlet:
  cudaMalloc((void **)&(d_dirichlet.up_x1), columns * sizeof(double));
  cudaMalloc((void **)&(d_dirichlet.bottom_x2), columns * sizeof(double));
  cudaMalloc((void **)&(d_dirichlet.left_y1), rows * sizeof(double));
  cudaMalloc((void **)&(d_dirichlet.right_y2), rows * sizeof(double));

  // copy contents of input to d_prev:
  cudaMemcpy(d_prev, input, size * sizeof(double),
             cudaMemcpyKind::cudaMemcpyHostToDevice);

  unsigned long long const threads = THREADS_PER_BLOCK_Y;
  const dim3 threads_per_block(threads, threads);
  const dim3 blocks_per_grid((columns + threads - 1) / threads,
                             (rows + threads - 1) / threads);
  // unpack the deltas and PDE coefficients:
  double const alpha = std::get<0>(coeffs_);
  double const beta = std::get<1>(coeffs_);
  double const gamma = std::get<2>(coeffs_);
  double const delta = std::get<3>(coeffs_);
  double const ni = std::get<4>(coeffs_);
  double const rho = std::get<5>(coeffs_);
  // calculate scheme coefficients:
  double const k = time_delta_;
  double const h_1 = std::get<0>(spatial_deltas_);
  double const h_2 = std::get<1>(spatial_deltas_);
  double const x_init = std::get<0>(spatial_inits_);
  double const y_init = std::get<1>(spatial_inits_);
  euler_coeffs_t *coeffs = (euler_coeffs_t *)malloc(sizeof(euler_coeffs_t));
  coeffs->A = (1.0f - 2.0f * alpha - 2.0f * beta + 2.0f * gamma + rho);
  coeffs->B_1 = (alpha - gamma + delta);
  coeffs->B_2 = (alpha - gamma - delta);
  coeffs->C_1 = (beta - gamma + ni);
  coeffs->C_2 = (beta - gamma - ni);
  coeffs->gamma = gamma;
  coeffs->k = k;
  // allocate euler coeffs on device and copy them there
  euler_coeffs_t *d_coeffs = NULL;
  cudaMalloc((void **)&d_coeffs, sizeof(euler_coeffs_t));
  cudaMemcpy(d_coeffs, coeffs, sizeof(euler_coeffs_t),
             cudaMemcpyKind::cudaMemcpyHostToDevice);
  free(coeffs);

  double time = k;

  if (is_source_set_) {
    // prepare a pointer for source on device:
    double *d_source = NULL;
    // allocate block memory on device:
    cudaMalloc((void **)&d_source, size * sizeof(double));
    // create vector on host:
    matrix_double h_source(rows, columns, NaN<double>());
    // source is zero:
    while (time <= time_) {
      // discretize source function on host:
      v_discretization_2d<double>::discretize_in_space(
          spatial_inits_, spatial_deltas_, time, source_, h_source);
      // discretize Dirichlet boundaries:
      dirichlet_boundary->fill(y_init, h_2, time, x1_dirichlet,
                               dirichlet_side_enum::Up);
      dirichlet_boundary->fill(y_init, h_2, time, x2_dirichlet,
                               dirichlet_side_enum::Bottom);
      dirichlet_boundary->fill(x_init, h_1, time, y1_dirichlet,
                               dirichlet_side_enum::Left);
      dirichlet_boundary->fill(x_init, h_1, time, y2_dirichlet,
                               dirichlet_side_enum::Right);

      // copy h_source contents to d_source (host => device ):
      cudaMemcpy(d_source, h_source.data().data(), size * sizeof(double),
                 cudaMemcpyKind::cudaMemcpyHostToDevice);
      // copy dirichlet boundaries to device:
      cudaMemcpy(d_dirichlet.up_x1, x1_dirichlet.data(),
                 columns * sizeof(double),
                 cudaMemcpyKind::cudaMemcpyHostToDevice);
      cudaMemcpy(d_dirichlet.bottom_x2, x2_dirichlet.data(),
                 columns * sizeof(double),
                 cudaMemcpyKind::cudaMemcpyHostToDevice);
      cudaMemcpy(d_dirichlet.left_y1, y1_dirichlet.data(),
                 rows * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
      cudaMemcpy(d_dirichlet.right_y2, y2_dirichlet.data(),
                 rows * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
      // populate new solution in d_next:
      explicit_euler_iterate_1d<double><<<blocks_per_grid, threads_per_block>>>(
          d_prev, d_next, d_source, d_coeffs, rows, columns);
      // fill in the dirichlet boundaries in d_next:
      fill_dirichlet_bc_1d<double><<<blocks_per_grid, threads_per_block>>>(
          d_next, d_dirichlet.up_x1, d_dirichlet.bottom_x2, d_dirichlet.left_y1,
          d_dirichlet.right_y2, rows, columns);
      // swap the two pointers:
      swap(d_prev, d_next);
      time += k;
    }
    // free allocated memory blocks on device:
    cudaFree(d_source);
  } else {
    // source is zero:
    while (time <= time_) {
      // discretize Dirichlet boundaries:
      dirichlet_boundary->fill(y_init, h_2, time, x1_dirichlet,
                               dirichlet_side_enum::Up);
      dirichlet_boundary->fill(y_init, h_2, time, x2_dirichlet,
                               dirichlet_side_enum::Bottom);
      dirichlet_boundary->fill(x_init, h_1, time, y1_dirichlet,
                               dirichlet_side_enum::Left);
      dirichlet_boundary->fill(x_init, h_1, time, y2_dirichlet,
                               dirichlet_side_enum::Right);
      // copy dirichlet boundaries to device:
      cudaMemcpy(d_dirichlet.up_x1, x1_dirichlet.data(),
                 columns * sizeof(double),
                 cudaMemcpyKind::cudaMemcpyHostToDevice);
      cudaMemcpy(d_dirichlet.bottom_x2, x2_dirichlet.data(),
                 columns * sizeof(double),
                 cudaMemcpyKind::cudaMemcpyHostToDevice);
      cudaMemcpy(d_dirichlet.left_y1, y1_dirichlet.data(),
                 rows * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
      cudaMemcpy(d_dirichlet.right_y2, y2_dirichlet.data(),
                 rows * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
      // populate new solution in d_next:
      explicit_euler_iterate_1d<double><<<blocks_per_grid, threads_per_block>>>(
          d_prev, d_next, d_coeffs, rows, columns);
      // fill in the dirichlet boundaries in d_next:
      fill_dirichlet_bc_1d<double><<<blocks_per_grid, threads_per_block>>>(
          d_next, d_dirichlet.up_x1, d_dirichlet.bottom_x2, d_dirichlet.left_y1,
          d_dirichlet.right_y2, rows, columns);
      // swap the two pointers:
      swap(d_prev, d_next);
      time += k;
    }
  }
  // copy the contents of d_next to the solution pointer:
  cudaMemcpy(solution, d_prev, size * sizeof(double),
             cudaMemcpyKind::cudaMemcpyDeviceToHost);
  // free allocated memory blocks on device:
  cudaFree(d_coeffs);
  cudaFree(d_prev);
  cudaFree(d_next);
  cudaFree(d_dirichlet.up_x1);
  cudaFree(d_dirichlet.bottom_x2);
  cudaFree(d_dirichlet.left_y1);
  cudaFree(d_dirichlet.right_y2);
}

void euler_2d_loop_sp::operator()(
    float const *input, sptr_t<robin_boundary_2d<float>> const &robin_boundary,
    unsigned long long const rows, unsigned long long const columns,
    unsigned long long const size, float *solution) const {
  throw std::exception("Not yet implemented");
}

void euler_2d_loop_dp::operator()(
    double const *input,
    sptr_t<robin_boundary_2d<double>> const &robin_boundary,
    unsigned long long const rows, unsigned long long const columns,
    unsigned long long const size, double *solution) const {
  throw std::exception("Not yet implemented");
}

}  // namespace lss_two_dim_heat_explicit_schemes_cuda
