
#include <device_launch_parameters.h>

#include "lss_space_variable_heat_cuda_kernels.h"
#include "lss_space_variable_heat_explicit_schemes_cuda_policy.h"

namespace lss_one_dim_space_variable_heat_explicit_schemes_cuda_policy {

using lss_one_dim_pde_utility::dirichlet_boundary;
using lss_one_dim_pde_utility::robin_boundary;
using lss_one_dim_pde_utility::v_discretization;
using lss_one_dim_space_variable_heat_cuda_kernels::explicit_euler_iterate_1d;
using lss_one_dim_space_variable_heat_cuda_kernels::fill_dirichlet_bc_1d;
using lss_one_dim_space_variable_heat_cuda_kernels::fill_robin_bc_1d;
using lss_utility::NaN;
using lss_utility::swap;

// ==========================================================================
// ==================== heat_euler_scheme_forward_policy ====================
// ==========================================================================

template <>
void heat_euler_scheme_forward_policy<float,
                                      pde_coefficient_holder_fun_1_arg<float>>::
    traverse(float *solution, float const *init_solution,
             unsigned long long size,
             sptr_t<dirichlet_boundary_1d<float>> const &dirichlet_boundary,
             std::pair<float, float> const &deltas,
             pde_coefficient_holder_fun_1_arg<float> const &holder,
             float const &terminal_time, float const &space_start) {
  // prepare pointers on device:
  float *d_prev = NULL;
  float *d_next = NULL;
  // allocate block of memory on device:
  cudaMalloc((void **)&d_prev, size * sizeof(float));
  cudaMalloc((void **)&d_next, size * sizeof(float));
  // copy contents of input to d_prev:
  cudaMemcpy(d_prev, init_solution, size * sizeof(float),
             cudaMemcpyKind::cudaMemcpyHostToDevice);

  unsigned int const threads_per_block = THREADS_PER_BLOCK;
  unsigned int const blocks_per_grid =
      (size + threads_per_block - 1) / threads_per_block;
  // unpack the deltas and PDE coefficients:
  float const k = std::get<0>(deltas);
  float const h = std::get<1>(deltas);
  // const coefficients:
  float const lambda = k / (h * h);
  float const gamma = k / (2.0 * h);
  float const delta = 0.5 * k;
  // unpack PDE coefficients:
  auto const &a = std::get<0>(holder);
  auto const &b = std::get<1>(holder);
  auto const &c = std::get<2>(holder);
  // create scheme coefficients:
  auto const &A = [&](float x, float t) {
    return (lambda * a(x) - gamma * b(x));
  };
  auto const &B = [&](float x, float t) {
    return (lambda * a(x) - delta * c(x));
  };
  auto const &D = [&](float x, float t) {
    return (lambda * a(x) + gamma * b(x));
  };
  // store bc:
  auto const &left = dirichlet_boundary->first;
  auto const &right = dirichlet_boundary->second;

  float time = k;

  // prepare pointers for PDE space variable coeffs on device:
  float *d_A = NULL;
  float *d_B = NULL;
  float *d_D = NULL;
  // allocate block memory on device for PDE coeffs:
  cudaMalloc((void **)&d_A, size * sizeof(float));
  cudaMalloc((void **)&d_B, size * sizeof(float));
  cudaMalloc((void **)&d_D, size * sizeof(float));
  // create vector for PDE coeffs on host:
  std::vector<float> h_A(size, NaN<float>());
  std::vector<float> h_B(size, NaN<float>());
  std::vector<float> h_D(size, NaN<float>());

  // source is zero:
  while (time <= terminal_time) {
    // discretize PDE space variable coeffs on host:
    v_discretization<float>::discretize_in_space(h, space_start, time, A, h_A);
    v_discretization<float>::discretize_in_space(h, space_start, time, B, h_B);
    v_discretization<float>::discretize_in_space(h, space_start, time, D, h_D);
    // copy h_A,h_B,h_D over to d_A,d_B,d_D (host => device ):
    cudaMemcpy(d_A, h_A.data(), size * sizeof(float),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size * sizeof(float),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, h_D.data(), size * sizeof(float),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    // populate new solution in d_next:
    explicit_euler_iterate_1d<float><<<threads_per_block, blocks_per_grid>>>(
        d_prev, d_next, d_A, d_B, d_D, size);
    // fill in the dirichlet boundaries in d_next:
    fill_dirichlet_bc_1d<float><<<threads_per_block, blocks_per_grid>>>(
        d_next, left(time), right(time), size);
    // swap the two pointers:
    swap(d_prev, d_next);
    time += k;
  }

  // copy the contents of d_next to the solution pointer:
  cudaMemcpy(solution, d_prev, size * sizeof(float),
             cudaMemcpyKind::cudaMemcpyDeviceToHost);
  // free allocated memory blocks on device:
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_D);
  cudaFree(d_prev);
  cudaFree(d_next);
}

template <>
void heat_euler_scheme_forward_policy<
    double, pde_coefficient_holder_fun_1_arg<double>>::
    traverse(double *solution, double const *init_solution,
             unsigned long long size,
             sptr_t<dirichlet_boundary_1d<double>> const &dirichlet_boundary,
             std::pair<double, double> const &deltas,
             pde_coefficient_holder_fun_1_arg<double> const &holder,
             double const &terminal_time, double const &space_start) {
  // prepare pointers on device:
  double *d_prev = NULL;
  double *d_next = NULL;
  // allocate block of memory on device:
  cudaMalloc((void **)&d_prev, size * sizeof(double));
  cudaMalloc((void **)&d_next, size * sizeof(double));
  // copy contents of input to d_prev:
  cudaMemcpy(d_prev, init_solution, size * sizeof(double),
             cudaMemcpyKind::cudaMemcpyHostToDevice);

  unsigned int const threads_per_block = THREADS_PER_BLOCK;
  unsigned int const blocks_per_grid =
      (size + threads_per_block - 1) / threads_per_block;
  // unpack the deltas and PDE coefficients:
  double const k = std::get<0>(deltas);
  double const h = std::get<1>(deltas);
  // const coefficients:
  double const lambda = k / (h * h);
  double const gamma = k / (2.0 * h);
  double const delta = 0.5 * k;
  // unpack PDE coefficients:
  auto const &a = std::get<0>(holder);
  auto const &b = std::get<1>(holder);
  auto const &c = std::get<2>(holder);
  // create scheme coefficients:
  auto const &A = [&](double x, double t) {
    return (lambda * a(x) - gamma * b(x));
  };
  auto const &B = [&](double x, double t) {
    return (lambda * a(x) - delta * c(x));
  };
  auto const &D = [&](double x, double t) {
    return (lambda * a(x) + gamma * b(x));
  };
  // store bc:
  auto const &left = dirichlet_boundary->first;
  auto const &right = dirichlet_boundary->second;

  float time = k;

  // prepare pointers for PDE space variable coeffs on device:
  double *d_A = NULL;
  double *d_B = NULL;
  double *d_D = NULL;
  // allocate block memory on device for PDE coeffs:
  cudaMalloc((void **)&d_A, size * sizeof(double));
  cudaMalloc((void **)&d_B, size * sizeof(double));
  cudaMalloc((void **)&d_D, size * sizeof(double));
  // create vector for PDE coeffs on host:
  std::vector<double> h_A(size, NaN<double>());
  std::vector<double> h_B(size, NaN<double>());
  std::vector<double> h_D(size, NaN<double>());

  // source is zero:
  while (time <= terminal_time) {
    // discretize PDE space variable coeffs on host:
    v_discretization<double>::discretize_in_space(h, space_start, time, A, h_A);
    v_discretization<double>::discretize_in_space(h, space_start, time, B, h_B);
    v_discretization<double>::discretize_in_space(h, space_start, time, D, h_D);
    // copy h_A,h_B,h_D over to d_A,d_B,d_D (host => device ):
    cudaMemcpy(d_A, h_A.data(), size * sizeof(double),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size * sizeof(double),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, h_D.data(), size * sizeof(double),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    // populate new solution in d_next:
    explicit_euler_iterate_1d<double><<<threads_per_block, blocks_per_grid>>>(
        d_prev, d_next, d_A, d_B, d_D, size);
    // fill in the dirichlet boundaries in d_next:
    fill_dirichlet_bc_1d<double><<<threads_per_block, blocks_per_grid>>>(
        d_next, left(time), right(time), size);
    // swap the two pointers:
    swap(d_prev, d_next);
    time += k;
  }

  // copy the contents of d_next to the solution pointer:
  cudaMemcpy(solution, d_prev, size * sizeof(double),
             cudaMemcpyKind::cudaMemcpyDeviceToHost);
  // free allocated memory blocks on device:
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_D);
  cudaFree(d_prev);
  cudaFree(d_next);
}

template <>
void heat_euler_scheme_forward_policy<float,
                                      pde_coefficient_holder_fun_1_arg<float>>::
    traverse(float *solution, float const *init_solution,
             unsigned long long size,
             sptr_t<dirichlet_boundary_1d<float>> const &dirichlet_boundary,
             std::pair<float, float> const &deltas,
             pde_coefficient_holder_fun_1_arg<float> const &holder,
             float const &terminal_time, float const &space_start,
             std::function<float(float, float)> const &source) {
  // prepare pointers on device:
  float *d_prev = NULL;
  float *d_next = NULL;
  // allocate block of memory on device:
  cudaMalloc((void **)&d_prev, size * sizeof(float));
  cudaMalloc((void **)&d_next, size * sizeof(float));
  // copy contents of input to d_prev:
  cudaMemcpy(d_prev, init_solution, size * sizeof(float),
             cudaMemcpyKind::cudaMemcpyHostToDevice);

  unsigned int const threads_per_block = THREADS_PER_BLOCK;
  unsigned int const blocks_per_grid =
      (size + threads_per_block - 1) / threads_per_block;
  // unpack the deltas and PDE coefficients:
  float const k = std::get<0>(deltas);
  float const h = std::get<1>(deltas);
  // const coefficients:
  float const lambda = k / (h * h);
  float const gamma = k / (2.0 * h);
  float const delta = 0.5 * k;
  // unpack PDE coefficients:
  auto const &a = std::get<0>(holder);
  auto const &b = std::get<1>(holder);
  auto const &c = std::get<2>(holder);
  // create scheme coefficients:
  auto const &A = [&](float x, float t) {
    return (lambda * a(x) - gamma * b(x));
  };
  auto const &B = [&](float x, float t) {
    return (lambda * a(x) - delta * c(x));
  };
  auto const &D = [&](float x, float t) {
    return (lambda * a(x) + gamma * b(x));
  };
  // store bc:
  auto const &left = dirichlet_boundary->first;
  auto const &right = dirichlet_boundary->second;

  float time = k;

  // prepare pointers for PDE space variable coeffs on device:
  float *d_A = NULL;
  float *d_B = NULL;
  float *d_D = NULL;
  // allocate block memory on device for PDE coeffs:
  cudaMalloc((void **)&d_A, size * sizeof(float));
  cudaMalloc((void **)&d_B, size * sizeof(float));
  cudaMalloc((void **)&d_D, size * sizeof(float));
  // create vector for PDE coeffs on host:
  std::vector<float> h_A(size, NaN<float>());
  std::vector<float> h_B(size, NaN<float>());
  std::vector<float> h_D(size, NaN<float>());

  // prepare a pointer for source on device:
  float *d_source = NULL;
  // allocate block memory on device:
  cudaMalloc((void **)&d_source, size * sizeof(float));
  // create vector for source on host:
  std::vector<float> h_source(size, NaN<float>());
  // source is zero:
  while (time <= terminal_time) {
    // discretize source function on host:
    v_discretization<float>::discretize_in_space(h, space_start, time, source,
                                                 h_source);
    // discretize PDE space variable coeffs on host:
    v_discretization<float>::discretize_in_space(h, space_start, time, A, h_A);
    v_discretization<float>::discretize_in_space(h, space_start, time, B, h_B);
    v_discretization<float>::discretize_in_space(h, space_start, time, D, h_D);
    // copy h_source contents to d_source (host => device ):
    cudaMemcpy(d_source, h_source.data(), size * sizeof(float),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    // copy h_A,h_B,h_D over to d_A,d_B,d_D (host => device ):
    cudaMemcpy(d_A, h_A.data(), size * sizeof(float),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size * sizeof(float),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, h_D.data(), size * sizeof(float),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    // populate new solution in d_next:
    explicit_euler_iterate_1d<float><<<threads_per_block, blocks_per_grid>>>(
        d_prev, d_next, d_source, d_A, d_B, d_D, k, size);
    // fill in the dirichlet boundaries in d_next:
    fill_dirichlet_bc_1d<float><<<threads_per_block, blocks_per_grid>>>(
        d_next, left(time), right(time), size);
    // swap the two pointers:
    swap(d_prev, d_next);
    time += k;
  }
  // free allocated memory blocks on device:
  cudaFree(d_source);
  // copy the contents of d_next to the solution pointer:
  cudaMemcpy(solution, d_prev, size * sizeof(float),
             cudaMemcpyKind::cudaMemcpyDeviceToHost);
  // free allocated memory blocks on device:
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_D);
  cudaFree(d_prev);
  cudaFree(d_next);
}

template <>
void heat_euler_scheme_forward_policy<
    double, pde_coefficient_holder_fun_1_arg<double>>::
    traverse(double *solution, double const *init_solution,
             unsigned long long size,
             sptr_t<dirichlet_boundary_1d<double>> const &dirichlet_boundary,
             std::pair<double, double> const &deltas,
             pde_coefficient_holder_fun_1_arg<double> const &holder,
             double const &terminal_time, double const &space_start,
             std::function<double(double, double)> const &source) {
  // prepare pointers on device:
  double *d_prev = NULL;
  double *d_next = NULL;
  // allocate block of memory on device:
  cudaMalloc((void **)&d_prev, size * sizeof(double));
  cudaMalloc((void **)&d_next, size * sizeof(double));
  // copy contents of input to d_prev:
  cudaMemcpy(d_prev, init_solution, size * sizeof(double),
             cudaMemcpyKind::cudaMemcpyHostToDevice);

  unsigned int const threads_per_block = THREADS_PER_BLOCK;
  unsigned int const blocks_per_grid =
      (size + threads_per_block - 1) / threads_per_block;
  // unpack the deltas and PDE coefficients:
  double const k = std::get<0>(deltas);
  double const h = std::get<1>(deltas);
  // const coefficients:
  double const lambda = k / (h * h);
  double const gamma = k / (2.0 * h);
  double const delta = 0.5 * k;
  // unpack PDE coefficients:
  auto const &a = std::get<0>(holder);
  auto const &b = std::get<1>(holder);
  auto const &c = std::get<2>(holder);
  // create scheme coefficients:
  auto const &A = [&](double x, double t) {
    return (lambda * a(x) - gamma * b(x));
  };
  auto const &B = [&](double x, double t) {
    return (lambda * a(x) - delta * c(x));
  };
  auto const &D = [&](double x, double t) {
    return (lambda * a(x) + gamma * b(x));
  };
  // store bc:
  auto const &left = dirichlet_boundary->first;
  auto const &right = dirichlet_boundary->second;

  double time = k;

  // prepare pointers for PDE space variable coeffs on device:
  double *d_A = NULL;
  double *d_B = NULL;
  double *d_D = NULL;
  // allocate block memory on device for PDE coeffs:
  cudaMalloc((void **)&d_A, size * sizeof(double));
  cudaMalloc((void **)&d_B, size * sizeof(double));
  cudaMalloc((void **)&d_D, size * sizeof(double));
  // create vector for PDE coeffs on host:
  std::vector<double> h_A(size, NaN<double>());
  std::vector<double> h_B(size, NaN<double>());
  std::vector<double> h_D(size, NaN<double>());

  // prepare a pointer for source on device:
  double *d_source = NULL;
  // allocate block memory on device:
  cudaMalloc((void **)&d_source, size * sizeof(double));
  // create vector for source on host:
  std::vector<double> h_source(size, NaN<double>());
  // source is zero:
  while (time <= terminal_time) {
    // discretize source function on host:
    v_discretization<double>::discretize_in_space(h, space_start, time, source,
                                                  h_source);
    // discretize PDE space variable coeffs on host:
    v_discretization<double>::discretize_in_space(h, space_start, time, A, h_A);
    v_discretization<double>::discretize_in_space(h, space_start, time, B, h_B);
    v_discretization<double>::discretize_in_space(h, space_start, time, D, h_D);
    // copy h_source contents to d_source (host => device ):
    cudaMemcpy(d_source, h_source.data(), size * sizeof(double),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    // copy h_A,h_B,h_D over to d_A,d_B,d_D (host => device ):
    cudaMemcpy(d_A, h_A.data(), size * sizeof(double),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size * sizeof(double),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, h_D.data(), size * sizeof(double),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    // populate new solution in d_next:
    explicit_euler_iterate_1d<double><<<threads_per_block, blocks_per_grid>>>(
        d_prev, d_next, d_source, d_A, d_B, d_D, k, size);
    // fill in the dirichlet boundaries in d_next:
    fill_dirichlet_bc_1d<double><<<threads_per_block, blocks_per_grid>>>(
        d_next, left(time), right(time), size);
    // swap the two pointers:
    swap(d_prev, d_next);
    time += k;
  }
  // free allocated memory blocks on device:
  cudaFree(d_source);
  // copy the contents of d_next to the solution pointer:
  cudaMemcpy(solution, d_prev, size * sizeof(double),
             cudaMemcpyKind::cudaMemcpyDeviceToHost);
  // free allocated memory blocks on device:
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_D);
  cudaFree(d_prev);
  cudaFree(d_next);
}

template <>
void heat_euler_scheme_forward_policy<float,
                                      pde_coefficient_holder_fun_1_arg<float>>::
    traverse(float *solution, float const *init_solution,
             unsigned long long size,
             robin_boundary<float> const &robin_boundary,
             std::pair<float, float> const &deltas,
             pde_coefficient_holder_fun_1_arg<float> const &holder,
             float const &terminal_time, float const &space_start) {
  // prepare pointers on device:
  float *d_prev = NULL;
  float *d_next = NULL;
  // allocate block of memory on device:
  cudaMalloc((void **)&d_prev, size * sizeof(float));
  cudaMalloc((void **)&d_next, size * sizeof(float));
  // copy contents of input to d_prev:
  cudaMemcpy(d_prev, init_solution, size * sizeof(float),
             cudaMemcpyKind::cudaMemcpyHostToDevice);

  unsigned int const threads_per_block = THREADS_PER_BLOCK;
  unsigned int const blocks_per_grid =
      (size + threads_per_block - 1) / threads_per_block;
  // unpack the deltas and PDE coefficients:
  float const k = std::get<0>(deltas);
  float const h = std::get<1>(deltas);
  // const coefficients:
  float const lambda = k / (h * h);
  float const gamma = k / (2.0 * h);
  float const delta = 0.5 * k;
  // unpack PDE coefficients:
  auto const &a = std::get<0>(holder);
  auto const &b = std::get<1>(holder);
  auto const &c = std::get<2>(holder);
  // create scheme coefficients:
  auto const &A = [&](float x, float t) {
    return (lambda * a(x) - gamma * b(x));
  };
  auto const &B = [&](float x, float t) {
    return (lambda * a(x) - delta * c(x));
  };
  auto const &D = [&](float x, float t) {
    return (lambda * a(x) + gamma * b(x));
  };
  // store bc:
  float const left_linear = robin_boundary.left.first;
  float const left_const = robin_boundary.left.second;
  float const right_linear = robin_boundary.right.first;
  float const right_const = robin_boundary.right.second;

  float time = k;

  // prepare pointers for PDE space variable coeffs on device:
  float *d_A = NULL;
  float *d_B = NULL;
  float *d_D = NULL;
  // allocate block memory on device for PDE coeffs:
  cudaMalloc((void **)&d_A, size * sizeof(float));
  cudaMalloc((void **)&d_B, size * sizeof(float));
  cudaMalloc((void **)&d_D, size * sizeof(float));
  // create vector for PDE coeffs on host:
  std::vector<float> h_A(size, NaN<float>());
  std::vector<float> h_B(size, NaN<float>());
  std::vector<float> h_D(size, NaN<float>());

  while (time <= terminal_time) {
    // discretize PDE space variable coeffs on host:
    v_discretization<float>::discretize_in_space(h, space_start, time, A, h_A);
    v_discretization<float>::discretize_in_space(h, space_start, time, B, h_B);
    v_discretization<float>::discretize_in_space(h, space_start, time, D, h_D);
    // copy h_A,h_B,h_D over to d_A,d_B,d_D (host => device ):
    cudaMemcpy(d_A, h_A.data(), size * sizeof(float),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size * sizeof(float),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, h_D.data(), size * sizeof(float),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    // populate new solution in d_next:
    explicit_euler_iterate_1d<float><<<threads_per_block, blocks_per_grid>>>(
        d_prev, d_next, d_A, d_B, d_D, size);
    // fill in the dirichlet boundaries in d_next:
    fill_robin_bc_1d<float><<<threads_per_block, blocks_per_grid>>>(
        d_next, d_A, d_B, d_D, left_linear, left_const, right_linear,
        right_const, size);
    // swap the two pointers:
    swap(d_prev, d_next);
    time += k;
  }

  // copy the contents of d_next to the solution pointer:
  cudaMemcpy(solution, d_prev, size * sizeof(float),
             cudaMemcpyKind::cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_D);
  cudaFree(d_prev);
  cudaFree(d_next);
}

template <>
void heat_euler_scheme_forward_policy<
    double, pde_coefficient_holder_fun_1_arg<double>>::
    traverse(double *solution, double const *init_solution,
             unsigned long long size,
             robin_boundary<double> const &robin_boundary,
             std::pair<double, double> const &deltas,
             pde_coefficient_holder_fun_1_arg<double> const &holder,
             double const &terminal_time, double const &space_start) {
  // prepare pointers on device:
  double *d_prev = NULL;
  double *d_next = NULL;
  // allocate block of memory on device:
  cudaMalloc((void **)&d_prev, size * sizeof(double));
  cudaMalloc((void **)&d_next, size * sizeof(double));
  // copy contents of input to d_prev:
  cudaMemcpy(d_prev, init_solution, size * sizeof(double),
             cudaMemcpyKind::cudaMemcpyHostToDevice);

  unsigned int const threads_per_block = THREADS_PER_BLOCK;
  unsigned int const blocks_per_grid =
      (size + threads_per_block - 1) / threads_per_block;
  // unpack the deltas and PDE coefficients:
  double const k = std::get<0>(deltas);
  double const h = std::get<1>(deltas);
  // const coefficients:
  double const lambda = k / (h * h);
  double const gamma = k / (2.0 * h);
  double const delta = 0.5 * k;
  // unpack PDE coefficients:
  auto const &a = std::get<0>(holder);
  auto const &b = std::get<1>(holder);
  auto const &c = std::get<2>(holder);
  // create scheme coefficients:
  auto const &A = [&](double x, double t) {
    return (lambda * a(x) - gamma * b(x));
  };
  auto const &B = [&](double x, double t) {
    return (lambda * a(x) - delta * c(x));
  };
  auto const &D = [&](double x, double t) {
    return (lambda * a(x) + gamma * b(x));
  };
  // store bc:
  double const left_linear = robin_boundary.left.first;
  double const left_const = robin_boundary.left.second;
  double const right_linear = robin_boundary.right.first;
  double const right_const = robin_boundary.right.second;

  double time = k;

  // prepare pointers for PDE space variable coeffs on device:
  double *d_A = NULL;
  double *d_B = NULL;
  double *d_D = NULL;
  // allocate block memory on device for PDE coeffs:
  cudaMalloc((void **)&d_A, size * sizeof(double));
  cudaMalloc((void **)&d_B, size * sizeof(double));
  cudaMalloc((void **)&d_D, size * sizeof(double));
  // create vector for PDE coeffs on host:
  std::vector<double> h_A(size, NaN<double>());
  std::vector<double> h_B(size, NaN<double>());
  std::vector<double> h_D(size, NaN<double>());

  while (time <= terminal_time) {
    // discretize PDE space variable coeffs on host:
    v_discretization<double>::discretize_in_space(h, space_start, time, A, h_A);
    v_discretization<double>::discretize_in_space(h, space_start, time, B, h_B);
    v_discretization<double>::discretize_in_space(h, space_start, time, D, h_D);
    // copy h_A,h_B,h_D over to d_A,d_B,d_D (host => device ):
    cudaMemcpy(d_A, h_A.data(), size * sizeof(double),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size * sizeof(double),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, h_D.data(), size * sizeof(double),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    // populate new solution in d_next:
    explicit_euler_iterate_1d<double><<<threads_per_block, blocks_per_grid>>>(
        d_prev, d_next, d_A, d_B, d_D, size);
    // fill in the dirichlet boundaries in d_next:
    fill_robin_bc_1d<double><<<threads_per_block, blocks_per_grid>>>(
        d_next, d_A, d_B, d_D, left_linear, left_const, right_linear,
        right_const, size);
    // swap the two pointers:
    swap(d_prev, d_next);
    time += k;
  }

  // copy the contents of d_next to the solution pointer:
  cudaMemcpy(solution, d_prev, size * sizeof(double),
             cudaMemcpyKind::cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_D);
  cudaFree(d_prev);
  cudaFree(d_next);
}

template <>
void heat_euler_scheme_forward_policy<float,
                                      pde_coefficient_holder_fun_1_arg<float>>::
    traverse(float *solution, float const *init_solution,
             unsigned long long size,
             robin_boundary<float> const &robin_boundary,
             std::pair<float, float> const &deltas,
             pde_coefficient_holder_fun_1_arg<float> const &holder,
             float const &terminal_time, float const &space_start,
             std::function<float(float, float)> const &source) {
  // prepare pointers on device:
  float *d_prev = NULL;
  float *d_next = NULL;
  // allocate block of memory on device:
  cudaMalloc((void **)&d_prev, size * sizeof(float));
  cudaMalloc((void **)&d_next, size * sizeof(float));
  // copy contents of input to d_prev:
  cudaMemcpy(d_prev, init_solution, size * sizeof(float),
             cudaMemcpyKind::cudaMemcpyHostToDevice);

  unsigned int const threads_per_block = THREADS_PER_BLOCK;
  unsigned int const blocks_per_grid =
      (size + threads_per_block - 1) / threads_per_block;
  // unpack the deltas and PDE coefficients:
  float const k = std::get<0>(deltas);
  float const h = std::get<1>(deltas);
  // const coefficients:
  float const lambda = k / (h * h);
  float const gamma = k / (2.0 * h);
  float const delta = 0.5 * k;
  // unpack PDE coefficients:
  auto const &a = std::get<0>(holder);
  auto const &b = std::get<1>(holder);
  auto const &c = std::get<2>(holder);
  // create scheme coefficients:
  auto const &A = [&](float x, float t) {
    return (lambda * a(x) - gamma * b(x));
  };
  auto const &B = [&](float x, float t) {
    return (lambda * a(x) - delta * c(x));
  };
  auto const &D = [&](float x, float t) {
    return (lambda * a(x) + gamma * b(x));
  };
  // store bc:
  float const left_linear = robin_boundary.left.first;
  float const left_const = robin_boundary.left.second;
  float const right_linear = robin_boundary.right.first;
  float const right_const = robin_boundary.right.second;

  float time = k;

  // prepare pointers for PDE space variable coeffs on device:
  float *d_A = NULL;
  float *d_B = NULL;
  float *d_D = NULL;
  // allocate block memory on device for PDE coeffs:
  cudaMalloc((void **)&d_A, size * sizeof(float));
  cudaMalloc((void **)&d_B, size * sizeof(float));
  cudaMalloc((void **)&d_D, size * sizeof(float));
  // create vector for PDE coeffs on host:
  std::vector<float> h_A(size, NaN<float>());
  std::vector<float> h_B(size, NaN<float>());
  std::vector<float> h_D(size, NaN<float>());

  // prepare a pointer for source on device:
  float *d_source = NULL;
  // allocate block memory on device:
  cudaMalloc((void **)&d_source, size * sizeof(float));
  // create vector on host:
  std::vector<float> h_source(size, NaN<float>());
  // source is zero:
  while (time <= terminal_time) {
    // discretize source function on host:
    v_discretization<float>::discretize_in_space(h, space_start, time, source,
                                                 h_source);
    // discretize PDE space variable coeffs on host:
    v_discretization<float>::discretize_in_space(h, space_start, time, A, h_A);
    v_discretization<float>::discretize_in_space(h, space_start, time, B, h_B);
    v_discretization<float>::discretize_in_space(h, space_start, time, D, h_D);
    // copy h_source contents to d_source (host => device ):
    cudaMemcpy(d_source, h_source.data(), size * sizeof(float),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    // copy h_A,h_B,h_D over to d_A,d_B,d_D (host => device ):
    cudaMemcpy(d_A, h_A.data(), size * sizeof(float),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size * sizeof(float),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, h_D.data(), size * sizeof(float),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    // populate new solution in d_next:
    explicit_euler_iterate_1d<float><<<threads_per_block, blocks_per_grid>>>(
        d_prev, d_next, d_source, d_A, d_B, d_D, k, size);
    // fill in the dirichlet boundaries in d_next:
    fill_robin_bc_1d<float><<<threads_per_block, blocks_per_grid>>>(
        d_next, h_source.front(), h_source.back(), d_A, d_B, d_D, k,
        left_linear, left_const, right_linear, right_const, size);
    // swap the two pointers:
    swap(d_prev, d_next);
    time += k;
  }
  // free allocated memory blocks on device:
  cudaFree(d_source);
  // copy the contents of d_next to the solution pointer:
  cudaMemcpy(solution, d_prev, size * sizeof(float),
             cudaMemcpyKind::cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_D);
  cudaFree(d_prev);
  cudaFree(d_next);
}

template <>
void heat_euler_scheme_forward_policy<
    double, pde_coefficient_holder_fun_1_arg<double>>::
    traverse(double *solution, double const *init_solution,
             unsigned long long size,
             robin_boundary<double> const &robin_boundary,
             std::pair<double, double> const &deltas,
             pde_coefficient_holder_fun_1_arg<double> const &holder,
             double const &terminal_time, double const &space_start,
             std::function<double(double, double)> const &source) {
  // prepare pointers on device:
  double *d_prev = NULL;
  double *d_next = NULL;
  // allocate block of memory on device:
  cudaMalloc((void **)&d_prev, size * sizeof(double));
  cudaMalloc((void **)&d_next, size * sizeof(double));
  // copy contents of input to d_prev:
  cudaMemcpy(d_prev, init_solution, size * sizeof(double),
             cudaMemcpyKind::cudaMemcpyHostToDevice);

  unsigned int const threads_per_block = THREADS_PER_BLOCK;
  unsigned int const blocks_per_grid =
      (size + threads_per_block - 1) / threads_per_block;
  // unpack the deltas and PDE coefficients:
  double const k = std::get<0>(deltas);
  double const h = std::get<1>(deltas);
  // const coefficients:
  double const lambda = k / (h * h);
  double const gamma = k / (2.0 * h);
  double const delta = 0.5 * k;
  // unpack PDE coefficients:
  auto const &a = std::get<0>(holder);
  auto const &b = std::get<1>(holder);
  auto const &c = std::get<2>(holder);
  // create scheme coefficients:
  auto const &A = [&](double x, double t) {
    return (lambda * a(x) - gamma * b(x));
  };
  auto const &B = [&](double x, double t) {
    return (lambda * a(x) - delta * c(x));
  };
  auto const &D = [&](double x, double t) {
    return (lambda * a(x) + gamma * b(x));
  };
  // store bc:
  double const left_linear = robin_boundary.left.first;
  double const left_const = robin_boundary.left.second;
  double const right_linear = robin_boundary.right.first;
  double const right_const = robin_boundary.right.second;

  double time = k;

  // prepare pointers for PDE space variable coeffs on device:
  double *d_A = NULL;
  double *d_B = NULL;
  double *d_D = NULL;
  // allocate block memory on device for PDE coeffs:
  cudaMalloc((void **)&d_A, size * sizeof(double));
  cudaMalloc((void **)&d_B, size * sizeof(double));
  cudaMalloc((void **)&d_D, size * sizeof(double));
  // create vector for PDE coeffs on host:
  std::vector<double> h_A(size, NaN<double>());
  std::vector<double> h_B(size, NaN<double>());
  std::vector<double> h_D(size, NaN<double>());

  // prepare a pointer for source on device:
  double *d_source = NULL;
  // allocate block memory on device:
  cudaMalloc((void **)&d_source, size * sizeof(double));
  // create vector on host:
  std::vector<double> h_source(size, NaN<double>());
  // source is zero:
  while (time <= terminal_time) {
    // discretize source function on host:
    v_discretization<double>::discretize_in_space(h, space_start, time, source,
                                                  h_source);
    // discretize PDE space variable coeffs on host:
    v_discretization<double>::discretize_in_space(h, space_start, time, A, h_A);
    v_discretization<double>::discretize_in_space(h, space_start, time, B, h_B);
    v_discretization<double>::discretize_in_space(h, space_start, time, D, h_D);
    // copy h_source contents to d_source (host => device ):
    cudaMemcpy(d_source, h_source.data(), size * sizeof(double),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    // copy h_A,h_B,h_D over to d_A,d_B,d_D (host => device ):
    cudaMemcpy(d_A, h_A.data(), size * sizeof(double),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size * sizeof(double),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, h_D.data(), size * sizeof(double),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    // populate new solution in d_next:
    explicit_euler_iterate_1d<double><<<threads_per_block, blocks_per_grid>>>(
        d_prev, d_next, d_source, d_A, d_B, d_D, k, size);
    // fill in the dirichlet boundaries in d_next:
    fill_robin_bc_1d<double><<<threads_per_block, blocks_per_grid>>>(
        d_next, h_source.front(), h_source.back(), d_A, d_B, d_D, k,
        left_linear, left_const, right_linear, right_const, size);
    // swap the two pointers:
    swap(d_prev, d_next);
    time += k;
  }
  // free allocated memory blocks on device:
  cudaFree(d_source);
  // copy the contents of d_next to the solution pointer:
  cudaMemcpy(solution, d_prev, size * sizeof(double),
             cudaMemcpyKind::cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_D);
  cudaFree(d_prev);
  cudaFree(d_next);
}

// ==========================================================================
// =================== heat_euler_scheme_backward_policy ====================
// ==========================================================================

template <>
void heat_euler_scheme_backward_policy<
    float, pde_coefficient_holder_fun_1_arg<float>>::
    traverse(float *solution, float const *init_solution,
             unsigned long long size,
             sptr_t<dirichlet_boundary_1d<float>> const &dirichlet_boundary,
             std::pair<float, float> const &deltas,
             pde_coefficient_holder_fun_1_arg<float> const &holder,
             float const &terminal_time, float const &space_start) {
  // prepare pointers on device:
  float *d_prev = NULL;
  float *d_next = NULL;
  // allocate block of memory on device:
  cudaMalloc((void **)&d_prev, size * sizeof(float));
  cudaMalloc((void **)&d_next, size * sizeof(float));
  // copy contents of input to d_prev:
  cudaMemcpy(d_prev, init_solution, size * sizeof(float),
             cudaMemcpyKind::cudaMemcpyHostToDevice);

  unsigned int const threads_per_block = THREADS_PER_BLOCK;
  unsigned int const blocks_per_grid =
      (size + threads_per_block - 1) / threads_per_block;
  // unpack the deltas and PDE coefficients:
  float const k = std::get<0>(deltas);
  float const h = std::get<1>(deltas);
  // const coefficients:
  float const lambda = k / (h * h);
  float const gamma = k / (2.0 * h);
  float const delta = 0.5 * k;
  // unpack PDE coefficients:
  auto const &a = std::get<0>(holder);
  auto const &b = std::get<1>(holder);
  auto const &c = std::get<2>(holder);
  // create scheme coefficients:
  auto const &A = [&](float x, float t) {
    return (lambda * a(x) - gamma * b(x));
  };
  auto const &B = [&](float x, float t) {
    return (lambda * a(x) - delta * c(x));
  };
  auto const &D = [&](float x, float t) {
    return (lambda * a(x) + gamma * b(x));
  };
  // store bc:
  auto const &left = dirichlet_boundary->first;
  auto const &right = dirichlet_boundary->second;

  float time = terminal_time - k;

  // prepare pointers for PDE space variable coeffs on device:
  float *d_A = NULL;
  float *d_B = NULL;
  float *d_D = NULL;
  // allocate block memory on device for PDE coeffs:
  cudaMalloc((void **)&d_A, size * sizeof(float));
  cudaMalloc((void **)&d_B, size * sizeof(float));
  cudaMalloc((void **)&d_D, size * sizeof(float));
  // create vector for PDE coeffs on host:
  std::vector<float> h_A(size, NaN<float>());
  std::vector<float> h_B(size, NaN<float>());
  std::vector<float> h_D(size, NaN<float>());

  // source is zero:
  while (time >= 0.0) {
    // discretize PDE space variable coeffs on host:
    v_discretization<float>::discretize_in_space(h, space_start, time, A, h_A);
    v_discretization<float>::discretize_in_space(h, space_start, time, B, h_B);
    v_discretization<float>::discretize_in_space(h, space_start, time, D, h_D);
    // copy h_A,h_B,h_D over to d_A,d_B,d_D (host => device ):
    cudaMemcpy(d_A, h_A.data(), size * sizeof(float),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size * sizeof(float),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, h_D.data(), size * sizeof(float),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    // populate new solution in d_next:
    explicit_euler_iterate_1d<float><<<threads_per_block, blocks_per_grid>>>(
        d_prev, d_next, d_A, d_B, d_D, size);
    // fill in the dirichlet boundaries in d_next:
    fill_dirichlet_bc_1d<float><<<threads_per_block, blocks_per_grid>>>(
        d_next, left(time), right(time), size);
    // swap the two pointers:
    swap(d_prev, d_next);
    time -= k;
  }

  // copy the contents of d_next to the solution pointer:
  cudaMemcpy(solution, d_prev, size * sizeof(float),
             cudaMemcpyKind::cudaMemcpyDeviceToHost);
  // free allocated memory blocks on device:
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_D);
  cudaFree(d_prev);
  cudaFree(d_next);
}

template <>
void heat_euler_scheme_backward_policy<
    double, pde_coefficient_holder_fun_1_arg<double>>::
    traverse(double *solution, double const *init_solution,
             unsigned long long size,
             sptr_t<dirichlet_boundary_1d<double>> const &dirichlet_boundary,
             std::pair<double, double> const &deltas,
             pde_coefficient_holder_fun_1_arg<double> const &holder,
             double const &terminal_time, double const &space_start) {
  // prepare pointers on device:
  double *d_prev = NULL;
  double *d_next = NULL;
  // allocate block of memory on device:
  cudaMalloc((void **)&d_prev, size * sizeof(double));
  cudaMalloc((void **)&d_next, size * sizeof(double));
  // copy contents of input to d_prev:
  cudaMemcpy(d_prev, init_solution, size * sizeof(double),
             cudaMemcpyKind::cudaMemcpyHostToDevice);

  unsigned int const threads_per_block = THREADS_PER_BLOCK;
  unsigned int const blocks_per_grid =
      (size + threads_per_block - 1) / threads_per_block;
  // unpack the deltas and PDE coefficients:
  double const k = std::get<0>(deltas);
  double const h = std::get<1>(deltas);
  // const coefficients:
  double const lambda = k / (h * h);
  double const gamma = k / (2.0 * h);
  double const delta = 0.5 * k;
  // unpack PDE coefficients:
  auto const &a = std::get<0>(holder);
  auto const &b = std::get<1>(holder);
  auto const &c = std::get<2>(holder);
  // create scheme coefficients:
  auto const &A = [&](double x, double t) {
    return (lambda * a(x) - gamma * b(x));
  };
  auto const &B = [&](double x, double t) {
    return (lambda * a(x) - delta * c(x));
  };
  auto const &D = [&](double x, double t) {
    return (lambda * a(x) + gamma * b(x));
  };
  // store bc:
  auto const &left = dirichlet_boundary->first;
  auto const &right = dirichlet_boundary->second;

  double time = terminal_time - k;

  // prepare pointers for PDE space variable coeffs on device:
  double *d_A = NULL;
  double *d_B = NULL;
  double *d_D = NULL;
  // allocate block memory on device for PDE coeffs:
  cudaMalloc((void **)&d_A, size * sizeof(double));
  cudaMalloc((void **)&d_B, size * sizeof(double));
  cudaMalloc((void **)&d_D, size * sizeof(double));
  // create vector for PDE coeffs on host:
  std::vector<double> h_A(size, NaN<double>());
  std::vector<double> h_B(size, NaN<double>());
  std::vector<double> h_D(size, NaN<double>());

  // source is zero:
  while (time >= 0.0) {
    // discretize PDE space variable coeffs on host:
    v_discretization<double>::discretize_in_space(h, space_start, time, A, h_A);
    v_discretization<double>::discretize_in_space(h, space_start, time, B, h_B);
    v_discretization<double>::discretize_in_space(h, space_start, time, D, h_D);
    // copy h_A,h_B,h_D over to d_A,d_B,d_D (host => device ):
    cudaMemcpy(d_A, h_A.data(), size * sizeof(double),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size * sizeof(double),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, h_D.data(), size * sizeof(double),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    // populate new solution in d_next:
    explicit_euler_iterate_1d<double><<<threads_per_block, blocks_per_grid>>>(
        d_prev, d_next, d_A, d_B, d_D, size);
    // fill in the dirichlet boundaries in d_next:
    fill_dirichlet_bc_1d<double><<<threads_per_block, blocks_per_grid>>>(
        d_next, left(time), right(time), size);
    // swap the two pointers:
    swap(d_prev, d_next);
    time -= k;
  }

  // copy the contents of d_next to the solution pointer:
  cudaMemcpy(solution, d_prev, size * sizeof(double),
             cudaMemcpyKind::cudaMemcpyDeviceToHost);
  // free allocated memory blocks on device:
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_D);
  cudaFree(d_prev);
  cudaFree(d_next);
}

template <>
void heat_euler_scheme_backward_policy<
    float, pde_coefficient_holder_fun_1_arg<float>>::
    traverse(float *solution, float const *init_solution,
             unsigned long long size,
             sptr_t<dirichlet_boundary_1d<float>> const &dirichlet_boundary,
             std::pair<float, float> const &deltas,
             pde_coefficient_holder_fun_1_arg<float> const &holder,
             float const &terminal_time, float const &space_start,
             std::function<float(float, float)> const &source) {
  // prepare pointers on device:
  float *d_prev = NULL;
  float *d_next = NULL;
  // allocate block of memory on device:
  cudaMalloc((void **)&d_prev, size * sizeof(float));
  cudaMalloc((void **)&d_next, size * sizeof(float));
  // copy contents of input to d_prev:
  cudaMemcpy(d_prev, init_solution, size * sizeof(float),
             cudaMemcpyKind::cudaMemcpyHostToDevice);

  unsigned int const threads_per_block = THREADS_PER_BLOCK;
  unsigned int const blocks_per_grid =
      (size + threads_per_block - 1) / threads_per_block;
  // unpack the deltas and PDE coefficients:
  float const k = std::get<0>(deltas);
  float const h = std::get<1>(deltas);
  // const coefficients:
  float const lambda = k / (h * h);
  float const gamma = k / (2.0 * h);
  float const delta = 0.5 * k;
  // unpack PDE coefficients:
  auto const &a = std::get<0>(holder);
  auto const &b = std::get<1>(holder);
  auto const &c = std::get<2>(holder);
  // create scheme coefficients:
  auto const &A = [&](float x, float t) {
    return (lambda * a(x) - gamma * b(x));
  };
  auto const &B = [&](float x, float t) {
    return (lambda * a(x) - delta * c(x));
  };
  auto const &D = [&](float x, float t) {
    return (lambda * a(x) + gamma * b(x));
  };
  // store bc:
  auto const &left = dirichlet_boundary->first;
  auto const &right = dirichlet_boundary->second;

  float time = terminal_time - k;

  // prepare pointers for PDE space variable coeffs on device:
  float *d_A = NULL;
  float *d_B = NULL;
  float *d_D = NULL;
  // allocate block memory on device for PDE coeffs:
  cudaMalloc((void **)&d_A, size * sizeof(float));
  cudaMalloc((void **)&d_B, size * sizeof(float));
  cudaMalloc((void **)&d_D, size * sizeof(float));
  // create vector for PDE coeffs on host:
  std::vector<float> h_A(size, NaN<float>());
  std::vector<float> h_B(size, NaN<float>());
  std::vector<float> h_D(size, NaN<float>());

  // prepare a pointer for source on device:
  float *d_source = NULL;
  // allocate block memory on device:
  cudaMalloc((void **)&d_source, size * sizeof(float));
  // create vector for source on host:
  std::vector<float> h_source(size, NaN<float>());
  // source is zero:
  while (time >= 0.0) {
    // discretize source function on host:
    v_discretization<float>::discretize_in_space(h, space_start, time, source,
                                                 h_source);
    // discretize PDE space variable coeffs on host:
    v_discretization<float>::discretize_in_space(h, space_start, time, A, h_A);
    v_discretization<float>::discretize_in_space(h, space_start, time, B, h_B);
    v_discretization<float>::discretize_in_space(h, space_start, time, D, h_D);
    // copy h_source contents to d_source (host => device ):
    cudaMemcpy(d_source, h_source.data(), size * sizeof(float),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    // copy h_A,h_B,h_D over to d_A,d_B,d_D (host => device ):
    cudaMemcpy(d_A, h_A.data(), size * sizeof(float),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size * sizeof(float),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, h_D.data(), size * sizeof(float),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    // populate new solution in d_next:
    explicit_euler_iterate_1d<float><<<threads_per_block, blocks_per_grid>>>(
        d_prev, d_next, d_source, d_A, d_B, d_D, k, size);
    // fill in the dirichlet boundaries in d_next:
    fill_dirichlet_bc_1d<float><<<threads_per_block, blocks_per_grid>>>(
        d_next, left(time), right(time), size);
    // swap the two pointers:
    swap(d_prev, d_next);
    time -= k;
  }
  // free allocated memory blocks on device:
  cudaFree(d_source);
  // copy the contents of d_next to the solution pointer:
  cudaMemcpy(solution, d_prev, size * sizeof(float),
             cudaMemcpyKind::cudaMemcpyDeviceToHost);
  // free allocated memory blocks on device:
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_D);
  cudaFree(d_prev);
  cudaFree(d_next);
}

template <>
void heat_euler_scheme_backward_policy<
    double, pde_coefficient_holder_fun_1_arg<double>>::
    traverse(double *solution, double const *init_solution,
             unsigned long long size,
             sptr_t<dirichlet_boundary_1d<double>> const &dirichlet_boundary,
             std::pair<double, double> const &deltas,
             pde_coefficient_holder_fun_1_arg<double> const &holder,
             double const &terminal_time, double const &space_start,
             std::function<double(double, double)> const &source) {
  // prepare pointers on device:
  double *d_prev = NULL;
  double *d_next = NULL;
  // allocate block of memory on device:
  cudaMalloc((void **)&d_prev, size * sizeof(double));
  cudaMalloc((void **)&d_next, size * sizeof(double));
  // copy contents of input to d_prev:
  cudaMemcpy(d_prev, init_solution, size * sizeof(double),
             cudaMemcpyKind::cudaMemcpyHostToDevice);

  unsigned int const threads_per_block = THREADS_PER_BLOCK;
  unsigned int const blocks_per_grid =
      (size + threads_per_block - 1) / threads_per_block;
  // unpack the deltas and PDE coefficients:
  double const k = std::get<0>(deltas);
  double const h = std::get<1>(deltas);
  // const coefficients:
  double const lambda = k / (h * h);
  double const gamma = k / (2.0 * h);
  double const delta = 0.5 * k;
  // unpack PDE coefficients:
  auto const &a = std::get<0>(holder);
  auto const &b = std::get<1>(holder);
  auto const &c = std::get<2>(holder);
  // create scheme coefficients:
  auto const &A = [&](double x, double t) {
    return (lambda * a(x) - gamma * b(x));
  };
  auto const &B = [&](double x, double t) {
    return (lambda * a(x) - delta * c(x));
  };
  auto const &D = [&](double x, double t) {
    return (lambda * a(x) + gamma * b(x));
  };
  // store bc:
  auto const &left = dirichlet_boundary->first;
  auto const &right = dirichlet_boundary->second;

  double time = terminal_time - k;

  // prepare pointers for PDE space variable coeffs on device:
  double *d_A = NULL;
  double *d_B = NULL;
  double *d_D = NULL;
  // allocate block memory on device for PDE coeffs:
  cudaMalloc((void **)&d_A, size * sizeof(double));
  cudaMalloc((void **)&d_B, size * sizeof(double));
  cudaMalloc((void **)&d_D, size * sizeof(double));
  // create vector for PDE coeffs on host:
  std::vector<double> h_A(size, NaN<double>());
  std::vector<double> h_B(size, NaN<double>());
  std::vector<double> h_D(size, NaN<double>());

  // prepare a pointer for source on device:
  double *d_source = NULL;
  // allocate block memory on device:
  cudaMalloc((void **)&d_source, size * sizeof(double));
  // create vector for source on host:
  std::vector<double> h_source(size, NaN<double>());
  // source is zero:
  while (time >= 0.0) {
    // discretize source function on host:
    v_discretization<double>::discretize_in_space(h, space_start, time, source,
                                                  h_source);
    // discretize PDE space variable coeffs on host:
    v_discretization<double>::discretize_in_space(h, space_start, time, A, h_A);
    v_discretization<double>::discretize_in_space(h, space_start, time, B, h_B);
    v_discretization<double>::discretize_in_space(h, space_start, time, D, h_D);
    // copy h_source contents to d_source (host => device ):
    cudaMemcpy(d_source, h_source.data(), size * sizeof(double),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    // copy h_A,h_B,h_D over to d_A,d_B,d_D (host => device ):
    cudaMemcpy(d_A, h_A.data(), size * sizeof(double),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size * sizeof(double),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, h_D.data(), size * sizeof(double),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    // populate new solution in d_next:
    explicit_euler_iterate_1d<double><<<threads_per_block, blocks_per_grid>>>(
        d_prev, d_next, d_source, d_A, d_B, d_D, k, size);
    // fill in the dirichlet boundaries in d_next:
    fill_dirichlet_bc_1d<double><<<threads_per_block, blocks_per_grid>>>(
        d_next, left(time), right(time), size);
    // swap the two pointers:
    swap(d_prev, d_next);
    time -= k;
  }
  // free allocated memory blocks on device:
  cudaFree(d_source);
  // copy the contents of d_next to the solution pointer:
  cudaMemcpy(solution, d_prev, size * sizeof(double),
             cudaMemcpyKind::cudaMemcpyDeviceToHost);
  // free allocated memory blocks on device:
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_D);
  cudaFree(d_prev);
  cudaFree(d_next);
}

template <>
void heat_euler_scheme_backward_policy<
    float, pde_coefficient_holder_fun_1_arg<float>>::
    traverse(float *solution, float const *init_solution,
             unsigned long long size,
             robin_boundary<float> const &robin_boundary,
             std::pair<float, float> const &deltas,
             pde_coefficient_holder_fun_1_arg<float> const &holder,
             float const &terminal_time, float const &space_start) {
  // prepare pointers on device:
  float *d_prev = NULL;
  float *d_next = NULL;
  // allocate block of memory on device:
  cudaMalloc((void **)&d_prev, size * sizeof(float));
  cudaMalloc((void **)&d_next, size * sizeof(float));
  // copy contents of input to d_prev:
  cudaMemcpy(d_prev, init_solution, size * sizeof(float),
             cudaMemcpyKind::cudaMemcpyHostToDevice);

  unsigned int const threads_per_block = THREADS_PER_BLOCK;
  unsigned int const blocks_per_grid =
      (size + threads_per_block - 1) / threads_per_block;
  // unpack the deltas and PDE coefficients:
  float const k = std::get<0>(deltas);
  float const h = std::get<1>(deltas);
  // const coefficients:
  float const lambda = k / (h * h);
  float const gamma = k / (2.0 * h);
  float const delta = 0.5 * k;
  // unpack PDE coefficients:
  auto const &a = std::get<0>(holder);
  auto const &b = std::get<1>(holder);
  auto const &c = std::get<2>(holder);
  // create scheme coefficients:
  auto const &A = [&](float x, float t) {
    return (lambda * a(x) - gamma * b(x));
  };
  auto const &B = [&](float x, float t) {
    return (lambda * a(x) - delta * c(x));
  };
  auto const &D = [&](float x, float t) {
    return (lambda * a(x) + gamma * b(x));
  };
  // store bc:
  float const left_linear = robin_boundary.left.first;
  float const left_const = robin_boundary.left.second;
  float const right_linear = robin_boundary.right.first;
  float const right_const = robin_boundary.right.second;

  float time = terminal_time - k;

  // prepare pointers for PDE space variable coeffs on device:
  float *d_A = NULL;
  float *d_B = NULL;
  float *d_D = NULL;
  // allocate block memory on device for PDE coeffs:
  cudaMalloc((void **)&d_A, size * sizeof(float));
  cudaMalloc((void **)&d_B, size * sizeof(float));
  cudaMalloc((void **)&d_D, size * sizeof(float));
  // create vector for PDE coeffs on host:
  std::vector<float> h_A(size, NaN<float>());
  std::vector<float> h_B(size, NaN<float>());
  std::vector<float> h_D(size, NaN<float>());

  while (time >= 0.0) {
    // discretize PDE space variable coeffs on host:
    v_discretization<float>::discretize_in_space(h, space_start, time, A, h_A);
    v_discretization<float>::discretize_in_space(h, space_start, time, B, h_B);
    v_discretization<float>::discretize_in_space(h, space_start, time, D, h_D);
    // copy h_A,h_B,h_D over to d_A,d_B,d_D (host => device ):
    cudaMemcpy(d_A, h_A.data(), size * sizeof(float),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size * sizeof(float),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, h_D.data(), size * sizeof(float),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    // populate new solution in d_next:
    explicit_euler_iterate_1d<float><<<threads_per_block, blocks_per_grid>>>(
        d_prev, d_next, d_A, d_B, d_D, size);
    // fill in the dirichlet boundaries in d_next:
    fill_robin_bc_1d<float><<<threads_per_block, blocks_per_grid>>>(
        d_next, d_A, d_B, d_D, left_linear, left_const, right_linear,
        right_const, size);
    // swap the two pointers:
    swap(d_prev, d_next);
    time -= k;
  }

  // copy the contents of d_next to the solution pointer:
  cudaMemcpy(solution, d_prev, size * sizeof(float),
             cudaMemcpyKind::cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_D);
  cudaFree(d_prev);
  cudaFree(d_next);
}

template <>
void heat_euler_scheme_backward_policy<
    double, pde_coefficient_holder_fun_1_arg<double>>::
    traverse(double *solution, double const *init_solution,
             unsigned long long size,
             robin_boundary<double> const &robin_boundary,
             std::pair<double, double> const &deltas,
             pde_coefficient_holder_fun_1_arg<double> const &holder,
             double const &terminal_time, double const &space_start) {
  // prepare pointers on device:
  double *d_prev = NULL;
  double *d_next = NULL;
  // allocate block of memory on device:
  cudaMalloc((void **)&d_prev, size * sizeof(double));
  cudaMalloc((void **)&d_next, size * sizeof(double));
  // copy contents of input to d_prev:
  cudaMemcpy(d_prev, init_solution, size * sizeof(double),
             cudaMemcpyKind::cudaMemcpyHostToDevice);

  unsigned int const threads_per_block = THREADS_PER_BLOCK;
  unsigned int const blocks_per_grid =
      (size + threads_per_block - 1) / threads_per_block;
  // unpack the deltas and PDE coefficients:
  double const k = std::get<0>(deltas);
  double const h = std::get<1>(deltas);
  // const coefficients:
  double const lambda = k / (h * h);
  double const gamma = k / (2.0 * h);
  double const delta = 0.5 * k;
  // unpack PDE coefficients:
  auto const &a = std::get<0>(holder);
  auto const &b = std::get<1>(holder);
  auto const &c = std::get<2>(holder);
  // create scheme coefficients:
  auto const &A = [&](double x, double t) {
    return (lambda * a(x) - gamma * b(x));
  };
  auto const &B = [&](double x, double t) {
    return (lambda * a(x) - delta * c(x));
  };
  auto const &D = [&](double x, double t) {
    return (lambda * a(x) + gamma * b(x));
  };
  // store bc:
  double const left_linear = robin_boundary.left.first;
  double const left_const = robin_boundary.left.second;
  double const right_linear = robin_boundary.right.first;
  double const right_const = robin_boundary.right.second;

  double time = terminal_time - k;

  // prepare pointers for PDE space variable coeffs on device:
  double *d_A = NULL;
  double *d_B = NULL;
  double *d_D = NULL;
  // allocate block memory on device for PDE coeffs:
  cudaMalloc((void **)&d_A, size * sizeof(double));
  cudaMalloc((void **)&d_B, size * sizeof(double));
  cudaMalloc((void **)&d_D, size * sizeof(double));
  // create vector for PDE coeffs on host:
  std::vector<double> h_A(size, NaN<double>());
  std::vector<double> h_B(size, NaN<double>());
  std::vector<double> h_D(size, NaN<double>());

  while (time >= 0.0) {
    // discretize PDE space variable coeffs on host:
    v_discretization<double>::discretize_in_space(h, space_start, time, A, h_A);
    v_discretization<double>::discretize_in_space(h, space_start, time, B, h_B);
    v_discretization<double>::discretize_in_space(h, space_start, time, D, h_D);
    // copy h_A,h_B,h_D over to d_A,d_B,d_D (host => device ):
    cudaMemcpy(d_A, h_A.data(), size * sizeof(double),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size * sizeof(double),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, h_D.data(), size * sizeof(double),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    // populate new solution in d_next:
    explicit_euler_iterate_1d<double><<<threads_per_block, blocks_per_grid>>>(
        d_prev, d_next, d_A, d_B, d_D, size);
    // fill in the dirichlet boundaries in d_next:
    fill_robin_bc_1d<double><<<threads_per_block, blocks_per_grid>>>(
        d_next, d_A, d_B, d_D, left_linear, left_const, right_linear,
        right_const, size);
    // swap the two pointers:
    swap(d_prev, d_next);
    time -= k;
  }

  // copy the contents of d_next to the solution pointer:
  cudaMemcpy(solution, d_prev, size * sizeof(double),
             cudaMemcpyKind::cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_D);
  cudaFree(d_prev);
  cudaFree(d_next);
}

template <>
void heat_euler_scheme_backward_policy<
    float, pde_coefficient_holder_fun_1_arg<float>>::
    traverse(float *solution, float const *init_solution,
             unsigned long long size,
             robin_boundary<float> const &robin_boundary,
             std::pair<float, float> const &deltas,
             pde_coefficient_holder_fun_1_arg<float> const &holder,
             float const &terminal_time, float const &space_start,
             std::function<float(float, float)> const &source) {
  // prepare pointers on device:
  float *d_prev = NULL;
  float *d_next = NULL;
  // allocate block of memory on device:
  cudaMalloc((void **)&d_prev, size * sizeof(float));
  cudaMalloc((void **)&d_next, size * sizeof(float));
  // copy contents of input to d_prev:
  cudaMemcpy(d_prev, init_solution, size * sizeof(float),
             cudaMemcpyKind::cudaMemcpyHostToDevice);

  unsigned int const threads_per_block = THREADS_PER_BLOCK;
  unsigned int const blocks_per_grid =
      (size + threads_per_block - 1) / threads_per_block;
  // unpack the deltas and PDE coefficients:
  float const k = std::get<0>(deltas);
  float const h = std::get<1>(deltas);
  // const coefficients:
  float const lambda = k / (h * h);
  float const gamma = k / (2.0 * h);
  float const delta = 0.5 * k;
  // unpack PDE coefficients:
  auto const &a = std::get<0>(holder);
  auto const &b = std::get<1>(holder);
  auto const &c = std::get<2>(holder);
  // create scheme coefficients:
  auto const &A = [&](float x, float t) {
    return (lambda * a(x) - gamma * b(x));
  };
  auto const &B = [&](float x, float t) {
    return (lambda * a(x) - delta * c(x));
  };
  auto const &D = [&](float x, float t) {
    return (lambda * a(x) + gamma * b(x));
  };
  // store bc:
  float const left_linear = robin_boundary.left.first;
  float const left_const = robin_boundary.left.second;
  float const right_linear = robin_boundary.right.first;
  float const right_const = robin_boundary.right.second;

  float time = terminal_time - k;

  // prepare pointers for PDE space variable coeffs on device:
  float *d_A = NULL;
  float *d_B = NULL;
  float *d_D = NULL;
  // allocate block memory on device for PDE coeffs:
  cudaMalloc((void **)&d_A, size * sizeof(float));
  cudaMalloc((void **)&d_B, size * sizeof(float));
  cudaMalloc((void **)&d_D, size * sizeof(float));
  // create vector for PDE coeffs on host:
  std::vector<float> h_A(size, NaN<float>());
  std::vector<float> h_B(size, NaN<float>());
  std::vector<float> h_D(size, NaN<float>());

  // prepare a pointer for source on device:
  float *d_source = NULL;
  // allocate block memory on device:
  cudaMalloc((void **)&d_source, size * sizeof(float));
  // create vector on host:
  std::vector<float> h_source(size, NaN<float>());
  // source is zero:
  while (time >= 0.0) {
    // discretize source function on host:
    v_discretization<float>::discretize_in_space(h, space_start, time, source,
                                                 h_source);
    // discretize PDE space variable coeffs on host:
    v_discretization<float>::discretize_in_space(h, space_start, time, A, h_A);
    v_discretization<float>::discretize_in_space(h, space_start, time, B, h_B);
    v_discretization<float>::discretize_in_space(h, space_start, time, D, h_D);
    // copy h_source contents to d_source (host => device ):
    cudaMemcpy(d_source, h_source.data(), size * sizeof(float),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    // copy h_A,h_B,h_D over to d_A,d_B,d_D (host => device ):
    cudaMemcpy(d_A, h_A.data(), size * sizeof(float),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size * sizeof(float),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, h_D.data(), size * sizeof(float),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    // populate new solution in d_next:
    explicit_euler_iterate_1d<float><<<threads_per_block, blocks_per_grid>>>(
        d_prev, d_next, d_source, d_A, d_B, d_D, k, size);
    // fill in the dirichlet boundaries in d_next:
    fill_robin_bc_1d<float><<<threads_per_block, blocks_per_grid>>>(
        d_next, h_source.front(), h_source.back(), d_A, d_B, d_D, k,
        left_linear, left_const, right_linear, right_const, size);
    // swap the two pointers:
    swap(d_prev, d_next);
    time -= k;
  }
  // free allocated memory blocks on device:
  cudaFree(d_source);
  // copy the contents of d_next to the solution pointer:
  cudaMemcpy(solution, d_prev, size * sizeof(float),
             cudaMemcpyKind::cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_D);
  cudaFree(d_prev);
  cudaFree(d_next);
}

template <>
void heat_euler_scheme_backward_policy<
    double, pde_coefficient_holder_fun_1_arg<double>>::
    traverse(double *solution, double const *init_solution,
             unsigned long long size,
             robin_boundary<double> const &robin_boundary,
             std::pair<double, double> const &deltas,
             pde_coefficient_holder_fun_1_arg<double> const &holder,
             double const &terminal_time, double const &space_start,
             std::function<double(double, double)> const &source) {
  // prepare pointers on device:
  double *d_prev = NULL;
  double *d_next = NULL;
  // allocate block of memory on device:
  cudaMalloc((void **)&d_prev, size * sizeof(double));
  cudaMalloc((void **)&d_next, size * sizeof(double));
  // copy contents of input to d_prev:
  cudaMemcpy(d_prev, init_solution, size * sizeof(double),
             cudaMemcpyKind::cudaMemcpyHostToDevice);

  unsigned int const threads_per_block = THREADS_PER_BLOCK;
  unsigned int const blocks_per_grid =
      (size + threads_per_block - 1) / threads_per_block;
  // unpack the deltas and PDE coefficients:
  double const k = std::get<0>(deltas);
  double const h = std::get<1>(deltas);
  // const coefficients:
  double const lambda = k / (h * h);
  double const gamma = k / (2.0 * h);
  double const delta = 0.5 * k;
  // unpack PDE coefficients:
  auto const &a = std::get<0>(holder);
  auto const &b = std::get<1>(holder);
  auto const &c = std::get<2>(holder);
  // create scheme coefficients:
  auto const &A = [&](double x, double t) {
    return (lambda * a(x) - gamma * b(x));
  };
  auto const &B = [&](double x, double t) {
    return (lambda * a(x) - delta * c(x));
  };
  auto const &D = [&](double x, double t) {
    return (lambda * a(x) + gamma * b(x));
  };
  // store bc:
  double const left_linear = robin_boundary.left.first;
  double const left_const = robin_boundary.left.second;
  double const right_linear = robin_boundary.right.first;
  double const right_const = robin_boundary.right.second;

  double time = terminal_time - k;

  // prepare pointers for PDE space variable coeffs on device:
  double *d_A = NULL;
  double *d_B = NULL;
  double *d_D = NULL;
  // allocate block memory on device for PDE coeffs:
  cudaMalloc((void **)&d_A, size * sizeof(double));
  cudaMalloc((void **)&d_B, size * sizeof(double));
  cudaMalloc((void **)&d_D, size * sizeof(double));
  // create vector for PDE coeffs on host:
  std::vector<double> h_A(size, NaN<double>());
  std::vector<double> h_B(size, NaN<double>());
  std::vector<double> h_D(size, NaN<double>());

  // prepare a pointer for source on device:
  double *d_source = NULL;
  // allocate block memory on device:
  cudaMalloc((void **)&d_source, size * sizeof(double));
  // create vector on host:
  std::vector<double> h_source(size, NaN<double>());
  // source is zero:
  while (time >= 0.0) {
    // discretize source function on host:
    v_discretization<double>::discretize_in_space(h, space_start, time, source,
                                                  h_source);
    // discretize PDE space variable coeffs on host:
    v_discretization<double>::discretize_in_space(h, space_start, time, A, h_A);
    v_discretization<double>::discretize_in_space(h, space_start, time, B, h_B);
    v_discretization<double>::discretize_in_space(h, space_start, time, D, h_D);
    // copy h_source contents to d_source (host => device ):
    cudaMemcpy(d_source, h_source.data(), size * sizeof(double),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    // copy h_A,h_B,h_D over to d_A,d_B,d_D (host => device ):
    cudaMemcpy(d_A, h_A.data(), size * sizeof(double),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size * sizeof(double),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, h_D.data(), size * sizeof(double),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    // populate new solution in d_next:
    explicit_euler_iterate_1d<double><<<threads_per_block, blocks_per_grid>>>(
        d_prev, d_next, d_source, d_A, d_B, d_D, k, size);
    // fill in the dirichlet boundaries in d_next:
    fill_robin_bc_1d<double><<<threads_per_block, blocks_per_grid>>>(
        d_next, h_source.front(), h_source.back(), d_A, d_B, d_D, k,
        left_linear, left_const, right_linear, right_const, size);
    // swap the two pointers:
    swap(d_prev, d_next);
    time -= k;
  }
  // free allocated memory blocks on device:
  cudaFree(d_source);
  // copy the contents of d_next to the solution pointer:
  cudaMemcpy(solution, d_prev, size * sizeof(double),
             cudaMemcpyKind::cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_D);
  cudaFree(d_prev);
  cudaFree(d_next);
}

};  // namespace lss_one_dim_space_variable_heat_explicit_schemes_cuda_policy
