#include <device_launch_parameters.h>

#include <limits>

#include "lss_one_dim_space_variable_heat_cuda_kernels.h"
#include "lss_one_dim_space_variable_heat_explicit_schemes_cuda.h"

namespace lss_one_dim_space_variable_heat_explicit_schemes_cuda {

// Move this somewhere else:
template <typename T>
static constexpr T NaN() {
  return std::numeric_limits<T>::quiet_NaN();
}

using lss_one_dim_space_variable_heat_cuda_kernels::explicitEulerIterate1D;
using lss_one_dim_space_variable_heat_cuda_kernels::fillDirichletBC1D;
using lss_one_dim_space_variable_heat_cuda_kernels::fillRobinBC1D;
using lss_utility::swap;

void ExplicitEulerLoopSP::operator()(
    float const *input, std::pair<float, float> const &boundaryPair,
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

  unsigned int const threadsPerBlock = THREADS_PER_BLOCK;
  unsigned int const blocksPerGrid =
      (size + threadsPerBlock - 1) / threadsPerBlock;
  // unpack the deltas and PDE coefficients:
  float const k = std::get<0>(deltas_);
  float const h = std::get<1>(deltas_);
  // const coefficients:
  float const lambda = k / (h * h);
  float const gamma = k / (2.0 * h);
  float const delta = 0.5 * k;
  // unpack PDE coefficients:
  auto const &a = std::get<0>(coeffs_);
  auto const &b = std::get<1>(coeffs_);
  auto const &c = std::get<2>(coeffs_);
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
  float const left = boundaryPair.first;
  float const right = boundaryPair.second;

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

  std::numeric_limits<double>::quiet_NaN();
  if (isSourceSet_) {
    // prepare a pointer for source on device:
    float *d_source = NULL;
    // allocate block memory on device:
    cudaMalloc((void **)&d_source, size * sizeof(float));
    // create vector for source on host:
    std::vector<float> h_source(size, NaN<float>());
    // source is zero:
    while (time <= terminalT_) {
      // discretize source function on host:
      discretizeInSpace(h, spaceStart_, time, source_, h_source);
      // discretize PDE space variable coeffs on host:
      discretizeInSpace(h, spaceStart_, time, A, h_A);
      discretizeInSpace(h, spaceStart_, time, B, h_B);
      discretizeInSpace(h, spaceStart_, time, D, h_D);
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
      explicitEulerIterate1D<float><<<threadsPerBlock, blocksPerGrid>>>(
          d_prev, d_next, d_source, d_A, d_B, d_D, k, size);
      // fill in the dirichlet boundaries in d_next:
      fillDirichletBC1D<float>
          <<<threadsPerBlock, blocksPerGrid>>>(d_next, left, right, size);
      // swap the two pointers:
      swap(d_prev, d_next);
      time += k;
    }
    // free allocated memory blocks on device:
    cudaFree(d_source);
  } else {
    // source is zero:
    while (time <= terminalT_) {
      // discretize PDE space variable coeffs on host:
      discretizeInSpace(h, spaceStart_, time, A, h_A);
      discretizeInSpace(h, spaceStart_, time, B, h_B);
      discretizeInSpace(h, spaceStart_, time, D, h_D);
      // copy h_A,h_B,h_D over to d_A,d_B,d_D (host => device ):
      cudaMemcpy(d_A, h_A.data(), size * sizeof(float),
                 cudaMemcpyKind::cudaMemcpyHostToDevice);
      cudaMemcpy(d_B, h_B.data(), size * sizeof(float),
                 cudaMemcpyKind::cudaMemcpyHostToDevice);
      cudaMemcpy(d_D, h_D.data(), size * sizeof(float),
                 cudaMemcpyKind::cudaMemcpyHostToDevice);
      // populate new solution in d_next:
      explicitEulerIterate1D<float><<<threadsPerBlock, blocksPerGrid>>>(
          d_prev, d_next, d_A, d_B, d_D, size);
      // fill in the dirichlet boundaries in d_next:
      fillDirichletBC1D<float>
          <<<threadsPerBlock, blocksPerGrid>>>(d_next, left, right, size);
      // swap the two pointers:
      swap(d_prev, d_next);
      time += k;
    }
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

void ExplicitEulerLoopDP::operator()(
    double const *input, std::pair<double, double> const &boundaryPair,
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

  unsigned int const threadsPerBlock = THREADS_PER_BLOCK;
  unsigned int const blocksPerGrid =
      (size + threadsPerBlock - 1) / threadsPerBlock;
  // unpack the deltas and PDE coefficients:
  double const k = std::get<0>(deltas_);
  double const h = std::get<1>(deltas_);
  // const coefficients:
  double const lambda = k / (h * h);
  double const gamma = k / (2.0 * h);
  double const delta = 0.5 * k;
  // unpack PDE coefficients:
  auto const &a = std::get<0>(coeffs_);
  auto const &b = std::get<1>(coeffs_);
  auto const &c = std::get<2>(coeffs_);
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
  double const left = boundaryPair.first;
  double const right = boundaryPair.second;

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

  if (isSourceSet_) {
    // prepare a pointer for source on device:
    double *d_source = NULL;
    // allocate block memory on device:
    cudaMalloc((void **)&d_source, size * sizeof(double));
    // create vector for source on host:
    std::vector<double> h_source(size, NaN<double>());
    // source is zero:
    while (time <= terminalT_) {
      // discretize source function on host:
      discretizeInSpace(h, spaceStart_, time, source_, h_source);
      // discretize PDE space variable coeffs on host:
      discretizeInSpace(h, spaceStart_, time, A, h_A);
      discretizeInSpace(h, spaceStart_, time, B, h_B);
      discretizeInSpace(h, spaceStart_, time, D, h_D);
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
      explicitEulerIterate1D<double><<<threadsPerBlock, blocksPerGrid>>>(
          d_prev, d_next, d_source, d_A, d_B, d_D, k, size);
      // fill in the dirichlet boundaries in d_next:
      fillDirichletBC1D<double>
          <<<threadsPerBlock, blocksPerGrid>>>(d_next, left, right, size);
      // swap the two pointers:
      swap(d_prev, d_next);
      time += k;
    }
    // free allocated memory blocks on device:
    cudaFree(d_source);
  } else {
    // source is zero:
    while (time <= terminalT_) {
      // discretize PDE space variable coeffs on host:
      discretizeInSpace(h, spaceStart_, time, A, h_A);
      discretizeInSpace(h, spaceStart_, time, B, h_B);
      discretizeInSpace(h, spaceStart_, time, D, h_D);
      // copy h_A,h_B,h_D over to d_A,d_B,d_D (host => device ):
      cudaMemcpy(d_A, h_A.data(), size * sizeof(double),
                 cudaMemcpyKind::cudaMemcpyHostToDevice);
      cudaMemcpy(d_B, h_B.data(), size * sizeof(double),
                 cudaMemcpyKind::cudaMemcpyHostToDevice);
      cudaMemcpy(d_D, h_D.data(), size * sizeof(double),
                 cudaMemcpyKind::cudaMemcpyHostToDevice);
      // populate new solution in d_next:
      explicitEulerIterate1D<double><<<threadsPerBlock, blocksPerGrid>>>(
          d_prev, d_next, d_A, d_B, d_D, size);
      // fill in the dirichlet boundaries in d_next:
      fillDirichletBC1D<double>
          <<<threadsPerBlock, blocksPerGrid>>>(d_next, left, right, size);
      // swap the two pointers:
      swap(d_prev, d_next);
      time += k;
    }
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

void ExplicitEulerLoopSP::operator()(float const *input,
                                     std::pair<float, float> const &leftPair,
                                     std::pair<float, float> const &rightPair,
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

  unsigned int const threadsPerBlock = THREADS_PER_BLOCK;
  unsigned int const blocksPerGrid =
      (size + threadsPerBlock - 1) / threadsPerBlock;
  // unpack the deltas and PDE coefficients:
  float const k = std::get<0>(deltas_);
  float const h = std::get<1>(deltas_);
  // const coefficients:
  float const lambda = k / (h * h);
  float const gamma = k / (2.0 * h);
  float const delta = 0.5 * k;
  // unpack PDE coefficients:
  auto const &a = std::get<0>(coeffs_);
  auto const &b = std::get<1>(coeffs_);
  auto const &c = std::get<2>(coeffs_);
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
  float const leftLinear = leftPair.first;
  float const leftConst = leftPair.second;
  float const rightLinear = rightPair.first;
  float const rightConst = rightPair.second;

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

  if (isSourceSet_) {
    // prepare a pointer for source on device:
    float *d_source = NULL;
    // allocate block memory on device:
    cudaMalloc((void **)&d_source, size * sizeof(float));
    // create vector on host:
    std::vector<float> h_source(size, NaN<float>());
    // source is zero:
    while (time <= terminalT_) {
      // discretize source function on host:
      discretizeInSpace(h, spaceStart_, time, source_, h_source);
      // discretize PDE space variable coeffs on host:
      discretizeInSpace(h, spaceStart_, time, A, h_A);
      discretizeInSpace(h, spaceStart_, time, B, h_B);
      discretizeInSpace(h, spaceStart_, time, D, h_D);
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
      explicitEulerIterate1D<float><<<threadsPerBlock, blocksPerGrid>>>(
          d_prev, d_next, d_source, d_A, d_B, d_D, k, size);
      // fill in the dirichlet boundaries in d_next:
      fillRobinBC1D<float><<<threadsPerBlock, blocksPerGrid>>>(
          d_next, h_source.front(), h_source.back(), d_A, d_B, d_D, k,
          leftLinear, leftConst, rightLinear, rightConst, size);
      // swap the two pointers:
      swap(d_prev, d_next);
      time += k;
    }
    // free allocated memory blocks on device:
    cudaFree(d_source);
  } else {
    while (time <= terminalT_) {
      // discretize PDE space variable coeffs on host:
      discretizeInSpace(h, spaceStart_, time, A, h_A);
      discretizeInSpace(h, spaceStart_, time, B, h_B);
      discretizeInSpace(h, spaceStart_, time, D, h_D);
      // copy h_A,h_B,h_D over to d_A,d_B,d_D (host => device ):
      cudaMemcpy(d_A, h_A.data(), size * sizeof(float),
                 cudaMemcpyKind::cudaMemcpyHostToDevice);
      cudaMemcpy(d_B, h_B.data(), size * sizeof(float),
                 cudaMemcpyKind::cudaMemcpyHostToDevice);
      cudaMemcpy(d_D, h_D.data(), size * sizeof(float),
                 cudaMemcpyKind::cudaMemcpyHostToDevice);
      // populate new solution in d_next:
      explicitEulerIterate1D<float><<<threadsPerBlock, blocksPerGrid>>>(
          d_prev, d_next, d_A, d_B, d_D, size);
      // fill in the dirichlet boundaries in d_next:
      fillRobinBC1D<float><<<threadsPerBlock, blocksPerGrid>>>(
          d_next, d_A, d_B, d_D, leftLinear, leftConst, rightLinear, rightConst,
          size);
      // swap the two pointers:
      swap(d_prev, d_next);
      time += k;
    }
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

void ExplicitEulerLoopDP::operator()(double const *input,
                                     std::pair<double, double> const &leftPair,
                                     std::pair<double, double> const &rightPair,
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

  unsigned int const threadsPerBlock = THREADS_PER_BLOCK;
  unsigned int const blocksPerGrid =
      (size + threadsPerBlock - 1) / threadsPerBlock;
  // unpack the deltas and PDE coefficients:
  double const k = std::get<0>(deltas_);
  double const h = std::get<1>(deltas_);
  // const coefficients:
  double const lambda = k / (h * h);
  double const gamma = k / (2.0 * h);
  double const delta = 0.5 * k;
  // unpack PDE coefficients:
  auto const &a = std::get<0>(coeffs_);
  auto const &b = std::get<1>(coeffs_);
  auto const &c = std::get<2>(coeffs_);
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
  double const leftLinear = leftPair.first;
  double const leftConst = leftPair.second;
  double const rightLinear = rightPair.first;
  double const rightConst = rightPair.second;

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

  if (isSourceSet_) {
    // prepare a pointer for source on device:
    double *d_source = NULL;
    // allocate block memory on device:
    cudaMalloc((void **)&d_source, size * sizeof(double));
    // create vector on host:
    std::vector<double> h_source(size, NaN<double>());
    // source is zero:
    while (time <= terminalT_) {
      // discretize source function on host:
      discretizeInSpace(h, spaceStart_, time, source_, h_source);
      // discretize PDE space variable coeffs on host:
      discretizeInSpace(h, spaceStart_, time, A, h_A);
      discretizeInSpace(h, spaceStart_, time, B, h_B);
      discretizeInSpace(h, spaceStart_, time, D, h_D);
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
      explicitEulerIterate1D<double><<<threadsPerBlock, blocksPerGrid>>>(
          d_prev, d_next, d_source, d_A, d_B, d_D, k, size);
      // fill in the dirichlet boundaries in d_next:
      fillRobinBC1D<double><<<threadsPerBlock, blocksPerGrid>>>(
          d_next, h_source.front(), h_source.back(), d_A, d_B, d_D, k,
          leftLinear, leftConst, rightLinear, rightConst, size);
      // swap the two pointers:
      swap(d_prev, d_next);
      time += k;
    }
    // free allocated memory blocks on device:
    cudaFree(d_source);
  } else {
    while (time <= terminalT_) {
      // discretize PDE space variable coeffs on host:
      discretizeInSpace(h, spaceStart_, time, A, h_A);
      discretizeInSpace(h, spaceStart_, time, B, h_B);
      discretizeInSpace(h, spaceStart_, time, D, h_D);
      // copy h_A,h_B,h_D over to d_A,d_B,d_D (host => device ):
      cudaMemcpy(d_A, h_A.data(), size * sizeof(double),
                 cudaMemcpyKind::cudaMemcpyHostToDevice);
      cudaMemcpy(d_B, h_B.data(), size * sizeof(double),
                 cudaMemcpyKind::cudaMemcpyHostToDevice);
      cudaMemcpy(d_D, h_D.data(), size * sizeof(double),
                 cudaMemcpyKind::cudaMemcpyHostToDevice);
      // populate new solution in d_next:
      explicitEulerIterate1D<double><<<threadsPerBlock, blocksPerGrid>>>(
          d_prev, d_next, d_A, d_B, d_D, size);
      // fill in the dirichlet boundaries in d_next:
      fillRobinBC1D<double><<<threadsPerBlock, blocksPerGrid>>>(
          d_next, d_A, d_B, d_D, leftLinear, leftConst, rightLinear, rightConst,
          size);
      // swap the two pointers:
      swap(d_prev, d_next);
      time += k;
    }
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

}  // namespace lss_one_dim_space_variable_heat_explicit_schemes_cuda
