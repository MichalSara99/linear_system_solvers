#pragma once
#if !defined(_LSS_ONE_DIM_HEAT_EXPLICIT_SCHEMES_CUDA)
#define _LSS_ONE_DIM_HEAT_EXPLICIT_SCHEMES_CUDA

#include <tuple>

#include "common/lss_macros.h"
#include "common/lss_utility.h"
#include "pde_solvers/one_dim/lss_one_dim_pde_utility.h"

namespace lss_one_dim_heat_explicit_schemes_cuda {

using lss_one_dim_pde_utility::DirichletBoundary;
using lss_one_dim_pde_utility::Discretization;
using lss_one_dim_pde_utility::RobinBoundary;

class ExplicitEulerLoopSP
    : public Discretization<float, std::vector, std::allocator<float>> {
 private:
  float spaceStart_;
  float terminalT_;
  std::pair<float, float> deltas_;  // first = delta time, second = delta space;
  std::tuple<float, float, float> coeffs_;  // coefficients of PDE
  std::function<float(float, float)> source_;
  bool isSourceSet_;

 public:
  ~ExplicitEulerLoopSP() {}
  explicit ExplicitEulerLoopSP() = delete;
  explicit ExplicitEulerLoopSP(float spaceStart, float terminalTime,
                               std::pair<float, float> const &deltas,
                               std::tuple<float, float, float> const &coeffs,
                               std::function<float(float, float)> const &source,
                               bool isSourceSet = false)
      : spaceStart_{spaceStart},
        terminalT_{terminalTime},
        deltas_{deltas},
        coeffs_{coeffs},
        source_{source},
        isSourceSet_{isSourceSet} {}

  void operator()(float const *input,
                  DirichletBoundary<float> const &dirichletBoundary,
                  unsigned long long const size, float *solution) const;
  void operator()(float const *input, RobinBoundary<float> const &robinBoundary,
                  unsigned long long const size, float *solution) const;
};

class ExplicitEulerLoopDP
    : public Discretization<double, std::vector, std::allocator<double>> {
 private:
  double spaceStart_;
  double terminalT_;
  std::pair<double, double>
      deltas_;  // first = delta time, second = delta space;
  std::tuple<double, double, double> coeffs_;  // coefficients of PDE
  std::function<double(double, double)> source_;
  bool isSourceSet_;

 public:
  ~ExplicitEulerLoopDP() {}
  explicit ExplicitEulerLoopDP() = delete;
  explicit ExplicitEulerLoopDP(
      double spaceStart, double terminalTime,
      std::pair<double, double> const &deltas,
      std::tuple<double, double, double> const &coeffs,
      std::function<double(double, double)> const &source,
      bool isSourceSet = false)
      : spaceStart_{spaceStart},
        terminalT_{terminalTime},
        deltas_{deltas},
        coeffs_{coeffs},
        source_{source},
        isSourceSet_{isSourceSet} {}

  void operator()(double const *input,
                  DirichletBoundary<double> const &dirichletBoundary,
                  unsigned long long const size, double *solution) const;
  void operator()(double const *input,
                  RobinBoundary<double> const &robinBoundary,
                  unsigned long long const size, double *solution) const;
};

// ============================================================================
// ============ ExplicitEulerHeatEquationScheme general template ==============
// ============================================================================

template <typename T, template <typename, typename> typename Container,
          typename Alloc>
class ExplicitEulerHeatEquationScheme {};

// ============================================================================
// ==== Single-Precision Floating-Point ExplicitEulerHeatEquationScheme =======
// partial specialization template ============================================
// ============================================================================

template <template <typename, typename> typename Container, typename Alloc>
class ExplicitEulerHeatEquationScheme<float, Container, Alloc> {
 private:
  float spaceStart_;
  float terminalT_;
  std::pair<float, float> deltas_;  // first = delta time, second = delta space;
  std::tuple<float, float, float> coeffs_;  // coefficients of PDE
  Container<float, Alloc> init_;
  std::function<float(float, float)> source_;
  bool isSourceSet_;

 public:
  typedef float value_type;
  explicit ExplicitEulerHeatEquationScheme() = delete;
  explicit ExplicitEulerHeatEquationScheme(
      float spaceStart, float terminalTime,
      std::pair<float, float> const &deltas,
      std::tuple<float, float, float> const &coeffs,
      Container<float, Alloc> const &init,
      std::function<float(float, float)> const &source = nullptr,
      bool isSourceSet = false)
      : spaceStart_{spaceStart},
        terminalT_{terminalTime},
        deltas_{deltas},
        coeffs_{coeffs},
        init_{init},
        source_{source},
        isSourceSet_{isSourceSet} {}

  ~ExplicitEulerHeatEquationScheme() {}

  ExplicitEulerHeatEquationScheme(ExplicitEulerHeatEquationScheme const &) =
      delete;
  ExplicitEulerHeatEquationScheme(ExplicitEulerHeatEquationScheme &&) = delete;
  ExplicitEulerHeatEquationScheme &operator=(
      ExplicitEulerHeatEquationScheme const &) = delete;
  ExplicitEulerHeatEquationScheme &operator=(
      ExplicitEulerHeatEquationScheme &&) = delete;

  void operator()(DirichletBoundary<float> const &dirichletBoundary,
                  Container<float, Alloc> &solution) const;
  void operator()(RobinBoundary<float> const &robinBoundary,
                  Container<float, Alloc> &solution) const;
};

// ============================================================================
// ==== Double-Precision Floating-Point ExplicitEulerHeatEquationScheme partial
// specialization template ====================================================
// ============================================================================

template <template <typename, typename> typename Container, typename Alloc>
class ExplicitEulerHeatEquationScheme<double, Container, Alloc> {
 private:
  double spaceStart_;
  double terminalT_;
  std::pair<double, double>
      deltas_;  // first = delta time, second = delta space;
  std::tuple<double, double, double> coeffs_;  // coefficients of PDE
  Container<double, Alloc> init_;
  std::function<double(double, double)> source_;
  bool isSourceSet_;

 public:
  typedef double value_type;
  explicit ExplicitEulerHeatEquationScheme() = delete;
  explicit ExplicitEulerHeatEquationScheme(
      double spaceStart, double terminalTime,
      std::pair<double, double> const &deltas,
      std::tuple<double, double, double> const &coeffs,
      Container<double, Alloc> const &init,
      std::function<double(double, double)> const &source = nullptr,
      bool isSourceSet = false)
      : spaceStart_{spaceStart},
        terminalT_{terminalTime},
        deltas_{deltas},
        coeffs_{coeffs},
        init_{init},
        source_{source},
        isSourceSet_{isSourceSet} {}

  ~ExplicitEulerHeatEquationScheme() {}

  ExplicitEulerHeatEquationScheme(ExplicitEulerHeatEquationScheme const &) =
      delete;
  ExplicitEulerHeatEquationScheme(ExplicitEulerHeatEquationScheme &&) = delete;
  ExplicitEulerHeatEquationScheme &operator=(
      ExplicitEulerHeatEquationScheme const &) = delete;
  ExplicitEulerHeatEquationScheme &operator=(
      ExplicitEulerHeatEquationScheme &&) = delete;

  void operator()(DirichletBoundary<double> const &dirichletBoundary,
                  Container<double, Alloc> &solution) const;
  void operator()(RobinBoundary<double> const &robinBoundary,
                  Container<double, Alloc> &solution) const;
};

// ============================================================================
// ========================= IMPLEMENTATIONS  =================================

// ============================================================================
// ================ ExplicitEulerHeatEquationScheme implementation ============
// ============================================================================

template <template <typename, typename> typename Container, typename Alloc>
void ExplicitEulerHeatEquationScheme<float, Container, Alloc>::operator()(
    DirichletBoundary<float> const &dirichletBoundary,
    Container<float, Alloc> &solution) const {
  LSS_ASSERT(init_.size() == solution.size(),
             "Initial and final solution must have the same size");
  // get the size of the vector:
  std::size_t const size = solution.size();
  // create prev pointer:
  float *prev = (float *)malloc(size * sizeof(float));
  std::copy(init_.begin(), init_.end(), prev);
  // create next pointer:
  float *next = (float *)malloc(size * sizeof(float));
  // launch the Euler loop:
  ExplicitEulerLoopSP loop{spaceStart_, terminalT_, deltas_,
                           coeffs_,     source_,    isSourceSet_};
  loop(prev, dirichletBoundary, size, next);
  // next point to the solution
  std::copy(next, next + size, solution.begin());
  free(prev);
  free(next);
}

template <template <typename, typename> typename Container, typename Alloc>
void ExplicitEulerHeatEquationScheme<double, Container, Alloc>::operator()(
    DirichletBoundary<double> const &dirichletBoundary,
    Container<double, Alloc> &solution) const {
  LSS_ASSERT(init_.size() == solution.size(),
             "Initial and final solution must have the same size");
  // get the size of the vector:
  std::size_t const size = solution.size();
  // create prev pointer:
  double *prev = (double *)malloc(size * sizeof(double));
  std::copy(init_.begin(), init_.end(), prev);
  // create next pointer:
  double *next = (double *)malloc(size * sizeof(double));
  // launch the Euler loop:
  ExplicitEulerLoopDP loop{spaceStart_, terminalT_, deltas_,
                           coeffs_,     source_,    isSourceSet_};
  loop(prev, dirichletBoundary, size, next);
  // next point to the solution
  std::copy(next, next + size, solution.begin());
  free(prev);
  free(next);
}

template <template <typename, typename> typename Container, typename Alloc>
void ExplicitEulerHeatEquationScheme<float, Container, Alloc>::operator()(
    RobinBoundary<float> const &robinBoundary,
    Container<float, Alloc> &solution) const {
  LSS_ASSERT(init_.size() == solution.size(),
             "Initial and final solution must have the same size");
  // get the size of the vector:
  std::size_t const size = solution.size();
  // create prev pointer:
  float *prev = (float *)malloc(size * sizeof(float));
  std::copy(init_.begin(), init_.end(), prev);
  // create next pointer:
  float *next = (float *)malloc(size * sizeof(float));
  // launch the Euler loop:
  ExplicitEulerLoopSP loop{spaceStart_, terminalT_, deltas_,
                           coeffs_,     source_,    isSourceSet_};
  loop(prev, robinBoundary, size, next);
  // next point to the solution
  std::copy(next, next + size, solution.begin());
  free(prev);
  free(next);
}

template <template <typename, typename> typename Container, typename Alloc>
void ExplicitEulerHeatEquationScheme<double, Container, Alloc>::operator()(
    RobinBoundary<double> const &robinBoundary,
    Container<double, Alloc> &solution) const {
  LSS_ASSERT(init_.size() == solution.size(),
             "Initial and final solution must have the same size");
  // get the size of the vector:
  std::size_t const size = solution.size();
  // create prev pointer:
  double *prev = (double *)malloc(size * sizeof(double));
  std::copy(init_.begin(), init_.end(), prev);
  // create next pointer:
  double *next = (double *)malloc(size * sizeof(double));
  // launch the Euler loop:
  ExplicitEulerLoopDP loop{spaceStart_, terminalT_, deltas_,
                           coeffs_,     source_,    isSourceSet_};
  loop(prev, robinBoundary, size, next);
  // next point to the solution
  std::copy(next, next + size, solution.begin());
  free(prev);
  free(next);
}

}  // namespace lss_one_dim_heat_explicit_schemes_cuda

#endif  ///_LSS_ONE_DIM_HEAT_EXPLICIT_SCHEMES_CUDA
