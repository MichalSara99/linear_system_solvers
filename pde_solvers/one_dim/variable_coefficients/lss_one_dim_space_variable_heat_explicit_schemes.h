#pragma once
#if !defined(_LSS_ONE_DIM_SPACE_VARIABLE_HEAT_EXPLICIT_SCHEMES)
#define _LSS_ONE_DIM_SPACE_VARIABLE_HEAT_EXPLICIT_SCHEMES

#pragma warning(disable : 4244)

#include <thread>

#include "common/lss_enumerations.h"
#include "pde_solvers/one_dim/lss_one_dim_base_explicit_schemes.h"
#include "pde_solvers/one_dim/lss_one_dim_pde_utility.h"

namespace lss_one_dim_space_variable_heat_explicit_schemes {

using lss_enumerations::BoundaryConditionType;
using lss_enumerations::ExplicitPDESchemes;
using lss_enumerations::ImplicitPDESchemes;
using lss_one_dim_base_explicit_schemes::Explicit1DHeatSchemeBase;
using lss_one_dim_pde_utility::DirichletBoundary;
using lss_one_dim_pde_utility::Discretization;
using lss_one_dim_pde_utility::PDECoefficientHolderFun1Arg;

// Alias for Scheme coefficients (A(x),B(x),D(x))
template <typename T>
using SchemeCoefficientHolder =
    std::tuple<std::function<T(T)>, std::function<T(T)>, std::function<T(T)>>;
// Alias for PDE coefficients (a(x),b(x),c(x))
template <typename T>
using PDECoefficientHolder =
    std::tuple<std::function<T(T)>, std::function<T(T)>, std::function<T(T)>>;

// ============================================================================
// ======================= ExplicitHeatEulerScheme ============================
// ============================================================================

template <typename T>
class ExplicitHeatEulerScheme
    : public Explicit1DHeatSchemeBase<T, SchemeCoefficientHolder<T>> {
 private:
  PDECoefficientHolder<T> pdeCoeffs_;

 public:
  explicit ExplicitHeatEulerScheme() = delete;
  explicit ExplicitHeatEulerScheme(
      T spaceStart, T terminalTime, std::pair<T, T> const &deltas,
      PDECoefficientHolder<T> const &pdeCoeffs,
      SchemeCoefficientHolder<T> const &coeffs,
      std::vector<T> const &initialCondition,
      std::function<T(T, T)> const &source = nullptr, bool isSourceSet = false)
      : Explicit1DHeatSchemeBase<T, SchemeCoefficientHolder<T>>(
            spaceStart, terminalTime, deltas, coeffs, initialCondition, source,
            isSourceSet),
        pdeCoeffs_{pdeCoeffs} {}

  ~ExplicitHeatEulerScheme() {}

  ExplicitHeatEulerScheme(ExplicitHeatEulerScheme const &) = delete;
  ExplicitHeatEulerScheme(ExplicitHeatEulerScheme &&) = delete;
  ExplicitHeatEulerScheme &operator=(ExplicitHeatEulerScheme const &) = delete;
  ExplicitHeatEulerScheme &operator=(ExplicitHeatEulerScheme &&) = delete;

  // stability check:
  bool isStable() const override;

  // for Dirichlet BC
  void operator()(DirichletBoundary<T> const &dirichletBCPair,
                  std::vector<T> &solution) const override;
  // for Robin BC
  void operator()(std::pair<T, T> const &leftRobinBCPair,
                  std::pair<T, T> const &rightRobinBCPair,
                  std::vector<T> &solution) const override;
};

// ============================================================================
// ====================== ADEHeatBakaratClarkScheme ===========================
// ============================================================================

template <typename T>
class ADEHeatBakaratClarkScheme
    : public Explicit1DHeatSchemeBase<T, SchemeCoefficientHolder<T>> {
 public:
  explicit ADEHeatBakaratClarkScheme() = delete;
  explicit ADEHeatBakaratClarkScheme(
      T spaceStart, T terminalTime, std::pair<T, T> const &deltas,
      SchemeCoefficientHolder<T> const &coeffs,
      std::vector<T> const &initialCondition,
      std::function<T(T, T)> const &source = nullptr, bool isSourceSet = false)
      : Explicit1DHeatSchemeBase<T, SchemeCoefficientHolder<T>>(
            spaceStart, terminalTime, deltas, coeffs, initialCondition, source,
            isSourceSet) {}

  ~ADEHeatBakaratClarkScheme() {}

  ADEHeatBakaratClarkScheme(ADEHeatBakaratClarkScheme const &) = delete;
  ADEHeatBakaratClarkScheme(ADEHeatBakaratClarkScheme &&) = delete;
  ADEHeatBakaratClarkScheme &operator=(ADEHeatBakaratClarkScheme const &) =
      delete;
  ADEHeatBakaratClarkScheme &operator=(ADEHeatBakaratClarkScheme &&) = delete;

  // stability check:
  bool isStable() const override { return true; };

  // for Dirichlet BC
  void operator()(DirichletBoundary<T> const &dirichletBCPair,
                  std::vector<T> &solution) const override;
  // for Robin BC
  void operator()(std::pair<T, T> const &leftRobinBCPair,
                  std::pair<T, T> const &rightRobinBCPair,
                  std::vector<T> &solution) const override;
};

// ============================================================================
// ======================= ADEHeatSaulyevScheme ===============================
// ============================================================================

template <typename T>
class ADEHeatSaulyevScheme
    : public Explicit1DHeatSchemeBase<T, SchemeCoefficientHolder<T>> {
 public:
  explicit ADEHeatSaulyevScheme() = delete;
  explicit ADEHeatSaulyevScheme(T spaceStart, T terminalTime,
                                std::pair<T, T> const &deltas,
                                SchemeCoefficientHolder<T> const &coeffs,
                                std::vector<T> const &initialCondition,
                                std::function<T(T, T)> const &source = nullptr,
                                bool isSourceSet = false)
      : Explicit1DHeatSchemeBase<T, SchemeCoefficientHolder<T>>(
            spaceStart, terminalTime, deltas, coeffs, initialCondition, source,
            isSourceSet) {}

  ~ADEHeatSaulyevScheme() {}

  ADEHeatSaulyevScheme(ADEHeatSaulyevScheme const &) = delete;
  ADEHeatSaulyevScheme(ADEHeatSaulyevScheme &&) = delete;
  ADEHeatSaulyevScheme &operator=(ADEHeatSaulyevScheme const &) = delete;
  ADEHeatSaulyevScheme &operator=(ADEHeatSaulyevScheme &&) = delete;

  // stability check:
  bool isStable() const override { return true; };

  // for Dirichlet BC
  void operator()(DirichletBoundary<T> const &dirichletBCPair,
                  std::vector<T> &solution) const override;
  // for Robin BC
  void operator()(std::pair<T, T> const &leftRobinBCPair,
                  std::pair<T, T> const &rightRobinBCPair,
                  std::vector<T> &solution) const override;
};

}  // namespace lss_one_dim_space_variable_heat_explicit_schemes

// ============================================================================
// ======================== IMPLEMENTATIONS ===================================

template <typename T>
bool lss_one_dim_space_variable_heat_explicit_schemes::ExplicitHeatEulerScheme<
    T>::isStable() const {
  auto const &a = std::get<0>(pdeCoeffs_);
  auto const &b = std::get<1>(pdeCoeffs_);
  auto const &c = std::get<2>(pdeCoeffs_);
  T const k = std::get<0>(deltas_);
  T const h = std::get<1>(deltas_);
  T const lambda = k / (h * h);
  T const gamma = k / h;

  const std::size_t spaceSize = initialCondition_.size();
  for (std::size_t i = 0; i < spaceSize; ++i) {
    if (c(i * h) > 0.0) return false;
    if ((2.0 * lambda * a(i * h) - k * c(i * h)) > 1.0) return false;
    if (((gamma * std::abs(b(i * h))) * (gamma * std::abs(b(i * h)))) >
        (2.0 * lambda * a(i * h)))
      return false;
  }
  return true;
}

template <typename T>
void lss_one_dim_space_variable_heat_explicit_schemes::ExplicitHeatEulerScheme<
    T>::operator()(DirichletBoundary<T> const &dirichletBCPair,
                   std::vector<T> &solution) const {
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");
  LSS_ASSERT(
      solution.size() == initialCondition_.size(),
      "Entered solution vector size differs from initialCondition vector.");
  LSS_ASSERT(isStable() == true, "This discretization is not stable.");
  // get delta time:
  T const k = std::get<0>(deltas_);
  // get delta space:
  T const h = std::get<1>(deltas_);
  // create first time point:
  T time = k;
  // get coefficients:
  auto const &A = std::get<0>(coeffs_);
  auto const &B = std::get<1>(coeffs_);
  auto const &D = std::get<2>(coeffs_);
  // previous solution:
  std::vector<T> prevSol = initialCondition_;
  // left space boundary:
  auto const &left = dirichletBCPair.first;
  // right space boundary:
  auto const &right = dirichletBCPair.second;
  // size of the space vector:
  std::size_t const spaceSize = solution.size();
  if (!isSourceSet_) {
    // loop for stepping in time:
    while (time <= terminalTime_) {
      solution[0] = left(time);
      solution[solution.size() - 1] = right(time);
      for (std::size_t t = 1; t < spaceSize - 1; ++t) {
        solution[t] = (1.0 - 2.0 * B(t * h)) * prevSol[t] +
                      D(t * h) * prevSol[t + 1] + A(t * h) * prevSol[t - 1];
      }
      prevSol = solution;
      time += k;
    }
  } else {
    // create a container to carry discretized source heat
    std::vector<T> sourceCurr(solution.size(), T{});
    discretizeInSpace(h, spaceStart_, 0.0, source_, sourceCurr);
    // loop for stepping in time:
    while (time <= terminalTime_) {
      solution[0] = left(time);
      solution[solution.size() - 1] = right(time);
      for (std::size_t t = 1; t < spaceSize - 1; ++t) {
        solution[t] = solution[t] =
            (1.0 - 2.0 * B(t * h)) * prevSol[t] + D(t * h) * prevSol[t + 1] +
            A(t * h) * prevSol[t - 1] + k * sourceCurr[t];
      }
      discretizeInSpace(h, spaceStart_, time, source_, sourceCurr);
      prevSol = solution;
      time += k;
    }
  }
}

template <typename T>
void lss_one_dim_space_variable_heat_explicit_schemes::ExplicitHeatEulerScheme<
    T>::operator()(std::pair<T, T> const &leftRobinBCPair,
                   std::pair<T, T> const &rightRobinBCPair,
                   std::vector<T> &solution) const {
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");
  LSS_ASSERT(
      solution.size() == initialCondition_.size(),
      "Entered solution vector size differs from initialCondition vector.");
  LSS_ASSERT(isStable() == true, "This discretization is not stable.");
  // get delta time:
  T const k = std::get<0>(deltas_);
  // get delta space:
  T const h = std::get<1>(deltas_);
  // create first time point:
  T time = k;
  // get coefficients:
  auto const &A = std::get<0>(coeffs_);
  auto const &B = std::get<1>(coeffs_);
  auto const &D = std::get<2>(coeffs_);
  // left space boundary:
  T const leftLin = leftRobinBCPair.first;
  T const leftConst = leftRobinBCPair.second;
  // right space boundary:
  T const rightLin_ = rightRobinBCPair.first;
  T const rightConst_ = rightRobinBCPair.second;
  // conversion of right hand boundaries:
  T const rightLin = 1.0 / rightLin_;
  T const rightConst = -1.0 * (rightConst_ / rightLin_);
  // previous solution:
  std::vector<T> prevSol = initialCondition_;
  // size of the space vector:
  std::size_t const spaceSize = solution.size();
  if (!isSourceSet_) {
    // loop for stepping in time:
    while (time <= terminalTime_) {
      solution[0] = (D(0) + (A(0) * leftLin)) * prevSol[1] +
                    (1.0 - 2.0 * B(0)) * prevSol[0] + A(0) * leftConst;
      solution[spaceSize - 1] =
          (A((spaceSize - 1) * h) + (D((spaceSize - 1) * h) * rightLin)) *
              prevSol[spaceSize - 2] +
          (1.0 - 2.0 * B((spaceSize - 1) * h)) * prevSol[spaceSize - 1] +
          D((spaceSize - 1) * h) * rightConst;
      for (std::size_t t = 1; t < spaceSize - 1; ++t) {
        solution[t] = (1.0 - 2.0 * B(t * h)) * prevSol[t] +
                      D(t * h) * prevSol[t + 1] + A(t * h) * prevSol[t - 1];
      }
      prevSol = solution;
      time += k;
    }
  } else {
    // create a container to carry discretized source heat
    std::vector<T> sourceCurr(solution.size(), T{});
    discretizeInSpace(h, spaceStart_, 0.0, source_, sourceCurr);
    // loop for stepping in time:
    while (time <= terminalTime_) {
      solution[0] = (D(0) + (A(0) * leftLin)) * prevSol[1] +
                    (1.0 - 2.0 * B(0)) * prevSol[0] + A(0) * leftConst +
                    k * sourceCurr[0];
      solution[spaceSize - 1] =
          (A((spaceSize - 1) * h) + (D((spaceSize - 1) * h) * rightLin)) *
              prevSol[spaceSize - 2] +
          (1.0 - 2.0 * B((spaceSize - 1) * h)) * prevSol[spaceSize - 1] +
          D((spaceSize - 1) * h) * rightConst + k * sourceCurr[spaceSize - 1];
      for (std::size_t t = 1; t < spaceSize - 1; ++t) {
        solution[t] = (1.0 - 2.0 * B(t * h)) * prevSol[t] +
                      D(t * h) * prevSol[t + 1] + A(t * h) * prevSol[t - 1] +
                      k * sourceCurr[t];
      }
      discretizeInSpace(h, spaceStart_, time, source_, sourceCurr);
      prevSol = solution;
      time += k;
    }
  }
}

template <typename T>
void lss_one_dim_space_variable_heat_explicit_schemes::
    ADEHeatBakaratClarkScheme<T>::operator()(
        DirichletBoundary<T> const &dirichletBCPair,
        std::vector<T> &solution) const {
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");
  LSS_ASSERT(
      solution.size() == initialCondition_.size(),
      "Entered solution vector size differs from initialCondition vector.");
  // get delta time:
  T const k = std::get<0>(deltas_);
  // get delta space:
  T const h = std::get<1>(deltas_);
  // create first time point:
  T time = k;
  // get coefficients:
  auto const &A = std::get<0>(coeffs_);
  auto const &B = std::get<1>(coeffs_);
  auto const &D = std::get<2>(coeffs_);
  // calculate scheme coefficients:
  auto const &a = [&](T x) { return (A(x) / (1.0 + B(x))); };
  auto const &b = [&](T x) { return ((1.0 - B(x)) / (1.0 + B(x))); };
  auto const &d = [&](T x) { return (D(x) / (1.0 + B(x))); };
  auto const &f = [&](T x) { return (k / (1.0 + B(x))); };
  // left space boundary:
  auto const &left = dirichletBCPair.first;
  // right space boundary:
  auto const &right = dirichletBCPair.second;
  // conmponents of the solution:
  std::vector<T> com1(initialCondition_);
  std::vector<T> com2(initialCondition_);
  // size of the space vector:
  std::size_t const spaceSize = solution.size();
  // create a container to carry discretized source heat
  std::vector<T> sourceCurr(spaceSize, T{});
  std::vector<T> sourceNext(spaceSize, T{});
  // create upsweep anonymous function:
  auto upSweep = [=](std::vector<T> &upComponent, std::vector<T> const &rhs,
                     T rhsCoeff) {
    for (std::size_t t = 1; t < spaceSize - 1; ++t) {
      upComponent[t] =
          b(t * h) * upComponent[t] + d(t * h) * upComponent[t + 1] +
          a(t * h) * upComponent[t - 1] + f(t * h) * rhsCoeff * rhs[t];
    }
  };
  // create downsweep anonymous function:
  auto downSweep = [=](std::vector<T> &downComponent, std::vector<T> const &rhs,
                       T rhsCoeff) {
    for (std::size_t t = spaceSize - 2; t >= 1; --t) {
      downComponent[t] =
          b(t * h) * downComponent[t] + d(t * h) * downComponent[t + 1] +
          a(t * h) * downComponent[t - 1] + f(t * h) * rhsCoeff * rhs[t];
    }
  };

  if (!isSourceSet_) {
    // loop for stepping in time:
    while (time <= terminalTime_) {
      com1[0] = com2[0] = left(time);
      com1[solution.size() - 1] = com2[solution.size() - 1] = right(time);
      std::thread upSweepTr(std::move(upSweep), std::ref(com1), sourceCurr,
                            0.0);
      std::thread downSweepTr(std::move(downSweep), std::ref(com2), sourceCurr,
                              0.0);
      upSweepTr.join();
      downSweepTr.join();
      for (std::size_t t = 0; t < spaceSize; ++t) {
        solution[t] = 0.5 * (com1[t] + com2[t]);
      }
      time += k;
    }
  } else {
    discretizeInSpace(h, spaceStart_, 0.0, source_, sourceCurr);
    discretizeInSpace(h, spaceStart_, time, source_, sourceNext);
    // loop for stepping in time:
    while (time <= terminalTime_) {
      com1[0] = com2[0] = left(time);
      com1[solution.size() - 1] = com2[solution.size() - 1] = right(time);
      std::thread upSweepTr(std::move(upSweep), std::ref(com1), sourceNext,
                            1.0);
      std::thread downSweepTr(std::move(downSweep), std::ref(com2), sourceCurr,
                              1.0);
      upSweepTr.join();
      downSweepTr.join();
      for (std::size_t t = 0; t < spaceSize; ++t) {
        solution[t] = 0.5 * (com1[t] + com2[t]);
      }
      discretizeInSpace(h, spaceStart_, time, source_, sourceCurr);
      discretizeInSpace(h, spaceStart_, 2.0 * time, source_, sourceNext);
      time += k;
    }
  }
}

template <typename T>
void lss_one_dim_space_variable_heat_explicit_schemes::
    ADEHeatBakaratClarkScheme<T>::operator()(
        std::pair<T, T> const &leftRobinBCPair,
        std::pair<T, T> const &rightRobinBCPair,
        std::vector<T> &solution) const {
  throw new std::exception("Not available.");
}

template <typename T>
void lss_one_dim_space_variable_heat_explicit_schemes::ADEHeatSaulyevScheme<
    T>::operator()(DirichletPair<T> const &dirichletBCPair,
                   std::vector<T> &solution) const {
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");
  LSS_ASSERT(
      solution.size() == initialCondition_.size(),
      "Entered solution vector size differs from initialCondition vector.");
  // get delta time:
  T const k = std::get<0>(deltas_);
  // get delta space:
  T const h = std::get<1>(deltas_);
  // create first time point:
  T time = k;
  // get coefficients:
  auto const &A = std::get<0>(coeffs_);
  auto const &B = std::get<1>(coeffs_);
  auto const &D = std::get<2>(coeffs_);
  // calculate scheme coefficients:
  auto const &a = [&](T x) { return (A(x) / (1.0 + B(x))); };
  auto const &b = [&](T x) { return ((1.0 - B(x)) / (1.0 + B(x))); };
  auto const &d = [&](T x) { return (D(x) / (1.0 + B(x))); };
  auto const &f = [&](T x) { return (k / (1.0 + B(x))); };
  // left space boundary:
  auto const &left = dirichletBCPair.first;
  // right space boundary:
  auto const &right = dirichletBCPair.second;
  // get the initial condition :
  solution = initialCondition_;
  // size of the space vector:
  std::size_t const spaceSize = solution.size();
  // create a container to carry discretized source heat
  std::vector<T> sourceCurr(spaceSize, T{});
  std::vector<T> sourceNext(spaceSize, T{});
  // create upsweep anonymous function:
  auto upSweep = [=](std::vector<T> &upComponent, std::vector<T> const &rhs,
                     T rhsCoeff) {
    for (std::size_t t = 1; t < spaceSize - 1; ++t) {
      upComponent[t] =
          b(t * h) * upComponent[t] + d(t * h) * upComponent[t + 1] +
          a(t * h) * upComponent[t - 1] + f(t * h) * rhsCoeff * rhs[t];
    }
  };
  // create downsweep anonymous function:
  auto downSweep = [=](std::vector<T> &downComponent, std::vector<T> const &rhs,
                       T rhsCoeff) {
    for (std::size_t t = spaceSize - 2; t >= 1; --t) {
      downComponent[t] =
          b(t * h) * downComponent[t] + d(t * h) * downComponent[t + 1] +
          a(t * h) * downComponent[t - 1] + f(t * h) * rhsCoeff * rhs[t];
    }
  };

  if (!isSourceSet_) {
    // loop for stepping in time:
    std::size_t t = 1;
    while (time <= terminalTime_) {
      solution[0] = left(time);
      solution[solution.size() - 1] = right(time);
      if (t % 2 == 0)
        downSweep(solution, sourceCurr, 0.0);
      else
        upSweep(solution, sourceCurr, 0.0);
      ++t;
      time += k;
    }
  } else {
    discretizeInSpace(h, spaceStart_, 0.0, source_, sourceCurr);
    discretizeInSpace(h, spaceStart_, time, source_, sourceNext);
    // loop for stepping in time:
    std::size_t t = 1;
    while (time <= terminalTime_) {
      solution[0] = left(time);
      solution[solution.size() - 1] = right(time);
      if (t % 2 == 0)
        downSweep(solution, sourceCurr, 1.0);
      else
        upSweep(solution, sourceNext, 1.0);
      ++t;
      discretizeInSpace(h, spaceStart_, time, source_, sourceCurr);
      discretizeInSpace(h, spaceStart_, 2.0 * time, source_, sourceNext);
      time += k;
    }
  }
}

template <typename T>
void lss_one_dim_space_variable_heat_explicit_schemes::ADEHeatSaulyevScheme<
    T>::operator()(std::pair<T, T> const &leftRobinBCPair,
                   std::pair<T, T> const &rightRobinBCPair,
                   std::vector<T> &solution) const {
  throw new std::exception("Not available.");
}

#endif  //_LSS_ONE_DIM_SPACE_VARIABLE_HEAT_EXPLICIT_SCHEMES
