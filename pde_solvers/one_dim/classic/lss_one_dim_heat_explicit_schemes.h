#pragma once
#if !defined(_LSS_ONE_DIM_HEAT_EXPLICIT_SCHEMES)
#define _LSS_ONE_DIM_HEAT_EXPLICIT_SCHEMES

#pragma warning(disable : 4244)

#include <thread>

#include "common/lss_types.h"
#include "pde_solvers/one_dim/lss_one_dim_base_explicit_schemes.h"
#include "pde_solvers/one_dim/lss_one_dim_pde_utility.h"

namespace lss_one_dim_heat_explicit_schemes {

using lss_one_dim_base_explicit_schemes::Explicit1DHeatSchemeBase;
using lss_one_dim_pde_utility::Discretization;
using lss_types::BoundaryConditionType;
using lss_types::ExplicitPDESchemes;
using lss_types::ImplicitPDESchemes;

// Alias for Scheme coefficients (A(x),B(x),D(x))
template <typename T>
using SchemeCoefficientHolder = std::tuple<T, T, T>;

// ============================================================================
// ================= ExplicitHeatEulerScheme ==================================
// ============================================================================

template <typename T>
class ExplicitHeatEulerScheme
    : public Explicit1DHeatSchemeBase<T, SchemeCoefficientHolder<T>> {
 public:
  explicit ExplicitHeatEulerScheme() = delete;
  explicit ExplicitHeatEulerScheme(
      T spaceStart, T terminalTime, std::pair<T, T> const &deltas,
      SchemeCoefficientHolder<T> const &coeffs,
      std::vector<T> const &initialCondition,
      std::function<T(T, T)> const &source = nullptr, bool isSourceSet = false)
      : Explicit1DHeatSchemeBase<T, SchemeCoefficientHolder<T>>(
            spaceStart, terminalTime, deltas, coeffs, initialCondition, source,
            isSourceSet) {}

  ~ExplicitHeatEulerScheme() {}

  ExplicitHeatEulerScheme(ExplicitHeatEulerScheme const &) = delete;
  ExplicitHeatEulerScheme(ExplicitHeatEulerScheme &&) = delete;
  ExplicitHeatEulerScheme &operator=(ExplicitHeatEulerScheme const &) = delete;
  ExplicitHeatEulerScheme &operator=(ExplicitHeatEulerScheme &&) = delete;

  // stability check:
  bool isStable() const override;

  // for Dirichlet BC
  void operator()(std::pair<T, T> const &dirichletBCPair,
                  std::vector<T> &solution) const override;
  // for Robin BC
  void operator()(std::pair<T, T> const &leftRobinBCPair,
                  std::pair<T, T> const &rightRobinBCPair,
                  std::vector<T> &solution) const override;
};

// ============================================================================
// ====================== ADEHeatBakaratClarkScheme  ==========================
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
  void operator()(std::pair<T, T> const &dirichletBCPair,
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
  void operator()(std::pair<T, T> const &dirichletBCPair,
                  std::vector<T> &solution) const override;
  // for Robin BC
  void operator()(std::pair<T, T> const &leftRobinBCPair,
                  std::pair<T, T> const &rightRobinBCPair,
                  std::vector<T> &solution) const override;
};

}  // namespace lss_one_dim_heat_explicit_schemes

// ============================================================================
// =========================== IMPLEMENTATIONS ================================

template <typename T>
bool lss_one_dim_heat_explicit_schemes::ExplicitHeatEulerScheme<T>::isStable()
    const {
  T const A = std::get<0>(coeffs_);
  T const B = std::get<1>(coeffs_);
  T const k = std::get<0>(deltas_);
  T const h = std::get<1>(deltas_);

  return (((2.0 * A * k / (h * h)) <= 1.0) && (B * (k / h) <= 1.0));
}

template <typename T>
void lss_one_dim_heat_explicit_schemes::ExplicitHeatEulerScheme<T>::operator()(
    std::pair<T, T> const &dirichletBCPair, std::vector<T> &solution) const {
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
  T const A = std::get<0>(coeffs_);
  T const B = std::get<1>(coeffs_);
  T const C = std::get<2>(coeffs_);
  // calculate scheme coefficients:
  T const lambda = (A * k) / (h * h);
  T const gamma = (B * k) / (2.0 * h);
  T const delta = C * k;
  // set up coefficients:
  T const a = 1.0 - (2.0 * lambda - delta);
  T const b = lambda + gamma;
  T const c = lambda - gamma;
  // previous solution:
  std::vector<T> prevSol = initialCondition_;
  // left space boundary:
  T const left = dirichletBCPair.first;
  // right space boundary:
  T const right = dirichletBCPair.second;
  // size of the space vector:
  std::size_t const spaceSize = solution.size();
  if (!isSourceSet_) {
    // loop for stepping in time:
    while (time <= terminalTime_) {
      solution[0] = left;
      solution[solution.size() - 1] = right;
      for (std::size_t t = 1; t < spaceSize - 1; ++t) {
        solution[t] = a * prevSol[t] + b * prevSol[t + 1] + c * prevSol[t - 1];
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
      solution[0] = left;
      solution[solution.size() - 1] = right;
      for (std::size_t t = 1; t < spaceSize - 1; ++t) {
        solution[t] = a * prevSol[t] + b * prevSol[t + 1] + c * prevSol[t - 1] +
                      k * sourceCurr[t];
      }
      discretizeInSpace(h, spaceStart_, time, source_, sourceCurr);
      prevSol = solution;
      time += k;
    }
  }
}

template <typename T>
void lss_one_dim_heat_explicit_schemes::ExplicitHeatEulerScheme<T>::operator()(
    std::pair<T, T> const &leftRobinBCPair,
    std::pair<T, T> const &rightRobinBCPair, std::vector<T> &solution) const {
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
  T const A = std::get<0>(coeffs_);
  T const B = std::get<1>(coeffs_);
  T const C = std::get<2>(coeffs_);
  // calculate scheme coefficients:
  T const lambda = (A * k) / (h * h);
  T const gamma = (B * k) / (2.0 * h);
  T const delta = C * k;
  // left space boundary:
  T const leftLin = leftRobinBCPair.first;
  T const leftConst = leftRobinBCPair.second;
  // right space boundary:
  T const rightLin_ = rightRobinBCPair.first;
  T const rightConst_ = rightRobinBCPair.second;
  // conversion of right hand boundaries:
  T const rightLin = 1.0 / rightLin_;
  T const rightConst = -1.0 * (rightConst_ / rightLin_);
  // set up coefficients:
  T const a = 1.0 - (2.0 * lambda - delta);
  T const b = lambda + gamma;
  T const c = lambda - gamma;
  // previous solution:
  std::vector<T> prevSol = initialCondition_;
  // size of the space vector:
  std::size_t const spaceSize = solution.size();
  if (!isSourceSet_) {
    // loop for stepping in time:
    while (time <= terminalTime_) {
      solution[0] =
          (b + (c * leftLin)) * prevSol[1] + a * prevSol[0] + c * leftConst;
      solution[solution.size() - 1] =
          (c + (b * rightLin)) * prevSol[solution.size() - 2] +
          a * prevSol[solution.size() - 1] + b * rightConst;
      for (std::size_t t = 1; t < spaceSize - 1; ++t) {
        solution[t] = a * prevSol[t] + b * prevSol[t + 1] + c * prevSol[t - 1];
      }
      prevSol = solution;
      time += k;
    }
  } else {
    // create a container to carry discretized source heat
    std::vector<T> sourceCurr(solution.size(), T{});
    discretizeInSpace(h, spaceStart_, 0.0, source_, sourceCurr);
    // loop for stepping in time:
    // loop for stepping in time:
    while (time <= terminalTime_) {
      solution[0] =
          (b + (c * leftLin)) * prevSol[1] + a * prevSol[0] + c * leftConst;
      solution[solution.size() - 1] =
          (c + (b * rightLin)) * prevSol[solution.size() - 2] +
          a * prevSol[solution.size() - 1] + b * rightConst;
      for (std::size_t t = 1; t < spaceSize - 1; ++t) {
        solution[t] = a * prevSol[t] + b * prevSol[t + 1] + c * prevSol[t - 1] +
                      k * sourceCurr[t];
      }
      discretizeInSpace(h, spaceStart_, time, source_, sourceCurr);
      prevSol = solution;
      time += k;
    }
  }
}

template <typename T>
void lss_one_dim_heat_explicit_schemes::ADEHeatBakaratClarkScheme<
    T>::operator()(std::pair<T, T> const &dirichletBCPair,
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
  T const A = std::get<0>(coeffs_);
  T const B = std::get<1>(coeffs_);
  T const C = std::get<2>(coeffs_);
  // calculate scheme coefficients:
  T const lambda = (A * k) / (h * h);
  T const gamma = (B * k) / (2.0 * h);
  T const delta = C * k / 2.0;
  // set up coefficients:
  T const divisor = 1.0 + lambda - delta;
  T const a = (1.0 - lambda + delta) / divisor;
  T const b = (lambda + gamma) / divisor;
  T const c = (lambda - gamma) / divisor;
  T const d = k / divisor;
  // left space boundary:
  T const left = dirichletBCPair.first;
  // right space boundary:
  T const right = dirichletBCPair.second;
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
      upComponent[t] = a * upComponent[t] + b * upComponent[t + 1] +
                       c * upComponent[t - 1] + d * rhsCoeff * rhs[t];
    }
  };
  // create downsweep anonymous function:
  auto downSweep = [=](std::vector<T> &downComponent, std::vector<T> const &rhs,
                       T rhsCoeff) {
    for (std::size_t t = spaceSize - 2; t >= 1; --t) {
      downComponent[t] = a * downComponent[t] + b * downComponent[t + 1] +
                         c * downComponent[t - 1] + d * rhsCoeff * rhs[t];
    }
  };

  if (!isSourceSet_) {
    // loop for stepping in time:
    while (time <= terminalTime_) {
      com1[0] = com2[0] = left;
      com1[solution.size() - 1] = com2[solution.size() - 1] = right;
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
      com1[0] = com2[0] = left;
      com1[solution.size() - 1] = com2[solution.size() - 1] = right;
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
void lss_one_dim_heat_explicit_schemes::ADEHeatBakaratClarkScheme<
    T>::operator()(std::pair<T, T> const &leftRobinBCPair,
                   std::pair<T, T> const &rightRobinBCPair,
                   std::vector<T> &solution) const {
  throw new std::exception("Not available.");
}

template <typename T>
void lss_one_dim_heat_explicit_schemes::ADEHeatSaulyevScheme<T>::operator()(
    std::pair<T, T> const &dirichletBCPair, std::vector<T> &solution) const {
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
  T const A = std::get<0>(coeffs_);
  T const B = std::get<1>(coeffs_);
  T const C = std::get<2>(coeffs_);
  // calculate scheme coefficients:
  T const lambda = (A * k) / (h * h);
  T const gamma = (B * k) / (2.0 * h);
  T const delta = C * k / 2.0;
  // set up coefficients:
  T const divisor = 1.0 + lambda - delta;
  T const a = (1.0 - lambda + delta) / divisor;
  T const b = (lambda + gamma) / divisor;
  T const c = (lambda - gamma) / divisor;
  T const d = k / divisor;
  // left space boundary:
  T const left = dirichletBCPair.first;
  // right space boundary:
  T const right = dirichletBCPair.second;
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
      upComponent[t] = a * upComponent[t] + b * upComponent[t + 1] +
                       c * upComponent[t - 1] + d * rhsCoeff * rhs[t];
    }
  };
  // create downsweep anonymous function:
  auto downSweep = [=](std::vector<T> &downComponent, std::vector<T> const &rhs,
                       T rhsCoeff) {
    for (std::size_t t = spaceSize - 2; t >= 1; --t) {
      downComponent[t] = a * downComponent[t] + b * downComponent[t + 1] +
                         c * downComponent[t - 1] + d * rhsCoeff * rhs[t];
    }
  };

  if (!isSourceSet_) {
    // loop for stepping in time:
    std::size_t t = 1;
    while (time <= terminalTime_) {
      solution[0] = left;
      solution[solution.size() - 1] = right;
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
      solution[0] = left;
      solution[solution.size() - 1] = right;
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
void lss_one_dim_heat_explicit_schemes::ADEHeatSaulyevScheme<T>::operator()(
    std::pair<T, T> const &leftRobinBCPair,
    std::pair<T, T> const &rightRobinBCPair, std::vector<T> &solution) const {
  throw new std::exception("Not available.");
}

#endif  ///_LSS_ONE_DIM_HEAT_EXPLICIT_SCHEMES
