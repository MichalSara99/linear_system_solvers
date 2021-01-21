#pragma once
#if !defined(_LSS_ONE_DIM_SPACE_VARIABLE_GENERAL_HEAT_EQUATION_SOLVERS)
#define _LSS_ONE_DIM_SPACE_VARIABLE_GENERAL_HEAT_EQUATION_SOLVERS

#include <functional>

#include "common/lss_enumerations.h"
#include "common/lss_macros.h"
#include "common/lss_utility.h"
#include "lss_one_dim_space_variable_heat_explicit_schemes.h"
#include "lss_one_dim_space_variable_heat_implicit_schemes.h"
#include "pde_solvers/one_dim/lss_one_dim_pde_utility.h"

namespace lss_one_dim_space_variable_general_heat_equation_solvers {

using lss_enumerations::BoundaryConditionType;
using lss_enumerations::ExplicitPDESchemes;
using lss_enumerations::ImplicitPDESchemes;
using lss_one_dim_pde_utility::DirichletBoundary;
using lss_one_dim_pde_utility::Discretization;
using lss_one_dim_pde_utility::HeatData;
using lss_one_dim_pde_utility::PDECoefficientHolderFun1Arg;
using lss_one_dim_pde_utility::RobinBoundary;
using lss_one_dim_space_variable_heat_explicit_schemes::
    ADEHeatBakaratClarkScheme;
using lss_one_dim_space_variable_heat_explicit_schemes::ADEHeatSaulyevScheme;
using lss_one_dim_space_variable_heat_explicit_schemes::ExplicitHeatEulerScheme;
using lss_one_dim_space_variable_heat_implicit_schemes::
    ImplicitSpaceVariableHeatEquationSchemes;
using lss_utility::Range;
using lss_utility::uptr_t;

namespace implicit_solvers {

// ============================================================================
// ======= Implicit1DSpaceVariableGeneralHeatEquation General Template ========
// ============================================================================

template <typename T, BoundaryConditionType BType,
          template <typename, BoundaryConditionType,
                    template <typename, typename> typename Cont, typename>
          typename FDMSolver,
          template <typename, typename> typename Container, typename Alloc>
class Implicit1DSpaceVariableGeneralHeatEquation {};

// ============================================================================
// Implicit1DSpaceVariableGeneralHeatEquation Dirichlet Specialisation Template
// ============================================================================
//
//	u_t = a(x)*u_xx + b(x)*u_x + c(x)*u + F(x,t), t > 0, x_1 < x < x_2
//
//	with initial condition
//  u(x,0) = f(x)
//
//	and Dirichlet boundaries:
//  u(x_1,t) = A(t)
//	u(x_2,t) = B(t)
//
// ============================================================================

template <typename T,
          template <typename, BoundaryConditionType,
                    template <typename, typename> typename Cont, typename>
          typename FDMSolver,
          template <typename, typename> typename Container, typename Alloc>
class Implicit1DSpaceVariableGeneralHeatEquation<
    T, BoundaryConditionType::Dirichlet, FDMSolver, Container, Alloc>
    : public Discretization<T, Container, Alloc> {
 private:
  typedef FDMSolver<T, BoundaryConditionType::Dirichlet, Container, Alloc>
      fdm_solver_t;
  typedef HeatData<T> heat_data_t;

  uptr_t<fdm_solver_t> solverPtr_;         // finite-difference solver
  uptr_t<heat_data_t> dataPtr_;            // one-dim heat data
  DirichletBoundary<T> boundary_;          // boundaries
  PDECoefficientHolderFun1Arg<T> coeffs_;  // coefficients of PDE

 public:
  typedef T value_type;
  explicit Implicit1DSpaceVariableGeneralHeatEquation() = delete;
  explicit Implicit1DSpaceVariableGeneralHeatEquation(
      Range<T> const &spaceRange, T terminalTime,
      std::size_t const &spaceDiscretization,
      std::size_t const &timeDiscretization)
      : solverPtr_{std::make_unique<fdm_solver_t>(spaceDiscretization + 1)},
        dataPtr_{std::make_unique<heat_data_t>(
            spaceRange, Range<T>(T{}, terminalTime), spaceDiscretization,
            timeDiscretization, nullptr, nullptr, nullptr, false)} {}

  ~Implicit1DSpaceVariableGeneralHeatEquation() {}

  Implicit1DSpaceVariableGeneralHeatEquation(
      Implicit1DSpaceVariableGeneralHeatEquation const &) = delete;
  Implicit1DSpaceVariableGeneralHeatEquation(
      Implicit1DSpaceVariableGeneralHeatEquation &&) = delete;
  Implicit1DSpaceVariableGeneralHeatEquation &operator=(
      Implicit1DSpaceVariableGeneralHeatEquation const &) = delete;
  Implicit1DSpaceVariableGeneralHeatEquation &operator=(
      Implicit1DSpaceVariableGeneralHeatEquation &&) = delete;

  inline T spaceStep() const {
    return ((dataPtr_->spaceRange.spread()) /
            static_cast<T>(dataPtr_->spaceDivision));
  }
  inline T timeStep() const {
    return ((dataPtr_->timeRange.upper()) /
            static_cast<T>(dataPtr_->timeDivision));
  }

  inline void setBoundaryCondition(DirichletBoundary<T> const &boundaryPair) {
    boundary_ = boundaryPair;
  }
  inline void setInitialCondition(std::function<T(T)> const &initialCondition) {
    dataPtr_->initialCondition = initialCondition;
  }
  inline void setHeatSource(std::function<T(T, T)> const &heatSource) {
    dataPtr_->isSourceFunctionSet = true;
    dataPtr_->sourceFunction = heatSource;
  }
  inline void set2OrderCoefficient(std::function<T(T)> const &a) {
    std::get<0>(coeffs_) = a;
  }
  inline void set1OrderCoefficient(std::function<T(T)> const &b) {
    std::get<1>(coeffs_) = b;
  }
  inline void set0OrderCoefficient(std::function<T(T)> const &c) {
    std::get<2>(coeffs_) = c;
  }

  void solve(Container<T, Alloc> &solution,
             ImplicitPDESchemes scheme = ImplicitPDESchemes::CrankNicolson);
};

// ============================================================================
// = Implicit1DSpaceVariableGeneralHeatEquation Robin Specialisation Template =
// ============================================================================
//
//	u_t = a(x)*u_xx + b(x)*u_x + c(x)*u + F(x,t), t > 0, x_1 < x < x_2
//
//	with initial condition
//  u(x,0) = f(x)
//
//	and Robin boundaries:
//  d_1*u_x(x_1,t) + f_1*u(x_1,t) = A
//	d_2*u_x(x_2,t) + f_2*u(x_2,t) = B
//
//	It is assumed the Robin boundaries are discretised in following way:
//	d_1*(U_1 - U_0)/h + f_1*(U_0 + U_1)/2 = A
//	d_2*(U_N - U_N-1)/h + f_2*(U_N-1 + U_N)/2 = B
//
//	And therefore can be rewritten in form:
//
//	U_0 = ((2*d_1 + f_1*h)/(2*d_1 - f_1*h)) * U_1 +
//			(2*h/(f_1*h - 2*d_1))*A
//	U_N-1 = ((2*d_2 + f_2*h)/(2*d_2 - f_2*h)) * U_N +
//			(2*h/(f_2*h -2*d_2))*B
//
//	or
//
//	U_0 = alpha * U_1 + phi,
//	U_N-1 = beta * U_N + psi,
//
// ============================================================================

template <typename T,
          template <typename, BoundaryConditionType,
                    template <typename, typename> typename Cont, typename>
          typename FDMSolver,
          template <typename, typename> typename Container, typename Alloc>
class Implicit1DSpaceVariableGeneralHeatEquation<
    T, BoundaryConditionType::Robin, FDMSolver, Container, Alloc>
    : public Discretization<T, Container, Alloc> {
 private:
  typedef FDMSolver<T, BoundaryConditionType::Robin, Container, Alloc>
      fdm_solver_t;
  typedef HeatData<T> heat_data_t;

  uptr_t<fdm_solver_t> solverPtr_;         // finite-difference solver
  uptr_t<heat_data_t> dataPtr_;            // one-dim heat data
  RobinBoundary<T> boundary_;              // Robin boundary
  PDECoefficientHolderFun1Arg<T> coeffs_;  // coefficients of PDE

 public:
  typedef T value_type;
  explicit Implicit1DSpaceVariableGeneralHeatEquation() = delete;
  explicit Implicit1DSpaceVariableGeneralHeatEquation(
      Range<T> const &spaceRange, T terminalTime,
      std::size_t const &spaceDiscretization,
      std::size_t const &timeDiscretization)
      : solverPtr_{std::make_unique<fdm_solver_t>(spaceDiscretization + 1)},
        dataPtr_{std::make_unique<heat_data_t>(
            spaceRange, Range<T>(T{}, terminalTime), spaceDiscretization,
            timeDiscretization, nullptr, nullptr, nullptr, false)} {}

  ~Implicit1DSpaceVariableGeneralHeatEquation() {}

  Implicit1DSpaceVariableGeneralHeatEquation(
      Implicit1DSpaceVariableGeneralHeatEquation const &) = delete;
  Implicit1DSpaceVariableGeneralHeatEquation(
      Implicit1DSpaceVariableGeneralHeatEquation &&) = delete;
  Implicit1DSpaceVariableGeneralHeatEquation &operator=(
      Implicit1DSpaceVariableGeneralHeatEquation const &) = delete;
  Implicit1DSpaceVariableGeneralHeatEquation &operator=(
      Implicit1DSpaceVariableGeneralHeatEquation &&) = delete;

  inline T spaceStep() const {
    return ((dataPtr_->spaceRange.spread()) /
            static_cast<T>(dataPtr_->spaceDivision));
  }
  inline T timeStep() const {
    return ((dataPtr_->timeRange.upper()) /
            static_cast<T>(dataPtr_->timeDivision));
  }

  inline void setBoundaryCondition(RobinBoundary<T> const &boundary) {
    boundary_ = boundary;
    solverPtr_->setBoundaryCondition(boundary_.left, boundary_.right);
  }

  inline void setInitialCondition(std::function<T(T)> const &initialCondition) {
    dataPtr_->initialCondition = initialCondition;
  }
  inline void setHeatSource(std::function<T(T, T)> const &heatSource) {
    dataPtr_->isSourceFunctionSet = true;
    dataPtr_->sourceFunction = heatSource;
  }
  inline void set2OrderCoefficient(std::function<T(T)> const &a) {
    std::get<0>(coeffs_) = a;
  }
  inline void set1OrderCoefficient(std::function<T(T)> const &b) {
    std::get<1>(coeffs_) = b;
  }
  inline void set0OrderCoefficient(std::function<T(T)> const &c) {
    std::get<2>(coeffs_) = c;
  }

  void solve(Container<T, Alloc> &solution,
             ImplicitPDESchemes scheme = ImplicitPDESchemes::CrankNicolson);
};

}  // namespace implicit_solvers

namespace explicit_solvers {

// ============================================================================
// ====== Explicit1DSpaceVariableGeneralHeatEquation General Template =========
// ============================================================================

template <typename T, BoundaryConditionType BType,
          template <typename, typename> typename Container, typename Alloc>
class Explicit1DSpaceVariableGeneralHeatEquation {};

// ============================================================================
// Explicit1DSpaceVariableGeneralHeatEquation Dirichlet Specialisation Template
// ============================================================================
//
//	u_t = a(x)*u_xx + b(x)*u_x + c(x)*u + F(x,t), t > 0, x_1 < x < x_2
//
//	with initial condition
//  u(x,0) = f(x)
//
//	and Dirichlet boundaries:
//  u(x_1,t) = A(t)
//	u(x_2,t) = B(t)
//
// ============================================================================

template <typename T, template <typename, typename> typename Container,
          typename Alloc>
class Explicit1DSpaceVariableGeneralHeatEquation<
    T, BoundaryConditionType::Dirichlet, Container, Alloc>
    : public Discretization<T, Container, Alloc> {
 private:
  typedef HeatData<T> heat_data_t;

  uptr_t<heat_data_t> dataPtr_;            // one-dim heat data
  DirichletBoundary<T> boundary_;          // boundaries
  PDECoefficientHolderFun1Arg<T> coeffs_;  // coefficients of PDE

 public:
  typedef T value_type;
  explicit Explicit1DSpaceVariableGeneralHeatEquation() = delete;
  explicit Explicit1DSpaceVariableGeneralHeatEquation(
      Range<T> const &spaceRange, T terminalTime,
      std::size_t const &spaceDiscretization,
      std::size_t const &timeDiscretization)
      : dataPtr_{std::make_unique<heat_data_t>(
            spaceRange, Range<T>(T{}, terminalTime), spaceDiscretization,
            timeDiscretization, nullptr, nullptr, nullptr, false)} {}

  ~Explicit1DSpaceVariableGeneralHeatEquation() {}

  Explicit1DSpaceVariableGeneralHeatEquation(
      Explicit1DSpaceVariableGeneralHeatEquation const &) = delete;
  Explicit1DSpaceVariableGeneralHeatEquation(
      Explicit1DSpaceVariableGeneralHeatEquation &&) = delete;
  Explicit1DSpaceVariableGeneralHeatEquation &operator=(
      Explicit1DSpaceVariableGeneralHeatEquation const &) = delete;
  Explicit1DSpaceVariableGeneralHeatEquation &operator=(
      Explicit1DSpaceVariableGeneralHeatEquation &&) = delete;

  inline T spaceStep() const {
    return ((dataPtr_->spaceRange.spread()) /
            static_cast<T>(dataPtr_->spaceDivision));
  }
  inline T timeStep() const {
    return ((dataPtr_->timeRange.upper()) /
            static_cast<T>(dataPtr_->timeDivision));
  }

  inline void setBoundaryCondition(DirichletBoundary<T> const &boundaryPair) {
    boundary_ = boundaryPair;
  }
  inline void setInitialCondition(std::function<T(T)> const &initialCondition) {
    dataPtr_->initialCondition = initialCondition;
  }
  inline void setHeatSource(std::function<T(T, T)> const &heatSource) {
    dataPtr_->isSourceFunctionSet = true;
    dataPtr_->sourceFunction = heatSource;
  }
  inline void set2OrderCoefficient(std::function<T(T)> const &a) {
    std::get<0>(coeffs_) = a;
  }
  inline void set1OrderCoefficient(std::function<T(T)> const &b) {
    std::get<1>(coeffs_) = b;
  }
  inline void set0OrderCoefficient(std::function<T(T)> const &c) {
    std::get<2>(coeffs_) = c;
  }

  void solve(Container<T, Alloc> &solution,
             ExplicitPDESchemes scheme = ExplicitPDESchemes::ADEBarakatClark);
};

// ============================================================================
// = Explicit1DSpaceVariableGeneralHeatEquation Robin Specialisation Template =
// ============================================================================
//
//	u_t = a(x)*u_xx + b(x)*u_x + c(x)*u + F(x,t), t > 0, x_1 < x < x_2
//
//	with initial condition
//  u(x,0) = f(x)
//
//	and Robin boundaries:
//  d_1*u_x(x_1,t) + f_1*u(x_1,t) = A
//	d_2*u_x(x_2,t) + f_2*u(x_2,t) = B
//
//	It is assumed the Robin boundaries are discretised in following way:
//	d_1*(U_1 - U_0)/h + f_1*(U_0 + U_1)/2 = A
//	d_2*(U_N - U_N-1)/h + f_2*(U_N-1 + U_N)/2 = B
//
//	And therefore can be rewritten in form:
//
//	U_0 = ((2*d_1 + f_1*h)/(2*d_1 - f_1*h)) * U_1 +
//			(2*h/(f_1*h - 2*d_1))*A
//	U_N-1 = ((2*d_2 + f_2*h)/(2*d_2 - f_2*h)) * U_N +
//			(2*h/(f_2*h -2*d_2))*B
//
//	or
//
//	U_0 = alpha * U_1 + phi,
//	U_N-1 = beta * U_N + psi,
//
// ============================================================================

template <typename T, template <typename, typename> typename Container,
          typename Alloc>
class Explicit1DSpaceVariableGeneralHeatEquation<
    T, BoundaryConditionType::Robin, Container, Alloc>
    : public Discretization<T, Container, Alloc> {
 private:
  typedef HeatData<T> heat_data_t;

  uptr_t<heat_data_t> dataPtr_;            // one-dim heat data
  RobinBoundary<T> boundary_;              // Robin boundary
  PDECoefficientHolderFun1Arg<T> coeffs_;  // coefficients of PDE

 public:
  typedef T value_type;
  explicit Explicit1DSpaceVariableGeneralHeatEquation() = delete;
  explicit Explicit1DSpaceVariableGeneralHeatEquation(
      Range<T> const &spaceRange, T terminalTime,
      std::size_t const &spaceDiscretization,
      std::size_t const &timeDiscretization)
      : dataPtr_{std::make_unique<heat_data_t>(
            spaceRange, Range<T>(T{}, terminalTime), spaceDiscretization,
            timeDiscretization, nullptr, nullptr, nullptr, false)} {}

  ~Explicit1DSpaceVariableGeneralHeatEquation() {}

  Explicit1DSpaceVariableGeneralHeatEquation(
      Explicit1DSpaceVariableGeneralHeatEquation const &) = delete;
  Explicit1DSpaceVariableGeneralHeatEquation(
      Explicit1DSpaceVariableGeneralHeatEquation &&) = delete;
  Explicit1DSpaceVariableGeneralHeatEquation &operator=(
      Explicit1DSpaceVariableGeneralHeatEquation const &) = delete;
  Explicit1DSpaceVariableGeneralHeatEquation &operator=(
      Explicit1DSpaceVariableGeneralHeatEquation &&) = delete;

  inline T spaceStep() const {
    return ((dataPtr_->spaceRange.spread()) /
            static_cast<T>(dataPtr_->spaceDivision));
  }
  inline T timeStep() const {
    return ((dataPtr_->timeRange.upper()) /
            static_cast<T>(dataPtr_->timeDivision));
  }

  inline void setBoundaryCondition(RobinBoundary<T> const &boundary) {
    boundary_ = boundary;
  }
  inline void setInitialCondition(std::function<T(T)> const &initialCondition) {
    dataPtr_->initialCondition = initialCondition;
  }
  inline void setHeatSource(std::function<T(T, T)> const &heatSource) {
    dataPtr_->isSourceFunctionSet = true;
    dataPtr_->sourceFunction = heatSource;
  }
  inline void set2OrderCoefficient(std::function<T(T)> const &a) {
    std::get<0>(coeffs_) = a;
  }
  inline void set1OrderCoefficient(std::function<T(T)> const &b) {
    std::get<1>(coeffs_) = b;
  }
  inline void set0OrderCoefficient(std::function<T(T)> const &c) {
    std::get<2>(coeffs_) = c;
  }

  void solve(Container<T, Alloc> &solution);
};

}  // namespace explicit_solvers

// ============================================================================
// ========================= IMPLEMENTATIONS ==================================

// ============================================================================
// == Implicit1DSpaceVariableGeneralHeatEquation (Dirichlet) implementation ===
// ============================================================================

template <typename T,
          template <typename, BoundaryConditionType,
                    template <typename, typename> typename Cont, typename>
          typename FDMSolver,
          template <typename, typename> typename Container, typename Alloc>
void implicit_solvers::Implicit1DSpaceVariableGeneralHeatEquation<
    T, BoundaryConditionType::Dirichlet, FDMSolver, Container,
    Alloc>::solve(Container<T, Alloc> &solution, ImplicitPDESchemes scheme) {
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");
  // get correct theta according to the scheme:
  T const theta = ImplicitSpaceVariableHeatEquationSchemes<T>::getTheta(scheme);
  // get space step:
  T const h = spaceStep();
  // get time step:
  T const k = timeStep();
  // get space range:
  auto const &spaceRange = dataPtr_->spaceRange;
  // get source heat function:
  auto const &heatSource = dataPtr_->sourceFunction;
  // space divisions:
  T const &spaceSize = dataPtr_->spaceDivision;
  // calculate scheme const coefficients:
  T const lambda = k / (h * h);
  T const gamma = k / (2.0 * h);
  T const delta = 0.5 * k;
  // save scheme variable coefficients:
  auto const &a = std::get<0>(coeffs_);
  auto const &b = std::get<1>(coeffs_);
  auto const &c = std::get<2>(coeffs_);
  // create container to carry mesh in space and then previous solution:
  Container<T, Alloc> prevSol(spaceSize + 1, T{});
  // populate the container with mesh in space
  discretizeSpace(h, spaceRange.lower(), prevSol);
  // use the mesh in space to get values of initial condition
  discretizeInitialCondition(dataPtr_->initialCondition, prevSol);
  // since coefficients are different in space :
  Container<T, Alloc> low(spaceSize + 1, T{});
  Container<T, Alloc> diag(spaceSize + 1, T{});
  Container<T, Alloc> up(spaceSize + 1, T{});
  // prepare space variable coefficients:
  auto const &A = [&](T x) { return (lambda * a(x) - gamma * b(x)); };
  auto const &B = [&](T x) { return (lambda * a(x) - delta * c(x)); };
  auto const &D = [&](T x) { return (lambda * a(x) + gamma * b(x)); };
  for (std::size_t t = 0; t < low.size(); ++t) {
    low[t] = -1.0 * A(t * h) * theta;
    diag[t] = (1.0 + 2.0 * B(t * h) * theta);
    up[t] = -1.0 * D(t * h) * theta;
  }
  Container<T, Alloc> rhs(spaceSize + 1, T{});
  // create container to carry new solution:
  Container<T, Alloc> nextSol(spaceSize + 1, T{});
  // create first time point:
  T time = k;
  // store terminal time:
  T const lastTime = dataPtr_->timeRange.upper();
  // set properties of FDMSolver:
  solverPtr_->setDiagonals(std::move(low), std::move(diag), std::move(up));
  // differentiate between inhomogeneous and homogeneous PDE:
  if ((dataPtr_->isSourceFunctionSet)) {
    // wrap the scheme coefficients:
    const auto schemeCoeffs = std::make_tuple(A, B, D, h, k);
    // get the correct scheme:
    auto schemeFun =
        ImplicitSpaceVariableHeatEquationSchemes<T>::getInhomScheme(scheme);
    // create a container to carry discretized source heat
    Container<T, Alloc> sourceCurr(spaceSize + 1, T{});
    Container<T, Alloc> sourceNext(spaceSize + 1, T{});
    discretizeInSpace(h, spaceRange.lower(), 0.0, heatSource, sourceCurr);
    discretizeInSpace(h, spaceRange.lower(), time, heatSource, sourceNext);
    // loop for stepping in time:
    while (time <= lastTime) {
      schemeFun(schemeCoeffs, prevSol, sourceCurr, sourceNext, rhs);
      solverPtr_->setBoundaryCondition(
          std::make_pair(boundary_.first(time), boundary_.second(time)));
      solverPtr_->setRhs(rhs);
      solverPtr_->solve(nextSol);
      prevSol = nextSol;
      discretizeInSpace(h, spaceRange.lower(), time, heatSource, sourceCurr);
      discretizeInSpace(h, spaceRange.lower(), 2.0 * time, heatSource,
                        sourceNext);
      time += k;
    }
  } else {
    // wrap the scheme coefficients:
    const auto schemeCoeffs = std::make_tuple(A, B, D, h, T{});
    // get the correct scheme:
    auto schemeFun =
        ImplicitSpaceVariableHeatEquationSchemes<T>::getScheme(scheme);
    // loop for stepping in time:
    while (time <= lastTime) {
      schemeFun(schemeCoeffs, prevSol, Container<T, Alloc>(),
                Container<T, Alloc>(), rhs);
      solverPtr_->setBoundaryCondition(
          std::make_pair(boundary_.first(time), boundary_.second(time)));
      solverPtr_->setRhs(rhs);
      solverPtr_->solve(nextSol);
      prevSol = nextSol;
      time += k;
    }
  }
  // copy into solution vector
  std::copy(prevSol.begin(), prevSol.end(), solution.begin());
}

// ============================================================================
// ==== Implicit1DSpaceVariableGeneralHeatEquation (Robin) implementation =====
// ============================================================================

template <typename T,
          template <typename, BoundaryConditionType,
                    template <typename, typename> typename Cont, typename>
          typename FDMSolver,
          template <typename, typename> typename Container, typename Alloc>
void implicit_solvers::Implicit1DSpaceVariableGeneralHeatEquation<
    T, BoundaryConditionType::Robin, FDMSolver, Container,
    Alloc>::solve(Container<T, Alloc> &solution, ImplicitPDESchemes scheme) {
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");
  // get correct theta according to the scheme:
  T const theta = ImplicitSpaceVariableHeatEquationSchemes<T>::getTheta(scheme);
  // get space step:
  T const h = spaceStep();
  // get time step:
  T const k = timeStep();
  // get space range:
  auto const &spaceRange = dataPtr_->spaceRange;
  // space divisions:
  T const &spaceSize = dataPtr_->spaceDivision;
  // get source heat function:
  auto const &heatSource = dataPtr_->sourceFunction;
  // calculate scheme const coefficients:
  T const lambda = k / (h * h);
  T const gamma = k / (2.0 * h);
  T const delta = 0.5 * k;
  // save scheme variable coefficients:
  auto const &a = std::get<0>(coeffs_);
  auto const &b = std::get<1>(coeffs_);
  auto const &c = std::get<2>(coeffs_);
  // create container to carry mesh in space and then previous solution:
  Container<T, Alloc> prevSol(spaceSize + 1, T{});
  // populate the container with mesh in space
  discretizeSpace(h, spaceRange.lower(), prevSol);
  // use the mesh in space to get values of initial condition
  discretizeInitialCondition(dataPtr_->initialCondition, prevSol);
  // since coefficients are different in space :
  Container<T, Alloc> low(spaceSize + 1, T{});
  Container<T, Alloc> diag(spaceSize + 1, T{});
  Container<T, Alloc> up(spaceSize + 1, T{});
  // prepare space variable coefficients:
  auto const &A = [&](T x) { return (lambda * a(x) - gamma * b(x)); };
  auto const &B = [&](T x) { return (lambda * a(x) - delta * c(x)); };
  auto const &D = [&](T x) { return (lambda * a(x) + gamma * b(x)); };
  for (std::size_t t = 0; t < low.size(); ++t) {
    low[t] = -1.0 * A(t * h) * theta;
    diag[t] = (1.0 + 2.0 * B(t * h) * theta);
    up[t] = -1.0 * D(t * h) * theta;
  }
  Container<T, Alloc> rhs(spaceSize + 1, T{});
  // create container to carry new solution:
  Container<T, Alloc> nextSol(spaceSize + 1, T{});
  // create first time point:
  T time = k;
  // store terminal time:
  T const lastTime = dataPtr_->timeRange.upper();
  // set properties of FDMSolver:
  solverPtr_->setDiagonals(std::move(low), std::move(diag), std::move(up));
  // differentiate between inhomogeneous and homogeneous PDE:
  if ((dataPtr_->isSourceFunctionSet)) {
    // wrap the scheme coefficients:
    const auto schemeCoeffs = std::make_tuple(A, B, D, h, k);
    // get the correct scheme:
    auto schemeFun =
        ImplicitSpaceVariableHeatEquationSchemes<T>::getInhomScheme(scheme);
    // create a container to carry discretized source heat
    Container<T, Alloc> sourceCurr(spaceSize + 1, T{});
    Container<T, Alloc> sourceNext(spaceSize + 1, T{});
    discretizeInSpace(h, spaceRange.lower(), 0.0, heatSource, sourceCurr);
    discretizeInSpace(h, spaceRange.lower(), time, heatSource, sourceNext);
    // loop for stepping in time:
    while (time <= lastTime) {
      schemeFun(schemeCoeffs, prevSol, sourceCurr, sourceNext, rhs);
      solverPtr_->setRhs(rhs);
      solverPtr_->solve(nextSol);
      prevSol = nextSol;
      discretizeInSpace(h, spaceRange.lower(), time, heatSource, sourceCurr);
      discretizeInSpace(h, spaceRange.lower(), 2.0 * time, heatSource,
                        sourceNext);
      time += k;
    }
  } else {
    // wrap the scheme coefficients:
    const auto schemeCoeffs = std::make_tuple(A, B, D, h, T{});
    // get the correct scheme:
    auto schemeFun =
        ImplicitSpaceVariableHeatEquationSchemes<T>::getScheme(scheme);
    // loop for stepping in time:
    while (time <= lastTime) {
      schemeFun(schemeCoeffs, prevSol, Container<T, Alloc>(),
                Container<T, Alloc>(), rhs);
      solverPtr_->setRhs(rhs);
      solverPtr_->solve(nextSol);
      prevSol = nextSol;
      time += k;
    }
  }
  // copy into solution vector
  std::copy(prevSol.begin(), prevSol.end(), solution.begin());
}

// ============================================================================
// == Explicit1DSpaceVariableGeneralHeatEquation (Dirichlet) implementation ===
// ============================================================================

template <typename T, template <typename, typename> typename Container,
          typename Alloc>
void explicit_solvers::Explicit1DSpaceVariableGeneralHeatEquation<
    T, BoundaryConditionType::Dirichlet, Container,
    Alloc>::solve(Container<T, Alloc> &solution, ExplicitPDESchemes scheme) {
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");
  // get space step:
  T const h = spaceStep();
  // get time step:
  T const k = timeStep();
  // get space range:
  auto const &spaceRange = dataPtr_->spaceRange;
  // space divisions:
  T const &spaceSize = dataPtr_->spaceDivision;
  // get source heat function:
  auto const &heatSource = dataPtr_->sourceFunction;
  // calculate scheme const coefficients:
  T const lambda = k / (h * h);
  T const gamma = k / (2.0 * h);
  T const delta = 0.5 * k;
  // save scheme variable coefficients:
  auto const &a = std::get<0>(coeffs_);
  auto const &b = std::get<1>(coeffs_);
  auto const &c = std::get<2>(coeffs_);
  // prepare space variable coefficients:
  auto const &A = [&](T x) { return (lambda * a(x) - gamma * b(x)); };
  auto const &B = [&](T x) { return (lambda * a(x) - delta * c(x)); };
  auto const &D = [&](T x) { return (lambda * a(x) + gamma * b(x)); };
  // wrap up the scheme coefficients:
  auto schemeCoeffs = std::make_tuple(A, B, D);
  // create container to carry mesh in space and then previous solution:
  Container<T, Alloc> initCondition(spaceSize + 1, T{});
  // populate the container with mesh in space
  discretizeSpace(h, spaceRange.lower(), initCondition);
  // use the mesh in space to get values of initial condition
  discretizeInitialCondition(dataPtr_->initialCondition, initCondition);
  // get the correct scheme:
  if (scheme == ExplicitPDESchemes::Euler) {
    ExplicitHeatEulerScheme<T> euler{
        spaceRange.lower(),   dataPtr_->timeRange.upper(),
        std::make_pair(k, h), coeffs_,
        schemeCoeffs,         initCondition,
        heatSource,           dataPtr_->isSourceFunctionSet};
    euler(boundary_, solution);
  } else if (scheme == ExplicitPDESchemes::ADEBarakatClark) {
    ADEHeatBakaratClarkScheme<T> adebc{spaceRange.lower(),
                                       dataPtr_->timeRange.upper(),
                                       std::make_pair(k, h),
                                       schemeCoeffs,
                                       initCondition,
                                       heatSource,
                                       dataPtr_->isSourceFunctionSet};
    adebc(boundary_, solution);
  } else {
    ADEHeatSaulyevScheme<T> ades{spaceRange.lower(),
                                 dataPtr_->timeRange.upper(),
                                 std::make_pair(k, h),
                                 schemeCoeffs,
                                 initCondition,
                                 heatSource,
                                 dataPtr_->isSourceFunctionSet};
    ades(boundary_, solution);
  }
}

// ============================================================================
// ===== Explicit1DSpaceVariableGeneralHeatEquation (Robin) implementation ====
// ============================================================================

template <typename T, template <typename, typename> typename Container,
          typename Alloc>
void explicit_solvers::Explicit1DSpaceVariableGeneralHeatEquation<
    T, BoundaryConditionType::Robin, Container,
    Alloc>::solve(Container<T, Alloc> &solution) {
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");
  // get space step:
  T const h = spaceStep();
  // get time step:
  T const k = timeStep();
  // get space range:
  auto const &spaceRange = dataPtr_->spaceRange;
  // space divisions:
  T const &spaceSize = dataPtr_->spaceDivision;
  // get source heat function:
  auto const &heatSource = dataPtr_->sourceFunction;
  // calculate scheme const coefficients:
  T const lambda = k / (h * h);
  T const gamma = k / (2.0 * h);
  T const delta = 0.5 * k;
  // save scheme variable coefficients:
  auto const &a = std::get<0>(coeffs_);
  auto const &b = std::get<1>(coeffs_);
  auto const &c = std::get<2>(coeffs_);
  // prepare space variable coefficients:
  auto const &A = [&](T x) { return (lambda * a(x) - gamma * b(x)); };
  auto const &B = [&](T x) { return (lambda * a(x) - delta * c(x)); };
  auto const &D = [&](T x) { return (lambda * a(x) + gamma * b(x)); };
  // wrap up the scheme coefficients:
  auto schemeCoeffs = std::make_tuple(A, B, D);
  // create container to carry mesh in space and then previous solution:
  Container<T, Alloc> initCondition(spaceSize + 1, T{});
  // populate the container with mesh in space
  discretizeSpace(h, spaceRange.lower(), initCondition);
  // use the mesh in space to get values of initial condition
  discretizeInitialCondition(dataPtr_->initialCondition, initCondition);
  // get the correct scheme:
  // Here we have only ExplicitEulerScheme available
  ExplicitHeatEulerScheme<T> euler{
      spaceRange.lower(),   dataPtr_->timeRange.upper(),
      std::make_pair(k, h), coeffs_,
      schemeCoeffs,         initCondition,
      heatSource,           dataPtr_->isSourceFunctionSet};
  euler(boundary_.left, boundary_.right, solution);
}

}  // namespace lss_one_dim_space_variable_general_heat_equation_solvers

#endif  //_LSS_ONE_DIM_SPACE_VARIABLE_GENERAL_HEAT_EQUATION_SOLVERS
