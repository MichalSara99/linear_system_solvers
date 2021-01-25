#pragma once
#if !defined(_LSS_ONE_DIM_GENERAL_HEAT_EQUATION_SOLVERS)
#define _LSS_ONE_DIM_GENERAL_HEAT_EQUATION_SOLVERS

#include <functional>

#include "common/lss_enumerations.h"
#include "common/lss_macros.h"
#include "common/lss_utility.h"
#include "lss_one_dim_heat_explicit_schemes.h"
#include "lss_one_dim_heat_implicit_schemes.h"
#include "pde_solvers/one_dim/lss_one_dim_pde_utility.h"

namespace lss_one_dim_general_heat_equation_solvers {

using lss_enumerations::BoundaryConditionType;
using lss_enumerations::ExplicitPDESchemes;
using lss_enumerations::ImplicitPDESchemes;
using lss_one_dim_heat_explicit_schemes::ADEHeatBakaratClarkScheme;
using lss_one_dim_heat_explicit_schemes::ADEHeatSaulyevScheme;
using lss_one_dim_heat_explicit_schemes::ExplicitHeatEulerScheme;
using lss_one_dim_heat_implicit_schemes::ImplicitHeatEquationSchemes;
using lss_one_dim_pde_utility::DirichletBoundary;
using lss_one_dim_pde_utility::Discretization;
using lss_one_dim_pde_utility::RobinBoundary;
using lss_utility::Range;
using lss_utility::uptr_t;

namespace implicit_solvers {
// ============================================================================
// ============= Implicit1DGeneralHeatEquation General Template
// ===============
// ============================================================================

template <typename T, BoundaryConditionType BType,
          template <typename, BoundaryConditionType,
                    template <typename, typename> typename Cont, typename>
          typename FDMSolver,
          template <typename, typename> typename Container, typename Alloc>
class Implicit1DGeneralHeatEquation {};

// ============================================================================
// ===== Implicit1DGeneralHeatEquation Dirichlet Specialisation Template
// ======
// ============================================================================
//
//	u_t = a*u_xx + b*u_x + c*u + F(x,t), t > 0, x_1 < x < x_2
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
class Implicit1DGeneralHeatEquation<T, BoundaryConditionType::Dirichlet,
                                    FDMSolver, Container, Alloc>
    : public Discretization<T, Container, Alloc> {
 private:
  typedef FDMSolver<T, BoundaryConditionType::Dirichlet, Container, Alloc>
      fdm_solver_t;

  uptr_t<fdm_solver_t> solverPtr_;  // finite-difference solver
  Range<T> spacer_;                 // space range
  T terminalT_;                     // terminal time
  std::size_t timeN_;               // number of time subdivisions
  std::size_t spaceN_;              // number of space subdivisions
  std::function<T(T)> init_;        // init condition
  std::function<T(T, T)> source_;   // heat source F(x,t)
  DirichletBoundary<T> boundary_;   // boundaries
  std::tuple<T, T, T> coeffs_;      // coefficients a, b, c in PDE
  bool isSourceSet_;

 public:
  typedef T value_type;
  explicit Implicit1DGeneralHeatEquation() = delete;
  explicit Implicit1DGeneralHeatEquation(Range<T> const &spaceRange,
                                         T terminalTime,
                                         std::size_t const &spaceDiscretization,
                                         std::size_t const &timeDiscretization)
      : solverPtr_{std::make_unique<fdm_solver_t>(spaceDiscretization + 1)},
        spacer_{spaceRange},
        terminalT_{terminalTime},
        timeN_{timeDiscretization},
        spaceN_{spaceDiscretization},
        source_{nullptr},
        isSourceSet_{false} {}

  ~Implicit1DGeneralHeatEquation() {}

  Implicit1DGeneralHeatEquation(Implicit1DGeneralHeatEquation const &) = delete;
  Implicit1DGeneralHeatEquation(Implicit1DGeneralHeatEquation &&) = delete;
  Implicit1DGeneralHeatEquation &operator=(
      Implicit1DGeneralHeatEquation const &) = delete;
  Implicit1DGeneralHeatEquation &operator=(Implicit1DGeneralHeatEquation &&) =
      delete;

  inline T spaceStep() const {
    return (spacer_.spread() / static_cast<T>(spaceN_));
  }
  inline T timeStep() const { return (terminalT_ / static_cast<T>(timeN_)); }

  inline void setBoundaryCondition(
      DirichletBoundary<T> const &dirichletBoundary) {
    boundary_ = dirichletBoundary;
  }
  inline void setInitialCondition(std::function<T(T)> const &initialCondition) {
    init_ = initialCondition;
  }
  inline void setHeatSource(std::function<T(T, T)> const &heatSource) {
    isSourceSet_ = true;
    source_ = heatSource;
  }
  inline void set2OrderCoefficient(T value) { std::get<0>(coeffs_) = value; }
  inline void set1OrderCoefficient(T value) { std::get<1>(coeffs_) = value; }
  inline void set0OrderCoefficient(T value) { std::get<2>(coeffs_) = value; }

  void solve(Container<T, Alloc> &solution,
             ImplicitPDESchemes scheme = ImplicitPDESchemes::CrankNicolson);
};

// ============================================================================
// ========= Implicit1DGeneralHeatEquation Robin Specialisation Template
// ======
// ============================================================================
//
//	u_t = a*u_xx + b*u_x + c*u + F(x,t), t > 0, x_1 < x < x_2
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
//	U_0 = ((2*d_1 + f_1*h)/(2*d_1 - f_1*h)) * U_1 + (2*h/(f_1*h - 2*d_1))*A
//	U_N-1 = ((2*d_2 + f_2*h)/(2*d_2 - f_2*h)) * U_N + (2*h/(f_2*h -
// 2*d_2))*B
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
class Implicit1DGeneralHeatEquation<T, BoundaryConditionType::Robin, FDMSolver,
                                    Container, Alloc>
    : public Discretization<T, Container, Alloc> {
 private:
  typedef FDMSolver<T, BoundaryConditionType::Robin, Container, Alloc>
      fdm_solver_t;

  uptr_t<fdm_solver_t> solverPtr_;  // finite-difference solver
  Range<T> spacer_;                 // space range
  T terminalT_;                     // terminal time
  std::size_t timeN_;               // number of time subdivisions
  std::size_t spaceN_;              // number of space subdivisions
  std::function<T(T, T)> source_;   // heat source F(x,t)
  std::function<T(T)> init_;        // initi condition
  RobinBoundary<T> boundary_;       // boundaries
  std::tuple<T, T, T> coeffs_;      // coefficients a, b, c in PDE
  bool isSourceSet_;

 public:
  typedef T value_type;
  explicit Implicit1DGeneralHeatEquation() = delete;
  explicit Implicit1DGeneralHeatEquation(Range<T> const &spaceRange,
                                         T terminalTime,
                                         std::size_t const &spaceDiscretization,
                                         std::size_t const &timeDiscretization)
      : solverPtr_{std::make_unique<fdm_solver_t>(spaceDiscretization + 1)},
        spacer_{spaceRange},
        terminalT_{terminalTime},
        timeN_{timeDiscretization},
        spaceN_{spaceDiscretization},
        source_{nullptr},
        isSourceSet_{false} {}

  ~Implicit1DGeneralHeatEquation() {}

  Implicit1DGeneralHeatEquation(Implicit1DGeneralHeatEquation const &) = delete;
  Implicit1DGeneralHeatEquation(Implicit1DGeneralHeatEquation &&) = delete;
  Implicit1DGeneralHeatEquation &operator=(
      Implicit1DGeneralHeatEquation const &) = delete;
  Implicit1DGeneralHeatEquation &operator=(Implicit1DGeneralHeatEquation &&) =
      delete;

  inline T spaceStep() const {
    return (spacer_.spread() / static_cast<T>(spaceN_));
  }
  inline T timeStep() const { return (terminalT_ / static_cast<T>(timeN_)); }

  inline void setBoundaryCondition(RobinBoundary<T> const &robinBoundary) {
    boundary_ = robinBoundary;
    solverPtr_->setBoundaryCondition(robinBoundary.left, robinBoundary.right);
  }

  inline void setInitialCondition(std::function<T(T)> const &initialCondition) {
    init_ = initialCondition;
  }
  inline void setHeatSource(std::function<T(T, T)> const &heatSource) {
    isSourceSet_ = true;
    source_ = heatSource;
  }
  inline void set2OrderCoefficient(T value) { std::get<0>(coeffs_) = value; }
  inline void set1OrderCoefficient(T value) { std::get<1>(coeffs_) = value; }
  inline void set0OrderCoefficient(T value) { std::get<2>(coeffs_) = value; }

  void solve(Container<T, Alloc> &solution,
             ImplicitPDESchemes scheme = ImplicitPDESchemes::CrankNicolson);
};

}  // namespace implicit_solvers

namespace explicit_solvers {

// ============================================================================
// =========== Explicit1DGeneralHeatEquation General Template =================
// ============================================================================

template <typename T, BoundaryConditionType BType,
          template <typename, typename> typename Container, typename Alloc>
class Explicit1DGeneralHeatEquation {};

// ============================================================================
// ====== Explicit1DGeneralHeatEquation Dirichlet Specialisation Template =====
// ============================================================================
//
//	u_t = a*u_xx + b*u_x + c*u + F(x,t), t > 0, x_1 < x < x_2
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
class Explicit1DGeneralHeatEquation<T, BoundaryConditionType::Dirichlet,
                                    Container, Alloc>
    : public Discretization<T, Container, Alloc> {
 private:
  Range<T> spacer_;                // space range
  T terminalT_;                    // terminal time
  std::size_t timeN_;              // number of time subdivisions
  std::size_t spaceN_;             // number of space subdivisions
  std::function<T(T)> init_;       // initi condition
  std::function<T(T, T)> source_;  // heat source	F(x,t)
  DirichletBoundary<T> boundary_;  // boundaries
  std::tuple<T, T, T> coeffs_;     // coefficients a, b, c in PDE
  bool isSourceSet_;

 public:
  typedef T value_type;
  explicit Explicit1DGeneralHeatEquation() = delete;
  explicit Explicit1DGeneralHeatEquation(Range<T> const &spaceRange,
                                         T terminalTime,
                                         std::size_t const &spaceDiscretization,
                                         std::size_t const &timeDiscretization)
      : spacer_{spaceRange},
        terminalT_{terminalTime},
        timeN_{timeDiscretization},
        spaceN_{spaceDiscretization},
        source_{nullptr},
        isSourceSet_{false} {}

  ~Explicit1DGeneralHeatEquation() {}

  Explicit1DGeneralHeatEquation(Explicit1DGeneralHeatEquation const &) = delete;
  Explicit1DGeneralHeatEquation(Explicit1DGeneralHeatEquation &&) = delete;
  Explicit1DGeneralHeatEquation &operator=(
      Explicit1DGeneralHeatEquation const &) = delete;
  Explicit1DGeneralHeatEquation &operator=(Explicit1DGeneralHeatEquation &&) =
      delete;

  inline T spaceStep() const {
    return (spacer_.spread() / static_cast<T>(spaceN_));
  }
  inline T timeStep() const { return (terminalT_ / static_cast<T>(timeN_)); }

  inline void setBoundaryCondition(
      DirichletBoundary<T> const &dirichletBoundary) {
    boundary_ = dirichletBoundary;
  }
  inline void setInitialCondition(std::function<T(T)> const &initialCondition) {
    init_ = initialCondition;
  }
  inline void setHeatSource(std::function<T(T, T)> const &heatSource) {
    isSourceSet_ = true;
    source_ = heatSource;
  }
  inline void set2OrderCoefficient(T value) { std::get<0>(coeffs_) = value; }
  inline void set1OrderCoefficient(T value) { std::get<1>(coeffs_) = value; }
  inline void set0OrderCoefficient(T value) { std::get<2>(coeffs_) = value; }

  void solve(Container<T, Alloc> &solution,
             ExplicitPDESchemes scheme = ExplicitPDESchemes::ADEBarakatClark);
};

// ============================================================================
// ======= Explicit1DGeneralHeatEquation Robin Specialisation Template ========
// ============================================================================
//
//	u_t = a*u_xx + b*u_x + c*u + F(x,t), t > 0, x_1 < x < x_2
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
//			(2*h/(f_2*h - 2*d_2))*B
//
//	or
//
//	U_0 = alpha * U_1 + phi,
//	U_N-1 = beta * U_N + psi,
//
// ============================================================================

template <typename T, template <typename, typename> typename Container,
          typename Alloc>
class Explicit1DGeneralHeatEquation<T, BoundaryConditionType::Robin, Container,
                                    Alloc>
    : public Discretization<T, Container, Alloc> {
 private:
  Range<T> spacer_;                // space range
  T terminalT_;                    // terminal time
  std::size_t timeN_;              // number of time subdivisions
  std::size_t spaceN_;             // number of space subdivisions
  std::function<T(T, T)> source_;  // heat source F(x,t)
  std::function<T(T)> init_;       // initi condition
  RobinBoundary<T> boundary_;      // boundary
  std::tuple<T, T, T> coeffs_;     // coefficients a, b, c in PDE
  bool isSourceSet_;

 public:
  typedef T value_type;
  explicit Explicit1DGeneralHeatEquation() = delete;
  explicit Explicit1DGeneralHeatEquation(Range<T> const &spaceRange,
                                         T terminalTime,
                                         std::size_t const &spaceDiscretization,
                                         std::size_t const &timeDiscretization)
      : spacer_{spaceRange},
        terminalT_{terminalTime},
        timeN_{timeDiscretization},
        spaceN_{spaceDiscretization},
        source_{nullptr},
        isSourceSet_{false} {}

  ~Explicit1DGeneralHeatEquation() {}

  Explicit1DGeneralHeatEquation(Explicit1DGeneralHeatEquation const &) = delete;
  Explicit1DGeneralHeatEquation(Explicit1DGeneralHeatEquation &&) = delete;
  Explicit1DGeneralHeatEquation &operator=(
      Explicit1DGeneralHeatEquation const &) = delete;
  Explicit1DGeneralHeatEquation &operator=(Explicit1DGeneralHeatEquation &&) =
      delete;

  inline T spaceStep() const {
    return (spacer_.spread() / static_cast<T>(spaceN_));
  }
  inline T timeStep() const { return (terminalT_ / static_cast<T>(timeN_)); }

  inline void setBoundaryCondition(RobinBoundary<T> const &robinBoundary) {
    boundary_ = robinBoundary;
  }
  inline void setInitialCondition(std::function<T(T)> const &initialCondition) {
    init_ = initialCondition;
  }
  inline void setHeatSource(std::function<T(T, T)> const &heatSource) {
    isSourceSet_ = true;
    source_ = heatSource;
  }
  inline void set2OrderCoefficient(T value) { std::get<0>(coeffs_) = value; }
  inline void set1OrderCoefficient(T value) { std::get<1>(coeffs_) = value; }
  inline void set0OrderCoefficient(T value) { std::get<2>(coeffs_) = value; }

  void solve(Container<T, Alloc> &solution);
};

}  // namespace explicit_solvers

// ========================= IMPLEMENTATIONS ==================================

// ============================================================================
// ========= Implicit1DGeneralHeatEquation (Dirichlet) implementation =========
// ============================================================================

template <typename T,
          template <typename, BoundaryConditionType,
                    template <typename, typename> typename Cont, typename>
          typename FDMSolver,
          template <typename, typename> typename Container, typename Alloc>
void implicit_solvers::Implicit1DGeneralHeatEquation<
    T, BoundaryConditionType::Dirichlet, FDMSolver, Container,
    Alloc>::solve(Container<T, Alloc> &solution, ImplicitPDESchemes scheme) {
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");
  // get correct theta according to the scheme:
  T const theta = ImplicitHeatEquationSchemes<T>::getTheta(scheme);
  // get space step:
  T const h = spaceStep();
  // get time step:
  T const k = timeStep();
  // calculate scheme const coefficients:
  T const lambda = (std::get<0>(coeffs_) * k) / (h * h);
  T const gamma = (std::get<1>(coeffs_) * k) / (2.0 * h);
  T const delta = (std::get<2>(coeffs_) * k);
  // create container to carry mesh in space and then previous solution:
  Container<T, Alloc> prevSol(spaceN_ + 1, T{});
  // populate the container with mesh in space
  discretizeSpace(h, spacer_.lower(), prevSol);
  // use the mesh in space to get values of initial condition
  discretizeInitialCondition(init_, prevSol);
  // prepare containers for diagonal vectors for FDMSolver:
  Container<T, Alloc> low(spaceN_ + 1, -1.0 * (lambda - gamma) * theta);
  Container<T, Alloc> diag(spaceN_ + 1, (1.0 + (2.0 * lambda - delta) * theta));
  Container<T, Alloc> up(spaceN_ + 1, -1.0 * (lambda + gamma) * theta);
  Container<T, Alloc> rhs(spaceN_ + 1, T{});
  // create container to carry new solution:
  Container<T, Alloc> nextSol(spaceN_ + 1, T{});
  // create first time point:
  T time = k;
  // store terminal time:
  T const lastTime = terminalT_;
  // set properties of FDMSolver:
  solverPtr_->setDiagonals(std::move(low), std::move(diag), std::move(up));
  // differentiate between inhomogeneous and homogeneous PDE:
  if (isSourceSet_) {
    // wrap the scheme coefficients:
    const auto schemeCoeffs = std::make_tuple(lambda, gamma, delta, k);
    // get the correct scheme:
    auto schemeFun = ImplicitHeatEquationSchemes<T>::getInhomScheme(scheme);
    // create a container to carry discretized source heat
    Container<T, Alloc> sourceCurr(spaceN_ + 1, T{});
    Container<T, Alloc> sourceNext(spaceN_ + 1, T{});
    discretizeInSpace(h, spacer_.lower(), 0.0, source_, sourceCurr);
    discretizeInSpace(h, spacer_.lower(), time, source_, sourceNext);
    // loop for stepping in time:
    while (time <= lastTime) {
      schemeFun(schemeCoeffs, prevSol, sourceCurr, sourceNext, rhs);
      solverPtr_->setBoundaryCondition(
          std::make_pair(boundary_.first(time), boundary_.second(time)));
      solverPtr_->setRhs(rhs);
      solverPtr_->solve(nextSol);
      prevSol = nextSol;
      discretizeInSpace(h, spacer_.lower(), time, source_, sourceCurr);
      discretizeInSpace(h, spacer_.lower(), 2.0 * time, source_, sourceNext);
      time += k;
    }
  } else {
    // wrap the scheme coefficients:
    const auto schemeCoeffs = std::make_tuple(lambda, gamma, delta, T{});
    // get the correct scheme:
    auto schemeFun = ImplicitHeatEquationSchemes<T>::getScheme(scheme);
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
// =========== Implicit1DGeneralHeatEquation (Robin) implementation ===========
// ============================================================================

template <typename T,
          template <typename, BoundaryConditionType,
                    template <typename, typename> typename Cont, typename>
          typename FDMSolver,
          template <typename, typename> typename Container, typename Alloc>
void implicit_solvers::Implicit1DGeneralHeatEquation<
    T, BoundaryConditionType::Robin, FDMSolver, Container,
    Alloc>::solve(Container<T, Alloc> &solution, ImplicitPDESchemes scheme) {
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");
  // get correct theta according to the scheme:
  T const theta = ImplicitHeatEquationSchemes<T>::getTheta(scheme);
  // get space step:
  T const h = spaceStep();
  // get time step:
  T const k = timeStep();
  // calculate scheme const coefficients:
  T const lambda = (std::get<0>(coeffs_) * k) / (h * h);
  T const gamma = (std::get<1>(coeffs_) * k) / (2.0 * h);
  T const delta = (std::get<2>(coeffs_) * k);
  // create container to carry mesh in space and then previous solution:
  Container<T, Alloc> prevSol(spaceN_ + 1, T{});
  // populate the container with mesh in space
  discretizeSpace(h, spacer_.lower(), prevSol);
  // use the mesh in space to get values of initial condition
  discretizeInitialCondition(init_, prevSol);
  // prepare containers for diagonal vectors for FDMSolver:
  Container<T, Alloc> low(spaceN_ + 1, -1.0 * (lambda - gamma) * theta);
  Container<T, Alloc> diag(spaceN_ + 1, (1.0 + (2.0 * lambda - delta) * theta));
  Container<T, Alloc> up(spaceN_ + 1, -1.0 * (lambda + gamma) * theta);
  Container<T, Alloc> rhs(spaceN_ + 1, T{});
  // create container to carry new solution:
  Container<T, Alloc> nextSol(spaceN_ + 1, T{});
  // create first time point:
  T time = k;
  // store terminal time:
  T const lastTime = terminalT_;
  // set properties of FDMSolver:
  solverPtr_->setDiagonals(std::move(low), std::move(diag), std::move(up));
  // differentiate between inhomogeneous and homogeneous PDE:
  if (isSourceSet_) {
    // wrap the scheme coefficients:
    const auto schemeCoeffs = std::make_tuple(lambda, gamma, delta, k);
    // get the correct scheme:
    auto schemeFun = ImplicitHeatEquationSchemes<T>::getInhomScheme(scheme);
    // create a container to carry discretized source heat
    Container<T, Alloc> sourceCurr(spaceN_ + 1, T{});
    Container<T, Alloc> sourceNext(spaceN_ + 1, T{});
    discretizeInSpace(h, spacer_.lower(), 0.0, source_, sourceCurr);
    discretizeInSpace(h, spacer_.lower(), time, source_, sourceNext);
    // loop for stepping in time:
    while (time <= lastTime) {
      schemeFun(schemeCoeffs, prevSol, sourceCurr, sourceNext, rhs);
      solverPtr_->setRhs(rhs);
      solverPtr_->solve(nextSol);
      prevSol = nextSol;
      discretizeInSpace(h, spacer_.lower(), time, source_, sourceCurr);
      discretizeInSpace(h, spacer_.lower(), 2.0 * time, source_, sourceNext);
      time += k;
    }
  } else {
    // wrap the scheme coefficients:
    const auto schemeCoeffs = std::make_tuple(lambda, gamma, delta, T{});
    // get the correct scheme:
    auto schemeFun = ImplicitHeatEquationSchemes<T>::getScheme(scheme);
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
// ========== Explicit1DGeneralHeatEquation (Dirichlet) implementation ========
// ============================================================================

template <typename T, template <typename, typename> typename Container,
          typename Alloc>
void explicit_solvers::Explicit1DGeneralHeatEquation<
    T, BoundaryConditionType::Dirichlet, Container,
    Alloc>::solve(Container<T, Alloc> &solution, ExplicitPDESchemes scheme) {
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");
  // get space step:
  T const h = spaceStep();
  // get time step:
  T const k = timeStep();
  // create container to carry mesh in space and then previous solution:
  Container<T, Alloc> initCondition(spaceN_ + 1, T{});
  // populate the container with mesh in space
  discretizeSpace(h, spacer_.lower(), initCondition);
  // use the mesh in space to get values of initial condition
  discretizeInitialCondition(init_, initCondition);
  // get the correct scheme:
  if (scheme == ExplicitPDESchemes::Euler) {
    ExplicitHeatEulerScheme<T> euler{
        spacer_.lower(), terminalT_, std::make_pair(k, h), coeffs_,
        initCondition,   source_,    isSourceSet_};
    euler(boundary_, solution);
  } else if (scheme == ExplicitPDESchemes::ADEBarakatClark) {
    ADEHeatBakaratClarkScheme<T> adebc{
        spacer_.lower(), terminalT_, std::make_pair(k, h), coeffs_,
        initCondition,   source_,    isSourceSet_};
    adebc(boundary_, solution);
  } else {
    ADEHeatSaulyevScheme<T> ades{
        spacer_.lower(), terminalT_, std::make_pair(k, h), coeffs_,
        initCondition,   source_,    isSourceSet_};
    ades(boundary_, solution);
  }
}

// ============================================================================
// ============= Explicit1DGeneralHeatEquation (Robin) implementation =========
// ============================================================================

template <typename T, template <typename, typename> typename Container,
          typename Alloc>
void explicit_solvers::Explicit1DGeneralHeatEquation<
    T, BoundaryConditionType::Robin, Container,
    Alloc>::solve(Container<T, Alloc> &solution) {
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");
  // get space step:
  T const h = spaceStep();
  // get time step:
  T const k = timeStep();
  // create container to carry mesh in space and then previous solution:
  Container<T, Alloc> initCondition(spaceN_ + 1, T{});
  // populate the container with mesh in space
  discretizeSpace(h, spacer_.lower(), initCondition);
  // use the mesh in space to get values of initial condition
  discretizeInitialCondition(init_, initCondition);
  // get the correct scheme:
  // Here we have only ExplicitEulerScheme available
  ExplicitHeatEulerScheme<T> euler{
      spacer_.lower(), terminalT_, std::make_pair(k, h), coeffs_,
      initCondition,   source_,    isSourceSet_};
  euler(boundary_, solution);
}

}  // namespace lss_one_dim_general_heat_equation_solvers

#endif  //_LSS_ONE_DIM_GENERAL_HEAT_EQUATION_SOLVERS
