#pragma once
#if !defined(_LSS_ONE_DIM_GENERAL_HEAT_EQUATION_SOLVERS_CUDA)
#define _LSS_ONE_DIM_GENERAL_HEAT_EQUATION_SOLVERS_CUDA

#include "common/lss_enumerations.h"
#include "common/lss_utility.h"
#include "lss_one_dim_heat_explicit_schemes_cuda.h"
#include "lss_one_dim_heat_implicit_schemes_cuda.h"
#include "pde_solvers/one_dim/lss_one_dim_pde_utility.h"
#include "sparse_solvers/lss_sparse_solvers_cuda.h"

namespace lss_one_dim_general_heat_equation_solvers_cuda {

using lss_enumerations::BoundaryConditionType;
using lss_enumerations::ImplicitPDESchemes;
using lss_enumerations::MemorySpace;
using lss_one_dim_heat_explicit_schemes_cuda::ExplicitEulerHeatEquationScheme;
using lss_one_dim_heat_implicit_schemes_cuda::ImplicitHeatEquationSchemes;
using lss_one_dim_pde_utility::Discretization;
using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
using lss_utility::FlatMatrix;
using lss_utility::Range;
using lss_utility::uptr_t;

// move this somewhere else:
template <typename T>
using DirichletPair = std::pair<std::function<T(T)>, std::function<T(T)>>;

namespace implicit_solvers {

// ============================================================================
// ========== Implicit1DGeneralHeatEquationCUDA General Template ==============
// ============================================================================

template <typename T, BoundaryConditionType BType, MemorySpace MemSpace,
          template <MemorySpace, typename> typename RealSparsePolicyCUDA,
          template <typename, typename> typename Container, typename Alloc>
class Implicit1DGeneralHeatEquationCUDA {};

// ============================================================================
// === Implicit1DGeneralHeatEquationCUDA Dirichlet Specialisation Template ====
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

template <typename T, MemorySpace MemSpace,
          template <MemorySpace, typename> typename RealSparsePolicyCUDA,
          template <typename, typename> typename Container, typename Alloc>
class Implicit1DGeneralHeatEquationCUDA<T, BoundaryConditionType::Dirichlet,
                                        MemSpace, RealSparsePolicyCUDA,
                                        Container, Alloc>
    : public Discretization<T, Container, Alloc> {
 private:
  typedef RealSparsePolicyCUDA<MemSpace, T> cuda_solver_t;

  uptr_t<cuda_solver_t> solverPtr_;  // finite-difference solver
  Range<T> spacer_;                  // space range
  T terminalT_;                      // terminal time
  std::size_t timeN_;                // number of time subdivisions
  std::size_t spaceN_;               // number of space subdivisions
  std::function<T(T)> init_;         // initi condition
  std::function<T(T, T)> source_;    // heat source
  DirichletPair<T> boundary_;        // boundaries
  std::tuple<T, T, T> coeffs_;       // coefficients a, b, c in PDE
  bool isSourceSet_;

 public:
  typedef T value_type;
  explicit Implicit1DGeneralHeatEquationCUDA() = delete;
  explicit Implicit1DGeneralHeatEquationCUDA(
      Range<T> const &spaceRange, T terminalTime,
      std::size_t const &spaceDiscretization,
      std::size_t const &timeDiscretization)
      : solverPtr_{std::make_unique<cuda_solver_t>()},
        spacer_{spaceRange},
        terminalT_{terminalTime},
        timeN_{timeDiscretization},
        spaceN_{spaceDiscretization},
        source_{nullptr},
        isSourceSet_{false} {}

  ~Implicit1DGeneralHeatEquationCUDA() {}

  Implicit1DGeneralHeatEquationCUDA(Implicit1DGeneralHeatEquationCUDA const &) =
      delete;
  Implicit1DGeneralHeatEquationCUDA(Implicit1DGeneralHeatEquationCUDA &&) =
      delete;
  Implicit1DGeneralHeatEquationCUDA &operator=(
      Implicit1DGeneralHeatEquationCUDA const &) = delete;
  Implicit1DGeneralHeatEquationCUDA &operator=(
      Implicit1DGeneralHeatEquationCUDA &&) = delete;

  inline T spaceStep() const {
    return (spacer_.spread() / static_cast<T>(spaceN_));
  }
  inline T timeStep() const { return (terminalT_ / static_cast<T>(timeN_)); }

  inline void setBoundaryCondition(DirichletPair<T> const &boundaryPair) {
    boundary_ = boundaryPair;
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
// ===== Implicit1DGeneralHeatEquationCUDA Robin Specialisation Template ======
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

template <typename T, MemorySpace MemSpace,
          template <MemorySpace, typename> typename RealSparsePolicyCUDA,
          template <typename, typename> typename Container, typename Alloc>
class Implicit1DGeneralHeatEquationCUDA<T, BoundaryConditionType::Robin,
                                        MemSpace, RealSparsePolicyCUDA,
                                        Container, Alloc>
    : public Discretization<T, Container, Alloc> {
 private:
  typedef RealSparsePolicyCUDA<MemSpace, T> cuda_solver_t;

  uptr_t<cuda_solver_t> solverPtr_;  // finite-difference solver
  Range<T> spacer_;                  // space range
  T terminalT_;                      // terminal time
  std::size_t timeN_;                // number of time subdivisions
  std::size_t spaceN_;               // number of space subdivisions
  std::function<T(T)> init_;         // initi condition
  std::function<T(T, T)> source_;    // heat source
  std::pair<T, T> leftBoundary_;     // left boundaries
  std::pair<T, T> rightBoundary_;    // right boundaries
  std::tuple<T, T, T> coeffs_;       // coefficients a, b, c in PDE
  bool isSourceSet_;

 public:
  typedef T value_type;
  explicit Implicit1DGeneralHeatEquationCUDA() = delete;
  explicit Implicit1DGeneralHeatEquationCUDA(
      Range<T> const &spaceRange, T terminalTime,
      std::size_t const &spaceDiscretization,
      std::size_t const &timeDiscretization)
      : solverPtr_{std::make_unique<RealSparsePolicyCUDA<MemSpace, T>>()},
        spacer_{spaceRange},
        terminalT_{terminalTime},
        timeN_{timeDiscretization},
        spaceN_{spaceDiscretization},
        source_{nullptr},
        isSourceSet_{false} {}

  ~Implicit1DGeneralHeatEquationCUDA() {}

  Implicit1DGeneralHeatEquationCUDA(Implicit1DGeneralHeatEquationCUDA const &) =
      delete;
  Implicit1DGeneralHeatEquationCUDA(Implicit1DGeneralHeatEquationCUDA &&) =
      delete;
  Implicit1DGeneralHeatEquationCUDA &operator=(
      Implicit1DGeneralHeatEquationCUDA const &) = delete;
  Implicit1DGeneralHeatEquationCUDA &operator=(
      Implicit1DGeneralHeatEquationCUDA &&) = delete;

  inline T spaceStep() const {
    return (spacer_.spread() / static_cast<T>(spaceN_));
  }
  inline T timeStep() const { return (terminalT_ / static_cast<T>(timeN_)); }

  inline void setBoundaryCondition(std::pair<T, T> const &left,
                                   std::pair<T, T> const &right) {
    leftBoundary_ = left;
    T beta_ = static_cast<T>(1.0) / right.first;
    T psi_ = static_cast<T>(-1.0) * right.second / right.first;
    rightBoundary_ = std::make_pair(beta_, psi_);
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
// ============= Explicit1DGeneralHeatEquationCUDA General Template ===========
// ============================================================================

template <typename T, BoundaryConditionType BType,
          template <typename, typename> typename Container, typename Alloc>
class Explicit1DGeneralHeatEquationCUDA {};

// ============================================================================
// ==== Explicit1DGeneralHeatEquationCUDA Dirichlet Specialisation Template ===
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
class Explicit1DGeneralHeatEquationCUDA<T, BoundaryConditionType::Dirichlet,
                                        Container, Alloc>
    : public Discretization<T, Container, Alloc> {
 private:
  Range<T> spacer_;                // space range
  T terminalT_;                    // terminal time
  std::size_t timeN_;              // number of time subdivisions
  std::size_t spaceN_;             // number of space subdivisions
  std::function<T(T)> init_;       // initi condition
  std::function<T(T, T)> source_;  // heat source	F(x,t)
  DirichletPair<T> boundary_;      // boundaries
  std::tuple<T, T, T> coeffs_;     // coefficients a, b, c in PDE
  bool isSourceSet_;

 public:
  typedef T value_type;
  explicit Explicit1DGeneralHeatEquationCUDA() = delete;
  explicit Explicit1DGeneralHeatEquationCUDA(
      Range<T> const &spaceRange, T terminalTime,
      std::size_t const &spaceDiscretization,
      std::size_t const &timeDiscretization)
      : spacer_{spaceRange},
        terminalT_{terminalTime},
        timeN_{timeDiscretization},
        spaceN_{spaceDiscretization},
        source_{nullptr},
        isSourceSet_{false} {}

  ~Explicit1DGeneralHeatEquationCUDA() {}

  Explicit1DGeneralHeatEquationCUDA(Explicit1DGeneralHeatEquationCUDA const &) =
      delete;
  Explicit1DGeneralHeatEquationCUDA(Explicit1DGeneralHeatEquationCUDA &&) =
      delete;
  Explicit1DGeneralHeatEquationCUDA &operator=(
      Explicit1DGeneralHeatEquationCUDA const &) = delete;
  Explicit1DGeneralHeatEquationCUDA &operator=(
      Explicit1DGeneralHeatEquationCUDA &&) = delete;

  inline T spaceStep() const {
    return (spacer_.spread() / static_cast<T>(spaceN_));
  }
  inline T timeStep() const { return (terminalT_ / static_cast<T>(timeN_)); }

  inline void setBoundaryCondition(DirichletPair<T> const &boundaryPair) {
    boundary_ = boundaryPair;
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

  // stability check:
  bool isStable() const;

  void solve(Container<T, Alloc> &solution);
};

// ============================================================================
// ==== Explicit1DGeneralHeatEquationCUDA Robin Specialisation Template =======
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
class Explicit1DGeneralHeatEquationCUDA<T, BoundaryConditionType::Robin,
                                        Container, Alloc>
    : public Discretization<T, Container, Alloc> {
 private:
  Range<T> spacer_;                // space range
  T terminalT_;                    // terminal time
  std::size_t timeN_;              // number of time subdivisions
  std::size_t spaceN_;             // number of space subdivisions
  std::function<T(T)> init_;       // initi condition
  std::function<T(T, T)> source_;  // heat source F(x,t)
  std::pair<T, T> leftBoundary_;   // left boundary pair
  std::pair<T, T> rightBoundary_;  // right boundary pair
  std::tuple<T, T, T> coeffs_;     // coefficients a, b, c in PDE
  bool isSourceSet_;

 public:
  typedef T value_type;
  explicit Explicit1DGeneralHeatEquationCUDA() = delete;
  explicit Explicit1DGeneralHeatEquationCUDA(
      Range<T> const &spaceRange, T terminalTime,
      std::size_t const &spaceDiscretization,
      std::size_t const &timeDiscretization)
      : spacer_{spaceRange},
        terminalT_{terminalTime},
        timeN_{timeDiscretization},
        spaceN_{spaceDiscretization},
        source_{nullptr},
        isSourceSet_{false} {}

  ~Explicit1DGeneralHeatEquationCUDA() {}

  Explicit1DGeneralHeatEquationCUDA(Explicit1DGeneralHeatEquationCUDA const &) =
      delete;
  Explicit1DGeneralHeatEquationCUDA(Explicit1DGeneralHeatEquationCUDA &&) =
      delete;
  Explicit1DGeneralHeatEquationCUDA &operator=(
      Explicit1DGeneralHeatEquationCUDA const &) = delete;
  Explicit1DGeneralHeatEquationCUDA &operator=(
      Explicit1DGeneralHeatEquationCUDA &&) = delete;

  inline T spaceStep() const {
    return (spacer_.spread() / static_cast<T>(spaceN_));
  }
  inline T timeStep() const { return (terminalT_ / static_cast<T>(timeN_)); }

  inline void setBoundaryCondition(std::pair<T, T> const &left,
                                   std::pair<T, T> const &right) {
    leftBoundary_ = left;
    T beta_ = static_cast<T>(1.0) / right.first;
    T psi_ = static_cast<T>(-1.0) * right.second / right.first;
    rightBoundary_ = std::make_pair(beta_, psi_);
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

  // stability check:
  bool isStable() const;

  void solve(Container<T, Alloc> &solution);
};
}  // namespace explicit_solvers

// ============================= IMPLEMENTATIONS ==============================

// ============================================================================
// ====== Implicit1DGeneralHeatEquationCUDA (Dirichlet) implementation ========
// ============================================================================

template <typename T, MemorySpace MemSpace,
          template <MemorySpace, typename> typename RealSparsePolicyCUDA,
          template <typename, typename> typename Container, typename Alloc>
void implicit_solvers::Implicit1DGeneralHeatEquationCUDA<
    T, BoundaryConditionType::Dirichlet, MemSpace, RealSparsePolicyCUDA,
    Container, Alloc>::solve(Container<T, Alloc> &solution,
                             ImplicitPDESchemes scheme) {
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
  // store size of matrix:
  std::size_t const m = spaceN_ - 1;
  // create container to carry mesh in space and then previous solution:
  Container<T, Alloc> prevSol(m, T{});
  // populate the container with mesh in space
  discretizeSpace(h, (spacer_.lower() + h), prevSol);
  // use the mesh in space to get values of initial condition
  discretizeInitialCondition(init_, prevSol);
  // first create and populate the sparse matrix:
  FlatMatrix<T> fsm;
  fsm.setColumns(m);
  fsm.setRows(m);
  // populate the matrix:
  fsm.emplace_back(0, 0, (1.0 + (2.0 * lambda - delta) * theta));
  fsm.emplace_back(0, 1, -1.0 * (lambda + gamma) * theta);
  for (std::size_t t = 1; t < m - 1; ++t) {
    fsm.emplace_back(t, t - 1, -1.0 * (lambda - gamma) * theta);
    fsm.emplace_back(t, t, (1.0 + (2.0 * lambda - delta) * theta));
    fsm.emplace_back(t, t + 1, -1.0 * (lambda + gamma) * theta);
  }
  fsm.emplace_back(m - 1, m - 2, -1.0 * (lambda - gamma) * theta);
  fsm.emplace_back(m - 1, m - 1, (1.0 + (2.0 * lambda - delta) * theta));
  Container<T, Alloc> rhs(m, T{});
  // create container to carry new solution:
  Container<T, Alloc> nextSol(m, T{});
  // create first time point:
  T time = k;
  // store terminal time:
  T const lastTime = terminalT_;
  // initialize the solver:
  solverPtr_->initialize(m);
  // insert sparse matrix A and vector b:
  solverPtr_->setFlatSparseMatrix(std::move(fsm));
  if (isSourceSet_) {
    // wrap the scheme coefficients:
    const auto schemeCoeffs = std::make_tuple(lambda, gamma, delta, k);
    // get the correct scheme:
    auto schemeFun = ImplicitHeatEquationSchemes<T>::getInhomScheme(
        BoundaryConditionType::Dirichlet, scheme);
    // create a container to carry discretized source heat
    Container<T, Alloc> sourceCurr(m, T{});
    Container<T, Alloc> sourceNext(m, T{});
    discretizeInSpace(h, (spacer_.lower() + h), 0.0, source_, sourceCurr);
    discretizeInSpace(h, (spacer_.lower() + h), time, source_, sourceNext);
    // loop for stepping in time:
    while (time <= lastTime) {
      schemeFun(schemeCoeffs, prevSol, sourceCurr, sourceNext, rhs,
                std::make_pair(boundary_.first(time), boundary_.second(time)),
                std::pair<T, T>());
      solverPtr_->setRhs(rhs);
      solverPtr_->solve(nextSol);
      prevSol = nextSol;
      discretizeInSpace(h, (spacer_.lower() + h), time, source_, sourceCurr);
      discretizeInSpace(h, (spacer_.lower() + h), 2.0 * time, source_,
                        sourceNext);
      time += k;
    }
  } else {
    // wrap the scheme coefficients:
    const auto schemeCoeffs = std::make_tuple(lambda, gamma, delta, T{});
    // get the correct scheme:
    auto schemeFun = ImplicitHeatEquationSchemes<T>::getScheme(
        BoundaryConditionType::Dirichlet, scheme);
    // loop for stepping in time:
    while (time <= lastTime) {
      schemeFun(schemeCoeffs, prevSol, Container<T, Alloc>(),
                Container<T, Alloc>(), rhs,
                std::make_pair(boundary_.first(time), boundary_.second(time)),
                std::pair<T, T>());
      solverPtr_->setRhs(rhs);
      solverPtr_->solve(nextSol);
      prevSol = nextSol;
      time += k;
    }
  }

  // copy into solution vector
  solution[0] = boundary_.first(lastTime);
  std::copy(prevSol.begin(), prevSol.end(), std::next(solution.begin()));
  solution[solution.size() - 1] = boundary_.second(lastTime);
}

// ============================================================================
// ======= Implicit1DGeneralHeatEquationCUDA (Robin BC) implementation ========
// ============================================================================

template <typename T, MemorySpace MemSpace,
          template <MemorySpace, typename> typename RealSparsePolicyCUDA,
          template <typename, typename> typename Container, typename Alloc>
void implicit_solvers::Implicit1DGeneralHeatEquationCUDA<
    T, BoundaryConditionType::Robin, MemSpace, RealSparsePolicyCUDA, Container,
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
  // store size of matrix:
  std::size_t const m = spaceN_ + 1;
  // create container to carry mesh in space and then previous solution:
  Container<T, Alloc> prevSol(m, T{});
  // populate the container with mesh in space
  discretizeSpace(h, spacer_.lower(), prevSol);
  // use the mesh in space to get values of initial condition
  discretizeInitialCondition(init_, prevSol);
  // first create and populate the sparse matrix:
  FlatMatrix<T> fsm;
  fsm.setColumns(m);
  fsm.setRows(m);
  // populate the matrix:
  fsm.emplace_back(0, 0, (1.0 + (2.0 * lambda - delta) * theta));
  fsm.emplace_back(
      0, 1,
      -1.0 * ((lambda + gamma) + (lambda - gamma) * leftBoundary_.first) *
          theta);
  for (std::size_t t = 1; t < m - 1; ++t) {
    fsm.emplace_back(t, t - 1, -1.0 * (lambda - gamma) * theta);
    fsm.emplace_back(t, t, (1.0 + (2.0 * lambda - delta) * theta));
    fsm.emplace_back(t, t + 1, -1.0 * (lambda + gamma) * theta);
  }
  fsm.emplace_back(
      m - 1, m - 2,
      -1.0 * ((lambda - gamma) + (lambda + gamma) * rightBoundary_.first) *
          theta);
  fsm.emplace_back(m - 1, m - 1, (1.0 + (2.0 * lambda - delta) * theta));
  Container<T, Alloc> rhs(m, T{});
  // create container to carry new solution:
  Container<T, Alloc> nextSol(m, T{});
  // create first time point:
  T time = k;
  // store terminal time:
  T const lastTime = terminalT_;
  // initialize the solver:
  solverPtr_->initialize(m);
  // insert sparse matrix A and vector b:
  solverPtr_->setFlatSparseMatrix(std::move(fsm));
  // differentiate between inhomogeneous and homogeneous PDE:
  if (isSourceSet_) {
    // wrap the scheme coefficients:
    const auto schemeCoeffs = std::make_tuple(lambda, gamma, delta, k);
    // get the correct scheme:
    auto schemeFun = ImplicitHeatEquationSchemes<T>::getInhomScheme(
        BoundaryConditionType::Robin, scheme);
    // create a container to carry discretized source heat
    Container<T, Alloc> sourceCurr(m, T{});
    Container<T, Alloc> sourceNext(m, T{});
    discretizeInSpace(h, (spacer_.lower() + h), 0.0, source_, sourceCurr);
    discretizeInSpace(h, (spacer_.lower() + h), time, source_, sourceNext);
    // loop for stepping in time:
    while (time <= lastTime) {
      schemeFun(schemeCoeffs, prevSol, sourceCurr, sourceNext, rhs,
                leftBoundary_, rightBoundary_);
      solverPtr_->setRhs(rhs);
      solverPtr_->solve(nextSol);
      prevSol = nextSol;
      discretizeInSpace(h, (spacer_.lower() + h), time, source_, sourceCurr);
      discretizeInSpace(h, (spacer_.lower() + h), 2.0 * time, source_,
                        sourceNext);
      time += k;
    }
  } else {
    // wrap the scheme coefficients:
    const auto schemeCoeffs = std::make_tuple(lambda, gamma, delta, T{});
    // get the correct scheme:
    auto schemeFun = ImplicitHeatEquationSchemes<T>::getScheme(
        BoundaryConditionType::Robin, scheme);
    // loop for stepping in time:
    while (time <= lastTime) {
      schemeFun(schemeCoeffs, prevSol, Container<T, Alloc>(),
                Container<T, Alloc>(), rhs, leftBoundary_, rightBoundary_);
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
// ===== Explicit1DGeneralHeatEquationCUDA (Dirichlet BC) implementation ======
// ============================================================================

template <typename T, template <typename, typename> typename Container,
          typename Alloc>
bool explicit_solvers::Explicit1DGeneralHeatEquationCUDA<
    T, BoundaryConditionType::Dirichlet, Container, Alloc>::isStable() const {
  T const k = timeStep();
  T const h = spaceStep();
  T const A = std::get<0>(coeffs_);
  T const B = std::get<1>(coeffs_);

  return (((2.0 * A * k / (h * h)) <= 1.0) && (B * (k / h) <= 1.0));
}

template <typename T, template <typename, typename> typename Container,
          typename Alloc>
void explicit_solvers::Explicit1DGeneralHeatEquationCUDA<
    T, BoundaryConditionType::Dirichlet, Container,
    Alloc>::solve(Container<T, Alloc> &solution) {
  LSS_ASSERT(isStable() == true, "This discretization is not stable.");
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");
  // get space step:
  T const h = spaceStep();
  // get time step:
  T const k = timeStep();
  // space start:
  T const spaceStart = spacer_.lower();
  // create container to carry mesh in space and then previous solution:
  Container<T, Alloc> prevSol(spaceN_ + 1, T{});
  // populate the container with mesh in space
  discretizeSpace(h, spaceStart, prevSol);
  // use the mesh in space to get values of initial condition
  discretizeInitialCondition(init_, prevSol);

  ExplicitEulerHeatEquationScheme<T, Container, Alloc> eulerScheme(
      spaceStart, terminalT_, std::make_pair(k, h), coeffs_, prevSol, source_,
      isSourceSet_);
  eulerScheme(boundary_, solution);
}

// ============================================================================
// ======== Explicit1DGeneralHeatEquationCUDA (Robin BC) implementation =======
// ============================================================================

template <typename T, template <typename, typename> typename Container,
          typename Alloc>
bool explicit_solvers::Explicit1DGeneralHeatEquationCUDA<
    T, BoundaryConditionType::Robin, Container, Alloc>::isStable() const {
  T const k = timeStep();
  T const h = spaceStep();
  T const A = std::get<0>(coeffs_);
  T const B = std::get<1>(coeffs_);

  return (((2.0 * A * k / (h * h)) <= 1.0) && (B * (k / h) <= 1.0));
}

template <typename T, template <typename, typename> typename Container,
          typename Alloc>
void explicit_solvers::Explicit1DGeneralHeatEquationCUDA<
    T, BoundaryConditionType::Robin, Container,
    Alloc>::solve(Container<T, Alloc> &solution) {
  LSS_ASSERT(isStable() == true, "This discretization is not stable.");
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");
  // get space step:
  T const h = spaceStep();
  // get time step:
  T const k = timeStep();
  // space start:
  T const spaceStart = spacer_.lower();
  // create container to carry mesh in space and then previous solution:
  Container<T, Alloc> prevSol(spaceN_ + 1, T{});
  // populate the container with mesh in space
  discretizeSpace(h, spaceStart, prevSol);
  // use the mesh in space to get values of initial condition
  discretizeInitialCondition(init_, prevSol);

  ExplicitEulerHeatEquationScheme<T, Container, Alloc> eulerScheme(
      spaceStart, terminalT_, std::make_pair(k, h), coeffs_, prevSol, source_,
      isSourceSet_);
  eulerScheme(leftBoundary_, rightBoundary_, solution);
}

}  // namespace lss_one_dim_general_heat_equation_solvers_cuda

#endif  ///_LSS_ONE_DIM_GENERAL_HEAT_EQUATION_SOLVERS_CUDA
