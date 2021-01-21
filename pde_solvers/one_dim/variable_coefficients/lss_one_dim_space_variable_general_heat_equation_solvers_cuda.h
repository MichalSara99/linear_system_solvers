#pragma once
#if !defined(_LSS_ONE_DIM_SPACE_VARIABLE_GENERAL_HEAT_EQUATION_SOLVERS_CUDA)
#define _LSS_ONE_DIM_SPACE_VARIABLE_GENERAL_HEAT_EQUATION_SOLVERS_CUDA

#include "common/lss_enumerations.h"
#include "common/lss_utility.h"
#include "lss_one_dim_space_variable_heat_explicit_schemes_cuda.h"
#include "lss_one_dim_space_variable_heat_implicit_schemes_cuda.h"
#include "pde_solvers/one_dim/lss_one_dim_pde_utility.h"
#include "sparse_solvers/lss_sparse_solvers_cuda.h"

namespace lss_one_dim_space_variable_general_heat_equation_solvers_cuda {

using lss_enumerations::BoundaryConditionType;
using lss_enumerations::ImplicitPDESchemes;
using lss_enumerations::MemorySpace;
using lss_one_dim_pde_utility::DirichletBoundary;
using lss_one_dim_pde_utility::Discretization;
using lss_one_dim_pde_utility::HeatData;
using lss_one_dim_pde_utility::PDECoefficientHolderFun1Arg;
using lss_one_dim_pde_utility::RobinBoundary;
using lss_one_dim_space_variable_heat_explicit_schemes_cuda::
    ExplicitEulerHeatEquationScheme;
using lss_one_dim_space_variable_heat_implicit_schemes_cuda::
    ImplicitSpaceVariableHeatEquationSchemesCUDA;
using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
using lss_utility::FlatMatrix;
using lss_utility::Range;
using lss_utility::uptr_t;

namespace implicit_solvers {

// ============================================================================
// ===== Implicit1DSpaceVariableGeneralHeatEquationCUDA General Template ======
// ============================================================================

template <typename T, BoundaryConditionType BType, MemorySpace MemSpace,
          template <MemorySpace, typename> typename RealSparsePolicyCUDA,
          template <typename, typename> typename Container, typename Alloc>
class Implicit1DSpaceVariableGeneralHeatEquationCUDA {};

// ============================================================================
// =============== Implicit1DSpaceVariableGeneralHeatEquationCUDA =============
// ===================== Dirichlet Specialisation Template ====================
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

template <typename T, MemorySpace MemSpace,
          template <MemorySpace, typename> typename RealSparsePolicyCUDA,
          template <typename, typename> typename Container, typename Alloc>
class Implicit1DSpaceVariableGeneralHeatEquationCUDA<
    T, BoundaryConditionType::Dirichlet, MemSpace, RealSparsePolicyCUDA,
    Container, Alloc> : public Discretization<T, Container, Alloc> {
 private:
  typedef RealSparsePolicyCUDA<MemSpace, T> cuda_solver_t;
  typedef HeatData<T> heat_data_t;

  uptr_t<cuda_solver_t> solverPtr_;        // finite-difference solver
  uptr_t<heat_data_t> dataPtr_;            // one-dim heat data
  DirichletBoundary<T> boundary_;          // boundaries
  PDECoefficientHolderFun1Arg<T> coeffs_;  // coefficients of PDE

 public:
  typedef T value_type;
  explicit Implicit1DSpaceVariableGeneralHeatEquationCUDA() = delete;
  explicit Implicit1DSpaceVariableGeneralHeatEquationCUDA(
      Range<T> const &spaceRange, T terminalTime,
      std::size_t const &spaceDiscretization,
      std::size_t const &timeDiscretization)
      : solverPtr_{std::make_unique<cuda_solver_t>()},
        dataPtr_{std::make_unique<heat_data_t>(
            spaceRange, Range<T>(T{}, terminalTime), spaceDiscretization,
            timeDiscretization, nullptr, nullptr, nullptr, false)} {}

  ~Implicit1DSpaceVariableGeneralHeatEquationCUDA() {}

  Implicit1DSpaceVariableGeneralHeatEquationCUDA(
      Implicit1DSpaceVariableGeneralHeatEquationCUDA const &) = delete;
  Implicit1DSpaceVariableGeneralHeatEquationCUDA(
      Implicit1DSpaceVariableGeneralHeatEquationCUDA &&) = delete;
  Implicit1DSpaceVariableGeneralHeatEquationCUDA &operator=(
      Implicit1DSpaceVariableGeneralHeatEquationCUDA const &) = delete;
  Implicit1DSpaceVariableGeneralHeatEquationCUDA &operator=(
      Implicit1DSpaceVariableGeneralHeatEquationCUDA &&) = delete;

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
// ================ Implicit1DSpaceVariableGeneralHeatEquationCUDA ============
// ====================== Robin Specialisation Template =======================
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
class Implicit1DSpaceVariableGeneralHeatEquationCUDA<
    T, BoundaryConditionType::Robin, MemSpace, RealSparsePolicyCUDA, Container,
    Alloc> : public Discretization<T, Container, Alloc> {
 private:
  typedef RealSparsePolicyCUDA<MemSpace, T> cuda_solver_t;
  typedef HeatData<T> heat_data_t;

  uptr_t<cuda_solver_t> solverPtr_;        // finite-difference solver
  uptr_t<heat_data_t> dataPtr_;            // one-dim heat data
  RobinBoundary<T> boundary_;              // Robin boundary
  PDECoefficientHolderFun1Arg<T> coeffs_;  // coefficients of PDE
  void transformRobinBC(RobinBoundary<T> const &boundary);

 public:
  typedef T value_type;
  explicit Implicit1DSpaceVariableGeneralHeatEquationCUDA() = delete;
  explicit Implicit1DSpaceVariableGeneralHeatEquationCUDA(
      Range<T> const &spaceRange, T terminalTime,
      std::size_t const &spaceDiscretization,
      std::size_t const &timeDiscretization)
      : solverPtr_{std::make_unique<cuda_solver_t>()},
        dataPtr_{std::make_unique<heat_data_t>(
            spaceRange, Range<T>(T{}, terminalTime), spaceDiscretization,
            timeDiscretization, nullptr, nullptr, nullptr, false)} {}

  ~Implicit1DSpaceVariableGeneralHeatEquationCUDA() {}

  Implicit1DSpaceVariableGeneralHeatEquationCUDA(
      Implicit1DSpaceVariableGeneralHeatEquationCUDA const &) = delete;
  Implicit1DSpaceVariableGeneralHeatEquationCUDA(
      Implicit1DSpaceVariableGeneralHeatEquationCUDA &&) = delete;
  Implicit1DSpaceVariableGeneralHeatEquationCUDA &operator=(
      Implicit1DSpaceVariableGeneralHeatEquationCUDA const &) = delete;
  Implicit1DSpaceVariableGeneralHeatEquationCUDA &operator=(
      Implicit1DSpaceVariableGeneralHeatEquationCUDA &&) = delete;

  inline T spaceStep() const {
    return ((dataPtr_->spaceRange.spread()) /
            static_cast<T>(dataPtr_->spaceDivision));
  }
  inline T timeStep() const {
    return ((dataPtr_->timeRange.upper()) /
            static_cast<T>(dataPtr_->timeDivision));
  }
  inline void setBoundaryCondition(RobinBoundary<T> const &boundary) {
    transformRobinBC(boundary);
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
// ===== Explicit1DSpaceVariableGeneralHeatEquationCUDA General Template ======
// ============================================================================

template <typename T, BoundaryConditionType BType,
          template <typename, typename> typename Container, typename Alloc>
class Explicit1DSpaceVariableGeneralHeatEquationCUDA {};

// ============================================================================
// ============== Explicit1DSpaceVariableGeneralHeatEquationCUDA ==============
// ==================== Dirichlet Specialisation Template =====================
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
class Explicit1DSpaceVariableGeneralHeatEquationCUDA<
    T, BoundaryConditionType::Dirichlet, Container, Alloc>
    : public Discretization<T, Container, Alloc> {
 private:
  typedef HeatData<T> heat_data_t;

  uptr_t<heat_data_t> dataPtr_;            // one-dim heat data
  DirichletBoundary<T> boundary_;          // boundaries
  PDECoefficientHolderFun1Arg<T> coeffs_;  // coefficients of PDE

 public:
  typedef T value_type;
  explicit Explicit1DSpaceVariableGeneralHeatEquationCUDA() = delete;
  explicit Explicit1DSpaceVariableGeneralHeatEquationCUDA(
      Range<T> const &spaceRange, T terminalTime,
      std::size_t const &spaceDiscretization,
      std::size_t const &timeDiscretization)
      : dataPtr_{std::make_unique<heat_data_t>(
            spaceRange, Range<T>(T{}, terminalTime), spaceDiscretization,
            timeDiscretization, nullptr, nullptr, nullptr, false)} {}

  ~Explicit1DSpaceVariableGeneralHeatEquationCUDA() {}

  Explicit1DSpaceVariableGeneralHeatEquationCUDA(
      Explicit1DSpaceVariableGeneralHeatEquationCUDA const &) = delete;
  Explicit1DSpaceVariableGeneralHeatEquationCUDA(
      Explicit1DSpaceVariableGeneralHeatEquationCUDA &&) = delete;
  Explicit1DSpaceVariableGeneralHeatEquationCUDA &operator=(
      Explicit1DSpaceVariableGeneralHeatEquationCUDA const &) = delete;
  Explicit1DSpaceVariableGeneralHeatEquationCUDA &operator=(
      Explicit1DSpaceVariableGeneralHeatEquationCUDA &&) = delete;

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
  // stability check:
  bool isStable() const;

  void solve(Container<T, Alloc> &solution);
};

// ============================================================================
// =============== Explicit1DSpaceVariableGeneralHeatEquationCUDA =============
// ======================= Robin Specialisation Template ======================
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
class Explicit1DSpaceVariableGeneralHeatEquationCUDA<
    T, BoundaryConditionType::Robin, Container, Alloc>
    : public Discretization<T, Container, Alloc> {
 private:
  typedef HeatData<T> heat_data_t;

  uptr_t<heat_data_t> dataPtr_;            // one-dim heat data
  RobinBoundary<T> boundary_;              // Robin boundary
  PDECoefficientHolderFun1Arg<T> coeffs_;  // coefficients of PDE
  void transformRobinBC(RobinBoundary<T> const &boundary);

 public:
  typedef T value_type;
  explicit Explicit1DSpaceVariableGeneralHeatEquationCUDA() = delete;
  explicit Explicit1DSpaceVariableGeneralHeatEquationCUDA(
      Range<T> const &spaceRange, T terminalTime,
      std::size_t const &spaceDiscretization,
      std::size_t const &timeDiscretization)
      : dataPtr_{std::make_unique<heat_data_t>(
            spaceRange, Range<T>(T{}, terminalTime), spaceDiscretization,
            timeDiscretization, nullptr, nullptr, nullptr, false)} {}

  ~Explicit1DSpaceVariableGeneralHeatEquationCUDA() {}

  Explicit1DSpaceVariableGeneralHeatEquationCUDA(
      Explicit1DSpaceVariableGeneralHeatEquationCUDA const &) = delete;
  Explicit1DSpaceVariableGeneralHeatEquationCUDA(
      Explicit1DSpaceVariableGeneralHeatEquationCUDA &&) = delete;
  Explicit1DSpaceVariableGeneralHeatEquationCUDA &operator=(
      Explicit1DSpaceVariableGeneralHeatEquationCUDA const &) = delete;
  Explicit1DSpaceVariableGeneralHeatEquationCUDA &operator=(
      Explicit1DSpaceVariableGeneralHeatEquationCUDA &&) = delete;

  inline T spaceStep() const {
    return ((dataPtr_->spaceRange.spread()) /
            static_cast<T>(dataPtr_->spaceDivision));
  }
  inline T timeStep() const {
    return ((dataPtr_->timeRange.upper()) /
            static_cast<T>(dataPtr_->timeDivision));
  }
  inline void setBoundaryCondition(RobinBoundary<T> const &boundary) {
    transformRobinBC(boundary);
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

  // stability check:
  bool isStable() const;

  void solve(Container<T, Alloc> &solution);
};
}  // namespace explicit_solvers

// ============================================================================
// ============================= IMPLEMENTATIONS ==============================

// ============================================================================
// = Implicit1DSpaceVariableGeneralHeatEquationCUDA (Dirichlet) implementation
// ============================================================================

template <typename T, MemorySpace MemSpace,
          template <MemorySpace, typename> typename RealSparsePolicyCUDA,
          template <typename, typename> typename Container, typename Alloc>
void implicit_solvers::Implicit1DSpaceVariableGeneralHeatEquationCUDA<
    T, BoundaryConditionType::Dirichlet, MemSpace, RealSparsePolicyCUDA,
    Container, Alloc>::solve(Container<T, Alloc> &solution,
                             ImplicitPDESchemes scheme) {
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");
  // get correct theta according to the scheme:
  T const theta =
      ImplicitSpaceVariableHeatEquationSchemesCUDA<T>::getTheta(scheme);
  // get space step:
  T const h = spaceStep();
  // get time step:
  T const k = timeStep();
  // get space range:
  auto const &spaceRange = dataPtr_->spaceRange;
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
  // store size of matrix:
  std::size_t const m = dataPtr_->spaceDivision - 1;
  // create container to carry mesh in space and then previous solution:
  Container<T, Alloc> prevSol(m, T{});
  // populate the container with mesh in space
  discretizeSpace(h, (spaceRange.lower() + h), prevSol);
  // use the mesh in space to get values of initial condition
  discretizeInitialCondition(dataPtr_->initialCondition, prevSol);
  // first create and populate the sparse matrix:
  FlatMatrix<T> fsm;
  fsm.setColumns(m);
  fsm.setRows(m);
  // prepare space variable coefficients:
  auto const &A = [&](T x) { return (lambda * a(x) - gamma * b(x)); };
  auto const &B = [&](T x) { return (lambda * a(x) - delta * c(x)); };
  auto const &D = [&](T x) { return (lambda * a(x) + gamma * b(x)); };
  // populate the matrix:
  fsm.emplace_back(0, 0, (1.0 + 2.0 * B(1 * h) * theta));
  fsm.emplace_back(0, 1, (-1.0 * D(1 * h) * theta));
  for (std::size_t t = 1; t < m - 1; ++t) {
    fsm.emplace_back(t, t - 1, (-1.0 * A((t + 1) * h) * theta));
    fsm.emplace_back(t, t, (1.0 + 2.0 * B((t + 1) * h) * theta));
    fsm.emplace_back(t, t + 1, (-1.0 * D((t + 1) * h) * theta));
  }
  fsm.emplace_back(m - 1, m - 2, (-1.0 * A(m * h) * theta));
  fsm.emplace_back(m - 1, m - 1, (1.0 + 2.0 * B(m * h) * theta));
  Container<T, Alloc> rhs(m, T{});
  // create container to carry new solution:
  Container<T, Alloc> nextSol(m, T{});
  // create first time point:
  T time = k;
  // store terminal time:
  T const lastTime = dataPtr_->timeRange.upper();
  // initialize the solver:
  solverPtr_->initialize(m);
  // insert sparse matrix A and vector b:
  solverPtr_->setFlatSparseMatrix(std::move(fsm));
  if ((dataPtr_->isSourceFunctionSet)) {
    // wrap the scheme coefficients:
    const auto schemeCoeffs = std::make_tuple(A, B, D, h, k);
    // get the correct scheme:
    auto schemeFun =
        ImplicitSpaceVariableHeatEquationSchemesCUDA<T>::getInhomScheme(
            BoundaryConditionType::Dirichlet, scheme);
    // create a container to carry discretized source heat
    Container<T, Alloc> sourceCurr(m, T{});
    Container<T, Alloc> sourceNext(m, T{});
    discretizeInSpace(h, (spaceRange.lower() + h), 0.0, heatSource, sourceCurr);
    discretizeInSpace(h, (spaceRange.lower() + h), time, heatSource,
                      sourceNext);
    // loop for stepping in time:
    while (time <= lastTime) {
      schemeFun(schemeCoeffs, prevSol, sourceCurr, sourceNext, rhs,
                std::make_pair(boundary_.first(time), boundary_.second(time)),
                std::pair<T, T>());
      solverPtr_->setRhs(rhs);
      solverPtr_->solve(nextSol);
      prevSol = nextSol;
      discretizeInSpace(h, (spaceRange.lower() + h), time, heatSource,
                        sourceCurr);
      discretizeInSpace(h, (spaceRange.lower() + h), 2.0 * time, heatSource,
                        sourceNext);
      time += k;
    }
  } else {
    // wrap the scheme coefficients:
    const auto schemeCoeffs = std::make_tuple(A, B, D, h, T{});
    // get the correct scheme:
    auto schemeFun = ImplicitSpaceVariableHeatEquationSchemesCUDA<T>::getScheme(
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
// = Implicit1DSpaceVariableGeneralHeatEquationCUDA (Robin BC) implementation =
// ============================================================================

template <typename T, MemorySpace MemSpace,
          template <MemorySpace, typename> typename RealSparsePolicyCUDA,
          template <typename, typename> typename Container, typename Alloc>
void implicit_solvers::Implicit1DSpaceVariableGeneralHeatEquationCUDA<
    T, BoundaryConditionType::Robin, MemSpace, RealSparsePolicyCUDA, Container,
    Alloc>::transformRobinBC(RobinBoundary<T> const &boundary) {
  auto const &beta_ = static_cast<T>(1.0) / boundary.right.first;
  auto const &psi_ =
      static_cast<T>(-1.0) * boundary.right.second / boundary.right.first;
  boundary_.left = boundary.left;
  boundary_.right = std::make_pair(beta_, psi_);
}

template <typename T, MemorySpace MemSpace,
          template <MemorySpace, typename> typename RealSparsePolicyCUDA,
          template <typename, typename> typename Container, typename Alloc>
void implicit_solvers::Implicit1DSpaceVariableGeneralHeatEquationCUDA<
    T, BoundaryConditionType::Robin, MemSpace, RealSparsePolicyCUDA, Container,
    Alloc>::solve(Container<T, Alloc> &solution, ImplicitPDESchemes scheme) {
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");
  // get correct theta according to the scheme:
  T const theta =
      ImplicitSpaceVariableHeatEquationSchemesCUDA<T>::getTheta(scheme);
  // get space step:
  T const h = spaceStep();
  // get time step:
  T const k = timeStep();
  // get space range:
  auto const &spaceRange = dataPtr_->spaceRange;
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
  // store size of matrix:
  std::size_t const m = dataPtr_->spaceDivision + 1;
  // create container to carry mesh in space and then previous solution:
  Container<T, Alloc> prevSol(m, T{});
  // populate the container with mesh in space
  discretizeSpace(h, spaceRange.lower(), prevSol);
  // use the mesh in space to get values of initial condition
  discretizeInitialCondition(dataPtr_->initialCondition, prevSol);
  // first create and populate the sparse matrix:
  FlatMatrix<T> fsm;
  fsm.setColumns(m);
  fsm.setRows(m);
  // prepare space variable coefficients:
  auto const &A = [&](T x) { return (lambda * a(x) - gamma * b(x)); };
  auto const &B = [&](T x) { return (lambda * a(x) - delta * c(x)); };
  auto const &D = [&](T x) { return (lambda * a(x) + gamma * b(x)); };
  // populate the matrix:
  fsm.emplace_back(0, 0, (1.0 + 2.0 * B(0 * h) * theta));
  fsm.emplace_back(
      0, 1, (-1.0 * (boundary_.left.first * A(0 * h) + D(0 * h)) * theta));
  for (std::size_t t = 1; t < m - 1; ++t) {
    fsm.emplace_back(t, t - 1, (-1.0 * A(t * h) * theta));
    fsm.emplace_back(t, t, (1.0 + 2.0 * B(t * h) * theta));
    fsm.emplace_back(t, t + 1, (-1.0 * D(t * h) * theta));
  }
  fsm.emplace_back(
      m - 1, m - 2,
      (-1.0 * (A((m - 1) * h) + boundary_.right.first * D((m - 1) * h)) *
       theta));
  fsm.emplace_back(m - 1, m - 1, (1.0 + 2.0 * B((m - 1) * h) * theta));

  Container<T, Alloc> rhs(m, T{});
  // create container to carry new solution:
  Container<T, Alloc> nextSol(m, T{});
  // create first time point:
  T time = k;
  // store terminal time:
  T const lastTime = dataPtr_->timeRange.upper();
  // initialize the solver:
  solverPtr_->initialize(m);
  // insert sparse matrix A and vector b:
  solverPtr_->setFlatSparseMatrix(std::move(fsm));
  // differentiate between inhomogeneous and homogeneous PDE:
  if ((dataPtr_->isSourceFunctionSet)) {
    // wrap the scheme coefficients:
    const auto schemeCoeffs = std::make_tuple(A, B, D, h, k);
    // get the correct scheme:
    auto schemeFun =
        ImplicitSpaceVariableHeatEquationSchemesCUDA<T>::getInhomScheme(
            BoundaryConditionType::Robin, scheme);
    // create a container to carry discretized source heat
    Container<T, Alloc> sourceCurr(m, T{});
    Container<T, Alloc> sourceNext(m, T{});
    discretizeInSpace(h, (spaceRange.lower() + h), 0.0, heatSource, sourceCurr);
    discretizeInSpace(h, (spaceRange.lower() + h), time, heatSource,
                      sourceNext);
    // loop for stepping in time:
    while (time <= lastTime) {
      schemeFun(schemeCoeffs, prevSol, sourceCurr, sourceNext, rhs,
                boundary_.left, boundary_.right);
      solverPtr_->setRhs(rhs);
      solverPtr_->solve(nextSol);
      prevSol = nextSol;
      discretizeInSpace(h, (spaceRange.lower() + h), time, heatSource,
                        sourceCurr);
      discretizeInSpace(h, (spaceRange.lower() + h), 2.0 * time, heatSource,
                        sourceNext);
      time += k;
    }
  } else {
    // wrap the scheme coefficients:
    const auto schemeCoeffs = std::make_tuple(A, B, D, h, T{});
    // get the correct scheme:
    auto schemeFun = ImplicitSpaceVariableHeatEquationSchemesCUDA<T>::getScheme(
        BoundaryConditionType::Robin, scheme);
    // loop for stepping in time:
    while (time <= lastTime) {
      schemeFun(schemeCoeffs, prevSol, Container<T, Alloc>(),
                Container<T, Alloc>(), rhs, boundary_.left, boundary_.right);
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
// Explicit1DSpaceVariableGeneralHeatEquationCUDA (Dirichlet BC) implementation
// ============================================================================

template <typename T, template <typename, typename> typename Container,
          typename Alloc>
bool explicit_solvers::Explicit1DSpaceVariableGeneralHeatEquationCUDA<
    T, BoundaryConditionType::Dirichlet, Container, Alloc>::isStable() const {
  auto const &a = std::get<0>(coeffs_);
  auto const &b = std::get<1>(coeffs_);
  auto const &c = std::get<2>(coeffs_);
  T const k = timeStep();
  T const h = spaceStep();
  T const lambda = k / (h * h);
  T const gamma = k / h;

  const std::size_t spaceSize = dataPtr_->spaceDivision + 1;
  for (std::size_t i = 0; i < spaceSize; ++i) {
    if (c(i * h) > 0.0) return false;
    if ((2.0 * lambda * a(i * h) - k * c(i * h)) > 1.0) return false;
    if (((gamma * std::abs(b(i * h))) * (gamma * std::abs(b(i * h)))) >
        (2.0 * lambda * a(i * h)))
      return false;
  }
  return true;
}

template <typename T, template <typename, typename> typename Container,
          typename Alloc>
void explicit_solvers::Explicit1DSpaceVariableGeneralHeatEquationCUDA<
    T, BoundaryConditionType::Dirichlet, Container,
    Alloc>::solve(Container<T, Alloc> &solution) {
  LSS_ASSERT(isStable() == true, "This discretization is not stable.");
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");
  // get space step:
  T const h = spaceStep();
  // get time step:
  T const k = timeStep();
  // get space range:
  auto const &spaceRange = dataPtr_->spaceRange;
  // space start:
  T const spaceStart = spaceRange.lower();
  // create container to carry mesh in space and then previous solution:
  Container<T, Alloc> prevSol(dataPtr_->spaceDivision + 1, T{});
  // populate the container with mesh in space
  discretizeSpace(h, spaceStart, prevSol);
  // use the mesh in space to get values of initial condition
  discretizeInitialCondition(dataPtr_->initialCondition, prevSol);

  ExplicitEulerHeatEquationScheme<T, Container, Alloc> eulerScheme(
      spaceStart, dataPtr_->timeRange.upper(), std::make_pair(k, h), coeffs_,
      prevSol, dataPtr_->sourceFunction, dataPtr_->isSourceFunctionSet);
  eulerScheme(boundary_, solution);
}

// ============================================================================
// = Explicit1DSpaceVariableGeneralHeatEquationCUDA (Robin BC) implementation =
// ============================================================================

template <typename T, template <typename, typename> typename Container,
          typename Alloc>
void explicit_solvers::Explicit1DSpaceVariableGeneralHeatEquationCUDA<
    T, BoundaryConditionType::Robin, Container,
    Alloc>::transformRobinBC(RobinBoundary<T> const &boundary) {
  auto const &beta_ = static_cast<T>(1.0) / boundary.right.first;
  auto const &psi_ =
      static_cast<T>(-1.0) * boundary.right.second / boundary.right.first;
  boundary_.left = boundary.left;
  boundary_.right = std::make_pair(beta_, psi_);
}

template <typename T, template <typename, typename> typename Container,
          typename Alloc>
bool explicit_solvers::Explicit1DSpaceVariableGeneralHeatEquationCUDA<
    T, BoundaryConditionType::Robin, Container, Alloc>::isStable() const {
  auto const &a = std::get<0>(coeffs_);
  auto const &b = std::get<1>(coeffs_);
  auto const &c = std::get<2>(coeffs_);
  T const k = timeStep();
  T const h = spaceStep();
  T const lambda = k / (h * h);
  T const gamma = k / h;

  const std::size_t spaceSize = dataPtr_->spaceDivision + 1;
  for (std::size_t i = 0; i < spaceSize; ++i) {
    if (c(i * h) > 0.0) return false;
    if ((2.0 * lambda * a(i * h) - k * c(i * h)) > 1.0) return false;
    if (((gamma * std::abs(b(i * h))) * (gamma * std::abs(b(i * h)))) >
        (2.0 * lambda * a(i * h)))
      return false;
  }
  return true;
}

template <typename T, template <typename, typename> typename Container,
          typename Alloc>
void explicit_solvers::Explicit1DSpaceVariableGeneralHeatEquationCUDA<
    T, BoundaryConditionType::Robin, Container,
    Alloc>::solve(Container<T, Alloc> &solution) {
  LSS_ASSERT(isStable() == true, "This discretization is not stable.");
  LSS_ASSERT(solution.size() > 0,
             "The input solution container must be initialized.");
  // get space step:
  T const h = spaceStep();
  // get time step:
  T const k = timeStep();
  // get space range:
  auto const &spaceRange = dataPtr_->spaceRange;
  // space start:
  T const spaceStart = spaceRange.lower();
  // create container to carry mesh in space and then previous solution:
  Container<T, Alloc> prevSol(dataPtr_->spaceDivision + 1, T{});
  // populate the container with mesh in space
  discretizeSpace(h, spaceStart, prevSol);
  // use the mesh in space to get values of initial condition
  discretizeInitialCondition(dataPtr_->initialCondition, prevSol);

  ExplicitEulerHeatEquationScheme<T, Container, Alloc> eulerScheme(
      spaceStart, dataPtr_->timeRange.upper(), std::make_pair(k, h), coeffs_,
      prevSol, dataPtr_->sourceFunction, dataPtr_->isSourceFunctionSet);
  eulerScheme(boundary_.left, boundary_.right, solution);
}

}  // namespace lss_one_dim_space_variable_general_heat_equation_solvers_cuda

#endif  ///_LSS_ONE_DIM_SPACE_VARIABLE_GENERAL_HEAT_EQUATION_SOLVERS_CUDA
