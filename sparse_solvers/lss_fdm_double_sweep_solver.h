#pragma once
#if !defined(_LSS_FDM_DOUBLE_SWEEP_SOLVER)
#define _LSS_FDM_DOUBLE_SWEEP_SOLVER

#pragma warning(disable : 4244)

#include <type_traits>
#include <vector>

#include "common/lss_enumerations.h"
#include "common/lss_macros.h"
#include "common/lss_utility.h"

namespace lss_fdm_double_sweep_solver {

using lss_enumerations::BoundaryConditionType;

// =============================================================================
// ====================== FDMDoubleSweepSolverBase =============================
// =============================================================================

template <typename T,
          template <typename T, typename Allocator> typename Container =
              std::vector,
          typename Alloc = std::allocator<T>>
class FDMDoubleSweepSolverBase {
 protected:
  std::size_t discretizationSize_;
  Container<T, Alloc> a_, b_, c_, f_;
  Container<T, Alloc> L_, K_;

  // will be overriden and implemented in
  // concrete classes base on boundary conditions
  virtual void kernel(Container<T, Alloc>& solution) = 0;

 public:
  typedef T value_type;
  explicit FDMDoubleSweepSolverBase() = delete;
  explicit FDMDoubleSweepSolverBase(std::size_t discretizationSize)
      : discretizationSize_{discretizationSize} {}

  virtual ~FDMDoubleSweepSolverBase() {}

  void setDiagonals(Container<T, Alloc> lowerDiagonal,
                    Container<T, Alloc> diagonal,
                    Container<T, Alloc> upperDiagonal);

  void setRhs(Container<T, Alloc> const& rhs);

  void solve(Container<T, Alloc>& solution);

  Container<T, Alloc> const solve();
};

// =============================================================================
// =================== Concrete FDMDoubleSweepSolver ===========================
// =============================================================================

template <typename T, BoundaryConditionType BCType,
          template <typename T, typename Allocator> typename Container,
          typename Alloc>
class FDMDoubleSweepSolver
    : public FDMDoubleSweepSolverBase<T, Container, Alloc> {
 protected:
  void kernel(Container<T, Alloc>& solution) override{};
};

// Double-Sweep Solver specialization for Dirichlet BC:
template <typename T,
          template <typename T, typename Allocator> typename Container,
          typename Alloc>
class FDMDoubleSweepSolver<T, BoundaryConditionType::Dirichlet, Container,
                           Alloc>
    : public FDMDoubleSweepSolverBase<T, Container, Alloc> {
 private:
  std::pair<T, T> boundary_;

 public:
  typedef T value_type;
  explicit FDMDoubleSweepSolver() = delete;
  explicit FDMDoubleSweepSolver(std::size_t discretizationSize)
      : FDMDoubleSweepSolverBase<T, Container, Alloc>(discretizationSize) {}

  FDMDoubleSweepSolver(FDMDoubleSweepSolver const&) = delete;
  FDMDoubleSweepSolver& operator=(FDMDoubleSweepSolver const&) = delete;
  FDMDoubleSweepSolver(FDMDoubleSweepSolver&&) = delete;
  FDMDoubleSweepSolver& operator=(FDMDoubleSweepSolver&&) = delete;

  ~FDMDoubleSweepSolver() {}

  inline void setBoundaryCondition(std::pair<T, T> const& boundaryPair) {
    boundary_ = boundaryPair;
  }

 protected:
  void kernel(Container<T, Alloc>& solution) override;
};

// Double-Sweep Solver specialization for Robin BC:
template <typename T,
          template <typename T, typename Allocator> typename Container,
          typename Alloc>
class FDMDoubleSweepSolver<T, BoundaryConditionType::Robin, Container, Alloc>
    : public FDMDoubleSweepSolverBase<T, Container, Alloc> {
 private:
  std::pair<T, T> left_;
  std::pair<T, T> right_;

 public:
  typedef T value_type;
  explicit FDMDoubleSweepSolver() = delete;
  explicit FDMDoubleSweepSolver(std::size_t discretizationSize)
      : FDMDoubleSweepSolverBase<T, Container, Alloc>(discretizationSize) {}

  FDMDoubleSweepSolver(FDMDoubleSweepSolver const&) = delete;
  FDMDoubleSweepSolver& operator=(FDMDoubleSweepSolver const&) = delete;
  FDMDoubleSweepSolver(FDMDoubleSweepSolver&&) = delete;
  FDMDoubleSweepSolver& operator=(FDMDoubleSweepSolver&&) = delete;

  ~FDMDoubleSweepSolver() {}

  inline void setBoundaryCondition(std::pair<T, T> const& left,
                                   std::pair<T, T> const& right) {
    left_ = left;
    right_ = right;
  }

 protected:
  void kernel(Container<T, Alloc>& solution) override;
};

}  // namespace lss_fdm_double_sweep_solver

// =============================================================================
// ==================== DoubleSweepSolverBase implementation ===================

template <typename T, template <typename T, typename Alloc> typename Container,
          typename Alloc>
void lss_fdm_double_sweep_solver::FDMDoubleSweepSolverBase<
    T, Container, Alloc>::setDiagonals(Container<T, Alloc> lowerDiagonal,
                                       Container<T, Alloc> diagonal,
                                       Container<T, Alloc> upperDiagonal) {
  LSS_ASSERT(lowerDiagonal.size() == discretizationSize_,
             "Inncorect size for lowerDiagonal");
  LSS_ASSERT(diagonal.size() == discretizationSize_,
             "Inncorect size for diagonal");
  LSS_ASSERT(upperDiagonal.size() == discretizationSize_,
             "Inncorect size for upperDiagonal");
  a_ = std::move(lowerDiagonal);
  b_ = std::move(diagonal);
  c_ = std::move(upperDiagonal);
}

template <typename T, template <typename T, typename Alloc> typename Container,
          typename Alloc>
void lss_fdm_double_sweep_solver::FDMDoubleSweepSolverBase<
    T, Container, Alloc>::setRhs(Container<T, Alloc> const& rhs) {
  LSS_ASSERT(rhs.size() == discretizationSize_,
             "Inncorect size for right-hand side");
  f_ = rhs;
}

template <typename T, template <typename T, typename Alloc> typename Container,
          typename Alloc>
void lss_fdm_double_sweep_solver::FDMDoubleSweepSolverBase<
    T, Container, Alloc>::solve(Container<T, Alloc>& solution) {
  LSS_ASSERT(solution.size() == discretizationSize_,
             "Incorrect size of solution container");
  kernel(solution);
}

template <typename T, template <typename T, typename Alloc> typename Container,
          typename Alloc>
Container<T, Alloc> const lss_fdm_double_sweep_solver::FDMDoubleSweepSolverBase<
    T, Container, Alloc>::solve() {
  Container<T, Alloc> solution(discretizationSize_);
  kernel(solution);
  return solution;
}

// =============================================================================
// ==================== FDMDoubleSweepSolver implementation ====================

template <typename T, template <typename T, typename Alloc> typename Container,
          typename Alloc>
void lss_fdm_double_sweep_solver::FDMDoubleSweepSolver<
    T, lss_types::BoundaryConditionType::Dirichlet, Container,
    Alloc>::kernel(Container<T, Alloc>& solution) {
  // clear coefficients:
  K_.clear();
  L_.clear();
  // resize coefficients:
  K_.resize(discretizationSize_);
  L_.resize(discretizationSize_);
  // init coefficients:
  K_[0] = boundary_.first;
  L_[0] = 0.0;

  T tmp{};
  for (std::size_t t = 1; t < discretizationSize_; ++t) {
    tmp = b_[t] + (a_[t] * L_[t - 1]);
    L_[t] = -1.0 * c_[t] / tmp;
    K_[t] = (f_[t] - (a_[t] * K_[t - 1])) / tmp;
  }

  f_[discretizationSize_ - 1] = boundary_.second;

  for (std::size_t t = discretizationSize_ - 2; t >= 1; --t) {
    f_[t] = (L_[t] * f_[t + 1]) + K_[t];
  }

  f_[0] = boundary_.first;
  std::copy(f_.begin(), f_.end(), solution.begin());
}

template <typename T, template <typename T, typename Alloc> typename Container,
          typename Alloc>
void lss_fdm_double_sweep_solver::FDMDoubleSweepSolver<
    T, lss_types::BoundaryConditionType::Robin, Container,
    Alloc>::kernel(Container<T, Alloc>& solution) {
  // clear coefficients:
  K_.clear();
  L_.clear();
  // resize coefficients:
  K_.resize(discretizationSize_);
  L_.resize(discretizationSize_);
  // init coefficients:
  L_[0] = left_.first;
  K_[0] = left_.second;

  T tmp{};
  for (std::size_t t = 1; t < discretizationSize_; ++t) {
    tmp = b_[t] + (a_[t] * L_[t - 1]);
    L_[t] = -1.0 * c_[t] / tmp;
    K_[t] = (f_[t] - (a_[t] * K_[t - 1])) / tmp;
  }

  f_[discretizationSize_ - 1] = ((K_[discretizationSize_ - 1] - right_.second) /
                                 (right_.first - L_[discretizationSize_ - 1]));

  for (std::size_t t = discretizationSize_ - 2; t >= 1; --t) {
    f_[t] = (L_[t] * f_[t + 1]) + K_[t];
  }

  f_[0] = L_[0] * f_[1] + K_[0];
  std::copy(f_.begin(), f_.end(), solution.begin());
}

#endif  ///_LSS_FDM_DOUBLE_SWEEP_SOLVER
