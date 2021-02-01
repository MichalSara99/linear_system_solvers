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

using lss_enumerations::boundary_condition_enum;

// =============================================================================
// ================== fdm_double_sweep_solver_base =============================
// =============================================================================

template <typename T,
          template <typename T, typename allocator> typename container =
              std::vector,
          typename alloc = std::allocator<T>>
class fdm_double_sweep_solver_base {
 protected:
  std::size_t discretization_size_;
  container<T, alloc> a_, b_, c_, f_;
  container<T, alloc> L_, K_;

  // will be overriden and implemented in
  // concrete classes base on boundary conditions
  virtual void kernel(container<T, alloc>& solution) = 0;

 public:
  typedef T value_type;
  explicit fdm_double_sweep_solver_base() = delete;
  explicit fdm_double_sweep_solver_base(std::size_t discretization_size)
      : discretization_size_{discretization_size} {}

  virtual ~fdm_double_sweep_solver_base() {}

  void set_diagonals(container<T, alloc> lower_diagonal,
                     container<T, alloc> diagonal,
                     container<T, alloc> upper_diagonal);

  void set_rhs(container<T, alloc> const& rhs);

  void solve(container<T, alloc>& solution);

  container<T, alloc> const solve();
};

// =============================================================================
// =================== Concrete FDMDoubleSweepSolver ===========================
// =============================================================================

template <typename T, boundary_condition_enum BCType,
          template <typename T, typename allocator> typename container,
          typename alloc>
class fdm_double_sweep_solver
    : public fdm_double_sweep_solver_base<T, container, alloc> {
 protected:
  void kernel(container<T, alloc>& solution) override{};
};

// Double-Sweep Solver specialization for Dirichlet BC:
template <typename T,
          template <typename T, typename allocator> typename container,
          typename alloc>
class fdm_double_sweep_solver<T, boundary_condition_enum::Dirichlet, container,
                              alloc>
    : public fdm_double_sweep_solver_base<T, container, alloc> {
 private:
  std::pair<T, T> boundary_;

 public:
  typedef T value_type;
  explicit fdm_double_sweep_solver() = delete;
  explicit fdm_double_sweep_solver(std::size_t discretization_size)
      : fdm_double_sweep_solver_base<T, container, alloc>(discretization_size) {
  }

  fdm_double_sweep_solver(fdm_double_sweep_solver const&) = delete;
  fdm_double_sweep_solver& operator=(fdm_double_sweep_solver const&) = delete;
  fdm_double_sweep_solver(fdm_double_sweep_solver&&) = delete;
  fdm_double_sweep_solver& operator=(fdm_double_sweep_solver&&) = delete;

  ~fdm_double_sweep_solver() {}

  inline void set_boundary_condition(std::pair<T, T> const& boundary_pair) {
    boundary_ = boundary_pair;
  }

 protected:
  void kernel(container<T, alloc>& solution) override;
};

// Double-Sweep Solver specialization for Robin BC:
template <typename T,
          template <typename T, typename allocator> typename container,
          typename alloc>
class fdm_double_sweep_solver<T, boundary_condition_enum::Robin, container,
                              alloc>
    : public fdm_double_sweep_solver_base<T, container, alloc> {
 private:
  std::pair<T, T> left_;
  std::pair<T, T> right_;

 public:
  typedef T value_type;
  explicit fdm_double_sweep_solver() = delete;
  explicit fdm_double_sweep_solver(std::size_t discretization_size)
      : fdm_double_sweep_solver_base<T, container, alloc>(discretization_size) {
  }

  fdm_double_sweep_solver(fdm_double_sweep_solver const&) = delete;
  fdm_double_sweep_solver& operator=(fdm_double_sweep_solver const&) = delete;
  fdm_double_sweep_solver(fdm_double_sweep_solver&&) = delete;
  fdm_double_sweep_solver& operator=(fdm_double_sweep_solver&&) = delete;

  ~fdm_double_sweep_solver() {}

  inline void set_boundary_condition(std::pair<T, T> const& left,
                                     std::pair<T, T> const& right) {
    left_ = left;
    right_ = right;
  }

 protected:
  void kernel(container<T, alloc>& solution) override;
};

}  // namespace lss_fdm_double_sweep_solver

// =============================================================================
// ==================== DoubleSweepSolverBase implementation ===================

template <typename T, template <typename T, typename alloc> typename container,
          typename alloc>
void lss_fdm_double_sweep_solver::fdm_double_sweep_solver_base<
    T, container, alloc>::set_diagonals(container<T, alloc> lower_diagonal,
                                        container<T, alloc> diagonal,
                                        container<T, alloc> upper_diagonal) {
  LSS_ASSERT(lower_diagonal.size() == discretization_size_,
             "Inncorect size for lowerDiagonal");
  LSS_ASSERT(diagonal.size() == discretization_size_,
             "Inncorect size for diagonal");
  LSS_ASSERT(upper_diagonal.size() == discretization_size_,
             "Inncorect size for upperDiagonal");
  a_ = std::move(lower_diagonal);
  b_ = std::move(diagonal);
  c_ = std::move(upper_diagonal);
}

template <typename T, template <typename T, typename alloc> typename container,
          typename alloc>
void lss_fdm_double_sweep_solver::fdm_double_sweep_solver_base<
    T, container, alloc>::set_rhs(container<T, alloc> const& rhs) {
  LSS_ASSERT(rhs.size() == discretization_size_,
             "Inncorect size for right-hand side");
  f_ = rhs;
}

template <typename T, template <typename T, typename alloc> typename container,
          typename alloc>
void lss_fdm_double_sweep_solver::fdm_double_sweep_solver_base<
    T, container, alloc>::solve(container<T, alloc>& solution) {
  LSS_ASSERT(solution.size() == discretization_size_,
             "Incorrect size of solution container");
  kernel(solution);
}

template <typename T, template <typename T, typename alloc> typename container,
          typename alloc>
container<T, alloc> const
lss_fdm_double_sweep_solver::fdm_double_sweep_solver_base<T, container,
                                                          alloc>::solve() {
  container<T, alloc> solution(discretization_size_);
  kernel(solution);
  return solution;
}

// =============================================================================
// ==================== FDMDoubleSweepSolver implementation ====================

template <typename T, template <typename T, typename alloc> typename container,
          typename alloc>
void lss_fdm_double_sweep_solver::fdm_double_sweep_solver<
    T, lss_enumerations::boundary_condition_enum::Dirichlet, container,
    alloc>::kernel(container<T, alloc>& solution) {
  // clear coefficients:
  K_.clear();
  L_.clear();
  // resize coefficients:
  K_.resize(discretization_size_);
  L_.resize(discretization_size_);
  // init coefficients:
  K_[0] = boundary_.first;
  L_[0] = 0.0;

  T tmp{};
  for (std::size_t t = 1; t < discretization_size_; ++t) {
    tmp = b_[t] + (a_[t] * L_[t - 1]);
    L_[t] = -1.0 * c_[t] / tmp;
    K_[t] = (f_[t] - (a_[t] * K_[t - 1])) / tmp;
  }

  f_[discretization_size_ - 1] = boundary_.second;

  for (std::size_t t = discretization_size_ - 2; t >= 1; --t) {
    f_[t] = (L_[t] * f_[t + 1]) + K_[t];
  }

  f_[0] = boundary_.first;
  std::copy(f_.begin(), f_.end(), solution.begin());
}

template <typename T, template <typename T, typename alloc> typename container,
          typename alloc>
void lss_fdm_double_sweep_solver::fdm_double_sweep_solver<
    T, lss_enumerations::boundary_condition_enum::Robin, container,
    alloc>::kernel(container<T, alloc>& solution) {
  // clear coefficients:
  K_.clear();
  L_.clear();
  // resize coefficients:
  K_.resize(discretization_size_);
  L_.resize(discretization_size_);
  // init coefficients:
  L_[0] = left_.first;
  K_[0] = left_.second;

  T tmp{};
  for (std::size_t t = 1; t < discretization_size_; ++t) {
    tmp = b_[t] + (a_[t] * L_[t - 1]);
    L_[t] = -1.0 * c_[t] / tmp;
    K_[t] = (f_[t] - (a_[t] * K_[t - 1])) / tmp;
  }

  f_[discretization_size_ - 1] =
      ((K_[discretization_size_ - 1] - right_.second) /
       (right_.first - L_[discretization_size_ - 1]));

  for (std::size_t t = discretization_size_ - 2; t >= 1; --t) {
    f_[t] = (L_[t] * f_[t + 1]) + K_[t];
  }

  f_[0] = L_[0] * f_[1] + K_[0];
  std::copy(f_.begin(), f_.end(), solution.begin());
}

#endif  ///_LSS_FDM_DOUBLE_SWEEP_SOLVER
