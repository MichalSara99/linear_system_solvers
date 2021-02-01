#pragma once
#if !defined(_LSS_FDM_THOMAS_LU_SOLVER)
#define _LSS_FDM_THOMAS_LU_SOLVER

#include <type_traits>
#include <vector>

#include "common/lss_enumerations.h"
#include "common/lss_macros.h"

namespace lss_fdm_thomas_lu_solver {

using lss_enumerations::boundary_condition_enum;

// =============================================================================
// ========================== fdm_thomas_lu_solver_base ========================
// =============================================================================

template <typename T,
          template <typename T, typename allocator> typename container =
              std::vector,
          typename alloc = std::allocator<T>>
class fdm_thomas_lu_solver_base {
 protected:
  std::size_t system_size_, discretization_size_;
  container<T, alloc> a_, b_, c_, f_;
  container<T, alloc> beta_, gamma_;

  virtual void kernel(container<T, alloc>& solution) = 0;
  virtual bool is_diagonally_dominant() const = 0;

 public:
  typedef T value_type;
  explicit fdm_thomas_lu_solver_base() = delete;
  explicit fdm_thomas_lu_solver_base(std::size_t discretization_size)
      : discretization_size_{discretization_size},
        system_size_{discretization_size - 2} {
  }  // because we subtract the boundary values which are known{}

  virtual ~fdm_thomas_lu_solver_base() {}

  void set_diagonals(container<T, alloc> lower_diagonal,
                     container<T, alloc> diagonal,
                     container<T, alloc> upper_diagonal);

  void set_rhs(container<T, alloc> const& rhs);

  void solve(container<T, alloc>& solution);

  container<T, alloc> const solve();
};

// =============================================================================
// ==================== Concrete fdm_thomas_lu_solver
// =============================
// =============================================================================

template <typename T, boundary_condition_enum BCType,
          template <typename T, typename allocator> typename container,
          typename alloc>
class fdm_thomas_lu_solver {
 protected:
  void kernel(container<T, alloc>& solution) override {}
};

// Thomas LU Solver specialization for Dirichlet BC:
template <typename T,
          template <typename T, typename allocator> typename container,
          typename alloc>
class fdm_thomas_lu_solver<T, boundary_condition_enum::Dirichlet, container,
                           alloc>
    : public fdm_thomas_lu_solver_base<T, container, alloc> {
 private:
  std::pair<T, T> boundary_;

 public:
  typedef T value_type;
  explicit fdm_thomas_lu_solver() = delete;
  explicit fdm_thomas_lu_solver(std::size_t discretization_size)
      : fdm_thomas_lu_solver_base<T, container, alloc>(discretization_size) {}

  ~fdm_thomas_lu_solver() {}

  fdm_thomas_lu_solver(fdm_thomas_lu_solver const&) = delete;
  fdm_thomas_lu_solver(fdm_thomas_lu_solver&&) = delete;
  fdm_thomas_lu_solver& operator=(fdm_thomas_lu_solver const&) = delete;
  fdm_thomas_lu_solver& operator=(fdm_thomas_lu_solver&&) = delete;

  inline void set_boundary_condition(std::pair<T, T> const& boundary_pair) {
    boundary_ = boundary_pair;
  }

 protected:
  bool is_diagonally_dominant() const override;
  void kernel(container<T, alloc>& solution) override;
};

// Thomas LU Solver specialization for Robin BC:
template <typename T,
          template <typename T, typename allocator> typename container,
          typename alloc>
class fdm_thomas_lu_solver<T, boundary_condition_enum::Robin, container, alloc>
    : public fdm_thomas_lu_solver_base<T, container, alloc> {
 private:
  std::pair<T, T> left_;
  std::pair<T, T> right_;

 public:
  typedef T value_type;
  explicit fdm_thomas_lu_solver() = delete;
  explicit fdm_thomas_lu_solver(std::size_t discretization_size)
      : fdm_thomas_lu_solver_base<T, container, alloc>(discretization_size) {}

  ~fdm_thomas_lu_solver() {}

  fdm_thomas_lu_solver(fdm_thomas_lu_solver const&) = delete;
  fdm_thomas_lu_solver(fdm_thomas_lu_solver&&) = delete;
  fdm_thomas_lu_solver& operator=(fdm_thomas_lu_solver const&) = delete;
  fdm_thomas_lu_solver& operator=(fdm_thomas_lu_solver&&) = delete;

  inline void set_boundary_condition(std::pair<T, T> const& left,
                                     std::pair<T, T> const& right) {
    left_ = left;
    right_ = right;
  }

 protected:
  bool is_diagonally_dominant() const override;
  void kernel(container<T, alloc>& solution) override;
};

}  // namespace lss_fdm_thomas_lu_solver

// =============================================================================
// =================== fdm_thomas_lu_solver_base implementation
// ====================

template <typename T, template <typename T, typename alloc> typename container,
          typename alloc>
void lss_fdm_thomas_lu_solver::fdm_thomas_lu_solver_base<
    T, container, alloc>::set_rhs(container<T, alloc> const& rhs) {
  LSS_ASSERT(rhs.size() == discretization_size_,
             "Inncorect size for right-hand side");
  f_.clear();
  f_.resize(system_size_);
  for (std::size_t t = 0; t < system_size_; ++t) f_[t] = rhs[t + 1];
}

template <typename T, template <typename T, typename alloc> typename container,
          typename alloc>
void lss_fdm_thomas_lu_solver::fdm_thomas_lu_solver_base<
    T, container, alloc>::set_diagonals(container<T, alloc> lower_diagonal,
                                        container<T, alloc> diagonal,
                                        container<T, alloc> upper_diagonal) {
  LSS_ASSERT(lower_diagonal.size() == discretization_size_,
             "Inncorect size for lowerDiagonal");
  LSS_ASSERT(diagonal.size() == discretization_size_,
             "Inncorect size for diagonal");
  LSS_ASSERT(upper_diagonal.size() == discretization_size_,
             "Inncorect size for upperDiagonal");
  a_.clear();
  a_.resize(system_size_);
  b_.clear();
  b_.resize(system_size_);
  c_.clear();
  c_.resize(system_size_);
  for (std::size_t t = 0; t < system_size_; ++t) {
    a_[t] = std::move(lower_diagonal[t + 1]);
    b_[t] = std::move(diagonal[t + 1]);
    c_[t] = std::move(upper_diagonal[t + 1]);
  }

  // LSS_ASSERT(isDiagonallyDominant() == true,
  //	"Tridiagonal matrix must be diagonally dominant.");
}

template <typename T, template <typename T, typename alloc> typename container,
          typename alloc>
void lss_fdm_thomas_lu_solver::fdm_thomas_lu_solver_base<
    T, container, alloc>::solve(container<T, alloc>& solution) {
  LSS_ASSERT(solution.size() == discretization_size_,
             "Incorrect size of solution container");
  kernel(solution);
}

template <typename T, template <typename T, typename alloc> typename container,
          typename alloc>
container<T, alloc> const lss_fdm_thomas_lu_solver::fdm_thomas_lu_solver_base<
    T, container, alloc>::solve() {
  container<T, alloc> solution(discretization_size_);
  kernel(solution);
  return solution;
}

// =============================================================================
// ======================== ThomasLUSolver implementation ======================

template <typename T, template <typename T, typename alloc> typename container,
          typename alloc>
bool lss_fdm_thomas_lu_solver::fdm_thomas_lu_solver<
    T, lss_enumerations::boundary_condition_enum::Dirichlet, container,
    alloc>::is_diagonally_dominant() const {
  if (std::abs(b_[0]) < std::abs(c_[0])) return false;
  if (std::abs(b_[system_size_ - 1]) < std::abs(a_[system_size_ - 1]))
    return false;

  for (std::size_t t = 0; t < system_size_ - 1; ++t)
    if (std::abs(b_[t]) < (std::abs(a_[t]) + std::abs(c_[t]))) return false;
  return true;
}

template <typename T, template <typename T, typename alloc> typename container,
          typename alloc>
void lss_fdm_thomas_lu_solver::fdm_thomas_lu_solver<
    T, lss_enumerations::boundary_condition_enum::Dirichlet, container,
    alloc>::kernel(container<T, alloc>& solution) {
  // check the diagonal dominance:
  LSS_ASSERT(is_diagonally_dominant() == true,
             "Tridiagonal matrix must be diagonally dominant.");

  // clear the working containers:
  beta_.clear();
  gamma_.clear();

  // resize the working containers:
  beta_.resize(system_size_);
  gamma_.resize(system_size_);

  // init values for the working containers:
  beta_[0] = b_[0];
  gamma_[0] = c_[0] / beta_[0];

  for (std::size_t t = 1; t < system_size_ - 1; ++t) {
    beta_[t] = b_[t] - (a_[t] * gamma_[t - 1]);
    gamma_[t] = c_[t] / beta_[t];
  }
  beta_[system_size_ - 1] =
      b_[system_size_ - 1] - (a_[system_size_ - 1] * gamma_[system_size_ - 2]);

  solution[1] = (f_[0] - a_[0] * boundary_.first) / beta_[0];
  for (std::size_t t = 1; t < system_size_; ++t) {
    solution[t + 1] = (f_[t] - (a_[t] * solution[t])) / beta_[t];
  }
  solution[system_size_] =
      ((f_[system_size_ - 1] - c_[system_size_ - 1] * boundary_.second) -
       (a_[system_size_ - 1] * solution[system_size_ - 1])) /
      beta_[system_size_ - 1];

  f_[system_size_ - 1] = solution[system_size_];
  for (long t = system_size_ - 2; t >= 0; --t) {
    f_[t] = solution[t + 1] - (gamma_[t] * f_[t + 1]);
  }
  // fill in the known boundary values:
  solution[0] = boundary_.first;
  solution[discretization_size_ - 1] = boundary_.second;
  std::copy(f_.begin(), f_.end(), std::next(solution.begin()));
}

template <typename T, template <typename T, typename alloc> typename container,
          typename alloc>
bool lss_fdm_thomas_lu_solver::fdm_thomas_lu_solver<
    T, lss_enumerations::boundary_condition_enum::Robin, container,
    alloc>::is_diagonally_dominant() const {
  auto alpha = left_.first;
  auto beta = right_.first;
  if (std::abs(alpha * a_[0] + b_[0]) < std::abs(c_[0])) return false;
  if (std::abs(b_[system_size_ - 1] + (c_[system_size_ - 1] / beta)) <
      std::abs(a_[system_size_ - 1]))
    return false;
  for (std::size_t t = 0; t < system_size_ - 1; ++t)
    if (std::abs(b_[t]) < (std::abs(a_[t]) + std::abs(c_[t]))) return false;
  return true;
}

template <typename T, template <typename T, typename alloc> typename container,
          typename alloc>
void lss_fdm_thomas_lu_solver::fdm_thomas_lu_solver<
    T, lss_enumerations::boundary_condition_enum::Robin, container,
    alloc>::kernel(container<T, alloc>& solution) {
  // check the diagonal dominance:
  LSS_ASSERT(is_diagonally_dominant() == true,
             "Tridiagonal matrix must be diagonally dominant.");

  // clear the working containers:
  beta_.clear();
  gamma_.clear();

  // resize the working containers:
  beta_.resize(system_size_);
  gamma_.resize(system_size_);

  // init values for the working containers:
  beta_[0] = left_.first * a_[0] + b_[0];
  gamma_[0] = c_[0] / beta_[0];

  for (std::size_t t = 1; t < system_size_ - 1; ++t) {
    beta_[t] = b_[t] - (a_[t] * gamma_[t - 1]);
    gamma_[t] = c_[t] / beta_[t];
  }
  beta_[system_size_ - 1] =
      (b_[system_size_ - 1] + (c_[system_size_ - 1] / right_.first)) -
      (a_[system_size_ - 1] * gamma_[system_size_ - 2]);

  solution[1] = (f_[0] - a_[0] * left_.second) / beta_[0];
  for (std::size_t t = 1; t < system_size_; ++t) {
    solution[t + 1] = (f_[t] - (a_[t] * solution[t])) / beta_[t];
  }
  solution[system_size_] =
      ((f_[system_size_ - 1] +
        (c_[system_size_ - 1] * right_.second) / right_.first) -
       (a_[system_size_ - 1] * solution[system_size_ - 1])) /
      beta_[system_size_ - 1];

  f_[system_size_ - 1] = solution[system_size_];
  for (long t = system_size_ - 2; t >= 0; --t) {
    f_[t] = solution[t + 1] - (gamma_[t] * f_[t + 1]);
  }

  solution[0] = (left_.first * f_[0]) + left_.second;
  solution[discretization_size_ - 1] =
      (f_[system_size_ - 1] - right_.second) / right_.first;
  std::copy(f_.begin(), f_.end(), std::next(solution.begin()));
}

#endif  ///_LSS_THOMAS_LU_SOLVER
