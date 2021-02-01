#pragma once
#if !defined(_LSS_FDM_TRIDIAGONAL_SOLVERS)
#define _LSS_FDM_TRIDIAGONAL_SOLVERS

#include "common/lss_enumerations.h"
#include "lss_fdm_double_sweep_solver.h"
#include "lss_fdm_thomas_lu_solver.h"

namespace lss_fdm_tridiagonal_solvers {

using lss_enumerations::boundary_condition_enum;
using lss_fdm_double_sweep_solver::fdm_double_sweep_solver;
using lss_fdm_thomas_lu_solver::fdm_thomas_lu_solver;

template <typename T, boundary_condition_enum bc_type,
          template <typename, boundary_condition_enum,
                    template <typename, typename> typename cont, typename>
          typename fdm_solver,
          template <typename, typename> typename container, typename alloc>
class fdm_tridiagonal_solver {};

template <typename T,
          template <typename, boundary_condition_enum,
                    template <typename, typename> typename cont, typename>
          typename fdm_solver,
          template <typename, typename> typename container, typename alloc>
class fdm_tridiagonal_solver<T, boundary_condition_enum::Dirichlet, fdm_solver,
                             container, alloc> {
 private:
  fdm_solver<T, boundary_condition_enum::Dirichlet, container, alloc> solver_;

 public:
  typedef T value_type;
  explicit fdm_tridiagonal_solver() = delete;
  explicit fdm_tridiagonal_solver(std::size_t discretization_size)
      : solver_{discretization_size} {}

  fdm_tridiagonal_solver(fdm_tridiagonal_solver const &) = delete;
  fdm_tridiagonal_solver &operator=(fdm_tridiagonal_solver const &) = delete;
  fdm_tridiagonal_solver(fdm_tridiagonal_solver &&) = delete;
  fdm_tridiagonal_solver &operator=(fdm_tridiagonal_solver &&) = delete;

  ~fdm_tridiagonal_solver() {}

  void set_diagonals(container<T, alloc> lower_diagonal,
                     container<T, alloc> diagonal,
                     container<T, alloc> upper_diagonal) {
    solver_.set_diagonals(std::move(lower_diagonal), std::move(diagonal),
                          std::move(upper_diagonal));
  }

  void set_boundary_condition(std::pair<T, T> const &boundary_pair) {
    solver_.set_boundary_condition(boundary_pair);
  }

  void set_rhs(container<T, alloc> const &rhs) { solver_.set_rhs(rhs); }

  void solve(container<T, alloc> &solution) { solver_.solve(solution); }

  container<T, alloc> const solve() { return solver_.solve(); }
};

template <typename T,
          template <typename, boundary_condition_enum,
                    template <typename, typename> typename cont, typename>
          typename fdm_solver,
          template <typename, typename> typename container, typename alloc>
class fdm_tridiagonal_solver<T, boundary_condition_enum::Robin, fdm_solver,
                             container, alloc> {
 private:
  fdm_solver<T, boundary_condition_enum::Robin, container, alloc> solver_;

 public:
  typedef T value_type;
  explicit fdm_tridiagonal_solver() = delete;
  explicit fdm_tridiagonal_solver(std::size_t discretization_size)
      : solver_{discretization_size} {}

  fdm_tridiagonal_solver(fdm_tridiagonal_solver const &) = delete;
  fdm_tridiagonal_solver &operator=(fdm_tridiagonal_solver const &) = delete;
  fdm_tridiagonal_solver(fdm_tridiagonal_solver &&) = delete;
  fdm_tridiagonal_solver &operator=(fdm_tridiagonal_solver &&) = delete;

  ~fdm_tridiagonal_solver() {}

  void set_diagonals(container<T, alloc> lower_diagonal,
                     container<T, alloc> diagonal,
                     container<T, alloc> upper_diagonal) {
    solver_.set_diagonals(std::move(lower_diagonal), std::move(diagonal),
                          std::move(upper_diagonal));
  }

  void set_boundary_condition(std::pair<T, T> const &left,
                              std::pair<T, T> const &right) {
    solver_.set_boundary_condition(left, right);
  }

  void set_rhs(container<T, alloc> const &rhs) { solver_.set_rhs(rhs); }

  void solve(container<T, alloc> &solution) { solver_.solve(solution); }

  container<T, alloc> const solve() { return solver_.solve(); }
};
}  // namespace lss_fdm_tridiagonal_solvers

#endif  //_LSS_FDM_TRIDIAGONAL_SOLVERS
