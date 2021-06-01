#pragma once
#if !defined(_LSS_SOR_SOLVER)
#define _LSS_SOR_SOLVER

#include "common/lss_containers.h"
#include "common/lss_enumerations.h"
#include "common/lss_macros.h"
#include "common/lss_utility.h"
#include "lss_sor_solver_traits.h"

namespace lss_sor_solver {

using lss_containers::flat_matrix;
using lss_containers::flat_matrix_sort_enum;
using lss_sor_solver_traits::sor_solver_traits;
using lss_utility::NaN;
using lss_utility::uptr_t;

template <typename fp_type,
          template <typename fp_type, typename allocator> typename container =
              std::vector,
          typename alloc = std::allocator<fp_type>>
class sor_solver {
 protected:
  container<fp_type, alloc> b_;
  uptr_t<flat_matrix<fp_type>> matrix_data_ptr_;
  std::size_t system_size_;
  fp_type omega_;

  explicit sor_solver(){};

  template <template <typename fp_type> typename traits = sor_solver_traits>
  void kernel(container<fp_type, alloc>& solution);
  bool is_diagonally_dominant();

 public:
  typedef fp_type value_type;
  explicit sor_solver(std::size_t system_size) : system_size_{system_size} {}

  virtual ~sor_solver() {}

  sor_solver(sor_solver const&) = delete;
  sor_solver(sor_solver&&) = delete;
  sor_solver& operator=(sor_solver const&) = delete;
  sor_solver& operator=(sor_solver&&) = delete;

  void set_flat_sparse_matrix(flat_matrix<fp_type> flat_matrix);

  void set_rhs(container<fp_type, alloc> const& rhs);

  void set_omega(fp_type value);

  template <template <typename fp_type> typename traits = sor_solver_traits>
  void solve(container<fp_type, alloc>& solution);

  template <template <typename fp_type> typename traits = sor_solver_traits>
  container<fp_type, alloc> const solve();
};

}  // namespace lss_sor_solver

template <typename fp_type,
          template <typename fp_type, typename alloc> typename container,
          typename alloc>
void lss_sor_solver::sor_solver<fp_type, container, alloc>::set_omega(
    fp_type value) {
  LSS_ASSERT((value > static_cast<fp_type>(0.0)) &&
                 (value < static_cast<fp_type>(2.0)),
             "relaxation parameter must be inside (0,2) range");
  omega_ = value;
}

template <typename fp_type,
          template <typename fp_type, typename alloc> typename container,
          typename alloc>
void lss_sor_solver::sor_solver<fp_type, container, alloc>::set_rhs(
    container<fp_type, alloc> const& rhs) {
  LSS_ASSERT(rhs.size() == system_size_, "Inncorect size for right-hand side");
  b_ = rhs;
}

template <typename fp_type,
          template <typename fp_type, typename alloc> typename container,
          typename alloc>
void lss_sor_solver::sor_solver<fp_type, container, alloc>::
    set_flat_sparse_matrix(flat_matrix<fp_type> matrix) {
  LSS_ASSERT(matrix.rows() == system_size_,
             "Inncorect number of rows for the flat_raw_matrix");
  LSS_ASSERT(matrix.columns() == system_size_,
             "Inncorect number of columns for the flat_raw_matrix");
  matrix_data_ptr_ = std::make_unique<flat_matrix<fp_type>>(std::move(matrix));
}

template <typename fp_type,
          template <typename fp_type, typename alloc> typename container,
          typename alloc>
bool lss_sor_solver::sor_solver<fp_type, container,
                                alloc>::is_diagonally_dominant() {
  LSS_VERIFY(matrix_data_ptr_, "flat_matrix has not been provided.");
  LSS_ASSERT(b_.size() == system_size_, "Incorrect size for right-hand side");
  // first make sure the matrix is row-major sorted
  matrix_data_ptr_->sort(flat_matrix_sort_enum::RowMajor);
  fp_type diag{};
  std::tuple<std::size_t, std::size_t, fp_type> non_diag{};
  fp_type sum{};
  std::size_t cols{};
  // for index of the flat_matrix element:
  std::size_t flt_idx{0};
  for (std::size_t r = 0; r < matrix_data_ptr_->rows(); ++r, flt_idx += cols) {
    sum = static_cast<fp_type>(0.0);
    diag = std::abs(matrix_data_ptr_->diagonal_at_row(r));
    cols = matrix_data_ptr_->non_zero_column_size(r);
    for (std::size_t c = flt_idx; c < cols + flt_idx; ++c) {
      non_diag = matrix_data_ptr_->at(c);
      if (std::get<0>(non_diag) != std::get<1>(non_diag))
        sum += std::abs(std::get<2>(non_diag));
    }
    if (diag < sum) return false;
  }
  return true;
}

template <typename fp_type,
          template <typename fp_type, typename alloc> typename container,
          typename alloc>
template <template <typename fp_type> typename traits>
void lss_sor_solver::sor_solver<fp_type, container, alloc>::kernel(
    container<fp_type, alloc>& solution) {
  LSS_ASSERT(is_diagonally_dominant() == true,
             "flat_raw_matrix isd not diagonally dominant.");
  // set initial step:
  std::size_t step{0};
  // set iter_limit:
  std::size_t iter_limit = traits<fp_type>::iteration_limit();
  // set tolerance:
  fp_type const tol = traits<fp_type>::tolerance();
  // for error:
  fp_type error{};
  // for sigma_1,sigma_2:
  fp_type sigma_1{};
  fp_type sigma_2{};
  // for diagonal value:
  fp_type diag{};
  // for new solution:
  container<fp_type, alloc> x_new(solution);
  // for number of columns:
  std::size_t cols{};
  // for flat_matrix element:
  std::tuple<std::size_t, std::size_t, fp_type> mat_elm{};
  // for flat_matrix row and column index of the element:
  std::size_t mat_r{};
  std::size_t mat_c{};
  // for flat_matrix element value:
  fp_type mat_val{};
  // for index of the flat_matrix element:
  std::size_t flt_idx{0};

  while (iter_limit > step) {
    error = static_cast<fp_type>(0.0);
    flt_idx = 0;
    for (std::size_t r = 0; r < matrix_data_ptr_->rows();
         ++r, flt_idx += cols) {
      sigma_1 = sigma_2 = static_cast<fp_type>(0.0);
      diag = matrix_data_ptr_->diagonal_at_row(r);
      cols = matrix_data_ptr_->non_zero_column_size(r);
      for (std::size_t c = flt_idx; c < cols + flt_idx; ++c) {
        mat_elm = matrix_data_ptr_->at(c);
        mat_r = std::get<0>(mat_elm);
        mat_c = std::get<1>(mat_elm);
        mat_val = std::get<2>(mat_elm);
        if (mat_c < mat_r) sigma_1 += mat_val * x_new[mat_c];
        if (mat_c > mat_r) sigma_2 += mat_val * solution[mat_c];
      }
      x_new[r] = (1.0 - omega_) * solution[r] +
                 ((omega_ / diag) * (b_[r] - sigma_1 - sigma_2));
      error += (x_new[r] - solution[r]) * (x_new[r] - solution[r]);
    }

    if (error <= tol) break;
    solution = x_new;
    step++;
  }
}

template <typename fp_type,
          template <typename fp_type, typename alloc> typename container,
          typename alloc>
template <template <typename fp_type> typename traits>
void lss_sor_solver::sor_solver<fp_type, container, alloc>::solve(
    container<fp_type, alloc>& solution) {
  LSS_ASSERT(solution.size() == system_size_,
             "Incorrect size of solution container");
  kernel(solution);
}

template <typename fp_type,
          template <typename fp_type, typename alloc> typename container,
          typename alloc>
template <template <typename fp_type> typename traits>
container<fp_type, alloc> const
lss_sor_solver::sor_solver<fp_type, container, alloc>::solve() {
  container<fp_type, alloc> solution(system_size_);
  kernel(solution);
  return solution;
}

#endif  ///_LSS_SOR_SOLVER
