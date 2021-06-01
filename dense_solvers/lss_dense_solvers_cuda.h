#pragma once
#if !defined(_LSS_DENSE_SOLVERS_CUDA)
#define _LSS_DENSE_SOLVERS_CUDA

#include <cusolverDn.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <type_traits>

#include "common/lss_containers.h"
#include "common/lss_enumerations.h"
#include "common/lss_helpers.h"
#include "common/lss_macros.h"
#include "common/lss_utility.h"
#include "lss_dense_solvers_policy.h"

namespace lss_dense_solvers {

using lss_containers::flat_matrix;
using lss_dense_solvers_policy::dense_solver_device;
using lss_dense_solvers_policy::dense_solver_qr;
using lss_enumerations::flat_matrix_sort_enum;
using lss_helpers::real_dense_solver_cuda_helpers;
using lss_utility::uptr_t;

template <typename T,
          typename =
              typename std::enable_if<std::is_floating_point<T>::value>::type>
class real_dense_solver_cuda {};

template <typename T>
class real_dense_solver_cuda<T> {
 private:
  int matrix_rows_;
  int matrix_cols_;
  uptr_t<flat_matrix<T>> matrix_data_ptr_;

  thrust::host_vector<T> h_matrix_values_;
  thrust::host_vector<T> h_rhs_values_;

  void populate();

  explicit real_dense_solver_cuda() {}

 public:
  typedef T value_type;
  explicit real_dense_solver_cuda(int matrix_rows, int matrix_columns)
      : matrix_cols_{matrix_columns}, matrix_rows_{matrix_rows} {}
  virtual ~real_dense_solver_cuda() {}

  real_dense_solver_cuda(real_dense_solver_cuda const&) = delete;
  real_dense_solver_cuda& operator=(real_dense_solver_cuda const&) = delete;
  real_dense_solver_cuda(real_dense_solver_cuda&&) = delete;
  real_dense_solver_cuda& operator=(real_dense_solver_cuda&&) = delete;

  void initialize(int matrix_rows, int matrix_columns);

  template <template <typename T, typename alloc>
            typename container = std::vector,
            typename alloc = std::allocator<T>>
  inline void set_rhs(container<T, alloc> const& rhs) {
    LSS_ASSERT(rhs.size() == matrix_rows_,
               " Right-hand side vector of the system has incorrect size.");
    thrust::copy(rhs.begin(), rhs.end(), h_rhs_values_.begin());
  }

  inline void set_rhs_value(std::size_t idx, T value) {
    LSS_ASSERT((idx < matrix_rows_), "idx is outside range");
    h_rhs_values_[idx] = value;
  }

  inline void set_flat_dense_matrix(flat_matrix<T> matrix) {
    LSS_ASSERT(
        (matrix.rows() == matrix_rows_) && (matrix.columns() == matrix_cols_),
        " flat_matrix has incorrect number of rows or columns");
    matrix_data_ptr_ = std::make_unique<flat_matrix<T>>(std::move(matrix));
  }

  template <template <typename> typename dense_solver_policy = dense_solver_qr,
            template <typename T, typename alloc>
            typename container = std::vector,
            typename alloc = std::allocator<T>,
            typename = typename std::enable_if<std::is_base_of<
                dense_solver_device<T>, dense_solver_policy<T>>::value>::type>
  void solve(container<T, alloc>& container);

  template <template <typename> typename dense_solver_policy = dense_solver_qr,
            template <typename T, typename alloc>
            typename container = std::vector,
            typename alloc = std::allocator<T>,
            typename = typename std::enable_if<std::is_base_of<
                dense_solver_device<T>, dense_solver_policy<T>>::value>::type>
  container<T, alloc> const solve();
};

}  // namespace lss_dense_solvers

template <typename T>
void lss_dense_solvers::real_dense_solver_cuda<T>::populate() {
  LSS_VERIFY(matrix_data_ptr_, "flat_matrix has not been provided.");
  // CUDA Dense solver is column-major:
  matrix_data_ptr_->sort(flat_matrix_sort_enum::ColumnMajor);

  for (std::size_t t = 0; t < matrix_data_ptr_->size(); ++t) {
    h_matrix_values_[t] = std::get<2>(matrix_data_ptr_->at(t));
  }
}

template <typename T>
void lss_dense_solvers::real_dense_solver_cuda<T>::initialize(
    int matrix_rows, int matrix_columns) {
  // set the sizes of the system components:
  matrix_cols_ = matrix_columns;
  matrix_rows_ = matrix_rows;

  // clear the containers:
  h_matrix_values_.clear();
  h_rhs_values_.clear();

  // resize the containers to the correct size:
  h_matrix_values_.resize(matrix_cols_ * matrix_rows_);
  h_rhs_values_.resize(matrix_rows_);
}

template <typename T>
template <template <typename> typename dense_solver_policy,
          template <typename T, typename alloc> typename container,
          typename alloc, typename>
void lss_dense_solvers::real_dense_solver_cuda<T>::solve(
    container<T, alloc>& container) {
  populate();

  // get the dimensions:
  std::size_t lda =
      std::max(matrix_data_ptr_->rows(), matrix_data_ptr_->columns());
  std::size_t m =
      std::min(matrix_data_ptr_->rows(), matrix_data_ptr_->columns());
  std::size_t ldb = h_rhs_values_.size();

  // step 1: create device containers:
  thrust::device_vector<T> d_matrix_values = h_matrix_values_;
  thrust::device_vector<T> d_rhs_values = h_rhs_values_;
  thrust::device_vector<T> d_solution(m);

  // step 2: cast to raw pointers:
  T* d_mat_vals = thrust::raw_pointer_cast(d_matrix_values.data());
  T* d_rhs_vals = thrust::raw_pointer_cast(d_rhs_values.data());
  T* d_sol = thrust::raw_pointer_cast(d_solution.data());

  // create the helpers:
  real_dense_solver_cuda_helpers helpers;
  helpers.initialize();

  // call the DenseSolverPolicy:
  dense_solver_policy<T>::solve(helpers.get_dense_solver_handle(),
                                helpers.get_cublas_handle(), m, d_mat_vals, lda,
                                d_rhs_vals, d_sol);

  thrust::copy(d_solution.begin(), d_solution.end(), container.begin());
}

template <typename T>
template <template <typename> typename dense_solver_policy,
          template <typename T, typename alloc> typename container,
          typename alloc, typename>
container<T, alloc> const
lss_dense_solvers::real_dense_solver_cuda<T>::solve() {
  populate();

  // get the dimensions:
  std::size_t lda =
      std::max(matrix_data_ptr_->rows(), matrix_data_ptr_->columns());
  std::size_t m =
      std::min(matrix_data_ptr_->rows(), matrix_data_ptr_->columns());
  std::size_t ldb = h_rhs_values_.size();

  // prepare container for solution:
  thrust::host_vector<T> h_solution(m);

  // step 1: create device vectors:
  thrust::device_vector<T> d_matrix_values = h_matrix_values_;
  thrust::device_vector<T> d_rhs_values = h_rhs_values_;
  thrust::device_vector<T> d_solution = h_solution;

  // step 2: cast to raw pointers:
  T* d_mat_vals = thrust::raw_pointer_cast(d_matrix_values.data());
  T* d_rhs_vals = thrust::raw_pointer_cast(d_rhs_values.data());
  T* d_sol = thrust::raw_pointer_cast(d_solution.data());

  // create the helpers:
  real_dense_solver_cuda_helpers helpers;
  helpers.initialize();

  // call the DenseSolverPolicy:
  dense_solver_policy<T>::solve(helpers.get_dense_solver_handle(),
                                helpers.get_cublas_handle(), m, d_mat_vals, lda,
                                d_rhs_vals, d_sol);

  container<T, alloc> solution(h_solution.size());
  thrust::copy(d_solution.begin(), d_solution.end(), solution.begin());

  return solution;
}

#endif  ///_LSS_DENSE_SOLVERS_CUDA
