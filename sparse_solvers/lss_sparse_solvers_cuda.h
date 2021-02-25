#pragma once
#if !defined(_LSS_SPARSE_SPARSE_CUDA)
#define _LSS_SPARSE_SPARSE_CUDA

#pragma warning(disable : 4267)
#pragma warning(disable : 4244)

#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <type_traits>

#include "common/lss_enumerations.h"
#include "common/lss_helpers.h"
#include "common/lss_utility.h"
#include "lss_sparse_solvers_policy.h"

namespace lss_sparse_solvers {

using lss_enumerations::flat_matrix_sort_enum;
using lss_enumerations::memory_space_enum;
using lss_helpers::real_sparse_solver_cuda_helpers;
using lss_sparse_solvers_policy::sparse_solver_device;
using lss_sparse_solvers_policy::sparse_solver_device_cholesky;
using lss_sparse_solvers_policy::sparse_solver_device_qr;
using lss_sparse_solvers_policy::sparse_solver_host;
using lss_sparse_solvers_policy::sparse_solver_host_cholesky;
using lss_sparse_solvers_policy::sparse_solver_host_lu;
using lss_sparse_solvers_policy::sparse_solver_host_qr;
using lss_utility::flat_matrix;
using lss_utility::uptr_t;

template <memory_space_enum memory_space, typename T>
class real_sparse_solver_cuda {};

// =============================================================================
// ========== real_sparse_solver_cuda partial specialization for HOST ==========
// =============================================================================

template <typename T>
class real_sparse_solver_cuda<memory_space_enum::Host, T> {
 protected:
  int system_size_;
  uptr_t<flat_matrix<T>> matrix_data_ptr_;

  thrust::host_vector<T> h_matrix_values_;
  thrust::host_vector<T> h_vector_values_;  // of systemSize length
  thrust::host_vector<int> h_column_indices_;
  thrust::host_vector<int> h_row_counts_;  // of systemSize + 1 length

  void build_csr();
  explicit real_sparse_solver_cuda() {}

 public:
  typedef T value_type;
  void initialize(std::size_t system_size);

  explicit real_sparse_solver_cuda(int system_size)
      : system_size_{system_size} {}
  virtual ~real_sparse_solver_cuda() {}

  real_sparse_solver_cuda(real_sparse_solver_cuda const&) = delete;
  real_sparse_solver_cuda& operator=(real_sparse_solver_cuda const&) = delete;
  real_sparse_solver_cuda(real_sparse_solver_cuda&&) = delete;
  real_sparse_solver_cuda& operator=(real_sparse_solver_cuda&&) = delete;

  inline std::size_t non_zero_elements() const {
    return matrix_data_ptr_->size();
  }

  template <template <typename T, typename alloc>
            typename container = std::vector,
            typename alloc = std::allocator<T>>
  inline void set_rhs(container<T, alloc> const& rhs) {
    LSS_ASSERT(rhs.size() == system_size_, " rhs has incorrect size");
    thrust::copy(rhs.begin(), rhs.end(), h_vector_values_.begin());
  }

  inline void set_rhs_value(std::size_t idx, T value) {
    LSS_ASSERT((idx < system_size_), "idx is outside range");
    h_vector_values_[idx] = value;
  }

  inline void set_flat_sparse_matrix(flat_matrix<T> matrix) {
    LSS_ASSERT(matrix.columns() == system_size_,
               " Incorrect number of columns");
    LSS_ASSERT(matrix.rows() == system_size_, " Incorrect number of rows");
    matrix_data_ptr_ = std::make_unique<flat_matrix<T>>(std::move(matrix));
  }

  template <
      template <typename>
      typename sparse_solver_host_policy = sparse_solver_host_qr,
      template <typename T, typename alloc> typename container = std::vector,
      typename alloc = std::allocator<T>,
      typename = typename std::enable_if<std::is_base_of<
          sparse_solver_host<T>, sparse_solver_host_policy<T>>::value>::type>
  void solve(container<T, alloc>& solution);

  template <
      template <typename T>
      typename sparse_solver_host_policy = sparse_solver_host_qr,
      template <typename T, typename alloc> typename container = std::vector,
      typename alloc = std::allocator<T>,
      typename = typename std::enable_if<std::is_base_of<
          sparse_solver_host<T>, sparse_solver_host_policy<T>>::value>::type>
  container<T, alloc> const solve();
};

// =============================================================================
// ========== real_sparse_solver_cuda partial specialization for DEVICE
// ===========
// =============================================================================

template <typename T>
class real_sparse_solver_cuda<memory_space_enum::Device, T> {
 protected:
  int system_size_;
  uptr_t<flat_matrix<T>> matrix_data_ptr_;

  thrust::host_vector<T> h_matrix_values_;
  thrust::host_vector<T> h_vector_values_;  // of systemSize length
  thrust::host_vector<int> h_column_indices_;
  thrust::host_vector<int> h_row_counts_;  // of systemSize + 1 length

  void build_csr();
  explicit real_sparse_solver_cuda() {}

 public:
  typedef T value_type;
  explicit real_sparse_solver_cuda(int system_size)
      : system_size_{system_size} {}
  virtual ~real_sparse_solver_cuda() {}

  real_sparse_solver_cuda(real_sparse_solver_cuda const&) = delete;
  real_sparse_solver_cuda& operator=(real_sparse_solver_cuda const&) = delete;
  real_sparse_solver_cuda(real_sparse_solver_cuda&&) = delete;
  real_sparse_solver_cuda& operator=(real_sparse_solver_cuda&&) = delete;

  void initialize(std::size_t system_size);

  inline std::size_t non_zero_elements() const {
    LSS_VERIFY(matrix_data_ptr_, "flat_matrix has not been provided.");
    return matrix_data_ptr_->size();
  }

  template <template <typename T, typename alloc>
            typename container = std::vector,
            typename alloc = std::allocator<T>>
  inline void set_rhs(container<T, alloc> const& rhs) {
    LSS_ASSERT(rhs.size() == system_size_, " rhs has incorrect size");
    thrust::copy(rhs.begin(), rhs.end(), h_vector_values_.begin());
  }

  inline void set_rhs_value(std::size_t idx, T value) {
    LSS_ASSERT((idx < system_size_), "idx is outside range");
    h_vector_values_[idx] = value;
  }

  inline void set_flat_sparse_matrix(flat_matrix<T> matrix) {
    LSS_ASSERT(matrix.columns() == system_size_,
               " Incorrect number of columns");
    LSS_ASSERT(matrix.rows() == system_size_, " Incorrect number of rows");
    matrix_data_ptr_ = std::make_unique<flat_matrix<T>>(std::move(matrix));
  }

  template <template <typename T>
            typename sparse_solver_device_policy = sparse_solver_device_qr,
            template <typename T, typename alloc>
            typename container = std::vector,
            typename alloc = std::allocator<T>,
            typename = typename std::enable_if<
                std::is_base_of<sparse_solver_device<T>,
                                sparse_solver_device_policy<T>>::value>::type>
  void solve(container<T, alloc>& solution);

  template <template <typename T>
            typename sparse_solver_device_policy = sparse_solver_device_qr,
            template <typename T, typename alloc>
            typename container = std::vector,
            typename alloc = std::allocator<T>,
            typename = typename std::enable_if<
                std::is_base_of<sparse_solver_device<T>,
                                sparse_solver_device_policy<T>>::value>::type>
  container<T, alloc> const solve();
};

}  // namespace lss_sparse_solvers

template <typename T>
void lss_sparse_solvers::real_sparse_solver_cuda<
    lss_enumerations::memory_space_enum::Host, T>::initialize(std::size_t
                                                                  system_size) {
  // set the size of the linear system:
  system_size_ = system_size;

  // clear the containers:
  h_matrix_values_.clear();
  h_vector_values_.clear();
  h_column_indices_.clear();
  h_row_counts_.clear();

  // resize the containers to the correct size:
  h_vector_values_.resize(system_size_);
  h_row_counts_.resize((system_size_ + 1));
}

template <typename T>
void lss_sparse_solvers::real_sparse_solver_cuda<
    lss_enumerations::memory_space_enum::Device,
    T>::initialize(std::size_t system_size) {
  // set the size of the linear system:
  system_size_ = system_size;

  // clear the containers:
  h_matrix_values_.clear();
  h_vector_values_.clear();
  h_column_indices_.clear();
  h_row_counts_.clear();

  // resize the containers to the correct size:
  h_vector_values_.resize(system_size_);
  h_row_counts_.resize((system_size_ + 1));
}

template <typename T>
void lss_sparse_solvers::real_sparse_solver_cuda<
    lss_enumerations::memory_space_enum::Host, T>::build_csr() {
  int const nonZeroSize = non_zero_elements();

  LSS_VERIFY(matrix_data_ptr_, "flat_matrix has not been provided.");
  // CUDA sparse solver is row-major:
  matrix_data_ptr_->sort(flat_matrix_sort_enum::RowMajor);

  h_column_indices_.resize(nonZeroSize);
  h_matrix_values_.resize(nonZeroSize);

  int nElement{0};
  int nRowElement{0};
  int lastRow{0};
  h_row_counts_[nRowElement++] = nElement;

  for (int i = 0; i < nonZeroSize; ++i) {
    if (lastRow < std::get<0>(matrix_data_ptr_->at(i))) {
      h_row_counts_[nRowElement++] = i;
      lastRow++;
    }

    h_column_indices_[i] = std::get<1>(matrix_data_ptr_->at(i));
    h_matrix_values_[i] = std::get<2>(matrix_data_ptr_->at(i));
  }
  h_row_counts_[nRowElement] = nonZeroSize;
}

template <typename T>
void lss_sparse_solvers::real_sparse_solver_cuda<
    lss_enumerations::memory_space_enum::Device, T>::build_csr() {
  int const non_zero_size = non_zero_elements();

  LSS_VERIFY(matrix_data_ptr_, "flat_matrix has not been provided.");
  // CUDA sparse solver is row-major:
  matrix_data_ptr_->sort(flat_matrix_sort_enum::RowMajor);

  h_column_indices_.resize(non_zero_size);
  h_matrix_values_.resize(non_zero_size);

  int nElement{0};
  int nRowElement{0};
  int lastRow{0};
  h_row_counts_[nRowElement++] = nElement;

  for (std::size_t i = 0; i < non_zero_size; ++i) {
    if (lastRow < std::get<0>(matrix_data_ptr_->at(i))) {
      h_row_counts_[nRowElement++] = i;
      lastRow++;
    }

    h_column_indices_[i] = std::get<1>(matrix_data_ptr_->at(i));
    h_matrix_values_[i] = std::get<2>(matrix_data_ptr_->at(i));
  }
  h_row_counts_[nRowElement] = non_zero_size;
}

template <typename T>
template <template <typename T> typename sparse_solver_host_policy,
          template <typename T, typename alloc> typename container,
          typename alloc, typename>
void lss_sparse_solvers::real_sparse_solver_cuda<
    lss_enumerations::memory_space_enum::Host, T>::solve(container<T, alloc>&
                                                             solution) {
  build_csr();

  // get the non-zero size:
  int const non_zero_size = non_zero_elements();

  // integer for holding index of row where singularity occurs:
  int singular_idx{0};

  // prepare container for solution:
  thrust::host_vector<T> h_solution(system_size_);

  // get the raw host pointers
  T* h_mat_vals = thrust::raw_pointer_cast(h_matrix_values_.data());
  T* h_rhs_vals = thrust::raw_pointer_cast(h_vector_values_.data());
  T* h_sol = thrust::raw_pointer_cast(h_solution.data());
  int* h_col = thrust::raw_pointer_cast(h_column_indices_.data());
  int* h_row = thrust::raw_pointer_cast(h_row_counts_.data());

  // create the helpers:
  real_sparse_solver_cuda_helpers helpers;
  helpers.initialize();
  // call the sparse_solver_host_policy:
  sparse_solver_host_policy<T>::solve(
      helpers.get_solver_handle(), helpers.get_matrix_descriptor(),
      system_size_, non_zero_size, h_mat_vals, h_row, h_col, h_rhs_vals, 0.0, 0,
      h_sol, &singular_idx);

  if (singular_idx >= 0) {
    std::cerr << "Sparse matrix is singular at row: " << singular_idx << "\n";
  }
  thrust::copy(h_solution.begin(), h_solution.end(), solution.begin());
}

template <typename T>
template <template <typename T> typename sparse_solver_device_policy,
          template <typename T, typename alloc> typename container,
          typename alloc, typename>
void lss_sparse_solvers::real_sparse_solver_cuda<
    lss_enumerations::memory_space_enum::Device, T>::solve(container<T, alloc>&
                                                               solution) {
  build_csr();

  // get the non-zero size:
  int const non_zero_size = non_zero_elements();

  // integer for holding index of row where singularity occurs:
  int singular_idx{0};

  // prepare container for solution:
  thrust::device_vector<T> d_solution(system_size_);

  // copy to the device constainers:
  thrust::device_vector<T> d_matrix_values = h_matrix_values_;
  thrust::device_vector<T> d_vector_values = h_vector_values_;
  thrust::device_vector<int> d_column_indices = h_column_indices_;
  thrust::device_vector<int> d_row_counts = h_row_counts_;

  // get the raw host pointers
  T* d_mat_vals = thrust::raw_pointer_cast(d_matrix_values.data());
  T* d_rhs_vals = thrust::raw_pointer_cast(d_vector_values.data());
  T* d_sol = thrust::raw_pointer_cast(d_solution.data());
  int* d_col = thrust::raw_pointer_cast(d_column_indices.data());
  int* d_row = thrust::raw_pointer_cast(d_row_counts.data());

  // create the helpers:
  real_sparse_solver_cuda_helpers helpers;
  helpers.initialize();
  // call the sparse_solver_device_policy:
  sparse_solver_device_policy<T>::solve(
      helpers.get_solver_handle(), helpers.get_matrix_descriptor(),
      system_size_, non_zero_size, d_mat_vals, d_row, d_col, d_rhs_vals, 0.0, 0,
      d_sol, &singular_idx);

  if (singular_idx >= 0) {
    std::cerr << "Sparse matrix is singular at row: " << singular_idx << "\n";
  }
  thrust::copy(d_solution.begin(), d_solution.end(), solution.begin());
}

template <typename T>
template <template <typename T> typename sparse_solver_host_policy,
          template <typename T, typename alloc> typename container,
          typename alloc, typename>
container<T, alloc> const lss_sparse_solvers::real_sparse_solver_cuda<
    lss_enumerations::memory_space_enum::Host, T>::solve() {
  build_csr();

  // get the non-zero size:
  int const no_zero_size = non_zero_elements();

  // integer for holding index of row where singularity occurs:
  int singular_idx{0};

  // prepare container for solution:
  thrust::host_vector<T> h_solution(system_size_);

  // get the raw host pointers
  T* h_mat_vals = thrust::raw_pointer_cast(h_matrix_values_.data());
  T* h_rhs_vals = thrust::raw_pointer_cast(h_vector_values_.data());
  T* h_sol = thrust::raw_pointer_cast(h_solution.data());
  int* h_col = thrust::raw_pointer_cast(h_column_indices_.data());
  int* h_row = thrust::raw_pointer_cast(h_row_counts_.data());

  // create the helpers:
  real_sparse_solver_cuda_helpers helpers;
  helpers.initialize();
  // call the sparse_solver_host_policy:
  sparse_solver_host_policy<T>::solve(
      helpers.get_solver_handle(), helpers.get_matrix_descriptor(),
      system_size_, no_zero_size, h_mat_vals, h_row, h_col, h_rhs_vals, 0.0, 0,
      h_sol, &singular_idx);

  if (singular_idx >= 0) {
    std::cerr << "Sparse matrix is singular at row: " << singular_idx << "\n";
  }

  container<T, alloc> solution(h_solution.size());
  thrust::copy(h_solution.begin(), h_solution.end(), solution.begin());
  return solution;
}

template <typename T>
template <template <typename T> typename sparse_solver_device_policy,
          template <typename T, typename alloc> typename container,
          typename alloc, typename>
container<T, alloc> const lss_sparse_solvers::real_sparse_solver_cuda<
    lss_enumerations::memory_space_enum::Device, T>::solve() {
  build_csr();

  // get the non-zero size:
  int const non_zero_size = non_zero_elements();

  // integer for holding index of row where singularity occurs:
  int singular_idx{0};

  // prepare container for solution:
  thrust::device_vector<T> d_solution(system_size_);

  // copy to the device constainers:
  thrust::device_vector<T> d_matrix_values = h_matrix_values_;
  thrust::device_vector<T> d_vector_values = h_vector_values_;
  thrust::device_vector<int> d_column_indices = h_column_indices_;
  thrust::device_vector<int> d_row_counts = h_row_counts_;

  // get the raw host pointers
  T* d_mat_vals = thrust::raw_pointer_cast(d_matrix_values.data());
  T* d_rhs_vals = thrust::raw_pointer_cast(d_vector_values.data());
  T* d_sol = thrust::raw_pointer_cast(d_solution.data());
  int* d_col = thrust::raw_pointer_cast(d_column_indices.data());
  int* d_row = thrust::raw_pointer_cast(d_row_counts.data());

  // create the helpers:
  real_sparse_solver_cuda_helpers helpers;
  helpers.initialize();
  // call the sparse_solver_device_policy:
  sparse_solver_device_policy<T>::solve(
      helpers.get_solver_handle(), helpers.get_matrix_descriptor(),
      system_size_, non_zero_size, d_mat_vals, d_row, d_col, d_rhs_vals, 0.0, 0,
      d_sol, &singular_idx);

  if (singular_idx >= 0) {
    std::cerr << "Sparse matrix is singular at row: " << singular_idx << "\n";
  }

  container<T, alloc> solution(system_size_);
  thrust::copy(d_solution.begin(), d_solution.end(), solution.begin());

  return solution;
}

#endif  ///_LSS_SPARSE_SPARSE_CUDA
