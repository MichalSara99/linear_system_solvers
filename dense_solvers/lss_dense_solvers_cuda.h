#pragma once
#if !defined(_LSS_DENSE_SOLVERS_CUDA)
#define _LSS_DENSE_SOLVERS_CUDA

#include <cusolverDn.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <type_traits>

#include "common/lss_enumerations.h"
#include "common/lss_helpers.h"
#include "common/lss_macros.h"
#include "common/lss_utility.h"
#include "lss_dense_solvers_policy.h"

namespace lss_dense_solvers_cuda {

using lss_dense_solvers_policy::DenseSolverDevice;
using lss_dense_solvers_policy::DenseSolverQR;
using lss_enumerations::FlatMatrixSort;
using lss_helpers::RealDenseSolverCUDAHelpers;
using lss_utility::FlatMatrix;

template <typename T,
          typename =
              typename std::enable_if<std::is_floating_point<T>::value>::type>
class RealDenseSolverCUDA {};

template <typename T>
class RealDenseSolverCUDA<T> {
 private:
  std::size_t matrixRows_;
  std::size_t matrixCols_;
  FlatMatrix<T> matrixElements_;

  thrust::host_vector<T> h_matrixValues_;
  thrust::host_vector<T> h_rhsValues_;

  void populate();

 public:
  typedef T value_type;
  explicit RealDenseSolverCUDA() : matrixCols_{0}, matrixRows_{0} {}
  virtual ~RealDenseSolverCUDA() {}

  RealDenseSolverCUDA(RealDenseSolverCUDA const&) = delete;
  RealDenseSolverCUDA& operator=(RealDenseSolverCUDA const&) = delete;
  RealDenseSolverCUDA(RealDenseSolverCUDA&&) = delete;
  RealDenseSolverCUDA& operator=(RealDenseSolverCUDA&&) = delete;

  void initialize(std::size_t matrixRows, std::size_t matrixColumns);

  template <template <typename T, typename Alloc>
            typename Container = std::vector,
            typename Alloc = std::allocator<T>>
  inline void setRhs(Container<T, Alloc> const& rhs) {
    LSS_ASSERT(rhs.size() == matrixRows_,
               " Right-hand side vector of the system has incorrect size.");
    thrust::copy(rhs.begin(), rhs.end(), h_rhsValues_.begin());
  }

  inline void setRhsValue(std::size_t idx, T value) {
    LSS_ASSERT((idx < matrixRows_), "idx is outside range");
    h_rhsValues_[idx] = value;
  }

  inline void setFlatDenseMatrix(FlatMatrix<T> matrix) {
    LSS_ASSERT(
        (matrix.rows() == matrixRows_) && (matrix.columns() == matrixCols_),
        " FlatMatrix has incorrect number of rows or columns");
    matrixElements_ = std::move(matrix);
  }

  inline void setFlatDenseMatrixValue(std::size_t rowIdx, std::size_t colIdx,
                                      T value) {
    LSS_ASSERT((rowIdx < matrixRows_), " row index is out of range");
    LSS_ASSERT((colIdx < matrixCols_), " column index is out of range");
    matrixElements_.emplace_back(rowIdx, colIdx, value);
  }

  inline void setFlatDenseMatrixValue(
      std::tuple<std::size_t, std::size_t, T> triplet) {
    LSS_ASSERT((std::get<0>(triplet) < matrixRows_),
               " row index is out of range");
    LSS_ASSERT((std::get<1>(triplet) < matrixCols_),
               " column index is out of range");
    matrixElements_.emplace_back(std::move(triplet));
  }

  template <template <typename> typename DenseSolverPolicy = DenseSolverQR,
            template <typename T, typename Alloc>
            typename Container = std::vector,
            typename Alloc = std::allocator<T>,
            typename = typename std::enable_if<std::is_base_of<
                DenseSolverDevice<T>, DenseSolverPolicy<T>>::value>::type>
  void solve(Container<T, Alloc>& container);

  template <template <typename> typename DenseSolverPolicy = DenseSolverQR,
            template <typename T, typename Alloc>
            typename Container = std::vector,
            typename Alloc = std::allocator<T>,
            typename = typename std::enable_if<std::is_base_of<
                DenseSolverDevice<T>, DenseSolverPolicy<T>>::value>::type>
  Container<T, Alloc> const solve();
};

}  // namespace lss_dense_solvers_cuda

template <typename T>
void lss_dense_solvers_cuda::RealDenseSolverCUDA<T>::populate() {
  // CUDA Dense solver is column-major:
  matrixElements_.sort(FlatMatrixSort::ColumnMajor);

  for (std::size_t t = 0; t < matrixElements_.size(); ++t) {
    h_matrixValues_[t] = std::get<2>(matrixElements_.at(t));
  }
}

template <typename T>
void lss_dense_solvers_cuda::RealDenseSolverCUDA<T>::initialize(
    std::size_t matrixRows, std::size_t matrixColumns) {
  // set the sizes of the system components:
  matrixCols_ = matrixColumns;
  matrixRows_ = matrixRows;

  // clear the containers:
  matrixElements_.clear();
  h_matrixValues_.clear();
  h_rhsValues_.clear();

  // resize the containers to the correct size:
  h_matrixValues_.resize(matrixCols_ * matrixRows_);
  h_rhsValues_.resize(matrixRows_);
}

template <typename T>
template <template <typename> typename DenseSolverPolicy,
          template <typename T, typename Alloc> typename Container,
          typename Alloc, typename>
void lss_dense_solvers_cuda::RealDenseSolverCUDA<T>::solve(
    Container<T, Alloc>& container) {
  populate();

  // get the dimensions:
  std::size_t lda = std::max(matrixElements_.rows(), matrixElements_.columns());
  std::size_t m = std::min(matrixElements_.rows(), matrixElements_.columns());
  std::size_t ldb = h_rhsValues_.size();

  // step 1: create device containers:
  thrust::device_vector<T> d_matrixValues = h_matrixValues_;
  thrust::device_vector<T> d_rhsValues = h_rhsValues_;
  thrust::device_vector<T> d_solution(m);

  // step 2: cast to raw pointers:
  T* d_matVals = thrust::raw_pointer_cast(d_matrixValues.data());
  T* d_rhsVals = thrust::raw_pointer_cast(d_rhsValues.data());
  T* d_sol = thrust::raw_pointer_cast(d_solution.data());

  // create the helpers:
  RealDenseSolverCUDAHelpers helpers;
  helpers.initialize();

  // call the DenseSolverPolicy:
  DenseSolverPolicy<T>::solve(helpers.getDenseSolverHandle(),
                              helpers.getCublasHandle(), m, d_matVals, lda,
                              d_rhsVals, d_sol);

  thrust::copy(d_solution.begin(), d_solution.end(), container.begin());
}

template <typename T>
template <template <typename> typename DenseSolverPolicy,
          template <typename T, typename Alloc> typename Container,
          typename Alloc, typename>
Container<T, Alloc> const
lss_dense_solvers_cuda::RealDenseSolverCUDA<T>::solve() {
  populate();

  // get the dimensions:
  std::size_t lda = std::max(matrixElements_.rows(), matrixElements_.columns());
  std::size_t m = std::min(matrixElements_.rows(), matrixElements_.columns());
  std::size_t ldb = h_rhsValues_.size();

  // prepare container for solution:
  thrust::host_vector<T> h_solution(m);

  // step 1: create device vectors:
  thrust::device_vector<T> d_matrixValues = h_matrixValues_;
  thrust::device_vector<T> d_rhsValues = h_rhsValues_;
  thrust::device_vector<T> d_solution = h_solution;

  // step 2: cast to raw pointers:
  T* d_matVals = thrust::raw_pointer_cast(d_matrixValues.data());
  T* d_rhsVals = thrust::raw_pointer_cast(d_rhsValues.data());
  T* d_sol = thrust::raw_pointer_cast(d_solution.data());

  // create the helpers:
  RealDenseSolverCUDAHelpers helpers;
  helpers.initialize();

  // call the DenseSolverPolicy:
  DenseSolverPolicy<T>::solve(helpers.getDenseSolverHandle(),
                              helpers.getCublasHandle(), m, d_matVals, lda,
                              d_rhsVals, d_sol);

  Container<T, Alloc> solution(h_solution.size());
  thrust::copy(d_solution.begin(), d_solution.end(), solution.begin());

  return solution;
}

#endif  ///_LSS_DENSE_SOLVERS_CUDA
