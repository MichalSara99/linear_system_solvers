#pragma once
#if !defined(_LSS_DENSE_SOLVERS_CUDA_HPP_)
#define _LSS_DENSE_SOLVERS_CUDA_HPP_

#include <cusolverDn.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <type_traits>

#include "common/lss_enumerations.hpp"
#include "common/lss_helpers.hpp"
#include "common/lss_macros.hpp"
#include "common/lss_utility.hpp"
#include "containers/lss_flat_matrix.hpp"
#include "lss_dense_solvers_policy.hpp"

namespace lss_dense_solvers
{

using lss_containers::flat_matrix;
using lss_dense_solvers_policy::dense_solver_cholesky;
using lss_dense_solvers_policy::dense_solver_lu;
using lss_dense_solvers_policy::dense_solver_qr;
using lss_enumerations::factorization_enum;
using lss_enumerations::flat_matrix_sort_enum;
using lss_helpers::real_dense_solver_cuda_helpers;
using lss_utility::sptr_t;
using lss_utility::uptr_t;

template <typename fp_type, template <typename, typename> typename container = std::vector,
          typename allocator = std::allocator<fp_type>>
class real_dense_solver_cuda
{
  private:
    int matrix_rows_;
    int matrix_cols_;
    uptr_t<flat_matrix<fp_type>> matrix_data_ptr_;

    thrust::host_vector<fp_type> h_matrix_values_;
    thrust::host_vector<fp_type> h_rhs_values_;

    void populate();

    explicit real_dense_solver_cuda()
    {
    }

  public:
    typedef fp_type value_type;
    explicit real_dense_solver_cuda(int matrix_rows, int matrix_columns)
        : matrix_cols_{matrix_columns}, matrix_rows_{matrix_rows}
    {
    }
    virtual ~real_dense_solver_cuda()
    {
    }

    real_dense_solver_cuda(real_dense_solver_cuda const &) = delete;
    real_dense_solver_cuda &operator=(real_dense_solver_cuda const &) = delete;
    real_dense_solver_cuda(real_dense_solver_cuda &&) = delete;
    real_dense_solver_cuda &operator=(real_dense_solver_cuda &&) = delete;

    void initialize(int matrix_rows, int matrix_columns);

    inline void set_rhs(container<fp_type, allocator> const &rhs)
    {
        LSS_ASSERT(rhs.size() == matrix_rows_, " Right-hand side vector of the system has incorrect size.");
        thrust::copy(rhs.begin(), rhs.end(), h_rhs_values_.begin());
    }

    inline void set_rhs_value(std::size_t idx, fp_type value)
    {
        LSS_ASSERT((idx < matrix_rows_), "idx is outside range");
        h_rhs_values_[idx] = value;
    }

    inline void set_flat_dense_matrix(flat_matrix<fp_type> matrix)
    {
        LSS_ASSERT((matrix.rows() == matrix_rows_) && (matrix.columns() == matrix_cols_),
                   " flat_matrix has incorrect number of rows or columns");
        matrix_data_ptr_ = std::make_unique<flat_matrix<fp_type>>(std::move(matrix));
    }

    void solve(container<fp_type, allocator> &container,
               factorization_enum factorization = factorization_enum::QRMethod);

    container<fp_type, allocator> const solve(factorization_enum factorization = factorization_enum::QRMethod);
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void real_dense_solver_cuda<fp_type, container, allocator>::populate()
{
    LSS_VERIFY(matrix_data_ptr_, "flat_matrix has not been provided.");
    // CUDA Dense solver is column-major:
    matrix_data_ptr_->sort(flat_matrix_sort_enum::ColumnMajor);

    for (std::size_t t = 0; t < matrix_data_ptr_->size(); ++t)
    {
        h_matrix_values_[t] = std::get<2>(matrix_data_ptr_->at(t));
    }
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void real_dense_solver_cuda<fp_type, container, allocator>::initialize(int matrix_rows, int matrix_columns)
{
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

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void real_dense_solver_cuda<fp_type, container, allocator>::solve(container<fp_type, allocator> &container,
                                                                  factorization_enum factorization)
{
    populate();

    // get the dimensions:
    std::size_t lda = std::max(matrix_data_ptr_->rows(), matrix_data_ptr_->columns());
    std::size_t m = std::min(matrix_data_ptr_->rows(), matrix_data_ptr_->columns());
    std::size_t ldb = h_rhs_values_.size();

    // step 1: create device containers:
    thrust::device_vector<fp_type> d_matrix_values = h_matrix_values_;
    thrust::device_vector<fp_type> d_rhs_values = h_rhs_values_;
    thrust::device_vector<fp_type> d_solution(m);

    // step 2: cast to raw pointers:
    fp_type *d_mat_vals = thrust::raw_pointer_cast(d_matrix_values.data());
    fp_type *d_rhs_vals = thrust::raw_pointer_cast(d_rhs_values.data());
    fp_type *d_sol = thrust::raw_pointer_cast(d_solution.data());

    // create the helpers:
    real_dense_solver_cuda_helpers helpers;
    helpers.initialize();

    // call the DenseSolverPolicy:
    if (factorization == factorization_enum::QRMethod)
    {
        dense_solver_qr<fp_type>::solve(helpers.get_dense_solver_handle(), helpers.get_cublas_handle(), m, d_mat_vals,
                                        lda, d_rhs_vals, d_sol);
    }
    else if (factorization == factorization_enum::LUMethod)
    {
        dense_solver_lu<fp_type>::solve(helpers.get_dense_solver_handle(), helpers.get_cublas_handle(), m, d_mat_vals,
                                        lda, d_rhs_vals, d_sol);
    }
    else if (factorization == factorization_enum::CholeskyMethod)
    {
        dense_solver_cholesky<fp_type>::solve(helpers.get_dense_solver_handle(), helpers.get_cublas_handle(), m,
                                              d_mat_vals, lda, d_rhs_vals, d_sol);
    }
    else
    {
        throw std::exception("factorization not known");
    }
    thrust::copy(d_solution.begin(), d_solution.end(), container.begin());
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
container<fp_type, allocator> const real_dense_solver_cuda<fp_type, container, allocator>::solve(
    factorization_enum factorization)
{
    populate();

    // get the dimensions:
    std::size_t lda = std::max(matrix_data_ptr_->rows(), matrix_data_ptr_->columns());
    std::size_t m = std::min(matrix_data_ptr_->rows(), matrix_data_ptr_->columns());
    std::size_t ldb = h_rhs_values_.size();

    // prepare container for solution:
    thrust::host_vector<fp_type> h_solution(m);

    // step 1: create device vectors:
    thrust::device_vector<fp_type> d_matrix_values = h_matrix_values_;
    thrust::device_vector<fp_type> d_rhs_values = h_rhs_values_;
    thrust::device_vector<fp_type> d_solution = h_solution;

    // step 2: cast to raw pointers:
    fp_type *d_mat_vals = thrust::raw_pointer_cast(d_matrix_values.data());
    fp_type *d_rhs_vals = thrust::raw_pointer_cast(d_rhs_values.data());
    fp_type *d_sol = thrust::raw_pointer_cast(d_solution.data());

    // create the helpers:
    real_dense_solver_cuda_helpers helpers;
    helpers.initialize();

    // call the DenseSolverPolicy:
    if (factorization == factorization_enum::QRMethod)
    {
        dense_solver_qr<fp_type>::solve(helpers.get_dense_solver_handle(), helpers.get_cublas_handle(), m, d_mat_vals,
                                        lda, d_rhs_vals, d_sol);
    }
    else if (factorization == factorization_enum::LUMethod)
    {
        dense_solver_lu<fp_type>::solve(helpers.get_dense_solver_handle(), helpers.get_cublas_handle(), m, d_mat_vals,
                                        lda, d_rhs_vals, d_sol);
    }
    else if (factorization == factorization_enum::CholeskyMethod)
    {
        dense_solver_cholesky<fp_type>::solve(helpers.get_dense_solver_handle(), helpers.get_cublas_handle(), m,
                                              d_mat_vals, lda, d_rhs_vals, d_sol);
    }
    else
    {
        throw std::exception("factorization not known");
    }
    container<fp_type, allocator> solution(h_solution.size());
    thrust::copy(d_solution.begin(), d_solution.end(), solution.begin());

    return solution;
}

template <typename fp_type> using real_dense_solver_cuda_ptr = sptr_t<real_dense_solver_cuda<fp_type>>;

} // namespace lss_dense_solvers

#endif ///_LSS_DENSE_SOLVERS_CUDA_HPP_
