#pragma once
#if !defined(_LSS_CORE_CUDA_SOLVER_HPP_)
#define _LSS_CORE_CUDA_SOLVER_HPP_

#pragma warning(disable : 4267)
#pragma warning(disable : 4244)

#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <type_traits>
#include <vector>

#include "common/lss_enumerations.hpp"
#include "common/lss_helpers.hpp"
#include "common/lss_utility.hpp"
#include "containers/lss_flat_matrix.hpp"
#include "lss_core_cuda_solver_policy.hpp"

namespace lss_core_cuda_solver
{

using lss_containers::flat_matrix;
using lss_core_cuda_solver_policy::sparse_solver_device;
using lss_core_cuda_solver_policy::sparse_solver_device_cholesky;
using lss_core_cuda_solver_policy::sparse_solver_device_qr;
using lss_core_cuda_solver_policy::sparse_solver_host;
using lss_core_cuda_solver_policy::sparse_solver_host_cholesky;
using lss_core_cuda_solver_policy::sparse_solver_host_lu;
using lss_core_cuda_solver_policy::sparse_solver_host_qr;
using lss_enumerations::factorization_enum;
using lss_enumerations::flat_matrix_sort_enum;
using lss_enumerations::memory_space_enum;
using lss_helpers::real_sparse_solver_cuda_helpers;
using lss_utility::sptr_t;
using lss_utility::uptr_t;

template <memory_space_enum memory_space, typename fp_type, template <typename, typename> typename container,
          typename allocator>
class real_sparse_solver_cuda
{
};

// =============================================================================
// ========== real_sparse_solver_cuda partial specialization for HOST ==========
// =============================================================================

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class real_sparse_solver_cuda<memory_space_enum::Host, fp_type, container, allocator>
{
  protected:
    int system_size_;
    uptr_t<flat_matrix<fp_type>> matrix_data_ptr_;

    thrust::host_vector<fp_type> h_matrix_values_;
    thrust::host_vector<fp_type> h_vector_values_; // of systemSize length
    thrust::host_vector<int> h_column_indices_;
    thrust::host_vector<int> h_row_counts_; // of systemSize + 1 length

    void build_csr();
    explicit real_sparse_solver_cuda()
    {
    }

  public:
    typedef fp_type value_type;
    void initialize(std::size_t system_size);

    explicit real_sparse_solver_cuda(int system_size) : system_size_{system_size}
    {
    }
    virtual ~real_sparse_solver_cuda()
    {
    }

    real_sparse_solver_cuda(real_sparse_solver_cuda const &) = delete;
    real_sparse_solver_cuda &operator=(real_sparse_solver_cuda const &) = delete;
    real_sparse_solver_cuda(real_sparse_solver_cuda &&) = delete;
    real_sparse_solver_cuda &operator=(real_sparse_solver_cuda &&) = delete;

    inline std::size_t non_zero_elements() const
    {
        return matrix_data_ptr_->size();
    }

    inline void set_rhs(container<fp_type, allocator> const &rhs)
    {
        LSS_ASSERT(rhs.size() == system_size_, " rhs has incorrect size");
        thrust::copy(rhs.begin(), rhs.end(), h_vector_values_.begin());
    }

    inline void set_rhs_value(std::size_t idx, fp_type value)
    {
        LSS_ASSERT((idx < system_size_), "idx is outside range");
        h_vector_values_[idx] = value;
    }

    inline void set_flat_sparse_matrix(flat_matrix<fp_type> matrix)
    {
        LSS_ASSERT(matrix.columns() == system_size_, " Incorrect number of columns");
        LSS_ASSERT(matrix.rows() == system_size_, " Incorrect number of rows");
        matrix_data_ptr_ = std::make_unique<flat_matrix<fp_type>>(std::move(matrix));
    }

    void solve(container<fp_type, allocator> &solution,
               factorization_enum factorization = factorization_enum::QRMethod);

    container<fp_type, allocator> const solve(factorization_enum factorization = factorization_enum::QRMethod);
};

template <typename fp_type>
using host_real_sparse_solver_cuda_ptr =
    sptr_t<real_sparse_solver_cuda<memory_space_enum::Host, fp_type, std::vector, std::allocator<fp_type>>>;

// =============================================================================
// ========== real_sparse_solver_cuda partial specialization for DEVICE
// ===========
// =============================================================================

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class real_sparse_solver_cuda<memory_space_enum::Device, fp_type, container, allocator>
{
  protected:
    int system_size_;
    uptr_t<flat_matrix<fp_type>> matrix_data_ptr_;

    thrust::host_vector<fp_type> h_matrix_values_;
    thrust::host_vector<fp_type> h_vector_values_; // of systemSize length
    thrust::host_vector<int> h_column_indices_;
    thrust::host_vector<int> h_row_counts_; // of systemSize + 1 length

    void build_csr();
    explicit real_sparse_solver_cuda()
    {
    }

  public:
    typedef fp_type value_type;
    explicit real_sparse_solver_cuda(int system_size) : system_size_{system_size}
    {
    }
    virtual ~real_sparse_solver_cuda()
    {
    }

    real_sparse_solver_cuda(real_sparse_solver_cuda const &) = delete;
    real_sparse_solver_cuda &operator=(real_sparse_solver_cuda const &) = delete;
    real_sparse_solver_cuda(real_sparse_solver_cuda &&) = delete;
    real_sparse_solver_cuda &operator=(real_sparse_solver_cuda &&) = delete;

    void initialize(std::size_t system_size);

    inline std::size_t non_zero_elements() const
    {
        LSS_VERIFY(matrix_data_ptr_, "flat_matrix has not been provided.");
        return matrix_data_ptr_->size();
    }

    inline void set_rhs(container<fp_type, allocator> const &rhs)
    {
        LSS_ASSERT(rhs.size() == system_size_, " rhs has incorrect size");
        thrust::copy(rhs.begin(), rhs.end(), h_vector_values_.begin());
    }

    inline void set_rhs_value(std::size_t idx, fp_type value)
    {
        LSS_ASSERT((idx < system_size_), "idx is outside range");
        h_vector_values_[idx] = value;
    }

    inline void set_flat_sparse_matrix(flat_matrix<fp_type> matrix)
    {
        LSS_ASSERT(matrix.columns() == system_size_, " Incorrect number of columns");
        LSS_ASSERT(matrix.rows() == system_size_, " Incorrect number of rows");
        matrix_data_ptr_ = std::make_unique<flat_matrix<fp_type>>(std::move(matrix));
    }

    void solve(container<fp_type, allocator> &solution,
               factorization_enum factorization = factorization_enum::QRMethod);

    container<fp_type, allocator> const solve(factorization_enum factorization = factorization_enum::QRMethod);
};

template <typename fp_type>
using device_real_sparse_solver_cuda_ptr =
    sptr_t<real_sparse_solver_cuda<memory_space_enum::Device, fp_type, std::vector, std::allocator<fp_type>>>;

} // namespace lss_core_cuda_solver

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void lss_core_cuda_solver::real_sparse_solver_cuda<lss_enumerations::memory_space_enum::Host, fp_type, container,
                                                   allocator>::initialize(std::size_t system_size)
{
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

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void lss_core_cuda_solver::real_sparse_solver_cuda<lss_enumerations::memory_space_enum::Device, fp_type, container,
                                                   allocator>::initialize(std::size_t system_size)
{
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

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void lss_core_cuda_solver::real_sparse_solver_cuda<lss_enumerations::memory_space_enum::Host, fp_type, container,
                                                   allocator>::build_csr()
{
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

    for (int i = 0; i < nonZeroSize; ++i)
    {
        if (lastRow < std::get<0>(matrix_data_ptr_->at(i)))
        {
            h_row_counts_[nRowElement++] = i;
            lastRow++;
        }

        h_column_indices_[i] = std::get<1>(matrix_data_ptr_->at(i));
        h_matrix_values_[i] = std::get<2>(matrix_data_ptr_->at(i));
    }
    h_row_counts_[nRowElement] = nonZeroSize;
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void lss_core_cuda_solver::real_sparse_solver_cuda<lss_enumerations::memory_space_enum::Device, fp_type, container,
                                                   allocator>::build_csr()
{
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

    for (std::size_t i = 0; i < non_zero_size; ++i)
    {
        if (lastRow < std::get<0>(matrix_data_ptr_->at(i)))
        {
            h_row_counts_[nRowElement++] = i;
            lastRow++;
        }

        h_column_indices_[i] = std::get<1>(matrix_data_ptr_->at(i));
        h_matrix_values_[i] = std::get<2>(matrix_data_ptr_->at(i));
    }
    h_row_counts_[nRowElement] = non_zero_size;
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void lss_core_cuda_solver::real_sparse_solver_cuda<lss_enumerations::memory_space_enum::Host, fp_type, container,
                                                   allocator>::solve(container<fp_type, allocator> &solution,
                                                                     factorization_enum factorization)
{
    build_csr();

    // get the non-zero size:
    int const non_zero_size = non_zero_elements();

    // integer for holding index of row where singularity occurs:
    int singular_idx{0};

    // prepare container for solution:
    thrust::host_vector<fp_type> h_solution(system_size_);

    // get the raw host pointers
    fp_type *h_mat_vals = thrust::raw_pointer_cast(h_matrix_values_.data());
    fp_type *h_rhs_vals = thrust::raw_pointer_cast(h_vector_values_.data());
    fp_type *h_sol = thrust::raw_pointer_cast(h_solution.data());
    int *h_col = thrust::raw_pointer_cast(h_column_indices_.data());
    int *h_row = thrust::raw_pointer_cast(h_row_counts_.data());

    // create the helpers:
    real_sparse_solver_cuda_helpers helpers;
    helpers.initialize();
    // call the sparse_solver_host_policy:
    if (factorization == factorization_enum::QRMethod)
    {
        sparse_solver_host_qr<fp_type>::solve(helpers.get_solver_handle(), helpers.get_matrix_descriptor(),
                                              system_size_, non_zero_size, h_mat_vals, h_row, h_col, h_rhs_vals, 0.0, 0,
                                              h_sol, &singular_idx);
    }
    else if (factorization == factorization_enum::LUMethod)
    {
        sparse_solver_host_lu<fp_type>::solve(helpers.get_solver_handle(), helpers.get_matrix_descriptor(),
                                              system_size_, non_zero_size, h_mat_vals, h_row, h_col, h_rhs_vals, 0.0, 0,
                                              h_sol, &singular_idx);
    }
    else if (factorization == factorization_enum::CholeskyMethod)
    {
        sparse_solver_host_cholesky<fp_type>::solve(helpers.get_solver_handle(), helpers.get_matrix_descriptor(),
                                                    system_size_, non_zero_size, h_mat_vals, h_row, h_col, h_rhs_vals,
                                                    0.0, 0, h_sol, &singular_idx);
    }
    else
    {
        throw std::exception("factorization not known");
    }

    LSS_ASSERT(singular_idx < 0, "Sparse matrix is singular at row: " << singular_idx << "\n");
    thrust::copy(h_solution.begin(), h_solution.end(), solution.begin());
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void lss_core_cuda_solver::real_sparse_solver_cuda<lss_enumerations::memory_space_enum::Device, fp_type, container,
                                                   allocator>::solve(container<fp_type, allocator> &solution,
                                                                     factorization_enum factorization)
{
    build_csr();

    // get the non-zero size:
    int const non_zero_size = non_zero_elements();

    // integer for holding index of row where singularity occurs:
    int singular_idx{0};

    // prepare container for solution:
    thrust::device_vector<fp_type> d_solution(system_size_);

    // copy to the device constainers:
    thrust::device_vector<fp_type> d_matrix_values = h_matrix_values_;
    thrust::device_vector<fp_type> d_vector_values = h_vector_values_;
    thrust::device_vector<int> d_column_indices = h_column_indices_;
    thrust::device_vector<int> d_row_counts = h_row_counts_;

    // get the raw host pointers
    fp_type *d_mat_vals = thrust::raw_pointer_cast(d_matrix_values.data());
    fp_type *d_rhs_vals = thrust::raw_pointer_cast(d_vector_values.data());
    fp_type *d_sol = thrust::raw_pointer_cast(d_solution.data());
    int *d_col = thrust::raw_pointer_cast(d_column_indices.data());
    int *d_row = thrust::raw_pointer_cast(d_row_counts.data());

    // create the helpers:
    real_sparse_solver_cuda_helpers helpers;
    helpers.initialize();
    // call the sparse_solver_device_policy:
    if (factorization == factorization_enum::QRMethod)
    {
        sparse_solver_device_qr<fp_type>::solve(helpers.get_solver_handle(), helpers.get_matrix_descriptor(),
                                                system_size_, non_zero_size, d_mat_vals, d_row, d_col, d_rhs_vals, 0.0,
                                                0, d_sol, &singular_idx);
    }
    else if (factorization == factorization_enum::CholeskyMethod)
    {
        sparse_solver_device_cholesky<fp_type>::solve(helpers.get_solver_handle(), helpers.get_matrix_descriptor(),
                                                      system_size_, non_zero_size, d_mat_vals, d_row, d_col, d_rhs_vals,
                                                      0.0, 0, d_sol, &singular_idx);
    }
    else if (factorization == factorization_enum::LUMethod)
    {
        throw std::exception("factorization not supported on device.");
    }
    else
    {
        throw std::exception("factorization not known");
    }

    LSS_ASSERT(singular_idx < 0, "Sparse matrix is singular at row: " << singular_idx << "\n");
    thrust::copy(d_solution.begin(), d_solution.end(), solution.begin());
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
container<fp_type, allocator> const lss_core_cuda_solver::real_sparse_solver_cuda<
    lss_enumerations::memory_space_enum::Host, fp_type, container, allocator>::solve(factorization_enum factorization)
{
    build_csr();

    // get the non-zero size:
    int const non_zero_size = non_zero_elements();

    // integer for holding index of row where singularity occurs:
    int singular_idx{0};

    // prepare container for solution:
    thrust::host_vector<fp_type> h_solution(system_size_);

    // get the raw host pointers
    fp_type *h_mat_vals = thrust::raw_pointer_cast(h_matrix_values_.data());
    fp_type *h_rhs_vals = thrust::raw_pointer_cast(h_vector_values_.data());
    fp_type *h_sol = thrust::raw_pointer_cast(h_solution.data());
    int *h_col = thrust::raw_pointer_cast(h_column_indices_.data());
    int *h_row = thrust::raw_pointer_cast(h_row_counts_.data());

    // create the helpers:
    real_sparse_solver_cuda_helpers helpers;
    helpers.initialize();
    // call the sparse_solver_host_policy:
    if (factorization == factorization_enum::QRMethod)
    {
        sparse_solver_host_qr<fp_type>::solve(helpers.get_solver_handle(), helpers.get_matrix_descriptor(),
                                              system_size_, non_zero_size, h_mat_vals, h_row, h_col, h_rhs_vals, 0.0, 0,
                                              h_sol, &singular_idx);
    }
    else if (factorization == factorization_enum::LUMethod)
    {
        sparse_solver_host_lu<fp_type>::solve(helpers.get_solver_handle(), helpers.get_matrix_descriptor(),
                                              system_size_, non_zero_size, h_mat_vals, h_row, h_col, h_rhs_vals, 0.0, 0,
                                              h_sol, &singular_idx);
    }
    else if (factorization == factorization_enum::CholeskyMethod)
    {
        sparse_solver_host_cholesky<fp_type>::solve(helpers.get_solver_handle(), helpers.get_matrix_descriptor(),
                                                    system_size_, non_zero_size, h_mat_vals, h_row, h_col, h_rhs_vals,
                                                    0.0, 0, h_sol, &singular_idx);
    }
    else
    {
        throw std::exception("factorization not known");
    }

    LSS_ASSERT(singular_idx < 0, "Sparse matrix is singular at row: " << singular_idx << "\n");
    container<fp_type, allocator> solution(h_solution.size());
    thrust::copy(h_solution.begin(), h_solution.end(), solution.begin());
    return solution;
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
container<fp_type, allocator> const lss_core_cuda_solver::real_sparse_solver_cuda<
    lss_enumerations::memory_space_enum::Device, fp_type, container, allocator>::solve(factorization_enum factorization)
{
    build_csr();

    // get the non-zero size:
    int const non_zero_size = non_zero_elements();

    // integer for holding index of row where singularity occurs:
    int singular_idx{0};

    // prepare container for solution:
    thrust::device_vector<fp_type> d_solution(system_size_);

    // copy to the device constainers:
    thrust::device_vector<fp_type> d_matrix_values = h_matrix_values_;
    thrust::device_vector<fp_type> d_vector_values = h_vector_values_;
    thrust::device_vector<int> d_column_indices = h_column_indices_;
    thrust::device_vector<int> d_row_counts = h_row_counts_;

    // get the raw host pointers
    fp_type *d_mat_vals = thrust::raw_pointer_cast(d_matrix_values.data());
    fp_type *d_rhs_vals = thrust::raw_pointer_cast(d_vector_values.data());
    fp_type *d_sol = thrust::raw_pointer_cast(d_solution.data());
    int *d_col = thrust::raw_pointer_cast(d_column_indices.data());
    int *d_row = thrust::raw_pointer_cast(d_row_counts.data());

    // create the helpers:
    real_sparse_solver_cuda_helpers helpers;
    helpers.initialize();
    // call the sparse_solver_device_policy:
    if (factorization == factorization_enum::QRMethod)
    {
        sparse_solver_device_qr<fp_type>::solve(helpers.get_solver_handle(), helpers.get_matrix_descriptor(),
                                                system_size_, non_zero_size, d_mat_vals, d_row, d_col, d_rhs_vals, 0.0,
                                                0, d_sol, &singular_idx);
    }
    else if (factorization == factorization_enum::CholeskyMethod)
    {
        sparse_solver_device_cholesky<fp_type>::solve(helpers.get_solver_handle(), helpers.get_matrix_descriptor(),
                                                      system_size_, non_zero_size, d_mat_vals, d_row, d_col, d_rhs_vals,
                                                      0.0, 0, d_sol, &singular_idx);
    }
    else if (factorization == factorization_enum::LUMethod)
    {
        throw std::exception("factorization not supported on device.");
    }
    else
    {
        throw std::exception("factorization not known");
    }

    LSS_ASSERT(singular_idx < 0, "Sparse matrix is singular at row: " << singular_idx << "\n");
    container<fp_type, allocator> solution(system_size_);
    thrust::copy(d_solution.begin(), d_solution.end(), solution.begin());

    return solution;
}

#endif ///_LSS_CORE_CUDA_SOLVER_HPP_
