#pragma once
#if !defined(_LSS_CORE_CUDA_SOLVER_POLICY_HPP_)
#define _LSS_CORE_CUDA_SOLVER_POLICY_HPP_

#include <cusolverSp.h>

#include <type_traits>

#include "common/lss_macros.hpp"

namespace lss_core_cuda_solver_policy
{

/* Base for sparse factorization on Host */

template <typename T> struct sparse_solver_host
{
};

/* Sparse QR factorization on Host */
template <typename T> struct sparse_solver_host_qr : public sparse_solver_host<T>
{
  private:
    // for T = double
    static void _solve_impl(cusolverSpHandle_t handle, cusparseMatDescr_t mat_desc, int const system_size,
                            int const non_zero_size, T const *h_mat_vals, int const *h_row_counts,
                            int const *h_col_indices, T const *h_rhs, T tol, int reorder, T *h_solution,
                            int *singularity, std::true_type);

    // for T = float
    static void _solve_impl(cusolverSpHandle_t handle, cusparseMatDescr_t mat_desc, int const system_size,
                            int const non_zero_size, T const *h_mat_vals, int const *h_row_counts,
                            int const *h_col_indices, T const *h_rhs, T tol, int reorder, T *h_solution,
                            int *singularity, std::false_type);

  public:
    static void solve(cusolverSpHandle_t handle, cusparseMatDescr_t mat_desc, int const system_size,
                      int const non_zero_size, T const *h_mat_vals, int const *h_row_counts, int const *h_col_indices,
                      T const *h_rhs, T tol, int reorder, T *h_solution, int *singularity)
    {
        _solve_impl(handle, mat_desc, system_size, non_zero_size, h_mat_vals, h_row_counts, h_col_indices, h_rhs, tol,
                    reorder, h_solution, singularity, std::is_same<T, double>());
    }
};

/* Sparse LU factorization on Host */

template <typename T> struct sparse_solver_host_lu : public sparse_solver_host<T>
{
  private:
    // for T = double
    static void _solve_impl(cusolverSpHandle_t handle, cusparseMatDescr_t mat_desc, int const system_size,
                            int const non_zero_size, T const *h_mat_vals, int const *h_row_counts,
                            int const *h_col_indices, T const *h_rhs, T tol, int reorder, T *h_solution,
                            int *singularity, std::true_type);

    // for T = float
    static void _solve_impl(cusolverSpHandle_t handle, cusparseMatDescr_t mat_desc, int const system_size,
                            int const non_zero_size, T const *h_mat_vals, int const *h_row_counts,
                            int const *h_col_indices, T const *h_rhs, T tol, int reorder, T *h_solution,
                            int *singularity, std::false_type);

  public:
    static void solve(cusolverSpHandle_t handle, cusparseMatDescr_t mat_desc, int const system_size,
                      int const non_zero_size, T const *h_mat_vals, int const *h_row_counts, int const *h_col_indices,
                      T const *h_rhs, T tol, int reorder, T *h_solution, int *singularity)
    {
        _solve_impl(handle, mat_desc, system_size, non_zero_size, h_mat_vals, h_row_counts, h_col_indices, h_rhs, tol,
                    reorder, h_solution, singularity, std::is_same<T, double>());
    }
};

/* Sparse Cholesky factorization on Host */

template <typename T> struct sparse_solver_host_cholesky : public sparse_solver_host<T>
{
  private:
    // for T = double
    static void _solve_impl(cusolverSpHandle_t handle, cusparseMatDescr_t mat_desc, int const system_size,
                            int const non_zero_size, T const *h_mat_vals, int const *h_row_counts,
                            int const *h_col_indices, T const *h_rhs, T tol, int reorder, T *h_solution,
                            int *singularity, std::true_type);

    // for T = float
    static void _solve_impl(cusolverSpHandle_t handle, cusparseMatDescr_t mat_desc, int const system_size,
                            int const non_zero_size, T const *h_mat_vals, int const *h_row_counts,
                            int const *h_col_indices, T const *h_rhs, T tol, int reorder, T *h_solution,
                            int *singularity, std::false_type);

  public:
    static void solve(cusolverSpHandle_t handle, cusparseMatDescr_t mat_desc, int const system_size,
                      int const non_zero_size, T const *h_mat_vals, int const *h_row_counts, int const *h_col_indices,
                      T const *h_rhs, T tol, int reorder, T *h_solution, int *singularity)
    {
        _solve_impl(handle, mat_desc, system_size, non_zero_size, h_mat_vals, h_row_counts, h_col_indices, h_rhs, tol,
                    reorder, h_solution, singularity, std::is_same<T, double>());
    }
};

/* Base for sparse factorization on Device */

template <typename T> struct sparse_solver_device
{
};

/* Sparse QR factorization on Device */

template <typename T> struct sparse_solver_device_qr : public sparse_solver_device<T>
{
  private:
    // for T = double
    static void _solve_impl(cusolverSpHandle_t handle, cusparseMatDescr_t mat_desc, int const system_size,
                            int const non_zero_size, T const *d_mat_vals, int const *d_row_counts,
                            int const *d_col_indices, T const *d_rhs, T tol, int reorder, T *d_solution,
                            int *singularity, std::true_type);

    // for T = float
    static void _solve_impl(cusolverSpHandle_t handle, cusparseMatDescr_t mat_desc, int const system_size,
                            int const non_zero_size, T const *d_mat_vals, int const *d_row_counts,
                            int const *d_col_indices, T const *d_rhs, T tol, int reorder, T *d_solution,
                            int *singularity, std::false_type);

  public:
    static void solve(cusolverSpHandle_t handle, cusparseMatDescr_t mat_desc, int const system_size,
                      int const non_zero_size, T const *d_mat_vals, int const *d_row_counts, int const *d_col_indices,
                      T const *d_rhs, T tol, int reorder, T *d_solution, int *singularity)
    {
        _solve_impl(handle, mat_desc, system_size, non_zero_size, d_mat_vals, d_row_counts, d_col_indices, d_rhs, tol,
                    reorder, d_solution, singularity, std::is_same<T, double>());
    }
};

/* Sparse Cholesky factorization on Device */

template <typename T> struct sparse_solver_device_cholesky : public sparse_solver_device<T>
{
  private:
    // for T = double
    static void _solve_impl(cusolverSpHandle_t handle, cusparseMatDescr_t mat_desc, int const system_size,
                            int const non_zero_size, T const *d_mat_vals, int const *d_row_counts,
                            int const *d_col_indices, T const *d_rhs, T tol, int reorder, T *d_solution,
                            int *singularity, std::true_type);

    // for T = float
    static void _solve_impl(cusolverSpHandle_t handle, cusparseMatDescr_t mat_desc, int const system_size,
                            int const non_zero_size, T const *d_mat_vals, int const *d_row_counts,
                            int const *d_col_indices, T const *d_rhs, T tol, int reorder, T *d_solution,
                            int *singularity, std::false_type);

  public:
    static void solve(cusolverSpHandle_t handle, cusparseMatDescr_t mat_desc, int const system_size,
                      int const non_zero_size, T const *d_mat_vals, int const *d_row_counts, int const *d_col_indices,
                      T const *d_rhs, T tol, int reorder, T *d_solution, int *singularity)
    {
        _solve_impl(handle, mat_desc, system_size, non_zero_size, d_mat_vals, d_row_counts, d_col_indices, d_rhs, tol,
                    reorder, d_solution, singularity, std::is_same<T, double>());
    }
};

} // namespace lss_core_cuda_solver_policy

/* Sparse QR factorization on HOST */

template <typename T>
void lss_core_cuda_solver_policy::sparse_solver_host_qr<T>::_solve_impl(
    cusolverSpHandle_t handle, cusparseMatDescr_t mat_desc, int const system_size, int const non_zero_size,
    T const *h_mat_vals, int const *h_row_counts, int const *h_col_indices, T const *h_rhs, T tol, int reorder,
    T *h_solution, int *singularity, std::true_type)
{
    CUSOLVER_STATUS(cusolverSpDcsrlsvqrHost(handle, system_size, non_zero_size, mat_desc, h_mat_vals, h_row_counts,
                                            h_col_indices, h_rhs, tol, reorder, h_solution, singularity));
}

template <typename T>
void lss_core_cuda_solver_policy::sparse_solver_host_qr<T>::_solve_impl(
    cusolverSpHandle_t handle, cusparseMatDescr_t matDesc, int const systemSize, int const nonZeroSize,
    T const *h_matVals, int const *h_rowCounts, int const *h_colIndices, T const *h_rhs, T tol, int reorder,
    T *h_solution, int *singularity, std::false_type)
{
    CUSOLVER_STATUS(cusolverSpScsrlsvqrHost(handle, systemSize, nonZeroSize, matDesc, h_matVals, h_rowCounts,
                                            h_colIndices, h_rhs, tol, reorder, h_solution, singularity));
}

/* Sparse LU factorization on HOST */

template <typename T>
void lss_core_cuda_solver_policy::sparse_solver_host_lu<T>::_solve_impl(
    cusolverSpHandle_t handle, cusparseMatDescr_t mat_desc, int const system_size, int const non_zero_size,
    T const *h_mat_vals, int const *h_row_counts, int const *h_col_indices, T const *h_rhs, T tol, int reorder,
    T *h_solution, int *singularity, std::true_type)
{
    CUSOLVER_STATUS(cusolverSpDcsrlsvluHost(handle, system_size, non_zero_size, mat_desc, h_mat_vals, h_row_counts,
                                            h_col_indices, h_rhs, tol, reorder, h_solution, singularity));
}

template <typename T>
void lss_core_cuda_solver_policy::sparse_solver_host_lu<T>::_solve_impl(
    cusolverSpHandle_t handle, cusparseMatDescr_t mat_desc, int const system_size, int const non_zero_size,
    T const *h_mat_vals, int const *h_row_counts, int const *h_col_indices, T const *h_rhs, T tol, int reorder,
    T *h_solution, int *singularity, std::false_type)
{
    CUSOLVER_STATUS(cusolverSpScsrlsvluHost(handle, system_size, non_zero_size, mat_desc, h_mat_vals, h_row_counts,
                                            h_col_indices, h_rhs, tol, reorder, h_solution, singularity));
}

/* Sparse Cholesky factorization on HOST */

template <typename T>
void lss_core_cuda_solver_policy::sparse_solver_host_cholesky<T>::_solve_impl(
    cusolverSpHandle_t handle, cusparseMatDescr_t mat_desc, int const system_size, int const non_zero_size,
    T const *h_mat_vals, int const *h_row_counts, int const *h_col_indices, T const *h_rhs, T tol, int reorder,
    T *h_solution, int *singularity, std::true_type)
{
    CUSOLVER_STATUS(cusolverSpDcsrlsvcholHost(handle, system_size, non_zero_size, mat_desc, h_mat_vals, h_row_counts,
                                              h_col_indices, h_rhs, tol, reorder, h_solution, singularity));
}

template <typename T>
void lss_core_cuda_solver_policy::sparse_solver_host_cholesky<T>::_solve_impl(
    cusolverSpHandle_t handle, cusparseMatDescr_t mat_desc, int const system_size, int const non_zero_size,
    T const *h_mat_vals, int const *h_row_counts, int const *h_col_indices, T const *h_rhs, T tol, int reorder,
    T *h_solution, int *singularity, std::false_type)
{
    CUSOLVER_STATUS(cusolverSpScsrlsvcholHost(handle, system_size, non_zero_size, mat_desc, h_mat_vals, h_row_counts,
                                              h_col_indices, h_rhs, tol, reorder, h_solution, singularity));
}

/* Sparse QR factorization on DEVICE */

template <typename T>
void lss_core_cuda_solver_policy::sparse_solver_device_qr<T>::_solve_impl(
    cusolverSpHandle_t handle, cusparseMatDescr_t mat_desc, int const system_size, int const non_zero_size,
    T const *d_mat_vals, int const *d_row_counts, int const *d_col_indices, T const *d_rhs, T tol, int reorder,
    T *d_solution, int *singularity, std::true_type)
{
    CUSOLVER_STATUS(cusolverSpDcsrlsvqr(handle, system_size, non_zero_size, mat_desc, d_mat_vals, d_row_counts,
                                        d_col_indices, d_rhs, tol, reorder, d_solution, singularity));
}

template <typename T>
void lss_core_cuda_solver_policy::sparse_solver_device_qr<T>::_solve_impl(
    cusolverSpHandle_t handle, cusparseMatDescr_t mat_desc, int const system_size, int const non_zero_size,
    T const *d_mat_vals, int const *d_row_counts, int const *d_col_indices, T const *d_rhs, T tol, int reorder,
    T *d_solution, int *singularity, std::false_type)
{
    CUSOLVER_STATUS(cusolverSpScsrlsvqr(handle, system_size, non_zero_size, mat_desc, d_mat_vals, d_row_counts,
                                        d_col_indices, d_rhs, tol, reorder, d_solution, singularity));
}

/* Sparse Cholesky factorization on DEVICE */

template <typename T>
void lss_core_cuda_solver_policy::sparse_solver_device_cholesky<T>::_solve_impl(
    cusolverSpHandle_t handle, cusparseMatDescr_t mat_desc, int const system_size, int const non_zero_size,
    T const *d_mat_vals, int const *d_row_counts, int const *d_col_indices, T const *d_rhs, T tol, int reorder,
    T *d_solution, int *singularity, std::true_type)
{
    CUSOLVER_STATUS(cusolverSpDcsrlsvchol(handle, system_size, non_zero_size, mat_desc, d_mat_vals, d_row_counts,
                                          d_col_indices, d_rhs, tol, reorder, d_solution, singularity));
}

template <typename T>
void lss_core_cuda_solver_policy::sparse_solver_device_cholesky<T>::_solve_impl(
    cusolverSpHandle_t handle, cusparseMatDescr_t mat_desc, int const system_size, int const non_zero_size,
    T const *d_mat_vals, int const *d_row_counts, int const *d_col_indices, T const *d_rhs, T tol, int reorder,
    T *d_solution, int *singularity, std::false_type)
{
    CUSOLVER_STATUS(cusolverSpScsrlsvchol(handle, system_size, non_zero_size, mat_desc, d_mat_vals, d_row_counts,
                                          d_col_indices, d_rhs, tol, reorder, d_solution, singularity));
}

#endif ///_LSS_CORE_CUDA_SOLVER_POLICY_HPP_
