#pragma once
#if !defined(_LSS_DENSE_SOLVERS_POLICY_HPP_)
#define _LSS_DENSE_SOLVERS_POLICY_HPP_
#pragma warning(disable : 4267)

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <type_traits>

#include "common/lss_macros.hpp"

namespace lss_dense_solvers_policy
{

template <typename T> struct dense_solver_device
{
};

/* Dense QR factorization */

template <typename T> struct dense_solver_qr : public dense_solver_device<T>
{
  private:
    // T = double
    static void _solve_impl(cusolverDnHandle_t cusolver_handle, cublasHandle_t cublas_handle, std::size_t n,
                            const T *d_Acopy, std::size_t lda, const T *d_b, T *d_x, std::true_type);
    // T = float
    static void _solve_impl(cusolverDnHandle_t cusolver_handle, cublasHandle_t cublas_handle, std::size_t n,
                            const T *d_Acopy, std::size_t lda, const T *d_b, T *d_x, std::false_type);

  public:
    static void solve(cusolverDnHandle_t cusolver_handle, cublasHandle_t cublas_handle, std::size_t n, const T *d_Acopy,
                      std::size_t lda, const T *d_b, T *d_x)
    {
        _solve_impl(cusolver_handle, cublas_handle, n, d_Acopy, lda, d_b, d_x, std::is_same<T, double>());
    }
};

/* Dense LU factorization */

template <typename T> struct dense_solver_lu : public dense_solver_device<T>
{
  private:
    // T = double
    static void _solve_impl(cusolverDnHandle_t cusolver_handle, cublasHandle_t cublas_handle, std::size_t n,
                            const T *d_Acopy, std::size_t lda, const T *d_b, T *d_x, std::true_type);
    // T = float
    static void _solve_impl(cusolverDnHandle_t cusolver_handle, cublasHandle_t cublas_handle, std::size_t n,
                            const T *d_Acopy, std::size_t lda, const T *d_b, T *d_x, std::false_type);

  public:
    static void solve(cusolverDnHandle_t cusolver_handle, cublasHandle_t cublas_handle, std::size_t n, const T *d_Acopy,
                      std::size_t lda, const T *d_b, T *d_x)
    {
        _solve_impl(cusolver_handle, cublas_handle, n, d_Acopy, lda, d_b, d_x, std::is_same<T, double>());
    }
};

/* Dense Cholesky factorization */

template <typename T> struct dense_solver_cholesky : public dense_solver_device<T>
{
  private:
    // T = double
    static void _solve_impl(cusolverDnHandle_t cusolver_handle, cublasHandle_t cublas_handle, std::size_t n,
                            const T *d_Acopy, std::size_t lda, const T *d_b, T *d_x, std::true_type);
    // T = float
    static void _solve_impl(cusolverDnHandle_t cusolver_handle, cublasHandle_t cublas_handle, std::size_t n,
                            const T *d_Acopy, std::size_t lda, const T *d_b, T *d_x, std::false_type);

  public:
    static void solve(cusolverDnHandle_t cusolver_handle, cublasHandle_t cublas_handle, std::size_t n, const T *d_Acopy,
                      std::size_t lda, const T *d_b, T *d_x)
    {
        _solve_impl(cusolver_handle, cublas_handle, n, d_Acopy, lda, d_b, d_x, std::is_same<T, double>());
    }
};

} // namespace lss_dense_solvers_policy

template <typename T>
void lss_dense_solvers_policy::dense_solver_qr<T>::_solve_impl(cusolverDnHandle_t cusolver_handle,
                                                               cublasHandle_t cublas_handle, std::size_t n,
                                                               const T *d_Acopy, std::size_t lda, const T *d_b, T *d_x,
                                                               std::true_type)
{
    int bufferSize = 0;
    int bufferSize_geqrf = 0;
    int bufferSize_ormqr = 0;
    int *info = NULL;
    T *buffer = NULL;
    T *A = NULL;
    T *tau = NULL;
    int h_info = 0;
    const T one = 1.0;

    CUSOLVER_STATUS(cusolverDnDgeqrf_bufferSize(cusolver_handle, n, n, (T *)d_Acopy, lda, &bufferSize_geqrf));
    CUSOLVER_STATUS(cusolverDnDormqr_bufferSize(cusolver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, n, 1, n, A, lda, NULL,
                                                d_x, n, &bufferSize_ormqr));

    bufferSize = (bufferSize_geqrf > bufferSize_ormqr) ? bufferSize_geqrf : bufferSize_ormqr;

    CUDA_ERROR(cudaMalloc(&info, sizeof(int)));
    CUDA_ERROR(cudaMalloc(&buffer, sizeof(T) * bufferSize));
    CUDA_ERROR(cudaMalloc(&A, sizeof(T) * lda * n));
    CUDA_ERROR(cudaMalloc((void **)&tau, sizeof(T) * n));

    // prepare a copy of A because getrf will overwrite A with L
    CUDA_ERROR(cudaMemcpy(A, d_Acopy, sizeof(T) * lda * n, cudaMemcpyDeviceToDevice));

    CUDA_ERROR(cudaMemset(info, 0, sizeof(int)));

    // compute QR factorization
    CUSOLVER_STATUS(cusolverDnDgeqrf(cusolver_handle, n, n, A, lda, tau, buffer, bufferSize, info));

    CUDA_ERROR(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    LSS_ASSERT(h_info == 0, "LU factorization failed\n");

    CUDA_ERROR(cudaMemcpy(d_x, d_b, sizeof(T) * n, cudaMemcpyDeviceToDevice));

    // compute Q^T*b
    CUSOLVER_STATUS(cusolverDnDormqr(cusolver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, n, 1, n, A, lda, tau, d_x, n,
                                     buffer, bufferSize, info));

    // x = R \ Q^T*b
    CUBLAS_STATUS(cublasDtrsm_v2(cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                 CUBLAS_DIAG_NON_UNIT, n, 1, &one, A, lda, d_x, n));
    CUDA_ERROR(cudaDeviceSynchronize());

    if (info)
    {
        CUDA_ERROR(cudaFree(info));
    }
    if (buffer)
    {
        CUDA_ERROR(cudaFree(buffer));
    }
    if (A)
    {
        CUDA_ERROR(cudaFree(A));
    }
    if (tau)
    {
        CUDA_ERROR(cudaFree(tau));
    }
}

template <typename T>
void lss_dense_solvers_policy::dense_solver_qr<T>::_solve_impl(cusolverDnHandle_t cusolver_handle,
                                                               cublasHandle_t cublas_handle, std::size_t n,
                                                               const T *d_Acopy, std::size_t lda, const T *d_b, T *d_x,
                                                               std::false_type)
{
    int bufferSize = 0;
    int bufferSize_geqrf = 0;
    int bufferSize_ormqr = 0;
    int *info = NULL;
    T *buffer = NULL;
    T *A = NULL;
    T *tau = NULL;
    int h_info = 0;
    const T one = 1.0;

    CUSOLVER_STATUS(cusolverDnSgeqrf_bufferSize(cusolver_handle, n, n, (T *)d_Acopy, lda, &bufferSize_geqrf));
    CUSOLVER_STATUS(cusolverDnSormqr_bufferSize(cusolver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, n, 1, n, A, lda, NULL,
                                                d_x, n, &bufferSize_ormqr));

    bufferSize = (bufferSize_geqrf > bufferSize_ormqr) ? bufferSize_geqrf : bufferSize_ormqr;

    CUDA_ERROR(cudaMalloc(&info, sizeof(int)));
    CUDA_ERROR(cudaMalloc(&buffer, sizeof(T) * bufferSize));
    CUDA_ERROR(cudaMalloc(&A, sizeof(T) * lda * n));
    CUDA_ERROR(cudaMalloc((void **)&tau, sizeof(T) * n));

    // prepare a copy of A because getrf will overwrite A with L
    CUDA_ERROR(cudaMemcpy(A, d_Acopy, sizeof(T) * lda * n, cudaMemcpyDeviceToDevice));

    CUDA_ERROR(cudaMemset(info, 0, sizeof(int)));

    // compute QR factorization
    CUSOLVER_STATUS(cusolverDnSgeqrf(cusolver_handle, n, n, A, lda, tau, buffer, bufferSize, info));

    CUDA_ERROR(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    LSS_ASSERT(h_info == 0, "LU factorization failed\n");

    CUDA_ERROR(cudaMemcpy(d_x, d_b, sizeof(T) * n, cudaMemcpyDeviceToDevice));

    // compute Q^T*b
    CUSOLVER_STATUS(cusolverDnSormqr(cusolver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, n, 1, n, A, lda, tau, d_x, n,
                                     buffer, bufferSize, info));

    // x = R \ Q^T*b
    CUBLAS_STATUS(cublasStrsm_v2(cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                 CUBLAS_DIAG_NON_UNIT, n, 1, &one, A, lda, d_x, n));
    CUDA_ERROR(cudaDeviceSynchronize());

    if (info)
    {
        CUDA_ERROR(cudaFree(info));
    }
    if (buffer)
    {
        CUDA_ERROR(cudaFree(buffer));
    }
    if (A)
    {
        CUDA_ERROR(cudaFree(A));
    }
    if (tau)
    {
        CUDA_ERROR(cudaFree(tau));
    }
}

template <typename T>
void lss_dense_solvers_policy::dense_solver_cholesky<T>::_solve_impl(cusolverDnHandle_t cusolver_handle,
                                                                     cublasHandle_t cublas_handle, std::size_t n,
                                                                     const T *d_Acopy, std::size_t lda, const T *d_b,
                                                                     T *d_x, std::true_type)
{
    int bufferSize = 0;
    int *info = NULL;
    T *buffer = NULL;
    T *A = NULL;
    int h_info = 0;

    CUSOLVER_STATUS(
        cusolverDnDpotrf_bufferSize(cusolver_handle, CUBLAS_FILL_MODE_LOWER, n, (T *)d_Acopy, lda, &bufferSize));

    CUDA_ERROR(cudaMalloc(&info, sizeof(int)));
    CUDA_ERROR(cudaMalloc(&buffer, sizeof(T) * bufferSize));
    CUDA_ERROR(cudaMalloc(&A, sizeof(T) * lda * n));

    // prepare a copy of A because potrf will overwrite A with L
    CUDA_ERROR(cudaMemcpy(A, d_Acopy, sizeof(T) * lda * n, cudaMemcpyDeviceToDevice));
    CUDA_ERROR(cudaMemset(info, 0, sizeof(int)));

    CUSOLVER_STATUS(cusolverDnDpotrf(cusolver_handle, CUBLAS_FILL_MODE_LOWER, n, A, lda, buffer, bufferSize, info));

    CUDA_ERROR(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    LSS_ASSERT(h_info == 0, "Cholesky factorization failed\n");

    CUDA_ERROR(cudaMemcpy(d_x, d_b, sizeof(T) * n, cudaMemcpyDeviceToDevice));

    CUSOLVER_STATUS(cusolverDnDpotrs(cusolver_handle, CUBLAS_FILL_MODE_LOWER, n, 1, A, lda, d_x, n, info));

    CUDA_ERROR(cudaDeviceSynchronize());

    if (info)
    {
        CUDA_ERROR(cudaFree(info));
    }
    if (buffer)
    {
        CUDA_ERROR(cudaFree(buffer));
    }
    if (A)
    {
        CUDA_ERROR(cudaFree(A));
    }
}

template <typename T>
void lss_dense_solvers_policy::dense_solver_cholesky<T>::_solve_impl(cusolverDnHandle_t cusolver_handle,
                                                                     cublasHandle_t cublas_handle, std::size_t n,
                                                                     const T *d_Acopy, std::size_t lda, const T *d_b,
                                                                     T *d_x, std::false_type)
{
    int bufferSize = 0;
    int *info = NULL;
    T *buffer = NULL;
    T *A = NULL;
    int h_info = 0;

    CUSOLVER_STATUS(
        cusolverDnSpotrf_bufferSize(cusolver_handle, CUBLAS_FILL_MODE_LOWER, n, (T *)d_Acopy, lda, &bufferSize));

    CUDA_ERROR(cudaMalloc(&info, sizeof(int)));
    CUDA_ERROR(cudaMalloc(&buffer, sizeof(T) * bufferSize));
    CUDA_ERROR(cudaMalloc(&A, sizeof(T) * lda * n));

    // prepare a copy of A because potrf will overwrite A with L
    CUDA_ERROR(cudaMemcpy(A, d_Acopy, sizeof(T) * lda * n, cudaMemcpyDeviceToDevice));
    CUDA_ERROR(cudaMemset(info, 0, sizeof(int)));

    CUSOLVER_STATUS(cusolverDnSpotrf(cusolver_handle, CUBLAS_FILL_MODE_LOWER, n, A, lda, buffer, bufferSize, info));

    CUDA_ERROR(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    LSS_ASSERT(h_info == 0, "Cholesky factorization failed\n");

    CUDA_ERROR(cudaMemcpy(d_x, d_b, sizeof(T) * n, cudaMemcpyDeviceToDevice));

    CUSOLVER_STATUS(cusolverDnSpotrs(cusolver_handle, CUBLAS_FILL_MODE_LOWER, n, 1, A, lda, d_x, n, info));

    CUDA_ERROR(cudaDeviceSynchronize());

    if (info)
    {
        CUDA_ERROR(cudaFree(info));
    }
    if (buffer)
    {
        CUDA_ERROR(cudaFree(buffer));
    }
    if (A)
    {
        CUDA_ERROR(cudaFree(A));
    }
}

template <typename T>
void lss_dense_solvers_policy::dense_solver_lu<T>::_solve_impl(cusolverDnHandle_t cusolver_handle,
                                                               cublasHandle_t cublas_handle, std::size_t n,
                                                               const T *d_Acopy, std::size_t lda, const T *d_b, T *d_x,
                                                               std::true_type)
{
    int bufferSize = 0;
    int *info = NULL;
    T *buffer = NULL;
    T *A = NULL;
    int *ipiv = NULL; // pivoting sequence
    int h_info = 0;

    CUSOLVER_STATUS(cusolverDnDgetrf_bufferSize(cusolver_handle, n, n, (T *)d_Acopy, lda, &bufferSize));

    CUDA_ERROR(cudaMalloc(&info, sizeof(int)));
    CUDA_ERROR(cudaMalloc(&buffer, sizeof(T) * bufferSize));
    CUDA_ERROR(cudaMalloc(&A, sizeof(T) * lda * n));
    CUDA_ERROR(cudaMalloc(&ipiv, sizeof(int) * n));

    // prepare a copy of A because getrf will overwrite A with L
    CUDA_ERROR(cudaMemcpy(A, d_Acopy, sizeof(T) * lda * n, cudaMemcpyDeviceToDevice));
    CUDA_ERROR(cudaMemset(info, 0, sizeof(int)));

    CUSOLVER_STATUS(cusolverDnDgetrf(cusolver_handle, n, n, A, lda, buffer, ipiv, info));
    CUDA_ERROR(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    LSS_ASSERT(h_info == 0, " LU factorizartion failed\n");

    CUDA_ERROR(cudaMemcpy(d_x, d_b, sizeof(T) * n, cudaMemcpyDeviceToDevice));
    CUSOLVER_STATUS(cusolverDnDgetrs(cusolver_handle, CUBLAS_OP_N, n, 1, A, lda, ipiv, d_x, n, info));
    CUDA_ERROR(cudaDeviceSynchronize());

    if (info)
    {
        CUDA_ERROR(cudaFree(info));
    }
    if (buffer)
    {
        CUDA_ERROR(cudaFree(buffer));
    }
    if (A)
    {
        CUDA_ERROR(cudaFree(A));
    }
    if (ipiv)
    {
        CUDA_ERROR(cudaFree(ipiv));
    }
}

template <typename T>
void lss_dense_solvers_policy::dense_solver_lu<T>::_solve_impl(cusolverDnHandle_t cusolver_handle,
                                                               cublasHandle_t cublas_handle, std::size_t n,
                                                               const T *d_Acopy, std::size_t lda, const T *d_b, T *d_x,
                                                               std::false_type)
{
    int bufferSize = 0;
    int *info = NULL;
    T *buffer = NULL;
    T *A = NULL;
    int *ipiv = NULL; // pivoting sequence
    int h_info = 0;

    CUSOLVER_STATUS(cusolverDnSgetrf_bufferSize(cusolver_handle, n, n, (T *)d_Acopy, lda, &bufferSize));

    CUDA_ERROR(cudaMalloc(&info, sizeof(int)));
    CUDA_ERROR(cudaMalloc(&buffer, sizeof(T) * bufferSize));
    CUDA_ERROR(cudaMalloc(&A, sizeof(T) * lda * n));
    CUDA_ERROR(cudaMalloc(&ipiv, sizeof(int) * n));

    // prepare a copy of A because getrf will overwrite A with L
    CUDA_ERROR(cudaMemcpy(A, d_Acopy, sizeof(T) * lda * n, cudaMemcpyDeviceToDevice));
    CUDA_ERROR(cudaMemset(info, 0, sizeof(int)));

    CUSOLVER_STATUS(cusolverDnSgetrf(cusolver_handle, n, n, A, lda, buffer, ipiv, info));
    CUDA_ERROR(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    LSS_ASSERT(h_info == 0, " LU factorizartion failed\n");

    CUDA_ERROR(cudaMemcpy(d_x, d_b, sizeof(T) * n, cudaMemcpyDeviceToDevice));
    CUSOLVER_STATUS(cusolverDnSgetrs(cusolver_handle, CUBLAS_OP_N, n, 1, A, lda, ipiv, d_x, n, info));
    CUDA_ERROR(cudaDeviceSynchronize());

    if (info)
    {
        CUDA_ERROR(cudaFree(info));
    }
    if (buffer)
    {
        CUDA_ERROR(cudaFree(buffer));
    }
    if (A)
    {
        CUDA_ERROR(cudaFree(A));
    }
    if (ipiv)
    {
        CUDA_ERROR(cudaFree(ipiv));
    }
}

#endif ///_LSS_DENSE_SOLVERS_POLICY_HPP_
