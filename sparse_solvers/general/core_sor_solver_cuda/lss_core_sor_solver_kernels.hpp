#pragma once
#if !defined(_LSS_CORE_SOR_SOLVER_KERNELS_HPP_)
#define _LSS_CORE_SOR_SOLVER_KERNELS_HPP_

#include <device_launch_parameters.h>

#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "common/lss_utility.hpp"

#define THREADS_PER_BLOCK 256

namespace lss_core_sor_solver
{

template <typename fp_type>
__global__ void sor_kernel(fp_type const *matrix_vals, std::size_t row_size, std::size_t const *column_idx,
                           fp_type const *rhv_vals, fp_type const *diagonal_vals, std::size_t const *row_start_idx,
                           std::size_t const *row_end_idx, fp_type const omega, fp_type *sol, fp_type *new_sol,
                           fp_type *errors)
{
    std::size_t const tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= row_size)
        return;

    fp_type sigma_1{0.0};
    fp_type sigma_2{0.0};
    fp_type mat_val{0.0};
    const fp_type one = static_cast<fp_type>(1.0);
    fp_type const diag = diagonal_vals[tid];
    std::size_t col_idx{};
    std::size_t const start_idx = row_start_idx[tid];
    std::size_t const end_idx = row_end_idx[tid];
    for (std::size_t c = start_idx; c <= end_idx; ++c)
    {
        mat_val = matrix_vals[c];
        col_idx = column_idx[c];
        if (col_idx < tid)
            sigma_1 += mat_val * new_sol[col_idx];
        if (col_idx > tid)
            sigma_2 += mat_val * sol[col_idx];
    }
    new_sol[tid] = (one - omega) * sol[tid] + ((omega / diag) * (rhv_vals[tid] - sigma_1 - sigma_2));
    errors[tid] = (new_sol[tid] - sol[tid]) * (new_sol[tid] - sol[tid]);
}

} // namespace lss_core_sor_solver

#endif ///_LSS_CORE_SOR_SOLVER_KERNELS_HPP_
