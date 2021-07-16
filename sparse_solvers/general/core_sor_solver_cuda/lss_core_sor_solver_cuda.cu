#include "lss_core_sor_solver_cuda.hpp"
#include "lss_core_sor_solver_kernels.hpp"

namespace lss_core_sor_solver
{

template <>
void sor_solver_core<double>::operator()(thrust::device_vector<double> &solution,
                                         thrust::device_vector<double> &next_solution,
                                         thrust::device_vector<double> &errors, double omega)
{
    unsigned long long const threads_per_block = THREADS_PER_BLOCK;
    unsigned long long const blocks_per_grid = (nrows_ + threads_per_block - 1) / threads_per_block;

    double const *mat_data_ptr = thrust::raw_pointer_cast(&matrix_vals_[0]);
    std::size_t const *non_zero_col_idx_data_ptr = thrust::raw_pointer_cast(&non_zero_column_idx_[0]);
    double const *rhs_data_ptr = thrust::raw_pointer_cast(&rhs_vals_[0]);
    double const *diag_data_ptr = thrust::raw_pointer_cast(&diagonal_vals_[0]);
    std::size_t const *row_start_idx_data_ptr = thrust::raw_pointer_cast(&row_start_idx_[0]);
    std::size_t const *row_end_idx_data_ptr = thrust::raw_pointer_cast(&row_end_idx_[0]);
    double *sol_data_ptr = thrust::raw_pointer_cast(&solution[0]);
    double *next_sol_data_ptr = thrust::raw_pointer_cast(&next_solution[0]);
    double *errors_data_ptr = thrust::raw_pointer_cast(&errors[0]);

    sor_kernel<double><<<threads_per_block, blocks_per_grid>>>(
        mat_data_ptr, nrows_, non_zero_col_idx_data_ptr, rhs_data_ptr, diag_data_ptr, row_start_idx_data_ptr,
        row_end_idx_data_ptr, omega, sol_data_ptr, next_sol_data_ptr, errors_data_ptr);
}

template <>
void sor_solver_core<float>::operator()(thrust::device_vector<float> &solution,
                                        thrust::device_vector<float> &next_solution,
                                        thrust::device_vector<float> &errors, float omega)
{
    unsigned long long const threads_per_block = THREADS_PER_BLOCK;
    unsigned long long const blocks_per_grid = (nrows_ + threads_per_block - 1) / threads_per_block;

    float const *mat_data_ptr = thrust::raw_pointer_cast(&matrix_vals_[0]);
    std::size_t const *non_zero_col_idx_data_ptr = thrust::raw_pointer_cast(&non_zero_column_idx_[0]);
    float const *rhs_data_ptr = thrust::raw_pointer_cast(&rhs_vals_[0]);
    float const *diag_data_ptr = thrust::raw_pointer_cast(&diagonal_vals_[0]);
    std::size_t const *row_start_idx_data_ptr = thrust::raw_pointer_cast(&row_start_idx_[0]);
    std::size_t const *row_end_idx_data_ptr = thrust::raw_pointer_cast(&row_end_idx_[0]);
    float *sol_data_ptr = thrust::raw_pointer_cast(&solution[0]);
    float *next_sol_data_ptr = thrust::raw_pointer_cast(&next_solution[0]);
    float *errors_data_ptr = thrust::raw_pointer_cast(&errors[0]);

    sor_kernel<float><<<threads_per_block, blocks_per_grid>>>(
        mat_data_ptr, nrows_, non_zero_col_idx_data_ptr, rhs_data_ptr, diag_data_ptr, row_start_idx_data_ptr,
        row_end_idx_data_ptr, omega, sol_data_ptr, next_sol_data_ptr, errors_data_ptr);
}

} // namespace lss_core_sor_solver
