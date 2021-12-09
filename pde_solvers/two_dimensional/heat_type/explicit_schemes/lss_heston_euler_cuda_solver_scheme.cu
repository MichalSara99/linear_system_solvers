#include "pde_solvers/two_dimensional/heat_type/solver_method/lss_heston_euler_cuda_solver_method.hpp"

#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16

namespace lss_pde_solvers {

namespace two_dimensional {

template <>
void heston_euler_cuda_kernel<float>::launch(float time, thrust::device_vector<float> const &input,
                                            thrust::device_vector<float> &solution) {
  discretize_coefficients(time);
  const unsigned int threads_per_block_x = THREADS_PER_BLOCK_X;
  const unsigned int threads_per_block_y = THREADS_PER_BLOCK_Y;
  const unsigned int blocks_per_grid_x = static_cast<unsigned int>(size_x_ + threads_per_block_x - 1) / 
      threads_per_block_x;
  const unsigned int blocks_per_grid_y = static_cast<unsigned int>(size_y_ + threads_per_block_y - 1) / 
      threads_per_block_y;
  const dim3 blocks_per_grid(blocks_per_grid_y, blocks_per_grid_x);
  const dim3 threads_per_block(threads_per_block_y, threads_per_block_x);
  float *raw_m = thrust::raw_pointer_cast(d_m_.data());
  float *raw_m_tilde = thrust::raw_pointer_cast(d_m_tilde_.data());
  float *raw_p = thrust::raw_pointer_cast(d_p_.data());
  float *raw_p_tilde = thrust::raw_pointer_cast(d_p_tilde_.data());
  float *raw_c = thrust::raw_pointer_cast(d_c_.data());
  float *raw_z = thrust::raw_pointer_cast(d_z_.data());
  float *raw_w = thrust::raw_pointer_cast(d_w_.data());
  const float *raw_input = thrust::raw_pointer_cast(input.data());
  float *raw_solution = thrust::raw_pointer_cast(solution.data());
  heston_core_kernel<float><<<blocks_per_grid,threads_per_block>>>(
      raw_m, raw_m_tilde, raw_p, raw_p_tilde, raw_c, raw_z, raw_w, raw_input,
      raw_solution, size_x_, size_y_);
}

template <>
void heston_euler_cuda_kernel<double>::launch(double time, thrust::device_vector<double> const &input,
                                            thrust::device_vector<double> &solution) {
  discretize_coefficients(time);
  const unsigned int threads_per_block_x = THREADS_PER_BLOCK_X;
  const unsigned int threads_per_block_y = THREADS_PER_BLOCK_Y;
  const unsigned int blocks_per_grid_x =
      static_cast<unsigned int>(size_x_ + threads_per_block_x - 1) /
      threads_per_block_x;
  const unsigned int blocks_per_grid_y =
      static_cast<unsigned int>(size_y_ + threads_per_block_y - 1) /
      threads_per_block_y;
  const dim3 blocks_per_grid(blocks_per_grid_y, blocks_per_grid_x);
  const dim3 threads_per_block(threads_per_block_y, threads_per_block_x);
  double *raw_m = thrust::raw_pointer_cast(d_m_.data());
  double *raw_m_tilde = thrust::raw_pointer_cast(d_m_tilde_.data());
  double *raw_p = thrust::raw_pointer_cast(d_p_.data());
  double *raw_p_tilde = thrust::raw_pointer_cast(d_p_tilde_.data());
  double *raw_c = thrust::raw_pointer_cast(d_c_.data());
  double *raw_z = thrust::raw_pointer_cast(d_z_.data());
  double *raw_w = thrust::raw_pointer_cast(d_w_.data());
  const double *raw_input = thrust::raw_pointer_cast(input.data());
  double *raw_solution = thrust::raw_pointer_cast(solution.data());
  heston_core_kernel<double><<<blocks_per_grid, threads_per_block>>>(
      raw_m, raw_m_tilde, raw_p, raw_p_tilde, raw_c, raw_z, raw_w, raw_input,
      raw_solution, size_x_, size_y_);
}

template <>
void heston_euler_cuda_kernel<float>::launch(float time, thrust::device_vector<float> const &input,
                                            thrust::device_vector<float> const &source,thrust::device_vector<float> &solution) {
  discretize_coefficients(time);
  const unsigned int threads_per_block_x = THREADS_PER_BLOCK_X;
  const unsigned int threads_per_block_y = THREADS_PER_BLOCK_Y;
  const unsigned int blocks_per_grid_x =
      static_cast<unsigned int>(size_x_ + threads_per_block_x - 1) /
      threads_per_block_x;
  const unsigned int blocks_per_grid_y =
      static_cast<unsigned int>(size_y_ + threads_per_block_y - 1) /
      threads_per_block_y;
  const dim3 blocks_per_grid(blocks_per_grid_y, blocks_per_grid_x);
  const dim3 threads_per_block(threads_per_block_y, threads_per_block_x);
  float *raw_m = thrust::raw_pointer_cast(d_m_.data());
  float *raw_m_tilde = thrust::raw_pointer_cast(d_m_tilde_.data());
  float *raw_p = thrust::raw_pointer_cast(d_p_.data());
  float *raw_p_tilde = thrust::raw_pointer_cast(d_p_tilde_.data());
  float *raw_c = thrust::raw_pointer_cast(d_c_.data());
  float *raw_z = thrust::raw_pointer_cast(d_z_.data());
  float *raw_w = thrust::raw_pointer_cast(d_w_.data());
  const float *raw_source = thrust::raw_pointer_cast(source.data());
  const float *raw_input = thrust::raw_pointer_cast(input.data());
  float *raw_solution = thrust::raw_pointer_cast(solution.data());
  heston_core_kernel<float><<<blocks_per_grid, threads_per_block>>>(
      raw_m, raw_m_tilde, raw_p, raw_p_tilde, raw_c, raw_z, raw_w, k_, raw_input,
      raw_source, raw_solution, size_x_, size_y_);
}

template <>
void heston_euler_cuda_kernel<double>::launch(double time, thrust::device_vector<double> const &input,
                                            thrust::device_vector<double> const &source,thrust::device_vector<double> &solution) {
  discretize_coefficients(time);
  const unsigned int threads_per_block_x = THREADS_PER_BLOCK_X;
  const unsigned int threads_per_block_y = THREADS_PER_BLOCK_Y;
  const unsigned int blocks_per_grid_x =
      static_cast<unsigned int>(size_x_ + threads_per_block_x - 1) /
      threads_per_block_x;
  const unsigned int blocks_per_grid_y =
      static_cast<unsigned int>(size_y_ + threads_per_block_y - 1) /
      threads_per_block_y;
  const dim3 blocks_per_grid(blocks_per_grid_y, blocks_per_grid_x);
  const dim3 threads_per_block(threads_per_block_y, threads_per_block_x);
  double *raw_m = thrust::raw_pointer_cast(d_m_.data());
  double *raw_m_tilde = thrust::raw_pointer_cast(d_m_tilde_.data());
  double *raw_p = thrust::raw_pointer_cast(d_p_.data());
  double *raw_p_tilde = thrust::raw_pointer_cast(d_p_tilde_.data());
  double *raw_c = thrust::raw_pointer_cast(d_c_.data());
  double *raw_z = thrust::raw_pointer_cast(d_z_.data());
  double *raw_w = thrust::raw_pointer_cast(d_w_.data());
  const double *raw_source = thrust::raw_pointer_cast(source.data());
  const double *raw_input = thrust::raw_pointer_cast(input.data());
  double *raw_solution = thrust::raw_pointer_cast(solution.data());
  heston_core_kernel<double><<<blocks_per_grid, threads_per_block>>>(
      raw_m, raw_m_tilde, raw_p, raw_p_tilde, raw_c, raw_z, raw_w, k_, raw_input,
      raw_source, raw_solution, size_x_, size_y_);
}

}  // namespace two_dimensional

}  // namespace lss_pde_solvers
