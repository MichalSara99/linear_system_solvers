#if !defined(_LSS_HESTON_EULER_CUDA_SOLVER_METHOD_HPP_)
#define _LSS_HESTON_EULER_CUDA_SOLVER_METHOD_HPP_

#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "containers/lss_container_2d.hpp"
#include "discretization/lss_discretization.hpp"
#include "discretization/lss_grid.hpp"
#include "discretization/lss_grid_config.hpp"
#include "pde_solvers/lss_pde_discretization_config.hpp"
#include "pde_solvers/two_dimensional/heat_type/explicit_coefficients/lss_heston_euler_coefficients.hpp"

namespace lss_pde_solvers
{
namespace two_dimensional
{

using lss_containers::container_2d;
using lss_enumerations::by_enum;
using lss_utility::NaN;
using lss_utility::range;
using lss_utility::sptr_t;

template <typename fp_type>
__global__ void heston_core_kernel(fp_type const *m_coeff, fp_type const *m_tilde_coeff, fp_type const *p_coeff,
                                   fp_type const *p_tilde_coeff, fp_type const *c_coeff, fp_type const *z_coeff,
                                   fp_type const *w_coeff, fp_type const *input, fp_type *solution,
                                   const std::size_t size_x, const std::size_t size_y)
{
    const std::size_t c_id = blockDim.x * blockIdx.x + threadIdx.x; // size_y
    const std::size_t r_id = blockDim.y * blockIdx.y + threadIdx.y; // size_x
    const std::size_t tid = c_id + r_id * size_y;
    if (c_id == 0)
        return;
    if (c_id >= (size_y - 1))
        return;
    if (r_id == 0)
        return;
    if (r_id >= (size_x - 1))
        return;
    // cross neighbours:
    const std::size_t up_tid = tid - size_y;
    const std::size_t down_tid = tid + size_y;
    const std::size_t right_tid = tid + 1;
    const std::size_t left_tid = tid - 1;
    // star neighbours:
    const std::size_t up_r_tid = tid - size_y + 1;
    const std::size_t up_l_tid = tid - size_y - 1;
    const std::size_t down_r_tid = tid + size_y + 1;
    const std::size_t down_l_tid = tid + size_y - 1;
    const fp_type one = static_cast<fp_type>(1.0);

    solution[tid] = (c_coeff[tid] * input[up_l_tid]) + (m_coeff[tid] * input[up_tid]) -
                    (c_coeff[tid] * input[up_r_tid]) + (m_tilde_coeff[tid] * input[left_tid]) +
                    ((one - z_coeff[tid] - w_coeff[tid]) * input[tid]) + (p_tilde_coeff[tid] * input[right_tid]) -
                    (c_coeff[tid] * input[down_l_tid]) + (p_coeff[tid] * input[down_tid]) +
                    (c_coeff[tid] * input[down_r_tid]);
}

template <typename fp_type>
__global__ void heston_core_kernel(fp_type const *m_coeff, fp_type const *m_tilde_coeff, fp_type const *p_coeff,
                                   fp_type const *p_tilde_coeff, fp_type const *c_coeff, fp_type const *z_coeff,
                                   fp_type const *w_coeff, fp_type time_step, fp_type const *input,
                                   fp_type const *source, fp_type *solution, const std::size_t size_x,
                                   const std::size_t size_y)
{
    const std::size_t c_id = blockDim.x * blockIdx.x + threadIdx.x; // size_y
    const std::size_t r_id = blockDim.y * blockIdx.y + threadIdx.y; // size_x
    const std::size_t tid = c_id + r_id * size_y;
    if (c_id == 0)
        return;
    if (c_id >= (size_y - 1))
        return;
    if (r_id == 0)
        return;
    if (r_id >= (size_x - 1))
        return;
    // cross neighbours:
    const std::size_t up_tid = tid - size_y;
    const std::size_t down_tid = tid + size_y;
    const std::size_t right_tid = tid + 1;
    const std::size_t left_tid = tid - 1;
    // star neighbours:
    const std::size_t up_r_tid = tid - size_y + 1;
    const std::size_t up_l_tid = tid - size_y - 1;
    const std::size_t down_r_tid = tid + size_y + 1;
    const std::size_t down_l_tid = tid + size_y - 1;
    const fp_type one = static_cast<fp_type>(1.0);

    solution[tid] = (c_coeff[tid] * input[up_l_tid]) + (m_coeff[tid] * input[up_tid]) -
                    (c_coeff[tid] * input[up_r_tid]) + (m_tilde_coeff[tid] * input[left_tid]) +
                    ((one - z_coeff[tid] - w_coeff[tid]) * input[tid]) + (p_tilde_coeff[tid] * input[right_tid]) -
                    (c_coeff[tid] * input[down_l_tid]) + (p_coeff[tid] * input[down_tid]) +
                    (c_coeff[tid] * input[down_r_tid]) + (time_step * source[tid]);
}

/**
 * heston_euler_cuda_kernel object
 */
template <typename fp_type> class heston_euler_cuda_kernel
{
    typedef discretization<dimension_enum::Two, fp_type, thrust::host_vector, std::allocator<fp_type>> d_2d;

  private:
    fp_type k_;
    std::size_t size_x_, size_y_;
    // device containers:
    thrust::device_vector<fp_type> d_m_;
    thrust::device_vector<fp_type> d_m_tilde_;
    thrust::device_vector<fp_type> d_p_;
    thrust::device_vector<fp_type> d_p_tilde_;
    thrust::device_vector<fp_type> d_c_;
    thrust::device_vector<fp_type> d_z_;
    thrust::device_vector<fp_type> d_w_;
    // host containers:
    thrust::host_vector<fp_type> h_m_;
    thrust::host_vector<fp_type> h_m_tilde_;
    thrust::host_vector<fp_type> h_p_;
    thrust::host_vector<fp_type> h_p_tilde_;
    thrust::host_vector<fp_type> h_c_;
    thrust::host_vector<fp_type> h_z_;
    thrust::host_vector<fp_type> h_w_;
    grid_config_2d_ptr<fp_type> grid_cfg_;
    // coefficients:
    std::function<fp_type(fp_type, fp_type, fp_type)> m_;
    std::function<fp_type(fp_type, fp_type, fp_type)> m_tilde_;
    std::function<fp_type(fp_type, fp_type, fp_type)> p_;
    std::function<fp_type(fp_type, fp_type, fp_type)> p_tilde_;
    std::function<fp_type(fp_type, fp_type, fp_type)> c_;
    std::function<fp_type(fp_type, fp_type, fp_type)> z_;
    std::function<fp_type(fp_type, fp_type, fp_type)> w_;

    void initialize(heston_euler_coefficients_ptr<fp_type> const &coefficients)
    {
        size_x_ = coefficients->space_size_x_;
        size_y_ = coefficients->space_size_y_;
        const std::size_t total_size = size_x_ * size_y_;
        k_ = coefficients->k_;
        m_ = coefficients->M_;
        m_tilde_ = coefficients->M_tilde_;
        p_ = coefficients->P_;
        p_tilde_ = coefficients->P_tilde_;
        c_ = coefficients->C_;
        z_ = coefficients->Z_;
        w_ = coefficients->W_;
        // on host:
        h_m_.resize(total_size);
        h_m_tilde_.resize(total_size);
        h_p_.resize(total_size);
        h_p_tilde_.resize(total_size);
        h_c_.resize(total_size);
        h_z_.resize(total_size);
        h_w_.resize(total_size);
        // on device:
        d_m_.resize(total_size);
        d_m_tilde_.resize(total_size);
        d_p_.resize(total_size);
        d_p_tilde_.resize(total_size);
        d_c_.resize(total_size);
        d_z_.resize(total_size);
        d_w_.resize(total_size);
    }

    void discretize_coefficients(fp_type time)
    {
        // discretize on host
        d_2d::of_function(grid_cfg_, time, m_, size_x_, size_y_, h_m_);
        d_2d::of_function(grid_cfg_, time, m_tilde_, size_x_, size_y_, h_m_tilde_);
        d_2d::of_function(grid_cfg_, time, p_, size_x_, size_y_, h_p_);
        d_2d::of_function(grid_cfg_, time, p_tilde_, size_x_, size_y_, h_p_tilde_);
        d_2d::of_function(grid_cfg_, time, c_, size_x_, size_y_, h_c_);
        d_2d::of_function(grid_cfg_, time, z_, size_x_, size_y_, h_z_);
        d_2d::of_function(grid_cfg_, time, w_, size_x_, size_y_, h_w_);
        // copy to device
        thrust::copy(h_m_.begin(), h_m_.end(), d_m_.begin());
        thrust::copy(h_m_tilde_.begin(), h_m_tilde_.end(), d_m_tilde_.begin());
        thrust::copy(h_p_.begin(), h_p_.end(), d_p_.begin());
        thrust::copy(h_p_tilde_.begin(), h_p_tilde_.end(), d_p_tilde_.begin());
        thrust::copy(h_c_.begin(), h_c_.end(), d_c_.begin());
        thrust::copy(h_z_.begin(), h_z_.end(), d_z_.begin());
        thrust::copy(h_w_.begin(), h_w_.end(), d_w_.begin());
    }

  public:
    explicit heston_euler_cuda_kernel(heston_euler_coefficients_ptr<fp_type> const &coefficients,
                                      grid_config_2d_ptr<fp_type> const &grid_config)
        : grid_cfg_{grid_config}
    {
        initialize(coefficients);
    }
    ~heston_euler_cuda_kernel()
    {
    }
    void launch(fp_type time, thrust::device_vector<fp_type> const &input, thrust::device_vector<fp_type> &solution);

    void launch(fp_type time, thrust::device_vector<fp_type> const &input, thrust::device_vector<fp_type> const &source,
                thrust::device_vector<fp_type> &solution);
};

template <typename fp_type> using heston_euler_cuda_kernel_ptr = sptr_t<heston_euler_cuda_kernel<fp_type>>;

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class explicit_heston_cuda_scheme
{
    typedef container_2d<by_enum::Row, fp_type, container, allocator> rcontainer_2d_t;

  public:
    static void rhs(heston_euler_coefficients_ptr<fp_type> const &cfs,
                    heston_euler_cuda_kernel_ptr<fp_type> const &kernel, grid_config_2d_ptr<fp_type> const &grid_config,
                    rcontainer_2d_t const &input, fp_type const &time, rcontainer_2d_t &solution)
    {
        // light-weight object with cuda kernel computing the solution:
        thrust::device_vector<fp_type> d_input(input.data());
        thrust::device_vector<fp_type> d_solution(solution.data());
        kernel->launch(time, d_input, d_solution);
        container<fp_type, allocator> tmp_solution(solution.data().size());
        thrust::copy(d_solution.begin(), d_solution.end(), tmp_solution.begin());
        solution.from_data(tmp_solution);
    }

    static void rhs_source(heston_euler_coefficients_ptr<fp_type> const &cfs,
                           heston_euler_cuda_kernel_ptr<fp_type> const &kernel,
                           grid_config_2d_ptr<fp_type> const &grid_config, rcontainer_2d_t const &input,
                           fp_type const &time, rcontainer_2d_t const &inhom_input, rcontainer_2d_t &solution)
    {
        // light-weight object with cuda kernel computing the solution:
        thrust::device_vector<fp_type> d_input(input.data());
        thrust::device_vector<fp_type> d_inhom_input(inhom_input.data());
        thrust::device_vector<fp_type> d_solution(solution.data());
        kernel->launch(time, d_input, d_inhom_input, d_solution);
        container<fp_type, allocator> tmp_solution(solution.data().size());
        thrust::copy(d_solution.begin(), d_solution.end(), tmp_solution.begin());
        solution.from_data(tmp_solution);
    }
};

/**
    heston_euler_cuda_solver_method object
*/
template <typename fp_type, template <typename, typename> typename container, typename allocator>
class heston_euler_cuda_solver_method
{
    typedef explicit_heston_cuda_scheme<fp_type, container, allocator> heston_scheme;
    typedef discretization<dimension_enum::Two, fp_type, container, allocator> d_2d;
    typedef container_2d<by_enum::Row, fp_type, container, allocator> rcontainer_2d_t;
    typedef sptr_t<container_2d<by_enum::Row, fp_type, container, allocator>> rcontainer_2d_ptr;

  private:
    // scheme coefficients:
    heston_euler_coefficients_ptr<fp_type> coefficients_;
    grid_config_2d_ptr<fp_type> grid_cfg_;
    // cuda kernel:
    heston_euler_cuda_kernel_ptr<fp_type> kernel_;
    rcontainer_2d_ptr source_;

    explicit heston_euler_cuda_solver_method() = delete;

    void initialize(bool is_heat_source_set)
    {
        kernel_ = std::make_shared<heston_euler_cuda_kernel<fp_type>>(coefficients_, grid_cfg_);
        if (is_heat_source_set)
        {
            source_ = std::make_shared<container_2d<by_enum::Row, fp_type, container, allocator>>(
                coefficients_->space_size_x_, coefficients_->space_size_y_);
        }
    }

  public:
    explicit heston_euler_cuda_solver_method(heston_euler_coefficients_ptr<fp_type> const &coefficients,
                                             grid_config_2d_ptr<fp_type> const &grid_config, bool is_heat_source_set)
        : coefficients_{coefficients}, grid_cfg_{grid_config}
    {
        initialize(is_heat_source_set);
    }

    ~heston_euler_cuda_solver_method()
    {
    }

    heston_euler_cuda_solver_method(heston_euler_cuda_solver_method const &) = delete;
    heston_euler_cuda_solver_method(heston_euler_cuda_solver_method &&) = delete;
    heston_euler_cuda_solver_method &operator=(heston_euler_cuda_solver_method const &) = delete;
    heston_euler_cuda_solver_method &operator=(heston_euler_cuda_solver_method &&) = delete;

    void solve(container_2d<by_enum::Row, fp_type, container, allocator> &prev_solution, fp_type const &time,
               container_2d<by_enum::Row, fp_type, container, allocator> &solution);

    void solve(container_2d<by_enum::Row, fp_type, container, allocator> &prev_solution, fp_type const &time,
               std::function<fp_type(fp_type, fp_type, fp_type)> const &heat_source,
               container_2d<by_enum::Row, fp_type, container, allocator> &solution);
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void heston_euler_cuda_solver_method<fp_type, container, allocator>::solve(
    container_2d<by_enum::Row, fp_type, container, allocator> &prev_solution, fp_type const &time,
    container_2d<by_enum::Row, fp_type, container, allocator> &solution)
{
    heston_scheme::rhs(coefficients_, kernel_, grid_cfg_, prev_solution, time, solution);
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void heston_euler_cuda_solver_method<fp_type, container, allocator>::solve(
    container_2d<by_enum::Row, fp_type, container, allocator> &prev_solution, fp_type const &time,
    std::function<fp_type(fp_type, fp_type, fp_type)> const &heat_source,
    container_2d<by_enum::Row, fp_type, container, allocator> &solution)
{
    d_2d::of_function(grid_cfg_, time, heat_source, *source_);
    heston_scheme::rhs_source(coefficients_, kernel_, grid_cfg_, prev_solution, time, *source_, solution);
}

} // namespace two_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_HESTON_EULER_CUDA_SOLVER_METHOD_HPP_
