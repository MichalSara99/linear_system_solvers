#if !defined(_LSS_HEAT_EULER_CUDA_SOLVER_METHOD_HPP_)
#define _LSS_HEAT_EULER_CUDA_SOLVER_METHOD_HPP_

#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "boundaries/lss_boundary.hpp"
#include "boundaries/lss_dirichlet_boundary.hpp"
#include "boundaries/lss_neumann_boundary.hpp"
#include "boundaries/lss_robin_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "containers/lss_container_2d.hpp"
#include "discretization/lss_discretization.hpp"
#include "discretization/lss_grid.hpp"
#include "discretization/lss_grid_config.hpp"
#include "pde_solvers/lss_heat_data_config.hpp"
#include "pde_solvers/lss_pde_discretization_config.hpp"
#include "pde_solvers/one_dimensional/heat_type/explicit_coefficients/lss_heat_euler_coefficients.hpp"

namespace lss_pde_solvers
{

namespace one_dimensional
{

using lss_boundary::boundary_1d_pair;
using lss_boundary::boundary_1d_ptr;
using lss_boundary::dirichlet_boundary_1d;
using lss_boundary::neumann_boundary_1d;
using lss_boundary::robin_boundary_1d;
using lss_containers::container_2d;
using lss_enumerations::by_enum;
using lss_utility::NaN;
using lss_utility::range;
using lss_utility::sptr_t;

/**
 * heat_core_kernel function
 *
 * \param a_coeff
 * \param b_coeff
 * \param d_coeff
 * \param input
 * \param solution
 * \param size
 * \return
 */
template <typename fp_type>
__global__ void heat_core_kernel(fp_type const *a_coeff, fp_type const *b_coeff, fp_type const *d_coeff,
                                 fp_type const *input, fp_type *solution, const std::size_t size)
{
    const std::size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid == 0)
        return;
    if (tid >= (size - 1))
        return;
    solution[tid] = (d_coeff[tid] * input[tid + 1]) + (b_coeff[tid] * input[tid]) + (a_coeff[tid] * input[tid - 1]);
}

/**
 * heat_core_kernel function
 *
 * \param a_coeff
 * \param b_coeff
 * \param d_coeff
 * \param time_step
 * \param input
 * \param source
 * \param solution
 * \param size
 * \return
 */
template <typename fp_type>
__global__ void heat_core_kernel(fp_type const *a_coeff, fp_type const *b_coeff, fp_type const *d_coeff,
                                 fp_type time_step, fp_type const *input, fp_type const *source, fp_type *solution,
                                 const std::size_t size)
{
    const std::size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid == 0)
        return;
    if (tid >= (size - 1))
        return;
    solution[tid] = (d_coeff[tid] * input[tid + 1]) + (b_coeff[tid] * input[tid]) + (a_coeff[tid] * input[tid - 1]) +
                    (time_step * source[tid]);
}

/**
 * heat_euler_cuda_kernel object
 */
template <typename fp_type> class heat_euler_cuda_kernel
{
    typedef discretization<dimension_enum::One, fp_type, thrust::host_vector, std::allocator<fp_type>> d_1d;

  private:
    fp_type k_;
    thrust::device_vector<fp_type> d_a_;
    thrust::device_vector<fp_type> d_b_;
    thrust::device_vector<fp_type> d_d_;
    thrust::host_vector<fp_type> h_a_;
    thrust::host_vector<fp_type> h_b_;
    thrust::host_vector<fp_type> h_d_;
    grid_config_1d_ptr<fp_type> grid_cfg_;
    // coefficients:
    std::function<fp_type(fp_type, fp_type)> a_;
    std::function<fp_type(fp_type, fp_type)> b_;
    std::function<fp_type(fp_type, fp_type)> d_;

    void initialize(heat_euler_coefficients_ptr<fp_type> const &coefficients)
    {
        const std::size_t space_size = coefficients->space_size_;
        k_ = coefficients->k_;
        a_ = coefficients->A_;
        b_ = coefficients->B_;
        d_ = coefficients->D_;
        h_a_.resize(space_size);
        h_b_.resize(space_size);
        h_d_.resize(space_size);
        d_a_.resize(space_size);
        d_b_.resize(space_size);
        d_d_.resize(space_size);
    }

    void discretize_coefficients(fp_type time)
    {
        // discretize on host
        d_1d::of_function(grid_cfg_, time, a_, h_a_);
        d_1d::of_function(grid_cfg_, time, b_, h_b_);
        d_1d::of_function(grid_cfg_, time, d_, h_d_);
        // copy to device
        thrust::copy(h_a_.begin(), h_a_.end(), d_a_.begin());
        thrust::copy(h_b_.begin(), h_b_.end(), d_b_.begin());
        thrust::copy(h_d_.begin(), h_d_.end(), d_d_.begin());
    }

  public:
    explicit heat_euler_cuda_kernel(heat_euler_coefficients_ptr<fp_type> const &coefficients,
                                    grid_config_1d_ptr<fp_type> const &grid_config)
        : grid_cfg_{grid_config}
    {
        initialize(coefficients);
    }
    ~heat_euler_cuda_kernel()
    {
    }
    void launch(fp_type time, thrust::device_vector<fp_type> const &input, thrust::device_vector<fp_type> &solution);

    void launch(fp_type time, thrust::device_vector<fp_type> const &input, thrust::device_vector<fp_type> const &source,
                thrust::device_vector<fp_type> &solution);
};

template <typename fp_type> using heat_euler_cuda_kernel_ptr = sptr_t<heat_euler_cuda_kernel<fp_type>>;

/**
 * explicit_cuda_scheme object
 */
template <typename fp_type> class explicit_cuda_scheme
{
  public:
    static void rhs(heat_euler_coefficients_ptr<fp_type> const &cfs, heat_euler_cuda_kernel_ptr<fp_type> const &kernel,
                    grid_config_1d_ptr<fp_type> const &grid_config, thrust::host_vector<fp_type> const &input,
                    boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time,
                    thrust::host_vector<fp_type> &solution)
    {
        const fp_type two = static_cast<fp_type>(2.0);
        auto const &first_bnd = boundary_pair.first;
        auto const &second_bnd = boundary_pair.second;
        auto const &a = cfs->A_;
        auto const &b = cfs->B_;
        auto const &d = cfs->D_;
        auto const h = grid_1d<fp_type>::step(grid_config);
        fp_type x{};
        // for lower boundaries first:
        x = grid_1d<fp_type>::value(grid_config, 0);
        if (auto const &ptr = std::dynamic_pointer_cast<dirichlet_boundary_1d<fp_type>>(first_bnd))
        {
            solution[0] = ptr->value(time);
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(first_bnd))
        {
            const fp_type beta = two * h * ptr->value(time);
            solution[0] = beta * a(time, x) + b(time, x) * input[0] + (a(time, x) + d(time, x)) * input[1];
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_bnd))
        {
            const fp_type beta = two * h * ptr->value(time);
            const fp_type alpha = two * h * ptr->linear_value(time);
            solution[0] =
                (b(time, x) + alpha * a(time, x)) * input[0] + (a(time, x) + d(time, x)) * input[1] + beta * a(time, x);
        }
        // for upper boundaries second:
        const std::size_t N = solution.size() - 1;
        x = grid_1d<fp_type>::value(grid_config, N);
        if (auto const &ptr = std::dynamic_pointer_cast<dirichlet_boundary_1d<fp_type>>(second_bnd))
        {
            solution[N] = ptr->value(time);
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(second_bnd))
        {
            const fp_type delta = two * h * ptr->value(time);
            solution[N] = (a(time, x) + d(time, x)) * input[N - 1] + b(time, x) * input[N] - delta * d(time, x);
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
        {
            const fp_type delta = two * h * ptr->value(time);
            const fp_type gamma = two * h * ptr->linear_value(time);
            solution[N] = (a(time, x) + d(time, x)) * input[N - 1] + (b(time, x) - gamma * d(time, x)) * input[N] -
                          delta * d(time, x);
        }
        // light-weight object with cuda kernel computing the solution:
        thrust::device_vector<fp_type> d_input(input);
        thrust::device_vector<fp_type> d_solution(solution);
        kernel->launch(time, d_input, d_solution);
        thrust::copy(d_solution.begin(), d_solution.end(), solution.begin());
    }

    static void rhs_source(heat_euler_coefficients_ptr<fp_type> const &cfs,
                           heat_euler_cuda_kernel_ptr<fp_type> const &kernel,
                           grid_config_1d_ptr<fp_type> const &grid_config, thrust::host_vector<fp_type> const &input,
                           thrust::host_vector<fp_type> const &inhom_input,
                           boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time,
                           thrust::host_vector<fp_type> &solution)
    {
        const fp_type two = static_cast<fp_type>(2.0);
        auto const &first_bnd = boundary_pair.first;
        auto const &second_bnd = boundary_pair.second;
        auto const &a = cfs->A_;
        auto const &b = cfs->B_;
        auto const &d = cfs->D_;
        auto const k = cfs->k_;
        auto const h = grid_1d<fp_type>::step(grid_config);
        fp_type x{};

        // for lower boundaries first:
        x = grid_1d<fp_type>::value(grid_config, 0);
        if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(first_bnd))
        {
            const fp_type beta = two * h * ptr->value(time);
            solution[0] =
                beta * a(time, x) + b(time, x) * input[0] + (a(time, x) + d(time, x)) * input[1] + k * inhom_input[0];
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_bnd))
        {
            const fp_type beta = two * h * ptr->value(time);
            const fp_type alpha = two * h * ptr->linear_value(time);
            solution[0] = (b(time, x) + alpha * a(time, x)) * input[0] + (a(time, x) + d(time, x)) * input[1] +
                          beta * a(time, x) + k * inhom_input[0];
        }
        // for upper boundaries second:
        const std::size_t N = solution.size() - 1;
        x = grid_1d<fp_type>::value(grid_config, N);
        if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(second_bnd))
        {
            const fp_type delta = two * h * ptr->value(time);
            solution[N] = (a(time, x) + d(time, x)) * input[N - 1] + b(time, x) * input[N] - delta * d(time, x) +
                          k * inhom_input[N];
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
        {
            const fp_type delta = two * h * ptr->value(time);
            const fp_type gamma = two * h * ptr->linear_value(time);
            solution[N] = (a(time, x) + d(time, x)) * input[N - 1] + (b(time, x) - gamma * d(time, x)) * input[N] -
                          delta * d(time, x) + k * inhom_input[N];
        }
        // light-weight object with cuda kernel computing the solution:
        thrust::device_vector<fp_type> d_input(input);
        thrust::device_vector<fp_type> d_inhom_input(inhom_input);
        thrust::device_vector<fp_type> d_solution(solution);
        kernel->launch(time, d_input, d_inhom_input, d_solution);
        thrust::copy(d_solution.begin(), d_solution.end(), solution.begin());
    }
};

/**
heat_euler_cuda_solver_method object
*/
template <typename fp_type> class heat_euler_cuda_solver_method
{
    typedef explicit_cuda_scheme<fp_type> heat_scheme;
    typedef discretization<dimension_enum::One, fp_type, thrust::host_vector, std::allocator<fp_type>> d_1d;

  private:
    // scheme coefficients:
    heat_euler_coefficients_ptr<fp_type> coefficients_;
    grid_config_1d_ptr<fp_type> grid_cfg_;
    // cuda kernel:
    heat_euler_cuda_kernel_ptr<fp_type> kernel_;
    // containers:
    thrust::host_vector<fp_type> source_;

    explicit heat_euler_cuda_solver_method() = delete;

    void initialize(bool is_heat_source_set)
    {
        kernel_ = std::make_shared<heat_euler_cuda_kernel<fp_type>>(coefficients_, grid_cfg_);
        if (is_heat_source_set)
        {
            source_.resize(coefficients_->space_size_);
        }
    }

  public:
    explicit heat_euler_cuda_solver_method(heat_euler_coefficients_ptr<fp_type> const &coefficients,
                                           grid_config_1d_ptr<fp_type> const &grid_config, bool is_heat_source_set)
        : coefficients_{coefficients}, grid_cfg_{grid_config}
    {
        initialize(is_heat_source_set);
    }

    ~heat_euler_cuda_solver_method()
    {
    }

    heat_euler_cuda_solver_method(heat_euler_cuda_solver_method const &) = delete;
    heat_euler_cuda_solver_method(heat_euler_cuda_solver_method &&) = delete;
    heat_euler_cuda_solver_method &operator=(heat_euler_cuda_solver_method const &) = delete;
    heat_euler_cuda_solver_method &operator=(heat_euler_cuda_solver_method &&) = delete;

    void solve(thrust::host_vector<fp_type> &prev_solution, boundary_1d_pair<fp_type> const &boundary_pair,
               fp_type const &time, thrust::host_vector<fp_type> &solution);

    void solve(thrust::host_vector<fp_type> &prev_solution, boundary_1d_pair<fp_type> const &boundary_pair,
               fp_type const &time, std::function<fp_type(fp_type, fp_type)> const &heat_source,
               thrust::host_vector<fp_type> &solution);
};

template <typename fp_type>
void heat_euler_cuda_solver_method<fp_type>::solve(thrust::host_vector<fp_type> &prev_solution,
                                                   boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time,
                                                   thrust::host_vector<fp_type> &solution)
{
    heat_scheme::rhs(coefficients_, kernel_, grid_cfg_, prev_solution, boundary_pair, time, solution);
}

template <typename fp_type>
void heat_euler_cuda_solver_method<fp_type>::solve(thrust::host_vector<fp_type> &prev_solution,
                                                   boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time,
                                                   std::function<fp_type(fp_type, fp_type)> const &heat_source,
                                                   thrust::host_vector<fp_type> &solution)
{
    d_1d::of_function(grid_cfg_, time, heat_source, source_);
    heat_scheme::rhs_source(coefficients_, kernel_, grid_cfg_, prev_solution, source_, boundary_pair, time, solution);
}
} // namespace one_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_HEAT_EULER_CUDA_SOLVER_METHOD_HPP_
