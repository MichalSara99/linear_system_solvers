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
#include "pde_solvers/one_dimensional/heat_type/explicit_coefficients/lss_heat_euler_svc_coefficients.hpp"

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
using lss_utility::coefficient_sevenlet_t;
using lss_utility::function_2d_sevenlet_t; // ?
using lss_utility::NaN;
using lss_utility::pair_t;
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
 * heat_euler_svc_cuda_kernel object
 */
template <typename fp_type> class heat_euler_svc_cuda_kernel
{
    typedef discretization<dimension_enum::One, fp_type, thrust::host_vector, std::allocator<fp_type>> d_1d;

  private:
    fp_type k_;
    thrust::device_vector<fp_type> d_a_;
    thrust::device_vector<fp_type> d_b_;
    thrust::device_vector<fp_type> d_d_;
    grid_config_1d_ptr<fp_type> grid_cfg_;

    void initialize(heat_euler_svc_coefficients_ptr<fp_type> const &coefficients)
    {
        k_ = coefficients->k_;
        const std::size_t space_size = coefficients->space_size_;
        auto const &a = coefficients->A_;
        auto const &b = coefficients->B_;
        auto const &d = coefficients->D_;
        thrust::host_vector<fp_type> h_a(space_size);
        thrust::host_vector<fp_type> h_b(space_size);
        thrust::host_vector<fp_type> h_d(space_size);
        // discretize on host
        d_1d::of_function(grid_cfg_, a, h_a);
        d_1d::of_function(grid_cfg_, b, h_b);
        d_1d::of_function(grid_cfg_, d, h_d);
        // copy to device
        d_a_.resize(space_size);
        d_b_.resize(space_size);
        d_d_.resize(space_size);
        thrust::copy(h_a.begin(), h_a.end(), d_a_.begin());
        thrust::copy(h_b.begin(), h_b.end(), d_b_.begin());
        thrust::copy(h_d.begin(), h_d.end(), d_d_.begin());
    }

  public:
    explicit heat_euler_svc_cuda_kernel(heat_euler_svc_coefficients_ptr<fp_type> const &coefficients,
                                        grid_config_1d_ptr<fp_type> const &grid_config)
        : grid_cfg_{grid_config}
    {
        initialize(coefficients);
    }
    ~heat_euler_svc_cuda_kernel()
    {
    }
    void launch(thrust::device_vector<fp_type> const &input, thrust::device_vector<fp_type> &solution);

    void launch(thrust::device_vector<fp_type> const &input, thrust::device_vector<fp_type> const &source,
                thrust::device_vector<fp_type> &solution);
};

template <typename fp_type> using heat_euler_svc_cuda_kernel_ptr = sptr_t<heat_euler_svc_cuda_kernel<fp_type>>;

template <typename fp_type>
using explicit_heat_svc_cuda_scheme_function =
    std::function<void(heat_euler_svc_coefficients_ptr<fp_type> const &,
                       heat_euler_svc_cuda_kernel_ptr<fp_type> const &, grid_config_1d_ptr<fp_type> const &,
                       thrust::host_vector<fp_type> const &, thrust::host_vector<fp_type> const &,
                       boundary_1d_pair<fp_type> const &, fp_type const &, thrust::host_vector<fp_type> &)>;

/**
 * explicit_svc_cuda_scheme object
 */
template <typename fp_type> class explicit_heat_svc_cuda_scheme
{
    typedef explicit_heat_svc_cuda_scheme_function<fp_type> scheme_function_t;

  public:
    static scheme_function_t const get(bool is_homogeneus)
    {
        const fp_type two = static_cast<fp_type>(2.0);
        auto scheme_fun_h =
            [=](heat_euler_svc_coefficients_ptr<fp_type> const &cfs,
                heat_euler_svc_cuda_kernel_ptr<fp_type> const &kernel, grid_config_1d_ptr<fp_type> const &grid_config,
                thrust::host_vector<fp_type> const &input, thrust::host_vector<fp_type> const &inhom_input,
                boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time,
                thrust::host_vector<fp_type> &solution) {
                auto const &first_bnd = boundary_pair.first;
                auto const &second_bnd = boundary_pair.second;
                auto const &a = cfs->A_;
                auto const &b = cfs->B_;
                auto const &d = cfs->D_;
                auto const h = cfs->h_;
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
                    solution[0] = beta * a(x) + b(x) * input[0] + (a(x) + d(x)) * input[1];
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_bnd))
                {
                    const fp_type beta = two * h * ptr->value(time);
                    const fp_type alpha = two * h * ptr->linear_value(time);
                    solution[0] = (b(x) + alpha * a(x)) * input[0] + (a(x) + d(x)) * input[1] + beta * a(x);
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
                    solution[N] = (a(x) + d(x)) * input[N - 1] + b(x) * input[N] - delta * d(x);
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
                {
                    const fp_type delta = two * h * ptr->value(time);
                    const fp_type gamma = two * h * ptr->linear_value(time);
                    solution[N] = (a(x) + d(x)) * input[N - 1] + (b(x) - gamma * d(x)) * input[N] - delta * d(x);
                }
                // light-weight object with cuda kernel computing the solution:
                thrust::device_vector<fp_type> d_input(input);
                thrust::device_vector<fp_type> d_solution(solution);
                kernel->launch(d_input, d_solution);
                thrust::copy(d_solution.begin(), d_solution.end(), solution.begin());
            };
        auto scheme_fun_nh =
            [=](heat_euler_svc_coefficients_ptr<fp_type> const &cfs,
                heat_euler_svc_cuda_kernel_ptr<fp_type> const &kernel, grid_config_1d_ptr<fp_type> const &grid_config,
                thrust::host_vector<fp_type> const &input, thrust::host_vector<fp_type> const &inhom_input,
                boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time,
                thrust::host_vector<fp_type> &solution) {
                auto const &first_bnd = boundary_pair.first;
                auto const &second_bnd = boundary_pair.second;
                auto const &a = cfs->A_;
                auto const &b = cfs->B_;
                auto const &d = cfs->D_;
                auto const k = cfs->k_;
                auto const h = cfs->h_;
                fp_type x{};

                // for lower boundaries first:
                x = grid_1d<fp_type>::value(grid_config, 0);
                if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(first_bnd))
                {
                    const fp_type beta = two * h * ptr->value(time);
                    solution[0] = beta * a(x) + b(x) * input[0] + (a(x) + d(x)) * input[1] + k * inhom_input[0];
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_bnd))
                {
                    const fp_type beta = two * h * ptr->value(time);
                    const fp_type alpha = two * h * ptr->linear_value(time);
                    solution[0] =
                        (b(x) + alpha * a(x)) * input[0] + (a(x) + d(x)) * input[1] + beta * a(x) + k * inhom_input[0];
                }
                // for upper boundaries second:
                const std::size_t N = solution.size() - 1;
                x = grid_1d<fp_type>::value(grid_config, N);
                if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(second_bnd))
                {
                    const fp_type delta = two * h * ptr->value(time);
                    solution[N] = (a(x) + d(x)) * input[N - 1] + b(x) * input[N] - delta * d(x) + k * inhom_input[N];
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
                {
                    const fp_type delta = two * h * ptr->value(time);
                    const fp_type gamma = two * h * ptr->linear_value(time);
                    solution[N] = (a(x) + d(x)) * input[N - 1] + (b(x) - gamma * d(x)) * input[N] - delta * d(x) +
                                  k * inhom_input[N];
                }
                // light-weight object with cuda kernel computing the solution:
                thrust::device_vector<fp_type> d_input(input);
                thrust::device_vector<fp_type> d_inhom_input(inhom_input);
                thrust::device_vector<fp_type> d_solution(solution);
                kernel->launch(d_input, d_inhom_input, d_solution);
                thrust::copy(d_solution.begin(), d_solution.end(), solution.begin());
            };
        if (is_homogeneus)
        {
            return scheme_fun_h;
        }
        else
        {
            return scheme_fun_nh;
        }
    }
};

/**
heat_euler_cuda_solver_method object
*/
template <typename fp_type> class heat_euler_cuda_solver_method
{
  private:
    // scheme coefficients:
    heat_euler_svc_coefficients_ptr<fp_type> coefficients_;
    grid_config_1d_ptr<fp_type> grid_cfg_;
    // cuda kernel:
    heat_euler_svc_cuda_kernel_ptr<fp_type> kernel_;

    explicit heat_euler_cuda_solver_method() = delete;

    void initialize()
    {
        kernel_ = std::make_shared<heat_euler_svc_cuda_kernel<fp_type>>(coefficients_, grid_cfg_);
    }

  public:
    explicit heat_euler_cuda_solver_method(heat_euler_svc_coefficients_ptr<fp_type> const &coefficients,
                                           grid_config_1d_ptr<fp_type> const &grid_config)
        : coefficients_{coefficients}, grid_cfg_{grid_config}
    {
        initialize();
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
    typedef explicit_heat_svc_cuda_scheme<fp_type> heat_scheme;

    // get the right-hand side of the scheme:
    auto scheme = heat_scheme::get(true);
    scheme(coefficients_, kernel_, grid_cfg_, prev_solution, thrust::host_vector<fp_type>(), boundary_pair, time,
           solution);
}

template <typename fp_type>
void heat_euler_cuda_solver_method<fp_type>::solve(thrust::host_vector<fp_type> &prev_solution,
                                                   boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time,
                                                   std::function<fp_type(fp_type, fp_type)> const &heat_source,
                                                   thrust::host_vector<fp_type> &solution)
{
    typedef explicit_heat_svc_cuda_scheme<fp_type> heat_scheme;
    typedef discretization<dimension_enum::One, fp_type, thrust::host_vector, std::allocator<fp_type>> d_1d;

    // get the right-hand side of the scheme:
    auto scheme = heat_scheme::get(false);
    thrust::host_vector<fp_type> source(prev_solution.size());
    d_1d::of_function(grid_cfg_, time, heat_source, source);
    scheme(coefficients_, kernel_, grid_cfg_, prev_solution, source, boundary_pair, time, solution);
}
} // namespace one_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_HEAT_EULER_CUDA_SOLVER_METHOD_HPP_
