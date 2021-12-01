#if !defined(_LSS_WAVE_EULER_CUDA_SOLVER_METHOD_HPP_)
#define _LSS_WAVE_EULER_CUDA_SOLVER_METHOD_HPP_

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
#include "lss_wave_euler_solver_method.hpp"
#include "pde_solvers/lss_pde_discretization_config.hpp"
#include "pde_solvers/lss_wave_data_config.hpp"
#include "pde_solvers/one_dimensional/wave_type/explicit_coefficients/lss_wave_explicit_coefficients.hpp"

namespace lss_pde_solvers
{

namespace one_dimensional
{

using lss_boundary::boundary_1d_pair;
using lss_boundary::boundary_1d_ptr;
using lss_boundary::dirichlet_boundary_1d;
using lss_boundary::neumann_boundary_1d;
using lss_containers::container_2d;
using lss_enumerations::by_enum;
using lss_utility::NaN;
using lss_utility::range;
using lss_utility::sptr_t;

/**
 * core_kernel function
 *
 * \param a_coeff
 * \param b_coeff
 * \param c_coeff
 * \param d_coeff
 * \param input_0
 * \param input_1
 * \param solution
 * \param size
 * \return
 */
template <typename fp_type>
__global__ void wave_core_kernel(fp_type const *a_coeff, fp_type const *b_coeff, fp_type const *c_coeff,
                                 fp_type const *d_coeff, fp_type const *input_0, fp_type const *input_1,
                                 fp_type *solution, const std::size_t size)
{
    const std::size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid == 0)
        return;
    if (tid >= (size - 1))
        return;

    solution[tid] = (a_coeff[tid] * input_1[tid - 1]) + (c_coeff[tid] * input_1[tid]) +
                    (b_coeff[tid] * input_1[tid + 1]) - (d_coeff[tid] * input_0[tid]);
}

/**
 * core_kernel function
 *
 * \param a_coeff
 * \param b_coeff
 * \param c_coeff
 * \param d_coeff
 * \param input_0
 * \param input_1
 * \param source
 * \param solution
 * \param size
 * \return
 */
template <typename fp_type>
__global__ void wave_core_kernel(fp_type const *a_coeff, fp_type const *b_coeff, fp_type const *c_coeff,
                                 fp_type const *d_coeff, fp_type const *input_0, fp_type const *input_1,
                                 fp_type const *source, fp_type *solution, const std::size_t size)
{
    const std::size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid == 0)
        return;
    if (tid >= (size - 1))
        return;

    solution[tid] = (a_coeff[tid] * input_1[tid - 1]) + (c_coeff[tid] * input_1[tid]) +
                    (b_coeff[tid] * input_1[tid + 1]) - (d_coeff[tid] * input_0[tid]) + source[tid];
}

/**
 * wave_euler_cuda_kernel object
 */
template <typename fp_type> class wave_euler_cuda_kernel
{
    typedef discretization<dimension_enum::One, fp_type, thrust::host_vector, std::allocator<fp_type>> d_1d;

  private:
    fp_type k_;
    thrust::device_vector<fp_type> d_a_;
    thrust::device_vector<fp_type> d_b_;
    thrust::device_vector<fp_type> d_c_;
    thrust::device_vector<fp_type> d_d_;
    thrust::host_vector<fp_type> h_a_;
    thrust::host_vector<fp_type> h_b_;
    thrust::host_vector<fp_type> h_c_;
    thrust::host_vector<fp_type> h_d_;
    grid_config_1d_ptr<fp_type> grid_cfg_;
    // coefficients:
    std::function<fp_type(fp_type, fp_type)> a_;
    std::function<fp_type(fp_type, fp_type)> b_;
    std::function<fp_type(fp_type, fp_type)> c_;
    std::function<fp_type(fp_type, fp_type)> d_;

    void initialize(wave_explicit_coefficients_ptr<fp_type> const &coefficients)
    {
        const std::size_t space_size = coefficients->space_size_;
        k_ = coefficients->k_;
        a_ = coefficients->A_;
        b_ = coefficients->B_;
        c_ = coefficients->C_;
        d_ = coefficients->D_;
        h_a_.resize(space_size);
        h_b_.resize(space_size);
        h_c_.resize(space_size);
        h_d_.resize(space_size);
        d_a_.resize(space_size);
        d_b_.resize(space_size);
        d_c_.resize(space_size);
        d_d_.resize(space_size);
    }

    void discretize_coefficients(fp_type time)
    {
        // discretize on host
        d_1d::of_function(grid_cfg_, time, a_, h_a_);
        d_1d::of_function(grid_cfg_, time, b_, h_b_);
        d_1d::of_function(grid_cfg_, time, c_, h_c_);
        d_1d::of_function(grid_cfg_, time, d_, h_d_);
        thrust::copy(h_a_.begin(), h_a_.end(), d_a_.begin());
        thrust::copy(h_b_.begin(), h_b_.end(), d_b_.begin());
        thrust::copy(h_c_.begin(), h_c_.end(), d_c_.begin());
        thrust::copy(h_d_.begin(), h_d_.end(), d_d_.begin());
    }

  public:
    explicit wave_euler_cuda_kernel(wave_explicit_coefficients_ptr<fp_type> const &coefficients,
                                    grid_config_1d_ptr<fp_type> const &grid_config)
        : grid_cfg_{grid_config}
    {
        initialize(coefficients);
    }
    ~wave_euler_cuda_kernel()
    {
    }
    void launch(fp_type time, thrust::device_vector<fp_type> const &input_0,
                thrust::device_vector<fp_type> const &input_1, thrust::device_vector<fp_type> &solution);

    void launch(fp_type time, thrust::device_vector<fp_type> const &input_0,
                thrust::device_vector<fp_type> const &input_1, thrust::device_vector<fp_type> const &source,
                thrust::device_vector<fp_type> &solution);
};

template <typename fp_type> using wave_euler_cuda_kernel_ptr = sptr_t<wave_euler_cuda_kernel<fp_type>>;

/**
 * explicit_wave_cuda_scheme object
 */
template <typename fp_type> class explicit_wave_cuda_scheme
{

  public:
    static void rhs(wave_explicit_coefficients_ptr<fp_type> const &cfs,
                    wave_euler_cuda_kernel_ptr<fp_type> const &kernel, grid_config_1d_ptr<fp_type> const &grid_config,
                    thrust::host_vector<fp_type> const &input_0, thrust::host_vector<fp_type> const &input_1,
                    boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time,
                    thrust::host_vector<fp_type> &solution)
    {
        const fp_type two = static_cast<fp_type>(2.0);
        auto const &first_bnd = boundary_pair.first;
        auto const &second_bnd = boundary_pair.second;
        auto const &a = cfs->A_;
        auto const &b = cfs->B_;
        auto const &c = cfs->C_;
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
            solution[0] = beta * a(time, x) + c(time, x) * input_1[0] + (a(time, x) + b(time, x)) * input_1[1] -
                          d(time, x) * input_0[0];
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_bnd))
        {
            const fp_type beta = two * h * ptr->value(time);
            const fp_type alpha = two * h * ptr->linear_value(time);
            solution[0] = beta * a(time, x) + (c(time, x) + alpha * a(time, x)) * input_1[0] +
                          (a(time, x) + b(time, x)) * input_1[1] - d(time, x) * input_0[0];
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
            solution[N] = (a(time, x) + b(time, x)) * input_1[N - 1] + c(time, x) * input_1[N] - delta * b(time, x) -
                          d(time, x) * input_0[N];
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
        {
            const fp_type delta = two * h * ptr->value(time);
            const fp_type gamma = two * h * ptr->linear_value(time);
            solution[N] = (a(time, x) + b(time, x)) * input_1[N - 1] + (c(time, x) - gamma * b(time, x)) * input_1[N] -
                          delta * b(time, x) - d(time, x) * input_0[N];
        }

        thrust::device_vector<fp_type> d_input_0(input_0);
        thrust::device_vector<fp_type> d_input_1(input_1);
        thrust::device_vector<fp_type> d_solution(solution);
        kernel->launch(time, d_input_0, d_input_1, d_solution);
        thrust::copy(d_solution.begin(), d_solution.end(), solution.begin());
    }

    static void rhs_source(wave_explicit_coefficients_ptr<fp_type> const &cfs,
                           wave_euler_cuda_kernel_ptr<fp_type> const &kernel,
                           grid_config_1d_ptr<fp_type> const &grid_config, thrust::host_vector<fp_type> const &input_0,
                           thrust::host_vector<fp_type> const &input_1, thrust::host_vector<fp_type> const &inhom_input,
                           boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time,
                           thrust::host_vector<fp_type> &solution)
    {
        const fp_type two = static_cast<fp_type>(2.0);
        auto const &first_bnd = boundary_pair.first;
        auto const &second_bnd = boundary_pair.second;
        auto const &a = cfs->A_;
        auto const &b = cfs->B_;
        auto const &c = cfs->C_;
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
            solution[0] = beta * a(time, x) + c(time, x) * input_1[0] + (a(time, x) + b(time, x)) * input_1[1] -
                          d(time, x) * input_0[0] + inhom_input[0];
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_bnd))
        {
            const fp_type beta = two * h * ptr->value(time);
            const fp_type alpha = two * h * ptr->linear_value(time);
            solution[0] = beta * a(time, x) + (c(time, x) + alpha * a(time, x)) * input_1[0] +
                          (a(time, x) + b(time, x)) * input_1[1] - d(time, x) * input_0[0] + inhom_input[0];
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
            solution[N] = (a(time, x) + b(time, x)) * input_1[N - 1] + c(time, x) * input_1[N] - delta * b(time, x) -
                          d(time, x) * input_0[N] + inhom_input[N];
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
        {
            const fp_type delta = two * h * ptr->value(time);
            const fp_type gamma = two * h * ptr->linear_value(time);
            solution[N] = (a(time, x) + b(time, x)) * input_1[N - 1] + (c(time, x) - gamma * b(time, x)) * input_1[N] -
                          delta * b(time, x) - d(time, x) * input_0[N] + inhom_input[N];
        }

        thrust::device_vector<fp_type> d_input_0(input_0);
        thrust::device_vector<fp_type> d_input_1(input_1);
        thrust::device_vector<fp_type> d_source(inhom_input);
        thrust::device_vector<fp_type> d_solution(solution);
        kernel->launch(time, d_input_0, d_input_1, d_source, d_solution);
        thrust::copy(d_solution.begin(), d_solution.end(), solution.begin());
    }
};

/**
wave_euler_cuda_solver_method object
*/
template <typename fp_type> class wave_euler_cuda_solver_method
{
  private:
    // scheme coefficients:
    wave_explicit_coefficients_ptr<fp_type> coefficients_;
    grid_config_1d_ptr<fp_type> grid_cfg_;
    // cuda kernel:
    wave_euler_cuda_kernel_ptr<fp_type> kernel_;
    thrust::host_vector<fp_type> source_;

    explicit wave_euler_cuda_solver_method() = delete;

    void initialize(bool is_wave_source_set)
    {
        kernel_ = std::make_shared<wave_euler_cuda_kernel<fp_type>>(coefficients_, grid_cfg_);
        if (is_wave_source_set)
        {
            source_.resize(coefficients_->space_size_);
        }
    }

  public:
    explicit wave_euler_cuda_solver_method(wave_explicit_coefficients_ptr<fp_type> const &coefficients,
                                           grid_config_1d_ptr<fp_type> const &grid_config, bool is_wave_source_set)
        : coefficients_{coefficients}, grid_cfg_{grid_config}
    {
        initialize(is_wave_source_set);
    }

    ~wave_euler_cuda_solver_method()
    {
    }

    wave_euler_cuda_solver_method(wave_euler_cuda_solver_method const &) = delete;
    wave_euler_cuda_solver_method(wave_euler_cuda_solver_method &&) = delete;
    wave_euler_cuda_solver_method &operator=(wave_euler_cuda_solver_method const &) = delete;
    wave_euler_cuda_solver_method &operator=(wave_euler_cuda_solver_method &&) = delete;

    void solve_initial(thrust::host_vector<fp_type> &prev_solution_0, thrust::host_vector<fp_type> &prev_solution_1,
                       boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, fp_type const &next_time,
                       thrust::host_vector<fp_type> &solution);

    void solve_initial(thrust::host_vector<fp_type> &prev_solution_0, thrust::host_vector<fp_type> &prev_solution_1,
                       boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, fp_type const &next_time,
                       std::function<fp_type(fp_type, fp_type)> const &wave_source,
                       thrust::host_vector<fp_type> &solution);

    void solve_terminal(thrust::host_vector<fp_type> &prev_solution_0, thrust::host_vector<fp_type> &prev_solution_1,
                        boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, fp_type const &next_time,
                        thrust::host_vector<fp_type> &solution);

    void solve_terminal(thrust::host_vector<fp_type> &prev_solution_0, thrust::host_vector<fp_type> &prev_solution_1,
                        boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, fp_type const &next_time,
                        std::function<fp_type(fp_type, fp_type)> const &wave_source,
                        thrust::host_vector<fp_type> &solution);

    void solve(thrust::host_vector<fp_type> &prev_solution_0, thrust::host_vector<fp_type> &prev_solution_1,
               boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, fp_type const &next_time,
               thrust::host_vector<fp_type> &solution);

    void solve(thrust::host_vector<fp_type> &prev_solution_0, thrust::host_vector<fp_type> &prev_solution_1,
               boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, fp_type const &next_time,
               std::function<fp_type(fp_type, fp_type)> const &wave_source, thrust::host_vector<fp_type> &solution);
};

template <typename fp_type>
void wave_euler_cuda_solver_method<fp_type>::solve_initial(thrust::host_vector<fp_type> &prev_solution_0,
                                                           thrust::host_vector<fp_type> &prev_solution_1,
                                                           boundary_1d_pair<fp_type> const &boundary_pair,
                                                           fp_type const &time, fp_type const &next_time,
                                                           thrust::host_vector<fp_type> &solution)
{
    typedef explicit_wave_scheme<fp_type, thrust::host_vector, std::allocator<fp_type>> wave_scheme;
    wave_scheme::rhs_initial(coefficients_, grid_cfg_, prev_solution_0, prev_solution_1, boundary_pair, time, solution);
}

template <typename fp_type>
void wave_euler_cuda_solver_method<fp_type>::solve_initial(thrust::host_vector<fp_type> &prev_solution_0,
                                                           thrust::host_vector<fp_type> &prev_solution_1,
                                                           boundary_1d_pair<fp_type> const &boundary_pair,
                                                           fp_type const &time, fp_type const &next_time,
                                                           std::function<fp_type(fp_type, fp_type)> const &wave_source,
                                                           thrust::host_vector<fp_type> &solution)
{
    typedef explicit_wave_scheme<fp_type, thrust::host_vector, std::allocator<fp_type>> wave_scheme;
    typedef discretization<dimension_enum::One, fp_type, thrust::host_vector, std::allocator<fp_type>> d_1d;
    // get the right-hand side of the scheme:
    d_1d::of_function(grid_cfg_, time, wave_source, source_);
    wave_scheme::rhs_initial_source(coefficients_, grid_cfg_, prev_solution_0, prev_solution_1, source_, boundary_pair,
                                    time, solution);
}

template <typename fp_type>
void wave_euler_cuda_solver_method<fp_type>::solve_terminal(thrust::host_vector<fp_type> &prev_solution_0,
                                                            thrust::host_vector<fp_type> &prev_solution_1,
                                                            boundary_1d_pair<fp_type> const &boundary_pair,
                                                            fp_type const &time, fp_type const &next_time,
                                                            thrust::host_vector<fp_type> &solution)
{
    typedef explicit_wave_scheme<fp_type, thrust::host_vector, std::allocator<fp_type>> wave_scheme;
    // get the right-hand side of the scheme:
    wave_scheme::rhs_terminal(coefficients_, grid_cfg_, prev_solution_0, prev_solution_1, boundary_pair, time,
                              solution);
}

template <typename fp_type>
void wave_euler_cuda_solver_method<fp_type>::solve_terminal(thrust::host_vector<fp_type> &prev_solution_0,
                                                            thrust::host_vector<fp_type> &prev_solution_1,
                                                            boundary_1d_pair<fp_type> const &boundary_pair,
                                                            fp_type const &time, fp_type const &next_time,
                                                            std::function<fp_type(fp_type, fp_type)> const &wave_source,
                                                            thrust::host_vector<fp_type> &solution)
{
    typedef explicit_wave_scheme<fp_type, thrust::host_vector, std::allocator<fp_type>> wave_scheme;
    typedef discretization<dimension_enum::One, fp_type, thrust::host_vector, std::allocator<fp_type>> d_1d;
    // get the right-hand side of the scheme:
    d_1d::of_function(grid_cfg_, time, wave_source, source_);
    wave_scheme::rhs_terminal_source(coefficients_, grid_cfg_, prev_solution_0, prev_solution_1, source_, boundary_pair,
                                     time, solution);
}

template <typename fp_type>
void wave_euler_cuda_solver_method<fp_type>::solve(thrust::host_vector<fp_type> &prev_solution_0,
                                                   thrust::host_vector<fp_type> &prev_solution_1,
                                                   boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time,
                                                   fp_type const &next_time, thrust::host_vector<fp_type> &solution)
{
    typedef explicit_wave_cuda_scheme<fp_type> wave_scheme;
    // get the right-hand side of the scheme:
    wave_scheme::rhs(coefficients_, kernel_, grid_cfg_, prev_solution_0, prev_solution_1, boundary_pair, time,
                     solution);
}

template <typename fp_type>
void wave_euler_cuda_solver_method<fp_type>::solve(thrust::host_vector<fp_type> &prev_solution_0,
                                                   thrust::host_vector<fp_type> &prev_solution_1,
                                                   boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time,
                                                   fp_type const &next_time,
                                                   std::function<fp_type(fp_type, fp_type)> const &wave_source,
                                                   thrust::host_vector<fp_type> &solution)
{
    typedef explicit_wave_cuda_scheme<fp_type> wave_scheme;
    typedef discretization<dimension_enum::One, fp_type, thrust::host_vector, std::allocator<fp_type>> d_1d;
    // get the right-hand side of the scheme:
    d_1d::of_function(grid_cfg_, time, wave_source, source_);
    wave_scheme::rhs_source(coefficients_, kernel_, grid_cfg_, prev_solution_0, prev_solution_1, source_, boundary_pair,
                            time, solution);
}
} // namespace one_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_WAVE_EULER_CUDA_SOLVER_METHOD_HPP_
