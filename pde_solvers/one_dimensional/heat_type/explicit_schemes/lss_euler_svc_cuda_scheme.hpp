#if !defined(_LSS_EULER_SVC_CUDA_SCHEME_HPP_)
#define _LSS_EULER_SVC_CUDA_SCHEME_HPP_

#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "boundaries/lss_boundary.hpp"
#include "boundaries/lss_dirichlet_boundary.hpp"
#include "boundaries/lss_neumann_boundary.hpp"
#include "boundaries/lss_robin_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include "containers/lss_container_2d.hpp"
#include "discretization/lss_discretization.hpp"
#include "pde_solvers/lss_pde_discretization_config.hpp"

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
using lss_enumerations::traverse_direction_enum;
using lss_utility::function_triplet_t;
using lss_utility::NaN;
using lss_utility::pair_t;
using lss_utility::range;
using lss_utility::sptr_t;

/**
 * core_kernel function
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
__global__ void core_kernel(fp_type const *a_coeff, fp_type const *b_coeff, fp_type const *d_coeff,
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
 * core_kernel function
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
__global__ void core_kernel(fp_type const *a_coeff, fp_type const *b_coeff, fp_type const *d_coeff, fp_type time_step,
                            fp_type const *input, fp_type const *source, fp_type *solution, const std::size_t size)
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
 * euler_svc_cuda_kernel object
 */
template <typename fp_type> class euler_svc_cuda_kernel
{
    typedef discretization<dimension_enum::One, fp_type, thrust::host_vector, std::allocator<fp_type>> d_1d;

  private:
    pair_t<fp_type> steps_;
    range<fp_type> spacer_;
    std::size_t space_size_;
    thrust::device_vector<fp_type> d_a_;
    thrust::device_vector<fp_type> d_b_;
    thrust::device_vector<fp_type> d_d_;

    void initialize(function_triplet_t<fp_type> const &fun_triplet)
    {
        auto const &start_x = spacer_.lower();
        auto const &h = steps_.second;
        auto const &a = std::get<0>(fun_triplet);
        auto const &b = std::get<1>(fun_triplet);
        auto const &d = std::get<2>(fun_triplet);
        thrust::host_vector<fp_type> h_a(space_size_);
        thrust::host_vector<fp_type> h_b(space_size_);
        thrust::host_vector<fp_type> h_d(space_size_);
        // discretize on host
        d_1d::of_function(start_x, h, a, h_a);
        d_1d::of_function(start_x, h, b, h_b);
        d_1d::of_function(start_x, h, d, h_d);
        // copy to device
        d_a_.resize(space_size_);
        d_b_.resize(space_size_);
        d_d_.resize(space_size_);
        thrust::copy(h_a.begin(), h_a.end(), d_a_.begin());
        thrust::copy(h_b.begin(), h_b.end(), d_b_.begin());
        thrust::copy(h_d.begin(), h_d.end(), d_d_.begin());
    }

  public:
    explicit euler_svc_cuda_kernel(function_triplet_t<fp_type> const &fun_triplet, pair_t<fp_type> const &steps,
                                   range<fp_type> const &space_range, std::size_t const space_size)
        : steps_{steps}, spacer_{space_range}, space_size_{space_size}
    {
        initialize(fun_triplet);
    }
    ~euler_svc_cuda_kernel()
    {
    }
    void launch(thrust::device_vector<fp_type> const &input, thrust::device_vector<fp_type> &solution);

    void launch(thrust::device_vector<fp_type> const &input, thrust::device_vector<fp_type> const &source,
                thrust::device_vector<fp_type> &solution);
};

template <typename fp_type> using euler_svc_cuda_kernel_ptr = sptr_t<euler_svc_cuda_kernel<fp_type>>;

template <typename fp_type>
using explicit_svc_cuda_scheme_function =
    std::function<void(function_triplet_t<fp_type> const &, euler_svc_cuda_kernel_ptr<fp_type> const &,
                       pair_t<fp_type> const &, thrust::host_vector<fp_type> const &,
                       thrust::host_vector<fp_type> const &, boundary_1d_pair<fp_type> const &, fp_type const &,
                       thrust::host_vector<fp_type> &)>;

/**
 * explicit_svc_cuda_scheme object
 */
template <typename fp_type> class explicit_svc_cuda_scheme
{
    typedef explicit_svc_cuda_scheme_function<fp_type> scheme_function_t;

  public:
    static scheme_function_t const get(bool is_homogeneus)
    {
        const fp_type two = static_cast<fp_type>(2.0);
        auto scheme_fun_h =
            [=](function_triplet_t<fp_type> const &coefficients, euler_svc_cuda_kernel_ptr<fp_type> const &kernel,
                std::pair<fp_type, fp_type> const &steps, thrust::host_vector<fp_type> const &input,
                thrust::host_vector<fp_type> const &inhom_input, boundary_1d_pair<fp_type> const &boundary_pair,
                fp_type const &time, thrust::host_vector<fp_type> &solution) {
                auto const &first_bnd = boundary_pair.first;
                auto const &second_bnd = boundary_pair.second;
                auto const &a = std::get<0>(coefficients);
                auto const &b = std::get<1>(coefficients);
                auto const &d = std::get<2>(coefficients);
                auto const h = steps.second;
                fp_type m{};
                // for lower boundaries first:
                if (auto const &ptr = std::dynamic_pointer_cast<dirichlet_boundary_1d<fp_type>>(first_bnd))
                {
                    solution[0] = ptr->value(time);
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(first_bnd))
                {
                    const fp_type beta = two * h * ptr->value(time);
                    m = static_cast<fp_type>(0);
                    solution[0] = beta * a(m * h) + b(m * h) * input[0] + (a(m * h) + d(m * h)) * input[1];
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_bnd))
                {
                    const fp_type beta = two * h * ptr->value(time);
                    const fp_type alpha = two * h * ptr->linear_value(time);
                    m = static_cast<fp_type>(0);
                    solution[0] =
                        (b(m * h) + alpha * a(m * h)) * input[0] + (a(m * h) + d(m * h)) * input[1] + beta * a(m * h);
                }
                // for upper boundaries second:
                const std::size_t N = solution.size() - 1;
                if (auto const &ptr = std::dynamic_pointer_cast<dirichlet_boundary_1d<fp_type>>(second_bnd))
                {
                    solution[N] = ptr->value(time);
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(second_bnd))
                {
                    const fp_type delta = two * h * ptr->value(time);
                    m = static_cast<fp_type>(N);
                    solution[N] = (a(m * h) + d(m * h)) * input[N - 1] + b(m * h) * input[N] - delta * d(m * h);
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
                {
                    const fp_type delta = two * h * ptr->value(time);
                    const fp_type gamma = two * h * ptr->linear_value(time);
                    m = static_cast<fp_type>(N);
                    solution[N] = (a(m * h) + d(m * h)) * input[N - 1] + (b(m * h) - gamma * d(m * h)) * input[N] -
                                  delta * d(m * h);
                }
                // light-weight object with cuda kernel computing the solution:
                thrust::device_vector<fp_type> d_input(input);
                thrust::device_vector<fp_type> d_solution(solution);
                kernel->launch(d_input, d_solution);
                thrust::copy(d_solution.begin(), d_solution.end(), solution.begin());
            };
        auto scheme_fun_nh =
            [=](function_triplet_t<fp_type> const &coefficients, euler_svc_cuda_kernel_ptr<fp_type> const &kernel,
                std::pair<fp_type, fp_type> const &steps, thrust::host_vector<fp_type> const &input,
                thrust::host_vector<fp_type> const &inhom_input, boundary_1d_pair<fp_type> const &boundary_pair,
                fp_type const &time, thrust::host_vector<fp_type> &solution) {
                auto const &first_bnd = boundary_pair.first;
                auto const &second_bnd = boundary_pair.second;
                auto const &a = std::get<0>(coefficients);
                auto const &b = std::get<1>(coefficients);
                auto const &d = std::get<2>(coefficients);
                auto const k = steps.first;
                auto const h = steps.second;
                fp_type m{};

                // for lower boundaries first:
                if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(first_bnd))
                {
                    const fp_type beta = two * h * ptr->value(time);
                    m = static_cast<fp_type>(0);
                    solution[0] =
                        beta * a(m * h) + b(m * h) * input[0] + (a(m * h) + d(m * h)) * input[1] + k * inhom_input[0];
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_bnd))
                {
                    const fp_type beta = two * h * ptr->value(time);
                    const fp_type alpha = two * h * ptr->linear_value(time);
                    m = static_cast<fp_type>(0);
                    solution[0] = (b(m * h) + alpha * a(m * h)) * input[0] + (a(m * h) + d(m * h)) * input[1] +
                                  beta * a(m * h) + k * inhom_input[0];
                }
                // for upper boundaries second:
                const std::size_t N = solution.size() - 1;
                if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(second_bnd))
                {
                    const fp_type delta = two * h * ptr->value(time);
                    m = static_cast<fp_type>(N);
                    solution[N] = (a(m * h) + d(m * h)) * input[N - 1] + b(m * h) * input[N] - delta * d(m * h) +
                                  k * inhom_input[N];
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
                {
                    const fp_type delta = two * h * ptr->value(time);
                    const fp_type gamma = two * h * ptr->linear_value(time);
                    m = static_cast<fp_type>(N);
                    solution[N] = (a(m * h) + d(m * h)) * input[N - 1] + (b(m * h) - gamma * d(m * h)) * input[N] -
                                  delta * d(m * h) + k * inhom_input[N];
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
 * euler_svc_cuda_time_loop object
 */
template <typename fp_type, template <typename, typename> typename container, typename allocator>
class euler_svc_cuda_time_loop
{
    typedef container<fp_type, allocator> container_t;
    typedef container_2d<fp_type, container, allocator> container_2d_t;

  public:
    template <typename scheme_function>
    static void run(function_triplet_t<fp_type> const &func_triplet, scheme_function &scheme_fun,
                    boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &space_range,
                    range<fp_type> const &time_range, std::size_t const &last_time_idx,
                    std::pair<fp_type, fp_type> const &steps, traverse_direction_enum const &traverse_dir,
                    container_t &solution);

    template <typename scheme_function>
    static void run(function_triplet_t<fp_type> const &func_triplet, scheme_function &scheme_fun,
                    boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &space_range,
                    range<fp_type> const &time_range, std::size_t const &last_time_idx,
                    std::pair<fp_type, fp_type> const &steps, traverse_direction_enum const &traverse_dir,
                    container_t &solution, std::function<fp_type(fp_type, fp_type)> const &heat_source,
                    container_t &source);

    template <typename scheme_function>
    static void run_with_stepping(function_triplet_t<fp_type> const &func_triplet, scheme_function &scheme_fun,
                                  boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &space_range,
                                  range<fp_type> const &time_range, std::size_t const &last_time_idx,
                                  std::pair<fp_type, fp_type> const &steps, traverse_direction_enum const &traverse_dir,
                                  container_t &solution, container_2d_t &solutions);

    template <typename scheme_function>
    static void run_with_stepping(function_triplet_t<fp_type> const &func_triplet, scheme_function &scheme_fun,
                                  boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &space_range,
                                  range<fp_type> const &time_range, std::size_t const &last_time_idx,
                                  std::pair<fp_type, fp_type> const &steps, traverse_direction_enum const &traverse_dir,
                                  container_t &solution, container_2d_t &solutions,
                                  std::function<fp_type(fp_type, fp_type)> const &heat_source, container_t &source);
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
template <typename scheme_function>
void euler_svc_cuda_time_loop<fp_type, container, allocator>::run(
    function_triplet_t<fp_type> const &func_triplet, scheme_function &scheme_fun,
    boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &space_range, range<fp_type> const &time_range,
    std::size_t const &last_time_idx, std::pair<fp_type, fp_type> const &steps,
    traverse_direction_enum const &traverse_dir, container_t &solution)
{
    const std::size_t sol_size = solution.size();
    // ranges and steps:
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type start_x = space_range.lower();
    const fp_type k = std::get<0>(steps);
    const fp_type h = std::get<1>(steps);
    // get function for sweeps:
    auto const &a = std::get<0>(func_triplet);
    auto const &b = std::get<1>(func_triplet);
    auto const &d = std::get<2>(func_triplet);
    // create a kernel:
    auto const &kernel = std::make_shared<euler_svc_cuda_kernel<fp_type>>(func_triplet, steps, space_range, sol_size);
    // create host vectors:
    thrust::host_vector<fp_type> h_solution(sol_size);
    thrust::copy(solution.begin(), solution.end(), h_solution.begin());
    thrust::host_vector<fp_type> h_next_solution(sol_size);

    fp_type time{};
    std::size_t time_idx{};
    if (traverse_dir == traverse_direction_enum::Forward)
    {
        time = start_time + k;
        time_idx = 1;
        while (time_idx <= last_time_idx)
        {
            scheme_fun(func_triplet, kernel, steps, h_solution, thrust::host_vector<fp_type>(), boundary_pair, time,
                       h_next_solution);
            h_solution = h_next_solution;
            time += k;
            time_idx++;
        }
    }
    else if (traverse_dir == traverse_direction_enum::Backward)
    {
        time = end_time - k;
        time_idx = last_time_idx;
        do
        {
            time_idx--;
            scheme_fun(func_triplet, kernel, steps, h_solution, thrust::host_vector<fp_type>(), boundary_pair, time,
                       h_next_solution);
            h_solution = h_next_solution;
            time -= k;
        } while (time_idx > 0);
    }
    else
    {
        throw std::exception("Unreachable");
    }
    thrust::copy(h_solution.begin(), h_solution.end(), solution.begin());
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
template <typename scheme_function>
void euler_svc_cuda_time_loop<fp_type, container, allocator>::run(
    function_triplet_t<fp_type> const &func_triplet, scheme_function &scheme_fun,
    boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &space_range, range<fp_type> const &time_range,
    std::size_t const &last_time_idx, std::pair<fp_type, fp_type> const &steps,
    traverse_direction_enum const &traverse_dir, container_t &solution,
    std::function<fp_type(fp_type, fp_type)> const &heat_source, container_t &source)
{
    typedef discretization<dimension_enum::One, fp_type, thrust::host_vector, std::allocator<fp_type>> d_1d;

    const std::size_t sol_size = solution.size();
    // ranges and steps:
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type start_x = space_range.lower();
    const fp_type k = std::get<0>(steps);
    const fp_type h = std::get<1>(steps);
    // get function for sweeps:
    auto const &a = std::get<0>(func_triplet);
    auto const &b = std::get<1>(func_triplet);
    auto const &d = std::get<2>(func_triplet);
    // create a kernel:
    auto const &kernel = std::make_shared<euler_svc_cuda_kernel<fp_type>>(func_triplet, steps, space_range, sol_size);
    // create host vectors:
    thrust::host_vector<fp_type> h_solution(sol_size);
    thrust::host_vector<fp_type> h_next_solution(sol_size);
    thrust::copy(solution.begin(), solution.end(), h_solution.begin());
    thrust::host_vector<fp_type> h_source(sol_size);

    fp_type time{};
    std::size_t time_idx{};
    if (traverse_dir == traverse_direction_enum::Forward)
    {
        d_1d::of_function(start_x, h, start_time, heat_source, h_source);
        time = start_time + k;
        time_idx = 1;
        while (time_idx <= last_time_idx)
        {
            scheme_fun(func_triplet, kernel, steps, h_solution, h_source, boundary_pair, time, h_next_solution);
            h_solution = h_next_solution;
            d_1d::of_function(start_x, h, time, heat_source, h_source);
            time += k;
            time_idx++;
        }
    }
    else if (traverse_dir == traverse_direction_enum::Backward)
    {
        d_1d::of_function(start_x, h, end_time, heat_source, h_source);
        time = end_time - k;
        time_idx = last_time_idx;
        do
        {
            time_idx--;
            scheme_fun(func_triplet, kernel, steps, h_solution, h_source, boundary_pair, time, h_next_solution);
            h_solution = h_next_solution;
            d_1d::of_function(start_x, h, time, heat_source, h_source);
            time -= k;
        } while (time_idx > 0);
    }
    else
    {
        throw std::exception("Unreachable");
    }
    thrust::copy(h_solution.begin(), h_solution.end(), solution.begin());
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
template <typename scheme_function>
void euler_svc_cuda_time_loop<fp_type, container, allocator>::run_with_stepping(
    function_triplet_t<fp_type> const &func_triplet, scheme_function &scheme_fun,
    boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &space_range, range<fp_type> const &time_range,
    std::size_t const &last_time_idx, std::pair<fp_type, fp_type> const &steps,
    traverse_direction_enum const &traverse_dir, container_t &solution, container_2d_t &solutions)
{
    const std::size_t sol_size = solution.size();
    // ranges and steps:
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type start_x = space_range.lower();
    const fp_type k = std::get<0>(steps);
    const fp_type h = std::get<1>(steps);
    // get function for sweeps:
    auto const &a = std::get<0>(func_triplet);
    auto const &b = std::get<1>(func_triplet);
    auto const &d = std::get<2>(func_triplet);
    // create a kernel:
    auto const &kernel = std::make_shared<euler_svc_cuda_kernel<fp_type>>(func_triplet, steps, space_range, sol_size);
    // create host vectors:
    thrust::host_vector<fp_type> h_solution(sol_size);
    thrust::copy(solution.begin(), solution.end(), h_solution.begin());
    thrust::host_vector<fp_type> h_next_solution(sol_size);

    fp_type time{};
    std::size_t time_idx{};
    if (traverse_dir == traverse_direction_enum::Forward)
    {
        // store the initial solution:
        solutions(0, solution);
        time = start_time + k;
        time_idx = 1;
        while (time_idx <= last_time_idx)
        {
            scheme_fun(func_triplet, kernel, steps, h_solution, thrust::host_vector<fp_type>(), boundary_pair, time,
                       h_next_solution);
            h_solution = h_next_solution;
            thrust::copy(h_solution.begin(), h_solution.end(), solution.begin());
            solutions(time_idx, solution);
            time += k;
            time_idx++;
        }
    }
    else if (traverse_dir == traverse_direction_enum::Backward)
    {
        // store the initial solution:
        solutions(last_time_idx, solution);
        time = end_time - k;
        time_idx = last_time_idx;
        do
        {
            time_idx--;
            scheme_fun(func_triplet, kernel, steps, h_solution, thrust::host_vector<fp_type>(), boundary_pair, time,
                       h_next_solution);
            h_solution = h_next_solution;
            thrust::copy(h_solution.begin(), h_solution.end(), solution.begin());
            solutions(time_idx, solution);
            time -= k;
        } while (time_idx > 0);
    }
    else
    {
        throw std::exception("Unreachable");
    }
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
template <typename scheme_function>
void euler_svc_cuda_time_loop<fp_type, container, allocator>::run_with_stepping(
    function_triplet_t<fp_type> const &func_triplet, scheme_function &scheme_fun,
    boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &space_range, range<fp_type> const &time_range,
    std::size_t const &last_time_idx, std::pair<fp_type, fp_type> const &steps,
    traverse_direction_enum const &traverse_dir, container_t &solution, container_2d_t &solutions,
    std::function<fp_type(fp_type, fp_type)> const &heat_source, container_t &source)
{
    typedef discretization<dimension_enum::One, fp_type, thrust::host_vector, std::allocator<fp_type>> d_1d;

    const std::size_t sol_size = solution.size();
    // ranges and steps:
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type start_x = space_range.lower();
    const fp_type k = std::get<0>(steps);
    const fp_type h = std::get<1>(steps);
    // get function for sweeps:
    auto const &a = std::get<0>(func_triplet);
    auto const &b = std::get<1>(func_triplet);
    auto const &d = std::get<2>(func_triplet);
    // create a kernel:
    auto const &kernel = std::make_shared<euler_svc_cuda_kernel<fp_type>>(func_triplet, steps, space_range, sol_size);
    // create host vectors:
    thrust::host_vector<fp_type> h_solution(sol_size);
    thrust::host_vector<fp_type> h_next_solution(sol_size);
    thrust::copy(solution.begin(), solution.end(), h_solution.begin());
    thrust::host_vector<fp_type> h_source(sol_size);

    fp_type time{};
    std::size_t time_idx{};
    if (traverse_dir == traverse_direction_enum::Forward)
    {
        // store the initial solution:
        solutions(0, solution);
        d_1d::of_function(start_x, h, start_time, heat_source, h_source);
        time = start_time + k;
        time_idx = 1;
        while (time_idx <= last_time_idx)
        {
            scheme_fun(func_triplet, kernel, steps, h_solution, source, boundary_pair, time, h_next_solution);
            h_solution = h_next_solution;
            d_1d::of_function(start_x, h, time, heat_source, h_source);
            thrust::copy(h_solution.begin(), h_solution.end(), solution.begin());
            solutions(time_idx, solution);
            time += k;
            time_idx++;
        }
    }
    else if (traverse_dir == traverse_direction_enum::Backward)
    {
        // store the initial solution:
        solutions(last_time_idx, solution);
        d_1d::of_function(start_x, h, end_time, heat_source, h_source);
        time = end_time - k;
        time_idx = last_time_idx;
        do
        {
            time_idx--;
            scheme_fun(func_triplet, kernel, steps, h_solution, source, boundary_pair, time, h_next_solution);
            h_solution = h_next_solution;
            d_1d::of_function(start_x, h, time, heat_source, h_source);
            thrust::copy(h_solution.begin(), h_solution.end(), solution.begin());
            solutions(time_idx, solution);
            time -= k;
        } while (time_idx > 0);
    }
    else
    {
        throw std::exception("Unreachable");
    }
}

/**
 * euler_svc_cuda_scheme object
 */
template <typename fp_type, template <typename, typename> typename container, typename allocator>
class euler_svc_cuda_scheme
{
    typedef euler_svc_cuda_time_loop<fp_type, container, allocator> loop;
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container<fp_type, allocator> container_t;

  private:
    function_triplet_t<fp_type> fun_triplet_;
    boundary_1d_pair<fp_type> boundary_pair_;
    pde_discretization_config_1d_ptr<fp_type> discretization_cfg_;

    bool is_stable()
    {
        const fp_type zero = static_cast<fp_type>(0.0);
        const fp_type half = static_cast<fp_type>(0.5);
        const fp_type one = static_cast<fp_type>(1.0);
        const fp_type two = static_cast<fp_type>(2.0);
        auto const &A = std::get<0>(fun_triplet_);
        auto const &B = std::get<1>(fun_triplet_);
        auto const &D = std::get<2>(fun_triplet_);
        const fp_type k = discretization_cfg_->time_step();
        const fp_type h = discretization_cfg_->space_step();
        const fp_type lambda = k / (h * h);
        const fp_type gamma = k / (two * h);
        const fp_type delta = half * k;
        auto const &a = [=](fp_type x) { return ((A(x) + D(x)) / (two * lambda)); };
        auto const &b = [=](fp_type x) { return ((D(x) - A(x)) / (two * gamma)); };
        auto const &c = [=](fp_type x) { return ((lambda * a(x) - B(x)) / delta); };
        const std::size_t space_size = discretization_cfg_->number_of_space_points();
        fp_type m{};
        for (std::size_t i = 0; i < space_size; ++i)
        {
            m = static_cast<fp_type>(i);
            if (c(m * h) > zero)
                return false;
            if ((two * lambda * a(m * h) - k * c(m * h)) > one)
                return false;
            if (((gamma * std::abs(b(m * h))) * (gamma * std::abs(b(m * h)))) > (two * lambda * a(m * h)))
                return false;
        }
        return true;
    }

    void initialize()
    {
        LSS_ASSERT(is_stable() == true, "The chosen scheme is not stable");
    }

    explicit euler_svc_cuda_scheme() = delete;

  public:
    euler_svc_cuda_scheme(function_triplet_t<fp_type> const &fun_triplet,
                          boundary_1d_pair<fp_type> const &boundary_pair,
                          pde_discretization_config_1d_ptr<fp_type> const &discretization_config)
        : fun_triplet_{fun_triplet}, boundary_pair_{boundary_pair}, discretization_cfg_{discretization_config}
    {
        initialize();
    }

    ~euler_svc_cuda_scheme()
    {
    }

    void operator()(container_t &solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type)> const &heat_source, traverse_direction_enum traverse_dir)
    {
        const fp_type one = static_cast<fp_type>(1.0);
        const fp_type two = static_cast<fp_type>(2.0);
        const range<fp_type> spacer = discretization_cfg_->space_range();
        const range<fp_type> timer = discretization_cfg_->time_range();
        const fp_type k = discretization_cfg_->time_step();
        const fp_type h = discretization_cfg_->space_step();
        auto const &A = std::get<0>(fun_triplet_);
        auto const &B = std::get<1>(fun_triplet_);
        auto const &D = std::get<2>(fun_triplet_);
        // build the scheme coefficients:
        auto const &a = [&](fp_type x) { return A(x); };
        auto const &b = [&](fp_type x) { return (one - two * B(x)); };
        auto const &d = [&](fp_type x) { return D(x); };
        // save solution size:
        const std::size_t sol_size = solution.size();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        // wrap up the functions:
        auto const &fun_trip = std::make_tuple(a, b, d);
        // create a container to carry discretized source heat
        container_t source(sol_size, NaN<fp_type>());
        auto const &steps = std::make_pair(k, h);
        const bool is_homogeneous = !is_heat_sourse_set;
        auto scheme_function = explicit_svc_cuda_scheme<fp_type>::get(is_homogeneous);
        if (is_heat_sourse_set)
        {
            loop::run(fun_trip, scheme_function, boundary_pair_, spacer, timer, last_time_idx, steps, traverse_dir,
                      solution, heat_source, source);
        }
        else
        {
            loop::run(fun_trip, scheme_function, boundary_pair_, spacer, timer, last_time_idx, steps, traverse_dir,
                      solution);
        }
    }

    void operator()(container_t &solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type)> const &heat_source, traverse_direction_enum traverse_dir,
                    container_2d<fp_type, container, allocator> &solutions)
    {
        const fp_type one = static_cast<fp_type>(1.0);
        const fp_type two = static_cast<fp_type>(2.0);
        const range<fp_type> spacer = discretization_cfg_->space_range();
        const range<fp_type> timer = discretization_cfg_->time_range();
        const fp_type k = discretization_cfg_->time_step();
        const fp_type h = discretization_cfg_->space_step();
        auto const &A = std::get<0>(fun_triplet_);
        auto const &B = std::get<1>(fun_triplet_);
        auto const &D = std::get<2>(fun_triplet_);
        // build the scheme coefficients:
        auto const &a = [&](fp_type x) { return A(x); };
        auto const &b = [&](fp_type x) { return (one - two * B(x)); };
        auto const &d = [&](fp_type x) { return D(x); };
        // save solution size:
        const std::size_t sol_size = solution.size();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        // wrap up the functions:
        auto const &fun_trip = std::make_tuple(a, b, d);
        // create a container to carry discretized source heat
        container_t source(sol_size, NaN<fp_type>());
        auto const &steps = std::make_pair(k, h);
        const bool is_homogeneous = !is_heat_sourse_set;
        auto scheme_function = explicit_svc_cuda_scheme<fp_type>::get(is_homogeneous);
        if (is_heat_sourse_set)
        {
            loop::run_with_stepping(fun_trip, scheme_function, boundary_pair_, spacer, timer, last_time_idx, steps,
                                    traverse_dir, solution, solutions, heat_source, source);
        }
        else
        {
            loop::run_with_stepping(fun_trip, scheme_function, boundary_pair_, spacer, timer, last_time_idx, steps,
                                    traverse_dir, solution, solutions);
        }
    }
};

} // namespace one_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_EULER_SVC_CUDA_SCHEME_HPP_
