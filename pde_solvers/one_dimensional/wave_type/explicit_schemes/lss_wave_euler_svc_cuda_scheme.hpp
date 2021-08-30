#if !defined(_LSS_WAVE_EULER_SVC_CUDA_SCHEME_HPP_)
#define _LSS_WAVE_EULER_SVC_CUDA_SCHEME_HPP_

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
#include "lss_wave_euler_svc_scheme.hpp"
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
using lss_utility::function_quintuple_t;
using lss_utility::NaN;
using lss_utility::pair_t;
using lss_utility::range;

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
    wave_euler_svc_cuda_kernel object
 */
template <typename fp_type> class wave_euler_svc_cuda_kernel
{
    typedef discretization<dimension_enum::One, fp_type, thrust::host_vector, std::allocator<fp_type>> d_1d;

  private:
    pair_t<fp_type> steps_;
    range<fp_type> spacer_;
    std::size_t space_size_;
    thrust::device_vector<fp_type> d_a_;
    thrust::device_vector<fp_type> d_b_;
    thrust::device_vector<fp_type> d_c_;
    thrust::device_vector<fp_type> d_d_;

    void initialize(function_quintuple_t<fp_type> const &fun_quintuple)
    {
        auto const &start_x = spacer_.lower();
        auto const &h = steps_.second;
        auto const &a = std::get<0>(fun_quintuple);
        auto const &b = std::get<1>(fun_quintuple);
        auto const &c = std::get<2>(fun_quintuple);
        auto const &d = std::get<3>(fun_quintuple);
        thrust::host_vector<fp_type> h_a(space_size_);
        thrust::host_vector<fp_type> h_b(space_size_);
        thrust::host_vector<fp_type> h_c(space_size_);
        thrust::host_vector<fp_type> h_d(space_size_);
        // discretize on host
        d_1d::of_function(start_x, h, a, h_a);
        d_1d::of_function(start_x, h, b, h_b);
        d_1d::of_function(start_x, h, c, h_c);
        d_1d::of_function(start_x, h, d, h_d);
        // copy to device
        d_a_.resize(space_size_);
        d_b_.resize(space_size_);
        d_c_.resize(space_size_);
        d_d_.resize(space_size_);
        thrust::copy(h_a.begin(), h_a.end(), d_a_.begin());
        thrust::copy(h_b.begin(), h_b.end(), d_b_.begin());
        thrust::copy(h_c.begin(), h_c.end(), d_c_.begin());
        thrust::copy(h_d.begin(), h_d.end(), d_d_.begin());
    }

  public:
    explicit wave_euler_svc_cuda_kernel(function_quintuple_t<fp_type> const &fun_quintuple,
                                        pair_t<fp_type> const &steps, range<fp_type> const &space_range,
                                        std::size_t const space_size)
        : steps_{steps}, spacer_{space_range}, space_size_{space_size}
    {
        initialize(fun_quintuple);
    }
    ~wave_euler_svc_cuda_kernel()
    {
    }
    void launch(thrust::device_vector<fp_type> const &input_0, thrust::device_vector<fp_type> const &input_1,
                thrust::device_vector<fp_type> &solution);

    void launch(thrust::device_vector<fp_type> const &input_0, thrust::device_vector<fp_type> const &input_1,
                thrust::device_vector<fp_type> const &source, thrust::device_vector<fp_type> &solution);
};

template <typename fp_type> using wave_euler_svc_cuda_kernel_ptr = sptr_t<wave_euler_svc_cuda_kernel<fp_type>>;

template <typename fp_type>
using explicit_wave_svc_cuda_scheme_function =
    std::function<void(function_quintuple_t<fp_type> const &, wave_euler_svc_cuda_kernel_ptr<fp_type> const &kernel,
                       pair_t<fp_type> const &, thrust::host_vector<fp_type> const &,
                       thrust::host_vector<fp_type> const &, thrust::host_vector<fp_type> const &,
                       boundary_1d_pair<fp_type> const &, fp_type const &, thrust::host_vector<fp_type> &)>;

/**
    explicit_wave_svc_cuda_scheme object
 */
template <typename fp_type> class explicit_wave_svc_cuda_scheme
{
    typedef explicit_wave_svc_cuda_scheme_function<fp_type> cuda_scheme_function_t;

  public:
    static cuda_scheme_function_t const get(bool is_homogeneus)
    {
        const fp_type two = static_cast<fp_type>(2.0);
        auto scheme_fun_h =
            [=](function_quintuple_t<fp_type> const &coefficients,
                wave_euler_svc_cuda_kernel_ptr<fp_type> const &kernel, std::pair<fp_type, fp_type> const &steps,
                thrust::host_vector<fp_type> const &input_0, thrust::host_vector<fp_type> const &input_1,
                thrust::host_vector<fp_type> const &inhom_input, boundary_1d_pair<fp_type> const &boundary_pair,
                fp_type const &time, thrust::host_vector<fp_type> &solution) {
                auto const &first_bnd = boundary_pair.first;
                auto const &second_bnd = boundary_pair.second;
                auto const &a = std::get<0>(coefficients);
                auto const &b = std::get<1>(coefficients);
                auto const &c = std::get<2>(coefficients);
                auto const &d = std::get<3>(coefficients);
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
                    solution[0] = beta * a(m * h) + c(m * h) * input_1[0] + (a(m * h) + b(m * h)) * input_1[1] -
                                  d(m * h) * input_0[0];
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_bnd))
                {
                    const fp_type beta = two * h * ptr->value(time);
                    const fp_type alpha = two * h * ptr->linear_value(time);
                    m = static_cast<fp_type>(0);
                    solution[0] = beta * a(m * h) + (c(m * h) + alpha * a(m * h)) * input_1[0] +
                                  (a(m * h) + b(m * h)) * input_1[1] - d(m * h) * input_0[0];
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
                    solution[N] = (a(m * h) + b(m * h)) * input_1[N - 1] + c(m * h) * input_1[N] - delta * b(m * h) -
                                  d(m * h) * input_0[N];
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
                {
                    const fp_type delta = two * h * ptr->value(time);
                    const fp_type gamma = two * h * ptr->linear_value(time);
                    m = static_cast<fp_type>(N);
                    solution[N] = (a(m * h) + b(m * h)) * input_1[N - 1] + (c(m * h) - gamma * b(m * h)) * input_1[N] -
                                  delta * b(m * h) - d(m * h) * input_0[N];
                }

                thrust::device_vector<fp_type> d_input_0(input_0);
                thrust::device_vector<fp_type> d_input_1(input_1);
                thrust::device_vector<fp_type> d_solution(solution);
                kernel->launch(d_input_0, d_input_1, d_solution);
                thrust::copy(d_solution.begin(), d_solution.end(), solution.begin());
            };
        auto scheme_fun_nh =
            [=](function_quintuple_t<fp_type> const &coefficients,
                wave_euler_svc_cuda_kernel_ptr<fp_type> const &kernel, std::pair<fp_type, fp_type> const &steps,
                thrust::host_vector<fp_type> const &input_0, thrust::host_vector<fp_type> const &input_1,
                thrust::host_vector<fp_type> const &inhom_input, boundary_1d_pair<fp_type> const &boundary_pair,
                fp_type const &time, thrust::host_vector<fp_type> &solution) {
                auto const &first_bnd = boundary_pair.first;
                auto const &second_bnd = boundary_pair.second;
                auto const &a = std::get<0>(coefficients);
                auto const &b = std::get<1>(coefficients);
                auto const &c = std::get<2>(coefficients);
                auto const &d = std::get<3>(coefficients);
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
                    solution[0] = beta * a(m * h) + c(m * h) * input_1[0] + (a(m * h) + b(m * h)) * input_1[1] -
                                  d(m * h) * input_0[0] + inhom_input[0];
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_bnd))
                {
                    const fp_type beta = two * h * ptr->value(time);
                    const fp_type alpha = two * h * ptr->linear_value(time);
                    m = static_cast<fp_type>(0);
                    solution[0] = beta * a(m * h) + (c(m * h) + alpha * a(m * h)) * input_1[0] +
                                  (a(m * h) + b(m * h)) * input_1[1] - d(m * h) * input_0[0] + inhom_input[0];
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
                    solution[N] = (a(m * h) + b(m * h)) * input_1[N - 1] + c(m * h) * input_1[N] - delta * b(m * h) -
                                  d(m * h) * input_0[N] + inhom_input[N];
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
                {
                    const fp_type delta = two * h * ptr->value(time);
                    const fp_type gamma = two * h * ptr->linear_value(time);
                    m = static_cast<fp_type>(N);
                    solution[N] = (a(m * h) + b(m * h)) * input_1[N - 1] + (c(m * h) - gamma * b(m * h)) * input_1[N] -
                                  delta * b(m * h) - d(m * h) * input_0[N] + inhom_input[N];
                }

                thrust::device_vector<fp_type> d_input_0(input_0);
                thrust::device_vector<fp_type> d_input_1(input_1);
                thrust::device_vector<fp_type> d_source(inhom_input);
                thrust::device_vector<fp_type> d_solution(solution);
                kernel->launch(d_input_0, d_input_1, d_source, d_solution);
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
 * wave_euler_svc_cuda_time_loop object
 */
template <typename fp_type, template <typename, typename> typename container, typename allocator>
class wave_euler_svc_cuda_time_loop
{
    typedef container<fp_type, allocator> container_t;
    typedef container_2d<by_enum::Row, fp_type, container, allocator> container_2d_t;

  public:
    static void run(function_quintuple_t<fp_type> const &fun_quintuple, boundary_1d_pair<fp_type> const &boundary_pair,
                    range<fp_type> const &space_range, range<fp_type> const &time_range,
                    std::size_t const &last_time_idx, std::pair<fp_type, fp_type> const &steps,
                    traverse_direction_enum const &traverse_dir, container_t &prev_solution_0,
                    container_t &prev_solution_1, container_t &next_solution);

    static void run(function_quintuple_t<fp_type> const &fun_quintuple, boundary_1d_pair<fp_type> const &boundary_pair,
                    range<fp_type> const &space_range, range<fp_type> const &time_range,
                    std::size_t const &last_time_idx, std::pair<fp_type, fp_type> const &steps,
                    traverse_direction_enum const &traverse_dir, container_t &prev_solution_0,
                    container_t &prev_solution_1, container_t &next_solution,
                    std::function<fp_type(fp_type, fp_type)> const &wave_source, container_t &source);

    static void run_with_stepping(function_quintuple_t<fp_type> const &fun_quintuple,
                                  boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &space_range,
                                  range<fp_type> const &time_range, std::size_t const &last_time_idx,
                                  std::pair<fp_type, fp_type> const &steps, traverse_direction_enum const &traverse_dir,
                                  container_t &prev_solution_0, container_t &prev_solution_1,
                                  container_t &next_solution, container_2d_t &solutions);

    static void run_with_stepping(function_quintuple_t<fp_type> const &fun_quintuple,
                                  boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &space_range,
                                  range<fp_type> const &time_range, std::size_t const &last_time_idx,
                                  std::pair<fp_type, fp_type> const &steps, traverse_direction_enum const &traverse_dir,
                                  container_t &prev_solution_0, container_t &prev_solution_1,
                                  container_t &next_solution,
                                  std::function<fp_type(fp_type, fp_type)> const &wave_source, container_t &source,
                                  container_2d_t &solutions);
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void wave_euler_svc_cuda_time_loop<fp_type, container, allocator>::run(
    function_quintuple_t<fp_type> const &fun_quintuple, boundary_1d_pair<fp_type> const &boundary_pair,
    range<fp_type> const &space_range, range<fp_type> const &time_range, std::size_t const &last_time_idx,
    std::pair<fp_type, fp_type> const &steps, traverse_direction_enum const &traverse_dir, container_t &prev_solution_0,
    container_t &prev_solution_1, container_t &next_solution)
{
    typedef explicit_wave_svc_scheme<fp_type, container, allocator> explicit_scheme;
    typedef explicit_wave_svc_cuda_scheme<fp_type> explicit_cuda_scheme;

    const std::size_t sol_size = next_solution.size();
    // ranges and steps:
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type start_x = space_range.lower();
    const fp_type k = std::get<0>(steps);
    const fp_type h = std::get<1>(steps);
    // create a kernel:
    auto const &kernel =
        std::make_shared<wave_euler_svc_cuda_kernel<fp_type>>(fun_quintuple, steps, space_range, sol_size);
    // create host vectors:
    thrust::host_vector<fp_type> h_solution_0(sol_size);
    thrust::host_vector<fp_type> h_solution_1(sol_size);
    thrust::host_vector<fp_type> h_next_solution(sol_size);

    fp_type time{};
    std::size_t time_idx{};

    if (traverse_dir == traverse_direction_enum::Forward)
    {
        // solve for initial time step:
        auto init_scheme = explicit_scheme::get_initial(true);
        init_scheme(fun_quintuple, steps, prev_solution_0, prev_solution_1, container_t(), boundary_pair, start_time,
                    next_solution);
        time = start_time + k;
        time_idx = 1;
        prev_solution_1 = next_solution;

        // solve for rest of time steps:
        thrust::copy(prev_solution_0.begin(), prev_solution_0.end(), h_solution_0.begin());
        thrust::copy(prev_solution_1.begin(), prev_solution_1.end(), h_solution_1.begin());
        auto scheme = explicit_cuda_scheme::get(true);
        time_idx++;
        while (time_idx <= last_time_idx)
        {
            scheme(fun_quintuple, kernel, steps, h_solution_0, h_solution_1, thrust::host_vector<fp_type>(),
                   boundary_pair, time, h_next_solution);
            time += k;
            h_solution_0 = h_solution_1;
            h_solution_1 = h_next_solution;
            time_idx++;
        }
    }
    else if (traverse_dir == traverse_direction_enum::Backward)
    {
        time_idx = last_time_idx;
        // solve for initial time step:
        auto term_scheme = explicit_scheme::get_terminal(true);
        term_scheme(fun_quintuple, steps, prev_solution_0, prev_solution_1, container_t(), boundary_pair, end_time,
                    next_solution);
        time_idx--;
        time = end_time - time;
        prev_solution_1 = next_solution;

        // solve for rest of time steps:
        thrust::copy(prev_solution_0.begin(), prev_solution_0.end(), h_solution_0.begin());
        thrust::copy(prev_solution_1.begin(), prev_solution_1.end(), h_solution_1.begin());
        auto scheme = explicit_cuda_scheme::get(true);
        do
        {
            time_idx--;
            scheme(fun_quintuple, kernel, steps, h_solution_0, h_solution_1, thrust::host_vector<fp_type>(),
                   boundary_pair, time, h_next_solution);
            time -= k;
            h_solution_0 = h_solution_1;
            h_solution_1 = h_next_solution;
        } while (time_idx > 0);
    }
    else
    {
        throw std::exception("Unreachable");
    }
    thrust::copy(h_next_solution.begin(), h_next_solution.end(), next_solution.begin());
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void wave_euler_svc_cuda_time_loop<fp_type, container, allocator>::run(
    function_quintuple_t<fp_type> const &fun_quintuple, boundary_1d_pair<fp_type> const &boundary_pair,
    range<fp_type> const &space_range, range<fp_type> const &time_range, std::size_t const &last_time_idx,
    std::pair<fp_type, fp_type> const &steps, traverse_direction_enum const &traverse_dir, container_t &prev_solution_0,
    container_t &prev_solution_1, container_t &next_solution,
    std::function<fp_type(fp_type, fp_type)> const &wave_source, container_t &source)
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef discretization<dimension_enum::One, fp_type, thrust::host_vector, std::allocator<fp_type>> d_1dd;
    typedef explicit_wave_svc_scheme<fp_type, container, allocator> explicit_scheme;
    typedef explicit_wave_svc_cuda_scheme<fp_type> explicit_cuda_scheme;

    const std::size_t sol_size = next_solution.size();
    // ranges and steps:
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type start_x = space_range.lower();
    const fp_type k = std::get<0>(steps);
    const fp_type h = std::get<1>(steps);
    // create a kernel:
    auto const &kernel =
        std::make_shared<wave_euler_svc_cuda_kernel<fp_type>>(fun_quintuple, steps, space_range, sol_size);
    // create host vectors:
    thrust::host_vector<fp_type> h_solution_0(sol_size);
    thrust::host_vector<fp_type> h_solution_1(sol_size);
    thrust::host_vector<fp_type> h_source(sol_size);
    thrust::host_vector<fp_type> h_next_solution(sol_size);

    fp_type time{};
    std::size_t time_idx{};

    if (traverse_dir == traverse_direction_enum::Forward)
    {
        // solve for initial time step:
        auto init_scheme = explicit_scheme::get_initial(false);
        d_1d::of_function(start_x, h, start_time, wave_source, source);
        init_scheme(fun_quintuple, steps, prev_solution_0, prev_solution_1, source, boundary_pair, start_time,
                    next_solution);
        time = start_time + k;
        time_idx = 1;
        prev_solution_1 = next_solution;

        // solve for rest of time steps:
        thrust::copy(prev_solution_0.begin(), prev_solution_0.end(), h_solution_0.begin());
        thrust::copy(prev_solution_1.begin(), prev_solution_1.end(), h_solution_1.begin());
        auto scheme = explicit_cuda_scheme::get(false);
        d_1dd::of_function(start_x, h, time, wave_source, h_source);
        time_idx++;
        while (time_idx <= last_time_idx)
        {
            scheme(fun_quintuple, kernel, steps, h_solution_0, h_solution_1, h_source, boundary_pair, time,
                   h_next_solution);
            time += k;
            h_solution_0 = h_solution_1;
            h_solution_1 = h_next_solution;
            d_1dd::of_function(start_x, h, time, wave_source, h_source);
            time_idx++;
        }
    }
    else if (traverse_dir == traverse_direction_enum::Backward)
    {
        time_idx = last_time_idx;
        // solve for initial time step:
        auto term_scheme = explicit_scheme::get_terminal(false);
        d_1d::of_function(start_x, h, end_time, wave_source, source);
        term_scheme(fun_quintuple, steps, prev_solution_0, prev_solution_1, source, boundary_pair, end_time,
                    next_solution);
        time_idx--;
        time = end_time - time;
        prev_solution_1 = next_solution;

        // solve for rest of time steps:
        thrust::copy(prev_solution_0.begin(), prev_solution_0.end(), h_solution_0.begin());
        thrust::copy(prev_solution_1.begin(), prev_solution_1.end(), h_solution_1.begin());
        auto scheme = explicit_cuda_scheme::get(false);
        d_1dd::of_function(start_x, h, time, wave_source, h_source);
        do
        {
            time_idx--;
            scheme(fun_quintuple, kernel, steps, h_solution_0, h_solution_1, h_source, boundary_pair, time,
                   h_next_solution);
            time -= k;
            h_solution_0 = h_solution_1;
            h_solution_1 = h_next_solution;
            d_1dd::of_function(start_x, h, time, wave_source, h_source);
        } while (time_idx > 0);
    }
    else
    {
        throw std::exception("Unreachable");
    }
    thrust::copy(h_next_solution.begin(), h_next_solution.end(), next_solution.begin());
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void wave_euler_svc_cuda_time_loop<fp_type, container, allocator>::run_with_stepping(
    function_quintuple_t<fp_type> const &fun_quintuple, boundary_1d_pair<fp_type> const &boundary_pair,
    range<fp_type> const &space_range, range<fp_type> const &time_range, std::size_t const &last_time_idx,
    std::pair<fp_type, fp_type> const &steps, traverse_direction_enum const &traverse_dir, container_t &prev_solution_0,
    container_t &prev_solution_1, container_t &next_solution, container_2d_t &solutions)
{
    typedef explicit_wave_svc_scheme<fp_type, container, allocator> explicit_scheme;
    typedef explicit_wave_svc_cuda_scheme<fp_type> explicit_cuda_scheme;

    const std::size_t sol_size = next_solution.size();
    // ranges and steps:
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type start_x = space_range.lower();

    const fp_type k = std::get<0>(steps);
    const fp_type h = std::get<1>(steps);
    // create a kernel:
    auto const &kernel =
        std::make_shared<wave_euler_svc_cuda_kernel<fp_type>>(fun_quintuple, steps, space_range, sol_size);
    // create host vectors:
    thrust::host_vector<fp_type> h_solution_0(sol_size);
    thrust::host_vector<fp_type> h_solution_1(sol_size);
    thrust::host_vector<fp_type> h_next_solution(sol_size);

    fp_type time{};
    std::size_t time_idx{};

    if (traverse_dir == traverse_direction_enum::Forward)
    {
        // store the initial solution:
        solutions(0, prev_solution_0);
        // solve for initial time step:
        auto init_scheme = explicit_scheme::get_initial(true);
        init_scheme(fun_quintuple, steps, prev_solution_0, prev_solution_1, container_t(), boundary_pair, start_time,
                    next_solution);
        time = start_time + k;
        time_idx = 1;
        prev_solution_1 = next_solution;
        solutions(time_idx, next_solution);

        // solve for rest of time steps:
        thrust::copy(prev_solution_0.begin(), prev_solution_0.end(), h_solution_0.begin());
        thrust::copy(prev_solution_1.begin(), prev_solution_1.end(), h_solution_1.begin());
        auto scheme = explicit_cuda_scheme::get(true);
        time_idx++;
        while (time_idx <= last_time_idx)
        {
            scheme(fun_quintuple, kernel, steps, h_solution_0, h_solution_1, thrust::host_vector<fp_type>(),
                   boundary_pair, time, h_next_solution);
            time += k;
            h_solution_0 = h_solution_1;
            h_solution_1 = h_next_solution;
            thrust::copy(h_next_solution.begin(), h_next_solution.end(), next_solution.begin());
            solutions(time_idx, next_solution);
            time_idx++;
        }
    }
    else if (traverse_dir == traverse_direction_enum::Backward)
    {
        time_idx = last_time_idx;
        // store the terminal solution:
        solutions(last_time_idx, prev_solution_0);
        // solve for terminal time step:
        auto term_scheme = explicit_scheme::get_terminal(true);
        term_scheme(fun_quintuple, steps, prev_solution_0, prev_solution_1, container_t(), boundary_pair, end_time,
                    next_solution);
        time_idx--;
        time = end_time - time;
        prev_solution_1 = next_solution;
        solutions(time_idx, next_solution);

        // solve for rest of time steps:
        thrust::copy(prev_solution_0.begin(), prev_solution_0.end(), h_solution_0.begin());
        thrust::copy(prev_solution_1.begin(), prev_solution_1.end(), h_solution_1.begin());
        auto scheme = explicit_cuda_scheme::get(true);
        do
        {
            time_idx--;
            scheme(fun_quintuple, kernel, steps, h_solution_0, h_solution_1, thrust::host_vector<fp_type>(),
                   boundary_pair, time, h_next_solution);
            time -= k;
            h_solution_0 = h_solution_1;
            h_solution_1 = h_next_solution;
            thrust::copy(h_next_solution.begin(), h_next_solution.end(), next_solution.begin());
            solutions(time_idx, next_solution);
        } while (time_idx > 0);
    }
    else
    {
        throw std::exception("Unreachable");
    }
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void wave_euler_svc_cuda_time_loop<fp_type, container, allocator>::run_with_stepping(
    function_quintuple_t<fp_type> const &fun_quintuple, boundary_1d_pair<fp_type> const &boundary_pair,
    range<fp_type> const &space_range, range<fp_type> const &time_range, std::size_t const &last_time_idx,
    std::pair<fp_type, fp_type> const &steps, traverse_direction_enum const &traverse_dir, container_t &prev_solution_0,
    container_t &prev_solution_1, container_t &next_solution,
    std::function<fp_type(fp_type, fp_type)> const &wave_source, container_t &source, container_2d_t &solutions)
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef discretization<dimension_enum::One, fp_type, thrust::host_vector, std::allocator<fp_type>> d_1dd;
    typedef explicit_wave_svc_scheme<fp_type, container, allocator> explicit_scheme;
    typedef explicit_wave_svc_cuda_scheme<fp_type> explicit_cuda_scheme;

    const std::size_t sol_size = next_solution.size();
    // ranges and steps:
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type start_x = space_range.lower();
    const fp_type k = std::get<0>(steps);
    const fp_type h = std::get<1>(steps);
    // create a kernel:
    auto const &kernel =
        std::make_shared<wave_euler_svc_cuda_kernel<fp_type>>(fun_quintuple, steps, space_range, sol_size);
    // create host vectors:
    thrust::host_vector<fp_type> h_solution_0(sol_size);
    thrust::host_vector<fp_type> h_solution_1(sol_size);
    thrust::host_vector<fp_type> h_source(sol_size);
    thrust::host_vector<fp_type> h_next_solution(sol_size);

    fp_type time{};
    std::size_t time_idx{};

    if (traverse_dir == traverse_direction_enum::Forward)
    {
        // store the initial solution:
        solutions(0, prev_solution_0);
        // solve for initial time step:
        auto init_scheme = explicit_scheme::get_initial(false);
        d_1d::of_function(start_x, h, start_time, wave_source, source);
        init_scheme(fun_quintuple, steps, prev_solution_0, prev_solution_1, source, boundary_pair, start_time,
                    next_solution);
        time = start_time + k;
        time_idx = 1;
        prev_solution_1 = next_solution;
        solutions(time_idx, next_solution);

        // solve for rest of time steps:
        thrust::copy(prev_solution_0.begin(), prev_solution_0.end(), h_solution_0.begin());
        thrust::copy(prev_solution_1.begin(), prev_solution_1.end(), h_solution_1.begin());
        auto scheme = explicit_cuda_scheme::get(false);
        d_1dd::of_function(start_x, h, time, wave_source, h_source);
        time_idx++;
        while (time_idx <= last_time_idx)
        {
            scheme(fun_quintuple, kernel, steps, h_solution_0, h_solution_1, h_source, boundary_pair, time,
                   h_next_solution);
            time += k;
            h_solution_0 = h_solution_1;
            h_solution_1 = h_next_solution;
            thrust::copy(h_next_solution.begin(), h_next_solution.end(), next_solution.begin());
            solutions(time_idx, next_solution);
            d_1dd::of_function(start_x, h, time, wave_source, h_source);
            time_idx++;
        }
    }
    else if (traverse_dir == traverse_direction_enum::Backward)
    {
        time_idx = last_time_idx;
        // store the terminal solution:
        solutions(last_time_idx, prev_solution_0);
        // solve for terminal time step:
        auto term_scheme = explicit_scheme::get_terminal(false);
        d_1d::of_function(start_x, h, end_time, wave_source, source);
        term_scheme(fun_quintuple, steps, prev_solution_0, prev_solution_1, source, boundary_pair, end_time,
                    next_solution);
        time_idx--;
        time = end_time - time;
        prev_solution_1 = next_solution;
        solutions(time_idx, next_solution);

        // solve for rest of time steps:
        thrust::copy(prev_solution_0.begin(), prev_solution_0.end(), h_solution_0.begin());
        thrust::copy(prev_solution_1.begin(), prev_solution_1.end(), h_solution_1.begin());
        auto scheme = explicit_cuda_scheme::get(false);
        d_1dd::of_function(start_x, h, time, wave_source, h_source);
        do
        {
            time_idx--;
            scheme(fun_quintuple, kernel, steps, h_solution_0, h_solution_1, h_source, boundary_pair, time,
                   h_next_solution);
            time -= k;
            h_solution_0 = h_solution_1;
            h_solution_1 = h_next_solution;
            thrust::copy(h_next_solution.begin(), h_next_solution.end(), next_solution.begin());
            solutions(time_idx, next_solution);
            d_1dd::of_function(start_x, h, time, wave_source, h_source);
        } while (time_idx > 0);
    }
    else
    {
        throw std::exception("Unreachable");
    }
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class wave_euler_svc_cuda_scheme
{
    typedef wave_euler_svc_cuda_time_loop<fp_type, container, allocator> loop;
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container<fp_type, allocator> container_t;

  private:
    function_quintuple_t<fp_type> fun_quintuple_;
    boundary_1d_pair<fp_type> boundary_pair_;
    pde_discretization_config_1d_ptr<fp_type> discretization_cfg_;

    bool is_stable()
    {
        auto const &b = std::get<4>(fun_quintuple_);
        const fp_type k = discretization_cfg_->time_step();
        const fp_type h = discretization_cfg_->space_step();
        const fp_type ratio = h / k;
        const std::size_t space_size = discretization_cfg_->number_of_space_points();
        fp_type m{};
        for (std::size_t i = 0; i < space_size; ++i)
        {
            m = static_cast<fp_type>(i);
            if (b(m * h) >= ratio)
                return false;
        }
        return true;
    }

    void initialize()
    {
        LSS_ASSERT(is_stable() == true, "The chosen scheme is not stable");
    }

    explicit wave_euler_svc_cuda_scheme() = delete;

  public:
    wave_euler_svc_cuda_scheme(function_quintuple_t<fp_type> const &fun_quintuple,
                               boundary_1d_pair<fp_type> const &boundary_pair,
                               pde_discretization_config_1d_ptr<fp_type> const &discretization_config)
        : fun_quintuple_{fun_quintuple}, boundary_pair_{boundary_pair}, discretization_cfg_{discretization_config}
    {
        initialize();
    }

    ~wave_euler_svc_cuda_scheme()
    {
    }

    void operator()(container_t &prev_solution_0, container_t &prev_solution_1, container_t &next_solution,
                    bool is_wave_sourse_set, std::function<fp_type(fp_type, fp_type)> const &wave_source,
                    traverse_direction_enum traverse_dir)
    {
        const range<fp_type> spacer = discretization_cfg_->space_range();
        const range<fp_type> timer = discretization_cfg_->time_range();
        const fp_type k = discretization_cfg_->time_step();
        const fp_type h = discretization_cfg_->space_step();
        // save solution size:
        const std::size_t sol_size = next_solution.size();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        auto const &steps = std::make_pair(k, h);
        if (is_wave_sourse_set)
        {
            container_t source(sol_size, NaN<fp_type>());
            loop::run(fun_quintuple_, boundary_pair_, spacer, timer, last_time_idx, steps, traverse_dir,
                      prev_solution_0, prev_solution_1, next_solution, wave_source, source);
        }
        else
        {
            loop::run(fun_quintuple_, boundary_pair_, spacer, timer, last_time_idx, steps, traverse_dir,
                      prev_solution_0, prev_solution_1, next_solution);
        }
    }

    void operator()(container_t &prev_solution_0, container_t &prev_solution_1, container_t &next_solution,
                    bool is_wave_sourse_set, std::function<fp_type(fp_type, fp_type)> const &wave_source,
                    traverse_direction_enum traverse_dir,
                    container_2d<by_enum::Row, fp_type, container, allocator> &solutions)
    {
        const range<fp_type> spacer = discretization_cfg_->space_range();
        const range<fp_type> timer = discretization_cfg_->time_range();
        const fp_type k = discretization_cfg_->time_step();
        const fp_type h = discretization_cfg_->space_step();
        // save solution size:
        const std::size_t sol_size = next_solution.size();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        auto const &steps = std::make_pair(k, h);
        if (is_wave_sourse_set)
        {
            // create a container to carry discretized source heat
            container_t source(sol_size, NaN<fp_type>());
            loop::run_with_stepping(fun_quintuple_, boundary_pair_, spacer, timer, last_time_idx, steps, traverse_dir,
                                    prev_solution_0, prev_solution_1, next_solution, wave_source, source, solutions);
        }
        else
        {
            loop::run_with_stepping(fun_quintuple_, boundary_pair_, spacer, timer, last_time_idx, steps, traverse_dir,
                                    prev_solution_0, prev_solution_1, next_solution, solutions);
        }
    }
};

} // namespace one_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_WAVE_EULER_SVC_CUDA_SCHEME_HPP_
