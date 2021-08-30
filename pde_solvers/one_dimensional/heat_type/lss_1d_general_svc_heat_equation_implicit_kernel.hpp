#if !defined(_LSS_1D_GENERAL_SVC_HEAT_EQUATION_IMPLICIT_KERNEL_HPP_)
#define _LSS_1D_GENERAL_SVC_HEAT_EQUATION_IMPLICIT_KERNEL_HPP_

#include <vector>

#include "boundaries/lss_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include "containers/lss_container_2d.hpp"
#include "discretization/lss_discretization.hpp"
#include "pde_solvers/lss_heat_solver_config.hpp"
#include "pde_solvers/lss_pde_discretization_config.hpp"
#include "sparse_solvers/tridiagonal/cuda_solver/lss_cuda_solver.hpp"
#include "sparse_solvers/tridiagonal/double_sweep_solver/lss_double_sweep_solver.hpp"
#include "sparse_solvers/tridiagonal/sor_solver/lss_sor_solver.hpp"
#include "sparse_solvers/tridiagonal/sor_solver_cuda/lss_sor_solver_cuda.hpp"
#include "sparse_solvers/tridiagonal/thomas_lu_solver/lss_thomas_lu_solver.hpp"

namespace lss_pde_solvers
{
namespace one_dimensional
{

using lss_boundary::boundary_1d_pair;
using lss_boundary::neumann_boundary_1d;
using lss_boundary::robin_boundary_1d;
using lss_containers::container_2d;
using lss_cuda_solver::cuda_solver;
using lss_double_sweep_solver::double_sweep_solver;
using lss_enumerations::by_enum;
using lss_enumerations::dimension_enum;
using lss_enumerations::implicit_pde_schemes_enum;
using lss_enumerations::memory_space_enum;
using lss_enumerations::traverse_direction_enum;
using lss_enumerations::tridiagonal_method_enum;
using lss_sor_solver::sor_solver;
using lss_sor_solver_cuda::sor_solver_cuda;
using lss_thomas_lu_solver::thomas_lu_solver;
using lss_utility::diagonal_triplet_t;
using lss_utility::function_triplet_t;
using lss_utility::NaN;
using lss_utility::pair_t;
using lss_utility::range;

template <template <typename, typename> typename container, typename fp_type, typename alloc>
using implicit_heat_scheme_function_t =
    std::function<void(function_triplet_t<fp_type> const &, pair_t<fp_type> const &, container<fp_type, alloc> const &,
                       container<fp_type, alloc> const &, container<fp_type, alloc> const &,
                       boundary_1d_pair<fp_type> const &, fp_type const &, container<fp_type, alloc> &)>;

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class implicit_heat_scheme
{
    typedef container<fp_type, allocator> container_t;
    typedef implicit_heat_scheme_function_t<container, fp_type, allocator> scheme_function_t;

  private:
    static fp_type const get_scheme_theta(implicit_pde_schemes_enum scheme)
    {
        const fp_type one = static_cast<fp_type>(1.0);
        const fp_type half = static_cast<fp_type>(0.5);
        if (scheme == implicit_pde_schemes_enum::Euler)
            return one;
        else
            return half;
    }

  public:
    static scheme_function_t const get(implicit_pde_schemes_enum scheme, bool is_homogeneus)
    {
        const fp_type theta = get_scheme_theta(scheme);
        const fp_type two = static_cast<fp_type>(2.0);
        const fp_type one = static_cast<fp_type>(1.0);
        auto scheme_fun_h =
            [=](function_triplet_t<fp_type> const &coefficients, std::pair<fp_type, fp_type> const &steps,
                container_t const &input, container_t const &inhom_input, container_t const &inhom_input_next,
                boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, container_t &solution) {
                auto const &first_bnd = boundary_pair.first;
                auto const &second_bnd = boundary_pair.second;
                auto const &A = std::get<0>(coefficients);
                auto const &B = std::get<1>(coefficients);
                auto const &D = std::get<2>(coefficients);
                auto const h = steps.second;
                fp_type m{};

                // for lower boundaries first:
                if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(first_bnd))
                {
                    const fp_type beta = two * h * ptr->value(time);
                    m = static_cast<fp_type>(0);
                    solution[0] = (one - theta) * beta * A(m * h) + (one - two * (one - theta) * B(m * h)) * input[0] +
                                  (one - theta) * (A(m * h) + D(m * h)) * input[1];
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_bnd))
                {
                    const fp_type beta = two * h * ptr->value(time);
                    const fp_type alpha = two * h * ptr->linear_value(time);
                    m = static_cast<fp_type>(0);
                    solution[0] = (one - theta) * beta * A(m * h) +
                                  (one - (one - theta) * (two * B(m * h) - alpha * A(m * h))) * input[0] +
                                  (one - theta) * (A(m * h) + D(m * h)) * input[1];
                }
                // for upper boundaries second:
                const std::size_t N = solution.size() - 1;
                if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(second_bnd))
                {
                    const fp_type delta = two * h * ptr->value(time);
                    m = static_cast<fp_type>(N);
                    solution[N] = (one - theta) * (A(m * h) + D(m * h)) * input[N - 1] +
                                  (one - two * (one - theta) * B(m * h)) * input[N] - (one - theta) * delta * D(m * h);
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
                {
                    const fp_type delta = two * h * ptr->value(time);
                    const fp_type gamma = two * h * ptr->linear_value(time);
                    m = static_cast<fp_type>(N);
                    solution[N] = (one - theta) * (A(m * h) + D(m * h)) * input[N - 1] +
                                  (one - (one - theta) * (two * B(m * h) + gamma * D(m * h))) * input[N] -
                                  (one - theta) * delta * D(m * h);
                }

                for (std::size_t t = 1; t < N; ++t)
                {
                    m = static_cast<fp_type>(t);
                    solution[t] = (D(m * h) * (one - theta) * input[t + 1]) +
                                  ((one - two * B(m * h) * (one - theta)) * input[t]) +
                                  (A(m * h) * (one - theta) * input[t - 1]);
                }
            };
        auto scheme_fun_nh =
            [=](function_triplet_t<fp_type> const &coefficients, std::pair<fp_type, fp_type> const &steps,
                container_t const &input, container_t const &inhom_input, container_t const &inhom_input_next,
                boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, container_t &solution) {
                auto const &first_bnd = boundary_pair.first;
                auto const &second_bnd = boundary_pair.second;
                auto const &A = std::get<0>(coefficients);
                auto const &B = std::get<1>(coefficients);
                auto const &D = std::get<2>(coefficients);
                auto const k = steps.first;
                auto const h = steps.second;
                fp_type m{};

                // for lower boundaries first:
                if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(first_bnd))
                {
                    const fp_type beta = two * h * ptr->value(time);
                    m = static_cast<fp_type>(0);
                    solution[0] = (one - theta) * beta * A(m * h) + (one - two * (one - theta) * B(m * h)) * input[0] +
                                  (one - theta) * (A(m * h) + D(m * h)) * input[1] + theta * k * inhom_input_next[0] +
                                  (one - theta) * k * inhom_input[0];
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_bnd))
                {
                    const fp_type beta = two * h * ptr->value(time);
                    const fp_type alpha = two * h * ptr->linear_value(time);
                    m = static_cast<fp_type>(0);
                    solution[0] = (one - theta) * beta * A(m * h) +
                                  (one - (one - theta) * (two * B(m * h) - alpha * A(m * h))) * input[0] +
                                  (one - theta) * (A(m * h) + D(m * h)) * input[1] + theta * k * inhom_input_next[0] +
                                  (one - theta) * k * inhom_input[0];
                    ;
                }
                // for upper boundaries second:
                const std::size_t N = solution.size() - 1;
                if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(second_bnd))
                {
                    const fp_type delta = two * h * ptr->value(time);
                    m = static_cast<fp_type>(N);
                    solution[N] = (one - theta) * (A(m * h) + D(m * h)) * input[N - 1] +
                                  (one - two * (one - theta) * B(m * h)) * input[N] - (one - theta) * delta * D(m * h) +
                                  theta * k * inhom_input_next[N] + (one - theta) * k * inhom_input[N];
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
                {
                    const fp_type delta = two * h * ptr->value(time);
                    const fp_type gamma = two * h * ptr->linear_value(time);
                    m = static_cast<fp_type>(N);
                    solution[N] = (one - theta) * (A(m * h) + D(m * h)) * input[N - 1] +
                                  (one - (one - theta) * (two * B(m * h) + gamma * D(m * h))) * input[N] -
                                  (one - theta) * delta * D(m * h) + theta * k * inhom_input_next[N] +
                                  (one - theta) * k * inhom_input[N];
                }
                for (std::size_t t = 1; t < N; ++t)
                {
                    m = static_cast<fp_type>(t);
                    solution[t] = (D(m * h) * (one - theta) * input[t + 1]) +
                                  ((one - two * B(m * h) * (one - theta)) * input[t]) +
                                  (A(m * h) * (one - theta) * input[t - 1]) +
                                  k * (theta * inhom_input_next[t] + (one - theta) * inhom_input[t]);
                }
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
 * heat_time_loop object
 */
template <typename fp_type, template <typename, typename> typename container = std::vector,
          typename allocator = std::allocator<fp_type>>
class heat_time_loop
{
    typedef container<fp_type, allocator> container_t;

  public:
    template <typename solver_object, typename scheme_function>
    static void run(solver_object &solver_obj, scheme_function &scheme_fun,
                    boundary_1d_pair<fp_type> const &boundary_pair, function_triplet_t<fp_type> const &fun_triplet,
                    range<fp_type> const &space_range, range<fp_type> const &time_range,
                    std::size_t const &last_time_idx, std::pair<fp_type, fp_type> const &steps,
                    traverse_direction_enum const &traverse_dir, container_t &prev_solution, container_t &next_solution,
                    container_t &rhs, std::function<fp_type(fp_type, fp_type)> const &heat_source,
                    container_t &curr_source, container_t &next_source);
    template <typename solver_object, typename scheme_function>
    static void run(solver_object &solver_obj, scheme_function &scheme_fun,
                    boundary_1d_pair<fp_type> const &boundary_pair, function_triplet_t<fp_type> const &fun_triplet,
                    range<fp_type> const &space_range, range<fp_type> const &time_range,
                    std::size_t const &last_time_idx, std::pair<fp_type, fp_type> const &steps,
                    traverse_direction_enum const &traverse_dir, container_t &prev_solution, container_t &next_solution,
                    container_t &rhs);

    template <typename solver_object, typename scheme_function>
    static void run_with_stepping(solver_object &solver_obj, scheme_function &scheme_fun,
                                  boundary_1d_pair<fp_type> const &boundary_pair,
                                  function_triplet_t<fp_type> const &fun_triplet, range<fp_type> const &space_range,
                                  range<fp_type> const &time_range, std::size_t const &last_time_idx,
                                  std::pair<fp_type, fp_type> const &steps, traverse_direction_enum const &traverse_dir,
                                  container_t &prev_solution, container_t &next_solution, container_t &rhs,
                                  std::function<fp_type(fp_type, fp_type)> const &heat_source, container_t &curr_source,
                                  container_t &next_source,
                                  container_2d<by_enum::Row, fp_type, container, allocator> &solutions);
    template <typename solver_object, typename scheme_function>
    static void run_with_stepping(solver_object &solver_obj, scheme_function &scheme_fun,
                                  boundary_1d_pair<fp_type> const &boundary_pair,
                                  function_triplet_t<fp_type> const &fun_triplet, range<fp_type> const &space_range,
                                  range<fp_type> const &time_range, std::size_t const &last_time_idx,
                                  std::pair<fp_type, fp_type> const &steps, traverse_direction_enum const &traverse_dir,
                                  container_t &prev_solution, container_t &next_solution, container_t &rhs,
                                  container_2d<by_enum::Row, fp_type, container, allocator> &solutions);
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
template <typename solver_object, typename scheme_function>
void heat_time_loop<fp_type, container, allocator>::run(
    solver_object &solver_obj, scheme_function &scheme_fun, boundary_1d_pair<fp_type> const &boundary_pair,
    function_triplet_t<fp_type> const &fun_triplet, range<fp_type> const &space_range, range<fp_type> const &time_range,
    std::size_t const &last_time_idx, std::pair<fp_type, fp_type> const &steps,
    traverse_direction_enum const &traverse_dir, container_t &prev_solution, container_t &next_solution,
    container_t &rhs, std::function<fp_type(fp_type, fp_type)> const &heat_source, container_t &curr_source,
    container_t &next_source)
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type start_x = space_range.lower();
    const fp_type step_x = std::get<1>(steps);
    const fp_type k = std::get<0>(steps);
    fp_type time{};
    std::size_t time_idx{};
    if (traverse_dir == traverse_direction_enum::Forward)
    {
        d_1d::of_function(start_x, step_x, start_time, heat_source, curr_source);
        d_1d::of_function(start_x, step_x, start_time + k, heat_source, next_source);
        time = start_time + k;
        time_idx = 1;
        while (time_idx <= last_time_idx)
        {
            scheme_fun(fun_triplet, steps, prev_solution, curr_source, next_source, boundary_pair, time, rhs);
            solver_obj->set_rhs(rhs);
            solver_obj->solve(boundary_pair, next_solution, time);
            prev_solution = next_solution;
            d_1d::of_function(start_x, step_x, time, heat_source, curr_source);
            d_1d::of_function(start_x, step_x, time + k, heat_source, next_source);
            time += k;
            time_idx++;
        }
    }
    else
    {
        d_1d::of_function(start_x, step_x, end_time, heat_source, curr_source);
        d_1d::of_function(start_x, step_x, end_time - k, heat_source, next_source);
        time = end_time - time;
        time_idx = last_time_idx;
        do
        {
            time_idx--;
            scheme_fun(fun_triplet, steps, prev_solution, curr_source, next_source, boundary_pair, time, rhs);
            solver_obj->set_rhs(rhs);
            solver_obj->solve(boundary_pair, next_solution, time);
            prev_solution = next_solution;
            d_1d::of_function(start_x, step_x, time, heat_source, curr_source);
            d_1d::of_function(start_x, step_x, time - k, heat_source, next_source);
            time -= k;
        } while (time_idx > 0);
    }
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
template <typename solver_object, typename scheme_function>
void heat_time_loop<fp_type, container, allocator>::run(
    solver_object &solver_obj, scheme_function &scheme_fun, boundary_1d_pair<fp_type> const &boundary_pair,
    function_triplet_t<fp_type> const &fun_triplet, range<fp_type> const &space_range, range<fp_type> const &time_range,
    std::size_t const &last_time_idx, std::pair<fp_type, fp_type> const &steps,
    traverse_direction_enum const &traverse_dir, container_t &prev_solution, container_t &next_solution,
    container_t &rhs)
{
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type k = std::get<0>(steps);
    fp_type time{};
    std::size_t time_idx{};
    if (traverse_dir == traverse_direction_enum::Forward)
    {
        time = start_time + k;
        time_idx = 1;
        while (time_idx <= last_time_idx)
        {
            scheme_fun(fun_triplet, steps, prev_solution, container_t(), container_t(), boundary_pair, time, rhs);
            solver_obj->set_rhs(rhs);
            solver_obj->solve(boundary_pair, next_solution, time);
            prev_solution = next_solution;
            time += k;
            time_idx++;
        }
    }
    else
    {
        time = end_time - k;
        time_idx = last_time_idx;
        do
        {
            time_idx--;
            scheme_fun(fun_triplet, steps, prev_solution, container_t(), container_t(), boundary_pair, time, rhs);
            solver_obj->set_rhs(rhs);
            solver_obj->solve(boundary_pair, next_solution, time);
            prev_solution = next_solution;
            time -= k;
        } while (time_idx > 0);
    }
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
template <typename solver_object, typename scheme_function>
void heat_time_loop<fp_type, container, allocator>::run_with_stepping(
    solver_object &solver_obj, scheme_function &scheme_fun, boundary_1d_pair<fp_type> const &boundary_pair,
    function_triplet_t<fp_type> const &fun_triplet, range<fp_type> const &space_range, range<fp_type> const &time_range,
    std::size_t const &last_time_idx, std::pair<fp_type, fp_type> const &steps,
    traverse_direction_enum const &traverse_dir, container_t &prev_solution, container_t &next_solution,
    container_t &rhs, std::function<fp_type(fp_type, fp_type)> const &heat_source, container_t &curr_source,
    container_t &next_source, container_2d<by_enum::Row, fp_type, container, allocator> &solutions)
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type start_x = space_range.lower();
    const fp_type step_x = std::get<1>(steps);
    const fp_type k = std::get<0>(steps);
    fp_type time{};
    std::size_t time_idx{};
    if (traverse_dir == traverse_direction_enum::Forward)
    {
        // store the initial solution:
        solutions(0, prev_solution);
        d_1d::of_function(start_x, step_x, start_time, heat_source, curr_source);
        d_1d::of_function(start_x, step_x, start_time + k, heat_source, next_source);
        time = start_time + k;
        time_idx = 1;
        while (time_idx <= last_time_idx)
        {
            scheme_fun(fun_triplet, steps, prev_solution, curr_source, next_source, boundary_pair, time, rhs);
            solver_obj->set_rhs(rhs);
            solver_obj->solve(boundary_pair, next_solution, time);
            solutions(time_idx, next_solution);
            prev_solution = next_solution;
            d_1d::of_function(start_x, step_x, time, heat_source, curr_source);
            d_1d::of_function(start_x, step_x, time + k, heat_source, next_source);
            time += k;
            time_idx++;
        }
    }
    else
    {
        // store the initial solution:
        solutions(last_time_idx, prev_solution);
        d_1d::of_function(start_x, step_x, end_time, heat_source, curr_source);
        d_1d::of_function(start_x, step_x, end_time - k, heat_source, next_source);
        time = end_time - k;
        time_idx = last_time_idx;
        do
        {
            time_idx--;
            scheme_fun(fun_triplet, steps, prev_solution, curr_source, next_source, boundary_pair, time, rhs);
            solver_obj->set_rhs(rhs);
            solver_obj->solve(boundary_pair, next_solution, time);
            solutions(time_idx, next_solution);
            prev_solution = next_solution;
            d_1d::of_function(start_x, step_x, time, heat_source, curr_source);
            d_1d::of_function(start_x, step_x, time - k, heat_source, next_source);
            time -= k;
        } while (time_idx > 0);
    }
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
template <typename solver_object, typename scheme_function>
void heat_time_loop<fp_type, container, allocator>::run_with_stepping(
    solver_object &solver_obj, scheme_function &scheme_fun, boundary_1d_pair<fp_type> const &boundary_pair,
    function_triplet_t<fp_type> const &fun_triplet, range<fp_type> const &space_range, range<fp_type> const &time_range,
    std::size_t const &last_time_idx, std::pair<fp_type, fp_type> const &steps,
    traverse_direction_enum const &traverse_dir, container_t &prev_solution, container_t &next_solution,
    container_t &rhs, container_2d<by_enum::Row, fp_type, container, allocator> &solutions)
{
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type k = std::get<0>(steps);
    fp_type time{};
    std::size_t time_idx{};
    if (traverse_dir == traverse_direction_enum::Forward)
    {
        // store the initial solution:
        solutions(0, prev_solution);
        time = start_time + k;
        time_idx = 1;
        while (time_idx <= last_time_idx)
        {
            scheme_fun(fun_triplet, steps, prev_solution, container_t(), container_t(), boundary_pair, time, rhs);
            solver_obj->set_rhs(rhs);
            solver_obj->solve(boundary_pair, next_solution, time);
            solutions(time_idx, next_solution);
            prev_solution = next_solution;
            time += k;
            time_idx++;
        }
    }
    else
    {
        // store the initial solution:
        solutions(last_time_idx, prev_solution);
        time = end_time - k;
        time_idx = last_time_idx;
        do
        {
            time_idx--;
            scheme_fun(fun_triplet, steps, prev_solution, container_t(), container_t(), boundary_pair, time, rhs);
            solver_obj->set_rhs(rhs);
            solver_obj->solve(boundary_pair, next_solution, time);
            solutions(time_idx, next_solution);
            prev_solution = next_solution;
            time -= k;
        } while (time_idx > 0);
    }
}

template <memory_space_enum memory_enum, tridiagonal_method_enum tridiagonal_method, typename fp_type,
          template <typename, typename> typename container = std::vector, typename allocator = std::allocator<fp_type>>
class general_svc_heat_equation_implicit_kernel
{
};

// ===================================================================
// ============================== DEVICE =============================
// ===================================================================
template <typename fp_type, template <typename, typename> typename container, typename allocator>
class general_svc_heat_equation_implicit_kernel<memory_space_enum::Device, tridiagonal_method_enum::CUDASolver, fp_type,
                                                container, allocator>
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container<fp_type, allocator> container_t;
    typedef cuda_solver<memory_space_enum::Device, fp_type, container, allocator> cusolver;
    typedef heat_time_loop<fp_type, container, allocator> loop;

  private:
    diagonal_triplet_t<fp_type, container, allocator> diagonals_;
    function_triplet_t<fp_type> fun_triplet_;
    boundary_1d_pair<fp_type> boundary_pair_;
    pde_discretization_config_1d_ptr<fp_type> discretization_cfg_;
    heat_implicit_solver_config_ptr solver_cfg_;

  public:
    general_svc_heat_equation_implicit_kernel(diagonal_triplet_t<fp_type, container, allocator> const &diagonals,
                                              function_triplet_t<fp_type> const &fun_triplet,
                                              boundary_1d_pair<fp_type> const &boundary_pair,
                                              pde_discretization_config_1d_ptr<fp_type> const &discretization_config,
                                              heat_implicit_solver_config_ptr const &solver_config)
        : diagonals_{diagonals}, fun_triplet_{fun_triplet}, boundary_pair_{boundary_pair},
          discretization_cfg_{discretization_config}, solver_cfg_{solver_config}
    {
    }

    void operator()(container_t &prev_solution, container_t &next_solution, container_t &rhs, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type)> const &heat_source)
    {
        // get space range:
        const range<fp_type> space = discretization_cfg_->space_range();
        // get time range:
        const range<fp_type> time = discretization_cfg_->time_range();
        // get space step:
        const fp_type h = discretization_cfg_->space_step();
        // time step:
        const fp_type k = discretization_cfg_->time_step();
        // wrap up steps into pair:
        const std::pair<fp_type, fp_type> steps = std::make_pair(k, h);
        // size of space discretization:
        const std::size_t space_size = discretization_cfg_->number_of_space_points();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        // save traverse_direction
        const traverse_direction_enum traverse_dir = solver_cfg_->traverse_direction();
        // create and set up the solver:
        auto const &solver = std::make_shared<cusolver>(space, space_size);
        solver->set_diagonals(std::move(std::get<0>(diagonals_)), std::move(std::get<1>(diagonals_)),
                              std::move(std::get<2>(diagonals_)));
        solver->set_factorization(solver_cfg_->tridiagonal_factorization());
        // solver->set_boundary(std::get<0>(boundary_pair_), std::get<1>(boundary_pair_));
        if (is_heat_sourse_set)
        {
            auto scheme_function =
                implicit_heat_scheme<fp_type, container, allocator>::get(solver_cfg_->implicit_pde_scheme(), false);
            // create a container to carry discretized source heat
            container_t source_curr(space_size, NaN<fp_type>());
            container_t source_next(space_size, NaN<fp_type>());
            loop::run(solver, scheme_function, boundary_pair_, fun_triplet_, space, time, last_time_idx, steps,
                      traverse_dir, prev_solution, next_solution, rhs, heat_source, source_curr, source_next);
        }
        else
        {
            auto scheme_function =
                implicit_heat_scheme<fp_type, container, allocator>::get(solver_cfg_->implicit_pde_scheme(), true);
            loop::run(solver, scheme_function, boundary_pair_, fun_triplet_, space, time, last_time_idx, steps,
                      traverse_dir, prev_solution, next_solution, rhs);
        }
    }

    void operator()(container_t &prev_solution, container_t &next_solution, container_t &rhs, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type)> const &heat_source,
                    container_2d<by_enum::Row, fp_type, container, allocator> &solutions)
    {
        // get space range:
        const range<fp_type> space = discretization_cfg_->space_range();
        // get time range:
        const range<fp_type> time = discretization_cfg_->time_range();
        // get space step:
        const fp_type h = discretization_cfg_->space_step();
        // time step:
        const fp_type k = discretization_cfg_->time_step();
        // wrap up steps into pair:
        const std::pair<fp_type, fp_type> steps = std::make_pair(k, h);
        // size of space discretization:
        const std::size_t space_size = discretization_cfg_->number_of_space_points();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        // save traverse_direction
        const traverse_direction_enum traverse_dir = solver_cfg_->traverse_direction();
        // create and set up the solver:
        auto const &solver = std::make_shared<cusolver>(space, space_size);
        solver->set_diagonals(std::move(std::get<0>(diagonals_)), std::move(std::get<1>(diagonals_)),
                              std::move(std::get<2>(diagonals_)));
        solver->set_factorization(solver_cfg_->tridiagonal_factorization());
        // solver->set_boundary(std::get<0>(boundary_pair_), std::get<1>(boundary_pair_));
        if (is_heat_sourse_set)
        {
            auto scheme_function =
                implicit_heat_scheme<fp_type, container, allocator>::get(solver_cfg_->implicit_pde_scheme(), false);
            // create a container to carry discretized source heat
            container_t source_curr(space_size, NaN<fp_type>());
            container_t source_next(space_size, NaN<fp_type>());
            loop::run_with_stepping(solver, scheme_function, boundary_pair_, fun_triplet_, space, time, last_time_idx,
                                    steps, traverse_dir, prev_solution, next_solution, rhs, heat_source, source_curr,
                                    source_next, solutions);
        }
        else
        {
            auto scheme_function =
                implicit_heat_scheme<fp_type, container, allocator>::get(solver_cfg_->implicit_pde_scheme(), true);
            loop::run_with_stepping(solver, scheme_function, boundary_pair_, fun_triplet_, space, time, last_time_idx,
                                    steps, traverse_dir, prev_solution, next_solution, rhs, solutions);
        }
    }
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class general_svc_heat_equation_implicit_kernel<memory_space_enum::Device, tridiagonal_method_enum::SORSolver, fp_type,
                                                container, allocator>
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container<fp_type, allocator> container_t;
    typedef sor_solver_cuda<fp_type, container, allocator> sorcusolver;
    typedef heat_time_loop<fp_type, container, allocator> loop;

  private:
    diagonal_triplet_t<fp_type, container, allocator> diagonals_;
    function_triplet_t<fp_type> fun_triplet_;
    boundary_1d_pair<fp_type> boundary_pair_;
    pde_discretization_config_1d_ptr<fp_type> discretization_cfg_;
    heat_implicit_solver_config_ptr solver_cfg_;

  public:
    general_svc_heat_equation_implicit_kernel(diagonal_triplet_t<fp_type, container, allocator> const &diagonals,
                                              function_triplet_t<fp_type> const &fun_triplet,
                                              boundary_1d_pair<fp_type> const &boundary_pair,
                                              pde_discretization_config_1d_ptr<fp_type> const &discretization_config,
                                              heat_implicit_solver_config_ptr const &solver_config)
        : diagonals_{diagonals}, fun_triplet_{fun_triplet}, boundary_pair_{boundary_pair},
          discretization_cfg_{discretization_config}, solver_cfg_{solver_config}
    {
    }

    void operator()(container_t &prev_solution, container_t &next_solution, container_t &rhs, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type)> const &heat_source, fp_type omega_value)
    {
        // get space range:
        const range<fp_type> space = discretization_cfg_->space_range();
        // get time range:
        const range<fp_type> time = discretization_cfg_->time_range();
        // get space step:
        const fp_type h = discretization_cfg_->space_step();
        // time step:
        const fp_type k = discretization_cfg_->time_step();
        // wrap up steps into pair:
        const std::pair<fp_type, fp_type> steps = std::make_pair(k, h);
        // size of space discretization:
        const std::size_t space_size = discretization_cfg_->number_of_space_points();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        // save traverse_direction
        const traverse_direction_enum traverse_dir = solver_cfg_->traverse_direction();
        // create and set up the solver:
        auto const &solver = std::make_shared<sorcusolver>(space, space_size);
        solver->set_diagonals(std::move(std::get<0>(diagonals_)), std::move(std::get<1>(diagonals_)),
                              std::move(std::get<2>(diagonals_)));
        solver->set_omega(omega_value);
        // solver->set_boundary(std::get<0>(boundary_pair_), std::get<1>(boundary_pair_));
        if (is_heat_sourse_set)
        {
            auto scheme_function =
                implicit_heat_scheme<fp_type, container, allocator>::get(solver_cfg_->implicit_pde_scheme(), false);
            // create a container to carry discretized source heat
            container_t source_curr(space_size, NaN<fp_type>());
            container_t source_next(space_size, NaN<fp_type>());
            loop::run(solver, scheme_function, boundary_pair_, fun_triplet_, space, time, last_time_idx, steps,
                      traverse_dir, prev_solution, next_solution, rhs, heat_source, source_curr, source_next);
        }
        else
        {
            auto scheme_function =
                implicit_heat_scheme<fp_type, container, allocator>::get(solver_cfg_->implicit_pde_scheme(), true);
            loop::run(solver, scheme_function, boundary_pair_, fun_triplet_, space, time, last_time_idx, steps,
                      traverse_dir, prev_solution, next_solution, rhs);
        }
    }

    void operator()(container_t &prev_solution, container_t &next_solution, container_t &rhs, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type)> const &heat_source, fp_type omega_value,
                    container_2d<by_enum::Row, fp_type, container, allocator> &solutions)
    {
        // get space range:
        const range<fp_type> space = discretization_cfg_->space_range();
        // get time range:
        const range<fp_type> time = discretization_cfg_->time_range();
        // get space step:
        const fp_type h = discretization_cfg_->space_step();
        // time step:
        const fp_type k = discretization_cfg_->time_step();
        // wrap up steps into pair:
        const std::pair<fp_type, fp_type> steps = std::make_pair(k, h);
        // size of space discretization:
        const std::size_t space_size = discretization_cfg_->number_of_space_points();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        // save traverse_direction
        const traverse_direction_enum traverse_dir = solver_cfg_->traverse_direction();
        // create and set up the solver:
        auto const &solver = std::make_shared<sorcusolver>(space, space_size);
        solver->set_diagonals(std::move(std::get<0>(diagonals_)), std::move(std::get<1>(diagonals_)),
                              std::move(std::get<2>(diagonals_)));
        solver->set_omega(omega_value);
        // solver->set_boundary(std::get<0>(boundary_pair_), std::get<1>(boundary_pair_));
        if (is_heat_sourse_set)
        {
            auto scheme_function =
                implicit_heat_scheme<fp_type, container, allocator>::get(solver_cfg_->implicit_pde_scheme(), false);
            // create a container to carry discretized source heat
            container_t source_curr(space_size, NaN<fp_type>());
            container_t source_next(space_size, NaN<fp_type>());
            loop::run_with_stepping(solver, scheme_function, boundary_pair_, fun_triplet_, space, time, last_time_idx,
                                    steps, traverse_dir, prev_solution, next_solution, rhs, heat_source, source_curr,
                                    source_next, solutions);
        }
        else
        {
            auto scheme_function =
                implicit_heat_scheme<fp_type, container, allocator>::get(solver_cfg_->implicit_pde_scheme(), true);
            loop::run_with_stepping(solver, scheme_function, boundary_pair_, fun_triplet_, space, time, last_time_idx,
                                    steps, traverse_dir, prev_solution, next_solution, rhs, solutions);
        }
    }
};

// ===================================================================
// ================================ HOST =============================
// ===================================================================
template <typename fp_type, template <typename, typename> typename container, typename allocator>
class general_svc_heat_equation_implicit_kernel<memory_space_enum::Host, tridiagonal_method_enum::CUDASolver, fp_type,
                                                container, allocator>
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container<fp_type, allocator> container_t;
    typedef cuda_solver<memory_space_enum::Host, fp_type, container, allocator> cusolver;
    typedef heat_time_loop<fp_type, container, allocator> loop;

  private:
    diagonal_triplet_t<fp_type, container, allocator> diagonals_;
    function_triplet_t<fp_type> fun_triplet_;
    boundary_1d_pair<fp_type> boundary_pair_;
    pde_discretization_config_1d_ptr<fp_type> discretization_cfg_;
    heat_implicit_solver_config_ptr solver_cfg_;

  public:
    general_svc_heat_equation_implicit_kernel(diagonal_triplet_t<fp_type, container, allocator> const &diagonals,
                                              function_triplet_t<fp_type> const &fun_triplet,
                                              boundary_1d_pair<fp_type> const &boundary_pair,
                                              pde_discretization_config_1d_ptr<fp_type> const &discretization_config,
                                              heat_implicit_solver_config_ptr const &solver_config)
        : diagonals_{diagonals}, fun_triplet_{fun_triplet}, boundary_pair_{boundary_pair},
          discretization_cfg_{discretization_config}, solver_cfg_{solver_config}
    {
    }

    void operator()(container_t &prev_solution, container_t &next_solution, container_t &rhs, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type)> const &heat_source)
    {
        // get space range:
        const range<fp_type> space = discretization_cfg_->space_range();
        // get time range:
        const range<fp_type> time = discretization_cfg_->time_range();
        // get space step:
        const fp_type h = discretization_cfg_->space_step();
        // time step:
        const fp_type k = discretization_cfg_->time_step();
        // wrap up steps into pair:
        const std::pair<fp_type, fp_type> steps = std::make_pair(k, h);
        // size of space discretization:
        const std::size_t space_size = discretization_cfg_->number_of_space_points();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        // save traverse_direction
        const traverse_direction_enum traverse_dir = solver_cfg_->traverse_direction();
        // create and set up the solver:
        auto const &solver = std::make_shared<cusolver>(space, space_size);
        solver->set_diagonals(std::move(std::get<0>(diagonals_)), std::move(std::get<1>(diagonals_)),
                              std::move(std::get<2>(diagonals_)));
        solver->set_factorization(solver_cfg_->tridiagonal_factorization());
        // solver->set_boundary(std::get<0>(boundary_pair_), std::get<1>(boundary_pair_));
        if (is_heat_sourse_set)
        {
            auto scheme_function =
                implicit_heat_scheme<fp_type, container, allocator>::get(solver_cfg_->implicit_pde_scheme(), false);
            // create a container to carry discretized source heat
            container_t source_curr(space_size, NaN<fp_type>());
            container_t source_next(space_size, NaN<fp_type>());
            loop::run(solver, scheme_function, boundary_pair_, fun_triplet_, space, time, last_time_idx, steps,
                      traverse_dir, prev_solution, next_solution, rhs, heat_source, source_curr, source_next);
        }
        else
        {
            auto scheme_function =
                implicit_heat_scheme<fp_type, container, allocator>::get(solver_cfg_->implicit_pde_scheme(), true);
            loop::run(solver, scheme_function, boundary_pair_, fun_triplet_, space, time, last_time_idx, steps,
                      traverse_dir, prev_solution, next_solution, rhs);
        }
    }

    void operator()(container_t &prev_solution, container_t &next_solution, container_t &rhs, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type)> const &heat_source,
                    container_2d<by_enum::Row, fp_type, container, allocator> &solutions)
    {
        // get space range:
        const range<fp_type> space = discretization_cfg_->space_range();
        // get time range:
        const range<fp_type> time = discretization_cfg_->time_range();
        // get space step:
        const fp_type h = discretization_cfg_->space_step();
        // time step:
        const fp_type k = discretization_cfg_->time_step();
        // wrap up steps into pair:
        const std::pair<fp_type, fp_type> steps = std::make_pair(k, h);
        // size of space discretization:
        const std::size_t space_size = discretization_cfg_->number_of_space_points();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        // save traverse_direction
        const traverse_direction_enum traverse_dir = solver_cfg_->traverse_direction();
        // create and set up the solver:
        auto const &solver = std::make_shared<cusolver>(space, space_size);
        solver->set_diagonals(std::move(std::get<0>(diagonals_)), std::move(std::get<1>(diagonals_)),
                              std::move(std::get<2>(diagonals_)));
        solver->set_factorization(solver_cfg_->tridiagonal_factorization());
        // solver->set_boundary(std::get<0>(boundary_pair_), std::get<1>(boundary_pair_));
        if (is_heat_sourse_set)
        {
            auto scheme_function =
                implicit_heat_scheme<fp_type, container, allocator>::get(solver_cfg_->implicit_pde_scheme(), false);
            // create a container to carry discretized source heat
            container_t source_curr(space_size, NaN<fp_type>());
            container_t source_next(space_size, NaN<fp_type>());
            loop::run_with_stepping(solver, scheme_function, boundary_pair_, fun_triplet_, space, time, last_time_idx,
                                    steps, traverse_dir, prev_solution, next_solution, rhs, heat_source, source_curr,
                                    source_next, solutions);
        }
        else
        {
            auto scheme_function =
                implicit_heat_scheme<fp_type, container, allocator>::get(solver_cfg_->implicit_pde_scheme(), true);
            loop::run_with_stepping(solver, scheme_function, boundary_pair_, fun_triplet_, space, time, last_time_idx,
                                    steps, traverse_dir, prev_solution, next_solution, rhs, solutions);
        }
    }
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class general_svc_heat_equation_implicit_kernel<memory_space_enum::Host, tridiagonal_method_enum::SORSolver, fp_type,
                                                container, allocator>
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container<fp_type, allocator> container_t;
    typedef sor_solver<fp_type, container, allocator> sorsolver;
    typedef heat_time_loop<fp_type, container, allocator> loop;

  private:
    diagonal_triplet_t<fp_type, container, allocator> diagonals_;
    function_triplet_t<fp_type> fun_triplet_;
    boundary_1d_pair<fp_type> boundary_pair_;
    pde_discretization_config_1d_ptr<fp_type> discretization_cfg_;
    heat_implicit_solver_config_ptr solver_cfg_;

  public:
    general_svc_heat_equation_implicit_kernel(diagonal_triplet_t<fp_type, container, allocator> const &diagonals,
                                              function_triplet_t<fp_type> const &fun_triplet,
                                              boundary_1d_pair<fp_type> const &boundary_pair,
                                              pde_discretization_config_1d_ptr<fp_type> const &discretization_config,
                                              heat_implicit_solver_config_ptr const &solver_config)
        : diagonals_{diagonals}, fun_triplet_{fun_triplet}, boundary_pair_{boundary_pair},
          discretization_cfg_{discretization_config}, solver_cfg_{solver_config}
    {
    }

    void operator()(container_t &prev_solution, container_t &next_solution, container_t &rhs, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type)> const &heat_source, fp_type omega_value)
    {
        // get space range:
        const range<fp_type> space = discretization_cfg_->space_range();
        // get time range:
        const range<fp_type> time = discretization_cfg_->time_range();
        // get space step:
        const fp_type h = discretization_cfg_->space_step();
        // time step:
        const fp_type k = discretization_cfg_->time_step();
        // wrap up steps into pair:
        const std::pair<fp_type, fp_type> steps = std::make_pair(k, h);
        // size of space discretization:
        const std::size_t space_size = discretization_cfg_->number_of_space_points();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        // save traverse_direction
        const traverse_direction_enum traverse_dir = solver_cfg_->traverse_direction();
        // create and set up the solver:
        auto const &solver = std::make_shared<sorsolver>(space, space_size);
        solver->set_diagonals(std::move(std::get<0>(diagonals_)), std::move(std::get<1>(diagonals_)),
                              std::move(std::get<2>(diagonals_)));
        solver->set_omega(omega_value);
        // solver->set_boundary(std::get<0>(boundary_pair_), std::get<1>(boundary_pair_));
        if (is_heat_sourse_set)
        {
            auto scheme_function =
                implicit_heat_scheme<fp_type, container, allocator>::get(solver_cfg_->implicit_pde_scheme(), false);
            // create a container to carry discretized source heat
            container_t source_curr(space_size, NaN<fp_type>());
            container_t source_next(space_size, NaN<fp_type>());
            loop::run(solver, scheme_function, boundary_pair_, fun_triplet_, space, time, last_time_idx, steps,
                      traverse_dir, prev_solution, next_solution, rhs, heat_source, source_curr, source_next);
        }
        else
        {
            auto scheme_function =
                implicit_heat_scheme<fp_type, container, allocator>::get(solver_cfg_->implicit_pde_scheme(), true);
            loop::run(solver, scheme_function, boundary_pair_, fun_triplet_, space, time, last_time_idx, steps,
                      traverse_dir, prev_solution, next_solution, rhs);
        }
    }

    void operator()(container_t &prev_solution, container_t &next_solution, container_t &rhs, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type)> const &heat_source, fp_type omega_value,
                    container_2d<by_enum::Row, fp_type, container, allocator> &solutions)
    {
        // get space range:
        const range<fp_type> space = discretization_cfg_->space_range();
        // get time range:
        const range<fp_type> time = discretization_cfg_->time_range();
        // get space step:
        const fp_type h = discretization_cfg_->space_step();
        // time step:
        const fp_type k = discretization_cfg_->time_step();
        // wrap up steps into pair:
        const std::pair<fp_type, fp_type> steps = std::make_pair(k, h);
        // size of space discretization:
        const std::size_t space_size = discretization_cfg_->number_of_space_points();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        // save traverse_direction
        const traverse_direction_enum traverse_dir = solver_cfg_->traverse_direction();
        // create and set up the solver:
        auto const &solver = std::make_shared<sorsolver>(space, space_size);
        solver->set_diagonals(std::move(std::get<0>(diagonals_)), std::move(std::get<1>(diagonals_)),
                              std::move(std::get<2>(diagonals_)));
        solver->set_omega(omega_value);
        // solver->set_boundary(std::get<0>(boundary_pair_), std::get<1>(boundary_pair_));
        if (is_heat_sourse_set)
        {
            auto scheme_function =
                implicit_heat_scheme<fp_type, container, allocator>::get(solver_cfg_->implicit_pde_scheme(), false);
            // create a container to carry discretized source heat
            container_t source_curr(space_size, NaN<fp_type>());
            container_t source_next(space_size, NaN<fp_type>());
            loop::run_with_stepping(solver, scheme_function, boundary_pair_, fun_triplet_, space, time, last_time_idx,
                                    steps, traverse_dir, prev_solution, next_solution, rhs, heat_source, source_curr,
                                    source_next, solutions);
        }
        else
        {
            auto scheme_function =
                implicit_heat_scheme<fp_type, container, allocator>::get(solver_cfg_->implicit_pde_scheme(), true);
            loop::run_with_stepping(solver, scheme_function, boundary_pair_, fun_triplet_, space, time, last_time_idx,
                                    steps, traverse_dir, prev_solution, next_solution, rhs, solutions);
        }
    }
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class general_svc_heat_equation_implicit_kernel<memory_space_enum::Host, tridiagonal_method_enum::DoubleSweepSolver,
                                                fp_type, container, allocator>
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container<fp_type, allocator> container_t;
    typedef double_sweep_solver<fp_type, container, allocator> ds_solver;
    typedef heat_time_loop<fp_type, container, allocator> loop;

  private:
    diagonal_triplet_t<fp_type, container, allocator> diagonals_;
    function_triplet_t<fp_type> fun_triplet_;
    boundary_1d_pair<fp_type> boundary_pair_;
    pde_discretization_config_1d_ptr<fp_type> discretization_cfg_;
    heat_implicit_solver_config_ptr solver_cfg_;

  public:
    general_svc_heat_equation_implicit_kernel(diagonal_triplet_t<fp_type, container, allocator> const &diagonals,
                                              function_triplet_t<fp_type> const &fun_triplet,
                                              boundary_1d_pair<fp_type> const &boundary_pair,
                                              pde_discretization_config_1d_ptr<fp_type> const &discretization_config,
                                              heat_implicit_solver_config_ptr const &solver_config)
        : diagonals_{diagonals}, fun_triplet_{fun_triplet}, boundary_pair_{boundary_pair},
          discretization_cfg_{discretization_config}, solver_cfg_{solver_config}
    {
    }

    void operator()(container_t &prev_solution, container_t &next_solution, container_t &rhs, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type)> const &heat_source)
    {
        // get space range:
        const range<fp_type> space = discretization_cfg_->space_range();
        // get time range:
        const range<fp_type> time = discretization_cfg_->time_range();
        // get space step:
        const fp_type h = discretization_cfg_->space_step();
        // time step:
        const fp_type k = discretization_cfg_->time_step();
        // wrap up steps into pair:
        const std::pair<fp_type, fp_type> steps = std::make_pair(k, h);
        // size of space discretization:
        const std::size_t space_size = discretization_cfg_->number_of_space_points();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        // save traverse_direction
        const traverse_direction_enum traverse_dir = solver_cfg_->traverse_direction();
        // create and set up the solver:
        auto const &solver = std::make_shared<ds_solver>(space, space_size);
        solver->set_diagonals(std::move(std::get<0>(diagonals_)), std::move(std::get<1>(diagonals_)),
                              std::move(std::get<2>(diagonals_)));
        // solver->set_boundary(std::get<0>(boundary_pair_), std::get<1>(boundary_pair_));
        if (is_heat_sourse_set)
        {
            auto scheme_function =
                implicit_heat_scheme<fp_type, container, allocator>::get(solver_cfg_->implicit_pde_scheme(), false);
            // create a container to carry discretized source heat
            container_t source_curr(space_size, NaN<fp_type>());
            container_t source_next(space_size, NaN<fp_type>());
            loop::run(solver, scheme_function, boundary_pair_, fun_triplet_, space, time, last_time_idx, steps,
                      traverse_dir, prev_solution, next_solution, rhs, heat_source, source_curr, source_next);
        }
        else
        {
            auto scheme_function =
                implicit_heat_scheme<fp_type, container, allocator>::get(solver_cfg_->implicit_pde_scheme(), true);
            loop::run(solver, scheme_function, boundary_pair_, fun_triplet_, space, time, last_time_idx, steps,
                      traverse_dir, prev_solution, next_solution, rhs);
        }
    }

    void operator()(container_t &prev_solution, container_t &next_solution, container_t &rhs, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type)> const &heat_source,
                    container_2d<by_enum::Row, fp_type, container, allocator> &solutions)
    {
        // get space range:
        const range<fp_type> space = discretization_cfg_->space_range();
        // get time range:
        const range<fp_type> time = discretization_cfg_->time_range();
        // get space step:
        const fp_type h = discretization_cfg_->space_step();
        // time step:
        const fp_type k = discretization_cfg_->time_step();
        // wrap up steps into pair:
        const std::pair<fp_type, fp_type> steps = std::make_pair(k, h);
        // size of space discretization:
        const std::size_t space_size = discretization_cfg_->number_of_space_points();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        // save traverse_direction
        const traverse_direction_enum traverse_dir = solver_cfg_->traverse_direction();
        // create and set up the solver:
        auto const &solver = std::make_shared<ds_solver>(space, space_size);
        solver->set_diagonals(std::move(std::get<0>(diagonals_)), std::move(std::get<1>(diagonals_)),
                              std::move(std::get<2>(diagonals_)));
        // solver->set_boundary(std::get<0>(boundary_pair_), std::get<1>(boundary_pair_));
        if (is_heat_sourse_set)
        {
            auto scheme_function =
                implicit_heat_scheme<fp_type, container, allocator>::get(solver_cfg_->implicit_pde_scheme(), false);
            // create a container to carry discretized source heat
            container_t source_curr(space_size, NaN<fp_type>());
            container_t source_next(space_size, NaN<fp_type>());
            loop::run_with_stepping(solver, scheme_function, boundary_pair_, fun_triplet_, space, time, last_time_idx,
                                    steps, traverse_dir, prev_solution, next_solution, rhs, heat_source, source_curr,
                                    source_next, solutions);
        }
        else
        {
            auto scheme_function =
                implicit_heat_scheme<fp_type, container, allocator>::get(solver_cfg_->implicit_pde_scheme(), true);
            loop::run_with_stepping(solver, scheme_function, boundary_pair_, fun_triplet_, space, time, last_time_idx,
                                    steps, traverse_dir, prev_solution, next_solution, rhs, solutions);
        }
    }
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class general_svc_heat_equation_implicit_kernel<memory_space_enum::Host, tridiagonal_method_enum::ThomasLUSolver,
                                                fp_type, container, allocator>
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container<fp_type, allocator> container_t;
    typedef thomas_lu_solver<fp_type, container, allocator> tlu_solver;
    typedef heat_time_loop<fp_type, container, allocator> loop;

  private:
    diagonal_triplet_t<fp_type, container, allocator> diagonals_;
    function_triplet_t<fp_type> fun_triplet_;
    boundary_1d_pair<fp_type> boundary_pair_;
    pde_discretization_config_1d_ptr<fp_type> discretization_cfg_;
    heat_implicit_solver_config_ptr solver_cfg_;

  public:
    general_svc_heat_equation_implicit_kernel(diagonal_triplet_t<fp_type, container, allocator> const &diagonals,
                                              function_triplet_t<fp_type> const &fun_triplet,
                                              boundary_1d_pair<fp_type> const &boundary_pair,
                                              pde_discretization_config_1d_ptr<fp_type> const &discretization_config,
                                              heat_implicit_solver_config_ptr const &solver_config)
        : diagonals_{diagonals}, fun_triplet_{fun_triplet}, boundary_pair_{boundary_pair},
          discretization_cfg_{discretization_config}, solver_cfg_{solver_config}
    {
    }

    void operator()(container_t &prev_solution, container_t &next_solution, container_t &rhs, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type)> const &heat_source)
    {
        // get space range:
        const range<fp_type> space = discretization_cfg_->space_range();
        // get time range:
        const range<fp_type> time = discretization_cfg_->time_range();
        // get space step:
        const fp_type h = discretization_cfg_->space_step();
        // time step:
        const fp_type k = discretization_cfg_->time_step();
        // wrap up steps into pair:
        const std::pair<fp_type, fp_type> steps = std::make_pair(k, h);
        // size of space discretization:
        const std::size_t space_size = discretization_cfg_->number_of_space_points();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        // save traverse_direction
        const traverse_direction_enum traverse_dir = solver_cfg_->traverse_direction();
        // create and set up the solver:
        auto const &solver = std::make_shared<tlu_solver>(space, space_size);
        solver->set_diagonals(std::move(std::get<0>(diagonals_)), std::move(std::get<1>(diagonals_)),
                              std::move(std::get<2>(diagonals_)));
        // solver->set_boundary(std::get<0>(boundary_pair_), std::get<1>(boundary_pair_));
        if (is_heat_sourse_set)
        {
            auto scheme_function =
                implicit_heat_scheme<fp_type, container, allocator>::get(solver_cfg_->implicit_pde_scheme(), false);
            // create a container to carry discretized source heat
            container_t source_curr(space_size, NaN<fp_type>());
            container_t source_next(space_size, NaN<fp_type>());
            loop::run(solver, scheme_function, boundary_pair_, fun_triplet_, space, time, last_time_idx, steps,
                      traverse_dir, prev_solution, next_solution, rhs, heat_source, source_curr, source_next);
        }
        else
        {
            auto scheme_function =
                implicit_heat_scheme<fp_type, container, allocator>::get(solver_cfg_->implicit_pde_scheme(), true);
            loop::run(solver, scheme_function, boundary_pair_, fun_triplet_, space, time, last_time_idx, steps,
                      traverse_dir, prev_solution, next_solution, rhs);
        }
    }

    void operator()(container_t &prev_solution, container_t &next_solution, container_t &rhs, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type)> const &heat_source,
                    container_2d<by_enum::Row, fp_type, container, allocator> &solutions)
    {
        // get space range:
        const range<fp_type> space = discretization_cfg_->space_range();
        // get time range:
        const range<fp_type> time = discretization_cfg_->time_range();
        // get space step:
        const fp_type h = discretization_cfg_->space_step();
        // time step:
        const fp_type k = discretization_cfg_->time_step();
        // wrap up steps into pair:
        const std::pair<fp_type, fp_type> steps = std::make_pair(k, h);
        // size of space discretization:
        const std::size_t space_size = discretization_cfg_->number_of_space_points();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        // save traverse_direction
        const traverse_direction_enum traverse_dir = solver_cfg_->traverse_direction();
        // create and set up the solver:
        auto const &solver = std::make_shared<tlu_solver>(space, space_size);
        solver->set_diagonals(std::move(std::get<0>(diagonals_)), std::move(std::get<1>(diagonals_)),
                              std::move(std::get<2>(diagonals_)));
        // solver->set_boundary(std::get<0>(boundary_pair_), std::get<1>(boundary_pair_));
        if (is_heat_sourse_set)
        {
            auto scheme_function =
                implicit_heat_scheme<fp_type, container, allocator>::get(solver_cfg_->implicit_pde_scheme(), false);
            // create a container to carry discretized source heat
            container_t source_curr(space_size, NaN<fp_type>());
            container_t source_next(space_size, NaN<fp_type>());
            loop::run_with_stepping(solver, scheme_function, boundary_pair_, fun_triplet_, space, time, last_time_idx,
                                    steps, traverse_dir, prev_solution, next_solution, rhs, heat_source, source_curr,
                                    source_next, solutions);
        }
        else
        {
            auto scheme_function =
                implicit_heat_scheme<fp_type, container, allocator>::get(solver_cfg_->implicit_pde_scheme(), true);
            loop::run_with_stepping(solver, scheme_function, boundary_pair_, fun_triplet_, space, time, last_time_idx,
                                    steps, traverse_dir, prev_solution, next_solution, rhs, solutions);
        }
    }
};
} // namespace one_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_1D_GENERAL_SVC_HEAT_EQUATION_IMPLICIT_KERNEL_HPP_
