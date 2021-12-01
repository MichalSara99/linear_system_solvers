#if !defined(_LSS_WAVE_EULER_CUDA_SCHEME_HPP_)
#define _LSS_WAVE_EULER_CUDA_SCHEME_HPP_

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
#include "discretization/lss_grid.hpp"
#include "discretization/lss_grid_config.hpp"
#include "lss_wave_euler_scheme.hpp"
#include "pde_solvers/lss_pde_discretization_config.hpp"
#include "pde_solvers/one_dimensional/wave_type/explicit_coefficients/lss_wave_explicit_coefficients.hpp"
#include "pde_solvers/one_dimensional/wave_type/solver_method/lss_wave_euler_cuda_solver_method.hpp"

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
using lss_utility::NaN;
using lss_utility::range;

/**
 * wave_euler_cuda_time_loop object
 */
template <typename fp_type, template <typename, typename> typename container, typename allocator>
class wave_euler_cuda_time_loop
{
    typedef container<fp_type, allocator> container_t;
    typedef container_2d<by_enum::Row, fp_type, container, allocator> container_2d_t;

  public:
    template <typename solver>
    static void run(solver const &solver_ptr, boundary_1d_pair<fp_type> const &boundary_pair,
                    range<fp_type> const &time_range, std::size_t const &last_time_idx, fp_type const time_step,
                    traverse_direction_enum const &traverse_dir, container_t &prev_solution_0,
                    container_t &prev_solution_1, container_t &next_solution);

    template <typename solver>
    static void run(solver const &solver_ptr, boundary_1d_pair<fp_type> const &boundary_pair,
                    range<fp_type> const &time_range, std::size_t const &last_time_idx, fp_type const time_step,
                    traverse_direction_enum const &traverse_dir, container_t &prev_solution_0,
                    container_t &prev_solution_1, std::function<fp_type(fp_type, fp_type)> const &wave_source,
                    container_t &next_solution);

    template <typename solver>
    static void run_with_stepping(solver const &solver_ptr, boundary_1d_pair<fp_type> const &boundary_pair,
                                  range<fp_type> const &time_range, std::size_t const &last_time_idx,
                                  fp_type const time_step, traverse_direction_enum const &traverse_dir,
                                  container_t &prev_solution_0, container_t &prev_solution_1,
                                  container_t &next_solution, container_2d_t &solutions);

    template <typename solver>
    static void run_with_stepping(solver const &solver_ptr, boundary_1d_pair<fp_type> const &boundary_pair,
                                  range<fp_type> const &time_range, std::size_t const &last_time_idx,
                                  fp_type const time_step, traverse_direction_enum const &traverse_dir,
                                  container_t &prev_solution_0, container_t &prev_solution_1,
                                  std::function<fp_type(fp_type, fp_type)> const &wave_source,
                                  container_t &next_solution, container_2d_t &solutions);
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
template <typename solver>
void wave_euler_cuda_time_loop<fp_type, container, allocator>::run(
    solver const &solver_ptr, boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &time_range,
    std::size_t const &last_time_idx, fp_type const time_step, traverse_direction_enum const &traverse_dir,
    container_t &prev_solution_0, container_t &prev_solution_1, container_t &next_solution)
{
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type k = time_step;
    // create host vectors:
    const std::size_t sol_size = prev_solution_0.size();
    thrust::host_vector<fp_type> h_solution_0(sol_size);
    thrust::copy(prev_solution_0.begin(), prev_solution_0.end(), h_solution_0.begin());
    thrust::host_vector<fp_type> h_solution_1(sol_size);
    thrust::copy(prev_solution_1.begin(), prev_solution_1.end(), h_solution_1.begin());
    thrust::host_vector<fp_type> h_next_solution(sol_size);

    fp_type time{start_time};
    fp_type next_time{time + k};
    std::size_t time_idx{};
    if (traverse_dir == traverse_direction_enum::Forward)
    {
        // solve for initial time step:
        solver_ptr->solve_initial(h_solution_0, h_solution_1, boundary_pair, time, next_time, h_next_solution);
        h_solution_1 = h_next_solution;
        time_idx = 1;

        // solve for rest of time steps:
        time += k;
        next_time += k;
        time_idx++;
        while (time_idx <= last_time_idx)
        {
            solver_ptr->solve(h_solution_0, h_solution_1, boundary_pair, time, next_time, h_next_solution);
            h_solution_0 = h_solution_1;
            h_solution_1 = h_next_solution;
            time += k;
            next_time += k;
            time_idx++;
        }
    }
    else if (traverse_dir == traverse_direction_enum::Backward)
    {
        time_idx = last_time_idx;
        time = end_time;
        next_time = time - k;
        // solve for initial time step:
        solver_ptr->solve_terminal(h_solution_0, h_solution_1, boundary_pair, time, next_time, h_next_solution);
        h_solution_1 = h_next_solution;

        // solve for rest of time steps:
        time -= k;
        next_time -= k;
        do
        {
            time_idx--;
            solver_ptr->solve(h_solution_0, h_solution_1, boundary_pair, time, next_time, h_next_solution);
            h_solution_0 = h_solution_1;
            h_solution_1 = h_next_solution;
            time -= k;
            next_time -= k;
        } while (time_idx > 0);
    }
    else
    {
        throw std::exception("Unreachable");
    }
    thrust::copy(h_next_solution.begin(), h_next_solution.end(), next_solution.begin());
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
template <typename solver>
void wave_euler_cuda_time_loop<fp_type, container, allocator>::run(
    solver const &solver_ptr, boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &time_range,
    std::size_t const &last_time_idx, fp_type const time_step, traverse_direction_enum const &traverse_dir,
    container_t &prev_solution_0, container_t &prev_solution_1,
    std::function<fp_type(fp_type, fp_type)> const &wave_source, container_t &next_solution)
{
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type k = time_step;
    fp_type time{start_time};
    fp_type next_time{time + k};
    std::size_t time_idx{};

    // create host vectors:
    const std::size_t sol_size = prev_solution_0.size();
    thrust::host_vector<fp_type> h_solution_0(sol_size);
    thrust::copy(prev_solution_0.begin(), prev_solution_0.end(), h_solution_0.begin());
    thrust::host_vector<fp_type> h_solution_1(sol_size);
    thrust::copy(prev_solution_1.begin(), prev_solution_1.end(), h_solution_1.begin());
    thrust::host_vector<fp_type> h_next_solution(sol_size);

    if (traverse_dir == traverse_direction_enum::Forward)
    {
        // solve for initial time step:
        solver_ptr->solve_initial(h_solution_0, h_solution_1, boundary_pair, time, next_time, wave_source,
                                  h_next_solution);
        h_solution_1 = h_next_solution;
        time_idx = 1;

        // solve for rest of time steps:
        time += k;
        next_time += k;
        time_idx++;
        while (time_idx <= last_time_idx)
        {
            solver_ptr->solve(h_solution_0, h_solution_1, boundary_pair, time, next_time, wave_source, h_next_solution);
            h_solution_0 = h_solution_1;
            h_solution_1 = h_next_solution;
            time += k;
            next_time += k;
            time_idx++;
        }
    }
    else if (traverse_dir == traverse_direction_enum::Backward)
    {
        time_idx = last_time_idx;
        time = end_time;
        next_time = time - k;
        // solve for initial time step:
        solver_ptr->solve_terminal(h_solution_0, h_solution_1, boundary_pair, time, next_time, wave_source,
                                   h_next_solution);
        h_solution_1 = h_next_solution;
        time_idx--;

        // solve for rest of time steps:
        time -= k;
        next_time -= k;
        do
        {
            time_idx--;
            solver_ptr->solve(h_solution_0, h_solution_1, boundary_pair, time, next_time, wave_source, h_next_solution);
            h_solution_0 = h_solution_1;
            h_solution_1 = h_next_solution;
            time -= k;
            next_time -= k;
        } while (time_idx > 0);
    }
    else
    {
        throw std::exception("Unreachable");
    }
    thrust::copy(h_next_solution.begin(), h_next_solution.end(), next_solution.begin());
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
template <typename solver>
void wave_euler_cuda_time_loop<fp_type, container, allocator>::run_with_stepping(
    solver const &solver_ptr, boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &time_range,
    std::size_t const &last_time_idx, fp_type const time_step, traverse_direction_enum const &traverse_dir,
    container_t &prev_solution_0, container_t &prev_solution_1, container_t &next_solution, container_2d_t &solutions)
{
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type k = time_step;
    fp_type time{start_time};
    fp_type next_time{time + k};
    std::size_t time_idx{};

    // create host vectors:
    const std::size_t sol_size = prev_solution_0.size();
    thrust::host_vector<fp_type> h_solution_0(sol_size);
    thrust::copy(prev_solution_0.begin(), prev_solution_0.end(), h_solution_0.begin());
    thrust::host_vector<fp_type> h_solution_1(sol_size);
    thrust::copy(prev_solution_1.begin(), prev_solution_1.end(), h_solution_1.begin());
    thrust::host_vector<fp_type> h_next_solution(sol_size);

    if (traverse_dir == traverse_direction_enum::Forward)
    {
        // store the initial solution:
        solutions(0, prev_solution_0);
        // solve for initial time step:
        solver_ptr->solve_initial(h_solution_0, h_solution_1, boundary_pair, time, next_time, h_next_solution);
        h_solution_1 = h_next_solution;
        time_idx = 1;
        thrust::copy(h_next_solution.begin(), h_next_solution.end(), next_solution.begin());
        solutions(time_idx, next_solution);

        // solve for rest of time steps:
        time += k;
        next_time += k;
        time_idx++;
        while (time_idx <= last_time_idx)
        {
            solver_ptr->solve(h_solution_0, h_solution_1, boundary_pair, time, next_time, h_next_solution);
            h_solution_0 = h_solution_1;
            h_solution_1 = h_next_solution;
            thrust::copy(h_next_solution.begin(), h_next_solution.end(), next_solution.begin());
            solutions(time_idx, next_solution);
            time += k;
            next_time += k;
            time_idx++;
        }
    }
    else if (traverse_dir == traverse_direction_enum::Backward)
    {
        time_idx = last_time_idx;
        // store the terminal solution:
        solutions(last_time_idx, prev_solution_0);
        time = end_time;
        next_time = time - k;
        // solve for terminal time step:
        solver_ptr->solve_terminal(h_solution_0, h_solution_1, boundary_pair, time, next_time, h_next_solution);
        h_solution_1 = h_next_solution;
        time_idx--;
        thrust::copy(h_next_solution.begin(), h_next_solution.end(), next_solution.begin());
        solutions(time_idx, next_solution);

        // solve for rest of time steps:
        time -= k;
        next_time -= k;
        do
        {
            time_idx--;
            solver_ptr->solve(h_solution_0, h_solution_1, boundary_pair, time, next_time, h_next_solution);
            h_solution_0 = h_solution_1;
            h_solution_1 = h_next_solution;
            thrust::copy(h_next_solution.begin(), h_next_solution.end(), next_solution.begin());
            solutions(time_idx, next_solution);
            time -= k;
            next_time -= k;
        } while (time_idx > 0);
    }
    else
    {
        throw std::exception("Unreachable");
    }
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
template <typename solver>
void wave_euler_cuda_time_loop<fp_type, container, allocator>::run_with_stepping(
    solver const &solver_ptr, boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &time_range,
    std::size_t const &last_time_idx, fp_type const time_step, traverse_direction_enum const &traverse_dir,
    container_t &prev_solution_0, container_t &prev_solution_1,
    std::function<fp_type(fp_type, fp_type)> const &wave_source, container_t &next_solution, container_2d_t &solutions)
{
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type k = time_step;
    fp_type time{start_time};
    fp_type next_time{time + k};
    std::size_t time_idx{};

    // create host vectors:
    const std::size_t sol_size = prev_solution_0.size();
    thrust::host_vector<fp_type> h_solution_0(sol_size);
    thrust::copy(prev_solution_0.begin(), prev_solution_0.end(), h_solution_0.begin());
    thrust::host_vector<fp_type> h_solution_1(sol_size);
    thrust::copy(prev_solution_1.begin(), prev_solution_1.end(), h_solution_1.begin());
    thrust::host_vector<fp_type> h_next_solution(sol_size);

    if (traverse_dir == traverse_direction_enum::Forward)
    {
        // store the initial solution:
        solutions(0, prev_solution_0);
        // solve for initial time step:
        solver_ptr->solve_initial(h_solution_0, h_solution_1, boundary_pair, time, next_time, wave_source,
                                  h_next_solution);
        h_solution_1 = h_next_solution;
        time_idx = 1;
        thrust::copy(h_next_solution.begin(), h_next_solution.end(), next_solution.begin());
        solutions(time_idx, next_solution);

        // solve for rest of time steps:
        time += k;
        next_time += k;
        time_idx++;
        while (time_idx <= last_time_idx)
        {
            solver_ptr->solve(h_solution_0, h_solution_1, boundary_pair, time, next_time, wave_source, h_next_solution);
            h_solution_0 = h_solution_1;
            h_solution_1 = h_next_solution;
            thrust::copy(h_next_solution.begin(), h_next_solution.end(), next_solution.begin());
            solutions(time_idx, next_solution);
            time += k;
            next_time += k;
            time_idx++;
        }
    }
    else if (traverse_dir == traverse_direction_enum::Backward)
    {
        time_idx = last_time_idx;
        // store the terminal solution:
        solutions(last_time_idx, prev_solution_0);
        time = end_time;
        next_time = time - k;
        // solve for terminal time step:
        solver_ptr->solve_terminal(h_solution_0, h_solution_1, boundary_pair, time, next_time, wave_source,
                                   h_next_solution);
        h_solution_1 = h_next_solution;
        time_idx--;
        thrust::copy(h_next_solution.begin(), h_next_solution.end(), next_solution.begin());
        solutions(time_idx, next_solution);

        // solve for rest of time steps:
        time -= k;
        next_time -= k;
        do
        {
            time_idx--;
            solver_ptr->solve(h_solution_0, h_solution_1, boundary_pair, time, next_time, wave_source, h_next_solution);
            h_solution_0 = h_solution_1;
            h_solution_1 = h_next_solution;
            thrust::copy(h_next_solution.begin(), h_next_solution.end(), next_solution.begin());
            solutions(time_idx, next_solution);
            time -= k;
            next_time -= k;
        } while (time_idx > 0);
    }
    else
    {
        throw std::exception("Unreachable");
    }
}
/**
 * wave_euler_cuda_scheme object
 */
template <typename fp_type, template <typename, typename> typename container, typename allocator>
class wave_euler_cuda_scheme
{
    typedef wave_euler_cuda_time_loop<fp_type, container, allocator> loop;
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container<fp_type, allocator> container_t;

  private:
    wave_explicit_coefficients_ptr<fp_type> euler_coeffs_;
    boundary_1d_pair<fp_type> boundary_pair_;
    pde_discretization_config_1d_ptr<fp_type> discretization_cfg_;
    grid_config_1d_ptr<fp_type> grid_cfg_;

    bool is_stable()
    {
        auto const &b = euler_coeffs_->b_;
        const fp_type k = euler_coeffs_->k_;
        const fp_type h = grid_1d<fp_type>::step(grid_cfg_);
        const fp_type ratio = h / k;
        const std::size_t space_size = discretization_cfg_->number_of_space_points();
        auto const ftime = discretization_cfg_->time_range().upper();
        fp_type x{}, t{k};
        while (t <= ftime)
        {
            for (std::size_t i = 0; i < space_size; ++i)
            {
                x = grid_1d<fp_type>::value(grid_cfg_, i);
                if (b(t, x) >= ratio)
                    return false;
            }
            t += k;
        }
        return true;
    }

    void initialize()
    {
        LSS_ASSERT(is_stable() == true, "The chosen scheme is not stable");
    }

    explicit wave_euler_cuda_scheme() = delete;

  public:
    wave_euler_cuda_scheme(wave_explicit_coefficients_ptr<fp_type> const &coefficients,
                           boundary_1d_pair<fp_type> const &boundary_pair,
                           pde_discretization_config_1d_ptr<fp_type> const &discretization_config,
                           grid_config_1d_ptr<fp_type> const &grid_config)
        : euler_coeffs_{coefficients}, boundary_pair_{boundary_pair},
          discretization_cfg_{discretization_config}, grid_cfg_{grid_config}
    {
        initialize();
    }

    ~wave_euler_cuda_scheme()
    {
    }

    void operator()(container_t &prev_solution_0, container_t &prev_solution_1, container_t &next_solution,
                    bool is_wave_sourse_set, std::function<fp_type(fp_type, fp_type)> const &wave_source,
                    traverse_direction_enum traverse_dir)
    {
        const range<fp_type> timer = discretization_cfg_->time_range();
        const fp_type k = discretization_cfg_->time_step();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        auto const &solver_method_ptr =
            std::make_shared<wave_euler_cuda_solver_method<fp_type>>(euler_coeffs_, grid_cfg_, is_wave_sourse_set);
        if (is_wave_sourse_set)
        {

            loop::run(solver_method_ptr, boundary_pair_, timer, last_time_idx, k, traverse_dir, prev_solution_0,
                      prev_solution_1, wave_source, next_solution);
        }
        else
        {
            loop::run(solver_method_ptr, boundary_pair_, timer, last_time_idx, k, traverse_dir, prev_solution_0,
                      prev_solution_1, next_solution);
        }
    }

    void operator()(container_t &prev_solution_0, container_t &prev_solution_1, container_t &next_solution,
                    bool is_wave_sourse_set, std::function<fp_type(fp_type, fp_type)> const &wave_source,
                    traverse_direction_enum traverse_dir,
                    container_2d<by_enum::Row, fp_type, container, allocator> &solutions)
    {
        const range<fp_type> timer = discretization_cfg_->time_range();
        const fp_type k = discretization_cfg_->time_step();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        auto const &solver_method_ptr =
            std::make_shared<wave_euler_cuda_solver_method<fp_type>>(euler_coeffs_, grid_cfg_, is_wave_sourse_set);
        if (is_wave_sourse_set)
        {

            loop::run_with_stepping(solver_method_ptr, boundary_pair_, timer, last_time_idx, k, traverse_dir,
                                    prev_solution_0, prev_solution_1, wave_source, next_solution, solutions);
        }
        else
        {
            loop::run_with_stepping(solver_method_ptr, boundary_pair_, timer, last_time_idx, k, traverse_dir,
                                    prev_solution_0, prev_solution_1, next_solution, solutions);
        }
    }
};

} // namespace one_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_WAVE_EULER_CUDA_SCHEME_HPP_
