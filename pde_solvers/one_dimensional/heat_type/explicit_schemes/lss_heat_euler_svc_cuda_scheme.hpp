#if !defined(_LSS_HEAT_EULER_SVC_CUDA_SCHEME_HPP_)
#define _LSS_HEAT_EULER_SVC_CUDA_SCHEME_HPP_

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
#include "discretization/lss_grid.hpp"
#include "discretization/lss_grid_config.hpp"
#include "pde_solvers/lss_pde_discretization_config.hpp"
#include "pde_solvers/one_dimensional/heat_type/explicit_coefficients/lss_heat_euler_svc_coefficients.hpp"
#include "pde_solvers/one_dimensional/heat_type/solver_method/lss_heat_euler_cuda_solver_method.hpp"

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
using lss_grids::grid_1d;
using lss_utility::NaN;
using lss_utility::pair_t;
using lss_utility::range;
using lss_utility::sptr_t;

/**
 * heat_euler_svc_cuda_time_loop object
 */
template <typename fp_type, template <typename, typename> typename container, typename allocator>
class heat_euler_svc_cuda_time_loop
{
    typedef container<fp_type, allocator> container_t;
    typedef container_2d<by_enum::Row, fp_type, container, allocator> container_2d_t;

  public:
    template <typename solver>
    static void run(solver const &solver_ptr, boundary_1d_pair<fp_type> const &boundary_pair,
                    range<fp_type> const &time_range, std::size_t const &last_time_idx, fp_type const time_step,
                    traverse_direction_enum const &traverse_dir, container_t &solution);

    template <typename solver>
    static void run(solver const &solver_ptr, boundary_1d_pair<fp_type> const &boundary_pair,
                    range<fp_type> const &time_range, std::size_t const &last_time_idx, fp_type const time_step,
                    traverse_direction_enum const &traverse_dir,
                    std::function<fp_type(fp_type, fp_type)> const &heat_source, container_t &solution);

    template <typename solver>
    static void run_with_stepping(solver const &solver_ptr, boundary_1d_pair<fp_type> const &boundary_pair,
                                  range<fp_type> const &time_range, std::size_t const &last_time_idx,
                                  fp_type const time_step, traverse_direction_enum const &traverse_dir,
                                  container_t &solution, container_2d_t &solutions);

    template <typename solver>
    static void run_with_stepping(solver const &solver_ptr, boundary_1d_pair<fp_type> const &boundary_pair,
                                  range<fp_type> const &time_range, std::size_t const &last_time_idx,
                                  fp_type const time_step, traverse_direction_enum const &traverse_dir,
                                  std::function<fp_type(fp_type, fp_type)> const &heat_source, container_t &solution,
                                  container_2d_t &solutions);
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
template <typename solver>
void heat_euler_svc_cuda_time_loop<fp_type, container, allocator>::run(
    solver const &solver_ptr, boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &time_range,
    std::size_t const &last_time_idx, fp_type const time_step, traverse_direction_enum const &traverse_dir,
    container_t &solution)
{
    const std::size_t sol_size = solution.size();
    // ranges and steps:
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type k = time_step;
    // create host vectors:
    thrust::host_vector<fp_type> h_solution(sol_size);
    thrust::copy(solution.begin(), solution.end(), h_solution.begin());
    thrust::host_vector<fp_type> h_next_solution(sol_size);

    fp_type time{start_time + k};
    std::size_t time_idx{};
    if (traverse_dir == traverse_direction_enum::Forward)
    {
        time_idx = 1;
        while (time_idx <= last_time_idx)
        {
            solver_ptr->solve(h_solution, boundary_pair, time, h_next_solution);
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
            solver_ptr->solve(h_solution, boundary_pair, time, h_next_solution);
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
template <typename solver>
void heat_euler_svc_cuda_time_loop<fp_type, container, allocator>::run(
    solver const &solver_ptr, boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &time_range,
    std::size_t const &last_time_idx, fp_type const time_step, traverse_direction_enum const &traverse_dir,
    std::function<fp_type(fp_type, fp_type)> const &heat_source, container_t &solution)
{
    const std::size_t sol_size = solution.size();
    // ranges and steps:
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type k = time_step;
    // create host vectors:
    thrust::host_vector<fp_type> h_solution(sol_size);
    thrust::copy(solution.begin(), solution.end(), h_solution.begin());
    thrust::host_vector<fp_type> h_next_solution(sol_size);

    fp_type time{start_time + k};
    std::size_t time_idx{};
    if (traverse_dir == traverse_direction_enum::Forward)
    {
        time_idx = 1;
        while (time_idx <= last_time_idx)
        {
            solver_ptr->solve(h_solution, boundary_pair, time, heat_source, h_next_solution);
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
            solver_ptr->solve(h_solution, boundary_pair, time, heat_source, h_next_solution);
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
template <typename solver>
void heat_euler_svc_cuda_time_loop<fp_type, container, allocator>::run_with_stepping(
    solver const &solver_ptr, boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &time_range,
    std::size_t const &last_time_idx, fp_type const time_step, traverse_direction_enum const &traverse_dir,
    container_t &solution, container_2d_t &solutions)
{
    const std::size_t sol_size = solution.size();
    // ranges and steps:
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type k = time_step;
    // create host vectors:
    thrust::host_vector<fp_type> h_solution(sol_size);
    thrust::copy(solution.begin(), solution.end(), h_solution.begin());
    thrust::host_vector<fp_type> h_next_solution(sol_size);

    fp_type time{start_time + k};
    std::size_t time_idx{};
    if (traverse_dir == traverse_direction_enum::Forward)
    {
        // store the initial solution:
        solutions(0, solution);
        time_idx = 1;
        while (time_idx <= last_time_idx)
        {
            solver_ptr->solve(h_solution, boundary_pair, time, h_next_solution);
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
            solver_ptr->solve(h_solution, boundary_pair, time, h_next_solution);
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
template <typename solver>
void heat_euler_svc_cuda_time_loop<fp_type, container, allocator>::run_with_stepping(
    solver const &solver_ptr, boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &time_range,
    std::size_t const &last_time_idx, fp_type const time_step, traverse_direction_enum const &traverse_dir,
    std::function<fp_type(fp_type, fp_type)> const &heat_source, container_t &solution, container_2d_t &solutions)
{
    typedef discretization<dimension_enum::One, fp_type, thrust::host_vector, std::allocator<fp_type>> d_1d;

    const std::size_t sol_size = solution.size();
    // ranges and steps:
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type k = time_step;
    // create host vectors:
    thrust::host_vector<fp_type> h_solution(sol_size);
    thrust::host_vector<fp_type> h_next_solution(sol_size);
    thrust::copy(solution.begin(), solution.end(), h_solution.begin());

    fp_type time{start_time + k};
    std::size_t time_idx{};
    if (traverse_dir == traverse_direction_enum::Forward)
    {
        // store the initial solution:
        solutions(0, solution);
        time_idx = 1;
        while (time_idx <= last_time_idx)
        {
            solver_ptr->solve(h_solution, boundary_pair, time, heat_source, h_next_solution);
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
            solver_ptr->solve(h_solution, boundary_pair, time, heat_source, h_next_solution);
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

/**
 * heat_euler_svc_cuda_scheme object
 */
template <typename fp_type, template <typename, typename> typename container, typename allocator>
class heat_euler_svc_cuda_scheme
{
    typedef heat_euler_svc_cuda_time_loop<fp_type, container, allocator> loop;
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container<fp_type, allocator> container_t;

  private:
    heat_euler_svc_coefficients_ptr<fp_type> euler_coeffs_;
    boundary_1d_pair<fp_type> boundary_pair_;
    pde_discretization_config_1d_ptr<fp_type> discretization_cfg_;
    grid_config_1d_ptr<fp_type> grid_cfg_;

    bool is_stable(general_svc_heat_equation_implicit_coefficients_ptr<fp_type> const &coefficients)
    {
        const fp_type zero = static_cast<fp_type>(0.0);
        const fp_type two = static_cast<fp_type>(2.0);
        auto const &A = coefficients->A_;
        auto const &B = coefficients->B_;
        auto const &D = coefficients->D_;
        const fp_type k = coefficients->k_;
        const fp_type lambda = coefficients->lambda_;
        const fp_type gamma = coefficients->gamma_;
        const fp_type delta = coefficients->delta_;
        auto const &a = [=](fp_type x) { return ((A(x) + D(x)) / (two * lambda)); };
        auto const &b = [=](fp_type x) { return ((D(x) - A(x)) / (two * gamma)); };
        auto const &c = [=](fp_type x) { return ((lambda * a(x) - B(x)) / delta); };
        const std::size_t space_size = discretization_cfg_->number_of_space_points();
        fp_type x{};
        for (std::size_t i = 0; i < space_size; ++i)
        {
            x = grid_1d<fp_type>::value(grid_cfg_, i);
            if (c(x) > zero)
                return false;
            if ((gamma * gamma * b(x) * b(x)) > (two * lambda * a(x)))
                return false;
        }
        return true;
    }

    void initialize(general_svc_heat_equation_implicit_coefficients_ptr<fp_type> const &coefficients)
    {
        LSS_ASSERT(is_stable(coefficients) == true, "The chosen scheme is not stable");
        euler_coeffs_ = std::make_shared<heat_euler_svc_coefficients<fp_type>>(coefficients);
    }

    explicit heat_euler_svc_cuda_scheme() = delete;

  public:
    heat_euler_svc_cuda_scheme(general_svc_heat_equation_implicit_coefficients_ptr<fp_type> const &coefficients,
                               boundary_1d_pair<fp_type> const &boundary_pair,
                               pde_discretization_config_1d_ptr<fp_type> const &discretization_config,
                               grid_config_1d_ptr<fp_type> const &grid_config)
        : boundary_pair_{boundary_pair}, discretization_cfg_{discretization_config}, grid_cfg_{grid_config}
    {
        initialize(coefficients);
    }

    ~heat_euler_svc_cuda_scheme()
    {
    }

    void operator()(container_t &solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type)> const &heat_source, traverse_direction_enum traverse_dir)
    {
        const range<fp_type> timer = discretization_cfg_->time_range();
        const fp_type k = discretization_cfg_->time_step();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        auto const &solver_method_ptr =
            std::make_shared<heat_euler_cuda_solver_method<fp_type>>(euler_coeffs_, grid_cfg_);
        if (is_heat_sourse_set)
        {
            loop::run(solver_method_ptr, boundary_pair_, timer, last_time_idx, k, traverse_dir, heat_source, solution);
        }
        else
        {
            loop::run(solver_method_ptr, boundary_pair_, timer, last_time_idx, k, traverse_dir, solution);
        }
    }

    void operator()(container_t &solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type)> const &heat_source, traverse_direction_enum traverse_dir,
                    container_2d<by_enum::Row, fp_type, container, allocator> &solutions)
    {
        const range<fp_type> timer = discretization_cfg_->time_range();
        const fp_type k = discretization_cfg_->time_step();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        auto const &solver_method_ptr =
            std::make_shared<heat_euler_cuda_solver_method<fp_type>>(euler_coeffs_, grid_cfg_);
        if (is_heat_sourse_set)
        {

            loop::run_with_stepping(solver_method_ptr, boundary_pair_, timer, last_time_idx, k, traverse_dir,
                                    heat_source, solution, solutions);
        }
        else
        {
            loop::run_with_stepping(solver_method_ptr, boundary_pair_, timer, last_time_idx, k, traverse_dir, solution,
                                    solutions);
        }
    }
};

} // namespace one_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_EULER_SVC_CUDA_SCHEME_HPP_
