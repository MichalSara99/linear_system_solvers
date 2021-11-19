#if !defined(_LSS_HEAT_BARAKAT_CLARK_SVC_SCHEME_HPP_)
#define _LSS_HEAT_BARAKAT_CLARK_SVC_SCHEME_HPP_

#include <thread>

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
#include "pde_solvers/one_dimensional/heat_type/explicit_coefficients/lss_heat_barakat_clark_svc_coefficients.hpp"
#include "pde_solvers/one_dimensional/heat_type/solver_method/lss_heat_barakat_clark_solver_method.hpp"

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
using lss_enumerations::traverse_direction_enum;
using lss_utility::function_quad_t;
using lss_utility::function_triplet_t;
using lss_utility::NaN;
using lss_utility::range;

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class heat_barakat_clark_svc_time_loop
{
    typedef container<fp_type, allocator> container_t;
    typedef container_2d<by_enum::Row, fp_type, container, allocator> container_2d_t;

  public:
    template <typename solver>
    static void run(solver const &solver_ptr, boundary_1d_pair<fp_type> const &boundary_pair,
                    range<fp_type> const &time_range, std::size_t const &last_time_idx,
                    std::pair<fp_type, fp_type> const &steps, traverse_direction_enum const &traverse_dir,
                    container_t &solution);

    template <typename solver>
    static void run(solver const &solver_ptr, boundary_1d_pair<fp_type> const &boundary_pair,
                    range<fp_type> const &time_range, std::size_t const &last_time_idx,
                    std::pair<fp_type, fp_type> const &steps, traverse_direction_enum const &traverse_dir,
                    std::function<fp_type(fp_type, fp_type)> const &heat_source, container_t &solution);

    template <typename solver>
    static void run_with_stepping(solver const &solver_ptr, boundary_1d_pair<fp_type> const &boundary_pair,
                                  range<fp_type> const &time_range, std::size_t const &last_time_idx,
                                  std::pair<fp_type, fp_type> const &steps, traverse_direction_enum const &traverse_dir,
                                  container_t &solution, container_2d_t &solutions);

    template <typename solver>
    static void run_with_stepping(solver const &solver_ptr, boundary_1d_pair<fp_type> const &boundary_pair,
                                  range<fp_type> const &time_range, std::size_t const &last_time_idx,
                                  std::pair<fp_type, fp_type> const &steps, traverse_direction_enum const &traverse_dir,
                                  std::function<fp_type(fp_type, fp_type)> const &heat_source, container_t &solution,
                                  container_2d_t &solutions);
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
template <typename solver>
void heat_barakat_clark_svc_time_loop<fp_type, container, allocator>::run(
    solver const &solver_ptr, boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &time_range,
    std::size_t const &last_time_idx, std::pair<fp_type, fp_type> const &steps,
    traverse_direction_enum const &traverse_dir, container_t &solution)
{

    // ranges and steps:
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type k = std::get<0>(steps);

    fp_type time{start_time + k};
    std::size_t time_idx{};
    if (traverse_dir == traverse_direction_enum::Forward)
    {
        time_idx = 1;
        while (time_idx <= last_time_idx)
        {
            solver_ptr->solve(boundary_pair, time, solution);
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
            solver_ptr->solve(boundary_pair, time, solution);
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
void heat_barakat_clark_svc_time_loop<fp_type, container, allocator>::run(
    solver const &solver_ptr, boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &time_range,
    std::size_t const &last_time_idx, std::pair<fp_type, fp_type> const &steps,
    traverse_direction_enum const &traverse_dir, std::function<fp_type(fp_type, fp_type)> const &heat_source,
    container_t &solution)
{
    // ranges and steps:
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type k = std::get<0>(steps);

    fp_type time{start_time + k};
    fp_type next_time{time + k};
    std::size_t time_idx{};
    if (traverse_dir == traverse_direction_enum::Forward)
    {
        time_idx = 1;
        while (time_idx <= last_time_idx)
        {
            solver_ptr->solve(boundary_pair, time, next_time, heat_source, solution);
            time += k;
            next_time += k;
            time_idx++;
        }
    }
    else if (traverse_dir == traverse_direction_enum::Backward)
    {
        time = end_time - k;
        next_time = time - k;
        time_idx = last_time_idx;
        do
        {
            time_idx--;
            solver_ptr->solve(boundary_pair, time, next_time, heat_source, solution);
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
void heat_barakat_clark_svc_time_loop<fp_type, container, allocator>::run_with_stepping(
    solver const &solver_ptr, boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &time_range,
    std::size_t const &last_time_idx, std::pair<fp_type, fp_type> const &steps,
    traverse_direction_enum const &traverse_dir, container_t &solution, container_2d_t &solutions)
{
    // ranges and steps:
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type k = std::get<0>(steps);

    fp_type time{start_time + k};
    std::size_t time_idx{};
    if (traverse_dir == traverse_direction_enum::Forward)
    {
        // store the initial solution:
        solutions(0, solution);
        time_idx = 1;
        while (time_idx <= last_time_idx)
        {
            solver_ptr->solve(boundary_pair, time, solution);
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
            solver_ptr->solve(boundary_pair, time, solution);
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
void heat_barakat_clark_svc_time_loop<fp_type, container, allocator>::run_with_stepping(
    solver const &solver_ptr, boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &time_range,
    std::size_t const &last_time_idx, std::pair<fp_type, fp_type> const &steps,
    traverse_direction_enum const &traverse_dir, std::function<fp_type(fp_type, fp_type)> const &heat_source,
    container_t &solution, container_2d_t &solutions)
{
    // ranges and steps:
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type k = std::get<0>(steps);

    fp_type time{start_time + k};
    fp_type next_time{time + k};
    std::size_t time_idx{};
    if (traverse_dir == traverse_direction_enum::Forward)
    {
        // store the initial solution:
        solutions(0, solution);
        time_idx = 1;
        while (time_idx <= last_time_idx)
        {
            solver_ptr->solve(boundary_pair, time, next_time, heat_source, solution);
            solutions(time_idx, solution);
            time += k;
            next_time += k;
            time_idx++;
        }
    }
    else if (traverse_dir == traverse_direction_enum::Backward)
    {
        // store the initial solution:
        solutions(last_time_idx, solution);
        time = end_time - k;
        next_time = time - k;
        time_idx = last_time_idx;
        do
        {
            time_idx--;
            solver_ptr->solve(boundary_pair, time, next_time, heat_source, solution);
            solutions(time_idx, solution);
            next_time -= k;
            time -= k;
        } while (time_idx > 0);
    }
    else
    {
        throw std::exception("Unreachable");
    }
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class heat_barakat_clark_svc_scheme
{
    typedef heat_barakat_clark_svc_time_loop<fp_type, container, allocator> loop;
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container<fp_type, allocator> container_t;

  private:
    heat_barakat_clark_svc_coefficients_ptr<fp_type> bc_coeffs_;
    boundary_1d_pair<fp_type> boundary_pair_;
    pde_discretization_config_1d_ptr<fp_type> discretization_cfg_;
    grid_config_1d_ptr<fp_type> grid_cfg_;

    void initialize(general_svc_heat_equation_implicit_coefficients_ptr<fp_type> const &coefficients)
    {
        auto const &first = boundary_pair_.first;
        if (std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(first))
        {
            throw std::exception("Neumann boundary type is not supported for this scheme");
        }
        if (std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first))
        {
            throw std::exception("Robin boundary type is not supported for this scheme");
        }
        auto const &second = boundary_pair_.second;
        if (std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(second))
        {
            throw std::exception("Neumann boundary type is not supported for this scheme");
        }
        if (std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second))
        {
            throw std::exception("Robin boundary type is not supported for this scheme");
        }
        bc_coeffs_ = std::make_shared<heat_barakat_clark_svc_coefficients<fp_type>>(coefficients);
    }

    explicit heat_barakat_clark_svc_scheme() = delete;

  public:
    heat_barakat_clark_svc_scheme(general_svc_heat_equation_implicit_coefficients_ptr<fp_type> const &coefficients,
                                  boundary_1d_pair<fp_type> const &boundary_pair,
                                  pde_discretization_config_1d_ptr<fp_type> const &discretization_config,
                                  grid_config_1d_ptr<fp_type> const &grid_config)
        : boundary_pair_{boundary_pair}, discretization_cfg_{discretization_config}, grid_cfg_{grid_config}
    {
        initialize(coefficients);
    }

    ~heat_barakat_clark_svc_scheme()
    {
    }

    void operator()(container_t &solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type)> const &heat_source, traverse_direction_enum traverse_dir)
    {
        const range<fp_type> timer = discretization_cfg_->time_range();
        const fp_type k = discretization_cfg_->time_step();
        const fp_type h = discretization_cfg_->space_step();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        auto const &steps = std::make_pair(k, h);
        auto const &solver_method_ptr =
            std::make_shared<heat_barakat_clark_solver_method<fp_type, container, allocator>>(bc_coeffs_, grid_cfg_);
        if (is_heat_sourse_set)
        {

            loop::run(solver_method_ptr, boundary_pair_, timer, last_time_idx, steps, traverse_dir, heat_source,
                      solution);
        }
        else
        {
            loop::run(solver_method_ptr, boundary_pair_, timer, last_time_idx, steps, traverse_dir, solution);
        }
    }

    void operator()(container_t &solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type)> const &heat_source, traverse_direction_enum traverse_dir,
                    container_2d<by_enum::Row, fp_type, container, allocator> &solutions)
    {
        const range<fp_type> timer = discretization_cfg_->time_range();
        const fp_type k = discretization_cfg_->time_step();
        const fp_type h = discretization_cfg_->space_step();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        auto const &steps = std::make_pair(k, h);
        auto const &solver_method_ptr =
            std::make_shared<heat_barakat_clark_solver_method<fp_type, container, allocator>>(bc_coeffs_, grid_cfg_);
        if (is_heat_sourse_set)
        {
            loop::run_with_stepping(solver_method_ptr, boundary_pair_, timer, last_time_idx, steps, traverse_dir,
                                    heat_source, solution, solutions);
        }
        else
        {
            loop::run_with_stepping(solver_method_ptr, boundary_pair_, timer, last_time_idx, steps, traverse_dir,
                                    solution, solutions);
        }
    }
};

} // namespace one_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_HEAT_BARAKAT_CLARK_SVC_SCHEME_HPP_
