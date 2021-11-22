#if !defined(_LSS_1D_GENERAL_SVC_WAVE_EQUATION_IMPLICIT_KERNEL_HPP_)
#define _LSS_1D_GENERAL_SVC_WAVE_EQUATION_IMPLICIT_KERNEL_HPP_

#include <vector>

#include "boundaries/lss_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include "containers/lss_container_2d.hpp"
#include "discretization/lss_discretization.hpp"
#include "implicit_coefficients/lss_wave_svc_implicit_coefficients.hpp"
#include "pde_solvers/lss_pde_discretization_config.hpp"
#include "pde_solvers/lss_wave_solver_config.hpp"
#include "pde_solvers/transformation/lss_wave_data_transform.hpp"
#include "solver_method/lss_wave_implicit_solver_method.hpp"
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
using lss_enumerations::dimension_enum;
using lss_enumerations::implicit_pde_schemes_enum;
using lss_enumerations::memory_space_enum;
using lss_enumerations::traverse_direction_enum;
using lss_enumerations::tridiagonal_method_enum;
using lss_sor_solver::sor_solver;
using lss_sor_solver_cuda::sor_solver_cuda;
using lss_thomas_lu_solver::thomas_lu_solver;
using lss_utility::NaN;
using lss_utility::pair_t;
using lss_utility::range;

/**
 * wave_time_loop object
 */
template <typename fp_type, template <typename, typename> typename container = std::vector,
          typename allocator = std::allocator<fp_type>>
class wave_time_loop
{
    typedef container<fp_type, allocator> container_t;

  public:
    template <typename solver>
    static void run(solver &solver_ptr, boundary_1d_pair<fp_type> const &boundary_pair,
                    range<fp_type> const &time_range, std::size_t const &last_time_idx, fp_type const time_step,
                    traverse_direction_enum const &traverse_dir, container_t &prev_solution_0,
                    container_t &prev_solution_1, std::function<fp_type(fp_type, fp_type)> const &wave_source,
                    container_t &next_solution);
    template <typename solver>
    static void run(solver &solver_ptr, boundary_1d_pair<fp_type> const &boundary_pair,
                    range<fp_type> const &time_range, std::size_t const &last_time_idx, fp_type const time_step,
                    traverse_direction_enum const &traverse_dir, container_t &prev_solution_0,
                    container_t &prev_solution_1, container_t &next_solution);

    template <typename solver>
    static void run_with_stepping(solver &solver_ptr, boundary_1d_pair<fp_type> const &boundary_pair,
                                  range<fp_type> const &time_range, std::size_t const &last_time_idx,
                                  fp_type const time_step, traverse_direction_enum const &traverse_dir,
                                  container_t &prev_solution_0, container_t &prev_solution_1,
                                  std::function<fp_type(fp_type, fp_type)> const &wave_source,
                                  container_t &next_solution,
                                  container_2d<by_enum::Row, fp_type, container, allocator> &solutions);
    template <typename solver>
    static void run_with_stepping(solver &solver_ptr, boundary_1d_pair<fp_type> const &boundary_pair,
                                  range<fp_type> const &time_range, std::size_t const &last_time_idx,
                                  fp_type const time_step, traverse_direction_enum const &traverse_dir,
                                  container_t &prev_solution_0, container_t &prev_solution_1,
                                  container_t &next_solution,
                                  container_2d<by_enum::Row, fp_type, container, allocator> &solutions);
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
template <typename solver>
void wave_time_loop<fp_type, container, allocator>::run(
    solver &solver_ptr, boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &time_range,
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

    if (traverse_dir == traverse_direction_enum::Forward)
    {

        // solve for initial time step:
        solver_ptr->solve_initial(prev_solution_0, prev_solution_1, boundary_pair, time, next_time, wave_source,
                                  next_solution);
        prev_solution_1 = next_solution;
        time_idx = 1;

        // solve for rest of time steps:
        time += k;
        next_time += k;
        time_idx++;
        while (time_idx <= last_time_idx)
        {
            solver_ptr->solve(prev_solution_0, prev_solution_1, boundary_pair, time, next_time, wave_source,
                              next_solution);
            prev_solution_0 = prev_solution_1;
            prev_solution_1 = next_solution;
            time += k;
            next_time += k;
            time_idx++;
        }
    }
    else
    {
        time_idx = last_time_idx;
        time = end_time;
        next_time = time - k;
        // solve for initial time step:
        solver_ptr->solve_terminal(prev_solution_0, prev_solution_1, boundary_pair, time, next_time, wave_source,
                                   next_solution);
        prev_solution_1 = next_solution;
        time_idx--;

        // solve for rest of time steps:
        time -= k;
        next_time -= k;
        do
        {
            time_idx--;
            solver_ptr->solve(prev_solution_0, prev_solution_1, boundary_pair, time, next_time, wave_source,
                              next_solution);
            prev_solution_0 = prev_solution_1;
            prev_solution_1 = next_solution;
            time -= k;
            next_time -= k;
        } while (time_idx > 0);
    }
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
template <typename solver>
void wave_time_loop<fp_type, container, allocator>::run(
    solver &solver_ptr, boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &time_range,
    std::size_t const &last_time_idx, fp_type const time_step, traverse_direction_enum const &traverse_dir,
    container_t &prev_solution_0, container_t &prev_solution_1, container_t &next_solution)
{
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type k = time_step;
    fp_type time{start_time};
    fp_type next_time{time + k};
    std::size_t time_idx{};

    if (traverse_dir == traverse_direction_enum::Forward)
    {
        // solve for initial time step:
        solver_ptr->solve_initial(prev_solution_0, prev_solution_1, boundary_pair, time, next_time, next_solution);
        prev_solution_1 = next_solution;
        time_idx = 1;

        // solve for rest of time steps:
        time += k;
        next_time += k;
        time_idx++;
        while (time_idx <= last_time_idx)
        {
            solver_ptr->solve(prev_solution_0, prev_solution_1, boundary_pair, time, next_time, next_solution);
            prev_solution_0 = prev_solution_1;
            prev_solution_1 = next_solution;
            time += k;
            next_time += k;
            time_idx++;
        }
    }
    else
    {
        time_idx = last_time_idx;
        time = end_time;
        next_time = time - k;
        // solve for initial time step:
        solver_ptr->solve_terminal(prev_solution_0, prev_solution_1, boundary_pair, time, next_time, next_solution);
        prev_solution_1 = next_solution;
        time_idx--;

        // solve for rest of time steps:
        time -= k;
        next_time -= k;
        do
        {
            time_idx--;
            solver_ptr->solve(prev_solution_0, prev_solution_1, boundary_pair, time, next_time, next_solution);
            prev_solution_0 = prev_solution_1;
            prev_solution_1 = next_solution;
            time -= k;
            next_time -= k;
        } while (time_idx > 0);
    }
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
template <typename solver>
void wave_time_loop<fp_type, container, allocator>::run_with_stepping(
    solver &solver_ptr, boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &time_range,
    std::size_t const &last_time_idx, fp_type const time_step, traverse_direction_enum const &traverse_dir,
    container_t &prev_solution_0, container_t &prev_solution_1,
    std::function<fp_type(fp_type, fp_type)> const &wave_source, container_t &next_solution,
    container_2d<by_enum::Row, fp_type, container, allocator> &solutions)
{
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type k = time_step;
    fp_type time{start_time};
    fp_type next_time{time + k};
    std::size_t time_idx{};

    if (traverse_dir == traverse_direction_enum::Forward)
    {
        // store the initial solution:
        solutions(0, prev_solution_0);
        // solve for initial time step:
        solver_ptr->solve_initial(prev_solution_0, prev_solution_1, boundary_pair, time, next_time, wave_source,
                                  next_solution);
        prev_solution_1 = next_solution;
        time_idx = 1;
        solutions(time_idx, next_solution);

        // solve for rest of time steps:
        time += k;
        next_time += k;
        time_idx++;
        while (time_idx <= last_time_idx)
        {
            solver_ptr->solve(prev_solution_0, prev_solution_1, boundary_pair, time, next_time, wave_source,
                              next_solution);
            prev_solution_0 = prev_solution_1;
            prev_solution_1 = next_solution;
            solutions(time_idx, next_solution);
            time += k;
            next_time += k;
            time_idx++;
        }
    }
    else
    {
        time_idx = last_time_idx;
        // store the terminal solution:
        solutions(last_time_idx, prev_solution_0);
        time = end_time;
        next_time = time - k;
        // solve for initial time step:
        solver_ptr->solve_terminal(prev_solution_0, prev_solution_1, boundary_pair, time, next_time, wave_source,
                                   next_solution);
        prev_solution_1 = next_solution;
        time_idx--;
        solutions(time_idx, next_solution);

        // solve for rest of time steps:
        time -= k;
        next_time -= k;
        do
        {
            time_idx--;
            solver_ptr->solve(prev_solution_0, prev_solution_1, boundary_pair, time, next_time, wave_source,
                              next_solution);
            prev_solution_0 = prev_solution_1;
            prev_solution_1 = next_solution;
            solutions(time_idx, next_solution);
            time -= k;
            next_time -= k;
        } while (time_idx > 0);
    }
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
template <typename solver>
void wave_time_loop<fp_type, container, allocator>::run_with_stepping(
    solver &solver_ptr, boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &time_range,
    std::size_t const &last_time_idx, fp_type const time_step, traverse_direction_enum const &traverse_dir,
    container_t &prev_solution_0, container_t &prev_solution_1, container_t &next_solution,
    container_2d<by_enum::Row, fp_type, container, allocator> &solutions)
{
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type k = time_step;
    fp_type time{start_time};
    fp_type next_time{time + k};
    std::size_t time_idx{};
    if (traverse_dir == traverse_direction_enum::Forward)
    {
        // store the initial solution:
        solutions(0, prev_solution_0);
        // solve for initial time step:
        solver_ptr->solve_initial(prev_solution_0, prev_solution_1, boundary_pair, time, next_time, next_solution);
        prev_solution_1 = next_solution;
        time_idx = 1;
        solutions(time_idx, next_solution);

        // solve for rest of time steps:
        time += k;
        next_time += k;
        time_idx++;
        while (time_idx <= last_time_idx)
        {
            solver_ptr->solve(prev_solution_0, prev_solution_1, boundary_pair, time, next_time, next_solution);
            prev_solution_0 = prev_solution_1;
            prev_solution_1 = next_solution;
            solutions(time_idx, next_solution);
            time += k;
            next_time += k;
            time_idx++;
        }
    }
    else
    {
        time_idx = last_time_idx;
        // store the terminal solution:
        solutions(last_time_idx, prev_solution_0);
        time = end_time;
        next_time = time - k;
        // solve for initial time step:
        solver_ptr->solve_terminal(prev_solution_0, prev_solution_1, boundary_pair, time, next_time, next_solution);
        prev_solution_1 = next_solution;
        time_idx--;
        solutions(time_idx, next_solution);

        // solve for rest of time steps:
        time -= k;
        next_time -= k;
        do
        {
            time_idx--;
            solver_ptr->solve(prev_solution_0, prev_solution_1, boundary_pair, time, next_time, next_solution);
            prev_solution_0 = prev_solution_1;
            prev_solution_1 = next_solution;
            solutions(time_idx, next_solution);
            time -= k;
            next_time -= k;
        } while (time_idx > 0);
    }
}

template <memory_space_enum memory_enum, tridiagonal_method_enum tridiagonal_method, typename fp_type,
          template <typename, typename> typename container = std::vector, typename allocator = std::allocator<fp_type>>
class general_svc_wave_equation_implicit_kernel
{
};

// ===================================================================
// ============================== DEVICE =============================
// ===================================================================
template <typename fp_type, template <typename, typename> typename container, typename allocator>
class general_svc_wave_equation_implicit_kernel<memory_space_enum::Device, tridiagonal_method_enum::CUDASolver, fp_type,
                                                container, allocator>
{
    typedef container<fp_type, allocator> container_t;
    typedef cuda_solver<memory_space_enum::Device, fp_type, container, allocator> cusolver;
    typedef wave_time_loop<fp_type, container, allocator> loop;
    typedef wave_implicit_solver_method<fp_type, sptr_t<cusolver>, container, allocator> solver_method;

  private:
    boundary_1d_pair<fp_type> boundary_pair_;
    wave_data_transform_1d_ptr<fp_type> wave_data_cfg_;
    pde_discretization_config_1d_ptr<fp_type> discretization_cfg_;
    wave_implicit_solver_config_ptr solver_cfg_;
    grid_config_1d_ptr<fp_type> grid_cfg_;

  public:
    general_svc_wave_equation_implicit_kernel(boundary_1d_pair<fp_type> const &boundary_pair,
                                              wave_data_transform_1d_ptr<fp_type> const &wave_data_config,
                                              pde_discretization_config_1d_ptr<fp_type> const &discretization_config,
                                              wave_implicit_solver_config_ptr const &solver_config,
                                              grid_config_1d_ptr<fp_type> const &grid_config)
        : boundary_pair_{boundary_pair}, wave_data_cfg_{wave_data_config}, discretization_cfg_{discretization_config},
          solver_cfg_{solver_config}, grid_cfg_{grid_config}
    {
    }

    void operator()(container_t &prev_solution_0, container_t &prev_solution_1, container_t &next_solution,
                    bool is_wave_sourse_set, std::function<fp_type(fp_type, fp_type)> const &wave_source)
    {
        // get space range:
        const range<fp_type> space = discretization_cfg_->space_range();
        // get time range:
        const range<fp_type> time = discretization_cfg_->time_range();
        // time step:
        const fp_type k = discretization_cfg_->time_step();
        // size of space discretization:
        const std::size_t space_size = discretization_cfg_->number_of_space_points();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        // save traverse_direction
        const traverse_direction_enum traverse_dir = solver_cfg_->traverse_direction();
        // create a wave coefficient holder:
        auto const wave_coeff_holder =
            std::make_shared<wave_svc_implicit_coefficients<fp_type>>(wave_data_cfg_, discretization_cfg_);
        // create and set up the solver:
        auto const &solver = std::make_shared<cusolver>(space, space_size);
        solver->set_factorization(solver_cfg_->tridiagonal_factorization());
        auto const &solver_method_ptr = std::make_shared<solver_method>(solver, wave_coeff_holder, grid_cfg_);
        if (is_wave_sourse_set)
        {
            loop::run(solver_method_ptr, boundary_pair_, time, last_time_idx, k, traverse_dir, prev_solution_0,
                      prev_solution_1, wave_source, next_solution);
        }
        else
        {
            loop::run(solver_method_ptr, boundary_pair_, time, last_time_idx, k, traverse_dir, prev_solution_0,
                      prev_solution_1, next_solution);
        }
    }

    void operator()(container_t &prev_solution_0, container_t &prev_solution_1, container_t &next_solution,
                    bool is_wave_sourse_set, std::function<fp_type(fp_type, fp_type)> const &wave_source,
                    container_2d<by_enum::Row, fp_type, container, allocator> &solutions)
    {
        // get space range:
        const range<fp_type> space = discretization_cfg_->space_range();
        // get time range:
        const range<fp_type> time = discretization_cfg_->time_range();
        // time step:
        const fp_type k = discretization_cfg_->time_step();
        // size of space discretization:
        const std::size_t space_size = discretization_cfg_->number_of_space_points();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        // save traverse_direction
        const traverse_direction_enum traverse_dir = solver_cfg_->traverse_direction();
        // create a wave coefficient holder:
        auto const wave_coeff_holder =
            std::make_shared<wave_svc_implicit_coefficients<fp_type>>(wave_data_cfg_, discretization_cfg_);
        // create and set up the solver:
        auto const &solver = std::make_shared<cusolver>(space, space_size);
        solver->set_factorization(solver_cfg_->tridiagonal_factorization());
        auto const &solver_method_ptr = std::make_shared<solver_method>(solver, wave_coeff_holder, grid_cfg_);
        if (is_wave_sourse_set)
        {
            loop::run_with_stepping(solver_method_ptr, boundary_pair_, time, last_time_idx, k, traverse_dir,
                                    prev_solution_0, prev_solution_1, wave_source, next_solution, solutions);
        }
        else
        {
            loop::run_with_stepping(solver_method_ptr, boundary_pair_, time, last_time_idx, k, traverse_dir,
                                    prev_solution_0, prev_solution_1, next_solution, solutions);
        }
    }
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class general_svc_wave_equation_implicit_kernel<memory_space_enum::Device, tridiagonal_method_enum::SORSolver, fp_type,
                                                container, allocator>
{
    typedef container<fp_type, allocator> container_t;
    typedef sor_solver_cuda<fp_type, container, allocator> sorcusolver;
    typedef wave_time_loop<fp_type, container, allocator> loop;
    typedef wave_implicit_solver_method<fp_type, sptr_t<sorcusolver>, container, allocator> solver_method;

  private:
    boundary_1d_pair<fp_type> boundary_pair_;
    wave_data_transform_1d_ptr<fp_type> wave_data_cfg_;
    pde_discretization_config_1d_ptr<fp_type> discretization_cfg_;
    wave_implicit_solver_config_ptr solver_cfg_;
    grid_config_1d_ptr<fp_type> grid_cfg_;

  public:
    general_svc_wave_equation_implicit_kernel(boundary_1d_pair<fp_type> const &boundary_pair,
                                              wave_data_transform_1d_ptr<fp_type> const &wave_data_config,
                                              pde_discretization_config_1d_ptr<fp_type> const &discretization_config,
                                              wave_implicit_solver_config_ptr const &solver_config,
                                              grid_config_1d_ptr<fp_type> const &grid_config)
        : boundary_pair_{boundary_pair}, wave_data_cfg_{wave_data_config}, discretization_cfg_{discretization_config},
          solver_cfg_{solver_config}, grid_cfg_{grid_config}
    {
    }

    void operator()(container_t &prev_solution_0, container_t &prev_solution_1, container_t &next_solution,
                    bool is_wave_sourse_set, std::function<fp_type(fp_type, fp_type)> const &wave_source,
                    fp_type omega_value)
    {
        // get space range:
        const range<fp_type> space = discretization_cfg_->space_range();
        // get time range:
        const range<fp_type> time = discretization_cfg_->time_range();
        // time step:
        const fp_type k = discretization_cfg_->time_step();
        // size of space discretization:
        const std::size_t space_size = discretization_cfg_->number_of_space_points();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        // save traverse_direction
        const traverse_direction_enum traverse_dir = solver_cfg_->traverse_direction();
        // create a wave coefficient holder:
        auto const wave_coeff_holder =
            std::make_shared<wave_svc_implicit_coefficients<fp_type>>(wave_data_cfg_, discretization_cfg_);
        // create and set up the solver:
        auto const &solver = std::make_shared<sorcusolver>(space, space_size);
        solver->set_omega(omega_value);
        auto const &solver_method_ptr = std::make_shared<solver_method>(solver, wave_coeff_holder, grid_cfg_);
        if (is_wave_sourse_set)
        {
            loop::run(solver_method_ptr, boundary_pair_, time, last_time_idx, k, traverse_dir, prev_solution_0,
                      prev_solution_1, wave_source, next_solution);
        }
        else
        {
            loop::run(solver_method_ptr, boundary_pair_, time, last_time_idx, k, traverse_dir, prev_solution_0,
                      prev_solution_1, next_solution);
        }
    }

    void operator()(container_t &prev_solution_0, container_t &prev_solution_1, container_t &next_solution,
                    bool is_wave_sourse_set, std::function<fp_type(fp_type, fp_type)> const &wave_source,
                    fp_type omega_value, container_2d<by_enum::Row, fp_type, container, allocator> &solutions)
    {
        // get space range:
        const range<fp_type> space = discretization_cfg_->space_range();
        // get time range:
        const range<fp_type> time = discretization_cfg_->time_range();
        // time step:
        const fp_type k = discretization_cfg_->time_step();
        // size of space discretization:
        const std::size_t space_size = discretization_cfg_->number_of_space_points();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        // save traverse_direction
        const traverse_direction_enum traverse_dir = solver_cfg_->traverse_direction();
        // create a wave coefficient holder:
        auto const wave_coeff_holder =
            std::make_shared<wave_svc_implicit_coefficients<fp_type>>(wave_data_cfg_, discretization_cfg_);
        // create and set up the solver:
        auto const &solver = std::make_shared<sorcusolver>(space, space_size);
        solver->set_omega(omega_value);
        auto const &solver_method_ptr = std::make_shared<solver_method>(solver, wave_coeff_holder, grid_cfg_);
        if (is_wave_sourse_set)
        {
            loop::run_with_stepping(solver_method_ptr, boundary_pair_, time, last_time_idx, k, traverse_dir,
                                    prev_solution_0, prev_solution_1, wave_source, next_solution, solutions);
        }
        else
        {
            loop::run_with_stepping(solver_method_ptr, boundary_pair_, time, last_time_idx, k, traverse_dir,
                                    prev_solution_0, prev_solution_1, next_solution, solutions);
        }
    }
};

// ===================================================================
// ================================ HOST =============================
// ===================================================================
template <typename fp_type, template <typename, typename> typename container, typename allocator>
class general_svc_wave_equation_implicit_kernel<memory_space_enum::Host, tridiagonal_method_enum::CUDASolver, fp_type,
                                                container, allocator>
{
    typedef container<fp_type, allocator> container_t;
    typedef cuda_solver<memory_space_enum::Host, fp_type, container, allocator> cusolver;
    typedef wave_time_loop<fp_type, container, allocator> loop;
    typedef wave_implicit_solver_method<fp_type, sptr_t<cusolver>, container, allocator> solver_method;

  private:
    boundary_1d_pair<fp_type> boundary_pair_;
    wave_data_transform_1d_ptr<fp_type> wave_data_cfg_;
    pde_discretization_config_1d_ptr<fp_type> discretization_cfg_;
    wave_implicit_solver_config_ptr solver_cfg_;
    grid_config_1d_ptr<fp_type> grid_cfg_;

  public:
    general_svc_wave_equation_implicit_kernel(boundary_1d_pair<fp_type> const &boundary_pair,
                                              wave_data_transform_1d_ptr<fp_type> const &wave_data_config,
                                              pde_discretization_config_1d_ptr<fp_type> const &discretization_config,
                                              wave_implicit_solver_config_ptr const &solver_config,
                                              grid_config_1d_ptr<fp_type> const &grid_config)
        : boundary_pair_{boundary_pair}, wave_data_cfg_{wave_data_config}, discretization_cfg_{discretization_config},
          solver_cfg_{solver_config}, grid_cfg_{grid_config}
    {
    }

    void operator()(container_t &prev_solution_0, container_t &prev_solution_1, container_t &next_solution,
                    bool is_wave_sourse_set, std::function<fp_type(fp_type, fp_type)> const &wave_source)
    {
        // get space range:
        const range<fp_type> space = discretization_cfg_->space_range();
        // get time range:
        const range<fp_type> time = discretization_cfg_->time_range();
        // time step:
        const fp_type k = discretization_cfg_->time_step();
        // size of space discretization:
        const std::size_t space_size = discretization_cfg_->number_of_space_points();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        // save traverse_direction
        const traverse_direction_enum traverse_dir = solver_cfg_->traverse_direction();
        // create a wave coefficient holder:
        auto const wave_coeff_holder =
            std::make_shared<wave_svc_implicit_coefficients<fp_type>>(wave_data_cfg_, discretization_cfg_);
        // create and set up the solver:
        auto const &solver = std::make_shared<cusolver>(space, space_size);
        solver->set_factorization(solver_cfg_->tridiagonal_factorization());
        auto const &solver_method_ptr = std::make_shared<solver_method>(solver, wave_coeff_holder, grid_cfg_);
        if (is_wave_sourse_set)
        {
            loop::run(solver_method_ptr, boundary_pair_, time, last_time_idx, k, traverse_dir, prev_solution_0,
                      prev_solution_1, wave_source, next_solution);
        }
        else
        {
            loop::run(solver_method_ptr, boundary_pair_, time, last_time_idx, k, traverse_dir, prev_solution_0,
                      prev_solution_1, next_solution);
        }
    }

    void operator()(container_t &prev_solution_0, container_t &prev_solution_1, container_t &next_solution,
                    bool is_wave_sourse_set, std::function<fp_type(fp_type, fp_type)> const &wave_source,
                    container_2d<by_enum::Row, fp_type, container, allocator> &solutions)
    {
        // get space range:
        const range<fp_type> space = discretization_cfg_->space_range();
        // get time range:
        const range<fp_type> time = discretization_cfg_->time_range();
        // time step:
        const fp_type k = discretization_cfg_->time_step();
        // size of space discretization:
        const std::size_t space_size = discretization_cfg_->number_of_space_points();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        // save traverse_direction
        const traverse_direction_enum traverse_dir = solver_cfg_->traverse_direction();
        // create a wave coefficient holder:
        auto const wave_coeff_holder =
            std::make_shared<wave_svc_implicit_coefficients<fp_type>>(wave_data_cfg_, discretization_cfg_);
        // create and set up the solver:
        auto const &solver = std::make_shared<cusolver>(space, space_size);
        solver->set_factorization(solver_cfg_->tridiagonal_factorization());
        auto const &solver_method_ptr = std::make_shared<solver_method>(solver, wave_coeff_holder, grid_cfg_);
        if (is_wave_sourse_set)
        {
            loop::run_with_stepping(solver_method_ptr, boundary_pair_, time, last_time_idx, k, traverse_dir,
                                    prev_solution_0, prev_solution_1, wave_source, next_solution, solutions);
        }
        else
        {
            loop::run_with_stepping(solver_method_ptr, boundary_pair_, time, last_time_idx, k, traverse_dir,
                                    prev_solution_0, prev_solution_1, next_solution, solutions);
        }
    }
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class general_svc_wave_equation_implicit_kernel<memory_space_enum::Host, tridiagonal_method_enum::SORSolver, fp_type,
                                                container, allocator>
{
    typedef container<fp_type, allocator> container_t;
    typedef sor_solver<fp_type, container, allocator> sorsolver;
    typedef wave_time_loop<fp_type, container, allocator> loop;
    typedef wave_implicit_solver_method<fp_type, sptr_t<sorsolver>, container, allocator> solver_method;

  private:
    boundary_1d_pair<fp_type> boundary_pair_;
    wave_data_transform_1d_ptr<fp_type> wave_data_cfg_;
    pde_discretization_config_1d_ptr<fp_type> discretization_cfg_;
    wave_implicit_solver_config_ptr solver_cfg_;
    grid_config_1d_ptr<fp_type> grid_cfg_;

  public:
    general_svc_wave_equation_implicit_kernel(boundary_1d_pair<fp_type> const &boundary_pair,
                                              wave_data_transform_1d_ptr<fp_type> const &wave_data_config,
                                              pde_discretization_config_1d_ptr<fp_type> const &discretization_config,
                                              wave_implicit_solver_config_ptr const &solver_config,
                                              grid_config_1d_ptr<fp_type> const &grid_config)
        : boundary_pair_{boundary_pair}, wave_data_cfg_{wave_data_config}, discretization_cfg_{discretization_config},
          solver_cfg_{solver_config}, grid_cfg_{grid_config}
    {
    }

    void operator()(container_t &prev_solution_0, container_t &prev_solution_1, container_t &next_solution,
                    bool is_wave_sourse_set, std::function<fp_type(fp_type, fp_type)> const &wave_source,
                    fp_type omega_value)
    {
        // get space range:
        const range<fp_type> space = discretization_cfg_->space_range();
        // get time range:
        const range<fp_type> time = discretization_cfg_->time_range();
        // get space step:
        const fp_type k = discretization_cfg_->time_step();
        // size of space discretization:
        const std::size_t space_size = discretization_cfg_->number_of_space_points();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        // save traverse_direction
        const traverse_direction_enum traverse_dir = solver_cfg_->traverse_direction();
        // create a wave coefficient holder:
        auto const wave_coeff_holder =
            std::make_shared<wave_svc_implicit_coefficients<fp_type>>(wave_data_cfg_, discretization_cfg_);
        // create and set up the solver:
        auto const &solver = std::make_shared<sorsolver>(space, space_size);
        solver->set_omega(omega_value);
        auto const &solver_method_ptr = std::make_shared<solver_method>(solver, wave_coeff_holder, grid_cfg_);
        if (is_wave_sourse_set)
        {
            loop::run(solver_method_ptr, boundary_pair_, time, last_time_idx, k, traverse_dir, prev_solution_0,
                      prev_solution_1, wave_source, next_solution);
        }
        else
        {
            loop::run(solver_method_ptr, boundary_pair_, time, last_time_idx, k, traverse_dir, prev_solution_0,
                      prev_solution_1, next_solution);
        }
    }

    void operator()(container_t &prev_solution_0, container_t &prev_solution_1, container_t &next_solution,
                    bool is_wave_sourse_set, std::function<fp_type(fp_type, fp_type)> const &wave_source,
                    fp_type omega_value, container_2d<by_enum::Row, fp_type, container, allocator> &solutions)
    {
        // get space range:
        const range<fp_type> space = discretization_cfg_->space_range();
        // get time range:
        const range<fp_type> time = discretization_cfg_->time_range();
        // time step:
        const fp_type k = discretization_cfg_->time_step();
        // size of space discretization:
        const std::size_t space_size = discretization_cfg_->number_of_space_points();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        // save traverse_direction
        const traverse_direction_enum traverse_dir = solver_cfg_->traverse_direction();
        // create a wave coefficient holder:
        auto const wave_coeff_holder =
            std::make_shared<wave_svc_implicit_coefficients<fp_type>>(wave_data_cfg_, discretization_cfg_);
        // create and set up the solver:
        auto const &solver = std::make_shared<sorsolver>(space, space_size);
        solver->set_omega(omega_value);
        auto const &solver_method_ptr = std::make_shared<solver_method>(solver, wave_coeff_holder, grid_cfg_);
        if (is_wave_sourse_set)
        {
            loop::run_with_stepping(solver_method_ptr, boundary_pair_, time, last_time_idx, k, traverse_dir,
                                    prev_solution_0, prev_solution_1, wave_source, next_solution, solutions);
        }
        else
        {
            loop::run_with_stepping(solver_method_ptr, boundary_pair_, time, last_time_idx, k, traverse_dir,
                                    prev_solution_0, prev_solution_1, next_solution, solutions);
        }
    }
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class general_svc_wave_equation_implicit_kernel<memory_space_enum::Host, tridiagonal_method_enum::DoubleSweepSolver,
                                                fp_type, container, allocator>
{
    typedef container<fp_type, allocator> container_t;
    typedef double_sweep_solver<fp_type, container, allocator> ds_solver;
    typedef wave_time_loop<fp_type, container, allocator> loop;
    typedef wave_implicit_solver_method<fp_type, sptr_t<ds_solver>, container, allocator> solver_method;

  private:
    boundary_1d_pair<fp_type> boundary_pair_;
    wave_data_transform_1d_ptr<fp_type> wave_data_cfg_;
    pde_discretization_config_1d_ptr<fp_type> discretization_cfg_;
    wave_implicit_solver_config_ptr solver_cfg_;
    grid_config_1d_ptr<fp_type> grid_cfg_;

  public:
    general_svc_wave_equation_implicit_kernel(boundary_1d_pair<fp_type> const &boundary_pair,
                                              wave_data_transform_1d_ptr<fp_type> const &wave_data_config,
                                              pde_discretization_config_1d_ptr<fp_type> const &discretization_config,
                                              wave_implicit_solver_config_ptr const &solver_config,
                                              grid_config_1d_ptr<fp_type> const &grid_config)
        : boundary_pair_{boundary_pair}, wave_data_cfg_{wave_data_config}, discretization_cfg_{discretization_config},
          solver_cfg_{solver_config}, grid_cfg_{grid_config}
    {
    }

    void operator()(container_t &prev_solution_0, container_t &prev_solution_1, container_t &next_solution,
                    bool is_wave_sourse_set, std::function<fp_type(fp_type, fp_type)> const &wave_source)
    {
        // get space range:
        const range<fp_type> space = discretization_cfg_->space_range();
        // get time range:
        const range<fp_type> time = discretization_cfg_->time_range();
        // time step:
        const fp_type k = discretization_cfg_->time_step();
        // size of space discretization:
        const std::size_t space_size = discretization_cfg_->number_of_space_points();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        // save traverse_direction
        const traverse_direction_enum traverse_dir = solver_cfg_->traverse_direction();
        // create a wave coefficient holder:
        auto const wave_coeff_holder =
            std::make_shared<wave_svc_implicit_coefficients<fp_type>>(wave_data_cfg_, discretization_cfg_);
        // create and set up the solver:
        auto const &solver = std::make_shared<ds_solver>(space, space_size);
        auto const &solver_method_ptr = std::make_shared<solver_method>(solver, wave_coeff_holder, grid_cfg_);
        if (is_wave_sourse_set)
        {
            loop::run(solver_method_ptr, boundary_pair_, time, last_time_idx, k, traverse_dir, prev_solution_0,
                      prev_solution_1, wave_source, next_solution);
        }
        else
        {
            loop::run(solver_method_ptr, boundary_pair_, time, last_time_idx, k, traverse_dir, prev_solution_0,
                      prev_solution_1, next_solution);
        }
    }

    void operator()(container_t &prev_solution_0, container_t &prev_solution_1, container_t &next_solution,
                    bool is_wave_sourse_set, std::function<fp_type(fp_type, fp_type)> const &wave_source,
                    container_2d<by_enum::Row, fp_type, container, allocator> &solutions)
    {
        // get space range:
        const range<fp_type> space = discretization_cfg_->space_range();
        // get time range:
        const range<fp_type> time = discretization_cfg_->time_range();
        // time step:
        const fp_type k = discretization_cfg_->time_step();
        // size of space discretization:
        const std::size_t space_size = discretization_cfg_->number_of_space_points();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        // save traverse_direction
        const traverse_direction_enum traverse_dir = solver_cfg_->traverse_direction();
        // create a wave coefficient holder:
        auto const wave_coeff_holder =
            std::make_shared<wave_svc_implicit_coefficients<fp_type>>(wave_data_cfg_, discretization_cfg_);
        // create and set up the solver:
        auto const &solver = std::make_shared<ds_solver>(space, space_size);
        auto const &solver_method_ptr = std::make_shared<solver_method>(solver, wave_coeff_holder, grid_cfg_);
        if (is_wave_sourse_set)
        {
            loop::run_with_stepping(solver_method_ptr, boundary_pair_, time, last_time_idx, k, traverse_dir,
                                    prev_solution_0, prev_solution_1, wave_source, next_solution, solutions);
        }
        else
        {
            loop::run_with_stepping(solver_method_ptr, boundary_pair_, time, last_time_idx, k, traverse_dir,
                                    prev_solution_0, prev_solution_1, next_solution, solutions);
        }
    }
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class general_svc_wave_equation_implicit_kernel<memory_space_enum::Host, tridiagonal_method_enum::ThomasLUSolver,
                                                fp_type, container, allocator>
{
    typedef container<fp_type, allocator> container_t;
    typedef thomas_lu_solver<fp_type, container, allocator> tlu_solver;
    typedef wave_time_loop<fp_type, container, allocator> loop;
    typedef wave_implicit_solver_method<fp_type, sptr_t<tlu_solver>, container, allocator> solver_method;

  private:
    boundary_1d_pair<fp_type> boundary_pair_;
    wave_data_transform_1d_ptr<fp_type> wave_data_cfg_;
    pde_discretization_config_1d_ptr<fp_type> discretization_cfg_;
    wave_implicit_solver_config_ptr solver_cfg_;
    grid_config_1d_ptr<fp_type> grid_cfg_;

  public:
    general_svc_wave_equation_implicit_kernel(boundary_1d_pair<fp_type> const &boundary_pair,
                                              wave_data_transform_1d_ptr<fp_type> const &wave_data_config,
                                              pde_discretization_config_1d_ptr<fp_type> const &discretization_config,
                                              wave_implicit_solver_config_ptr const &solver_config,
                                              grid_config_1d_ptr<fp_type> const &grid_config)
        : boundary_pair_{boundary_pair}, wave_data_cfg_{wave_data_config}, discretization_cfg_{discretization_config},
          solver_cfg_{solver_config}, grid_cfg_{grid_config}
    {
    }

    void operator()(container_t &prev_solution_0, container_t &prev_solution_1, container_t &next_solution,
                    bool is_wave_sourse_set, std::function<fp_type(fp_type, fp_type)> const &wave_source)
    {
        // get space range:
        const range<fp_type> space = discretization_cfg_->space_range();
        // get time range:
        const range<fp_type> time = discretization_cfg_->time_range();
        // time step:
        const fp_type k = discretization_cfg_->time_step();
        // size of space discretization:
        const std::size_t space_size = discretization_cfg_->number_of_space_points();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        // save traverse_direction
        const traverse_direction_enum traverse_dir = solver_cfg_->traverse_direction();
        // create a wave coefficient holder:
        auto const wave_coeff_holder =
            std::make_shared<wave_svc_implicit_coefficients<fp_type>>(wave_data_cfg_, discretization_cfg_);
        // create and set up the solver:
        auto const &solver = std::make_shared<tlu_solver>(space, space_size);
        auto const &solver_method_ptr = std::make_shared<solver_method>(solver, wave_coeff_holder, grid_cfg_);
        if (is_wave_sourse_set)
        {
            loop::run(solver_method_ptr, boundary_pair_, time, last_time_idx, k, traverse_dir, prev_solution_0,
                      prev_solution_1, wave_source, next_solution);
        }
        else
        {
            loop::run(solver_method_ptr, boundary_pair_, time, last_time_idx, k, traverse_dir, prev_solution_0,
                      prev_solution_1, next_solution);
        }
    }

    void operator()(container_t &prev_solution_0, container_t &prev_solution_1, container_t &next_solution,
                    bool is_wave_sourse_set, std::function<fp_type(fp_type, fp_type)> const &wave_source,
                    container_2d<by_enum::Row, fp_type, container, allocator> &solutions)
    {
        // get space range:
        const range<fp_type> space = discretization_cfg_->space_range();
        // get time range:
        const range<fp_type> time = discretization_cfg_->time_range();
        // time step:
        const fp_type k = discretization_cfg_->time_step();
        // size of space discretization:
        const std::size_t space_size = discretization_cfg_->number_of_space_points();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        // save traverse_direction
        const traverse_direction_enum traverse_dir = solver_cfg_->traverse_direction();
        // create a wave coefficient holder:
        auto const wave_coeff_holder =
            std::make_shared<wave_svc_implicit_coefficients<fp_type>>(wave_data_cfg_, discretization_cfg_);
        // create and set up the solver:
        auto const &solver = std::make_shared<tlu_solver>(space, space_size);
        auto const &solver_method_ptr = std::make_shared<solver_method>(solver, wave_coeff_holder, grid_cfg_);
        if (is_wave_sourse_set)
        {
            loop::run_with_stepping(solver_method_ptr, boundary_pair_, time, last_time_idx, k, traverse_dir,
                                    prev_solution_0, prev_solution_1, wave_source, next_solution, solutions);
        }
        else
        {
            loop::run_with_stepping(solver_method_ptr, boundary_pair_, time, last_time_idx, k, traverse_dir,
                                    prev_solution_0, prev_solution_1, next_solution, solutions);
        }
    }
};
} // namespace one_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_1D_GENERAL_SVC_WAVE_EQUATION_IMPLICIT_KERNEL_HPP_
