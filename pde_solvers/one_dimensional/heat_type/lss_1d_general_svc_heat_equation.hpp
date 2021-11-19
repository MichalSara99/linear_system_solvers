#if !defined(_LSS_1D_GENERAL_SVC_HEAT_EQUATION_HPP_)
#define _LSS_1D_GENERAL_SVC_HEAT_EQUATION_HPP_

#include <functional>
#include <map>

#include "boundaries/lss_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "containers/lss_container_2d.hpp"
#include "discretization/lss_discretization.hpp"
#include "discretization/lss_grid.hpp"
#include "discretization/lss_grid_config.hpp"
#include "discretization/lss_grid_config_hints.hpp"
#include "lss_1d_general_svc_heat_equation_explicit_kernel.hpp"
#include "lss_1d_general_svc_heat_equation_implicit_kernel.hpp"
#include "pde_solvers/lss_heat_data_config.hpp"
#include "pde_solvers/lss_heat_solver_config.hpp"
#include "pde_solvers/lss_pde_discretization_config.hpp"

namespace lss_pde_solvers
{

namespace one_dimensional
{
using lss_boundary::boundary_1d_pair;
using lss_boundary::boundary_1d_ptr;
using lss_containers::container_2d;
using lss_enumerations::grid_enum;
using lss_grids::grid_config_hints_1d_ptr;
using lss_grids::nonuniform_grid_config_1d;
using lss_grids::uniform_grid_config_1d;

namespace implicit_solvers
{

/*!
============================================================================
Represents general spacial variable coefficient 1D heat equation solver

u_t = a(x)*u_xx + b(x)*u_x + c(x)*u + F(x,t),
x_1 < x < x_2
t_1 < t < t_2

with initial condition:

u(x,t_1) = f(x)

or terminal condition:

u(x,t_2) = f(x)


// ============================================================================
*/
template <typename fp_type, template <typename, typename> typename container = std::vector,
          typename allocator = std::allocator<fp_type>>
class general_svc_heat_equation
{

  private:
    boundary_1d_pair<fp_type> boundary_pair_;
    heat_data_config_1d_ptr<fp_type> heat_data_cfg_;
    pde_discretization_config_1d_ptr<fp_type> discretization_cfg_;
    grid_config_hints_1d_ptr<fp_type> grid_hints_cfg_;
    heat_implicit_solver_config_ptr solver_cfg_;
    std::map<std::string, fp_type> solver_config_details_;

    explicit general_svc_heat_equation() = delete;

    void initialize()
    {
        LSS_VERIFY(heat_data_cfg_, "heat_data_config must not be null");
        LSS_VERIFY(discretization_cfg_, "discretization_config must not be null");
        LSS_VERIFY(std::get<0>(boundary_pair_), "boundary_pair.first must not be null");
        LSS_VERIFY(std::get<1>(boundary_pair_), "boundary_pair.second must not be null");
        LSS_VERIFY(solver_cfg_, "solver_config must not be null");
        LSS_VERIFY(grid_hints_cfg_, "grid_config_hints must not be null");
        if (!solver_config_details_.empty())
        {
            auto const &it = solver_config_details_.find("sor_omega");
            LSS_ASSERT(it != solver_config_details_.end(), "sor_omega is not defined");
        }
    }

  public:
    explicit general_svc_heat_equation(
        heat_data_config_1d_ptr<fp_type> const &heat_data_config,
        pde_discretization_config_1d_ptr<fp_type> const &discretization_config,
        boundary_1d_pair<fp_type> const &boundary_pair, grid_config_hints_1d_ptr<fp_type> const &grid_config_hints,
        heat_implicit_solver_config_ptr const &solver_config =
            default_heat_solver_configs::host_fwd_dssolver_euler_solver_config_ptr,
        std::map<std::string, fp_type> const &solver_config_details = std::map<std::string, fp_type>())
        : heat_data_cfg_{heat_data_config}, discretization_cfg_{discretization_config}, boundary_pair_{boundary_pair},
          grid_hints_cfg_{grid_config_hints}, solver_cfg_{solver_config}, solver_config_details_{solver_config_details}
    {
        initialize();
    }

    ~general_svc_heat_equation()
    {
    }

    general_svc_heat_equation(general_svc_heat_equation const &) = delete;
    general_svc_heat_equation(general_svc_heat_equation &&) = delete;
    general_svc_heat_equation &operator=(general_svc_heat_equation const &) = delete;
    general_svc_heat_equation &operator=(general_svc_heat_equation &&) = delete;

    /**
     * Get the final solution of the PDE
     *
     * \param solution - container for solution
     */
    void solve(container<fp_type, allocator> &solution);

    /**
     * Get all solutions in time (surface) of the PDE
     *
     * \param solutions - 2D container for all the solutions in time
     */
    void solve(container_2d<by_enum::Row, fp_type, container, allocator> &solutions);
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void general_svc_heat_equation<fp_type, container, allocator>::solve(container<fp_type, allocator> &solution)
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container<fp_type, allocator> container_t;

    LSS_ASSERT(solution.size() > 0, "The input solution container must be initialized");

    // get space range:
    const range<fp_type> space = discretization_cfg_->space_range();
    // get space step:
    const fp_type h = discretization_cfg_->space_step();
    // time step:
    const fp_type k = discretization_cfg_->time_step();
    // size of space discretization:
    const std::size_t space_size = discretization_cfg_->number_of_space_points();
    // This is the proper size of the container:
    LSS_ASSERT(solution.size() == space_size, "The input solution container must have the correct size");
    grid_config_1d_ptr<fp_type> grid_cfg;
    if (grid_hints_cfg_->grid() == grid_enum::Uniform)
    {
        grid_cfg = std::make_shared<uniform_grid_config_1d<fp_type>>(discretization_cfg_);
    }
    else if (grid_hints_cfg_->grid() == grid_enum::Nonuniform)
    {
        grid_cfg = std::make_shared<nonuniform_grid_config_1d<fp_type>>(
            discretization_cfg_, grid_hints_cfg_->strike(), grid_hints_cfg_->p_scale(), grid_hints_cfg_->c_scale());
    }
    else
    {
        throw std::exception("Unreachable");
    }
    // create container to carry previous solution:
    container_t prev_sol(space_size, fp_type{});
    //  create container to carry new solution:
    container_t next_sol(space_size, fp_type{});
    // discretize initial condition
    d_1d::of_function(grid_cfg, heat_data_cfg_->initial_condition(), prev_sol);
    // get heat_source:
    const bool is_heat_source_set = heat_data_cfg_->is_heat_source_set();
    // get heat_source:
    auto const &heat_source = heat_data_cfg_->heat_source();

    if (solver_cfg_->memory_space() == memory_space_enum::Device)
    {
        if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::CUDASolver)
        {
            typedef general_svc_heat_equation_implicit_kernel<
                memory_space_enum::Device, tridiagonal_method_enum::CUDASolver, fp_type, container, allocator>
                dev_cu_solver;

            dev_cu_solver solver(boundary_pair_, heat_data_cfg_, discretization_cfg_, solver_cfg_, grid_cfg);
            solver(prev_sol, next_sol, is_heat_source_set, heat_source);
            std::copy(prev_sol.begin(), prev_sol.end(), solution.begin());
        }
        else if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::SORSolver)
        {
            typedef general_svc_heat_equation_implicit_kernel<
                memory_space_enum::Device, tridiagonal_method_enum::SORSolver, fp_type, container, allocator>
                dev_sor_solver;
            LSS_ASSERT(!solver_config_details_.empty(), "solver_config_details map must not be empty");
            fp_type omega_value = solver_config_details_["sor_omega"];

            dev_sor_solver solver(boundary_pair_, heat_data_cfg_, discretization_cfg_, solver_cfg_, grid_cfg);
            solver(prev_sol, next_sol, is_heat_source_set, heat_source, omega_value);
            std::copy(prev_sol.begin(), prev_sol.end(), solution.begin());
        }
        else
        {
            throw std::exception("Not supported on Device");
        }
    }
    else if (solver_cfg_->memory_space() == memory_space_enum::Host)
    {
        if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::CUDASolver)
        {
            typedef general_svc_heat_equation_implicit_kernel<
                memory_space_enum::Host, tridiagonal_method_enum::CUDASolver, fp_type, container, allocator>
                host_cu_solver;

            host_cu_solver solver(boundary_pair_, heat_data_cfg_, discretization_cfg_, solver_cfg_, grid_cfg);
            solver(prev_sol, next_sol, is_heat_source_set, heat_source);
            std::copy(prev_sol.begin(), prev_sol.end(), solution.begin());
        }
        else if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::SORSolver)
        {
            typedef general_svc_heat_equation_implicit_kernel<
                memory_space_enum::Host, tridiagonal_method_enum::SORSolver, fp_type, container, allocator>
                host_sor_solver;

            LSS_ASSERT(!solver_config_details_.empty(), "solver_config_details map must not be empty");
            fp_type omega_value = solver_config_details_["sor_omega"];

            host_sor_solver solver(boundary_pair_, heat_data_cfg_, discretization_cfg_, solver_cfg_, grid_cfg);
            solver(prev_sol, next_sol, is_heat_source_set, heat_source, omega_value);
            std::copy(prev_sol.begin(), prev_sol.end(), solution.begin());
        }
        else if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::DoubleSweepSolver)
        {
            typedef general_svc_heat_equation_implicit_kernel<
                memory_space_enum::Host, tridiagonal_method_enum::DoubleSweepSolver, fp_type, container, allocator>
                host_dss_solver;

            host_dss_solver solver(boundary_pair_, heat_data_cfg_, discretization_cfg_, solver_cfg_, grid_cfg);
            solver(prev_sol, next_sol, is_heat_source_set, heat_source);
            std::copy(prev_sol.begin(), prev_sol.end(), solution.begin());
        }
        else if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::ThomasLUSolver)
        {
            typedef general_svc_heat_equation_implicit_kernel<
                memory_space_enum::Host, tridiagonal_method_enum::ThomasLUSolver, fp_type, container, allocator>
                host_lus_solver;

            host_lus_solver solver(boundary_pair_, heat_data_cfg_, discretization_cfg_, solver_cfg_, grid_cfg);
            solver(prev_sol, next_sol, is_heat_source_set, heat_source);
            std::copy(prev_sol.begin(), prev_sol.end(), solution.begin());
        }
        else
        {
            throw std::exception("Not supported on Host");
        }
    }
    else
    {
        throw std::exception("Unreachable");
    }
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void general_svc_heat_equation<fp_type, container, allocator>::solve(
    container_2d<by_enum::Row, fp_type, container, allocator> &solutions)
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container<fp_type, allocator> container_t;

    LSS_ASSERT(((solutions.columns() > 0) && (solutions.rows() > 0)),
               "The input solution 2D container must be initialized");

    // get space range:
    const range<fp_type> space = discretization_cfg_->space_range();
    // get space step:
    const fp_type h = discretization_cfg_->space_step();
    // time step:
    const fp_type k = discretization_cfg_->time_step();
    // size of space discretization:
    const std::size_t space_size = discretization_cfg_->number_of_space_points();
    // size of time discretization:
    const std::size_t time_size = discretization_cfg_->number_of_time_points();
    // This is the proper size of the container:
    LSS_ASSERT((solutions.rows() == time_size) && (solutions.columns() == space_size),
               "The input solution 2D container must have the correct size");
    // grid:
    grid_config_1d_ptr<fp_type> grid_cfg;
    if (grid_hints_cfg_->grid() == grid_enum::Uniform)
    {
        grid_cfg = std::make_shared<uniform_grid_config_1d<fp_type>>(discretization_cfg_);
    }
    else if (grid_hints_cfg_->grid() == grid_enum::Nonuniform)
    {
        grid_cfg = std::make_shared<nonuniform_grid_config_1d<fp_type>>(
            discretization_cfg_, grid_hints_cfg_->strike(), grid_hints_cfg_->p_scale(), grid_hints_cfg_->c_scale());
    }
    else
    {
        throw std::exception("Unreachable");
    }
    // create container to carry previous solution:
    container_t prev_sol(space_size, fp_type{});
    //  create container to carry new solution:
    container_t next_sol(space_size, fp_type{});
    // discretize initial condition
    d_1d::of_function(grid_cfg, heat_data_cfg_->initial_condition(), prev_sol);
    // get heat_source:
    const bool is_heat_source_set = heat_data_cfg_->is_heat_source_set();
    // get heat_source:
    auto const &heat_source = heat_data_cfg_->heat_source();

    if (solver_cfg_->memory_space() == memory_space_enum::Device)
    {
        if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::CUDASolver)
        {
            typedef general_svc_heat_equation_implicit_kernel<
                memory_space_enum::Device, tridiagonal_method_enum::CUDASolver, fp_type, container, allocator>
                dev_cu_solver;

            dev_cu_solver solver(boundary_pair_, heat_data_cfg_, discretization_cfg_, solver_cfg_, grid_cfg);
            solver(prev_sol, next_sol, is_heat_source_set, heat_source, solutions);
        }
        else if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::SORSolver)
        {
            typedef general_svc_heat_equation_implicit_kernel<
                memory_space_enum::Device, tridiagonal_method_enum::SORSolver, fp_type, container, allocator>
                dev_sor_solver;
            LSS_ASSERT(!solver_config_details_.empty(), "solver_config_details map must not be empty");
            fp_type omega_value = solver_config_details_["sor_omega"];

            dev_sor_solver solver(boundary_pair_, heat_data_cfg_, discretization_cfg_, solver_cfg_, grid_cfg);
            solver(prev_sol, next_sol, is_heat_source_set, heat_source, omega_value, solutions);
        }
        else
        {
            throw std::exception("Not supported on Device");
        }
    }
    else if (solver_cfg_->memory_space() == memory_space_enum::Host)
    {
        if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::CUDASolver)
        {
            typedef general_svc_heat_equation_implicit_kernel<
                memory_space_enum::Host, tridiagonal_method_enum::CUDASolver, fp_type, container, allocator>
                host_cu_solver;

            host_cu_solver solver(boundary_pair_, heat_data_cfg_, discretization_cfg_, solver_cfg_, grid_cfg);
            solver(prev_sol, next_sol, is_heat_source_set, heat_source, solutions);
        }
        else if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::SORSolver)
        {
            typedef general_svc_heat_equation_implicit_kernel<
                memory_space_enum::Host, tridiagonal_method_enum::SORSolver, fp_type, container, allocator>
                host_sor_solver;

            LSS_ASSERT(!solver_config_details_.empty(), "solver_config_details map must not be empty");
            fp_type omega_value = solver_config_details_["sor_omega"];
            host_sor_solver solver(boundary_pair_, heat_data_cfg_, discretization_cfg_, solver_cfg_, grid_cfg);
            solver(prev_sol, next_sol, is_heat_source_set, heat_source, omega_value, solutions);
        }
        else if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::DoubleSweepSolver)
        {
            typedef general_svc_heat_equation_implicit_kernel<
                memory_space_enum::Host, tridiagonal_method_enum::DoubleSweepSolver, fp_type, container, allocator>
                host_dss_solver;
            host_dss_solver solver(boundary_pair_, heat_data_cfg_, discretization_cfg_, solver_cfg_, grid_cfg);
            solver(prev_sol, next_sol, is_heat_source_set, heat_source, solutions);
        }
        else if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::ThomasLUSolver)
        {
            typedef general_svc_heat_equation_implicit_kernel<
                memory_space_enum::Host, tridiagonal_method_enum::ThomasLUSolver, fp_type, container, allocator>
                host_lus_solver;
            host_lus_solver solver(boundary_pair_, heat_data_cfg_, discretization_cfg_, solver_cfg_, grid_cfg);
            solver(prev_sol, next_sol, is_heat_source_set, heat_source, solutions);
        }
        else
        {
            throw std::exception("Not supported on Host");
        }
    }
    else
    {
        throw std::exception("Unreachable");
    }
}

} // namespace implicit_solvers

namespace explicit_solvers
{

template <typename fp_type, template <typename, typename> typename container = std::vector,
          typename allocator = std::allocator<fp_type>>
class general_svc_heat_equation
{
  private:
    boundary_1d_pair<fp_type> boundary_pair_;
    heat_data_config_1d_ptr<fp_type> heat_data_cfg_;
    pde_discretization_config_1d_ptr<fp_type> discretization_cfg_;
    grid_config_hints_1d_ptr<fp_type> grid_hints_cfg_;
    heat_explicit_solver_config_ptr solver_cfg_;

    explicit general_svc_heat_equation() = delete;

    void initialize()
    {
        LSS_VERIFY(heat_data_cfg_, "heat_data_config must not be null");
        LSS_VERIFY(discretization_cfg_, "discretization_config must not be null");
        LSS_VERIFY(std::get<0>(boundary_pair_), "boundary_pair.first must not be null");
        LSS_VERIFY(std::get<1>(boundary_pair_), "boundary_pair.second must not be null");
        LSS_VERIFY(grid_hints_cfg_, "grid_config_hints must not be null");
        LSS_VERIFY(solver_cfg_, "solver_config must not be null");
    }

  public:
    explicit general_svc_heat_equation(heat_data_config_1d_ptr<fp_type> const &heat_data_config,
                                       pde_discretization_config_1d_ptr<fp_type> const &discretization_config,
                                       boundary_1d_pair<fp_type> const &boundary_pair,
                                       grid_config_hints_1d_ptr<fp_type> const &grid_config_hints,
                                       heat_explicit_solver_config_ptr const &solver_config =
                                           default_heat_solver_configs::dev_expl_fwd_euler_solver_config_ptr)
        : heat_data_cfg_{heat_data_config}, discretization_cfg_{discretization_config}, boundary_pair_{boundary_pair},
          grid_hints_cfg_{grid_config_hints}, solver_cfg_{solver_config}
    {
        initialize();
    }

    ~general_svc_heat_equation()
    {
    }

    general_svc_heat_equation(general_svc_heat_equation const &) = delete;
    general_svc_heat_equation(general_svc_heat_equation &&) = delete;
    general_svc_heat_equation &operator=(general_svc_heat_equation const &) = delete;
    general_svc_heat_equation &operator=(general_svc_heat_equation &&) = delete;

    /**
     * Get the final solution of the PDE
     *
     * \param solution - container for solution
     */
    void solve(container<fp_type, allocator> &solution);

    /**
     * Get all solutions in time (surface) of the PDE
     *
     * \param solutions - 2D container for all the solutions in time
     */
    void solve(container_2d<by_enum::Row, fp_type, container, allocator> &solutions);
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void general_svc_heat_equation<fp_type, container, allocator>::solve(container<fp_type, allocator> &solution)
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container<fp_type, allocator> container_t;

    LSS_ASSERT(solution.size() > 0, "The input solution container must be initialized");
    // get space range:
    const range<fp_type> space = discretization_cfg_->space_range();
    // get space step:
    const fp_type h = discretization_cfg_->space_step();
    // time step:
    const fp_type k = discretization_cfg_->time_step();
    // size of space discretization:
    const std::size_t space_size = discretization_cfg_->number_of_space_points();
    // This is the proper size of the container:
    LSS_ASSERT((solution.size() == space_size), "The input solution container must have the correct size");
    //  grid:
    grid_config_1d_ptr<fp_type> grid_cfg;
    if (grid_hints_cfg_->grid() == grid_enum::Uniform)
    {
        grid_cfg = std::make_shared<uniform_grid_config_1d<fp_type>>(discretization_cfg_);
    }
    else if (grid_hints_cfg_->grid() == grid_enum::Nonuniform)
    {
        grid_cfg = std::make_shared<nonuniform_grid_config_1d<fp_type>>(
            discretization_cfg_, grid_hints_cfg_->strike(), grid_hints_cfg_->p_scale(), grid_hints_cfg_->c_scale());
    }
    else
    {
        throw std::exception("Unreachable");
    }
    // create container to carry previous solution:
    container_t prev_sol(space_size, fp_type{});
    // discretize initial condition
    d_1d::of_function(grid_cfg, heat_data_cfg_->initial_condition(), prev_sol);
    // get heat_source:
    const bool is_heat_source_set = heat_data_cfg_->is_heat_source_set();
    // get heat_source:
    auto const &heat_source = heat_data_cfg_->heat_source();

    if (solver_cfg_->memory_space() == memory_space_enum::Device)
    {
        typedef general_svc_heat_equation_explicit_kernel<memory_space_enum::Device, fp_type, container, allocator>
            device_solver;
        device_solver solver(boundary_pair_, heat_data_cfg_, discretization_cfg_, solver_cfg_, grid_cfg);
        solver(prev_sol, is_heat_source_set, heat_source);
        std::copy(prev_sol.begin(), prev_sol.end(), solution.begin());
    }
    else if (solver_cfg_->memory_space() == memory_space_enum::Host)
    {
        typedef general_svc_heat_equation_explicit_kernel<memory_space_enum::Host, fp_type, container, allocator>
            host_solver;
        host_solver solver(boundary_pair_, heat_data_cfg_, discretization_cfg_, solver_cfg_, grid_cfg);
        solver(prev_sol, is_heat_source_set, heat_source);
        std::copy(prev_sol.begin(), prev_sol.end(), solution.begin());
    }
    else
    {
        throw std::exception("Unreachable");
    }
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void general_svc_heat_equation<fp_type, container, allocator>::solve(
    container_2d<by_enum::Row, fp_type, container, allocator> &solutions)
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container<fp_type, allocator> container_t;

    LSS_ASSERT(((solutions.columns() > 0) && (solutions.rows() > 0)),
               "The input solution 2D container must be initialized");

    // get space range:
    const range<fp_type> space = discretization_cfg_->space_range();
    // get space step:
    const fp_type h = discretization_cfg_->space_step();
    // time step:
    const fp_type k = discretization_cfg_->time_step();
    // size of space discretization:
    const std::size_t space_size = discretization_cfg_->number_of_space_points();
    // size of time discretization:
    const std::size_t time_size = discretization_cfg_->number_of_time_points();
    // This is the proper size of the container:
    LSS_ASSERT((solutions.rows() == time_size) && (solutions.columns() == space_size),
               "The input solution 2D container must have the correct size");
    //  grid:
    grid_config_1d_ptr<fp_type> grid_cfg;
    if (grid_hints_cfg_->grid() == grid_enum::Uniform)
    {
        grid_cfg = std::make_shared<uniform_grid_config_1d<fp_type>>(discretization_cfg_);
    }
    else if (grid_hints_cfg_->grid() == grid_enum::Nonuniform)
    {
        grid_cfg = std::make_shared<nonuniform_grid_config_1d<fp_type>>(
            discretization_cfg_, grid_hints_cfg_->strike(), grid_hints_cfg_->p_scale(), grid_hints_cfg_->c_scale());
    }
    else
    {
        throw std::exception("Unreachable");
    }
    // create container to carry previous solution:
    container_t prev_sol(space_size, fp_type{});
    // discretize initial condition
    d_1d::of_function(grid_cfg, heat_data_cfg_->initial_condition(), prev_sol);
    // get heat_source:
    const bool is_heat_source_set = heat_data_cfg_->is_heat_source_set();
    // get heat_source:
    auto const &heat_source = heat_data_cfg_->heat_source();

    if (solver_cfg_->memory_space() == memory_space_enum::Device)
    {
        typedef general_svc_heat_equation_explicit_kernel<memory_space_enum::Device, fp_type, container, allocator>
            device_solver;
        device_solver solver(boundary_pair_, heat_data_cfg_, discretization_cfg_, solver_cfg_, grid_cfg);
        solver(prev_sol, is_heat_source_set, heat_source, solutions);
    }
    else if (solver_cfg_->memory_space() == memory_space_enum::Host)
    {
        typedef general_svc_heat_equation_explicit_kernel<memory_space_enum::Host, fp_type, container, allocator>
            host_solver;
        host_solver solver(boundary_pair_, heat_data_cfg_, discretization_cfg_, solver_cfg_, grid_cfg);
        solver(prev_sol, is_heat_source_set, heat_source, solutions);
    }
    else
    {
        throw std::exception("Unreachable");
    }
}

} // namespace explicit_solvers

} // namespace one_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_1D_GENERAL_SVC_HEAT_EQUATION_HPP_
