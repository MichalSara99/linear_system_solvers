#if !defined(_LSS_1D_GENERAL_SVC_WAVE_EQUATION_HPP_)
#define _LSS_1D_GENERAL_SVC_WAVE_EQUATION_HPP_

#include <functional>
#include <map>

#include "boundaries/lss_boundary.hpp"
#include "common/lss_macros.hpp"
#include "containers/lss_container_2d.hpp"
#include "discretization/lss_discretization.hpp"
#include "lss_1d_general_svc_wave_equation_explicit_kernel.hpp"
#include "lss_1d_general_svc_wave_equation_implicit_kernel.hpp"
#include "pde_solvers/lss_pde_discretization_config.hpp"
#include "pde_solvers/lss_wave_data_config.hpp"
#include "pde_solvers/lss_wave_solver_config.hpp"
#include "pde_solvers/transformation/lss_boundary_transform.hpp"
#include "pde_solvers/transformation/lss_wave_data_transform.hpp"

namespace lss_pde_solvers
{

namespace one_dimensional
{
using lss_boundary::boundary_1d_pair;
using lss_boundary::boundary_1d_ptr;
using lss_containers::container_2d;

namespace implicit_solvers
{

/*!
============================================================================
Represents general spacial variable coefficient 1D wave equation solver

u_tt + a(x)*u_t = b(x)*u_xx + c(x)*u_x + d(x)*u + F(x,t),
x_1 < x < x_2
t_1 < t < t_2

with initial condition:

u(x,t_1) = f(x)

u_t(x,t_1) = g(x)

or terminal condition:

u(x,t_2) = f(x)

u_t(x,t_2) = g(x)

// ============================================================================
*/
template <typename fp_type, template <typename, typename> typename container = std::vector,
          typename allocator = std::allocator<fp_type>>
class general_svc_wave_equation
{

  private:
    wave_data_transform_1d_ptr<fp_type> wave_data_trans_cfg_;
    pde_discretization_config_1d_ptr<fp_type> discretization_cfg_;
    boundary_transform_1d_ptr<fp_type> boundary_;
    grid_transform_config_1d_ptr<fp_type> grid_trans_cfg_; // this may be removed as it is not used later
    wave_implicit_solver_config_ptr solver_cfg_;
    std::map<std::string, fp_type> solver_config_details_;

    explicit general_svc_wave_equation() = delete;

    void initialize(wave_data_config_1d_ptr<fp_type> const &wave_data_cfg,
                    grid_config_hints_1d_ptr<fp_type> const &grid_config_hints,
                    boundary_1d_pair<fp_type> const &boundary_pair)
    {
        LSS_VERIFY(wave_data_cfg, "wave_data_config must not be null");
        LSS_VERIFY(discretization_cfg_, "discretization_config must not be null");
        LSS_VERIFY(std::get<0>(boundary_pair), "boundary_pair.first must not be null");
        LSS_VERIFY(std::get<1>(boundary_pair), "boundary_pair.second must not be null");
        LSS_VERIFY(solver_cfg_, "solver_config must not be null");
        if (!solver_config_details_.empty())
        {
            auto const &it = solver_config_details_.find("sor_omega");
            LSS_ASSERT(it != solver_config_details_.end(), "sor_omega is not defined");
        }
        // make necessary transformations:
        // create grid_transform_config:
        grid_trans_cfg_ = std::make_shared<grid_transform_config_1d<fp_type>>(discretization_cfg_, grid_config_hints);
        // transform original wave data:
        wave_data_trans_cfg_ = std::make_shared<wave_data_transform_1d<fp_type>>(wave_data_cfg, grid_trans_cfg_);
        // transform original boundary:
        boundary_ = std::make_shared<boundary_transform_1d<fp_type>>(boundary_pair, grid_trans_cfg_);
    }

  public:
    explicit general_svc_wave_equation(
        wave_data_config_1d_ptr<fp_type> const &wave_data_config,
        pde_discretization_config_1d_ptr<fp_type> const &discretization_config,
        boundary_1d_pair<fp_type> const &boundary_pair, grid_config_hints_1d_ptr<fp_type> const &grid_config_hints,
        wave_implicit_solver_config_ptr const &solver_config =
            default_wave_solver_configs::dev_fwd_cusolver_qr_solver_config_ptr,
        std::map<std::string, fp_type> const &solver_config_details = std::map<std::string, fp_type>())
        : discretization_cfg_{discretization_config}, solver_cfg_{solver_config}, solver_config_details_{
                                                                                      solver_config_details}
    {
        initialize(wave_data_config, grid_config_hints, boundary_pair);
    }

    ~general_svc_wave_equation()
    {
    }

    general_svc_wave_equation(general_svc_wave_equation const &) = delete;
    general_svc_wave_equation(general_svc_wave_equation &&) = delete;
    general_svc_wave_equation &operator=(general_svc_wave_equation const &) = delete;
    general_svc_wave_equation &operator=(general_svc_wave_equation &&) = delete;

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
void general_svc_wave_equation<fp_type, container, allocator>::solve(container<fp_type, allocator> &solution)
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container<fp_type, allocator> container_t;

    LSS_ASSERT(solution.size() > 0, "The input solution container must be initialized");
    // size of space discretization:
    const std::size_t space_size = discretization_cfg_->number_of_space_points();
    // This is the proper size of the container:
    LSS_ASSERT(solution.size() == space_size, "The input solution container must have the correct size");
    // grid:
    auto const &grid_cfg = std::make_shared<grid_config_1d<fp_type>>(discretization_cfg_);
    auto const &boundary_pair = boundary_->boundary_pair();
    // create container to carry previous solution:
    container_t prev_sol_0(space_size, fp_type{});
    // discretize initial condition
    d_1d::of_function(grid_cfg, wave_data_trans_cfg_->first_initial_condition(), prev_sol_0);
    // create container to carry second initial condition:
    container_t prev_sol_1(space_size, fp_type{});
    // discretize first order initial condition:
    d_1d::of_function(grid_cfg, wave_data_trans_cfg_->second_initial_condition(), prev_sol_1);
    // create container to carry new solution:
    container_t next_sol(space_size, fp_type{});
    // get heat_source:
    const bool is_wave_source_set = wave_data_trans_cfg_->is_wave_source_set();
    // get heat_source:
    auto const &wave_source = wave_data_trans_cfg_->wave_source();

    if (solver_cfg_->memory_space() == memory_space_enum::Device)
    {
        if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::CUDASolver)
        {
            typedef general_svc_wave_equation_implicit_kernel<
                memory_space_enum::Device, tridiagonal_method_enum::CUDASolver, fp_type, container, allocator>
                dev_cu_solver;

            dev_cu_solver solver(boundary_pair, wave_data_trans_cfg_, discretization_cfg_, solver_cfg_, grid_cfg);
            solver(prev_sol_0, prev_sol_1, next_sol, is_wave_source_set, wave_source);
            std::copy(prev_sol_1.begin(), prev_sol_1.end(), solution.begin());
        }
        else if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::SORSolver)
        {
            typedef general_svc_wave_equation_implicit_kernel<
                memory_space_enum::Device, tridiagonal_method_enum::SORSolver, fp_type, container, allocator>
                dev_sor_solver;
            LSS_ASSERT(!solver_config_details_.empty(), "solver_config_details map must not be empty");
            fp_type omega_value = solver_config_details_["sor_omega"];
            dev_sor_solver solver(boundary_pair, wave_data_trans_cfg_, discretization_cfg_, solver_cfg_, grid_cfg);
            solver(prev_sol_0, prev_sol_1, next_sol, is_wave_source_set, wave_source, omega_value);
            std::copy(prev_sol_1.begin(), prev_sol_1.end(), solution.begin());
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
            typedef general_svc_wave_equation_implicit_kernel<
                memory_space_enum::Host, tridiagonal_method_enum::CUDASolver, fp_type, container, allocator>
                host_cu_solver;
            host_cu_solver solver(boundary_pair, wave_data_trans_cfg_, discretization_cfg_, solver_cfg_, grid_cfg);
            solver(prev_sol_0, prev_sol_1, next_sol, is_wave_source_set, wave_source);
            std::copy(prev_sol_1.begin(), prev_sol_1.end(), solution.begin());
        }
        else if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::SORSolver)
        {
            typedef general_svc_wave_equation_implicit_kernel<
                memory_space_enum::Host, tridiagonal_method_enum::SORSolver, fp_type, container, allocator>
                host_sor_solver;

            LSS_ASSERT(!solver_config_details_.empty(), "solver_config_details map must not be empty");
            fp_type omega_value = solver_config_details_["sor_omega"];
            host_sor_solver solver(boundary_pair, wave_data_trans_cfg_, discretization_cfg_, solver_cfg_, grid_cfg);
            solver(prev_sol_0, prev_sol_1, next_sol, is_wave_source_set, wave_source, omega_value);
            std::copy(prev_sol_1.begin(), prev_sol_1.end(), solution.begin());
        }
        else if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::DoubleSweepSolver)
        {
            typedef general_svc_wave_equation_implicit_kernel<
                memory_space_enum::Host, tridiagonal_method_enum::DoubleSweepSolver, fp_type, container, allocator>
                host_dss_solver;
            host_dss_solver solver(boundary_pair, wave_data_trans_cfg_, discretization_cfg_, solver_cfg_, grid_cfg);
            solver(prev_sol_0, prev_sol_1, next_sol, is_wave_source_set, wave_source);
            std::copy(prev_sol_1.begin(), prev_sol_1.end(), solution.begin());
        }
        else if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::ThomasLUSolver)
        {
            typedef general_svc_wave_equation_implicit_kernel<
                memory_space_enum::Host, tridiagonal_method_enum::ThomasLUSolver, fp_type, container, allocator>
                host_lus_solver;
            host_lus_solver solver(boundary_pair, wave_data_trans_cfg_, discretization_cfg_, solver_cfg_, grid_cfg);
            solver(prev_sol_0, prev_sol_1, next_sol, is_wave_source_set, wave_source);
            std::copy(prev_sol_1.begin(), prev_sol_1.end(), solution.begin());
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
void general_svc_wave_equation<fp_type, container, allocator>::solve(
    container_2d<by_enum::Row, fp_type, container, allocator> &solutions)
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container<fp_type, allocator> container_t;

    LSS_ASSERT(((solutions.columns() > 0) && (solutions.rows() > 0)),
               "The input solution 2D container must be initialized");
    // size of space discretization:
    const std::size_t space_size = discretization_cfg_->number_of_space_points();
    // size of time discretization:
    const std::size_t time_size = discretization_cfg_->number_of_time_points();
    // This is the proper size of the container:
    LSS_ASSERT((solutions.rows() == time_size) && (solutions.columns() == space_size),
               "The input solution 2D container must have the correct size");
    // grid:
    auto const &grid_cfg = std::make_shared<grid_config_1d<fp_type>>(discretization_cfg_);
    auto const &boundary_pair = boundary_->boundary_pair();
    // create container to carry previous solution:
    container_t prev_sol_0(space_size, fp_type{});
    // discretize initial condition
    d_1d::of_function(grid_cfg, wave_data_trans_cfg_->first_initial_condition(), prev_sol_0);
    // create container to carry second initial condition:
    container_t prev_sol_1(space_size, fp_type{});
    // discretize first order initial condition:
    d_1d::of_function(grid_cfg, wave_data_trans_cfg_->second_initial_condition(), prev_sol_1);
    // create container to carry new solution:
    container_t next_sol(space_size, fp_type{});
    // get heat_source:
    const bool is_wave_source_set = wave_data_trans_cfg_->is_wave_source_set();
    // get heat_source:
    auto const &wave_source = wave_data_trans_cfg_->wave_source();

    if (solver_cfg_->memory_space() == memory_space_enum::Device)
    {
        if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::CUDASolver)
        {
            typedef general_svc_wave_equation_implicit_kernel<
                memory_space_enum::Device, tridiagonal_method_enum::CUDASolver, fp_type, container, allocator>
                dev_cu_solver;

            dev_cu_solver solver(boundary_pair, wave_data_trans_cfg_, discretization_cfg_, solver_cfg_, grid_cfg);
            solver(prev_sol_0, prev_sol_1, next_sol, is_wave_source_set, wave_source, solutions);
        }
        else if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::SORSolver)
        {
            typedef general_svc_wave_equation_implicit_kernel<
                memory_space_enum::Device, tridiagonal_method_enum::SORSolver, fp_type, container, allocator>
                dev_sor_solver;
            LSS_ASSERT(!solver_config_details_.empty(), "solver_config_details map must not be empty");
            fp_type omega_value = solver_config_details_["sor_omega"];
            dev_sor_solver solver(boundary_pair, wave_data_trans_cfg_, discretization_cfg_, solver_cfg_, grid_cfg);
            solver(prev_sol_0, prev_sol_1, next_sol, is_wave_source_set, wave_source, omega_value, solutions);
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
            typedef general_svc_wave_equation_implicit_kernel<
                memory_space_enum::Host, tridiagonal_method_enum::CUDASolver, fp_type, container, allocator>
                host_cu_solver;
            host_cu_solver solver(boundary_pair, wave_data_trans_cfg_, discretization_cfg_, solver_cfg_, grid_cfg);
            solver(prev_sol_0, prev_sol_1, next_sol, is_wave_source_set, wave_source, solutions);
        }
        else if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::SORSolver)
        {
            typedef general_svc_wave_equation_implicit_kernel<
                memory_space_enum::Host, tridiagonal_method_enum::SORSolver, fp_type, container, allocator>
                host_sor_solver;

            LSS_ASSERT(!solver_config_details_.empty(), "solver_config_details map must not be empty");
            fp_type omega_value = solver_config_details_["sor_omega"];
            host_sor_solver solver(boundary_pair, wave_data_trans_cfg_, discretization_cfg_, solver_cfg_, grid_cfg);
            solver(prev_sol_0, prev_sol_1, next_sol, is_wave_source_set, wave_source, omega_value, solutions);
        }
        else if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::DoubleSweepSolver)
        {
            typedef general_svc_wave_equation_implicit_kernel<
                memory_space_enum::Host, tridiagonal_method_enum::DoubleSweepSolver, fp_type, container, allocator>
                host_dss_solver;
            host_dss_solver solver(boundary_pair, wave_data_trans_cfg_, discretization_cfg_, solver_cfg_, grid_cfg);
            solver(prev_sol_0, prev_sol_1, next_sol, is_wave_source_set, wave_source, solutions);
        }
        else if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::ThomasLUSolver)
        {
            typedef general_svc_wave_equation_implicit_kernel<
                memory_space_enum::Host, tridiagonal_method_enum::ThomasLUSolver, fp_type, container, allocator>
                host_lus_solver;
            host_lus_solver solver(boundary_pair, wave_data_trans_cfg_, discretization_cfg_, solver_cfg_, grid_cfg);
            solver(prev_sol_0, prev_sol_1, next_sol, is_wave_source_set, wave_source, solutions);
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
class general_svc_wave_equation
{
  private:
    wave_data_transform_1d_ptr<fp_type> wave_data_trans_cfg_;
    pde_discretization_config_1d_ptr<fp_type> discretization_cfg_;
    boundary_transform_1d_ptr<fp_type> boundary_;
    grid_transform_config_1d_ptr<fp_type> grid_trans_cfg_; // this may be removed as it is not used later
    wave_explicit_solver_config_ptr solver_cfg_;

    explicit general_svc_wave_equation() = delete;

    void initialize(wave_data_config_1d_ptr<fp_type> const &wave_data_cfg,
                    grid_config_hints_1d_ptr<fp_type> const &grid_config_hints,
                    boundary_1d_pair<fp_type> const &boundary_pair)
    {
        LSS_VERIFY(wave_data_cfg, "wave_data_config must not be null");
        LSS_VERIFY(discretization_cfg_, "discretization_config must not be null");
        LSS_VERIFY(std::get<0>(boundary_pair), "boundary_pair.first must not be null");
        LSS_VERIFY(std::get<1>(boundary_pair), "boundary_pair.second must not be null");
        LSS_VERIFY(solver_cfg_, "solver_config must not be null");

        // make necessary transformations:
        // create grid_transform_config:
        grid_trans_cfg_ = std::make_shared<grid_transform_config_1d<fp_type>>(discretization_cfg_, grid_config_hints);
        // transform original wave data:
        wave_data_trans_cfg_ = std::make_shared<wave_data_transform_1d<fp_type>>(wave_data_cfg, grid_trans_cfg_);
        // transform original boundary:
        boundary_ = std::make_shared<boundary_transform_1d<fp_type>>(boundary_pair, grid_trans_cfg_);
    }

  public:
    explicit general_svc_wave_equation(wave_data_config_1d_ptr<fp_type> const &wave_data_config,
                                       pde_discretization_config_1d_ptr<fp_type> const &discretization_config,
                                       boundary_1d_pair<fp_type> const &boundary_pair,
                                       grid_config_hints_1d_ptr<fp_type> const &grid_config_hints,
                                       wave_explicit_solver_config_ptr const &solver_config =
                                           default_wave_solver_configs::dev_expl_fwd_solver_config_ptr)
        : discretization_cfg_{discretization_config}, solver_cfg_{solver_config}
    {
        initialize(wave_data_config, grid_config_hints, boundary_pair);
    }

    ~general_svc_wave_equation()
    {
    }

    general_svc_wave_equation(general_svc_wave_equation const &) = delete;
    general_svc_wave_equation(general_svc_wave_equation &&) = delete;
    general_svc_wave_equation &operator=(general_svc_wave_equation const &) = delete;
    general_svc_wave_equation &operator=(general_svc_wave_equation &&) = delete;

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
void general_svc_wave_equation<fp_type, container, allocator>::solve(container<fp_type, allocator> &solution)
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container<fp_type, allocator> container_t;

    LSS_ASSERT(solution.size() > 0, "The input solution container must be initialized");
    // size of space discretization:
    const std::size_t space_size = discretization_cfg_->number_of_space_points();
    // This is the proper size of the container:
    LSS_ASSERT((solution.size() == space_size), "The input solution container must have the correct size");
    // grid:
    auto const &grid_cfg = std::make_shared<grid_config_1d<fp_type>>(discretization_cfg_);
    auto const &boundary_pair = boundary_->boundary_pair();
    // create container to carry previous solution:
    container_t prev_sol_0(space_size, fp_type{});
    // discretize initial condition
    d_1d::of_function(grid_cfg, wave_data_trans_cfg_->first_initial_condition(), prev_sol_0);
    // create container to carry second initial condition:
    container_t prev_sol_1(space_size, fp_type{});
    // discretize first order initial condition:
    d_1d::of_function(grid_cfg, wave_data_trans_cfg_->second_initial_condition(), prev_sol_1);
    // get wave_source:
    auto const &wave_source = wave_data_trans_cfg_->wave_source();
    // is wave_source set:
    const bool is_wave_source_set = wave_data_trans_cfg_->is_wave_source_set();
    // create container to carry new solution:
    container_t next_sol(space_size, fp_type{});

    if (solver_cfg_->memory_space() == memory_space_enum::Device)
    {
        typedef general_svc_wave_equation_explicit_kernel<memory_space_enum::Device, fp_type, container, allocator>
            device_solver;
        device_solver solver(boundary_pair, wave_data_trans_cfg_, discretization_cfg_, solver_cfg_, grid_cfg);
        solver(prev_sol_0, prev_sol_1, next_sol, is_wave_source_set, wave_source);
        std::copy(next_sol.begin(), next_sol.end(), solution.begin());
    }
    else if (solver_cfg_->memory_space() == memory_space_enum::Host)
    {
        typedef general_svc_wave_equation_explicit_kernel<memory_space_enum::Host, fp_type, container, allocator>
            host_solver;
        host_solver solver(boundary_pair, wave_data_trans_cfg_, discretization_cfg_, solver_cfg_, grid_cfg);
        solver(prev_sol_0, prev_sol_1, next_sol, is_wave_source_set, wave_source);
        std::copy(prev_sol_1.begin(), prev_sol_1.end(), solution.begin());
    }
    else
    {
        throw std::exception("Unreachable");
    }
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void general_svc_wave_equation<fp_type, container, allocator>::solve(
    container_2d<by_enum::Row, fp_type, container, allocator> &solutions)
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container<fp_type, allocator> container_t;

    LSS_ASSERT(((solutions.columns() > 0) && (solutions.rows() > 0)),
               "The input solution 2D container must be initialized");
    // size of space discretization:
    const std::size_t space_size = discretization_cfg_->number_of_space_points();
    // size of time discretization:
    const std::size_t time_size = discretization_cfg_->number_of_time_points();
    // This is the proper size of the container:
    LSS_ASSERT((solutions.rows() == time_size) && (solutions.columns() == space_size),
               "The input solution 2D container must have the correct size");
    // grid:
    auto const &grid_cfg = std::make_shared<grid_config_1d<fp_type>>(discretization_cfg_);
    auto const &boundary_pair = boundary_->boundary_pair();
    // create container to carry previous solution:
    container_t prev_sol_0(space_size, fp_type{});
    // discretize initial condition
    d_1d::of_function(grid_cfg, wave_data_trans_cfg_->first_initial_condition(), prev_sol_0);
    // create container to carry second initial condition:
    container_t prev_sol_1(space_size, fp_type{});
    // discretize first order initial condition:
    d_1d::of_function(grid_cfg, wave_data_trans_cfg_->second_initial_condition(), prev_sol_1);
    // get wave_source:
    auto const &wave_source = wave_data_trans_cfg_->wave_source();
    // is wave_source set:
    const bool is_wave_source_set = wave_data_trans_cfg_->is_wave_source_set();
    // create container to carry new solution:
    container_t next_sol(space_size, fp_type{});

    if (solver_cfg_->memory_space() == memory_space_enum::Device)
    {
        typedef general_svc_wave_equation_explicit_kernel<memory_space_enum::Device, fp_type, container, allocator>
            device_solver;
        device_solver solver(boundary_pair, wave_data_trans_cfg_, discretization_cfg_, solver_cfg_, grid_cfg);
        solver(prev_sol_0, prev_sol_1, next_sol, is_wave_source_set, wave_source, solutions);
    }
    else if (solver_cfg_->memory_space() == memory_space_enum::Host)
    {
        typedef general_svc_wave_equation_explicit_kernel<memory_space_enum::Host, fp_type, container, allocator>
            host_solver;
        host_solver solver(boundary_pair, wave_data_trans_cfg_, discretization_cfg_, solver_cfg_, grid_cfg);
        solver(prev_sol_0, prev_sol_1, next_sol, is_wave_source_set, wave_source, solutions);
    }
    else
    {
        throw std::exception("Unreachable");
    }
}

} // namespace explicit_solvers

} // namespace one_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_1D_GENERAL_SVC_WAVE_EQUATION_HPP_
