#if !defined(_LSS_GENERAL_SVC_HEAT_EQUATION_HPP_)
#define _LSS_GENERAL_SVC_HEAT_EQUATION_HPP_

#include <functional>
#include <map>

#include "boundaries/lss_boundary_1d.hpp"
#include "common/lss_macros.hpp"
#include "containers/lss_container_2d.hpp"
#include "lss_general_svc_heat_equation_explicit_kernel.hpp"
#include "lss_general_svc_heat_equation_implicit_kernel.hpp"
#include "pde_solvers/lss_discretization.hpp"
#include "pde_solvers/lss_discretization_config.hpp"
#include "pde_solvers/lss_heat_data_config.hpp"
#include "pde_solvers/lss_solver_config.hpp"

namespace lss_pde_solvers
{

namespace one_dimensional
{
using lss_boundary_1d::boundary_1d_pair;
using lss_boundary_1d::boundary_1d_ptr;
using lss_containers::container_2d;

namespace implicit_solvers
{

/*!
============================================================================
Represents general spacial variable coefficient 1D heat equation solver

u_t = a(x)*u_xx + b(x)*u_x + c(x)*u + F(x,t),
t > 0, x_1 < x < x_2

with initial condition:

u(x,0) = f(x)


// ============================================================================
*/
template <typename fp_type, template <typename, typename> typename container = std::vector,
          typename allocator = std::allocator<fp_type>>
class general_svc_heat_equation
{

  private:
    boundary_1d_pair<fp_type> boundary_pair_;
    heat_data_config_1d_ptr<fp_type> heat_data_cfg_;
    discretization_config_1d_ptr<fp_type> discretization_cfg_;
    implicit_solver_config_1d_ptr solver_cfg_;
    std::map<std::string, fp_type> solver_config_details_;

    explicit general_svc_heat_equation() = delete;

    void initialize()
    {
        LSS_VERIFY(heat_data_cfg_, "heat_data_config must not be null");
        LSS_VERIFY(discretization_cfg_, "discretization_config must not be null");
        LSS_VERIFY(std::get<0>(boundary_pair_), "boundary_pair.first must not be null");
        LSS_VERIFY(std::get<1>(boundary_pair_), "boundary_pair.second must not be null");
        LSS_VERIFY(solver_cfg_, "solver_config must not be null");
        if (!solver_config_details_.empty())
        {
            auto const &it = solver_config_details_.find("sor_omega");
            LSS_ASSERT(it != solver_config_details_.end(), "sor_omega is not defined");
        }
    }

  public:
    explicit general_svc_heat_equation(
        heat_data_config_1d_ptr<fp_type> const &heat_data_config,
        discretization_config_1d_ptr<fp_type> const &discretization_config,
        boundary_1d_pair<fp_type> const &boundary_pair,
        implicit_solver_config_1d_ptr const &solver_config = host_fwd_dssolver_euler_solver_config_ptr,
        std::map<std::string, fp_type> const &solver_config_details = std::map<std::string, fp_type>())
        : heat_data_cfg_{heat_data_config}, discretization_cfg_{discretization_config}, boundary_pair_{boundary_pair},
          solver_cfg_{solver_config}, solver_config_details_{solver_config_details}
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
    void solve(container_2d<fp_type, container, allocator> &solutions);
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
    // calculate scheme coefficients:
    const fp_type one = static_cast<fp_type>(1.0);
    const fp_type two = static_cast<fp_type>(2.0);
    const fp_type half = static_cast<fp_type>(0.5);
    const fp_type lambda = k / (h * h);
    const fp_type gamma = k / (two * h);
    const fp_type delta = half * k;
    // create container to carry previous solution:
    container_t prev_sol(space_size, fp_type{});
    // discretize initial condition
    d_1d::of_function(space.lower(), h, heat_data_cfg_->initial_condition(), prev_sol);
    // since coefficients are different in space :
    container_t low(space_size, fp_type{});
    container_t diag(space_size, fp_type{});
    container_t up(space_size, fp_type{});
    // save coefficients:
    auto const &a = heat_data_cfg_->a_coefficient();
    auto const &b = heat_data_cfg_->b_coefficient();
    auto const &c = heat_data_cfg_->c_coefficient();
    // prepare space variable coefficients:
    auto const &A = [&](fp_type x) { return (lambda * a(x) - gamma * b(x)); };
    auto const &B = [&](fp_type x) { return (lambda * a(x) - delta * c(x)); };
    auto const &D = [&](fp_type x) { return (lambda * a(x) + gamma * b(x)); };
    // wrap up the functions into tuple:
    auto const &fun_triplet = std::make_tuple(A, B, D);
    // get propper theta accoring to clients chosen scheme:
    fp_type theta{};
    if (solver_cfg_->implicit_pde_scheme() == implicit_pde_schemes_enum::Euler)
    {
        theta = one;
    }
    else if (solver_cfg_->implicit_pde_scheme() == implicit_pde_schemes_enum::CrankNicolson)
    {
        theta = half;
    }
    else
    {
        throw std::exception("Unreachable");
    }
    fp_type m{};
    for (std::size_t t = 0; t < low.size(); ++t)
    {
        m = static_cast<fp_type>(t);
        low[t] = -one * A(m * h) * theta;
        diag[t] = (one + two * B(m * h) * theta);
        up[t] = -one * D(m * h) * theta;
    }
    // wrap up the diagonals into tuple:
    auto const &diag_triplet = std::make_tuple(low, diag, up);
    container_t rhs(space_size, fp_type{});
    // create container to carry new solution:
    container_t next_sol(space_size, fp_type{});
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

            dev_cu_solver solver(diag_triplet, fun_triplet, boundary_pair_, discretization_cfg_, solver_cfg_);
            solver(prev_sol, next_sol, rhs, is_heat_source_set, heat_source);
            std::copy(prev_sol.begin(), prev_sol.end(), solution.begin());
        }
        else if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::SORSolver)
        {
            typedef general_svc_heat_equation_implicit_kernel<
                memory_space_enum::Device, tridiagonal_method_enum::SORSolver, fp_type, container, allocator>
                dev_sor_solver;
            LSS_ASSERT(!solver_config_details_.empty(), "solver_config_details map must not be empty");
            fp_type omega_value = solver_config_details_["sor_omega"];
            dev_sor_solver solver(diag_triplet, fun_triplet, boundary_pair_, discretization_cfg_, solver_cfg_);
            solver(prev_sol, next_sol, rhs, is_heat_source_set, heat_source, omega_value);
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
            host_cu_solver solver(diag_triplet, fun_triplet, boundary_pair_, discretization_cfg_, solver_cfg_);
            solver(prev_sol, next_sol, rhs, is_heat_source_set, heat_source);
            std::copy(prev_sol.begin(), prev_sol.end(), solution.begin());
        }
        else if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::SORSolver)
        {
            typedef general_svc_heat_equation_implicit_kernel<
                memory_space_enum::Host, tridiagonal_method_enum::SORSolver, fp_type, container, allocator>
                host_sor_solver;

            LSS_ASSERT(!solver_config_details_.empty(), "solver_config_details map must not be empty");
            fp_type omega_value = solver_config_details_["sor_omega"];
            host_sor_solver solver(diag_triplet, fun_triplet, boundary_pair_, discretization_cfg_, solver_cfg_);
            solver(prev_sol, next_sol, rhs, is_heat_source_set, heat_source, omega_value);
            std::copy(prev_sol.begin(), prev_sol.end(), solution.begin());
        }
        else if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::DoubleSweepSolver)
        {
            typedef general_svc_heat_equation_implicit_kernel<
                memory_space_enum::Host, tridiagonal_method_enum::DoubleSweepSolver, fp_type, container, allocator>
                host_dss_solver;
            host_dss_solver solver(diag_triplet, fun_triplet, boundary_pair_, discretization_cfg_, solver_cfg_);
            solver(prev_sol, next_sol, rhs, is_heat_source_set, heat_source);
            std::copy(prev_sol.begin(), prev_sol.end(), solution.begin());
        }
        else if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::ThomasLUSolver)
        {
            typedef general_svc_heat_equation_implicit_kernel<
                memory_space_enum::Host, tridiagonal_method_enum::ThomasLUSolver, fp_type, container, allocator>
                host_lus_solver;
            host_lus_solver solver(diag_triplet, fun_triplet, boundary_pair_, discretization_cfg_, solver_cfg_);
            solver(prev_sol, next_sol, rhs, is_heat_source_set, heat_source);
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
    container_2d<fp_type, container, allocator> &solutions)
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
    // calculate scheme coefficients:
    const fp_type one = static_cast<fp_type>(1.0);
    const fp_type two = static_cast<fp_type>(2.0);
    const fp_type half = static_cast<fp_type>(0.5);
    const fp_type lambda = k / (h * h);
    const fp_type gamma = k / (two * h);
    const fp_type delta = half * k;
    // create container to carry previous solution:
    container_t prev_sol(space_size, fp_type{});
    // discretize initial condition
    d_1d::of_function(space.lower(), h, heat_data_cfg_->initial_condition(), prev_sol);
    // since coefficients are different in space :
    container_t low(space_size, fp_type{});
    container_t diag(space_size, fp_type{});
    container_t up(space_size, fp_type{});
    // save coefficients:
    auto const &a = heat_data_cfg_->a_coefficient();
    auto const &b = heat_data_cfg_->b_coefficient();
    auto const &c = heat_data_cfg_->c_coefficient();
    // prepare space variable coefficients:
    auto const &A = [&](fp_type x) { return (lambda * a(x) - gamma * b(x)); };
    auto const &B = [&](fp_type x) { return (lambda * a(x) - delta * c(x)); };
    auto const &D = [&](fp_type x) { return (lambda * a(x) + gamma * b(x)); };
    // wrap up the functions into tuple:
    auto const &fun_triplet = std::make_tuple(A, B, D);
    // get propper theta accoring to clients chosen scheme:
    fp_type theta{};
    if (solver_cfg_->implicit_pde_scheme() == implicit_pde_schemes_enum::Euler)
    {
        theta = one;
    }
    else if (solver_cfg_->implicit_pde_scheme() == implicit_pde_schemes_enum::CrankNicolson)
    {
        theta = half;
    }
    else
    {
        throw std::exception("Unreachable");
    }
    fp_type m{};
    for (std::size_t t = 0; t < low.size(); ++t)
    {
        m = static_cast<fp_type>(t);
        low[t] = -one * A(m * h) * theta;
        diag[t] = (one + two * B(m * h) * theta);
        up[t] = -one * D(m * h) * theta;
    }
    // wrap up the diagonals into tuple:
    auto const &diag_triplet = std::make_tuple(low, diag, up);
    container_t rhs(space_size, fp_type{});
    // create container to carry new solution:
    container_t next_sol(space_size, fp_type{});
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

            dev_cu_solver solver(diag_triplet, fun_triplet, boundary_pair_, discretization_cfg_, solver_cfg_);
            solver(prev_sol, next_sol, rhs, is_heat_source_set, heat_source, solutions);
        }
        else if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::SORSolver)
        {
            typedef general_svc_heat_equation_implicit_kernel<
                memory_space_enum::Device, tridiagonal_method_enum::SORSolver, fp_type, container, allocator>
                dev_sor_solver;
            LSS_ASSERT(!solver_config_details_.empty(), "solver_config_details map must not be empty");
            fp_type omega_value = solver_config_details_["sor_omega"];
            dev_sor_solver solver(diag_triplet, fun_triplet, boundary_pair_, discretization_cfg_, solver_cfg_);
            solver(prev_sol, next_sol, rhs, is_heat_source_set, heat_source, omega_value, solutions);
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
            host_cu_solver solver(diag_triplet, fun_triplet, boundary_pair_, discretization_cfg_, solver_cfg_);
            solver(prev_sol, next_sol, rhs, is_heat_source_set, heat_source, solutions);
        }
        else if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::SORSolver)
        {
            typedef general_svc_heat_equation_implicit_kernel<
                memory_space_enum::Host, tridiagonal_method_enum::SORSolver, fp_type, container, allocator>
                host_sor_solver;

            LSS_ASSERT(!solver_config_details_.empty(), "solver_config_details map must not be empty");
            fp_type omega_value = solver_config_details_["sor_omega"];
            host_sor_solver solver(diag_triplet, fun_triplet, boundary_pair_, discretization_cfg_, solver_cfg_);
            solver(prev_sol, next_sol, rhs, is_heat_source_set, heat_source, omega_value, solutions);
        }
        else if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::DoubleSweepSolver)
        {
            typedef general_svc_heat_equation_implicit_kernel<
                memory_space_enum::Host, tridiagonal_method_enum::DoubleSweepSolver, fp_type, container, allocator>
                host_dss_solver;
            host_dss_solver solver(diag_triplet, fun_triplet, boundary_pair_, discretization_cfg_, solver_cfg_);
            solver(prev_sol, next_sol, rhs, is_heat_source_set, heat_source, solutions);
        }
        else if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::ThomasLUSolver)
        {
            typedef general_svc_heat_equation_implicit_kernel<
                memory_space_enum::Host, tridiagonal_method_enum::ThomasLUSolver, fp_type, container, allocator>
                host_lus_solver;
            host_lus_solver solver(diag_triplet, fun_triplet, boundary_pair_, discretization_cfg_, solver_cfg_);
            solver(prev_sol, next_sol, rhs, is_heat_source_set, heat_source, solutions);
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
    discretization_config_1d_ptr<fp_type> discretization_cfg_;
    explicit_solver_config_1d_ptr solver_cfg_;

    explicit general_svc_heat_equation() = delete;

    void initialize()
    {
        LSS_VERIFY(heat_data_cfg_, "heat_data_config must not be null");
        LSS_VERIFY(discretization_cfg_, "discretization_config must not be null");
        LSS_VERIFY(std::get<0>(boundary_pair_), "boundary_pair.first must not be null");
        LSS_VERIFY(std::get<1>(boundary_pair_), "boundary_pair.second must not be null");
        LSS_VERIFY(solver_cfg_, "solver_config must not be null");
    }

  public:
    explicit general_svc_heat_equation(
        heat_data_config_1d_ptr<fp_type> const &heat_data_config,
        discretization_config_1d_ptr<fp_type> const &discretization_config,
        boundary_1d_pair<fp_type> const &boundary_pair,
        explicit_solver_config_1d_ptr const &solver_config = dev_expl_fwd_euler_solver_config_ptr)
        : heat_data_cfg_{heat_data_config}, discretization_cfg_{discretization_config}, boundary_pair_{boundary_pair},
          solver_cfg_{solver_config}
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
    void solve(container_2d<fp_type, container, allocator> &solutions);
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
    // calculate scheme coefficients:
    const fp_type one = static_cast<fp_type>(1.0);
    const fp_type two = static_cast<fp_type>(2.0);
    const fp_type half = static_cast<fp_type>(0.5);
    const fp_type lambda = k / (h * h);
    const fp_type gamma = k / (two * h);
    const fp_type delta = half * k;
    // save coefficients:
    auto const &a = heat_data_cfg_->a_coefficient();
    auto const &b = heat_data_cfg_->b_coefficient();
    auto const &c = heat_data_cfg_->c_coefficient();
    // prepare space variable coefficients:
    auto const &A = [&](fp_type x) { return (lambda * a(x) - gamma * b(x)); };
    auto const &B = [&](fp_type x) { return (lambda * a(x) - delta * c(x)); };
    auto const &D = [&](fp_type x) { return (lambda * a(x) + gamma * b(x)); };
    // wrap up the functions into tuple:
    auto const &fun_triplet = std::make_tuple(A, B, D);
    // create container to carry previous solution:
    container_t prev_sol(space_size, fp_type{});
    // discretize initial condition
    d_1d::of_function(space.lower(), h, heat_data_cfg_->initial_condition(), prev_sol);
    // get heat_source:
    const bool is_heat_source_set = heat_data_cfg_->is_heat_source_set();
    // get heat_source:
    auto const &heat_source = heat_data_cfg_->heat_source();

    if (solver_cfg_->memory_space() == memory_space_enum::Device)
    {
        typedef general_svc_heat_equation_explicit_kernel<memory_space_enum::Device, fp_type, container, allocator>
            device_solver;
        device_solver solver(fun_triplet, boundary_pair_, discretization_cfg_, solver_cfg_);
        solver(prev_sol, is_heat_source_set, heat_source);
        std::copy(prev_sol.begin(), prev_sol.end(), solution.begin());
    }
    else if (solver_cfg_->memory_space() == memory_space_enum::Host)
    {
        typedef general_svc_heat_equation_explicit_kernel<memory_space_enum::Host, fp_type, container, allocator>
            host_solver;
        host_solver solver(fun_triplet, boundary_pair_, discretization_cfg_, solver_cfg_);
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
    container_2d<fp_type, container, allocator> &solutions)
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
    // calculate scheme coefficients:
    const fp_type one = static_cast<fp_type>(1.0);
    const fp_type two = static_cast<fp_type>(2.0);
    const fp_type half = static_cast<fp_type>(0.5);
    const fp_type lambda = k / (h * h);
    const fp_type gamma = k / (two * h);
    const fp_type delta = half * k;
    // save coefficients:
    auto const &a = heat_data_cfg_->a_coefficient();
    auto const &b = heat_data_cfg_->b_coefficient();
    auto const &c = heat_data_cfg_->c_coefficient();
    // prepare space variable coefficients:
    auto const &A = [&](fp_type x) { return (lambda * a(x) - gamma * b(x)); };
    auto const &B = [&](fp_type x) { return (lambda * a(x) - delta * c(x)); };
    auto const &D = [&](fp_type x) { return (lambda * a(x) + gamma * b(x)); };
    // wrap up the functions into tuple:
    auto const &fun_triplet = std::make_tuple(A, B, D);
    // create container to carry previous solution:
    container_t prev_sol(space_size, fp_type{});
    // discretize initial condition
    d_1d::of_function(space.lower(), h, heat_data_cfg_->initial_condition(), prev_sol);
    // get heat_source:
    const bool is_heat_source_set = heat_data_cfg_->is_heat_source_set();
    // get heat_source:
    auto const &heat_source = heat_data_cfg_->heat_source();

    if (solver_cfg_->memory_space() == memory_space_enum::Device)
    {
        typedef general_svc_heat_equation_explicit_kernel<memory_space_enum::Device, fp_type, container, allocator>
            device_solver;
        device_solver solver(fun_triplet, boundary_pair_, discretization_cfg_, solver_cfg_);
        solver(prev_sol, is_heat_source_set, heat_source, solutions);
    }
    else if (solver_cfg_->memory_space() == memory_space_enum::Host)
    {
        typedef general_svc_heat_equation_explicit_kernel<memory_space_enum::Host, fp_type, container, allocator>
            host_solver;
        host_solver solver(fun_triplet, boundary_pair_, discretization_cfg_, solver_cfg_);
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

#endif ///_LSS_GENERAL_SVC_HEAT_EQUATION_HPP_
