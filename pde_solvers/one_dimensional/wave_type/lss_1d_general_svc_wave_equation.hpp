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
    boundary_1d_pair<fp_type> boundary_pair_;
    wave_data_config_1d_ptr<fp_type> wave_data_cfg_;
    pde_discretization_config_1d_ptr<fp_type> discretization_cfg_;
    wave_implicit_solver_config_ptr solver_cfg_;
    std::map<std::string, fp_type> solver_config_details_;

    explicit general_svc_wave_equation() = delete;

    void initialize()
    {
        LSS_VERIFY(wave_data_cfg_, "wave_data_config must not be null");
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
    explicit general_svc_wave_equation(
        wave_data_config_1d_ptr<fp_type> const &wave_data_config,
        pde_discretization_config_1d_ptr<fp_type> const &discretization_config,
        boundary_1d_pair<fp_type> const &boundary_pair,
        wave_implicit_solver_config_ptr const &solver_config =
            default_wave_solver_configs::dev_fwd_cusolver_qr_solver_config_ptr,
        std::map<std::string, fp_type> const &solver_config_details = std::map<std::string, fp_type>())
        : wave_data_cfg_{wave_data_config}, discretization_cfg_{discretization_config}, boundary_pair_{boundary_pair},
          solver_cfg_{solver_config}, solver_config_details_{solver_config_details}
    {
        initialize();
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
    void solve(container_2d<fp_type, container, allocator> &solutions);
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void general_svc_wave_equation<fp_type, container, allocator>::solve(container<fp_type, allocator> &solution)
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
    const fp_type quater = static_cast<fp_type>(0.25);
    const fp_type lambda = one / (k * k);
    const fp_type gamma = one / (two * k);
    const fp_type delta = one / (h * h);
    const fp_type rho = one / (two * h);
    // create container to carry previous solution:
    container_t prev_sol_0(space_size, fp_type{});
    // discretize initial condition
    d_1d::of_function(space.lower(), h, wave_data_cfg_->first_initial_condition(), prev_sol_0);
    // create container to carry second initial condition:
    container_t prev_sol_1(space_size, fp_type{});
    // discretize first order initial condition:
    d_1d::of_function(space.lower(), h, wave_data_cfg_->second_initial_condition(), prev_sol_1);
    // since coefficients are different in space :
    container_t low_1(space_size, fp_type{});
    container_t diag_1(space_size, fp_type{});
    container_t up_1(space_size, fp_type{});
    // since coefficients are different in space :
    container_t low_0(space_size, fp_type{});
    container_t diag_0(space_size, fp_type{});
    container_t up_0(space_size, fp_type{});
    // save coefficients:
    auto const &a = wave_data_cfg_->a_coefficient();
    auto const &b = wave_data_cfg_->b_coefficient();
    auto const &c = wave_data_cfg_->c_coefficient();
    auto const &d = wave_data_cfg_->d_coefficient();
    // prepare space variable coefficients:
    auto const &A = [&](fp_type x) { return quater * (delta * b(x) - rho * c(x)); };
    auto const &B = [&](fp_type x) { return quater * (delta * b(x) + rho * c(x)); };
    auto const &C = [&](fp_type x) { return half * (delta * b(x) - half * d(x)); };
    auto const &D = [&](fp_type x) { return (lambda - gamma * a(x)); };
    auto const &E = [&](fp_type x) { return (lambda + gamma * a(x)); };
    // wrap up the functions into tuple:
    auto const &fun_quintaple = std::make_tuple(A, B, C, D, E);
    fp_type m{};
    fp_type up{};
    fp_type mid{};
    fp_type down{};
    for (std::size_t t = 0; t < low_0.size(); ++t)
    {
        m = static_cast<fp_type>(t);
        down = -one * A(m * h);
        mid = (E(m * h) + C(m * h));
        up = -one * B(m * h);
        low_1[t] = down;
        diag_1[t] = mid;
        up_1[t] = up;

        low_0[t] = two * down;
        diag_0[t] = (mid + C(m * h) + D(m * h));
        up_0[t] = two * up;
    }
    // wrap up the diagonals into tuple:
    auto const &diag_triplet_0 = std::make_tuple(low_0, diag_0, up_0);
    auto const &diag_triplet_1 = std::make_tuple(low_1, diag_1, up_1);
    auto const &diag_triplets = std::make_pair(diag_triplet_0, diag_triplet_1);
    container_t rhs(space_size, fp_type{});
    // create container to carry new solution:
    container_t next_sol(space_size, fp_type{});
    // get heat_source:
    const bool is_wave_source_set = wave_data_cfg_->is_wave_source_set();
    // get heat_source:
    auto const &wave_source = wave_data_cfg_->wave_source();

    if (solver_cfg_->memory_space() == memory_space_enum::Device)
    {
        if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::CUDASolver)
        {
            typedef general_svc_wave_equation_implicit_kernel<
                memory_space_enum::Device, tridiagonal_method_enum::CUDASolver, fp_type, container, allocator>
                dev_cu_solver;

            dev_cu_solver solver(diag_triplets, fun_quintaple, boundary_pair_, discretization_cfg_, solver_cfg_);
            solver(prev_sol_0, prev_sol_1, next_sol, rhs, is_wave_source_set, wave_source);
            std::copy(prev_sol_1.begin(), prev_sol_1.end(), solution.begin());
        }
        else if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::SORSolver)
        {
            typedef general_svc_wave_equation_implicit_kernel<
                memory_space_enum::Device, tridiagonal_method_enum::SORSolver, fp_type, container, allocator>
                dev_sor_solver;
            LSS_ASSERT(!solver_config_details_.empty(), "solver_config_details map must not be empty");
            fp_type omega_value = solver_config_details_["sor_omega"];
            dev_sor_solver solver(diag_triplets, fun_quintaple, boundary_pair_, discretization_cfg_, solver_cfg_);
            solver(prev_sol_0, prev_sol_1, next_sol, rhs, is_wave_source_set, wave_source, omega_value);
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
            host_cu_solver solver(diag_triplets, fun_quintaple, boundary_pair_, discretization_cfg_, solver_cfg_);
            solver(prev_sol_0, prev_sol_1, next_sol, rhs, is_wave_source_set, wave_source);
            std::copy(prev_sol_1.begin(), prev_sol_1.end(), solution.begin());
        }
        else if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::SORSolver)
        {
            typedef general_svc_wave_equation_implicit_kernel<
                memory_space_enum::Host, tridiagonal_method_enum::SORSolver, fp_type, container, allocator>
                host_sor_solver;

            LSS_ASSERT(!solver_config_details_.empty(), "solver_config_details map must not be empty");
            fp_type omega_value = solver_config_details_["sor_omega"];
            host_sor_solver solver(diag_triplets, fun_quintaple, boundary_pair_, discretization_cfg_, solver_cfg_);
            solver(prev_sol_0, prev_sol_1, next_sol, rhs, is_wave_source_set, wave_source, omega_value);
            std::copy(prev_sol_1.begin(), prev_sol_1.end(), solution.begin());
        }
        else if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::DoubleSweepSolver)
        {
            typedef general_svc_wave_equation_implicit_kernel<
                memory_space_enum::Host, tridiagonal_method_enum::DoubleSweepSolver, fp_type, container, allocator>
                host_dss_solver;
            host_dss_solver solver(diag_triplets, fun_quintaple, boundary_pair_, discretization_cfg_, solver_cfg_);
            solver(prev_sol_0, prev_sol_1, next_sol, rhs, is_wave_source_set, wave_source);
            std::copy(prev_sol_1.begin(), prev_sol_1.end(), solution.begin());
        }
        else if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::ThomasLUSolver)
        {
            typedef general_svc_wave_equation_implicit_kernel<
                memory_space_enum::Host, tridiagonal_method_enum::ThomasLUSolver, fp_type, container, allocator>
                host_lus_solver;
            host_lus_solver solver(diag_triplets, fun_quintaple, boundary_pair_, discretization_cfg_, solver_cfg_);
            solver(prev_sol_0, prev_sol_1, next_sol, rhs, is_wave_source_set, wave_source);
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
    const fp_type quater = static_cast<fp_type>(0.25);
    const fp_type lambda = one / (k * k);
    const fp_type gamma = one / (two * k);
    const fp_type delta = one / (h * h);
    const fp_type rho = one / (two * h);
    // create container to carry previous solution:
    container_t prev_sol_0(space_size, fp_type{});
    // discretize initial condition
    d_1d::of_function(space.lower(), h, wave_data_cfg_->first_initial_condition(), prev_sol_0);
    // create container to carry second initial condition:
    container_t prev_sol_1(space_size, fp_type{});
    // discretize first order initial condition:
    d_1d::of_function(space.lower(), h, wave_data_cfg_->second_initial_condition(), prev_sol_1);
    // since coefficients are different in space :
    container_t low_1(space_size, fp_type{});
    container_t diag_1(space_size, fp_type{});
    container_t up_1(space_size, fp_type{});
    // since coefficients are different in space :
    container_t low_0(space_size, fp_type{});
    container_t diag_0(space_size, fp_type{});
    container_t up_0(space_size, fp_type{});
    // save coefficients:
    auto const &a = wave_data_cfg_->a_coefficient();
    auto const &b = wave_data_cfg_->b_coefficient();
    auto const &c = wave_data_cfg_->c_coefficient();
    auto const &d = wave_data_cfg_->d_coefficient();
    // prepare space variable coefficients:
    auto const &A = [&](fp_type x) { return quater * (delta * b(x) - rho * c(x)); };
    auto const &B = [&](fp_type x) { return quater * (delta * b(x) + rho * c(x)); };
    auto const &C = [&](fp_type x) { return half * (delta * b(x) - half * d(x)); };
    auto const &D = [&](fp_type x) { return (lambda - gamma * a(x)); };
    auto const &E = [&](fp_type x) { return (lambda + gamma * a(x)); };
    // wrap up the functions into tuple:
    auto const &fun_quintaple = std::make_tuple(A, B, C, D, E);
    fp_type m{};
    fp_type up{};
    fp_type mid{};
    fp_type down{};
    for (std::size_t t = 0; t < low_0.size(); ++t)
    {
        m = static_cast<fp_type>(t);
        down = -one * A(m * h);
        mid = (E(m * h) + C(m * h));
        up = -one * B(m * h);
        low_1[t] = down;
        diag_1[t] = mid;
        up_1[t] = up;

        low_0[t] = two * down;
        diag_0[t] = (mid + C(m * h) + D(m * h));
        up_0[t] = two * up;
    }
    // wrap up the diagonals into tuple:
    auto const &diag_triplet_0 = std::make_tuple(low_0, diag_0, up_0);
    auto const &diag_triplet_1 = std::make_tuple(low_1, diag_1, up_1);
    auto const &diag_triplets = std::make_pair(diag_triplet_0, diag_triplet_1);
    container_t rhs(space_size, fp_type{});
    // create container to carry new solution:
    container_t next_sol(space_size, fp_type{});
    // get heat_source:
    const bool is_wave_source_set = wave_data_cfg_->is_wave_source_set();
    // get heat_source:
    auto const &wave_source = wave_data_cfg_->wave_source();

    if (solver_cfg_->memory_space() == memory_space_enum::Device)
    {
        if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::CUDASolver)
        {
            typedef general_svc_wave_equation_implicit_kernel<
                memory_space_enum::Device, tridiagonal_method_enum::CUDASolver, fp_type, container, allocator>
                dev_cu_solver;

            dev_cu_solver solver(diag_triplets, fun_quintaple, boundary_pair_, discretization_cfg_, solver_cfg_);
            solver(prev_sol_0, prev_sol_1, next_sol, rhs, is_wave_source_set, wave_source, solutions);
        }
        else if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::SORSolver)
        {
            typedef general_svc_wave_equation_implicit_kernel<
                memory_space_enum::Device, tridiagonal_method_enum::SORSolver, fp_type, container, allocator>
                dev_sor_solver;
            LSS_ASSERT(!solver_config_details_.empty(), "solver_config_details map must not be empty");
            fp_type omega_value = solver_config_details_["sor_omega"];
            dev_sor_solver solver(diag_triplets, fun_quintaple, boundary_pair_, discretization_cfg_, solver_cfg_);
            solver(prev_sol_0, prev_sol_1, next_sol, rhs, is_wave_source_set, wave_source, omega_value, solutions);
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
            host_cu_solver solver(diag_triplets, fun_quintaple, boundary_pair_, discretization_cfg_, solver_cfg_);
            solver(prev_sol_0, prev_sol_1, next_sol, rhs, is_wave_source_set, wave_source, solutions);
        }
        else if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::SORSolver)
        {
            typedef general_svc_wave_equation_implicit_kernel<
                memory_space_enum::Host, tridiagonal_method_enum::SORSolver, fp_type, container, allocator>
                host_sor_solver;

            LSS_ASSERT(!solver_config_details_.empty(), "solver_config_details map must not be empty");
            fp_type omega_value = solver_config_details_["sor_omega"];
            host_sor_solver solver(diag_triplets, fun_quintaple, boundary_pair_, discretization_cfg_, solver_cfg_);
            solver(prev_sol_0, prev_sol_1, next_sol, rhs, is_wave_source_set, wave_source, omega_value, solutions);
        }
        else if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::DoubleSweepSolver)
        {
            typedef general_svc_wave_equation_implicit_kernel<
                memory_space_enum::Host, tridiagonal_method_enum::DoubleSweepSolver, fp_type, container, allocator>
                host_dss_solver;
            host_dss_solver solver(diag_triplets, fun_quintaple, boundary_pair_, discretization_cfg_, solver_cfg_);
            solver(prev_sol_0, prev_sol_1, next_sol, rhs, is_wave_source_set, wave_source, solutions);
        }
        else if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::ThomasLUSolver)
        {
            typedef general_svc_wave_equation_implicit_kernel<
                memory_space_enum::Host, tridiagonal_method_enum::ThomasLUSolver, fp_type, container, allocator>
                host_lus_solver;
            host_lus_solver solver(diag_triplets, fun_quintaple, boundary_pair_, discretization_cfg_, solver_cfg_);
            solver(prev_sol_0, prev_sol_1, next_sol, rhs, is_wave_source_set, wave_source, solutions);
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
    boundary_1d_pair<fp_type> boundary_pair_;
    wave_data_config_1d_ptr<fp_type> wave_data_cfg_;
    pde_discretization_config_1d_ptr<fp_type> discretization_cfg_;
    wave_explicit_solver_config_ptr solver_cfg_;

    explicit general_svc_wave_equation() = delete;

    void initialize()
    {
        LSS_VERIFY(wave_data_cfg_, "wave_data_config must not be null");
        LSS_VERIFY(discretization_cfg_, "discretization_config must not be null");
        LSS_VERIFY(std::get<0>(boundary_pair_), "boundary_pair.first must not be null");
        LSS_VERIFY(std::get<1>(boundary_pair_), "boundary_pair.second must not be null");
        LSS_VERIFY(solver_cfg_, "solver_config must not be null");
    }

  public:
    explicit general_svc_wave_equation(wave_data_config_1d_ptr<fp_type> const &wave_data_config,
                                       pde_discretization_config_1d_ptr<fp_type> const &discretization_config,
                                       boundary_1d_pair<fp_type> const &boundary_pair,
                                       wave_explicit_solver_config_ptr const &solver_config =
                                           default_wave_solver_configs::dev_expl_fwd_solver_config_ptr)
        : wave_data_cfg_{wave_data_config}, discretization_cfg_{discretization_config}, boundary_pair_{boundary_pair},
          solver_cfg_{solver_config}
    {
        initialize();
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
    void solve(container_2d<fp_type, container, allocator> &solutions);
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void general_svc_wave_equation<fp_type, container, allocator>::solve(container<fp_type, allocator> &solution)
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
    const fp_type lambda = one / (k * k);
    const fp_type gamma = one / (two * k);
    const fp_type delta = one / (h * h);
    const fp_type rho = one / (two * h);
    // create container to carry previous solution:
    container_t prev_sol_0(space_size, fp_type{});
    // discretize initial condition
    d_1d::of_function(space.lower(), h, wave_data_cfg_->first_initial_condition(), prev_sol_0);
    // create container to carry second initial condition:
    container_t prev_sol_1(space_size, fp_type{});
    // discretize first order initial condition:
    d_1d::of_function(space.lower(), h, wave_data_cfg_->second_initial_condition(), prev_sol_1);
    // save coefficients:
    auto const &a = wave_data_cfg_->a_coefficient();
    auto const &b = wave_data_cfg_->b_coefficient();
    auto const &c = wave_data_cfg_->c_coefficient();
    auto const &d = wave_data_cfg_->d_coefficient();
    // get wave_source:
    auto const &wave_source = wave_data_cfg_->wave_source();
    // prepare space variable coefficients:
    auto const &E = [&](fp_type x) { return (lambda + a(x) * gamma); };
    auto const &A = [&](fp_type x) { return ((delta * b(x) - rho * c(x)) / E(x)); };
    auto const &B = [&](fp_type x) { return ((delta * b(x) + rho * c(x)) / E(x)); };
    auto const &C = [&](fp_type x) { return ((two * (lambda - (delta * b(x) - half * d(x)))) / E(x)); };
    auto const &D = [&](fp_type x) { return ((lambda - a(x) * gamma) / E(x)); };
    auto const &wave_source_modified = [&](fp_type x, fp_type t) { return (wave_source(x, t) / E(x)); };
    // wrap up the functions into tuple:
    auto const &fun_quintuple = std::make_tuple(A, B, C, D, b);
    // is wave_source set:
    const bool is_wave_source_set = wave_data_cfg_->is_wave_source_set();
    // create container to carry new solution:
    container_t next_sol(space_size, fp_type{});

    if (solver_cfg_->memory_space() == memory_space_enum::Device)
    {
        typedef general_svc_wave_equation_explicit_kernel<memory_space_enum::Device, fp_type, container, allocator>
            device_solver;
        device_solver solver(fun_quintuple, boundary_pair_, discretization_cfg_, solver_cfg_);
        solver(prev_sol_0, prev_sol_1, next_sol, is_wave_source_set, wave_source_modified);
        std::copy(next_sol.begin(), next_sol.end(), solution.begin());
    }
    else if (solver_cfg_->memory_space() == memory_space_enum::Host)
    {
        typedef general_svc_wave_equation_explicit_kernel<memory_space_enum::Host, fp_type, container, allocator>
            host_solver;
        host_solver solver(fun_quintuple, boundary_pair_, discretization_cfg_, solver_cfg_);
        solver(prev_sol_0, prev_sol_1, next_sol, is_wave_source_set, wave_source_modified);
        std::copy(prev_sol_1.begin(), prev_sol_1.end(), solution.begin());
    }
    else
    {
        throw std::exception("Unreachable");
    }
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void general_svc_wave_equation<fp_type, container, allocator>::solve(
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
    const fp_type lambda = one / (k * k);
    const fp_type gamma = one / (two * k);
    const fp_type delta = one / (h * h);
    const fp_type rho = one / (two * h);
    // create container to carry previous solution:
    container_t prev_sol_0(space_size, fp_type{});
    // discretize initial condition
    d_1d::of_function(space.lower(), h, wave_data_cfg_->first_initial_condition(), prev_sol_0);
    // create container to carry second initial condition:
    container_t prev_sol_1(space_size, fp_type{});
    // discretize first order initial condition:
    d_1d::of_function(space.lower(), h, wave_data_cfg_->second_initial_condition(), prev_sol_1);
    // save coefficients:
    auto const &a = wave_data_cfg_->a_coefficient();
    auto const &b = wave_data_cfg_->b_coefficient();
    auto const &c = wave_data_cfg_->c_coefficient();
    auto const &d = wave_data_cfg_->d_coefficient();
    // get wave_source:
    auto const &wave_source = wave_data_cfg_->wave_source();
    // prepare space variable coefficients:
    auto const &E = [&](fp_type x) { return (lambda + a(x) * gamma); };
    auto const &A = [&](fp_type x) { return ((delta * b(x) - rho * c(x)) / E(x)); };
    auto const &B = [&](fp_type x) { return ((delta * b(x) + rho * c(x)) / E(x)); };
    auto const &C = [&](fp_type x) { return ((two * (lambda - (delta * b(x) - half * d(x)))) / E(x)); };
    auto const &D = [&](fp_type x) { return ((lambda - a(x) * gamma) / E(x)); };
    auto const &wave_source_modified = [&](fp_type x, fp_type t) { return (wave_source(x, t) / E(x)); };
    // wrap up the functions into tuple:
    auto const &fun_quintuple = std::make_tuple(A, B, C, D, b);
    // is wave_source set:
    const bool is_wave_source_set = wave_data_cfg_->is_wave_source_set();
    // create container to carry new solution:
    container_t next_sol(space_size, fp_type{});

    if (solver_cfg_->memory_space() == memory_space_enum::Device)
    {
        typedef general_svc_wave_equation_explicit_kernel<memory_space_enum::Device, fp_type, container, allocator>
            device_solver;
        device_solver solver(fun_quintuple, boundary_pair_, discretization_cfg_, solver_cfg_);
        solver(prev_sol_0, prev_sol_1, next_sol, is_wave_source_set, wave_source_modified, solutions);
    }
    else if (solver_cfg_->memory_space() == memory_space_enum::Host)
    {
        typedef general_svc_wave_equation_explicit_kernel<memory_space_enum::Host, fp_type, container, allocator>
            host_solver;
        host_solver solver(fun_quintuple, boundary_pair_, discretization_cfg_, solver_cfg_);
        solver(prev_sol_0, prev_sol_1, next_sol, is_wave_source_set, wave_source_modified, solutions);
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
