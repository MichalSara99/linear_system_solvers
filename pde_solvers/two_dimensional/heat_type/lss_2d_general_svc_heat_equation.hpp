#if !defined(_LSS_2D_GENERAL_SVC_HEAT_EQUATION_HPP_)
#define _LSS_2D_GENERAL_SVC_HEAT_EQUATION_HPP_

#include <functional>
#include <map>

#include "boundaries/lss_boundary.hpp"
#include "common/lss_macros.hpp"
#include "containers/lss_container_2d.hpp"
//#include "lss_general_svc_heat_equation_explicit_kernel.hpp"
//#include "lss_general_svc_heat_equation_implicit_kernel.hpp"
#include "pde_solvers/lss_discretization.hpp"
#include "pde_solvers/lss_discretization_config.hpp"
#include "pde_solvers/lss_heat_data_config.hpp"
#include "pde_solvers/lss_solver_config.hpp"

namespace lss_pde_solvers
{

namespace two_dimensional
{
using lss_boundary::boundary_1d_pair;
using lss_boundary::boundary_1d_ptr;
using lss_containers::container_2d;

namespace implicit_solvers
{

/*!
============================================================================
Represents general spacial variable coefficient 2D heat equation solver

u_t = a(x,y)*u_xx + b(x,y)*u_yy + c(x,y)*u_xy + d(x,y)*u_x + e(x,y)*u_y +
        f(x,y)*u + F(x,y,t)

t > 0, x_1 < x < x_2, y_1 < y < y_2

with initial condition:

u(x,y,0) = G(x,y)

or terminal condition:

u(x,y,T) = G(x,y)


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
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void general_svc_heat_equation<fp_type, container, allocator>::solve(
    container_2d<fp_type, container, allocator> &solutions)
{
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
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void general_svc_heat_equation<fp_type, container, allocator>::solve(
    container_2d<fp_type, container, allocator> &solutions)
{
}

} // namespace explicit_solvers

} // namespace two_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_2D_GENERAL_SVC_HEAT_EQUATION_HPP_
