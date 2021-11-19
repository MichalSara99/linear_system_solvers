#if !defined(_LSS_2D_GENERAL_SVC_HESTON_EQUATION_HPP_)
#define _LSS_2D_GENERAL_SVC_HESTON_EQUATION_HPP_

#include <functional>
#include <map>

#include "boundaries/lss_boundary.hpp"
#include "boundaries/lss_dirichlet_boundary.hpp"
#include "boundaries/lss_neumann_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "containers/lss_container_2d.hpp"
#include "containers/lss_container_3d.hpp"
//#include "lss_general_svc_heat_equation_explicit_kernel.hpp"
#include "discretization/lss_discretization.hpp"
#include "discretization/lss_grid.hpp"
#include "discretization/lss_grid_config.hpp"
#include "discretization/lss_grid_config_hints.hpp"
#include "lss_2d_general_svc_heston_equation_implicit_kernel.hpp"
#include "pde_solvers/lss_heat_data_config.hpp"
#include "pde_solvers/lss_heat_solver_config.hpp"
#include "pde_solvers/lss_pde_discretization_config.hpp"
#include "pde_solvers/lss_splitting_method_config.hpp"
#include "pde_solvers/lss_weighted_scheme_config.hpp"

namespace lss_pde_solvers
{

namespace two_dimensional
{
using lss_boundary::boundary_2d_pair;
using lss_boundary::boundary_2d_ptr;
using lss_boundary::dirichlet_boundary_2d;
using lss_boundary::neumann_boundary_2d;
using lss_containers::container_2d;
using lss_containers::container_3d;
using lss_enumerations::by_enum;
using lss_enumerations::grid_enum;
using lss_grids::grid_config_hints_2d_ptr;
using lss_grids::nonuniform_grid_config_2d;
using lss_grids::uniform_grid_config_2d;

namespace implicit_solvers
{

/*!
============================================================================
Represents general spacial variable coefficient Heston type equation

u_t = a(x,y)*u_xx + b(x,y)*u_yy + c(x,y)*u_xy + d(x,y)*u_x + e(x,y)*u_y +
        f(x,y)*u + F(t,x,y)

t > 0, x_1 < x < x_2, y_1 < y < y_2

with initial condition:

u(x,y,0) = G(x,y)

or terminal condition:

u(x,y,T) = G(x,y)

horizontal_boundary_pair = S = (S_1,S_2) boundary

             vol (Y)
        ________________
        |S_1,S_1,S_1,S_1|
        |               |
        |               |
S (X)   |               |
        |               |
        |               |
        |               |
        |               |
        |S_2,S_2,S_2,S_2|
        |_______________|

// ============================================================================
*/
template <typename fp_type, template <typename, typename> typename container = std::vector,
          typename allocator = std::allocator<fp_type>>
class general_svc_heston_equation
{

  private:
    heat_data_config_2d_ptr<fp_type> heat_data_cfg_;
    pde_discretization_config_2d_ptr<fp_type> discretization_cfg_;
    boundary_2d_ptr<fp_type> boundary_ver_;
    boundary_2d_pair<fp_type> boundary_pair_hor_;
    splitting_method_config_ptr<fp_type> splitting_method_cfg_;
    weighted_scheme_config_ptr<fp_type> weighted_scheme_cfg_;
    grid_config_hints_2d_ptr<fp_type> grid_hints_cfg_;
    heat_implicit_solver_config_ptr solver_cfg_;
    std::map<std::string, fp_type> solver_config_details_;

    explicit general_svc_heston_equation() = delete;

    void initialize()
    {
        LSS_VERIFY(heat_data_cfg_, "heat_data_config must not be null");
        LSS_VERIFY(discretization_cfg_, "discretization_config must not be null");

        if (auto ver_ptr = std::dynamic_pointer_cast<dirichlet_boundary_2d<fp_type>>(boundary_ver_))
        {
            LSS_VERIFY(ver_ptr, "vertical_upper_boundary_ptr must be of dirichlet type only");
        }

        if (auto hor_ptr = std::dynamic_pointer_cast<dirichlet_boundary_2d<fp_type>>(std::get<0>(boundary_pair_hor_)))
        {
            LSS_VERIFY(hor_ptr, "horizontal_boundary_pair.first must be of dirichlet type only");
        }
        if (auto hor_ptr = std::dynamic_pointer_cast<neumann_boundary_2d<fp_type>>(std::get<1>(boundary_pair_hor_)))
        {
            LSS_VERIFY(hor_ptr, "horizontal_boundary_pair.second must be of neumann type only");
        }

        LSS_VERIFY(splitting_method_cfg_, "splitting_method_config must not be null");
        LSS_VERIFY(weighted_scheme_cfg_, "weighted_scheme_config must not be null");
        LSS_VERIFY(solver_cfg_, "solver_config must not be null");
        LSS_VERIFY(grid_hints_cfg_, "grid_config_hints must not be null");
        if (!solver_config_details_.empty())
        {
            auto const &it = solver_config_details_.find("sor_omega");
            LSS_ASSERT(it != solver_config_details_.end(), "sor_omega is not defined");
        }
    }

  public:
    explicit general_svc_heston_equation(
        heat_data_config_2d_ptr<fp_type> const &heat_data_config,
        pde_discretization_config_2d_ptr<fp_type> const &discretization_config,
        boundary_2d_ptr<fp_type> const &vertical_upper_boundary_ptr,
        boundary_2d_pair<fp_type> const &horizontal_boundary_pair,
        splitting_method_config_ptr<fp_type> const &splitting_method_config,
        weighted_scheme_config_ptr<fp_type> const &weighted_scheme_config,
        grid_config_hints_2d_ptr<fp_type> const &grid_config_hints,
        heat_implicit_solver_config_ptr const &solver_config =
            default_heat_solver_configs::host_fwd_dssolver_euler_solver_config_ptr,
        std::map<std::string, fp_type> const &solver_config_details = std::map<std::string, fp_type>())
        : heat_data_cfg_{heat_data_config}, discretization_cfg_{discretization_config},
          boundary_ver_{vertical_upper_boundary_ptr}, boundary_pair_hor_{horizontal_boundary_pair},
          splitting_method_cfg_{splitting_method_config}, weighted_scheme_cfg_{weighted_scheme_config},
          grid_hints_cfg_{grid_config_hints}, solver_cfg_{solver_config}, solver_config_details_{solver_config_details}
    {
        initialize();
    }

    ~general_svc_heston_equation()
    {
    }

    general_svc_heston_equation(general_svc_heston_equation const &) = delete;
    general_svc_heston_equation(general_svc_heston_equation &&) = delete;
    general_svc_heston_equation &operator=(general_svc_heston_equation const &) = delete;
    general_svc_heston_equation &operator=(general_svc_heston_equation &&) = delete;

    /**
     * Get the final solution of the PDE
     *
     * \param solution -  2D container for solution
     */
    void solve(container_2d<by_enum::Row, fp_type, container, allocator> &solution);

    /**
     * Get all solutions in time (surface) of the PDE
     *
     * \param solutions - 3D container for all the solutions in time
     */
    void solve(container_3d<by_enum::Row, fp_type, container, allocator> &solutions);
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void general_svc_heston_equation<fp_type, container, allocator>::solve(
    container_2d<by_enum::Row, fp_type, container, allocator> &solution)
{
    typedef discretization<dimension_enum::Two, fp_type, container, allocator> d_2d;
    typedef container_2d<by_enum::Row, fp_type, container, allocator> rcontainer_2d_t;
    typedef container<fp_type, allocator> container_t;

    LSS_ASSERT((solution.rows()) > 0 && (solution.columns() > 0), "The input solution container must be initialized");

    // get space ranges:
    const auto &spaces = discretization_cfg_->space_range();
    // across X:
    const auto space_x = spaces.first;
    // across Y:
    const auto space_y = spaces.second;
    // get space steps:
    const auto &hs = discretization_cfg_->space_step();
    // size of spaces discretization:
    const auto &space_sizes = discretization_cfg_->number_of_space_points();
    const std::size_t space_size_x = std::get<0>(space_sizes);
    const std::size_t space_size_y = std::get<1>(space_sizes);
    // This is the proper size of the container:
    LSS_ASSERT((solution.columns() == space_size_y) && (solution.rows() == space_size_x),
               "The input solution container must have the correct size");
    // grid:
    grid_config_2d_ptr<fp_type> grid_cfg;
    if (grid_hints_cfg_->grid() == grid_enum::Uniform)
    {
        grid_cfg = std::make_shared<uniform_grid_config_2d<fp_type>>(discretization_cfg_);
    }
    else if (grid_hints_cfg_->grid() == grid_enum::Nonuniform)
    {
        grid_cfg = std::make_shared<nonuniform_grid_config_2d<fp_type>>(
            discretization_cfg_, grid_hints_cfg_->strike(), grid_hints_cfg_->p_scale(), grid_hints_cfg_->c_scale(),
            grid_hints_cfg_->d_scale());
    }
    else
    {
        throw std::exception("Unreachable");
    }

    // create container to carry previous solution:
    rcontainer_2d_t prev_sol(space_size_x, space_size_y, fp_type{});
    // create container to carry next solution:
    rcontainer_2d_t next_sol(space_size_x, space_size_y, fp_type{});
    // discretize initial condition
    d_2d::of_function(grid_cfg, heat_data_cfg_->initial_condition(), prev_sol);
    // get heat_source:
    const bool is_heat_source_set = heat_data_cfg_->is_heat_source_set();
    // get heat_source:
    auto const &heat_source = heat_data_cfg_->heat_source();

    if (solver_cfg_->memory_space() == memory_space_enum::Device)
    {
        if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::CUDASolver)
        {
            typedef general_svc_heston_equation_implicit_kernel<
                memory_space_enum::Device, tridiagonal_method_enum::CUDASolver, fp_type, container, allocator>
                dev_cu_solver;

            dev_cu_solver solver(boundary_ver_, boundary_pair_hor_, heat_data_cfg_, discretization_cfg_,
                                 splitting_method_cfg_, weighted_scheme_cfg_, solver_cfg_, grid_cfg);
            solver(prev_sol, next_sol, is_heat_source_set, heat_source);
            solution = prev_sol;
        }
        else if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::SORSolver)
        {
            typedef general_svc_heston_equation_implicit_kernel<
                memory_space_enum::Device, tridiagonal_method_enum::SORSolver, fp_type, container, allocator>
                dev_sor_solver;

            LSS_ASSERT(!solver_config_details_.empty(), "solver_config_details map must not be empty");
            fp_type omega_value = solver_config_details_["sor_omega"];
            dev_sor_solver solver(boundary_ver_, boundary_pair_hor_, heat_data_cfg_, discretization_cfg_,
                                  splitting_method_cfg_, weighted_scheme_cfg_, solver_cfg_, grid_cfg);
            solver(prev_sol, next_sol, is_heat_source_set, heat_source, omega_value);
            solution = prev_sol;
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
            typedef general_svc_heston_equation_implicit_kernel<
                memory_space_enum::Host, tridiagonal_method_enum::CUDASolver, fp_type, container, allocator>
                host_cu_solver;

            host_cu_solver solver(boundary_ver_, boundary_pair_hor_, heat_data_cfg_, discretization_cfg_,
                                  splitting_method_cfg_, weighted_scheme_cfg_, solver_cfg_, grid_cfg);
            solver(prev_sol, next_sol, is_heat_source_set, heat_source);
            solution = next_sol;
        }
        else if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::SORSolver)
        {
            typedef general_svc_heston_equation_implicit_kernel<
                memory_space_enum::Host, tridiagonal_method_enum::SORSolver, fp_type, container, allocator>
                host_sor_solver;

            LSS_ASSERT(!solver_config_details_.empty(), "solver_config_details map must not be empty");
            fp_type omega_value = solver_config_details_["sor_omega"];
            host_sor_solver solver(boundary_ver_, boundary_pair_hor_, heat_data_cfg_, discretization_cfg_,
                                   splitting_method_cfg_, weighted_scheme_cfg_, solver_cfg_, grid_cfg);
            solver(prev_sol, next_sol, is_heat_source_set, heat_source, omega_value);
            solution = next_sol;
        }
        else if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::DoubleSweepSolver)
        {
            typedef general_svc_heston_equation_implicit_kernel<
                memory_space_enum::Host, tridiagonal_method_enum::DoubleSweepSolver, fp_type, container, allocator>
                host_dss_solver;
            host_dss_solver solver(boundary_ver_, boundary_pair_hor_, heat_data_cfg_, discretization_cfg_,
                                   splitting_method_cfg_, weighted_scheme_cfg_, solver_cfg_, grid_cfg);
            solver(prev_sol, next_sol, is_heat_source_set, heat_source);
            solution = next_sol;
        }
        else if (solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::ThomasLUSolver)
        {
            typedef general_svc_heston_equation_implicit_kernel<
                memory_space_enum::Host, tridiagonal_method_enum::ThomasLUSolver, fp_type, container, allocator>
                host_lus_solver;
            host_lus_solver solver(boundary_ver_, boundary_pair_hor_, heat_data_cfg_, discretization_cfg_,
                                   splitting_method_cfg_, weighted_scheme_cfg_, solver_cfg_, grid_cfg);
            solver(prev_sol, next_sol, is_heat_source_set, heat_source);
            solution = next_sol;
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
void general_svc_heston_equation<fp_type, container, allocator>::solve(
    container_3d<by_enum::Row, fp_type, container, allocator> &solutions)
{
}

} // namespace implicit_solvers

namespace explicit_solvers
{

/*!
============================================================================
Represents general spacial variable coefficient Heston type equation

u_t = a(x,y)*u_xx + b(x,y)*u_yy + c(x,y)*u_xy + d(x,y)*u_x + e(x,y)*u_y +
f(x,y)*u + F(t,x,y)

t > 0, x_1 < x < x_2, y_1 < y < y_2

with initial condition:

u(x,y,0) = G(x,y)

or terminal condition:

u(x,y,T) = G(x,y)


// ============================================================================
*/

template <typename fp_type, template <typename, typename> typename container = std::vector,
          typename allocator = std::allocator<fp_type>>
class general_svc_heston_equation
{
  private:
    boundary_2d_ptr<fp_type> boundary_hor_;
    boundary_2d_pair<fp_type> boundary_pair_ver_;
    heat_data_config_2d_ptr<fp_type> heat_data_cfg_;
    pde_discretization_config_2d_ptr<fp_type> discretization_cfg_;
    weighted_scheme_config_ptr<fp_type> weighted_scheme_cfg_;
    grid_config_hints_2d_ptr<fp_type> grid_hints_cfg_;
    heat_explicit_solver_config_ptr solver_cfg_;

    explicit general_svc_heston_equation() = delete;

    void initialize()
    {
        LSS_VERIFY(heat_data_cfg_, "heat_data_config must not be null");
        LSS_VERIFY(discretization_cfg_, "discretization_config must not be null");

        if (auto hor_ptr = std::dynamic_pointer_cast<dirichlet_boundary_2d<fp_type>>(boundary_hor_))
        {
            LSS_VERIFY(hor_ptr, "horizontal_upper_boundary_ptr must be of dirichlet type only");
        }

        if (auto ver_ptr = std::dynamic_pointer_cast<dirichlet_boundary_2d<fp_type>>(std::get<0>(boundary_pair_ver_)))
        {
            LSS_VERIFY(ver_ptr, "vertical_boundary_pair.first must be of dirichlet type only");
        }
        if (auto ver_ptr = std::dynamic_pointer_cast<neumann_boundary_2d<fp_type>>(std::get<1>(boundary_pair_ver_)))
        {
            LSS_VERIFY(ver_ptr, "vertical_boundary_pair.second must be of neumann type only");
        }

        LSS_VERIFY(grid_hints_cfg_, "grid_config_hints must not be null");
        LSS_VERIFY(solver_cfg_, "solver_config must not be null");
    }

  public:
    explicit general_svc_heston_equation(heat_data_config_2d_ptr<fp_type> const &heat_data_config,
                                         pde_discretization_config_2d_ptr<fp_type> const &discretization_config,
                                         boundary_2d_ptr<fp_type> const &horizontal_upper_boundary_ptr,
                                         boundary_2d_pair<fp_type> const &vertical_boundary_pair,
                                         weighted_scheme_config_ptr<fp_type> const &weighted_scheme_config,
                                         grid_config_hints_2d_ptr<fp_type> const &grid_config_hints,
                                         heat_explicit_solver_config_ptr const &solver_config =
                                             default_heat_solver_configs::dev_expl_fwd_euler_solver_config_ptr)
        : heat_data_cfg_{heat_data_config}, discretization_cfg_{discretization_config},
          boundary_hor_{horizontal_upper_boundary_ptr}, boundary_pair_ver_{vertical_boundary_pair},
          weighted_scheme_cfg_{weighted_scheme_config}, grid_hints_cfg_{grid_config_hints}, solver_cfg_{solver_config}
    {
        initialize();
    }

    ~general_svc_heston_equation()
    {
    }

    general_svc_heston_equation(general_svc_heston_equation const &) = delete;
    general_svc_heston_equation(general_svc_heston_equation &&) = delete;
    general_svc_heston_equation &operator=(general_svc_heston_equation const &) = delete;
    general_svc_heston_equation &operator=(general_svc_heston_equation &&) = delete;

    /**
     * Get the final solution of the PDE
     *
     * \param solution - 2D container for solution
     */
    void solve(container_2d<by_enum::Row, fp_type, container, allocator> &solution);

    /**
     * Get all solutions in time (surface) of the PDE
     *
     * \param solutions - 3D container for all the solutions in time
     */
    void solve(container_3d<by_enum::Row, fp_type, container, allocator> &solutions);
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void general_svc_heston_equation<fp_type, container, allocator>::solve(
    container_2d<by_enum::Row, fp_type, container, allocator> &solution)
{
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void general_svc_heston_equation<fp_type, container, allocator>::solve(
    container_3d<by_enum::Row, fp_type, container, allocator> &solutions)
{
}

} // namespace explicit_solvers

} // namespace two_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_2D_GENERAL_SVC_HESTON_EQUATION_HPP_
