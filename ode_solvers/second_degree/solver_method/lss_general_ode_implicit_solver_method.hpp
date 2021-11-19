#if !defined(_LSS_GENERAL_ODE_IMPLICIT_SOLVER_METHOD_HPP_)
#define _LSS_GENERAL_ODE_IMPLICIT_SOLVER_METHOD_HPP_

#include "boundaries/lss_boundary.hpp"
#include "boundaries/lss_dirichlet_boundary.hpp"
#include "boundaries/lss_neumann_boundary.hpp"
#include "boundaries/lss_robin_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "containers/lss_container_2d.hpp"
#include "discretization/lss_discretization.hpp"
#include "discretization/lss_grid.hpp"
#include "discretization/lss_grid_config.hpp"
#include "ode_solvers/second_degree/implicit_coefficients/lss_general_ode_implicit_coefficients.hpp"

namespace lss_ode_solvers
{

using lss_boundary::boundary_1d_pair;
using lss_boundary::boundary_1d_ptr;
using lss_boundary::dirichlet_boundary_1d;
using lss_boundary::neumann_boundary_1d;
using lss_boundary::robin_boundary_1d;

/**
wave_implicit_solver_method object
*/
template <typename fp_type, typename solver, template <typename, typename> typename container = std::vector,
          typename allocator = std::allocator<fp_type>>
class ode_implicit_solver_method
{
    typedef container<fp_type, allocator> container_t;

  private:
    // solvers:
    solver solveru_ptr_;
    // scheme coefficients:
    general_ode_implicit_coefficients_ptr<fp_type> coefficients_;
    grid_config_1d_ptr<fp_type> grid_cfg_;

    explicit ode_implicit_solver_method() = delete;

    void initialize()
    {
    }

    void split(container_t &low, container_t &diag, container_t &high)
    {
        fp_type x{};
        for (std::size_t t = 0; t < low.size(); ++t)
        {
            x = grid_1d<fp_type>::value(grid_cfg_, t);
            low[t] = coefficients_->A_(x);
            diag[t] = coefficients_->C_(x);
            high[t] = coefficients_->B_(x);
        }
    }

  public:
    explicit ode_implicit_solver_method(solver const &solver_ptr,
                                        general_ode_implicit_coefficients_ptr<fp_type> const &coefficients,
                                        grid_config_1d_ptr<fp_type> const &grid_config)
        : solveru_ptr_{solver_ptr}, coefficients_{coefficients}, grid_cfg_{grid_config}
    {
        initialize();
    }

    ~ode_implicit_solver_method()
    {
    }

    ode_implicit_solver_method(ode_implicit_solver_method const &) = delete;
    ode_implicit_solver_method(ode_implicit_solver_method &&) = delete;
    ode_implicit_solver_method &operator=(ode_implicit_solver_method const &) = delete;
    ode_implicit_solver_method &operator=(ode_implicit_solver_method &&) = delete;

    void solve(boundary_1d_pair<fp_type> const &boundary_pair, container<fp_type, allocator> &solution);

    void solve(boundary_1d_pair<fp_type> const &boundary_pair, std::function<fp_type(fp_type)> const &source,
               container<fp_type, allocator> &solution);
};

template <typename fp_type, typename solver, template <typename, typename> typename container, typename allocator>
void ode_implicit_solver_method<fp_type, solver, container, allocator>::solve(
    boundary_1d_pair<fp_type> const &boundary_pair, container<fp_type, allocator> &solution)
{
    // containers for first split solver:
    container_t low(coefficients_->space_size_, fp_type{});
    container_t diag(coefficients_->space_size_, fp_type{});
    container_t high(coefficients_->space_size_, fp_type{});
    container_t rhs(coefficients_->space_size_, fp_type{0.0});
    // get the right-hand side of the scheme:
    split(low, diag, high);
    solveru_ptr_->set_diagonals(low, diag, high);
    solveru_ptr_->set_rhs(rhs);
    solveru_ptr_->solve(boundary_pair, solution);
}

template <typename fp_type, typename solver, template <typename, typename> typename container, typename allocator>
void ode_implicit_solver_method<fp_type, solver, container, allocator>::solve(
    boundary_1d_pair<fp_type> const &boundary_pair, std::function<fp_type(fp_type)> const &source,
    container<fp_type, allocator> &solution)
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    // containers for first split solver:
    container_t low(coefficients_->space_size_, fp_type{});
    container_t diag(coefficients_->space_size_, fp_type{});
    container_t high(coefficients_->space_size_, fp_type{});
    container_t rhs(coefficients_->space_size_, fp_type{});
    // get the right-hand side of the scheme:
    split(low, diag, high);
    d_1d::of_function(grid_cfg_, source, rhs);
    solveru_ptr_->set_diagonals(low, diag, high);
    solveru_ptr_->set_rhs(rhs);
    solveru_ptr_->solve(boundary_pair, solution);
}

} // namespace lss_ode_solvers

#endif ///_LSS_GENERAL_ODE_IMPLICIT_SOLVER_METHOD_HPP_
