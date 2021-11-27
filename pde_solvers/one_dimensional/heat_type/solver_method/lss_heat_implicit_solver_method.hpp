#if !defined(_LSS_HEAT_IMPLICIT_SOLVER_METHOD_HPP_)
#define _LSS_HEAT_IMPLICIT_SOLVER_METHOD_HPP_

#include "boundaries/lss_boundary.hpp"
#include "boundaries/lss_dirichlet_boundary.hpp"
#include "boundaries/lss_neumann_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "containers/lss_container_2d.hpp"
#include "discretization/lss_discretization.hpp"
#include "discretization/lss_grid.hpp"
#include "discretization/lss_grid_config.hpp"
#include "pde_solvers/lss_heat_data_config.hpp"
#include "pde_solvers/lss_heat_solver_config.hpp"
#include "pde_solvers/lss_pde_discretization_config.hpp"
#include "pde_solvers/one_dimensional/heat_type/implicit_coefficients/lss_1d_general_svc_heat_equation_implicit_coefficients.hpp"

namespace lss_pde_solvers
{

namespace one_dimensional
{

using lss_boundary::boundary_1d_pair;
using lss_boundary::boundary_1d_ptr;
using lss_boundary::dirichlet_boundary_1d;
using lss_boundary::neumann_boundary_1d;
using lss_containers::container_2d;
using lss_enumerations::by_enum;
using lss_utility::pair_t;

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class implicit_heat_scheme
{
    typedef container<fp_type, allocator> container_t;

  public:
    static void rhs(general_svc_heat_equation_implicit_coefficients_ptr<fp_type> const &cfs,
                    grid_config_1d_ptr<fp_type> const &grid_cfg, container_t const &input,
                    boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, container_t &solution)
    {
        auto const two = static_cast<fp_type>(2.0);
        auto const one = static_cast<fp_type>(1.0);
        auto const &first_bnd = boundary_pair.first;
        auto const &second_bnd = boundary_pair.second;
        auto const &A = cfs->A_;
        auto const &B = cfs->B_;
        auto const &D = cfs->D_;
        auto const theta = cfs->theta_;
        auto const h = grid_1d<fp_type>::step(grid_cfg);
        fp_type x{};
        // for lower boundaries first:
        x = grid_1d<fp_type>::value(grid_cfg, 0);
        if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(first_bnd))
        {
            const fp_type beta = two * h * ptr->value(time);
            solution[0] = (one - theta) * beta * A(x) + (one - two * (one - theta) * B(x)) * input[0] +
                          (one - theta) * (A(x) + D(x)) * input[1];
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_bnd))
        {
            const fp_type beta = two * h * ptr->value(time);
            const fp_type alpha = two * h * ptr->linear_value(time);
            solution[0] = (one - theta) * beta * A(x) + (one - (one - theta) * (two * B(x) - alpha * A(x))) * input[0] +
                          (one - theta) * (A(x) + D(x)) * input[1];
        }
        // for upper boundaries second:
        const std::size_t N = solution.size() - 1;
        x = grid_1d<fp_type>::value(grid_cfg, N);
        if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(second_bnd))
        {
            const fp_type delta = two * h * ptr->value(time);
            solution[N] = (one - theta) * (A(x) + D(x)) * input[N - 1] + (one - two * (one - theta) * B(x)) * input[N] -
                          (one - theta) * delta * D(x);
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
        {
            const fp_type delta = two * h * ptr->value(time);
            const fp_type gamma = two * h * ptr->linear_value(time);
            solution[N] = (one - theta) * (A(x) + D(x)) * input[N - 1] +
                          (one - (one - theta) * (two * B(x) + gamma * D(x))) * input[N] - (one - theta) * delta * D(x);
        }

        for (std::size_t t = 1; t < N; ++t)
        {
            x = grid_1d<fp_type>::value(grid_cfg, t);
            solution[t] = (D(x) * (one - theta) * input[t + 1]) + ((one - two * B(x) * (one - theta)) * input[t]) +
                          (A(x) * (one - theta) * input[t - 1]);
        }
    }

    static void rhs_source(general_svc_heat_equation_implicit_coefficients_ptr<fp_type> const &cfs,
                           grid_config_1d_ptr<fp_type> const &grid_cfg, container_t const &input,
                           container_t const &inhom_input, container_t const &inhom_input_next,
                           boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, container_t &solution)
    {
        auto const two = static_cast<fp_type>(2.0);
        auto const one = static_cast<fp_type>(1.0);
        auto const &first_bnd = boundary_pair.first;
        auto const &second_bnd = boundary_pair.second;
        auto const &A = cfs->A_;
        auto const &B = cfs->B_;
        auto const &D = cfs->D_;
        auto const k = cfs->k_;
        auto const theta = cfs->theta_;
        auto const h = grid_1d<fp_type>::step(grid_cfg);
        fp_type x{};

        // for lower boundaries first:
        x = grid_1d<fp_type>::value(grid_cfg, 0);
        if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(first_bnd))
        {
            const fp_type beta = two * h * ptr->value(time);
            solution[0] = (one - theta) * beta * A(x) + (one - two * (one - theta) * B(x)) * input[0] +
                          (one - theta) * (A(x) + D(x)) * input[1] + theta * k * inhom_input_next[0] +
                          (one - theta) * k * inhom_input[0];
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_bnd))
        {
            const fp_type beta = two * h * ptr->value(time);
            const fp_type alpha = two * h * ptr->linear_value(time);
            solution[0] = (one - theta) * beta * A(x) + (one - (one - theta) * (two * B(x) - alpha * A(x))) * input[0] +
                          (one - theta) * (A(x) + D(x)) * input[1] + theta * k * inhom_input_next[0] +
                          (one - theta) * k * inhom_input[0];
        }
        // for upper boundaries second:
        const std::size_t N = solution.size() - 1;
        x = grid_1d<fp_type>::value(grid_cfg, N);
        if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(second_bnd))
        {
            const fp_type delta = two * h * ptr->value(time);
            solution[N] = (one - theta) * (A(x) + D(x)) * input[N - 1] + (one - two * (one - theta) * B(x)) * input[N] -
                          (one - theta) * delta * D(x) + theta * k * inhom_input_next[N] +
                          (one - theta) * k * inhom_input[N];
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
        {
            const fp_type delta = two * h * ptr->value(time);
            const fp_type gamma = two * h * ptr->linear_value(time);
            solution[N] = (one - theta) * (A(x) + D(x)) * input[N - 1] +
                          (one - (one - theta) * (two * B(x) + gamma * D(x))) * input[N] -
                          (one - theta) * delta * D(x) + theta * k * inhom_input_next[N] +
                          (one - theta) * k * inhom_input[N];
        }
        for (std::size_t t = 1; t < N; ++t)
        {
            x = grid_1d<fp_type>::value(grid_cfg, t);
            solution[t] = (D(x) * (one - theta) * input[t + 1]) + ((one - two * B(x) * (one - theta)) * input[t]) +
                          (A(x) * (one - theta) * input[t - 1]) +
                          k * (theta * inhom_input_next[t] + (one - theta) * inhom_input[t]);
        }
    }
};

/**
heat_implicit_solver_method object
*/
template <typename fp_type, typename solver, template <typename, typename> typename container = std::vector,
          typename allocator = std::allocator<fp_type>>
class heat_implicit_solver_method
{
    typedef container<fp_type, allocator> container_t;

  private:
    // private constats:
    const fp_type cone_ = static_cast<fp_type>(1.0);
    const fp_type ctwo_ = static_cast<fp_type>(2.0);
    // solvers:
    solver solveru_ptr_;
    // scheme coefficients:
    general_svc_heat_equation_implicit_coefficients_ptr<fp_type> coefficients_;
    grid_config_1d_ptr<fp_type> grid_cfg_;
    // prepare containers:
    container_t low_, diag_, high_;
    container_t source_, source_next_;
    container_t rhs_;

    explicit heat_implicit_solver_method() = delete;

    void initialize(bool is_heat_sourse_set)
    {
        // prepare containers:
        low_.resize(coefficients_->space_size_);
        diag_.resize(coefficients_->space_size_);
        high_.resize(coefficients_->space_size_);
        rhs_.resize(coefficients_->space_size_);
        if (is_heat_sourse_set)
        {
            source_.resize(coefficients_->space_size_);
            source_next_.resize(coefficients_->space_size_);
        }
    }

    void split(container_t &low, container_t &diag, container_t &high)
    {
        fp_type x{};
        for (std::size_t t = 0; t < low.size(); ++t)
        {
            x = grid_1d<fp_type>::value(grid_cfg_, t);
            low[t] = (-coefficients_->theta_ * coefficients_->A_(x));
            diag[t] = (cone_ + ctwo_ * coefficients_->theta_ * coefficients_->B_(x));
            high[t] = (-coefficients_->theta_ * coefficients_->D_(x));
        }
    }

  public:
    explicit heat_implicit_solver_method(
        solver const &solver_ptr, general_svc_heat_equation_implicit_coefficients_ptr<fp_type> const &coefficients,
        grid_config_1d_ptr<fp_type> const &grid_config, bool is_heat_sourse_set)
        : solveru_ptr_{solver_ptr}, coefficients_{coefficients}, grid_cfg_{grid_config}
    {
        initialize(is_heat_sourse_set);
    }

    ~heat_implicit_solver_method()
    {
    }

    heat_implicit_solver_method(heat_implicit_solver_method const &) = delete;
    heat_implicit_solver_method(heat_implicit_solver_method &&) = delete;
    heat_implicit_solver_method &operator=(heat_implicit_solver_method const &) = delete;
    heat_implicit_solver_method &operator=(heat_implicit_solver_method &&) = delete;

    void solve(container<fp_type, allocator> &prev_solution, boundary_1d_pair<fp_type> const &boundary_pair,
               fp_type const &time, container<fp_type, allocator> &solution);

    void solve(container<fp_type, allocator> &prev_solution, boundary_1d_pair<fp_type> const &boundary_pair,
               fp_type const &time, fp_type const &next_time,
               std::function<fp_type(fp_type, fp_type)> const &heat_source, container<fp_type, allocator> &solution);
};

template <typename fp_type, typename solver, template <typename, typename> typename container, typename allocator>
void heat_implicit_solver_method<fp_type, solver, container, allocator>::solve(
    container<fp_type, allocator> &prev_solution, boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time,
    container<fp_type, allocator> &solution)
{
    typedef implicit_heat_scheme<fp_type, container, allocator> heat_scheme;

    heat_scheme::rhs(coefficients_, grid_cfg_, prev_solution, boundary_pair, time, rhs_);
    split(low_, diag_, high_);
    solveru_ptr_->set_diagonals(low_, diag_, high_);
    solveru_ptr_->set_rhs(rhs_);
    solveru_ptr_->solve(boundary_pair, solution, time);
}

template <typename fp_type, typename solver, template <typename, typename> typename container, typename allocator>
void heat_implicit_solver_method<fp_type, solver, container, allocator>::solve(
    container<fp_type, allocator> &prev_solution, boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time,
    fp_type const &next_time, std::function<fp_type(fp_type, fp_type)> const &heat_source,
    container<fp_type, allocator> &solution)
{
    typedef implicit_heat_scheme<fp_type, container, allocator> heat_scheme;
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;

    split(low_, diag_, high_);
    d_1d::of_function(grid_cfg_, time, heat_source, source_);
    d_1d::of_function(grid_cfg_, next_time, heat_source, source_next_);
    heat_scheme::rhs_source(coefficients_, grid_cfg_, prev_solution, source_, source_next_, boundary_pair, time, rhs_);
    solveru_ptr_->set_diagonals(low_, diag_, high_);
    solveru_ptr_->set_rhs(rhs_);
    solveru_ptr_->solve(boundary_pair, solution, time);
}
} // namespace one_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_HEAT_IMPLICIT_SOLVER_METHOD_HPP_
