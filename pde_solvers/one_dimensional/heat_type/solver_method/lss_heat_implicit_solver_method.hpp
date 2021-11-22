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

template <template <typename, typename> typename container, typename fp_type, typename alloc>
using implicit_heat_scheme_function_t =
    std::function<void(general_svc_heat_equation_implicit_coefficients_ptr<fp_type> const &,
                       grid_config_1d_ptr<fp_type> const &, container<fp_type, alloc> const &,
                       container<fp_type, alloc> const &, container<fp_type, alloc> const &,
                       boundary_1d_pair<fp_type> const &, fp_type const &, container<fp_type, alloc> &)>;

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class implicit_heat_scheme
{
    typedef container<fp_type, allocator> container_t;
    typedef implicit_heat_scheme_function_t<container, fp_type, allocator> scheme_function_t;

  public:
    static scheme_function_t const get(bool is_homogeneus)
    {
        const fp_type two = static_cast<fp_type>(2.0);
        const fp_type one = static_cast<fp_type>(1.0);
        auto scheme_fun_h = [=](general_svc_heat_equation_implicit_coefficients_ptr<fp_type> const &cfs,
                                grid_config_1d_ptr<fp_type> const &grid_cfg, container_t const &input,
                                container_t const &inhom_input, container_t const &inhom_input_next,
                                boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time,
                                container_t &solution) {
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
                solution[0] = (one - theta) * beta * A(x) +
                              (one - (one - theta) * (two * B(x) - alpha * A(x))) * input[0] +
                              (one - theta) * (A(x) + D(x)) * input[1];
            }
            // for upper boundaries second:
            const std::size_t N = solution.size() - 1;
            x = grid_1d<fp_type>::value(grid_cfg, N);
            if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(second_bnd))
            {
                const fp_type delta = two * h * ptr->value(time);
                solution[N] = (one - theta) * (A(x) + D(x)) * input[N - 1] +
                              (one - two * (one - theta) * B(x)) * input[N] - (one - theta) * delta * D(x);
            }
            else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
            {
                const fp_type delta = two * h * ptr->value(time);
                const fp_type gamma = two * h * ptr->linear_value(time);
                solution[N] = (one - theta) * (A(x) + D(x)) * input[N - 1] +
                              (one - (one - theta) * (two * B(x) + gamma * D(x))) * input[N] -
                              (one - theta) * delta * D(x);
            }

            for (std::size_t t = 1; t < N; ++t)
            {
                x = grid_1d<fp_type>::value(grid_cfg, t);
                solution[t] = (D(x) * (one - theta) * input[t + 1]) + ((one - two * B(x) * (one - theta)) * input[t]) +
                              (A(x) * (one - theta) * input[t - 1]);
            }
        };
        auto scheme_fun_nh = [=](general_svc_heat_equation_implicit_coefficients_ptr<fp_type> const &cfs,
                                 grid_config_1d_ptr<fp_type> const &grid_cfg, container_t const &input,
                                 container_t const &inhom_input, container_t const &inhom_input_next,
                                 boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time,
                                 container_t &solution) {
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
                solution[0] = (one - theta) * beta * A(x) +
                              (one - (one - theta) * (two * B(x) - alpha * A(x))) * input[0] +
                              (one - theta) * (A(x) + D(x)) * input[1] + theta * k * inhom_input_next[0] +
                              (one - theta) * k * inhom_input[0];
            }
            // for upper boundaries second:
            const std::size_t N = solution.size() - 1;
            x = grid_1d<fp_type>::value(grid_cfg, N);
            if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(second_bnd))
            {
                const fp_type delta = two * h * ptr->value(time);
                solution[N] = (one - theta) * (A(x) + D(x)) * input[N - 1] +
                              (one - two * (one - theta) * B(x)) * input[N] - (one - theta) * delta * D(x) +
                              theta * k * inhom_input_next[N] + (one - theta) * k * inhom_input[N];
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
        };
        if (is_homogeneus)
        {
            return scheme_fun_h;
        }
        else
        {
            return scheme_fun_nh;
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
    // solvers:
    solver solveru_ptr_;
    // scheme coefficients:
    general_svc_heat_equation_implicit_coefficients_ptr<fp_type> coefficients_;
    grid_config_1d_ptr<fp_type> grid_cfg_;

    explicit heat_implicit_solver_method() = delete;

    void initialize()
    {
    }

    void split(container_t &low, container_t &diag, container_t &high)
    {
        const fp_type one = static_cast<fp_type>(1.0);
        const fp_type two = static_cast<fp_type>(2.0);
        fp_type x{};
        for (std::size_t t = 0; t < low.size(); ++t)
        {
            x = grid_1d<fp_type>::value(grid_cfg_, t);
            low[t] = (-coefficients_->theta_ * coefficients_->A_(x));
            diag[t] = (one + two * coefficients_->theta_ * coefficients_->B_(x));
            high[t] = (-coefficients_->theta_ * coefficients_->D_(x));
        }
    }

  public:
    explicit heat_implicit_solver_method(
        solver const &solver_ptr, general_svc_heat_equation_implicit_coefficients_ptr<fp_type> const &coefficients,
        grid_config_1d_ptr<fp_type> const &grid_config)
        : solveru_ptr_{solver_ptr}, coefficients_{coefficients}, grid_cfg_{grid_config}
    {
        initialize();
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

    // containers for first split solver:
    container_t low(coefficients_->space_size_, fp_type{});
    container_t diag(coefficients_->space_size_, fp_type{});
    container_t high(coefficients_->space_size_, fp_type{});
    container_t rhs(coefficients_->space_size_, fp_type{});
    // get the right-hand side of the scheme:
    auto scheme = heat_scheme::get(true);
    split(low, diag, high);
    scheme(coefficients_, grid_cfg_, prev_solution, container_t(), container_t(), boundary_pair, time, rhs);
    solveru_ptr_->set_diagonals(low, diag, high);
    solveru_ptr_->set_rhs(rhs);
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

    // containers for first split solver:
    container_t low(coefficients_->space_size_, fp_type{});
    container_t diag(coefficients_->space_size_, fp_type{});
    container_t high(coefficients_->space_size_, fp_type{});
    container_t rhs(coefficients_->space_size_, fp_type{});
    container_t source(coefficients_->space_size_, fp_type{});
    container_t source_next(coefficients_->space_size_, fp_type{});
    // get the right-hand side of the scheme:
    auto scheme = heat_scheme::get(false);
    split(low, diag, high);
    d_1d::of_function(grid_cfg_, time, heat_source, source);
    d_1d::of_function(grid_cfg_, next_time, heat_source, source_next);
    scheme(coefficients_, grid_cfg_, prev_solution, source, source_next, boundary_pair, time, rhs);
    solveru_ptr_->set_diagonals(low, diag, high);
    solveru_ptr_->set_rhs(rhs);
    solveru_ptr_->solve(boundary_pair, solution, time);
}
} // namespace one_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_HEAT_IMPLICIT_SOLVER_METHOD_HPP_
