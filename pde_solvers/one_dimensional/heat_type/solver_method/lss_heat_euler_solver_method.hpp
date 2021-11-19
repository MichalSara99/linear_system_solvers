#if !defined(_LSS_HEAT_EULER_SOLVER_METHOD_HPP_)
#define _LSS_HEAT_EULER_SOLVER_METHOD_HPP_

#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

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
#include "pde_solvers/lss_heat_data_config.hpp"
#include "pde_solvers/lss_heat_solver_config.hpp"
#include "pde_solvers/lss_pde_discretization_config.hpp"
#include "pde_solvers/one_dimensional/heat_type/explicit_coefficients/lss_heat_euler_svc_coefficients.hpp"

namespace lss_pde_solvers
{

namespace one_dimensional
{

using lss_boundary::boundary_1d_pair;
using lss_boundary::boundary_1d_ptr;
using lss_boundary::dirichlet_boundary_1d;
using lss_boundary::neumann_boundary_1d;
using lss_boundary::robin_boundary_1d;
using lss_containers::container_2d;
using lss_enumerations::by_enum;
using lss_utility::coefficient_sevenlet_t;
using lss_utility::function_2d_sevenlet_t; // ?
using lss_utility::NaN;
using lss_utility::pair_t;
using lss_utility::range;
using lss_utility::sptr_t;

template <template <typename, typename> typename container, typename fp_type, typename alloc>
using explicit_heat_svc_scheme_function =
    std::function<void(heat_euler_svc_coefficients_ptr<fp_type> const &, grid_config_1d_ptr<fp_type> const &,
                       container<fp_type, alloc> const &, container<fp_type, alloc> const &,
                       boundary_1d_pair<fp_type> const &, fp_type const &, container<fp_type, alloc> &)>;

/**
    explicit_heat_svc_scheme object
 */
template <typename fp_type, template <typename, typename> typename container, typename allocator>
class explicit_heat_svc_scheme
{
    typedef container<fp_type, allocator> container_t;
    typedef explicit_heat_svc_scheme_function<container, fp_type, allocator> scheme_function_t;

  public:
    static scheme_function_t const get(bool is_homogeneus)
    {
        const fp_type two = static_cast<fp_type>(2.0);
        auto scheme_fun_h = [=](heat_euler_svc_coefficients_ptr<fp_type> const &cfs,
                                grid_config_1d_ptr<fp_type> const &grid_config, container_t const &input,
                                container_t const &inhom_input, boundary_1d_pair<fp_type> const &boundary_pair,
                                fp_type const &time, container_t &solution) {
            auto const &first_bnd = boundary_pair.first;
            auto const &second_bnd = boundary_pair.second;
            auto const &a = cfs->A_;
            auto const &b = cfs->B_;
            auto const &d = cfs->D_;
            auto const h = cfs->h_;
            fp_type x{};
            // for lower boundaries first:
            x = grid_1d<fp_type>::value(grid_config, 0);
            if (auto const &ptr = std::dynamic_pointer_cast<dirichlet_boundary_1d<fp_type>>(first_bnd))
            {
                solution[0] = ptr->value(time);
            }
            else if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(first_bnd))
            {
                const fp_type beta = two * h * ptr->value(time);
                solution[0] = beta * a(x) + b(x) * input[0] + (a(x) + d(x)) * input[1];
            }
            else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_bnd))
            {
                const fp_type beta = two * h * ptr->value(time);
                const fp_type alpha = two * h * ptr->linear_value(time);
                solution[0] = (b(x) + alpha * a(x)) * input[0] + (a(x) + d(x)) * input[1] + beta * a(x);
            }
            // for upper boundaries second:
            const std::size_t N = solution.size() - 1;
            x = grid_1d<fp_type>::value(grid_config, N);
            if (auto const &ptr = std::dynamic_pointer_cast<dirichlet_boundary_1d<fp_type>>(second_bnd))
            {
                solution[N] = ptr->value(time);
            }
            else if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(second_bnd))
            {
                const fp_type delta = two * h * ptr->value(time);
                solution[N] = (a(x) + d(x)) * input[N - 1] + b(x) * input[N] - delta * d(x);
            }
            else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
            {
                const fp_type delta = two * h * ptr->value(time);
                const fp_type gamma = two * h * ptr->linear_value(time);
                solution[N] = (a(x) + d(x)) * input[N - 1] + (b(x) - gamma * d(x)) * input[N] - delta * d(x);
            }

            for (std::size_t t = 1; t < N; ++t)
            {
                x = grid_1d<fp_type>::value(grid_config, t);
                solution[t] = (d(x) * input[t + 1]) + (b(x) * input[t]) + (a(x) * input[t - 1]);
            }
        };
        auto scheme_fun_nh = [=](heat_euler_svc_coefficients_ptr<fp_type> const &cfs,
                                 grid_config_1d_ptr<fp_type> const &grid_config, container_t const &input,
                                 container_t const &inhom_input, boundary_1d_pair<fp_type> const &boundary_pair,
                                 fp_type const &time, container_t &solution) {
            auto const &first_bnd = boundary_pair.first;
            auto const &second_bnd = boundary_pair.second;
            auto const &a = cfs->A_;
            auto const &b = cfs->B_;
            auto const &d = cfs->D_;
            auto const k = cfs->k_;
            auto const h = cfs->h_;
            fp_type x{};

            // for lower boundaries first:
            x = grid_1d<fp_type>::value(grid_config, 0);
            if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(first_bnd))
            {
                const fp_type beta = two * h * ptr->value(time);
                solution[0] = beta * a(x) + b(x) * input[0] + (a(x) + d(x)) * input[1] + k * inhom_input[0];
            }
            else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_bnd))
            {
                const fp_type beta = two * h * ptr->value(time);
                const fp_type alpha = two * h * ptr->linear_value(time);
                solution[0] =
                    (b(x) + alpha * a(x)) * input[0] + (a(x) + d(x)) * input[1] + beta * a(x) + k * inhom_input[0];
            }
            // for upper boundaries second:
            const std::size_t N = solution.size() - 1;
            x = grid_1d<fp_type>::value(grid_config, N);
            if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(second_bnd))
            {
                const fp_type delta = two * h * ptr->value(time);
                solution[N] = (a(x) + d(x)) * input[N - 1] + b(x) * input[N] - delta * d(x) + k * inhom_input[N];
            }
            else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
            {
                const fp_type delta = two * h * ptr->value(time);
                const fp_type gamma = two * h * ptr->linear_value(time);
                solution[N] =
                    (a(x) + d(x)) * input[N - 1] + (b(x) - gamma * d(x)) * input[N] - delta * d(x) + k * inhom_input[N];
                ;
            }
            for (std::size_t t = 1; t < N; ++t)
            {
                x = grid_1d<fp_type>::value(grid_config, t);
                solution[t] = (d(x) * input[t + 1]) + (b(x) * input[t]) + (a(x) * input[t - 1]) + (k * inhom_input[t]);
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
template <typename fp_type> class heat_euler_solver_method
 object
*/
template <typename fp_type, template <typename, typename> typename container, typename allocator>
class heat_euler_solver_method
{
    typedef container<fp_type, allocator> container_t;

  private:
    // scheme coefficients:
    heat_euler_svc_coefficients_ptr<fp_type> coefficients_;
    grid_config_1d_ptr<fp_type> grid_cfg_;

    explicit heat_euler_solver_method() = delete;

    void initialize()
    {
    }

  public:
    explicit heat_euler_solver_method(heat_euler_svc_coefficients_ptr<fp_type> const &coefficients,
                                      grid_config_1d_ptr<fp_type> const &grid_config)
        : coefficients_{coefficients}, grid_cfg_{grid_config}
    {
        initialize();
    }

    ~heat_euler_solver_method()
    {
    }

    heat_euler_solver_method(heat_euler_solver_method const &) = delete;
    heat_euler_solver_method(heat_euler_solver_method &&) = delete;
    heat_euler_solver_method &operator=(heat_euler_solver_method const &) = delete;
    heat_euler_solver_method &operator=(heat_euler_solver_method &&) = delete;

    void solve(container<fp_type, allocator> &prev_solution, boundary_1d_pair<fp_type> const &boundary_pair,
               fp_type const &time, container<fp_type, allocator> &solution);

    void solve(container<fp_type, allocator> &prev_solution, boundary_1d_pair<fp_type> const &boundary_pair,
               fp_type const &time, std::function<fp_type(fp_type, fp_type)> const &heat_source,
               container<fp_type, allocator> &solution);
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void heat_euler_solver_method<fp_type, container, allocator>::solve(container<fp_type, allocator> &prev_solution,
                                                                    boundary_1d_pair<fp_type> const &boundary_pair,
                                                                    fp_type const &time,
                                                                    container<fp_type, allocator> &solution)
{
    typedef explicit_heat_svc_scheme<fp_type, container, allocator> heat_scheme;

    // get the right-hand side of the scheme:
    auto scheme = heat_scheme::get(true);
    scheme(coefficients_, grid_cfg_, prev_solution, container<fp_type, allocator>(), boundary_pair, time, solution);
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void heat_euler_solver_method<fp_type, container, allocator>::solve(
    container<fp_type, allocator> &prev_solution, boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time,
    std::function<fp_type(fp_type, fp_type)> const &heat_source, container<fp_type, allocator> &solution)
{
    typedef explicit_heat_svc_scheme<fp_type, container, allocator> heat_scheme;
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    // get the right-hand side of the scheme:
    auto scheme = heat_scheme::get(false);
    container<fp_type, allocator> source(prev_solution.size());
    d_1d::of_function(grid_cfg_, time, heat_source, source);
    scheme(coefficients_, grid_cfg_, prev_solution, source, boundary_pair, time, solution);
}
} // namespace one_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_HEAT_EULER_SOLVER_METHOD_HPP_
