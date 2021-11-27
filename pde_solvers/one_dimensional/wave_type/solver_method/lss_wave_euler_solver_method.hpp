#if !defined(_LSS_WAVE_EULER_SOLVER_METHOD_HPP_)
#define _LSS_WAVE_EULER_SOLVER_METHOD_HPP_

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
#include "pde_solvers/lss_pde_discretization_config.hpp"
#include "pde_solvers/lss_wave_data_config.hpp"
#include "pde_solvers/one_dimensional/wave_type/explicit_coefficients/lss_wave_svc_explicit_coefficients.hpp"

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
using lss_utility::NaN;
using lss_utility::range;
using lss_utility::sptr_t;

/**
    explicit_wave_svc_scheme object
 */
template <typename fp_type, template <typename, typename> typename container, typename allocator>
class explicit_wave_svc_scheme
{
    typedef container<fp_type, allocator> container_t;

  public:
    static void rhs(wave_svc_explicit_coefficients_ptr<fp_type> const &cfs,
                    grid_config_1d_ptr<fp_type> const &grid_config, container_t const &input_0,
                    container_t const &input_1, boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time,
                    container_t &solution)
    {
        const fp_type two = static_cast<fp_type>(2.0);
        auto const &first_bnd = boundary_pair.first;
        auto const &second_bnd = boundary_pair.second;
        auto const &a = cfs->A_;
        auto const &b = cfs->B_;
        auto const &c = cfs->C_;
        auto const &d = cfs->D_;
        auto const h = grid_1d<fp_type>::step(grid_config);
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
            solution[0] = beta * a(x) + c(x) * input_1[0] + (a(x) + b(x)) * input_1[1] - d(x) * input_0[0];
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_bnd))
        {
            const fp_type beta = two * h * ptr->value(time);
            const fp_type alpha = two * h * ptr->linear_value(time);
            solution[0] =
                beta * a(x) + (c(x) + alpha * a(x)) * input_1[0] + (a(x) + b(x)) * input_1[1] - d(x) * input_0[0];
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
            solution[N] = (a(x) + b(x)) * input_1[N - 1] + c(x) * input_1[N] - delta * b(x) - d(x) * input_0[N];
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
        {
            const fp_type delta = two * h * ptr->value(time);
            const fp_type gamma = two * h * ptr->linear_value(time);
            solution[N] =
                (a(x) + b(x)) * input_1[N - 1] + (c(x) - gamma * b(x)) * input_1[N] - delta * b(x) - d(x) * input_0[N];
        }

        for (std::size_t t = 1; t < N; ++t)
        {
            x = grid_1d<fp_type>::value(grid_config, t);
            solution[t] = (a(x) * input_1[t - 1]) + (c(x) * input_1[t]) + (b(x) * input_1[t + 1]) - (d(x) * input_0[t]);
        }
    }

    static void rhs_source(wave_svc_explicit_coefficients_ptr<fp_type> const &cfs,
                           grid_config_1d_ptr<fp_type> const &grid_config, container_t const &input_0,
                           container_t const &input_1, container_t const &inhom_input,
                           boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, container_t &solution)
    {
        const fp_type two = static_cast<fp_type>(2.0);
        auto const &first_bnd = boundary_pair.first;
        auto const &second_bnd = boundary_pair.second;
        auto const &a = cfs->A_;
        auto const &b = cfs->B_;
        auto const &c = cfs->C_;
        auto const &d = cfs->D_;
        auto const h = grid_1d<fp_type>::step(grid_config);
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
            solution[0] =
                beta * a(x) + c(x) * input_1[0] + (a(x) + b(x)) * input_1[1] - d(x) * input_0[0] + inhom_input[0];
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_bnd))
        {
            const fp_type beta = two * h * ptr->value(time);
            const fp_type alpha = two * h * ptr->linear_value(time);
            solution[0] = beta * a(x) + (c(x) + alpha * a(x)) * input_1[0] + (a(x) + b(x)) * input_1[1] -
                          d(x) * input_0[0] + inhom_input[0];
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
            solution[N] =
                (a(x) + b(x)) * input_1[N - 1] + c(x) * input_1[N] - delta * b(x) - d(x) * input_0[N] + inhom_input[N];
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
        {
            const fp_type delta = two * h * ptr->value(time);
            const fp_type gamma = two * h * ptr->linear_value(time);
            solution[N] = (a(x) + b(x)) * input_1[N - 1] + (c(x) - gamma * b(x)) * input_1[N] - delta * b(x) -
                          d(x) * input_0[N] + inhom_input[N];
        }

        for (std::size_t t = 1; t < N; ++t)
        {
            x = grid_1d<fp_type>::value(grid_config, t);
            solution[t] = (a(x) * input_1[t - 1]) + (c(x) * input_1[t]) + (b(x) * input_1[t + 1]) -
                          (d(x) * input_0[t]) + inhom_input[t];
        }
    }

    static void rhs_initial(wave_svc_explicit_coefficients_ptr<fp_type> const &cfs,
                            grid_config_1d_ptr<fp_type> const &grid_config, container_t const &input_0,
                            container_t const &input_1, boundary_1d_pair<fp_type> const &boundary_pair,
                            fp_type const &time, container_t &solution)
    {
        const fp_type one = static_cast<fp_type>(1.0);
        const fp_type two = static_cast<fp_type>(2.0);
        auto const &first_bnd = boundary_pair.first;
        auto const &second_bnd = boundary_pair.second;
        auto const &a = cfs->A_;
        auto const &b = cfs->B_;
        auto const &c = cfs->C_;
        auto const &d = cfs->D_;
        auto const h = grid_1d<fp_type>::step(grid_config);
        auto const k = cfs->k_;
        auto const one_gamma = (two * k);
        auto const &defl = [&](fp_type x) { return (one + d(x)); };
        auto const &A = [&](fp_type x) { return (a(x) / defl(x)); };
        auto const &B = [&](fp_type x) { return (b(x) / defl(x)); };
        auto const &C = [&](fp_type x) { return (c(x) / defl(x)); };
        auto const &D = [&](fp_type x) { return (one_gamma * (d(x) / defl(x))); };

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
            solution[0] = beta * A(x) + (C(x) * input_0[0]) + ((A(x) + B(x)) * input_0[1]) + (D(x) * input_1[0]);
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_bnd))
        {
            const fp_type beta = two * h * ptr->value(time);
            const fp_type alpha = two * h * ptr->linear_value(time);
            solution[0] =
                beta * A(x) + (C(x) + alpha * A(x)) * input_0[0] + (A(x) + B(x)) * input_0[1] + (D(x) * input_1[0]);
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
            solution[N] = (A(x) + B(x)) * input_0[N - 1] + C(x) * input_0[N] - delta * B(x) + (D(x) * input_1[N]);
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
        {
            const fp_type delta = two * h * ptr->value(time);
            const fp_type gamma = two * h * ptr->linear_value(time);
            solution[N] = (A(x) + B(x)) * input_0[N - 1] + (C(x) - gamma * B(x)) * input_0[N] - delta * B(x) +
                          (D(x) * input_1[N]);
        }

        for (std::size_t t = 1; t < N; ++t)
        {
            x = grid_1d<fp_type>::value(grid_config, t);
            solution[t] = (A(x) * input_0[t - 1]) + (C(x) * input_0[t]) + (B(x) * input_0[t + 1]) + (D(x) * input_1[t]);
        }
    }

    static void rhs_initial_source(wave_svc_explicit_coefficients_ptr<fp_type> const &cfs,
                                   grid_config_1d_ptr<fp_type> const &grid_config, container_t const &input_0,
                                   container_t const &input_1, container_t const &inhom_input,
                                   boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time,
                                   container_t &solution)
    {
        const fp_type one = static_cast<fp_type>(1.0);
        const fp_type two = static_cast<fp_type>(2.0);
        auto const &first_bnd = boundary_pair.first;
        auto const &second_bnd = boundary_pair.second;
        auto const &a = cfs->A_;
        auto const &b = cfs->B_;
        auto const &c = cfs->C_;
        auto const &d = cfs->D_;
        auto const h = grid_1d<fp_type>::step(grid_config);
        auto const k = cfs->k_;
        auto const &defl = [=](fp_type x) { return (one + d(x)); };
        auto const &A = [=](fp_type x) { return (a(x) / defl(x)); };
        auto const &B = [=](fp_type x) { return (b(x) / defl(x)); };
        auto const &C = [=](fp_type x) { return (c(x) / defl(x)); };
        auto const one_gamma = (two * k);

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
            solution[0] = beta * A(x) + C(x) * input_0[0] + (A(x) + B(x)) * input_0[1] +
                          ((one_gamma * d(x)) / defl(x)) * input_1[0] + (inhom_input[0] / defl(x));
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_bnd))
        {
            const fp_type beta = two * h * ptr->value(time);
            const fp_type alpha = two * h * ptr->linear_value(time);
            solution[0] = beta * A(x) + (C(x) + alpha * A(x)) * input_0[0] + (A(x) + B(x)) * input_0[1] +
                          ((one_gamma * d(x)) / defl(x)) * input_1[0] + (inhom_input[0] / defl(x));
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
            solution[N] = (A(x) + B(x)) * input_0[N - 1] + C(x) * input_0[N] - delta * B(x) +
                          ((one_gamma * d(x)) / defl(x)) * input_1[N] + (inhom_input[N] / defl(x));
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
        {
            const fp_type delta = two * h * ptr->value(time);
            const fp_type gamma = two * h * ptr->linear_value(time);
            solution[N] = (A(x) + B(x)) * input_0[N - 1] + (C(x) - gamma * B(x)) * input_0[N] - delta * B(x) +
                          ((one_gamma * d(x)) / defl(x)) * input_1[N] + (inhom_input[N] / defl(x));
        }

        for (std::size_t t = 1; t < N; ++t)
        {
            x = grid_1d<fp_type>::value(grid_config, t);
            solution[t] = (A(x) * input_0[t - 1]) + (C(x) * input_0[t]) + (B(x) * input_0[t + 1]) +
                          ((one_gamma * d(x) / defl(x)) * input_1[t]) + (inhom_input[t] / defl(x));
        }
    }

    static void rhs_terminal(wave_svc_explicit_coefficients_ptr<fp_type> const &cfs,
                             grid_config_1d_ptr<fp_type> const &grid_config, container_t const &input_0,
                             container_t const &input_1, boundary_1d_pair<fp_type> const &boundary_pair,
                             fp_type const &time, container_t &solution)
    {
        const fp_type one = static_cast<fp_type>(1.0);
        const fp_type two = static_cast<fp_type>(2.0);
        auto const &first_bnd = boundary_pair.first;
        auto const &second_bnd = boundary_pair.second;
        auto const &a = cfs->A_;
        auto const &b = cfs->B_;
        auto const &c = cfs->C_;
        auto const &d = cfs->D_;
        auto const h = grid_1d<fp_type>::step(grid_config);
        auto const k = cfs->k_;
        auto const &defl = [=](fp_type x) { return (one + d(x)); };
        auto const &A = [=](fp_type x) { return (a(x) / defl(x)); };
        auto const &B = [=](fp_type x) { return (b(x) / defl(x)); };
        auto const &C = [=](fp_type x) { return (c(x) / defl(x)); };
        auto const one_gamma = (two * k);

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
            solution[0] = beta * A(x) + C(x) * input_0[0] + (A(x) + B(x)) * input_0[1] -
                          ((one_gamma * d(x)) / defl(x)) * input_1[0];
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_bnd))
        {
            const fp_type beta = two * h * ptr->value(time);
            const fp_type alpha = two * h * ptr->linear_value(time);
            solution[0] = beta * A(x) + (C(x) + alpha * A(x)) * input_0[0] + (A(x) + B(x)) * input_0[1] -
                          ((one_gamma * d(x)) / defl(x)) * input_1[0];
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
            solution[N] = (A(x) + B(x)) * input_0[N - 1] + C(x) * input_0[N] - delta * B(x) -
                          ((one_gamma * d(x)) / defl(x)) * input_1[N];
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
        {
            const fp_type delta = two * h * ptr->value(time);
            const fp_type gamma = two * h * ptr->linear_value(time);
            solution[N] = (A(x) + B(x)) * input_0[N - 1] + (C(x) - gamma * B(x)) * input_0[N] - delta * B(x) -
                          ((one_gamma * d(x)) / defl(x)) * input_1[N];
        }

        for (std::size_t t = 1; t < N; ++t)
        {
            x = grid_1d<fp_type>::value(grid_config, t);
            solution[t] = (A(x) * input_0[t - 1]) + (C(x) * input_0[t]) + (B(x) * input_0[t + 1]) -
                          ((one_gamma * d(x) / defl(x)) * input_1[t]);
        }
    }

    static void rhs_terminal_source(wave_svc_explicit_coefficients_ptr<fp_type> const &cfs,
                                    grid_config_1d_ptr<fp_type> const &grid_config, container_t const &input_0,
                                    container_t const &input_1, container_t const &inhom_input,
                                    boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time,
                                    container_t &solution)
    {
        const fp_type one = static_cast<fp_type>(1.0);
        const fp_type two = static_cast<fp_type>(2.0);
        auto const &first_bnd = boundary_pair.first;
        auto const &second_bnd = boundary_pair.second;
        auto const &a = cfs->A_;
        auto const &b = cfs->B_;
        auto const &c = cfs->C_;
        auto const &d = cfs->D_;
        auto const h = grid_1d<fp_type>::step(grid_config);
        auto const k = cfs->k_;
        auto const &defl = [=](fp_type x) { return (one + d(x)); };
        auto const &A = [=](fp_type x) { return (a(x) / defl(x)); };
        auto const &B = [=](fp_type x) { return (b(x) / defl(x)); };
        auto const &C = [=](fp_type x) { return (c(x) / defl(x)); };
        auto const one_gamma = (two * k);

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
            solution[0] = beta * A(x) + C(x) * input_0[0] + (A(x) + B(x)) * input_0[1] -
                          ((one_gamma * d(x)) / defl(x)) * input_1[0] + (inhom_input[0] / defl(x));
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_bnd))
        {
            const fp_type beta = two * h * ptr->value(time);
            const fp_type alpha = two * h * ptr->linear_value(time);
            solution[0] = beta * A(x) + (C(x) + alpha * A(x)) * input_0[0] + (A(x) + B(x)) * input_0[1] -
                          ((one_gamma * d(x)) / defl(x)) * input_1[0] + (inhom_input[0] / defl(x));
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
            solution[N] = (A(x) + B(x)) * input_0[N - 1] + C(x) * input_0[N] - delta * B(x) -
                          ((one_gamma * d(x)) / defl(x)) * input_1[N] + (inhom_input[N] / defl(x));
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
        {
            const fp_type delta = two * h * ptr->value(time);
            const fp_type gamma = two * h * ptr->linear_value(time);
            solution[N] = (A(x) + B(x)) * input_0[N - 1] + (C(x) - gamma * B(x)) * input_0[N] - delta * B(x) -
                          ((one_gamma * d(x)) / defl(x)) * input_1[N] + (inhom_input[N] / defl(x));
        }

        for (std::size_t t = 1; t < N; ++t)
        {
            x = grid_1d<fp_type>::value(grid_config, t);
            solution[t] = (A(x) * input_0[t - 1]) + (C(x) * input_0[t]) + (B(x) * input_0[t + 1]) -
                          ((one_gamma * d(x) / defl(x)) * input_1[t]) + (inhom_input[t] / defl(x));
        }
    }
};

/**
wave_euler_solver_method object
*/
template <typename fp_type, template <typename, typename> typename container, typename allocator>
class wave_euler_solver_method
{
    typedef container<fp_type, allocator> container_t;
    typedef explicit_wave_svc_scheme<fp_type, container, allocator> wave_scheme;
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;

  private:
    // scheme coefficients:
    wave_svc_explicit_coefficients_ptr<fp_type> coefficients_;
    grid_config_1d_ptr<fp_type> grid_cfg_;
    // containers:
    container_t source_;

    explicit wave_euler_solver_method() = delete;

    void initialize(bool is_wave_source_set)
    {
        if (is_wave_source_set)
        {
            source_.resize(coefficients_->space_size_);
        }
    }

  public:
    explicit wave_euler_solver_method(wave_svc_explicit_coefficients_ptr<fp_type> const &coefficients,
                                      grid_config_1d_ptr<fp_type> const &grid_config, bool is_wave_source_set)
        : coefficients_{coefficients}, grid_cfg_{grid_config}
    {
        initialize(is_wave_source_set);
    }

    ~wave_euler_solver_method()
    {
    }

    wave_euler_solver_method(wave_euler_solver_method const &) = delete;
    wave_euler_solver_method(wave_euler_solver_method &&) = delete;
    wave_euler_solver_method &operator=(wave_euler_solver_method const &) = delete;
    wave_euler_solver_method &operator=(wave_euler_solver_method &&) = delete;

    void solve_initial(container<fp_type, allocator> &prev_solution_0, container<fp_type, allocator> &prev_solution_1,
                       boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, fp_type const &next_time,
                       container<fp_type, allocator> &solution);

    void solve_initial(container<fp_type, allocator> &prev_solution_0, container<fp_type, allocator> &prev_solution_1,
                       boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, fp_type const &next_time,
                       std::function<fp_type(fp_type, fp_type)> const &wave_source,
                       container<fp_type, allocator> &solution);

    void solve_terminal(container<fp_type, allocator> &prev_solution_0, container<fp_type, allocator> &prev_solution_1,
                        boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, fp_type const &next_time,
                        container<fp_type, allocator> &solution);

    void solve_terminal(container<fp_type, allocator> &prev_solution_0, container<fp_type, allocator> &prev_solution_1,
                        boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, fp_type const &next_time,
                        std::function<fp_type(fp_type, fp_type)> const &wave_source,
                        container<fp_type, allocator> &solution);

    void solve(container<fp_type, allocator> &prev_solution_0, container<fp_type, allocator> &prev_solution_1,
               boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, fp_type const &next_time,
               container<fp_type, allocator> &solution);

    void solve(container<fp_type, allocator> &prev_solution_0, container<fp_type, allocator> &prev_solution_1,
               boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, fp_type const &next_time,
               std::function<fp_type(fp_type, fp_type)> const &wave_source, container<fp_type, allocator> &solution);
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void wave_euler_solver_method<fp_type, container, allocator>::solve_initial(
    container<fp_type, allocator> &prev_solution_0, container<fp_type, allocator> &prev_solution_1,
    boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, fp_type const &next_time,
    container<fp_type, allocator> &solution)
{
    // get the right-hand side of the scheme:
    wave_scheme::rhs_initial(coefficients_, grid_cfg_, prev_solution_0, prev_solution_1, boundary_pair, time, solution);
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void wave_euler_solver_method<fp_type, container, allocator>::solve_initial(
    container<fp_type, allocator> &prev_solution_0, container<fp_type, allocator> &prev_solution_1,
    boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, fp_type const &next_time,
    std::function<fp_type(fp_type, fp_type)> const &wave_source, container<fp_type, allocator> &solution)
{
    // get the right-hand side of the scheme:
    d_1d::of_function(grid_cfg_, time, wave_source, source_);
    wave_scheme::rhs_initial_source(coefficients_, grid_cfg_, prev_solution_0, prev_solution_1, source_, boundary_pair,
                                    time, solution);
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void wave_euler_solver_method<fp_type, container, allocator>::solve_terminal(
    container<fp_type, allocator> &prev_solution_0, container<fp_type, allocator> &prev_solution_1,
    boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, fp_type const &next_time,
    container<fp_type, allocator> &solution)
{
    // get the right-hand side of the scheme:
    wave_scheme::rhs_terminal(coefficients_, grid_cfg_, prev_solution_0, prev_solution_1, boundary_pair, time,
                              solution);
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void wave_euler_solver_method<fp_type, container, allocator>::solve_terminal(
    container<fp_type, allocator> &prev_solution_0, container<fp_type, allocator> &prev_solution_1,
    boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, fp_type const &next_time,
    std::function<fp_type(fp_type, fp_type)> const &wave_source, container<fp_type, allocator> &solution)
{
    // get the right-hand side of the scheme:
    d_1d::of_function(grid_cfg_, time, wave_source, source_);
    wave_scheme::rhs_terminal_source(coefficients_, grid_cfg_, prev_solution_0, prev_solution_1, source_, boundary_pair,
                                     time, solution);
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void wave_euler_solver_method<fp_type, container, allocator>::solve(container<fp_type, allocator> &prev_solution_0,
                                                                    container<fp_type, allocator> &prev_solution_1,
                                                                    boundary_1d_pair<fp_type> const &boundary_pair,
                                                                    fp_type const &time, fp_type const &next_time,
                                                                    container<fp_type, allocator> &solution)
{
    // get the right-hand side of the scheme:
    wave_scheme::rhs(coefficients_, grid_cfg_, prev_solution_0, prev_solution_1, boundary_pair, time, solution);
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void wave_euler_solver_method<fp_type, container, allocator>::solve(
    container<fp_type, allocator> &prev_solution_0, container<fp_type, allocator> &prev_solution_1,
    boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, fp_type const &next_time,
    std::function<fp_type(fp_type, fp_type)> const &wave_source, container<fp_type, allocator> &solution)
{
    d_1d::of_function(grid_cfg_, time, wave_source, source_);
    wave_scheme::rhs_source(coefficients_, grid_cfg_, prev_solution_0, prev_solution_1, source_, boundary_pair, time,
                            solution);
}
} // namespace one_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_WAVE_EULER_SOLVER_METHOD_HPP_
