#if !defined(_LSS_WAVE_IMPLICIT_SOLVER_METHOD_HPP_)
#define _LSS_WAVE_IMPLICIT_SOLVER_METHOD_HPP_

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
#include "pde_solvers/one_dimensional/wave_type/implicit_coefficients/lss_wave_svc_implicit_coefficients.hpp"

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

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class implicit_wave_scheme
{
    typedef container<fp_type, allocator> container_t;

  public:
    static void rhs(wave_svc_implicit_coefficients_ptr<fp_type> const &cfs, grid_config_1d_ptr<fp_type> const &grid_cfg,
                    container_t const &input_0, container_t const &input_1,
                    boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, container_t &solution)
    {
        const fp_type two = static_cast<fp_type>(2.0);
        const fp_type one = static_cast<fp_type>(1.0);
        auto const &first_bnd = boundary_pair.first;
        auto const &second_bnd = boundary_pair.second;
        auto const &A = cfs->A_;
        auto const &B = cfs->B_;
        auto const &C = cfs->C_;
        auto const &D = cfs->D_;
        auto const k = cfs->k_;
        auto const h = grid_1d<fp_type>::step(grid_cfg);
        auto const lambda = one / (k * k);

        fp_type x{};
        // for lower boundaries first:
        x = grid_1d<fp_type>::value(grid_cfg, 0);
        if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(first_bnd))
        {
            const fp_type beta_curr = two * h * ptr->value(time);
            const fp_type beta_prev = two * h * ptr->value(time - k);
            solution[0] = two * (A(x) + B(x)) * input_1[1] + two * (lambda - C(x)) * input_1[0] +
                          (A(x) + B(x)) * input_0[1] - (D(x) + C(x)) * input_0[0] +
                          (two * beta_curr + beta_prev) * A(x);
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_bnd))
        {
            const fp_type beta_curr = two * h * ptr->value(time);
            const fp_type beta_prev = two * h * ptr->value(time - k);
            const fp_type alpha_curr = two * h * ptr->linear_value(time);
            const fp_type alpha_prev = two * h * ptr->linear_value(time - k);
            solution[0] = two * (A(x) + B(x)) * input_1[1] + two * (lambda - C(x) + alpha_curr * A(x)) * input_1[0] +
                          (A(x) + B(x)) * input_0[1] - (D(x) + C(x) - alpha_prev * A(x)) * input_0[0] +
                          (two * beta_curr + beta_prev) * A(x);
        }
        // for upper boundaries second:
        const std::size_t N = solution.size() - 1;
        x = grid_1d<fp_type>::value(grid_cfg, N);
        if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(second_bnd))
        {
            const fp_type delta_curr = two * h * ptr->value(time);
            const fp_type delta_prev = two * h * ptr->value(time - k);
            solution[N] = two * (A(x) + B(x)) * input_1[N - 1] + two * (lambda - C(x)) * input_1[N] +
                          (A(x) + B(x)) * input_0[N - 1] - (D(x) + C(x)) * input_0[N] -
                          (two * delta_curr + delta_prev) * B(x);
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
        {
            const fp_type delta_curr = two * h * ptr->value(time);
            const fp_type delta_prev = two * h * ptr->value(time - k);
            const fp_type gamma_curr = two * h * ptr->linear_value(time);
            const fp_type gamma_prev = two * h * ptr->linear_value(time - k);
            solution[N] = two * (A(x) + B(x)) * input_1[N - 1] +
                          two * (lambda - C(x) - gamma_curr * B(x)) * input_1[N] + (A(x) + B(x)) * input_0[N - 1] -
                          (D(x) + C(x) + gamma_prev * B(x)) * input_0[N] - (two * delta_curr + delta_prev) * B(x);
        }

        for (std::size_t t = 1; t < N; ++t)
        {
            x = grid_1d<fp_type>::value(grid_cfg, t);
            solution[t] = (B(x) * input_0[t + 1]) - ((D(x) + C(x)) * input_0[t]) + (A(x) * input_0[t - 1]) +
                          (two * B(x) * input_1[t + 1]) + (two * (lambda - C(x)) * input_1[t]) +
                          (two * A(x) * input_1[t - 1]);
        }
    }

    static void rhs_source(wave_svc_implicit_coefficients_ptr<fp_type> const &cfs,
                           grid_config_1d_ptr<fp_type> const &grid_cfg, container_t const &input_0,
                           container_t const &input_1, container_t const &inhom_input,
                           boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, container_t &solution)
    {
        const fp_type two = static_cast<fp_type>(2.0);
        const fp_type one = static_cast<fp_type>(1.0);
        auto const &first_bnd = boundary_pair.first;
        auto const &second_bnd = boundary_pair.second;
        auto const &A = cfs->A_;
        auto const &B = cfs->B_;
        auto const &C = cfs->C_;
        auto const &D = cfs->D_;
        auto const k = cfs->k_;
        auto const h = grid_1d<fp_type>::step(grid_cfg);
        auto const lambda = one / (k * k);
        fp_type x{};

        // for lower boundaries first:
        x = grid_1d<fp_type>::value(grid_cfg, 0);
        if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(first_bnd))
        {
            const fp_type beta_curr = two * h * ptr->value(time);
            const fp_type beta_prev = two * h * ptr->value(time - k);
            solution[0] = two * (A(x) + B(x)) * input_1[1] + two * (lambda - C(x)) * input_1[0] +
                          (A(x) + B(x)) * input_0[1] - (D(x) + C(x)) * input_0[0] +
                          (two * beta_curr + beta_prev) * A(x) + inhom_input[0];
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_bnd))
        {
            const fp_type beta_curr = two * h * ptr->value(time);
            const fp_type beta_prev = two * h * ptr->value(time - k);
            const fp_type alpha_curr = two * h * ptr->linear_value(time);
            const fp_type alpha_prev = two * h * ptr->linear_value(time - k);
            solution[0] = two * (A(x) + B(x)) * input_1[1] + two * (lambda - C(x) + alpha_curr * A(x)) * input_1[0] +
                          (A(x) + B(x)) * input_0[1] - (D(x) + C(x) - alpha_prev * A(x)) * input_0[0] +
                          (two * beta_curr + beta_prev) * A(x) + inhom_input[0];
            ;
        }
        // for upper boundaries second:
        const std::size_t N = solution.size() - 1;
        x = grid_1d<fp_type>::value(grid_cfg, N);
        if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(second_bnd))
        {
            const fp_type delta_curr = two * h * ptr->value(time);
            const fp_type delta_prev = two * h * ptr->value(time - k);
            solution[N] = two * (A(x) + B(x)) * input_1[N - 1] + two * (lambda - C(x)) * input_1[N] +
                          (A(x) + B(x)) * input_0[N - 1] - (D(x) + C(x)) * input_0[N] -
                          (two * delta_curr + delta_prev) * B(x) + inhom_input[N];
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
        {
            const fp_type delta_curr = two * h * ptr->value(time);
            const fp_type delta_prev = two * h * ptr->value(time - k);
            const fp_type gamma_curr = two * h * ptr->linear_value(time);
            const fp_type gamma_prev = two * h * ptr->linear_value(time - k);
            solution[N] = two * (A(x) + B(x)) * input_1[N - 1] +
                          two * (lambda - C(x) - gamma_curr * B(x)) * input_1[N] + (A(x) + B(x)) * input_0[N - 1] -
                          (D(x) + C(x) + gamma_prev * B(x)) * input_0[N] - (two * delta_curr + delta_prev) * B(x) +
                          inhom_input[N];
        }
        for (std::size_t t = 1; t < N; ++t)
        {
            x = grid_1d<fp_type>::value(grid_cfg, t);
            solution[t] = (B(x) * input_0[t + 1]) - ((D(x) + C(x)) * input_0[t]) + (A(x) * input_0[t - 1]) +
                          (two * B(x) * input_1[t + 1]) + (two * (lambda - C(x)) * input_1[t]) +
                          (two * A(x) * input_1[t - 1]) + inhom_input[t];
        }
    }

    static void rhs_initial(wave_svc_implicit_coefficients_ptr<fp_type> const &cfs,
                            grid_config_1d_ptr<fp_type> const &grid_cfg, container_t const &input_0,
                            container_t const &input_1, boundary_1d_pair<fp_type> const &boundary_pair,
                            fp_type const &time, container_t &solution)
    {
        const fp_type two = static_cast<fp_type>(2.0);
        auto const &first_bnd = boundary_pair.first;
        auto const &second_bnd = boundary_pair.second;
        auto const &A = cfs->A_;
        auto const &B = cfs->B_;
        auto const &C = cfs->C_;
        auto const &D = cfs->D_;
        auto const k = cfs->k_;
        auto const h = grid_1d<fp_type>::step(grid_cfg);
        auto const lambda = cfs->lambda_;
        auto const one_gamma = (two * k);

        fp_type x{};
        // for lower boundaries first:
        x = grid_1d<fp_type>::value(grid_cfg, 0);
        if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(first_bnd))
        {
            const fp_type beta = two * h * ptr->value(time);
            solution[0] = (two * (A(x) + B(x)) * input_0[1]) + (two * (lambda - C(x)) * input_0[0]) +
                          (two * beta * A(x)) + (one_gamma * (D(x) + C(x) - A(x) - B(x)) * input_1[0]);
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_bnd))
        {
            const fp_type beta = two * h * ptr->value(time);
            const fp_type alpha = two * h * ptr->linear_value(time);
            solution[0] = (two * (A(x) + B(x)) * input_0[1]) + (two * (lambda - C(x) + alpha * A(x)) * input_0[0]) +
                          (two * beta * A(x)) + (one_gamma * (D(x) + C(x) - A(x) - B(x)) * input_1[0]);
        }
        // for upper boundaries second:
        const std::size_t N = solution.size() - 1;
        x = grid_1d<fp_type>::value(grid_cfg, N);
        if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(second_bnd))
        {
            const fp_type delta = two * h * ptr->value(time);
            solution[N] = (two * (A(x) + B(x)) * input_0[N - 1]) + (two * (lambda - C(x)) * input_0[N]) -
                          (two * delta * B(x)) + (one_gamma * (D(x) + C(x) - A(x) - B(x)) * input_1[N]);
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
        {
            const fp_type delta = two * h * ptr->value(time);
            const fp_type gamma = two * h * ptr->linear_value(time);
            solution[N] = (two * (A(x) + B(x)) * input_0[N - 1]) + (two * (lambda - C(x) - gamma * B(x)) * input_0[N]) -
                          (two * delta * B(x)) + (one_gamma * (D(x) + C(x) - A(x) - B(x)) * input_1[N]);
        }

        for (std::size_t t = 1; t < N; ++t)
        {
            x = grid_1d<fp_type>::value(grid_cfg, t);
            solution[t] = (two * B(x) * input_0[t + 1]) + (two * (lambda - C(x)) * input_0[t]) +
                          (two * A(x) * input_0[t - 1]) - (one_gamma * B(x) * input_1[t + 1]) +
                          (one_gamma * (D(x) + C(x)) * input_1[t]) - (one_gamma * A(x) * input_1[t - 1]);
        }
    }

    static void rhs_initial_source(wave_svc_implicit_coefficients_ptr<fp_type> const &cfs,
                                   grid_config_1d_ptr<fp_type> const &grid_cfg, container_t const &input_0,
                                   container_t const &input_1, container_t const &inhom_input,
                                   boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time,
                                   container_t &solution)
    {
        const fp_type two = static_cast<fp_type>(2.0);
        auto const &first_bnd = boundary_pair.first;
        auto const &second_bnd = boundary_pair.second;
        auto const &A = cfs->A_;
        auto const &B = cfs->B_;
        auto const &C = cfs->C_;
        auto const &D = cfs->D_;
        auto const k = cfs->k_;
        auto const h = grid_1d<fp_type>::step(grid_cfg);
        auto const lambda = cfs->lambda_;
        auto const one_gamma = (two * k);

        fp_type x{};
        // for lower boundaries first:
        x = grid_1d<fp_type>::value(grid_cfg, 0);
        if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(first_bnd))
        {
            const fp_type beta = two * h * ptr->value(time);
            solution[0] = (two * (A(x) + B(x)) * input_0[1]) + (two * (lambda - C(x)) * input_0[0]) +
                          (two * beta * A(x)) + (one_gamma * (D(x) + C(x) - A(x) - B(x)) * input_1[0]) + inhom_input[0];
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_bnd))
        {
            const fp_type beta = two * h * ptr->value(time);
            const fp_type alpha = two * h * ptr->linear_value(time);
            solution[0] = (two * (A(x) + B(x)) * input_0[1]) + (two * (lambda - C(x) + alpha * A(x)) * input_0[0]) +
                          (two * beta * A(x)) + (one_gamma * (D(x) + C(x) - A(x) - B(x)) * input_1[0]) + inhom_input[0];
        }
        // for upper boundaries second:
        const std::size_t N = solution.size() - 1;
        x = grid_1d<fp_type>::value(grid_cfg, N);
        if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(second_bnd))
        {
            const fp_type delta = two * h * ptr->value(time);
            solution[N] = (two * (A(x) + B(x)) * input_0[N - 1]) + (two * (lambda - C(x)) * input_0[N]) -
                          (two * delta * B(x)) + (one_gamma * (D(x) + C(x) - A(x) - B(x)) * input_1[N]) +
                          inhom_input[N];
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
        {
            const fp_type delta = two * h * ptr->value(time);
            const fp_type gamma = two * h * ptr->linear_value(time);
            solution[N] = (two * (A(x) + B(x)) * input_0[N - 1]) + (two * (lambda - C(x) - gamma * B(x)) * input_0[N]) -
                          (two * delta * B(x)) + (one_gamma * (D(x) + C(x) - A(x) - B(x)) * input_1[N]) +
                          inhom_input[N];
        }

        for (std::size_t t = 1; t < N; ++t)
        {
            x = grid_1d<fp_type>::value(grid_cfg, t);
            solution[t] = (two * B(x) * input_0[t + 1]) + (two * (lambda - C(x)) * input_0[t]) +
                          (two * A(x) * input_0[t - 1]) - (one_gamma * B(x) * input_1[t + 1]) +
                          (one_gamma * (D(x) + C(x)) * input_1[t]) - (one_gamma * A(x) * input_1[t - 1]) +
                          inhom_input[t];
        }
    }

    static void rhs_terminal(wave_svc_implicit_coefficients_ptr<fp_type> const &cfs,
                             grid_config_1d_ptr<fp_type> const &grid_cfg, container_t const &input_0,
                             container_t const &input_1, boundary_1d_pair<fp_type> const &boundary_pair,
                             fp_type const &time, container_t &solution)
    {
        const fp_type two = static_cast<fp_type>(2.0);
        auto const &first_bnd = boundary_pair.first;
        auto const &second_bnd = boundary_pair.second;
        auto const &A = cfs->A_;
        auto const &B = cfs->B_;
        auto const &C = cfs->C_;
        auto const &D = cfs->D_;
        auto const k = cfs->k_;
        auto const h = grid_1d<fp_type>::step(grid_cfg);
        auto const lambda = cfs->lambda_;
        auto const one_gamma = (two * k);

        fp_type x{};
        // for lower boundaries first:
        x = grid_1d<fp_type>::value(grid_cfg, 0);
        if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(first_bnd))
        {
            const fp_type beta = two * h * ptr->value(time);
            solution[0] = (two * (A(x) + B(x)) * input_0[1]) + (two * (lambda - C(x)) * input_0[0]) +
                          (two * beta * A(x)) + (one_gamma * (A(x) + B(x) - C(x) - D(x)) * input_1[0]);
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_bnd))
        {
            const fp_type beta = two * h * ptr->value(time);
            const fp_type alpha = two * h * ptr->linear_value(time);
            solution[0] = (two * (A(x) + B(x)) * input_0[1]) + (two * (lambda - C(x) + alpha * A(x)) * input_0[0]) +
                          (two * beta * A(x)) + (one_gamma * (A(x) + B(x) - C(x) - D(x)) * input_1[0]);
        }
        // for upper boundaries second:
        const std::size_t N = solution.size() - 1;
        x = grid_1d<fp_type>::value(grid_cfg, N);
        if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(second_bnd))
        {
            const fp_type delta = two * h * ptr->value(time);
            solution[N] = (two * (A(x) + B(x)) * input_0[N - 1]) + (two * (lambda - C(x)) * input_0[N]) -
                          (two * delta * B(x)) + (one_gamma * (A(x) + B(x) - C(x) - D(x)) * input_1[N]);
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
        {
            const fp_type delta = two * h * ptr->value(time);
            const fp_type gamma = two * h * ptr->linear_value(time);
            solution[N] = (two * (A(x) + B(x)) * input_0[N - 1]) + (two * (lambda - C(x) - gamma * B(x)) * input_0[N]) -
                          (two * delta * B(x)) + (one_gamma * (A(x) + B(x) - C(x) - D(x)) * input_1[N]);
        }

        for (std::size_t t = 1; t < N; ++t)
        {
            x = grid_1d<fp_type>::value(grid_cfg, t);
            solution[t] = (two * B(x) * input_0[t + 1]) + (two * (lambda - C(x)) * input_0[t]) +
                          (two * A(x) * input_0[t - 1]) + (one_gamma * B(x) * input_1[t + 1]) -
                          (one_gamma * (D(x) + C(x)) * input_1[t]) + (one_gamma * A(x) * input_1[t - 1]);
        }
    }

    static void rhs_terminal_source(wave_svc_implicit_coefficients_ptr<fp_type> const &cfs,
                                    grid_config_1d_ptr<fp_type> const &grid_cfg, container_t const &input_0,
                                    container_t const &input_1, container_t const &inhom_input,
                                    boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time,
                                    container_t &solution)
    {
        const fp_type two = static_cast<fp_type>(2.0);
        auto const &first_bnd = boundary_pair.first;
        auto const &second_bnd = boundary_pair.second;
        auto const &A = cfs->A_;
        auto const &B = cfs->B_;
        auto const &C = cfs->C_;
        auto const &D = cfs->D_;
        auto const k = cfs->k_;
        auto const h = grid_1d<fp_type>::step(grid_cfg);
        auto const lambda = cfs->lambda_;
        auto const one_gamma = (two * k);

        fp_type x{};
        // for lower boundaries first:
        x = grid_1d<fp_type>::value(grid_cfg, 0);
        if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(first_bnd))
        {
            const fp_type beta = two * h * ptr->value(time);
            solution[0] = (two * (A(x) + B(x)) * input_0[1]) + (two * (lambda - C(x)) * input_0[0]) +
                          (two * beta * A(x)) + (one_gamma * (A(x) + B(x) - C(x) - D(x)) * input_1[0]) + inhom_input[0];
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_bnd))
        {
            const fp_type beta = two * h * ptr->value(time);
            const fp_type alpha = two * h * ptr->linear_value(time);
            solution[0] = (two * (A(x) + B(x)) * input_0[1]) + (two * (lambda - C(x) + alpha * A(x)) * input_0[0]) +
                          (two * beta * A(x)) + (one_gamma * (A(x) + B(x) - C(x) - D(x)) * input_1[0]) + inhom_input[0];
        }
        // for upper boundaries second:
        const std::size_t N = solution.size() - 1;
        x = grid_1d<fp_type>::value(grid_cfg, N);
        if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(second_bnd))
        {
            const fp_type delta = two * h * ptr->value(time);
            solution[N] = (two * (A(x) + B(x)) * input_0[N - 1]) + (two * (lambda - C(x)) * input_0[N]) -
                          (two * delta * B(x)) + (one_gamma * (A(x) + B(x) - C(x) - D(x)) * input_1[N]) +
                          inhom_input[N];
        }
        else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
        {
            const fp_type delta = two * h * ptr->value(time);
            const fp_type gamma = two * h * ptr->linear_value(time);
            solution[N] = (two * (A(x) + B(x)) * input_0[N - 1]) + (two * (lambda - C(x) - gamma * B(x)) * input_0[N]) -
                          (two * delta * B(x)) + (one_gamma * (A(x) + B(x) - C(x) - D(x)) * input_1[N]) +
                          inhom_input[N];
        }

        for (std::size_t t = 1; t < N; ++t)
        {
            x = grid_1d<fp_type>::value(grid_cfg, t);
            solution[t] = (two * B(x) * input_0[t + 1]) + (two * (lambda - C(x)) * input_0[t]) +
                          (two * A(x) * input_0[t - 1]) + (one_gamma * B(x) * input_1[t + 1]) -
                          (one_gamma * (D(x) + C(x)) * input_1[t]) + (one_gamma * A(x) * input_1[t - 1]) +
                          inhom_input[t];
        }
    }
};

/**
wave_implicit_solver_method object
*/
template <typename fp_type, typename solver, template <typename, typename> typename container = std::vector,
          typename allocator = std::allocator<fp_type>>
class wave_implicit_solver_method
{
    typedef container<fp_type, allocator> container_t;

  private:
    // constant coeffs:
    const fp_type ctwo_ = static_cast<fp_type>(2.0);
    const fp_type cone_ = static_cast<fp_type>(1.0);
    // solvers:
    solver solveru_ptr_;
    // scheme coefficients:
    wave_svc_implicit_coefficients_ptr<fp_type> coefficients_;
    grid_config_1d_ptr<fp_type> grid_cfg_;
    // container:
    container_t low_, diag_, high_;
    container_t rhs_, source_;

    explicit wave_implicit_solver_method() = delete;

    void initialize(bool is_wave_source_set)
    {
        low_.resize(coefficients_->space_size_);
        diag_.resize(coefficients_->space_size_);
        high_.resize(coefficients_->space_size_);
        rhs_.resize(coefficients_->space_size_);
        if (is_wave_source_set)
        {
            source_.resize(coefficients_->space_size_);
        }
    }

    void split_0(container_t &low_0, container_t &diag_0, container_t &high_0)
    {
        fp_type x{};
        for (std::size_t t = 0; t < low_0.size(); ++t)
        {
            x = grid_1d<fp_type>::value(grid_cfg_, t);
            low_0[t] = ctwo_ * (-cone_ * coefficients_->A_(x));
            diag_0[t] = (coefficients_->E_(x) + ctwo_ * coefficients_->C_(x) + coefficients_->D_(x));
            high_0[t] = ctwo_ * (-cone_ * coefficients_->B_(x));
        }
    }

    void split_1(container_t &low_1, container_t &diag_1, container_t &high_1)
    {
        fp_type x{};
        for (std::size_t t = 0; t < low_1.size(); ++t)
        {
            x = grid_1d<fp_type>::value(grid_cfg_, t);
            low_1[t] = -cone_ * coefficients_->A_(x);
            diag_1[t] = (coefficients_->E_(x) + coefficients_->C_(x));
            high_1[t] = -cone_ * coefficients_->B_(x);
        }
    }

  public:
    explicit wave_implicit_solver_method(solver const &solver_ptr,
                                         wave_svc_implicit_coefficients_ptr<fp_type> const &coefficients,
                                         grid_config_1d_ptr<fp_type> const &grid_config, bool is_wave_source_set)
        : solveru_ptr_{solver_ptr}, coefficients_{coefficients}, grid_cfg_{grid_config}
    {
        initialize(is_wave_source_set);
    }

    ~wave_implicit_solver_method()
    {
    }

    wave_implicit_solver_method(wave_implicit_solver_method const &) = delete;
    wave_implicit_solver_method(wave_implicit_solver_method &&) = delete;
    wave_implicit_solver_method &operator=(wave_implicit_solver_method const &) = delete;
    wave_implicit_solver_method &operator=(wave_implicit_solver_method &&) = delete;

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

template <typename fp_type, typename solver, template <typename, typename> typename container, typename allocator>
void wave_implicit_solver_method<fp_type, solver, container, allocator>::solve_initial(
    container<fp_type, allocator> &prev_solution_0, container<fp_type, allocator> &prev_solution_1,
    boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, fp_type const &next_time,
    container<fp_type, allocator> &solution)
{
    typedef implicit_wave_scheme<fp_type, container, allocator> wave_scheme;

    // containers for first split solver:
    // container_t low_0(coefficients_->space_size_, fp_type{});
    // container_t diag_0(coefficients_->space_size_, fp_type{});
    // container_t high_0(coefficients_->space_size_, fp_type{});
    // container_t rhs(coefficients_->space_size_, fp_type{});
    // get the right-hand side of the scheme:
    wave_scheme::rhs_initial(coefficients_, grid_cfg_, prev_solution_0, prev_solution_1, boundary_pair, time, rhs_);
    split_0(low_, diag_, high_);
    solveru_ptr_->set_diagonals(low_, diag_, high_);
    solveru_ptr_->set_rhs(rhs_);
    solveru_ptr_->solve(boundary_pair, solution, next_time);
}

template <typename fp_type, typename solver, template <typename, typename> typename container, typename allocator>
void wave_implicit_solver_method<fp_type, solver, container, allocator>::solve_initial(
    container<fp_type, allocator> &prev_solution_0, container<fp_type, allocator> &prev_solution_1,
    boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, fp_type const &next_time,
    std::function<fp_type(fp_type, fp_type)> const &wave_source, container<fp_type, allocator> &solution)
{
    typedef implicit_wave_scheme<fp_type, container, allocator> wave_scheme;
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;

    // containers for first split solver:
    // container_t low_0(coefficients_->space_size_, fp_type{});
    // container_t diag_0(coefficients_->space_size_, fp_type{});
    // container_t high_0(coefficients_->space_size_, fp_type{});
    // container_t rhs(coefficients_->space_size_, fp_type{});
    // container_t source(coefficients_->space_size_, fp_type{});
    //// get the right-hand side of the scheme:
    // auto scheme = wave_scheme::get_initial(false);
    d_1d::of_function(grid_cfg_, time, wave_source, source_);
    wave_scheme::rhs_initial_source(coefficients_, grid_cfg_, prev_solution_0, prev_solution_1, source_, boundary_pair,
                                    time, rhs_);
    split_0(low_, diag_, high_);
    solveru_ptr_->set_diagonals(low_, diag_, high_);
    solveru_ptr_->set_rhs(rhs_);
    solveru_ptr_->solve(boundary_pair, solution, next_time);
}

template <typename fp_type, typename solver, template <typename, typename> typename container, typename allocator>
void wave_implicit_solver_method<fp_type, solver, container, allocator>::solve_terminal(
    container<fp_type, allocator> &prev_solution_0, container<fp_type, allocator> &prev_solution_1,
    boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, fp_type const &next_time,
    container<fp_type, allocator> &solution)
{
    typedef implicit_wave_scheme<fp_type, container, allocator> wave_scheme;

    // containers for first split solver:
    // container_t low_0(coefficients_->space_size_, fp_type{});
    // container_t diag_0(coefficients_->space_size_, fp_type{});
    // container_t high_0(coefficients_->space_size_, fp_type{});
    // container_t rhs(coefficients_->space_size_, fp_type{});
    // get the right-hand side of the scheme:
    wave_scheme::rhs_terminal(coefficients_, grid_cfg_, prev_solution_0, prev_solution_1, boundary_pair, time, rhs_);
    split_0(low_, diag_, high_);
    solveru_ptr_->set_diagonals(low_, diag_, high_);
    solveru_ptr_->set_rhs(rhs_);
    solveru_ptr_->solve(boundary_pair, solution, next_time);
}

template <typename fp_type, typename solver, template <typename, typename> typename container, typename allocator>
void wave_implicit_solver_method<fp_type, solver, container, allocator>::solve_terminal(
    container<fp_type, allocator> &prev_solution_0, container<fp_type, allocator> &prev_solution_1,
    boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, fp_type const &next_time,
    std::function<fp_type(fp_type, fp_type)> const &wave_source, container<fp_type, allocator> &solution)
{
    typedef implicit_wave_scheme<fp_type, container, allocator> wave_scheme;
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;

    // containers for first split solver:
    // container_t low_0(coefficients_->space_size_, fp_type{});
    // container_t diag_0(coefficients_->space_size_, fp_type{});
    // container_t high_0(coefficients_->space_size_, fp_type{});
    // container_t rhs(coefficients_->space_size_, fp_type{});
    // container_t source(coefficients_->space_size_, fp_type{});
    // get the right-hand side of the scheme:
    d_1d::of_function(grid_cfg_, time, wave_source, source_);
    wave_scheme::rhs_terminal_source(coefficients_, grid_cfg_, prev_solution_0, prev_solution_1, source_, boundary_pair,
                                     time, rhs_);
    split_0(low_, diag_, high_);
    solveru_ptr_->set_diagonals(low_, diag_, high_);
    solveru_ptr_->set_rhs(rhs_);
    solveru_ptr_->solve(boundary_pair, solution, next_time);
}

template <typename fp_type, typename solver, template <typename, typename> typename container, typename allocator>
void wave_implicit_solver_method<fp_type, solver, container, allocator>::solve(
    container<fp_type, allocator> &prev_solution_0, container<fp_type, allocator> &prev_solution_1,
    boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, fp_type const &next_time,
    container<fp_type, allocator> &solution)
{
    typedef implicit_wave_scheme<fp_type, container, allocator> wave_scheme;

    // containers for first split solver:
    // container_t low_1(coefficients_->space_size_, fp_type{});
    // container_t diag_1(coefficients_->space_size_, fp_type{});
    // container_t high_1(coefficients_->space_size_, fp_type{});
    // container_t rhs(coefficients_->space_size_, fp_type{});
    // get the right-hand side of the scheme:
    split_1(low_, diag_, high_);
    wave_scheme::rhs(coefficients_, grid_cfg_, prev_solution_0, prev_solution_1, boundary_pair, time, rhs_);
    solveru_ptr_->set_diagonals(low_, diag_, high_);
    solveru_ptr_->set_rhs(rhs_);
    solveru_ptr_->solve(boundary_pair, solution, next_time);
}

template <typename fp_type, typename solver, template <typename, typename> typename container, typename allocator>
void wave_implicit_solver_method<fp_type, solver, container, allocator>::solve(
    container<fp_type, allocator> &prev_solution_0, container<fp_type, allocator> &prev_solution_1,
    boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, fp_type const &next_time,
    std::function<fp_type(fp_type, fp_type)> const &wave_source, container<fp_type, allocator> &solution)
{
    typedef implicit_wave_scheme<fp_type, container, allocator> wave_scheme;
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;

    // containers for first split solver:
    // container_t low_1(coefficients_->space_size_, fp_type{});
    // container_t diag_1(coefficients_->space_size_, fp_type{});
    // container_t high_1(coefficients_->space_size_, fp_type{});
    // container_t rhs(coefficients_->space_size_, fp_type{});
    // container_t source(coefficients_->space_size_, fp_type{});
    // get the right-hand side of the scheme:
    split_1(low_, diag_, high_);
    d_1d::of_function(grid_cfg_, time, wave_source, source_);
    wave_scheme::rhs_source(coefficients_, grid_cfg_, prev_solution_0, prev_solution_1, source_, boundary_pair, time,
                            rhs_);
    solveru_ptr_->set_diagonals(low_, diag_, high_);
    solveru_ptr_->set_rhs(rhs_);
    solveru_ptr_->solve(boundary_pair, solution, next_time);
}
} // namespace one_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_WAVE_IMPLICIT_SOLVER_METHOD_HPP_
