#if !defined(_LSS_WAVE_EULER_SVC_SCHEME_HPP_)
#define _LSS_WAVE_EULER_SVC_SCHEME_HPP_

#include "boundaries/lss_boundary.hpp"
#include "boundaries/lss_dirichlet_boundary.hpp"
#include "boundaries/lss_neumann_boundary.hpp"
#include "boundaries/lss_robin_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include "containers/lss_container_2d.hpp"
#include "discretization/lss_discretization.hpp"
#include "pde_solvers/lss_pde_discretization_config.hpp"

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
using lss_enumerations::traverse_direction_enum;
using lss_utility::function_quintuple_t;
using lss_utility::function_triplet_t;
using lss_utility::NaN;
using lss_utility::pair_t;
using lss_utility::range;

template <template <typename, typename> typename container, typename fp_type, typename alloc>
using explicit_wave_scheme_function =
    std::function<void(function_quintuple_t<fp_type> const &, pair_t<fp_type> const &,
                       container<fp_type, alloc> const &, container<fp_type, alloc> const &,
                       container<fp_type, alloc> const &, boundary_1d_pair<fp_type> const &, fp_type const &,
                       container<fp_type, alloc> &)>;

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class explicit_wave_scheme
{
    typedef container<fp_type, allocator> container_t;
    typedef explicit_wave_scheme_function<container, fp_type, allocator> scheme_function_t;

  public:
    static scheme_function_t const get(bool is_homogeneus)
    {
        const fp_type two = static_cast<fp_type>(2.0);
        auto scheme_fun_h =
            [=](function_quintuple_t<fp_type> const &coefficients, std::pair<fp_type, fp_type> const &steps,
                container_t const &input_0, container_t const &input_1, container_t const &inhom_input,
                boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, container_t &solution) {
                auto const &first_bnd = boundary_pair.first;
                auto const &second_bnd = boundary_pair.second;
                auto const &a = std::get<0>(coefficients);
                auto const &b = std::get<1>(coefficients);
                auto const &c = std::get<2>(coefficients);
                auto const &d = std::get<3>(coefficients);
                auto const h = steps.second;
                fp_type m{};
                // for lower boundaries first:
                if (auto const &ptr = std::dynamic_pointer_cast<dirichlet_boundary_1d<fp_type>>(first_bnd))
                {
                    solution[0] = ptr->value(time);
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(first_bnd))
                {
                    const fp_type beta = two * h * ptr->value(time);
                    m = static_cast<fp_type>(0);
                    solution[0] = beta * a(m * h) + c(m * h) * input_1[0] + (a(m * h) + b(m * h)) * input_1[1] -
                                  d(m * h) * input_0[0];
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_bnd))
                {
                    const fp_type beta = two * h * ptr->value(time);
                    const fp_type alpha = two * h * ptr->linear_value(time);
                    m = static_cast<fp_type>(0);
                    solution[0] = beta * a(m * h) + (c(m * h) + alpha * a(m * h)) * input_1[0] +
                                  (a(m * h) + b(m * h)) * input_1[1] - d(m * h) * input_0[0];
                }
                // for upper boundaries second:
                const std::size_t N = solution.size() - 1;
                if (auto const &ptr = std::dynamic_pointer_cast<dirichlet_boundary_1d<fp_type>>(second_bnd))
                {
                    solution[N] = ptr->value(time);
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(second_bnd))
                {
                    const fp_type delta = two * h * ptr->value(time);
                    m = static_cast<fp_type>(N);
                    solution[N] = (a(m * h) + b(m * h)) * input_1[N - 1] + c(m * h) * input_1[N] - delta * b(m * h) -
                                  d(m * h) * input_0[N];
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
                {
                    const fp_type delta = two * h * ptr->value(time);
                    const fp_type gamma = two * h * ptr->linear_value(time);
                    m = static_cast<fp_type>(N);
                    solution[N] = (a(m * h) + b(m * h)) * input_1[N - 1] + (c(m * h) - gamma * b(m * h)) * input_1[N] -
                                  delta * b(m * h) - d(m * h) * input_0[N];
                }

                for (std::size_t t = 1; t < N; ++t)
                {
                    m = static_cast<fp_type>(t);
                    solution[t] = (a(m * h) * input_1[t - 1]) + (c(m * h) * input_1[t]) + (b(m * h) * input_1[t + 1]) -
                                  (d(m * h) * input_0[t]);
                }
            };
        auto scheme_fun_nh =
            [=](function_quintuple_t<fp_type> const &coefficients, std::pair<fp_type, fp_type> const &steps,
                container_t const &input_0, container_t const &input_1, container_t const &inhom_input,
                boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, container_t &solution) {
                auto const &first_bnd = boundary_pair.first;
                auto const &second_bnd = boundary_pair.second;
                auto const &a = std::get<0>(coefficients);
                auto const &b = std::get<1>(coefficients);
                auto const &c = std::get<2>(coefficients);
                auto const &d = std::get<3>(coefficients);
                auto const h = steps.second;
                fp_type m{};
                // for lower boundaries first:
                if (auto const &ptr = std::dynamic_pointer_cast<dirichlet_boundary_1d<fp_type>>(first_bnd))
                {
                    solution[0] = ptr->value(time);
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(first_bnd))
                {
                    const fp_type beta = two * h * ptr->value(time);
                    m = static_cast<fp_type>(0);
                    solution[0] = beta * a(m * h) + c(m * h) * input_1[0] + (a(m * h) + b(m * h)) * input_1[1] -
                                  d(m * h) * input_0[0] + inhom_input[0];
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_bnd))
                {
                    const fp_type beta = two * h * ptr->value(time);
                    const fp_type alpha = two * h * ptr->linear_value(time);
                    m = static_cast<fp_type>(0);
                    solution[0] = beta * a(m * h) + (c(m * h) + alpha * a(m * h)) * input_1[0] +
                                  (a(m * h) + b(m * h)) * input_1[1] - d(m * h) * input_0[0] + inhom_input[0];
                }
                // for upper boundaries second:
                const std::size_t N = solution.size() - 1;
                if (auto const &ptr = std::dynamic_pointer_cast<dirichlet_boundary_1d<fp_type>>(second_bnd))
                {
                    solution[N] = ptr->value(time);
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(second_bnd))
                {
                    const fp_type delta = two * h * ptr->value(time);
                    m = static_cast<fp_type>(N);
                    solution[N] = (a(m * h) + b(m * h)) * input_1[N - 1] + c(m * h) * input_1[N] - delta * b(m * h) -
                                  d(m * h) * input_0[N] + inhom_input[N];
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
                {
                    const fp_type delta = two * h * ptr->value(time);
                    const fp_type gamma = two * h * ptr->linear_value(time);
                    m = static_cast<fp_type>(N);
                    solution[N] = (a(m * h) + b(m * h)) * input_1[N - 1] + (c(m * h) - gamma * b(m * h)) * input_1[N] -
                                  delta * b(m * h) - d(m * h) * input_0[N] + inhom_input[N];
                }

                for (std::size_t t = 1; t < N; ++t)
                {
                    m = static_cast<fp_type>(t);
                    solution[t] = (a(m * h) * input_1[t - 1]) + (c(m * h) * input_1[t]) + (b(m * h) * input_1[t + 1]) -
                                  (d(m * h) * input_0[t]) + inhom_input[t];
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

    static scheme_function_t const get_initial(bool is_homogeneus)
    {
        const fp_type one = static_cast<fp_type>(1.0);
        const fp_type two = static_cast<fp_type>(2.0);
        auto scheme_fun_h =
            [=](function_quintuple_t<fp_type> const &coefficients, std::pair<fp_type, fp_type> const &steps,
                container_t const &input_0, container_t const &input_1, container_t const &inhom_input,
                boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, container_t &solution) {
                auto const &first_bnd = boundary_pair.first;
                auto const &second_bnd = boundary_pair.second;
                auto const &a = std::get<0>(coefficients);
                auto const &b = std::get<1>(coefficients);
                auto const &c = std::get<2>(coefficients);
                auto const &d = std::get<3>(coefficients);
                auto const h = steps.second;
                auto const k = steps.first;
                auto const one_gamma = (two * k);
                auto const &defl = [&](fp_type x) { return (one + d(x)); };
                auto const &A = [&](fp_type x) { return (a(x) / defl(x)); };
                auto const &B = [&](fp_type x) { return (b(x) / defl(x)); };
                auto const &C = [&](fp_type x) { return (c(x) / defl(x)); };
                auto const &D = [&](fp_type x) { return (one_gamma * (d(x) / defl(x))); };

                fp_type m{};
                // for lower boundaries first:
                if (auto const &ptr = std::dynamic_pointer_cast<dirichlet_boundary_1d<fp_type>>(first_bnd))
                {
                    solution[0] = ptr->value(time);
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(first_bnd))
                {
                    const fp_type beta = two * h * ptr->value(time);
                    m = static_cast<fp_type>(0);
                    solution[0] = beta * A(m * h) + (C(m * h) * input_0[0]) + ((A(m * h) + B(m * h)) * input_0[1]) +
                                  (D(m * h) * input_1[0]);
                    ;
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_bnd))
                {
                    const fp_type beta = two * h * ptr->value(time);
                    const fp_type alpha = two * h * ptr->linear_value(time);
                    m = static_cast<fp_type>(0);
                    solution[0] = beta * A(m * h) + (C(m * h) + alpha * A(m * h)) * input_0[0] +
                                  (A(m * h) + B(m * h)) * input_0[1] + (D(m * h) * input_1[0]);
                }
                // for upper boundaries second:
                const std::size_t N = solution.size() - 1;
                if (auto const &ptr = std::dynamic_pointer_cast<dirichlet_boundary_1d<fp_type>>(second_bnd))
                {
                    solution[N] = ptr->value(time);
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(second_bnd))
                {
                    const fp_type delta = two * h * ptr->value(time);
                    m = static_cast<fp_type>(N);
                    solution[N] = (A(m * h) + B(m * h)) * input_0[N - 1] + C(m * h) * input_0[N] - delta * B(m * h) +
                                  (D(m * h) * input_1[N]);
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
                {
                    const fp_type delta = two * h * ptr->value(time);
                    const fp_type gamma = two * h * ptr->linear_value(time);
                    m = static_cast<fp_type>(N);
                    solution[N] = (A(m * h) + B(m * h)) * input_0[N - 1] + (C(m * h) - gamma * B(m * h)) * input_0[N] -
                                  delta * B(m * h) + (D(m * h) * input_1[N]);
                }

                for (std::size_t t = 1; t < N; ++t)
                {
                    m = static_cast<fp_type>(t);
                    solution[t] = (A(m * h) * input_0[t - 1]) + (C(m * h) * input_0[t]) + (B(m * h) * input_0[t + 1]) +
                                  (D(m * h) * input_1[t]);
                }
            };
        auto scheme_fun_nh =
            [=](function_quintuple_t<fp_type> const &coefficients, std::pair<fp_type, fp_type> const &steps,
                container_t const &input_0, container_t const &input_1, container_t const &inhom_input,
                boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, container_t &solution) {
                auto const &first_bnd = boundary_pair.first;
                auto const &second_bnd = boundary_pair.second;
                auto const &a = std::get<0>(coefficients);
                auto const &b = std::get<1>(coefficients);
                auto const &c = std::get<2>(coefficients);
                auto const &d = std::get<3>(coefficients);
                auto const &defl = [=](fp_type x) { return (one + d(x)); };
                auto const &A = [=](fp_type x) { return (a(x) / defl(x)); };
                auto const &B = [=](fp_type x) { return (b(x) / defl(x)); };
                auto const &C = [=](fp_type x) { return (c(x) / defl(x)); };

                auto const h = steps.second;
                auto const k = steps.first;
                auto const one_gamma = (two * k);
                fp_type m{};
                // for lower boundaries first:
                if (auto const &ptr = std::dynamic_pointer_cast<dirichlet_boundary_1d<fp_type>>(first_bnd))
                {
                    solution[0] = ptr->value(time);
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(first_bnd))
                {
                    const fp_type beta = two * h * ptr->value(time);
                    m = static_cast<fp_type>(0);
                    solution[0] = beta * A(m * h) + C(m * h) * input_0[0] + (A(m * h) + B(m * h)) * input_0[1] +
                                  ((one_gamma * d(m * h)) / defl(m * h)) * input_1[0] + (inhom_input[0] / defl(m * h));
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_bnd))
                {
                    const fp_type beta = two * h * ptr->value(time);
                    const fp_type alpha = two * h * ptr->linear_value(time);
                    m = static_cast<fp_type>(0);
                    solution[0] = beta * A(m * h) + (C(m * h) + alpha * A(m * h)) * input_0[0] +
                                  (A(m * h) + B(m * h)) * input_0[1] +
                                  ((one_gamma * d(m * h)) / defl(m * h)) * input_1[0] + (inhom_input[0] / defl(m * h));
                }
                // for upper boundaries second:
                const std::size_t N = solution.size() - 1;
                if (auto const &ptr = std::dynamic_pointer_cast<dirichlet_boundary_1d<fp_type>>(second_bnd))
                {
                    solution[N] = ptr->value(time);
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(second_bnd))
                {
                    const fp_type delta = two * h * ptr->value(time);
                    m = static_cast<fp_type>(N);
                    solution[N] = (A(m * h) + B(m * h)) * input_0[N - 1] + C(m * h) * input_0[N] - delta * B(m * h) +
                                  ((one_gamma * d(m * h)) / defl(m * h)) * input_1[N] + (inhom_input[N] / defl(m * h));
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
                {
                    const fp_type delta = two * h * ptr->value(time);
                    const fp_type gamma = two * h * ptr->linear_value(time);
                    m = static_cast<fp_type>(N);
                    solution[N] = (A(m * h) + B(m * h)) * input_0[N - 1] + (C(m * h) - gamma * B(m * h)) * input_0[N] -
                                  delta * B(m * h) + ((one_gamma * d(m * h)) / defl(m * h)) * input_1[N] +
                                  (inhom_input[N] / defl(m * h));
                }

                for (std::size_t t = 1; t < N; ++t)
                {
                    m = static_cast<fp_type>(t);
                    solution[t] = (A(m * h) * input_0[t - 1]) + (C(m * h) * input_0[t]) + (B(m * h) * input_0[t + 1]) +
                                  ((one_gamma * d(m * h) / defl(m * h)) * input_1[t]) + (inhom_input[t] / defl(m * h));
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

    static scheme_function_t const get_terminal(bool is_homogeneus)
    {
        const fp_type one = static_cast<fp_type>(1.0);
        const fp_type two = static_cast<fp_type>(2.0);
        auto scheme_fun_h = [=](function_quintuple_t<fp_type> const &coefficients,
                                std::pair<fp_type, fp_type> const &steps, container_t const &input_0,
                                container_t const &input_1, container_t const &inhom_input,
                                boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time,
                                container_t &solution) {
            auto const &first_bnd = boundary_pair.first;
            auto const &second_bnd = boundary_pair.second;
            auto const &a = std::get<0>(coefficients);
            auto const &b = std::get<1>(coefficients);
            auto const &c = std::get<2>(coefficients);
            auto const &d = std::get<3>(coefficients);
            auto const &defl = [=](fp_type x) { return (one + d(x)); };
            auto const &A = [=](fp_type x) { return (a(x) / defl(x)); };
            auto const &B = [=](fp_type x) { return (b(x) / defl(x)); };
            auto const &C = [=](fp_type x) { return (c(x) / defl(x)); };

            auto const h = steps.second;
            auto const k = steps.first;
            auto const one_gamma = (two * k);
            fp_type m{};
            // for lower boundaries first:
            if (auto const &ptr = std::dynamic_pointer_cast<dirichlet_boundary_1d<fp_type>>(first_bnd))
            {
                solution[0] = ptr->value(time);
            }
            else if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(first_bnd))
            {
                const fp_type beta = two * h * ptr->value(time);
                m = static_cast<fp_type>(0);
                solution[0] = beta * A(m * h) + C(m * h) * input_0[0] + (A(m * h) + B(m * h)) * input_0[1] -
                              ((one_gamma * d(m * h)) / defl(m * h)) * input_1[0];
            }
            else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_bnd))
            {
                const fp_type beta = two * h * ptr->value(time);
                const fp_type alpha = two * h * ptr->linear_value(time);
                m = static_cast<fp_type>(0);
                solution[0] = beta * A(m * h) + (C(m * h) + alpha * A(m * h)) * input_0[0] +
                              (A(m * h) + B(m * h)) * input_0[1] - ((one_gamma * d(m * h)) / defl(m * h)) * input_1[0];
            }
            // for upper boundaries second:
            const std::size_t N = solution.size() - 1;
            if (auto const &ptr = std::dynamic_pointer_cast<dirichlet_boundary_1d<fp_type>>(second_bnd))
            {
                solution[N] = ptr->value(time);
            }
            else if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(second_bnd))
            {
                const fp_type delta = two * h * ptr->value(time);
                m = static_cast<fp_type>(N);
                solution[N] = (A(m * h) + B(m * h)) * input_0[N - 1] + C(m * h) * input_0[N] - delta * B(m * h) -
                              ((one_gamma * d(m * h)) / defl(m * h)) * input_1[N];
            }
            else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
            {
                const fp_type delta = two * h * ptr->value(time);
                const fp_type gamma = two * h * ptr->linear_value(time);
                m = static_cast<fp_type>(N);
                solution[N] = (A(m * h) + B(m * h)) * input_0[N - 1] + (C(m * h) - gamma * B(m * h)) * input_0[N] -
                              delta * B(m * h) - ((one_gamma * d(m * h)) / defl(m * h)) * input_1[N];
            }

            for (std::size_t t = 1; t < N; ++t)
            {
                m = static_cast<fp_type>(t);
                solution[t] = (A(m * h) * input_0[t - 1]) + (C(m * h) * input_0[t]) + (B(m * h) * input_0[t + 1]) -
                              ((one_gamma * d(m * h) / defl(m * h)) * input_1[t]);
            }
        };
        auto scheme_fun_nh =
            [=](function_quintuple_t<fp_type> const &coefficients, std::pair<fp_type, fp_type> const &steps,
                container_t const &input_0, container_t const &input_1, container_t const &inhom_input,
                boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, container_t &solution) {
                auto const &first_bnd = boundary_pair.first;
                auto const &second_bnd = boundary_pair.second;
                auto const &a = std::get<0>(coefficients);
                auto const &b = std::get<1>(coefficients);
                auto const &c = std::get<2>(coefficients);
                auto const &d = std::get<3>(coefficients);
                auto const &defl = [=](fp_type x) { return (one + d(x)); };
                auto const &A = [=](fp_type x) { return (a(x) / defl(x)); };
                auto const &B = [=](fp_type x) { return (b(x) / defl(x)); };
                auto const &C = [=](fp_type x) { return (c(x) / defl(x)); };

                auto const h = steps.second;
                auto const k = steps.first;
                auto const one_gamma = (two * k);
                fp_type m{};
                // for lower boundaries first:
                if (auto const &ptr = std::dynamic_pointer_cast<dirichlet_boundary_1d<fp_type>>(first_bnd))
                {
                    solution[0] = ptr->value(time);
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(first_bnd))
                {
                    const fp_type beta = two * h * ptr->value(time);
                    m = static_cast<fp_type>(0);
                    solution[0] = beta * A(m * h) + C(m * h) * input_0[0] + (A(m * h) + B(m * h)) * input_0[1] -
                                  ((one_gamma * d(m * h)) / defl(m * h)) * input_1[0] + (inhom_input[0] / defl(m * h));
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_bnd))
                {
                    const fp_type beta = two * h * ptr->value(time);
                    const fp_type alpha = two * h * ptr->linear_value(time);
                    m = static_cast<fp_type>(0);
                    solution[0] = beta * A(m * h) + (C(m * h) + alpha * A(m * h)) * input_0[0] +
                                  (A(m * h) + B(m * h)) * input_0[1] -
                                  ((one_gamma * d(m * h)) / defl(m * h)) * input_1[0] + (inhom_input[0] / defl(m * h));
                }
                // for upper boundaries second:
                const std::size_t N = solution.size() - 1;
                if (auto const &ptr = std::dynamic_pointer_cast<dirichlet_boundary_1d<fp_type>>(second_bnd))
                {
                    solution[N] = ptr->value(time);
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(second_bnd))
                {
                    const fp_type delta = two * h * ptr->value(time);
                    m = static_cast<fp_type>(N);
                    solution[N] = (A(m * h) + B(m * h)) * input_0[N - 1] + C(m * h) * input_0[N] - delta * B(m * h) -
                                  ((one_gamma * d(m * h)) / defl(m * h)) * input_1[N] + (inhom_input[N] / defl(m * h));
                }
                else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
                {
                    const fp_type delta = two * h * ptr->value(time);
                    const fp_type gamma = two * h * ptr->linear_value(time);
                    m = static_cast<fp_type>(N);
                    solution[N] = (A(m * h) + B(m * h)) * input_0[N - 1] + (C(m * h) - gamma * B(m * h)) * input_0[N] -
                                  delta * B(m * h) - ((one_gamma * d(m * h)) / defl(m * h)) * input_1[N] +
                                  (inhom_input[N] / defl(m * h));
                }

                for (std::size_t t = 1; t < N; ++t)
                {
                    m = static_cast<fp_type>(t);
                    solution[t] = (A(m * h) * input_0[t - 1]) + (C(m * h) * input_0[t]) + (B(m * h) * input_0[t + 1]) -
                                  ((one_gamma * d(m * h) / defl(m * h)) * input_1[t]) + (inhom_input[t] / defl(m * h));
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
 * wave_euler_svc_time_loop object
 */
template <typename fp_type, template <typename, typename> typename container, typename allocator>
class wave_euler_svc_time_loop
{
    typedef container<fp_type, allocator> container_t;
    typedef container_2d<fp_type, container, allocator> container_2d_t;

  public:
    static void run(function_quintuple_t<fp_type> const &fun_quintuple, boundary_1d_pair<fp_type> const &boundary_pair,
                    range<fp_type> const &space_range, range<fp_type> const &time_range,
                    std::size_t const &last_time_idx, std::pair<fp_type, fp_type> const &steps,
                    traverse_direction_enum const &traverse_dir, container_t &prev_solution_0,
                    container_t &prev_solution_1, container_t &next_solution);

    static void run(function_quintuple_t<fp_type> const &fun_quintuple, boundary_1d_pair<fp_type> const &boundary_pair,
                    range<fp_type> const &space_range, range<fp_type> const &time_range,
                    std::size_t const &last_time_idx, std::pair<fp_type, fp_type> const &steps,
                    traverse_direction_enum const &traverse_dir, container_t &prev_solution_0,
                    container_t &prev_solution_1, container_t &next_solution,
                    std::function<fp_type(fp_type, fp_type)> const &wave_source, container_t &source);

    static void run_with_stepping(function_quintuple_t<fp_type> const &fun_quintuple,
                                  boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &space_range,
                                  range<fp_type> const &time_range, std::size_t const &last_time_idx,
                                  std::pair<fp_type, fp_type> const &steps, traverse_direction_enum const &traverse_dir,
                                  container_t &prev_solution_0, container_t &prev_solution_1,
                                  container_t &next_solution, container_2d_t &solutions);

    static void run_with_stepping(function_quintuple_t<fp_type> const &fun_quintuple,
                                  boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &space_range,
                                  range<fp_type> const &time_range, std::size_t const &last_time_idx,
                                  std::pair<fp_type, fp_type> const &steps, traverse_direction_enum const &traverse_dir,
                                  container_t &prev_solution_0, container_t &prev_solution_1,
                                  container_t &next_solution,
                                  std::function<fp_type(fp_type, fp_type)> const &wave_source, container_t &source,
                                  container_2d_t &solutions);
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void wave_euler_svc_time_loop<fp_type, container, allocator>::run(
    function_quintuple_t<fp_type> const &fun_quintuple, boundary_1d_pair<fp_type> const &boundary_pair,
    range<fp_type> const &space_range, range<fp_type> const &time_range, std::size_t const &last_time_idx,
    std::pair<fp_type, fp_type> const &steps, traverse_direction_enum const &traverse_dir, container_t &prev_solution_0,
    container_t &prev_solution_1, container_t &next_solution)
{
    typedef explicit_wave_scheme<fp_type, container, allocator> explicit_scheme;

    const std::size_t sol_size = next_solution.size();
    // ranges and steps:
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type start_x = space_range.lower();
    const fp_type k = std::get<0>(steps);
    const fp_type h = std::get<1>(steps);
    fp_type time{};
    std::size_t time_idx{};

    if (traverse_dir == traverse_direction_enum::Forward)
    {
        // solve for initial time step:
        auto init_scheme = explicit_scheme::get_initial(true);
        init_scheme(fun_quintuple, steps, prev_solution_0, prev_solution_1, container_t(), boundary_pair, start_time,
                    next_solution);
        time = start_time + k;
        time_idx = 1;
        prev_solution_1 = next_solution;

        // solve for rest of time steps:
        auto scheme = explicit_scheme::get(true);
        time_idx++;
        while (time_idx <= last_time_idx)
        {
            scheme(fun_quintuple, steps, prev_solution_0, prev_solution_1, container_t(), boundary_pair, time,
                   next_solution);
            time += k;
            prev_solution_0 = prev_solution_1;
            prev_solution_1 = next_solution;
            time_idx++;
        }
    }
    else if (traverse_dir == traverse_direction_enum::Backward)
    {
        time_idx = last_time_idx;
        // solve for initial time step:
        auto term_scheme = explicit_scheme::get_terminal(true);
        term_scheme(fun_quintuple, steps, prev_solution_0, prev_solution_1, container_t(), boundary_pair, end_time,
                    next_solution);
        time_idx--;
        time = end_time - time;
        prev_solution_1 = next_solution;

        // solve for rest of time steps:
        auto scheme = explicit_scheme::get(true);
        do
        {
            time_idx--;
            scheme(fun_quintuple, steps, prev_solution_0, prev_solution_1, container_t(), boundary_pair, time,
                   next_solution);
            time -= k;
            prev_solution_0 = prev_solution_1;
            prev_solution_1 = next_solution;
        } while (time_idx > 0);
    }
    else
    {
        throw std::exception("Unreachable");
    }
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void wave_euler_svc_time_loop<fp_type, container, allocator>::run(
    function_quintuple_t<fp_type> const &fun_quintuple, boundary_1d_pair<fp_type> const &boundary_pair,
    range<fp_type> const &space_range, range<fp_type> const &time_range, std::size_t const &last_time_idx,
    std::pair<fp_type, fp_type> const &steps, traverse_direction_enum const &traverse_dir, container_t &prev_solution_0,
    container_t &prev_solution_1, container_t &next_solution,
    std::function<fp_type(fp_type, fp_type)> const &wave_source, container_t &source)
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef explicit_wave_scheme<fp_type, container, allocator> explicit_scheme;

    const std::size_t sol_size = next_solution.size();
    // ranges and steps:
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type start_x = space_range.lower();
    const fp_type k = std::get<0>(steps);
    const fp_type h = std::get<1>(steps);
    fp_type time{};
    std::size_t time_idx{};

    if (traverse_dir == traverse_direction_enum::Forward)
    {
        // solve for initial time step:
        auto init_scheme = explicit_scheme::get_initial(false);
        d_1d::of_function(start_x, h, start_time, wave_source, source);
        init_scheme(fun_quintuple, steps, prev_solution_0, prev_solution_1, source, boundary_pair, start_time,
                    next_solution);
        time = start_time + k;
        time_idx = 1;
        prev_solution_1 = next_solution;

        // solve for rest of time steps:
        auto scheme = explicit_scheme::get(false);
        d_1d::of_function(start_x, h, time, wave_source, source);
        time_idx++;
        while (time_idx <= last_time_idx)
        {
            scheme(fun_quintuple, steps, prev_solution_0, prev_solution_1, source, boundary_pair, time, next_solution);
            time += k;
            prev_solution_0 = prev_solution_1;
            prev_solution_1 = next_solution;
            d_1d::of_function(start_x, h, time, wave_source, source);
            time_idx++;
        }
    }
    else if (traverse_dir == traverse_direction_enum::Backward)
    {
        time_idx = last_time_idx;
        // solve for initial time step:
        auto term_scheme = explicit_scheme::get_terminal(false);
        d_1d::of_function(start_x, h, end_time, wave_source, source);
        term_scheme(fun_quintuple, steps, prev_solution_0, prev_solution_1, source, boundary_pair, end_time,
                    next_solution);
        time_idx--;
        time = end_time - time;
        prev_solution_1 = next_solution;

        // solve for rest of time steps:
        auto scheme = explicit_scheme::get(false);
        d_1d::of_function(start_x, h, time, wave_source, source);
        do
        {
            time_idx--;
            scheme(fun_quintuple, steps, prev_solution_0, prev_solution_1, source, boundary_pair, time, next_solution);
            time -= k;
            prev_solution_0 = prev_solution_1;
            prev_solution_1 = next_solution;
            d_1d::of_function(start_x, h, time, wave_source, source);
        } while (time_idx > 0);
    }
    else
    {
        throw std::exception("Unreachable");
    }
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void wave_euler_svc_time_loop<fp_type, container, allocator>::run_with_stepping(
    function_quintuple_t<fp_type> const &fun_quintuple, boundary_1d_pair<fp_type> const &boundary_pair,
    range<fp_type> const &space_range, range<fp_type> const &time_range, std::size_t const &last_time_idx,
    std::pair<fp_type, fp_type> const &steps, traverse_direction_enum const &traverse_dir, container_t &prev_solution_0,
    container_t &prev_solution_1, container_t &next_solution, container_2d_t &solutions)
{

    typedef explicit_wave_scheme<fp_type, container, allocator> explicit_scheme;

    const std::size_t sol_size = next_solution.size();
    // ranges and steps:
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type start_x = space_range.lower();
    const fp_type k = std::get<0>(steps);
    const fp_type h = std::get<1>(steps);
    fp_type time{};
    std::size_t time_idx{};

    if (traverse_dir == traverse_direction_enum::Forward)
    {
        // store the initial solution:
        solutions(0, prev_solution_0);
        // solve for initial time step:
        auto init_scheme = explicit_scheme::get_initial(true);
        init_scheme(fun_quintuple, steps, prev_solution_0, prev_solution_1, container_t(), boundary_pair, start_time,
                    next_solution);
        time = start_time + k;
        time_idx = 1;
        prev_solution_1 = next_solution;
        solutions(time_idx, next_solution);

        // solve for rest of time steps:
        auto scheme = explicit_scheme::get(true);
        time_idx++;
        while (time_idx <= last_time_idx)
        {
            scheme(fun_quintuple, steps, prev_solution_0, prev_solution_1, container_t(), boundary_pair, time,
                   next_solution);
            time += k;
            prev_solution_0 = prev_solution_1;
            prev_solution_1 = next_solution;
            solutions(time_idx, next_solution);
            time_idx++;
        }
    }
    else if (traverse_dir == traverse_direction_enum::Backward)
    {
        time_idx = last_time_idx;
        // store the terminal solution:
        solutions(last_time_idx, prev_solution_0);
        // solve for terminal time step:
        auto term_scheme = explicit_scheme::get_terminal(true);
        term_scheme(fun_quintuple, steps, prev_solution_0, prev_solution_1, container_t(), boundary_pair, end_time,
                    next_solution);
        time_idx--;
        time = end_time - time;
        prev_solution_1 = next_solution;
        solutions(time_idx, next_solution);

        // solve for rest of time steps:
        auto scheme = explicit_scheme::get(true);
        do
        {
            time_idx--;
            scheme(fun_quintuple, steps, prev_solution_0, prev_solution_1, container_t(), boundary_pair, time,
                   next_solution);
            time -= k;
            prev_solution_0 = prev_solution_1;
            prev_solution_1 = next_solution;
            solutions(time_idx, next_solution);
        } while (time_idx > 0);
    }
    else
    {
        throw std::exception("Unreachable");
    }
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void wave_euler_svc_time_loop<fp_type, container, allocator>::run_with_stepping(
    function_quintuple_t<fp_type> const &fun_quintuple, boundary_1d_pair<fp_type> const &boundary_pair,
    range<fp_type> const &space_range, range<fp_type> const &time_range, std::size_t const &last_time_idx,
    std::pair<fp_type, fp_type> const &steps, traverse_direction_enum const &traverse_dir, container_t &prev_solution_0,
    container_t &prev_solution_1, container_t &next_solution,
    std::function<fp_type(fp_type, fp_type)> const &wave_source, container_t &source, container_2d_t &solutions)
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef explicit_wave_scheme<fp_type, container, allocator> explicit_scheme;

    const std::size_t sol_size = next_solution.size();
    // ranges and steps:
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type start_x = space_range.lower();
    const fp_type k = std::get<0>(steps);
    const fp_type h = std::get<1>(steps);
    fp_type time{};
    std::size_t time_idx{};

    if (traverse_dir == traverse_direction_enum::Forward)
    {
        // store the initial solution:
        solutions(0, prev_solution_0);
        // solve for initial time step:
        auto init_scheme = explicit_scheme::get_initial(false);
        d_1d::of_function(start_x, h, start_time, wave_source, source);
        init_scheme(fun_quintuple, steps, prev_solution_0, prev_solution_1, source, boundary_pair, start_time,
                    next_solution);
        time = start_time + k;
        time_idx = 1;
        prev_solution_1 = next_solution;
        solutions(time_idx, next_solution);

        // solve for rest of time steps:
        auto scheme = explicit_scheme::get(false);
        d_1d::of_function(start_x, h, time, wave_source, source);
        time_idx++;
        while (time_idx <= last_time_idx)
        {
            scheme(fun_quintuple, steps, prev_solution_0, prev_solution_1, source, boundary_pair, time, next_solution);
            time += k;
            prev_solution_0 = prev_solution_1;
            prev_solution_1 = next_solution;
            solutions(time_idx, next_solution);
            d_1d::of_function(start_x, h, time, wave_source, source);
            time_idx++;
        }
    }
    else if (traverse_dir == traverse_direction_enum::Backward)
    {
        time_idx = last_time_idx;
        // store the terminal solution:
        solutions(last_time_idx, prev_solution_0);
        // solve for terminal time step:
        auto term_scheme = explicit_scheme::get_terminal(false);
        d_1d::of_function(start_x, h, end_time, wave_source, source);
        term_scheme(fun_quintuple, steps, prev_solution_0, prev_solution_1, source, boundary_pair, end_time,
                    next_solution);
        time_idx--;
        time = end_time - time;
        prev_solution_1 = next_solution;
        solutions(time_idx, next_solution);

        // solve for rest of time steps:
        auto scheme = explicit_scheme::get(false);
        d_1d::of_function(start_x, h, time, wave_source, source);
        do
        {
            time_idx--;
            scheme(fun_quintuple, steps, prev_solution_0, prev_solution_1, source, boundary_pair, time, next_solution);
            time -= k;
            prev_solution_0 = prev_solution_1;
            prev_solution_1 = next_solution;
            solutions(time_idx, next_solution);
            d_1d::of_function(start_x, h, time, wave_source, source);
        } while (time_idx > 0);
    }
    else
    {
        throw std::exception("Unreachable");
    }
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class wave_euler_svc_scheme
{
    typedef wave_euler_svc_time_loop<fp_type, container, allocator> loop;
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container<fp_type, allocator> container_t;

  private:
    function_quintuple_t<fp_type> fun_quintuple_;
    boundary_1d_pair<fp_type> boundary_pair_;
    pde_discretization_config_1d_ptr<fp_type> discretization_cfg_;

    bool is_stable()
    {
        auto const &b = std::get<4>(fun_quintuple_);
        const fp_type k = discretization_cfg_->time_step();
        const fp_type h = discretization_cfg_->space_step();
        const fp_type ratio = h / k;
        const std::size_t space_size = discretization_cfg_->number_of_space_points();
        fp_type m{};
        for (std::size_t i = 0; i < space_size; ++i)
        {
            m = static_cast<fp_type>(i);
            if (b(m * h) >= ratio)
                return false;
        }
        return true;
    }

    void initialize()
    {
        LSS_ASSERT(is_stable() == true, "The chosen scheme is not stable");
    }

    explicit wave_euler_svc_scheme() = delete;

  public:
    wave_euler_svc_scheme(function_quintuple_t<fp_type> const &fun_quintuple,
                          boundary_1d_pair<fp_type> const &boundary_pair,
                          pde_discretization_config_1d_ptr<fp_type> const &discretization_config)
        : fun_quintuple_{fun_quintuple}, boundary_pair_{boundary_pair}, discretization_cfg_{discretization_config}
    {
        initialize();
    }

    ~wave_euler_svc_scheme()
    {
    }

    void operator()(container_t &prev_solution_0, container_t &prev_solution_1, container_t &next_solution,
                    bool is_wave_sourse_set, std::function<fp_type(fp_type, fp_type)> const &wave_source,
                    traverse_direction_enum traverse_dir)
    {
        const range<fp_type> spacer = discretization_cfg_->space_range();
        const range<fp_type> timer = discretization_cfg_->time_range();
        const fp_type k = discretization_cfg_->time_step();
        const fp_type h = discretization_cfg_->space_step();
        // save solution size:
        const std::size_t sol_size = next_solution.size();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        auto const &steps = std::make_pair(k, h);
        if (is_wave_sourse_set)
        {
            container_t source(sol_size, NaN<fp_type>());
            loop::run(fun_quintuple_, boundary_pair_, spacer, timer, last_time_idx, steps, traverse_dir,
                      prev_solution_0, prev_solution_1, next_solution, wave_source, source);
        }
        else
        {
            loop::run(fun_quintuple_, boundary_pair_, spacer, timer, last_time_idx, steps, traverse_dir,
                      prev_solution_0, prev_solution_1, next_solution);
        }
    }

    void operator()(container_t &prev_solution_0, container_t &prev_solution_1, container_t &next_solution,
                    bool is_wave_sourse_set, std::function<fp_type(fp_type, fp_type)> const &wave_source,
                    traverse_direction_enum traverse_dir, container_2d<fp_type, container, allocator> &solutions)
    {

        const range<fp_type> spacer = discretization_cfg_->space_range();
        const range<fp_type> timer = discretization_cfg_->time_range();
        const fp_type k = discretization_cfg_->time_step();
        const fp_type h = discretization_cfg_->space_step();
        // save solution size:
        const std::size_t sol_size = next_solution.size();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        auto const &steps = std::make_pair(k, h);
        if (is_wave_sourse_set)
        {
            // create a container to carry discretized source heat
            container_t source(sol_size, NaN<fp_type>());
            loop::run_with_stepping(fun_quintuple_, boundary_pair_, spacer, timer, last_time_idx, steps, traverse_dir,
                                    prev_solution_0, prev_solution_1, next_solution, wave_source, source, solutions);
        }
        else
        {
            loop::run_with_stepping(fun_quintuple_, boundary_pair_, spacer, timer, last_time_idx, steps, traverse_dir,
                                    prev_solution_0, prev_solution_1, next_solution, solutions);
        }
    }
};

} // namespace one_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_WAVE_EULER_SVC_SCHEME_HPP_
