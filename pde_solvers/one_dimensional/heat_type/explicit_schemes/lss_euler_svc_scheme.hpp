#if !defined(_LSS_EULER_SVC_SCHEME_HPP_)
#define _LSS_EULER_SVC_SCHEME_HPP_

#include "boundaries/lss_boundary.hpp"
#include "boundaries/lss_dirichlet_boundary.hpp"
#include "boundaries/lss_neumann_boundary.hpp"
#include "boundaries/lss_robin_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include "containers/lss_container_2d.hpp"
#include "pde_solvers/lss_discretization.hpp"
#include "pde_solvers/lss_discretization_config.hpp"

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
using lss_utility::function_triplet_t;
using lss_utility::NaN;
using lss_utility::pair_t;
using lss_utility::range;

template <template <typename, typename> typename container, typename fp_type, typename alloc>
using explicit_scheme_function =
    std::function<void(function_triplet_t<fp_type> const &, pair_t<fp_type> const &, container<fp_type, alloc> const &,
                       container<fp_type, alloc> const &, boundary_1d_pair<fp_type> const &, fp_type const &,
                       container<fp_type, alloc> &)>;

template <typename fp_type, template <typename, typename> typename container, typename allocator> class explicit_scheme
{
    typedef container<fp_type, allocator> container_t;
    typedef explicit_scheme_function<container, fp_type, allocator> scheme_function_t;

  public:
    static scheme_function_t const get(bool is_homogeneus)
    {
        const fp_type two = static_cast<fp_type>(2.0);
        auto scheme_fun_h = [=](function_triplet_t<fp_type> const &coefficients,
                                std::pair<fp_type, fp_type> const &steps, container_t const &input,
                                container_t const &inhom_input, boundary_1d_pair<fp_type> const &boundary_pair,
                                fp_type const &time, container_t &solution) {
            auto const &first_bnd = boundary_pair.first;
            auto const &second_bnd = boundary_pair.second;
            auto const &a = std::get<0>(coefficients);
            auto const &b = std::get<1>(coefficients);
            auto const &d = std::get<2>(coefficients);
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
                solution[0] = beta * a(m * h) + b(m * h) * input[0] + (a(m * h) + d(m * h)) * input[1];
            }
            else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_bnd))
            {
                const fp_type beta = two * h * ptr->value(time);
                const fp_type alpha = two * h * ptr->linear_value(time);
                m = static_cast<fp_type>(0);
                solution[0] =
                    (b(m * h) + alpha * a(m * h)) * input[0] + (a(m * h) + d(m * h)) * input[1] + beta * a(m * h);
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
                solution[N] = (a(m * h) + d(m * h)) * input[N - 1] + b(m * h) * input[N] - delta * d(m * h);
            }
            else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
            {
                const fp_type delta = two * h * ptr->value(time);
                const fp_type gamma = two * h * ptr->linear_value(time);
                m = static_cast<fp_type>(N);
                solution[N] =
                    (a(m * h) + d(m * h)) * input[N - 1] + (b(m * h) - gamma * d(m * h)) * input[N] - delta * d(m * h);
            }

            for (std::size_t t = 1; t < N; ++t)
            {
                m = static_cast<fp_type>(t);
                solution[t] = (d(m * h) * input[t + 1]) + (b(m * h) * input[t]) + (a(m * h) * input[t - 1]);
            }
        };
        auto scheme_fun_nh = [=](function_triplet_t<fp_type> const &coefficients,
                                 std::pair<fp_type, fp_type> const &steps, container_t const &input,
                                 container_t const &inhom_input, boundary_1d_pair<fp_type> const &boundary_pair,
                                 fp_type const &time, container_t &solution) {
            auto const &first_bnd = boundary_pair.first;
            auto const &second_bnd = boundary_pair.second;
            auto const &a = std::get<0>(coefficients);
            auto const &b = std::get<1>(coefficients);
            auto const &d = std::get<2>(coefficients);
            auto const k = steps.first;
            auto const h = steps.second;
            fp_type m{};

            // for lower boundaries first:
            if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(first_bnd))
            {
                const fp_type beta = two * h * ptr->value(time);
                m = static_cast<fp_type>(0);
                solution[0] =
                    beta * a(m * h) + b(m * h) * input[0] + (a(m * h) + d(m * h)) * input[1] + k * inhom_input[0];
            }
            else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first_bnd))
            {
                const fp_type beta = two * h * ptr->value(time);
                const fp_type alpha = two * h * ptr->linear_value(time);
                m = static_cast<fp_type>(0);
                solution[0] = (b(m * h) + alpha * a(m * h)) * input[0] + (a(m * h) + d(m * h)) * input[1] +
                              beta * a(m * h) + k * inhom_input[0];
                ;
            }
            // for upper boundaries second:
            const std::size_t N = solution.size() - 1;
            if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(second_bnd))
            {
                const fp_type delta = two * h * ptr->value(time);
                m = static_cast<fp_type>(N);
                solution[N] =
                    (a(m * h) + d(m * h)) * input[N - 1] + b(m * h) * input[N] - delta * d(m * h) + k * inhom_input[N];
                ;
            }
            else if (auto const &ptr = std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second_bnd))
            {
                const fp_type delta = two * h * ptr->value(time);
                const fp_type gamma = two * h * ptr->linear_value(time);
                m = static_cast<fp_type>(N);
                solution[N] = (a(m * h) + d(m * h)) * input[N - 1] + (b(m * h) - gamma * d(m * h)) * input[N] -
                              delta * d(m * h) + k * inhom_input[N];
                ;
            }
            for (std::size_t t = 1; t < N; ++t)
            {
                m = static_cast<fp_type>(t);
                solution[t] = (d(m * h) * input[t + 1]) + (b(m * h) * input[t]) + (a(m * h) * input[t - 1]) +
                              (k * inhom_input[t]);
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
 * euler_svc_time_loop object
 */
template <typename fp_type, template <typename, typename> typename container, typename allocator>
class euler_svc_time_loop
{
    typedef container<fp_type, allocator> container_t;
    typedef container_2d<fp_type, container, allocator> container_2d_t;

  public:
    template <typename scheme_function>
    static void run(function_triplet_t<fp_type> const &func_triplet, scheme_function &scheme_fun,
                    boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &space_range,
                    range<fp_type> const &time_range, std::size_t const &last_time_idx,
                    std::pair<fp_type, fp_type> const &steps, traverse_direction_enum const &traverse_dir,
                    container_t &solution);

    template <typename scheme_function>
    static void run(function_triplet_t<fp_type> const &func_triplet, scheme_function &scheme_fun,
                    boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &space_range,
                    range<fp_type> const &time_range, std::size_t const &last_time_idx,
                    std::pair<fp_type, fp_type> const &steps, traverse_direction_enum const &traverse_dir,
                    container_t &solution, std::function<fp_type(fp_type, fp_type)> const &heat_source,
                    container_t &source);

    template <typename scheme_function>
    static void run_with_stepping(function_triplet_t<fp_type> const &func_triplet, scheme_function &scheme_fun,
                                  boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &space_range,
                                  range<fp_type> const &time_range, std::size_t const &last_time_idx,
                                  std::pair<fp_type, fp_type> const &steps, traverse_direction_enum const &traverse_dir,
                                  container_t &solution, container_2d_t &solutions);

    template <typename scheme_function>
    static void run_with_stepping(function_triplet_t<fp_type> const &func_triplet, scheme_function &scheme_fun,
                                  boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &space_range,
                                  range<fp_type> const &time_range, std::size_t const &last_time_idx,
                                  std::pair<fp_type, fp_type> const &steps, traverse_direction_enum const &traverse_dir,
                                  container_t &solution, container_2d_t &solutions,
                                  std::function<fp_type(fp_type, fp_type)> const &heat_source, container_t &source);
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
template <typename scheme_function>
void euler_svc_time_loop<fp_type, container, allocator>::run(
    function_triplet_t<fp_type> const &func_triplet, scheme_function &scheme_fun,
    boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &space_range, range<fp_type> const &time_range,
    std::size_t const &last_time_idx, std::pair<fp_type, fp_type> const &steps,
    traverse_direction_enum const &traverse_dir, container_t &solution)
{
    const fp_type zero = static_cast<fp_type>(0.0);
    const std::size_t sol_size = solution.size();
    // ranges and steps:
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type start_x = space_range.lower();
    const fp_type k = std::get<0>(steps);
    const fp_type h = std::get<1>(steps);
    // container for next solution:
    container_t next_solution(sol_size, zero);
    // get function for sweeps:
    auto const &a = std::get<0>(func_triplet);
    auto const &b = std::get<1>(func_triplet);
    auto const &d = std::get<2>(func_triplet);

    fp_type time{};
    std::size_t time_idx{};
    if (traverse_dir == traverse_direction_enum::Forward)
    {
        time = start_time + k;
        time_idx = 1;
        while (time_idx <= last_time_idx)
        {
            scheme_fun(func_triplet, steps, solution, container_t(), boundary_pair, time, next_solution);
            solution = next_solution;
            time += k;
            time_idx++;
        }
    }
    else if (traverse_dir == traverse_direction_enum::Backward)
    {
        time = end_time - k;
        time_idx = last_time_idx;
        do
        {
            time_idx--;
            scheme_fun(func_triplet, steps, solution, container_t(), boundary_pair, time, next_solution);
            solution = next_solution;
            time -= k;
        } while (time_idx > 0);
    }
    else
    {
        throw std::exception("Unreachable");
    }
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
template <typename scheme_function>
void euler_svc_time_loop<fp_type, container, allocator>::run(
    function_triplet_t<fp_type> const &func_triplet, scheme_function &scheme_fun,
    boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &space_range, range<fp_type> const &time_range,
    std::size_t const &last_time_idx, std::pair<fp_type, fp_type> const &steps,
    traverse_direction_enum const &traverse_dir, container_t &solution,
    std::function<fp_type(fp_type, fp_type)> const &heat_source, container_t &source)
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;

    const fp_type zero = static_cast<fp_type>(0.0);
    const std::size_t sol_size = solution.size();
    // ranges and steps:
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type start_x = space_range.lower();
    const fp_type k = std::get<0>(steps);
    const fp_type h = std::get<1>(steps);
    // container for next solution:
    container_t next_solution(sol_size, zero);
    // get function for sweeps:
    auto const &a = std::get<0>(func_triplet);
    auto const &b = std::get<1>(func_triplet);
    auto const &d = std::get<2>(func_triplet);

    fp_type time{};
    std::size_t time_idx{};
    if (traverse_dir == traverse_direction_enum::Forward)
    {
        d_1d::of_function(start_x, h, start_time, heat_source, source);
        time = start_time + k;
        time_idx = 1;
        while (time_idx <= last_time_idx)
        {
            scheme_fun(func_triplet, steps, solution, source, boundary_pair, time, next_solution);
            solution = next_solution;
            d_1d::of_function(start_x, h, time, heat_source, source);
            time += k;
            time_idx++;
        }
    }
    else if (traverse_dir == traverse_direction_enum::Backward)
    {
        d_1d::of_function(start_x, h, end_time, heat_source, source);
        time = end_time - k;
        time_idx = last_time_idx;
        do
        {
            time_idx--;
            scheme_fun(func_triplet, steps, solution, source, boundary_pair, time, next_solution);
            solution = next_solution;
            d_1d::of_function(start_x, h, time, heat_source, source);
            time -= k;
        } while (time_idx > 0);
    }
    else
    {
        throw std::exception("Unreachable");
    }
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
template <typename scheme_function>
void euler_svc_time_loop<fp_type, container, allocator>::run_with_stepping(
    function_triplet_t<fp_type> const &func_triplet, scheme_function &scheme_fun,
    boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &space_range, range<fp_type> const &time_range,
    std::size_t const &last_time_idx, std::pair<fp_type, fp_type> const &steps,
    traverse_direction_enum const &traverse_dir, container_t &solution, container_2d_t &solutions)
{
    const fp_type zero = static_cast<fp_type>(0.0);
    const std::size_t sol_size = solution.size();
    // ranges and steps:
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type start_x = space_range.lower();
    const fp_type k = std::get<0>(steps);
    const fp_type h = std::get<1>(steps);
    // container for next solution:
    container_t next_solution(sol_size, zero);
    // get function for sweeps:
    auto const &a = std::get<0>(func_triplet);
    auto const &b = std::get<1>(func_triplet);
    auto const &d = std::get<2>(func_triplet);

    fp_type time{};
    std::size_t time_idx{};
    if (traverse_dir == traverse_direction_enum::Forward)
    {
        // store the initial solution:
        solutions(0, solution);
        time = start_time + k;
        time_idx = 1;
        while (time_idx <= last_time_idx)
        {
            scheme_fun(func_triplet, steps, solution, container_t(), boundary_pair, time, next_solution);
            solution = next_solution;
            solutions(time_idx, solution);
            time += k;
            time_idx++;
        }
    }
    else if (traverse_dir == traverse_direction_enum::Backward)
    {
        // store the initial solution:
        solutions(last_time_idx, solution);
        time = end_time - k;
        time_idx = last_time_idx;
        do
        {
            time_idx--;
            scheme_fun(func_triplet, steps, solution, container_t(), boundary_pair, time, next_solution);
            solution = next_solution;
            solutions(time_idx, solution);
            time -= k;
        } while (time_idx > 0);
    }
    else
    {
        throw std::exception("Unreachable");
    }
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
template <typename scheme_function>
void euler_svc_time_loop<fp_type, container, allocator>::run_with_stepping(
    function_triplet_t<fp_type> const &func_triplet, scheme_function &scheme_fun,
    boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &space_range, range<fp_type> const &time_range,
    std::size_t const &last_time_idx, std::pair<fp_type, fp_type> const &steps,
    traverse_direction_enum const &traverse_dir, container_t &solution, container_2d_t &solutions,
    std::function<fp_type(fp_type, fp_type)> const &heat_source, container_t &source)
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;

    const fp_type zero = static_cast<fp_type>(0.0);
    const std::size_t sol_size = solution.size();
    // ranges and steps:
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type start_x = space_range.lower();
    const fp_type k = std::get<0>(steps);
    const fp_type h = std::get<1>(steps);
    // container for next_solution:
    container_t next_solution(sol_size, zero);
    // get function for sweeps:
    auto const &a = std::get<0>(func_triplet);
    auto const &b = std::get<1>(func_triplet);
    auto const &d = std::get<2>(func_triplet);

    fp_type time{};
    std::size_t time_idx{};
    if (traverse_dir == traverse_direction_enum::Forward)
    {
        // store the initial solution:
        solutions(0, solution);
        d_1d::of_function(start_x, h, start_time, heat_source, source);
        time = start_time + k;
        time_idx = 1;
        while (time_idx <= last_time_idx)
        {
            scheme_fun(func_triplet, steps, solution, source, boundary_pair, time, next_solution);
            solution = next_solution;
            d_1d::of_function(start_x, h, time, heat_source, source);
            solutions(time_idx, solution);
            time += k;
            time_idx++;
        }
    }
    else if (traverse_dir == traverse_direction_enum::Backward)
    {
        // store the initial solution:
        solutions(last_time_idx, solution);
        d_1d::of_function(start_x, h, end_time, heat_source, source);
        time = end_time - k;
        time_idx = last_time_idx;
        do
        {
            time_idx--;
            scheme_fun(func_triplet, steps, solution, source, boundary_pair, time, next_solution);
            solution = next_solution;
            d_1d::of_function(start_x, h, time, heat_source, source);
            solutions(time_idx, solution);
            time -= k;
        } while (time_idx > 0);
    }
    else
    {
        throw std::exception("Unreachable");
    }
}

template <typename fp_type, template <typename, typename> typename container, typename allocator> class euler_svc_scheme
{
    typedef euler_svc_time_loop<fp_type, container, allocator> loop;
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container<fp_type, allocator> container_t;

  private:
    function_triplet_t<fp_type> fun_triplet_;
    boundary_1d_pair<fp_type> boundary_pair_;
    discretization_config_1d_ptr<fp_type> discretization_cfg_;

    bool is_stable()
    {
        const fp_type zero = static_cast<fp_type>(0.0);
        const fp_type half = static_cast<fp_type>(0.5);
        const fp_type one = static_cast<fp_type>(1.0);
        const fp_type two = static_cast<fp_type>(2.0);
        auto const &A = std::get<0>(fun_triplet_);
        auto const &B = std::get<1>(fun_triplet_);
        auto const &D = std::get<2>(fun_triplet_);
        const fp_type k = discretization_cfg_->time_step();
        const fp_type h = discretization_cfg_->space_step();
        const fp_type lambda = k / (h * h);
        const fp_type gamma = k / (two * h);
        const fp_type delta = half * k;
        auto const &a = [=](fp_type x) { return ((A(x) + D(x)) / (two * lambda)); };
        auto const &b = [=](fp_type x) { return ((D(x) - A(x)) / (two * gamma)); };
        auto const &c = [=](fp_type x) { return ((lambda * a(x) - B(x)) / delta); };
        const std::size_t space_size = discretization_cfg_->number_of_space_points();
        fp_type m{};
        for (std::size_t i = 0; i < space_size; ++i)
        {
            m = static_cast<fp_type>(i);
            if (c(m * h) > zero)
                return false;
            if ((two * lambda * a(m * h) - k * c(m * h)) > one)
                return false;
            if (((gamma * std::abs(b(m * h))) * (gamma * std::abs(b(m * h)))) > (two * lambda * a(m * h)))
                return false;
        }
        return true;
    }

    void initialize()
    {
        LSS_ASSERT(is_stable() == true, "The chosen scheme is not stable");
    }

    explicit euler_svc_scheme() = delete;

  public:
    euler_svc_scheme(function_triplet_t<fp_type> const &fun_triplet, boundary_1d_pair<fp_type> const &boundary_pair,
                     discretization_config_1d_ptr<fp_type> const &discretization_config)
        : fun_triplet_{fun_triplet}, boundary_pair_{boundary_pair}, discretization_cfg_{discretization_config}
    {
        initialize();
    }

    ~euler_svc_scheme()
    {
    }

    void operator()(container_t &solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type)> const &heat_source, traverse_direction_enum traverse_dir)
    {
        const fp_type one = static_cast<fp_type>(1.0);
        const fp_type two = static_cast<fp_type>(2.0);
        const range<fp_type> spacer = discretization_cfg_->space_range();
        const range<fp_type> timer = discretization_cfg_->time_range();
        const fp_type k = discretization_cfg_->time_step();
        const fp_type h = discretization_cfg_->space_step();
        auto const &A = std::get<0>(fun_triplet_);
        auto const &B = std::get<1>(fun_triplet_);
        auto const &D = std::get<2>(fun_triplet_);
        // build the scheme coefficients:
        auto const &a = [&](fp_type x) { return A(x); };
        auto const &b = [&](fp_type x) { return (one - two * B(x)); };
        auto const &d = [&](fp_type x) { return D(x); };
        // save solution size:
        const std::size_t sol_size = solution.size();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        // wrap up the functions:
        auto const &fun_trip = std::make_tuple(a, b, d);
        // create a container to carry discretized source heat
        container_t source(sol_size, NaN<fp_type>());
        auto const &steps = std::make_pair(k, h);
        const bool is_homogeneous = !is_heat_sourse_set;
        auto scheme_function = explicit_scheme<fp_type, container, allocator>::get(is_homogeneous);
        if (is_heat_sourse_set)
        {
            loop::run(fun_trip, scheme_function, boundary_pair_, spacer, timer, last_time_idx, steps, traverse_dir,
                      solution, heat_source, source);
        }
        else
        {
            loop::run(fun_trip, scheme_function, boundary_pair_, spacer, timer, last_time_idx, steps, traverse_dir,
                      solution);
        }
    }

    void operator()(container_t &solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type)> const &heat_source, traverse_direction_enum traverse_dir,
                    container_2d<fp_type, container, allocator> &solutions)
    {
        const fp_type one = static_cast<fp_type>(1.0);
        const fp_type two = static_cast<fp_type>(2.0);
        const range<fp_type> spacer = discretization_cfg_->space_range();
        const range<fp_type> timer = discretization_cfg_->time_range();
        const fp_type k = discretization_cfg_->time_step();
        const fp_type h = discretization_cfg_->space_step();
        auto const &A = std::get<0>(fun_triplet_);
        auto const &B = std::get<1>(fun_triplet_);
        auto const &D = std::get<2>(fun_triplet_);
        // build the scheme coefficients:
        auto const &a = [&](fp_type x) { return A(x); };
        auto const &b = [&](fp_type x) { return (one - two * B(x)); };
        auto const &d = [&](fp_type x) { return D(x); };
        // save solution size:
        const std::size_t sol_size = solution.size();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        // wrap up the functions:
        auto const &fun_trip = std::make_tuple(a, b, d);
        // create a container to carry discretized source heat
        container_t source(sol_size, NaN<fp_type>());
        auto const &steps = std::make_pair(k, h);
        const bool is_homogeneous = !is_heat_sourse_set;
        auto scheme_function = explicit_scheme<fp_type, container, allocator>::get(is_homogeneous);
        if (is_heat_sourse_set)
        {
            loop::run_with_stepping(fun_trip, scheme_function, boundary_pair_, spacer, timer, last_time_idx, steps,
                                    traverse_dir, solution, solutions, heat_source, source);
        }
        else
        {
            loop::run_with_stepping(fun_trip, scheme_function, boundary_pair_, spacer, timer, last_time_idx, steps,
                                    traverse_dir, solution, solutions);
        }
    }
};

} // namespace one_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_EULER_SVC_SCHEME_HPP_
