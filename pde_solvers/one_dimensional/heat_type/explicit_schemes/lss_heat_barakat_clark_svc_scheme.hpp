#if !defined(_LSS_HEAT_BARAKAT_CLARK_SVC_SCHEME_HPP_)
#define _LSS_HEAT_BARAKAT_CLARK_SVC_SCHEME_HPP_

#include <thread>

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
using lss_enumerations::by_enum;
using lss_enumerations::traverse_direction_enum;
using lss_utility::function_quad_t;
using lss_utility::function_triplet_t;
using lss_utility::NaN;
using lss_utility::range;

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class heat_barakat_clark_svc_time_loop
{
    typedef container<fp_type, allocator> container_t;
    typedef container_2d<by_enum::Row, fp_type, container, allocator> container_2d_t;

  public:
    static void run(function_quad_t<fp_type> const &func_quad, boundary_1d_pair<fp_type> const &boundary_pair,
                    range<fp_type> const &space_range, range<fp_type> const &time_range,
                    std::size_t const &last_time_idx, std::pair<fp_type, fp_type> const &steps,
                    traverse_direction_enum const &traverse_dir, container_t &solution);

    static void run(function_quad_t<fp_type> const &func_quad, boundary_1d_pair<fp_type> const &boundary_pair,
                    range<fp_type> const &space_range, range<fp_type> const &time_range,
                    std::size_t const &last_time_idx, std::pair<fp_type, fp_type> const &steps,
                    traverse_direction_enum const &traverse_dir, container_t &solution,
                    std::function<fp_type(fp_type, fp_type)> const &heat_source, container_t &curr_source,
                    container_t &next_source);

    static void run_with_stepping(function_quad_t<fp_type> const &func_quad,
                                  boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &space_range,
                                  range<fp_type> const &time_range, std::size_t const &last_time_idx,
                                  std::pair<fp_type, fp_type> const &steps, traverse_direction_enum const &traverse_dir,
                                  container_t &solution, container_2d_t &solutions);

    static void run_with_stepping(function_quad_t<fp_type> const &func_quad,
                                  boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &space_range,
                                  range<fp_type> const &time_range, std::size_t const &last_time_idx,
                                  std::pair<fp_type, fp_type> const &steps, traverse_direction_enum const &traverse_dir,
                                  container_t &solution, container_2d_t &solutions,
                                  std::function<fp_type(fp_type, fp_type)> const &heat_source, container_t &curr_source,
                                  container_t &next_source);
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void heat_barakat_clark_svc_time_loop<fp_type, container, allocator>::run(
    function_quad_t<fp_type> const &func_quad, boundary_1d_pair<fp_type> const &boundary_pair,
    range<fp_type> const &space_range, range<fp_type> const &time_range, std::size_t const &last_time_idx,
    std::pair<fp_type, fp_type> const &steps, traverse_direction_enum const &traverse_dir, container_t &solution)
{

    const fp_type zero = static_cast<fp_type>(0.0);
    const fp_type half = static_cast<fp_type>(0.5);
    const std::size_t sol_size = solution.size();
    // ranges and steps:
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type start_x = space_range.lower();
    const fp_type k = std::get<0>(steps);
    const fp_type h = std::get<1>(steps);
    // conmponents of the solution:
    container_t cont_1(solution);
    container_t cont_2(solution);
    // dummy container for source:
    container_t source_dummy(sol_size, zero);
    // get Dirichlet BC:
    auto const &first_bnd = std::dynamic_pointer_cast<dirichlet_boundary_1d<fp_type>>(boundary_pair.first);
    auto const &second_bnd = std::dynamic_pointer_cast<dirichlet_boundary_1d<fp_type>>(boundary_pair.second);
    // get function for sweeps:
    auto const &a = std::get<0>(func_quad);
    auto const &b = std::get<1>(func_quad);
    auto const &d = std::get<2>(func_quad);
    auto const &K = std::get<3>(func_quad);
    // create upsweep anonymous function:
    auto up_sweep = [=](container_t &up_component, container_t const &rhs, fp_type rhs_coeff) {
        fp_type m{};
        for (std::size_t t = 1; t < sol_size - 1; ++t)
        {
            m = static_cast<fp_type>(t);
            up_component[t] = b(m * h) * up_component[t] + d(m * h) * up_component[t + 1] +
                              a(m * h) * up_component[t - 1] + K(m * h) * rhs_coeff * rhs[t];
        }
    };
    // create downsweep anonymous function:
    auto down_sweep = [=](container_t &down_component, container_t const &rhs, fp_type rhs_coeff) {
        fp_type m{};
        for (std::size_t t = sol_size - 2; t >= 1; --t)
        {
            m = static_cast<fp_type>(t);
            down_component[t] = b(m * h) * down_component[t] + d(m * h) * down_component[t + 1] +
                                a(m * h) * down_component[t - 1] + K(m * h) * rhs_coeff * rhs[t];
        }
    };

    fp_type time{};
    std::size_t time_idx{};
    if (traverse_dir == traverse_direction_enum::Forward)
    {
        time = start_time + k;
        time_idx = 1;
        while (time_idx <= last_time_idx)
        {
            std::thread up_sweep_tr(std::move(up_sweep), std::ref(cont_1), source_dummy, zero);
            std::thread down_sweep_tr(std::move(down_sweep), std::ref(cont_2), source_dummy, zero);
            up_sweep_tr.join();
            down_sweep_tr.join();
            cont_1[0] = cont_2[0] = first_bnd->value(time);
            cont_1[sol_size - 1] = cont_2[sol_size - 1] = second_bnd->value(time);
            for (std::size_t t = 0; t < sol_size; ++t)
            {
                solution[t] = half * (cont_1[t] + cont_2[t]);
            }
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
            std::thread up_sweep_tr(std::move(up_sweep), std::ref(cont_1), source_dummy, zero);
            std::thread down_sweep_tr(std::move(down_sweep), std::ref(cont_2), source_dummy, zero);
            up_sweep_tr.join();
            down_sweep_tr.join();
            cont_1[0] = cont_2[0] = first_bnd->value(time);
            cont_1[sol_size - 1] = cont_2[sol_size - 1] = second_bnd->value(time);
            for (std::size_t t = 0; t < sol_size; ++t)
            {
                solution[t] = half * (cont_1[t] + cont_2[t]);
            }
            time -= k;
        } while (time_idx > 0);
    }
    else
    {
        throw std::exception("Unreachable");
    }
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void heat_barakat_clark_svc_time_loop<fp_type, container, allocator>::run(
    function_quad_t<fp_type> const &func_quad, boundary_1d_pair<fp_type> const &boundary_pair,
    range<fp_type> const &space_range, range<fp_type> const &time_range, std::size_t const &last_time_idx,
    std::pair<fp_type, fp_type> const &steps, traverse_direction_enum const &traverse_dir, container_t &solution,
    std::function<fp_type(fp_type, fp_type)> const &heat_source, container_t &curr_source, container_t &next_source)
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;

    const fp_type one = static_cast<fp_type>(1.0);
    const fp_type half = static_cast<fp_type>(0.5);
    const std::size_t sol_size = solution.size();
    // ranges and steps:
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type start_x = space_range.lower();
    const fp_type k = std::get<0>(steps);
    const fp_type h = std::get<1>(steps);
    // conmponents of the solution:
    container_t cont_1(solution);
    container_t cont_2(solution);
    // get Dirichlet BC:
    auto const &first_bnd = std::dynamic_pointer_cast<dirichlet_boundary_1d<fp_type>>(boundary_pair.first);
    auto const &second_bnd = std::dynamic_pointer_cast<dirichlet_boundary_1d<fp_type>>(boundary_pair.second);
    // get function for sweeps:
    auto const &a = std::get<0>(func_quad);
    auto const &b = std::get<1>(func_quad);
    auto const &d = std::get<2>(func_quad);
    auto const &K = std::get<3>(func_quad);
    // create upsweep anonymous function:
    auto up_sweep = [=](container_t &up_component, container_t const &rhs, fp_type rhs_coeff) {
        fp_type m{};
        for (std::size_t t = 1; t < sol_size - 1; ++t)
        {
            m = static_cast<fp_type>(t);
            up_component[t] = b(m * h) * up_component[t] + d(m * h) * up_component[t + 1] +
                              a(m * h) * up_component[t - 1] + K(m * h) * rhs_coeff * rhs[t];
        }
    };
    // create downsweep anonymous function:
    auto down_sweep = [=](container_t &down_component, container_t const &rhs, fp_type rhs_coeff) {
        fp_type m{};
        for (std::size_t t = sol_size - 2; t >= 1; --t)
        {
            m = static_cast<fp_type>(t);
            down_component[t] = b(m * h) * down_component[t] + d(m * h) * down_component[t + 1] +
                                a(m * h) * down_component[t - 1] + K(m * h) * rhs_coeff * rhs[t];
        }
    };
    fp_type time{};
    std::size_t time_idx{};
    if (traverse_dir == traverse_direction_enum::Forward)
    {
        d_1d::of_function(start_x, h, start_time, heat_source, curr_source);
        d_1d::of_function(start_x, h, start_time + k, heat_source, next_source);
        time = start_time + k;
        time_idx = 1;
        while (time_idx <= last_time_idx)
        {
            std::thread up_sweep_tr(std::move(up_sweep), std::ref(cont_1), next_source, one);
            std::thread down_sweep_tr(std::move(down_sweep), std::ref(cont_2), curr_source, one);
            up_sweep_tr.join();
            down_sweep_tr.join();
            cont_1[0] = cont_2[0] = first_bnd->value(time);
            cont_1[sol_size - 1] = cont_2[sol_size - 1] = second_bnd->value(time);
            for (std::size_t t = 0; t < sol_size; ++t)
            {
                solution[t] = half * (cont_1[t] + cont_2[t]);
            }
            d_1d::of_function(start_x, h, time, heat_source, curr_source);
            d_1d::of_function(start_x, h, time + k, heat_source, next_source);
            time += k;
            time_idx++;
        }
    }
    else if (traverse_dir == traverse_direction_enum::Backward)
    {
        d_1d::of_function(start_x, h, end_time, heat_source, curr_source);
        d_1d::of_function(start_x, h, end_time - k, heat_source, next_source);
        time = end_time - k;
        time_idx = last_time_idx;
        do
        {
            time_idx--;
            std::thread up_sweep_tr(std::move(up_sweep), std::ref(cont_1), next_source, one);
            std::thread down_sweep_tr(std::move(down_sweep), std::ref(cont_2), curr_source, one);
            up_sweep_tr.join();
            down_sweep_tr.join();
            cont_1[0] = cont_2[0] = first_bnd->value(time);
            cont_1[sol_size - 1] = cont_2[sol_size - 1] = second_bnd->value(time);
            for (std::size_t t = 0; t < sol_size; ++t)
            {
                solution[t] = half * (cont_1[t] + cont_2[t]);
            }
            d_1d::of_function(start_x, h, time, heat_source, curr_source);
            d_1d::of_function(start_x, h, time - k, heat_source, next_source);
            time -= k;
        } while (time_idx > 0);
    }
    else
    {
        throw std::exception("Unreachable");
    }
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void heat_barakat_clark_svc_time_loop<fp_type, container, allocator>::run_with_stepping(
    function_quad_t<fp_type> const &func_quad, boundary_1d_pair<fp_type> const &boundary_pair,
    range<fp_type> const &space_range, range<fp_type> const &time_range, std::size_t const &last_time_idx,
    std::pair<fp_type, fp_type> const &steps, traverse_direction_enum const &traverse_dir, container_t &solution,
    container_2d_t &solutions)
{
    const fp_type zero = static_cast<fp_type>(0.0);
    const fp_type half = static_cast<fp_type>(0.5);
    const std::size_t sol_size = solution.size();
    // ranges and steps:
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type start_x = space_range.lower();
    const fp_type k = std::get<0>(steps);
    const fp_type h = std::get<1>(steps);
    // conmponents of the solution:
    container_t cont_1(solution);
    container_t cont_2(solution);
    // dummy container for source:
    container_t source_dummy(sol_size, NaN<fp_type>());
    // get Dirichlet BC:
    auto const &first_bnd = std::dynamic_pointer_cast<dirichlet_boundary_1d<fp_type>>(boundary_pair.first);
    auto const &second_bnd = std::dynamic_pointer_cast<dirichlet_boundary_1d<fp_type>>(boundary_pair.second);
    // get function for sweeps:
    auto const &a = std::get<0>(func_quad);
    auto const &b = std::get<1>(func_quad);
    auto const &d = std::get<2>(func_quad);
    auto const &K = std::get<3>(func_quad);
    // create upsweep anonymous function:
    auto up_sweep = [=](container_t &up_component, container_t const &rhs, fp_type rhs_coeff) {
        fp_type m{};
        for (std::size_t t = 1; t < sol_size - 1; ++t)
        {
            m = static_cast<fp_type>(t);
            up_component[t] = b(m * h) * up_component[t] + d(m * h) * up_component[t + 1] +
                              a(m * h) * up_component[t - 1] + K(m * h) * rhs_coeff * rhs[t];
        }
    };
    // create downsweep anonymous function:
    auto down_sweep = [=](container_t &down_component, container_t const &rhs, fp_type rhs_coeff) {
        fp_type m{};
        for (std::size_t t = sol_size - 2; t >= 1; --t)
        {
            m = static_cast<fp_type>(t);
            down_component[t] = b(m * h) * down_component[t] + d(m * h) * down_component[t + 1] +
                                a(m * h) * down_component[t - 1] + K(m * h) * rhs_coeff * rhs[t];
        }
    };
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
            std::thread up_sweep_tr(std::move(up_sweep), std::ref(cont_1), source_dummy, zero);
            std::thread down_sweep_tr(std::move(down_sweep), std::ref(cont_2), source_dummy, zero);
            up_sweep_tr.join();
            down_sweep_tr.join();
            cont_1[0] = cont_2[0] = first_bnd->value(time);
            cont_1[sol_size - 1] = cont_2[sol_size - 1] = second_bnd->value(time);
            for (std::size_t t = 0; t < sol_size; ++t)
            {
                solution[t] = half * (cont_1[t] + cont_2[t]);
            }
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
            std::thread up_sweep_tr(std::move(up_sweep), std::ref(cont_1), source_dummy, zero);
            std::thread down_sweep_tr(std::move(down_sweep), std::ref(cont_2), source_dummy, zero);
            up_sweep_tr.join();
            down_sweep_tr.join();
            cont_1[0] = cont_2[0] = first_bnd->value(time);
            cont_1[sol_size - 1] = cont_2[sol_size - 1] = second_bnd->value(time);
            for (std::size_t t = 0; t < sol_size; ++t)
            {
                solution[t] = half * (cont_1[t] + cont_2[t]);
            }
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
void heat_barakat_clark_svc_time_loop<fp_type, container, allocator>::run_with_stepping(
    function_quad_t<fp_type> const &func_quad, boundary_1d_pair<fp_type> const &boundary_pair,
    range<fp_type> const &space_range, range<fp_type> const &time_range, std::size_t const &last_time_idx,
    std::pair<fp_type, fp_type> const &steps, traverse_direction_enum const &traverse_dir, container_t &solution,
    container_2d_t &solutions, std::function<fp_type(fp_type, fp_type)> const &heat_source, container_t &curr_source,
    container_t &next_source)
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;

    const fp_type one = static_cast<fp_type>(1.0);
    const fp_type half = static_cast<fp_type>(0.5);
    const std::size_t sol_size = solution.size();
    // ranges and steps:
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type start_x = space_range.lower();
    const fp_type k = std::get<0>(steps);
    const fp_type h = std::get<1>(steps);
    // conmponents of the solution:
    container_t cont_1(solution);
    container_t cont_2(solution);
    // get Dirichlet BC:
    auto const &first_bnd = std::dynamic_pointer_cast<dirichlet_boundary_1d<fp_type>>(boundary_pair.first);
    auto const &second_bnd = std::dynamic_pointer_cast<dirichlet_boundary_1d<fp_type>>(boundary_pair.second);
    // get function for sweeps:
    auto const &a = std::get<0>(func_quad);
    auto const &b = std::get<1>(func_quad);
    auto const &d = std::get<2>(func_quad);
    auto const &K = std::get<3>(func_quad);
    // create upsweep anonymous function:
    auto up_sweep = [=](container_t &up_component, container_t const &rhs, fp_type rhs_coeff) {
        fp_type m{};
        for (std::size_t t = 1; t < sol_size - 1; ++t)
        {
            m = static_cast<fp_type>(t);
            up_component[t] = b(m * h) * up_component[t] + d(m * h) * up_component[t + 1] +
                              a(m * h) * up_component[t - 1] + K(m * h) * rhs_coeff * rhs[t];
        }
    };
    // create downsweep anonymous function:
    auto down_sweep = [=](container_t &down_component, container_t const &rhs, fp_type rhs_coeff) {
        fp_type m{};
        for (std::size_t t = sol_size - 2; t >= 1; --t)
        {
            m = static_cast<fp_type>(t);
            down_component[t] = b(m * h) * down_component[t] + d(m * h) * down_component[t + 1] +
                                a(m * h) * down_component[t - 1] + K(m * h) * rhs_coeff * rhs[t];
        }
    };
    fp_type time{};
    std::size_t time_idx{};
    if (traverse_dir == traverse_direction_enum::Forward)
    {
        // store the initial solution:
        solutions(0, solution);
        d_1d::of_function(start_x, h, start_time, heat_source, curr_source);
        d_1d::of_function(start_x, h, start_time + k, heat_source, next_source);
        time = start_time + k;
        time_idx = 1;
        while (time_idx <= last_time_idx)
        {
            std::thread up_sweep_tr(std::move(up_sweep), std::ref(cont_1), next_source, one);
            std::thread down_sweep_tr(std::move(down_sweep), std::ref(cont_2), curr_source, one);
            up_sweep_tr.join();
            down_sweep_tr.join();
            cont_1[0] = cont_2[0] = first_bnd->value(time);
            cont_1[sol_size - 1] = cont_2[sol_size - 1] = second_bnd->value(time);
            for (std::size_t t = 0; t < sol_size; ++t)
            {
                solution[t] = half * (cont_1[t] + cont_2[t]);
            }
            d_1d::of_function(start_x, h, time, heat_source, curr_source);
            d_1d::of_function(start_x, h, time + k, heat_source, next_source);
            solutions(time_idx, solution);
            time += k;
            time_idx++;
        }
    }
    else if (traverse_dir == traverse_direction_enum::Backward)
    {
        // store the initial solution:
        solutions(last_time_idx, solution);
        d_1d::of_function(start_x, h, end_time, heat_source, curr_source);
        d_1d::of_function(start_x, h, end_time - k, heat_source, next_source);
        time = end_time - k;
        time_idx = last_time_idx;
        do
        {
            time_idx--;
            std::thread up_sweep_tr(std::move(up_sweep), std::ref(cont_1), next_source, one);
            std::thread down_sweep_tr(std::move(down_sweep), std::ref(cont_2), curr_source, one);
            up_sweep_tr.join();
            down_sweep_tr.join();
            cont_1[0] = cont_2[0] = first_bnd->value(time);
            cont_1[sol_size - 1] = cont_2[sol_size - 1] = second_bnd->value(time);
            for (std::size_t t = 0; t < sol_size; ++t)
            {
                solution[t] = half * (cont_1[t] + cont_2[t]);
            }
            d_1d::of_function(start_x, h, time, heat_source, curr_source);
            d_1d::of_function(start_x, h, time - k, heat_source, next_source);
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
class heat_barakat_clark_svc_scheme
{
    typedef heat_barakat_clark_svc_time_loop<fp_type, container, allocator> loop;
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container<fp_type, allocator> container_t;

  private:
    function_triplet_t<fp_type> fun_triplet_;
    boundary_1d_pair<fp_type> boundary_pair_;
    pde_discretization_config_1d_ptr<fp_type> discretization_cfg_;

    void initialize()
    {
        auto const &first = boundary_pair_.first;
        if (std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(first))
        {
            throw std::exception("Neumann boundary type is not supported for this scheme");
        }
        if (std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(first))
        {
            throw std::exception("Robin boundary type is not supported for this scheme");
        }
        auto const &second = boundary_pair_.second;
        if (std::dynamic_pointer_cast<neumann_boundary_1d<fp_type>>(second))
        {
            throw std::exception("Neumann boundary type is not supported for this scheme");
        }
        if (std::dynamic_pointer_cast<robin_boundary_1d<fp_type>>(second))
        {
            throw std::exception("Robin boundary type is not supported for this scheme");
        }
    }

    explicit heat_barakat_clark_svc_scheme() = delete;

  public:
    heat_barakat_clark_svc_scheme(function_triplet_t<fp_type> const &fun_triplet,
                                  boundary_1d_pair<fp_type> const &boundary_pair,
                                  pde_discretization_config_1d_ptr<fp_type> const &discretization_config)
        : fun_triplet_{fun_triplet}, boundary_pair_{boundary_pair}, discretization_cfg_{discretization_config}
    {
        initialize();
    }

    ~heat_barakat_clark_svc_scheme()
    {
    }

    void operator()(container_t &solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type)> const &heat_source, traverse_direction_enum traverse_dir)
    {
        const fp_type one = static_cast<fp_type>(1.0);
        const range<fp_type> spacer = discretization_cfg_->space_range();
        const range<fp_type> timer = discretization_cfg_->time_range();
        const fp_type k = discretization_cfg_->time_step();
        const fp_type h = discretization_cfg_->space_step();
        auto const &A = std::get<0>(fun_triplet_);
        auto const &B = std::get<1>(fun_triplet_);
        auto const &D = std::get<2>(fun_triplet_);
        // build the scheme coefficients:
        auto const &a = [&](fp_type x) { return (A(x) / (one + B(x))); };
        auto const &b = [&](fp_type x) { return ((one - B(x)) / (one + B(x))); };
        auto const &d = [&](fp_type x) { return (D(x) / (one + B(x))); };
        auto const &K = [&](fp_type x) { return (k / (one + B(x))); };
        // save solution size:
        const std::size_t sol_size = solution.size();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        // wrap up the functions:
        auto const &fun_quad = std::make_tuple(a, b, d, K);
        // create a container to carry discretized source heat
        container_t source_curr(sol_size, NaN<fp_type>());
        container_t source_next(sol_size, NaN<fp_type>());
        auto const &steps = std::make_pair(k, h);
        if (is_heat_sourse_set)
        {
            loop::run(fun_quad, boundary_pair_, spacer, timer, last_time_idx, steps, traverse_dir, solution,
                      heat_source, source_curr, source_next);
        }
        else
        {
            loop::run(fun_quad, boundary_pair_, spacer, timer, last_time_idx, steps, traverse_dir, solution);
        }
    }

    void operator()(container_t &solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type)> const &heat_source, traverse_direction_enum traverse_dir,
                    container_2d<by_enum::Row, fp_type, container, allocator> &solutions)
    {
        const fp_type one = static_cast<fp_type>(1.0);
        const range<fp_type> spacer = discretization_cfg_->space_range();
        const range<fp_type> timer = discretization_cfg_->time_range();
        const fp_type k = discretization_cfg_->time_step();
        const fp_type h = discretization_cfg_->space_step();
        auto const &A = std::get<0>(fun_triplet_);
        auto const &B = std::get<1>(fun_triplet_);
        auto const &D = std::get<2>(fun_triplet_);
        // build the scheme coefficients:
        auto const &a = [&](fp_type x) { return (A(x) / (one + B(x))); };
        auto const &b = [&](fp_type x) { return ((one - B(x)) / (one + B(x))); };
        auto const &d = [&](fp_type x) { return (D(x) / (one + B(x))); };
        auto const &K = [&](fp_type x) { return (k / (one + B(x))); };
        // save solution size:
        const std::size_t sol_size = solution.size();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        // wrap up the functions:
        auto const &fun_quad = std::make_tuple(a, b, d, K);
        // create a container to carry discretized source heat
        container_t source_curr(sol_size, NaN<fp_type>());
        container_t source_next(sol_size, NaN<fp_type>());

        auto const &steps = std::make_pair(k, h);
        if (is_heat_sourse_set)
        {
            loop::run_with_stepping(fun_quad, boundary_pair_, spacer, timer, last_time_idx, steps, traverse_dir,
                                    solution, solutions, heat_source, source_curr, source_next);
        }
        else
        {
            loop::run_with_stepping(fun_quad, boundary_pair_, spacer, timer, last_time_idx, steps, traverse_dir,
                                    solution, solutions);
        }
    }
};

} // namespace one_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_HEAT_BARAKAT_CLARK_SVC_SCHEME_HPP_
