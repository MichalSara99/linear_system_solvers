#if !defined(_LSS_HESTON_EULER_SCHEME_HPP_)
#define _LSS_HESTON_EULER_SCHEME_HPP_

#include "boundaries/lss_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include "containers/lss_container_2d.hpp"
#include "discretization/lss_discretization.hpp"
#include "discretization/lss_grid.hpp"
#include "discretization/lss_grid_config.hpp"
#include "pde_solvers/lss_pde_discretization_config.hpp"
#include "pde_solvers/two_dimensional/heat_type/explicit_coefficients/lss_heston_euler_coefficients.hpp"
#include "pde_solvers/two_dimensional/heat_type/lss_2d_general_heston_equation_explicit_boundary.hpp"
#include "pde_solvers/two_dimensional/heat_type/solver_method/lss_heston_euler_solver_method.hpp"

namespace lss_pde_solvers
{

namespace two_dimensional
{

using lss_boundary::boundary_2d_pair;
using lss_boundary::boundary_2d_ptr;
using lss_enumerations::traverse_direction_enum;

/**
 * heston_euler_time_loop object
 */
template <typename fp_type, template <typename, typename> typename container, typename allocator>
class heston_euler_time_loop
{
    typedef container_2d<by_enum::Row, fp_type, container, allocator> container_2d_t;

  public:
    template <typename solver, typename boundary_solver>
    static void run(solver const &solver_ptr, boundary_solver const &boundary_solver_ptr,
                    boundary_2d_pair<fp_type> const &horizontal_boundary_pair,
                    boundary_2d_ptr<fp_type> const &vertical_upper_boundary_ptr,
                    grid_config_2d_ptr<fp_type> const &grid_config, range<fp_type> const &time_range,
                    std::size_t const &last_time_idx, fp_type const time_step,
                    traverse_direction_enum const &traverse_dir, container_2d_t &prev_solution,
                    container_2d_t &next_solution);
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
template <typename solver, typename boundary_solver>
void heston_euler_time_loop<fp_type, container, allocator>::run(
    solver const &solver_ptr, boundary_solver const &boundary_solver_ptr,
    boundary_2d_pair<fp_type> const &horizontal_boundary_pair,
    boundary_2d_ptr<fp_type> const &vertical_upper_boundary_ptr, grid_config_2d_ptr<fp_type> const &grid_config,
    range<fp_type> const &time_range, std::size_t const &last_time_idx, fp_type const time_step,
    traverse_direction_enum const &traverse_dir, container_2d_t &prev_solution, container_2d_t &next_solution)
{
    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type k = time_step;

    fp_type time{start_time + k};
    std::size_t time_idx{};
    if (traverse_dir == traverse_direction_enum::Forward)
    {
        time_idx = 1;
        while (time_idx <= last_time_idx)
        {
            solver_ptr->solve(prev_solution, time, next_solution);
            boundary_solver_ptr->solve(prev_solution, horizontal_boundary_pair, vertical_upper_boundary_ptr, time,
                                       next_solution);
            boundary_solver_ptr->solve(prev_solution, horizontal_boundary_pair, time, next_solution);
            prev_solution = next_solution;
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
            solver_ptr->solve(prev_solution, time, next_solution);
            boundary_solver_ptr->solve(prev_solution, horizontal_boundary_pair, vertical_upper_boundary_ptr, time,
                                       next_solution);
            boundary_solver_ptr->solve(prev_solution, horizontal_boundary_pair, time, next_solution);
            prev_solution = next_solution;

            time -= k;
        } while (time_idx > 0);
    }
    else
    {
        throw std::exception("Unreachable");
    }
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class heston_euler_scheme
{
    typedef heston_euler_time_loop<fp_type, container, allocator> loop;
    typedef discretization<dimension_enum::Two, fp_type, container, allocator> d_2d;
    typedef container_2d<by_enum::Row, fp_type, container, allocator> container_2d_t;
    typedef sptr_t<general_heston_equation_explicit_boundary<fp_type, container, allocator>> heston_boundary_ptr;

  private:
    heston_euler_coefficients_ptr<fp_type> euler_coeffs_;
    heston_boundary_ptr heston_boundary_;
    boundary_2d_ptr<fp_type> boundary_ver_;
    boundary_2d_pair<fp_type> boundary_pair_hor_;
    pde_discretization_config_2d_ptr<fp_type> discretization_cfg_;
    grid_config_2d_ptr<fp_type> grid_cfg_;

    bool is_stable(general_heston_equation_coefficients_ptr<fp_type> const &coefficients)
    {
        // TODO: this needs to be implemented !!!
        return true;
    }

    void initialize(general_heston_equation_coefficients_ptr<fp_type> const &coefficients)
    {
        LSS_ASSERT(is_stable(coefficients) == true, "The chosen scheme is not stable");
        euler_coeffs_ = std::make_shared<heston_euler_coefficients<fp_type>>(coefficients);
        heston_boundary_ = std::make_shared<general_heston_equation_explicit_boundary<fp_type, container, allocator>>(
            coefficients, grid_cfg_);
    }

    explicit heston_euler_scheme() = delete;

  public:
    heston_euler_scheme(general_heston_equation_coefficients_ptr<fp_type> const &coefficients,
                        boundary_2d_ptr<fp_type> const &vertical_upper_boundary_ptr,
                        boundary_2d_pair<fp_type> const &horizontal_boundary_pair,
                        pde_discretization_config_2d_ptr<fp_type> const &discretization_config,
                        grid_config_2d_ptr<fp_type> const &grid_config)
        : boundary_ver_{vertical_upper_boundary_ptr}, boundary_pair_hor_{horizontal_boundary_pair},
          discretization_cfg_{discretization_config}, grid_cfg_{grid_config}
    {
        initialize(coefficients);
    }

    ~heston_euler_scheme()
    {
    }

    void operator()(container_2d_t &prev_solution, container_2d_t &next_solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type, fp_type)> const &heat_source,
                    traverse_direction_enum traverse_dir)
    {
        const range<fp_type> timer = discretization_cfg_->time_range();
        const fp_type k = discretization_cfg_->time_step();
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        auto const &solver_method_ptr = std::make_shared<heston_euler_solver_method<fp_type, container, allocator>>(
            euler_coeffs_, grid_cfg_, is_heat_sourse_set);
        if (is_heat_sourse_set)
        {
            // TODO: to be implemented!!!
            //  loop::run(solver_method_ptr, boundary_pair_, timer, last_time_idx, k, traverse_dir, heat_source,
            //  solution);
        }
        else
        {
            loop::run(solver_method_ptr, heston_boundary_, boundary_pair_hor_, boundary_ver_, grid_cfg_, timer,
                      last_time_idx, k, traverse_dir, prev_solution, next_solution);
        }
    }
};

} // namespace two_dimensional
} // namespace lss_pde_solvers
#endif ///_LSS_HESTON_EULER_SCHEME_HPP_
