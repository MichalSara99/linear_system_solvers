#if !defined(_LSS_2D_GENERAL_HESTON_EQUATION_EXPLICIT_BOUNDARY_HPP_)
#define _LSS_2D_GENERAL_HESTON_EQUATION_EXPLICIT_BOUNDARY_HPP_

#include <vector>

#include "boundaries/lss_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include "containers/lss_container_2d.hpp"
#include "discretization/lss_discretization.hpp"
#include "discretization/lss_grid.hpp"
#include "discretization/lss_grid_config.hpp"
#include "pde_solvers/lss_pde_discretization_config.hpp"
#include "pde_solvers/two_dimensional/heat_type/implicit_coefficients/lss_2d_general_heston_equation_coefficients.hpp"

namespace lss_pde_solvers
{
namespace two_dimensional
{

using lss_boundary::boundary_2d_pair;
using lss_boundary::boundary_2d_ptr;
using lss_boundary::dirichlet_boundary_2d;
using lss_boundary::neumann_boundary_2d;
using lss_containers::container_2d;
using lss_enumerations::by_enum;
using lss_utility::range;

/**
    explicit_heston_boundary_scheme object
 */
template <typename fp_type, template <typename, typename> typename container, typename allocator>
class explicit_heston_boundary_scheme
{
    typedef container<fp_type, allocator> container_t;
    typedef container_2d<by_enum::Row, fp_type, container, allocator> rcontainer_2d_t;

  public:
    static void rhs(general_heston_equation_coefficients_ptr<fp_type> const &cfg,
                    grid_config_2d_ptr<fp_type> const &grid_cfg, std::size_t const &y_index, fp_type const &y,
                    boundary_2d_pair<fp_type> const &horizontal_boundary_pair, rcontainer_2d_t const &input,
                    fp_type const &time, container_t &solution)
    {
        auto const four = static_cast<fp_type>(4.0);
        auto const three = static_cast<fp_type>(3.0);
        auto const two = static_cast<fp_type>(2.0);
        auto const one = static_cast<fp_type>(1.0);
        auto const &D = cfg->D_;
        auto const &E = cfg->E_;
        auto const &F = cfg->F_;
        auto const delta = cfg->delta_;
        auto const ni = cfg->ni_;
        auto const rho = cfg->rho_;

        auto const &first_bnd = horizontal_boundary_pair.first;
        auto const &second_bnd = horizontal_boundary_pair.second;

        fp_type x{}, h_1{};
        if (auto const &ptr = std::dynamic_pointer_cast<dirichlet_boundary_2d<fp_type>>(first_bnd))
        {
            solution[0] = ptr->value(time, y);
        }

        const std::size_t N = solution.size() - 1;
        if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_2d<fp_type>>(second_bnd))
        {
            x = grid_2d<fp_type>::value_1(grid_cfg, N);
            h_1 = grid_2d<fp_type>::step_1(grid_cfg);
            const fp_type delta_ = two * h_1 * ptr->value(time, y);
            solution[N] = ((one - three * ni * E(time, x, y) + rho * F(time, x, y)) * input(N, y_index)) +
                          (four * ni * E(time, x, y) * input(N, y_index + 1)) -
                          (ni * E(time, x, y) * input(N, y_index + 2)) - (delta * delta_ * D(time, x, y));
        }

        for (std::size_t t = 1; t < N; ++t)
        {
            x = grid_2d<fp_type>::value_1(grid_cfg, t);
            solution[t] = (-delta * D(time, x, y) * input(t - 1, y_index)) +
                          ((one - three * ni * E(time, x, y) + rho * F(time, x, y)) * input(t, y_index)) +
                          (delta * D(time, x, y) * input(t + 1, y_index)) -
                          (ni * E(time, x, y) * input(t, y_index + 2)) +
                          (four * ni * E(time, x, y) * input(t, y_index + 1));
        }
    }

    static void rhs_source(general_heston_equation_coefficients_ptr<fp_type> const &cfg,
                           grid_config_2d_ptr<fp_type> const &grid_cfg, std::size_t const &y_index, fp_type const &y,
                           boundary_2d_pair<fp_type> const &horizontal_boundary_pair, rcontainer_2d_t const &input,
                           rcontainer_2d_t const &inhom_input, fp_type const &time, container_t &solution)
    {
        auto const four = static_cast<fp_type>(4.0);
        auto const three = static_cast<fp_type>(3.0);
        auto const two = static_cast<fp_type>(2.0);
        auto const one = static_cast<fp_type>(1.0);
        auto const &D = cfg->D_;
        auto const &E = cfg->E_;
        auto const &F = cfg->F_;
        auto const delta = cfg->delta_;
        auto const ni = cfg->ni_;
        auto const rho = cfg->rho_;

        auto const &first_bnd = horizontal_boundary_pair.first;
        auto const &second_bnd = horizontal_boundary_pair.second;

        fp_type x{}, h_1{};
        if (auto const &ptr = std::dynamic_pointer_cast<dirichlet_boundary_2d<fp_type>>(first_bnd))
        {
            solution[0] = ptr->value(time, y);
        }

        const std::size_t N = solution.size() - 1;
        if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_2d<fp_type>>(second_bnd))
        {
            x = grid_2d<fp_type>::value_1(grid_cfg, N);
            h_1 = grid_2d<fp_type>::step_1(grid_cfg);
            const fp_type delta_ = two * h_1 * ptr->value(time, y);
            solution[N] = ((one - three * ni * E(time, x, y) + rho * F(time, x, y)) * input(N, y_index)) +
                          (four * ni * E(time, x, y) * input(N, y_index + 1)) -
                          (ni * E(time, x, y) * input(N, y_index + 2)) - (delta * delta_ * D(time, x, y)) +
                          (rho * inhom_input(N, y_index));
        }

        for (std::size_t t = 1; t < N; ++t)
        {
            x = grid_2d<fp_type>::value_1(grid_cfg, t);
            solution[t] = (-delta * D(time, x, y) * input(t - 1, y_index)) +
                          ((one - three * ni * E(time, x, y) + rho * F(time, x, y)) * input(t, y_index)) +
                          (delta * D(time, x, y) * input(t + 1, y_index)) -
                          (ni * E(time, x, y) * input(t, y_index + 2)) +
                          (four * ni * E(time, x, y) * input(t, y_index + 1)) + (rho * inhom_input(N, y_index));
        }
    }
};

/**
    general_heston_equation_explicit_boundary object
 */
template <typename fp_type, template <typename, typename> typename container = std::vector,
          typename allocator = std::allocator<fp_type>>
class general_heston_equation_explicit_boundary
{
    typedef container_2d<by_enum::Row, fp_type, container, allocator> rcontainer_2d_t;
    typedef container_2d<by_enum::Column, fp_type, container, allocator> ccontainer_2d_t;
    typedef container<fp_type, allocator> container_t;
    typedef explicit_heston_boundary_scheme<fp_type, container, allocator> heston_boundary_scheme;
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;

  private:
    general_heston_equation_coefficients_ptr<fp_type> coefficients_;
    grid_config_2d_ptr<fp_type> grid_cfg_;

    explicit general_heston_equation_explicit_boundary() = delete;

  public:
    explicit general_heston_equation_explicit_boundary(
        general_heston_equation_coefficients_ptr<fp_type> const &coefficients,
        grid_config_2d_ptr<fp_type> const &grid_config)
        : coefficients_{coefficients}, grid_cfg_{grid_config}
    {
    }

    ~general_heston_equation_explicit_boundary()
    {
    }

    general_heston_equation_explicit_boundary(general_heston_equation_explicit_boundary const &) = delete;
    general_heston_equation_explicit_boundary(general_heston_equation_explicit_boundary &&) = delete;
    general_heston_equation_explicit_boundary &operator=(general_heston_equation_explicit_boundary const &) = delete;
    general_heston_equation_explicit_boundary &operator=(general_heston_equation_explicit_boundary &&) = delete;

    void solve(container_2d<by_enum::Row, fp_type, container, allocator> const &prev_solution,
               boundary_2d_pair<fp_type> const &horizonatal_boundary_pair,
               boundary_2d_ptr<fp_type> const &vertical_upper_boundary_ptr, fp_type const &time,
               container_2d<by_enum::Row, fp_type, container, allocator> &solution);

    void solve(container_2d<by_enum::Row, fp_type, container, allocator> const &prev_solution,
               boundary_2d_pair<fp_type> const &horizonatal_boundary_pair, fp_type const &time,
               container_2d<by_enum::Row, fp_type, container, allocator> &solution);
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void general_heston_equation_explicit_boundary<fp_type, container, allocator>::solve(
    container_2d<by_enum::Row, fp_type, container, allocator> const &prev_solution,
    boundary_2d_pair<fp_type> const &horizonatal_boundary_pair,
    boundary_2d_ptr<fp_type> const &vertical_upper_boundary_ptr, fp_type const &time,
    container_2d<by_enum::Row, fp_type, container, allocator> &solution)
{
    ccontainer_2d_t csolution(solution);
    // 1D container for intermediate solution:
    container_t solution_v(coefficients_->space_size_x_, fp_type{});
    // get the right-hand side of the scheme:
    heston_boundary_scheme::rhs(coefficients_, grid_cfg_, 0, fp_type(0.0), horizonatal_boundary_pair, prev_solution,
                                time, solution_v);
    csolution(0, solution_v);
    auto const &upper_bnd_ptr = std::dynamic_pointer_cast<dirichlet_boundary_2d<fp_type>>(vertical_upper_boundary_ptr);
    auto const &upper_bnd = [=](fp_type s, fp_type t) { return upper_bnd_ptr->value(t, s); };
    d_1d::of_function(grid_cfg_->grid_1(), time, upper_bnd, solution_v);
    csolution(coefficients_->space_size_y_ - 1, solution_v);
    solution = csolution;
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void general_heston_equation_explicit_boundary<fp_type, container, allocator>::solve(
    container_2d<by_enum::Row, fp_type, container, allocator> const &prev_solution,
    boundary_2d_pair<fp_type> const &horizonatal_boundary_pair, fp_type const &time,
    container_2d<by_enum::Row, fp_type, container, allocator> &solution)
{
    // 1D container for intermediate solution:
    container_t solution_v(coefficients_->space_size_y_, fp_type{});
    // some constants:
    auto const &start_y = coefficients_->rangey_.lower();
    // prepare grid_1:
    auto const &grid_1 = grid_cfg_->grid_1();
    // prepare grid_2:
    auto const &grid_2 = grid_cfg_->grid_2();
    // populating lower horizontal:
    auto const &lower_bnd_ptr =
        std::dynamic_pointer_cast<dirichlet_boundary_2d<fp_type>>(horizonatal_boundary_pair.first);
    auto const &lower_bnd = [=](fp_type v, fp_type t) { return lower_bnd_ptr->value(t, v); };
    d_1d::of_function(grid_2, time, lower_bnd, solution_v);
    solution(0, solution_v);
    // populating upper horizontal:
    auto const lri = solution.rows() - 1;
    auto const two = static_cast<fp_type>(2.0);
    auto const three = static_cast<fp_type>(3.0);
    auto const four = static_cast<fp_type>(4.0);
    auto const N_x = coefficients_->space_size_x_;
    auto const &upper_bnd_ptr =
        std::dynamic_pointer_cast<neumann_boundary_2d<fp_type>>(horizonatal_boundary_pair.second);
    auto const &upper_bnd = [=](fp_type v, fp_type t) {
        const std::size_t j = grid_1d<fp_type>::index_of(grid_2, v);
        auto const bnd_val = upper_bnd_ptr->value(t, v);
        auto const h_1 = grid_1d<fp_type>::step(grid_1);
        return (((four * solution(lri - 1, j)) - solution(lri - 2, j) - (two * h_1 * bnd_val)) / three);
    };
    d_1d::of_function(grid_2, time, upper_bnd, solution_v);
    solution(N_x - 1, solution_v);
}

} // namespace two_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_2D_GENERAL_HESTON_EQUATION_EXPLICIT_BOUNDARY_HPP_
