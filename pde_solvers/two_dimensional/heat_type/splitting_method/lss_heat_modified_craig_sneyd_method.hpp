#if !defined(_LSS_HEAT_MODIFIED_CRAIG_SNEYD_METHOD_HPP_)
#define _LSS_HEAT_MODIFIED_CRAIG_SNEYD_METHOD_HPP_

#include <functional>
#include <map>

#include "boundaries/lss_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "containers/lss_container_2d.hpp"
#include "discretization/lss_discretization.hpp"
#include "discretization/lss_grid.hpp"
#include "discretization/lss_grid_config.hpp"
#include "lss_heat_splitting_method.hpp"
#include "pde_solvers/lss_pde_discretization_config.hpp"
#include "pde_solvers/two_dimensional/heat_type/implicit_coefficients/lss_2d_general_heston_equation_coefficients.hpp"

namespace lss_pde_solvers
{

namespace two_dimensional
{
using lss_boundary::boundary_2d_pair;
using lss_boundary::boundary_2d_ptr;
using lss_containers::container_2d;
using lss_enumerations::by_enum;
using lss_grids::grid_2d;
using lss_grids::grid_config_2d_ptr;

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class implicit_heston_scheme_mcs
{
    typedef container<fp_type, allocator> container_t;
    typedef container_2d<by_enum::Row, fp_type, container, allocator> rcontainer_2d_t;

  public:
    static void rhs_intermed_1(general_heston_equation_coefficients_ptr<fp_type> const &cfs,
                               grid_config_2d_ptr<fp_type> const &grid_cfg, std::size_t const &y_index,
                               fp_type const &y, rcontainer_2d_t const &input, fp_type const &time,
                               container_t &solution)
    {
        auto const one = static_cast<fp_type>(1.0);
        auto const &M = cfs->M_;
        auto const &M_tilde = cfs->M_tilde_;
        auto const &P = cfs->P_;
        auto const &P_tilde = cfs->P_tilde_;
        auto const &Z = cfs->Z_;
        auto const &W = cfs->W_;
        auto const &C = cfs->C_;

        auto const gamma = cfs->gamma_;
        auto const theta = cfs->theta_;

        const std::size_t N = solution.size() - 1;
        fp_type x{};
        for (std::size_t t = 1; t < N; ++t)
        {
            x = grid_2d<fp_type>::value_1(grid_cfg, t);
            solution[t] =
                (gamma * C(time, x, y) * input(t - 1, y_index - 1)) +
                ((one - theta) * M(time, x, y) * input(t - 1, y_index)) -
                (gamma * C(time, x, y) * input(t - 1, y_index + 1)) + (M_tilde(time, x, y) * input(t, y_index - 1)) +
                ((one - W(time, x, y) - (one - theta) * Z(time, x, y)) * input(t, y_index)) +
                (P_tilde(time, x, y) * input(t, y_index + 1)) - (gamma * C(time, x, y) * input(t + 1, y_index - 1)) +
                ((one - theta) * P(time, x, y) * input(t + 1, y_index)) +
                (gamma * C(time, x, y) * input(t + 1, y_index + 1));
        }
    }

    static void rhs_intermed_1_source(general_heston_equation_coefficients_ptr<fp_type> const &cfs,
                                      grid_config_2d_ptr<fp_type> const &grid_cfg, std::size_t const &y_index,
                                      fp_type const &y, rcontainer_2d_t const &input,
                                      rcontainer_2d_t const &inhom_input, rcontainer_2d_t const &inhom_input_next,
                                      fp_type const &time, container_t &solution)
    {
        auto const one = static_cast<fp_type>(1.0);
        auto const &M = cfs->M_;
        auto const &M_tilde = cfs->M_tilde_;
        auto const &P = cfs->P_;
        auto const &P_tilde = cfs->P_tilde_;
        auto const &Z = cfs->Z_;
        auto const &W = cfs->W_;
        auto const &C = cfs->C_;

        auto const gamma = cfs->gamma_;
        auto const theta = cfs->theta_;
        auto const rho = cfs->rho_;

        const std::size_t N = solution.size() - 1;
        fp_type x{};
        for (std::size_t t = 1; t < N; ++t)
        {
            x = grid_2d<fp_type>::value_1(grid_cfg, t);
            solution[t] =
                (gamma * C(time, x, y) * input(t - 1, y_index - 1)) +
                ((one - theta) * M(time, x, y) * input(t - 1, y_index)) -
                (gamma * C(time, x, y) * input(t - 1, y_index + 1)) + (M_tilde(time, x, y) * input(t, y_index - 1)) +
                ((one - W(time, x, y) - (one - theta) * Z(time, x, y)) * input(t, y_index)) +
                (P_tilde(time, x, y) * input(t, y_index + 1)) - (gamma * C(time, x, y) * input(t + 1, y_index - 1)) +
                ((one - theta) * P(time, x, y) * input(t + 1, y_index)) +
                (gamma * C(time, x, y) * input(t + 1, y_index + 1)) + (theta * rho * inhom_input_next(t, y_index)) +
                ((one - theta) * rho * inhom_input(t, y_index));
        }
    }

    static void rhs_intermed_2(general_heston_equation_coefficients_ptr<fp_type> const &cfs,
                               grid_config_2d_ptr<fp_type> const &grid_cfg, std::size_t const &x_index,
                               fp_type const &x, rcontainer_2d_t const &input, rcontainer_2d_t const &inhom_input,
                               fp_type const &time, container_t &solution)
    {
        auto const &M_tilde = cfs->M_tilde_;
        auto const &P_tilde = cfs->P_tilde_;
        auto const &W = cfs->W_;

        auto const theta = cfs->theta_;

        const std::size_t N = solution.size() - 1;
        fp_type y{};
        for (std::size_t t = 1; t < N; ++t)
        {
            y = grid_2d<fp_type>::value_2(grid_cfg, t);
            solution[t] = (-theta * M_tilde(time, x, y) * input(x_index, t - 1)) +
                          (theta * W(time, x, y) * input(x_index, t)) -
                          (theta * P_tilde(time, x, y) * input(x_index, t + 1)) + inhom_input(x_index, t);
        }
    }

    static void rhs_intermed_3(general_heston_equation_coefficients_ptr<fp_type> const &cfs,
                               grid_config_2d_ptr<fp_type> const &grid_cfg, std::size_t const &y_index,
                               fp_type const &y, rcontainer_2d_t const &input, rcontainer_2d_t const &inhom_input,
                               rcontainer_2d_t const &inhom_input_next, fp_type const &time, container_t &solution)
    {
        auto const &M = cfs->M_;
        auto const &P = cfs->P_;
        auto const &M_tilde = cfs->M_tilde_;
        auto const &P_tilde = cfs->P_tilde_;
        auto const &Z = cfs->Z_;
        auto const &C = cfs->C_;
        auto const &W = cfs->W_;

        auto const gamma = cfs->gamma_;
        auto const theta = cfs->theta_;
        auto const zeta = cfs->zeta_;

        const std::size_t N = solution.size() - 1;
        fp_type x{};
        for (std::size_t t = 1; t < N; ++t)
        {
            x = grid_2d<fp_type>::value_1(grid_cfg, t);
            solution[t] = (-theta * M(time, x, y) * input(t - 1, y_index)) +
                          (theta * Z(time, x, y) * input(t, y_index)) -
                          (theta * P(time, x, y) * input(t + 1, y_index)) + (inhom_input(t, y_index)) +
                          (zeta * gamma * C(time, x, y) *
                           ((inhom_input_next(t + 1, y_index + 1) - input(t + 1, y_index + 1)) -
                            (inhom_input_next(t + 1, y_index - 1) - input(t + 1, y_index - 1)) -
                            (inhom_input_next(t - 1, y_index + 1) - input(t - 1, y_index + 1)) +
                            (inhom_input_next(t - 1, y_index - 1) - input(t - 1, y_index - 1)))) +
                          ((zeta - theta) *
                           (M(time, x, y) * (inhom_input_next(t - 1, y_index) - input(t - 1, y_index)) -
                            (Z(time, x, y) + W(time, x, y)) * (inhom_input_next(t, y_index) - input(t, y_index)) -
                            P(time, x, y) * (inhom_input_next(t + 1, y_index) - input(t + 1, y_index)) +
                            M_tilde(time, x, y) * (inhom_input_next(t, y_index - 1) - input(t, y_index - 1)) +
                            P_tilde(time, x, y) * (inhom_input_next(t, y_index + 1) - input(t, y_index + 1))));
        }
    }

    static void rhs(general_heston_equation_coefficients_ptr<fp_type> const &cfs,
                    grid_config_2d_ptr<fp_type> const &grid_cfg, std::size_t const &x_index, fp_type const &x,
                    rcontainer_2d_t const &input, rcontainer_2d_t const &inhom_input, fp_type const &time,
                    container_t &solution)
    {
        auto const &M_tilde = cfs->M_tilde_;
        auto const &P_tilde = cfs->P_tilde_;
        auto const &W = cfs->W_;

        auto const theta = cfs->theta_;

        const std::size_t N = solution.size() - 1;
        fp_type y{};
        for (std::size_t t = 1; t < N; ++t)
        {
            y = grid_2d<fp_type>::value_2(grid_cfg, t);
            solution[t] = (-theta * M_tilde(time, x, y) * input(x_index, t - 1)) +
                          (theta * W(time, x, y) * input(x_index, t)) -
                          (theta * P_tilde(time, x, y) * input(x_index, t + 1)) + inhom_input(x_index, t);
        }
    }
};

/**
    heat_modified_craig_sneyd_method object
 */
template <typename fp_type, typename solver, template <typename, typename> typename container = std::vector,
          typename allocator = std::allocator<fp_type>>
class heat_modified_craig_sneyd_method : public heat_splitting_method<fp_type, container, allocator>
{
    typedef container_2d<by_enum::Row, fp_type, container, allocator> rcontainer_2d_t;
    typedef container_2d<by_enum::Column, fp_type, container, allocator> ccontainer_2d_t;
    typedef container<fp_type, allocator> container_t;
    typedef implicit_heston_scheme_mcs<fp_type, container, allocator> heston_scheme;

  private:
    // constant:
    const fp_type cone_ = static_cast<fp_type>(1.0);
    // solvers:
    solver solvery_ptr_;
    solver solveru_ptr_;
    // scheme coefficients:
    general_heston_equation_coefficients_ptr<fp_type> coefficients_;
    grid_config_2d_ptr<fp_type> grid_cfg_;
    // container:
    container_t low_, diag_, high_, rhs_;

    explicit heat_modified_craig_sneyd_method() = delete;

    void initialize(bool is_heat_source_set)
    {
    }

    void split_0(fp_type const &y, fp_type const &time, container_t &low, container_t &diag, container_t &high)
    {
        fp_type x{};
        for (std::size_t t = 0; t < low.size(); ++t)
        {
            x = grid_2d<fp_type>::value_1(grid_cfg_, t);
            low[t] = (-coefficients_->theta_ * coefficients_->M_(time, x, y));
            diag[t] = (cone_ + coefficients_->theta_ * coefficients_->Z_(time, x, y));
            high[t] = (-coefficients_->theta_ * coefficients_->P_(time, x, y));
        }
    }

    void split_1(fp_type const &x, fp_type const &time, container_t &low, container_t &diag, container_t &high)
    {
        fp_type y{};
        for (std::size_t t = 0; t < low.size(); ++t)
        {
            y = grid_2d<fp_type>::value_2(grid_cfg_, t);
            low[t] = (-coefficients_->theta_ * coefficients_->M_tilde_(time, x, y));
            diag[t] = (cone_ + coefficients_->theta_ * coefficients_->W_(time, x, y));
            high[t] = (-coefficients_->theta_ * coefficients_->P_tilde_(time, x, y));
        }
    }

  public:
    explicit heat_modified_craig_sneyd_method(solver const &solvery_ptr, solver const &solveru_ptr,
                                              general_heston_equation_coefficients_ptr<fp_type> const &coefficients,
                                              grid_config_2d_ptr<fp_type> const &grid_config, bool is_heat_source_set)
        : solvery_ptr_{solvery_ptr}, solveru_ptr_{solveru_ptr}, coefficients_{coefficients}, grid_cfg_{grid_config}
    {
        initialize(is_heat_source_set);
    }

    ~heat_modified_craig_sneyd_method()
    {
    }

    heat_modified_craig_sneyd_method(heat_modified_craig_sneyd_method const &) = delete;
    heat_modified_craig_sneyd_method(heat_modified_craig_sneyd_method &&) = delete;
    heat_modified_craig_sneyd_method &operator=(heat_modified_craig_sneyd_method const &) = delete;
    heat_modified_craig_sneyd_method &operator=(heat_modified_craig_sneyd_method &&) = delete;

    void solve(container_2d<by_enum::Row, fp_type, container, allocator> const &prev_solution,
               boundary_2d_pair<fp_type> const &horizontal_boundary_pair,
               boundary_2d_pair<fp_type> const &vertical_boundary_pair, fp_type const &time,
               container_2d<by_enum::Row, fp_type, container, allocator> &solution) override;

    void solve(container_2d<by_enum::Row, fp_type, container, allocator> const &prev_solution,
               boundary_2d_pair<fp_type> const &horizontal_boundary_pair,
               boundary_2d_pair<fp_type> const &vertical_boundary_pair, fp_type const &time,
               std::function<fp_type(fp_type, fp_type)> const &heat_source,
               container_2d<by_enum::Row, fp_type, container, allocator> &solution) override;
};

template <typename fp_type, typename solver, template <typename, typename> typename container, typename allocator>
void heat_modified_craig_sneyd_method<fp_type, solver, container, allocator>::solve(
    container_2d<by_enum::Row, fp_type, container, allocator> const &prev_solution,
    boundary_2d_pair<fp_type> const &horizontal_boundary_pair, boundary_2d_pair<fp_type> const &vertical_boundary_pair,
    fp_type const &time, container_2d<by_enum::Row, fp_type, container, allocator> &solution)
{

    // 2D container for intermediate solution Y_1:
    ccontainer_2d_t inter_solution_1(coefficients_->space_size_x_, coefficients_->space_size_y_, fp_type{});
    // 1D container for intermediate solution Y_1:
    container_t solution_v(coefficients_->space_size_x_, fp_type{});
    // containers for first split solver:
    low_.resize(coefficients_->space_size_x_);
    diag_.resize(coefficients_->space_size_x_);
    high_.resize(coefficients_->space_size_x_);
    rhs_.resize(coefficients_->space_size_x_);
    // get the right-hand side of the scheme:
    fp_type y{};
    for (std::size_t j = 1; j < coefficients_->space_size_y_ - 1; ++j)
    {
        y = grid_2d<fp_type>::value_2(grid_cfg_, j);
        split_0(y, time, low_, diag_, high_);
        heston_scheme::rhs_intermed_1(coefficients_, grid_cfg_, j, y, prev_solution, time, rhs_);
        solvery_ptr_->set_diagonals(low_, diag_, high_);
        solvery_ptr_->set_rhs(rhs_);
        solvery_ptr_->solve(horizontal_boundary_pair, solution_v, time, y);
        inter_solution_1(j, solution_v);
    }

    // 2D container for intermediate solution Y_2:
    rcontainer_2d_t inter_solution_2(coefficients_->space_size_x_, coefficients_->space_size_y_, fp_type{});
    // 1D container for intermediate solution Y_2:
    solution_v.resize(coefficients_->space_size_y_);
    // containers for second split solver:
    low_.resize(coefficients_->space_size_y_);
    diag_.resize(coefficients_->space_size_y_);
    high_.resize(coefficients_->space_size_y_);
    rhs_.resize(coefficients_->space_size_y_);
    // get the right-hand side of the scheme:
    fp_type x{};
    for (std::size_t i = 1; i < coefficients_->space_size_x_ - 1; ++i)
    {
        x = grid_2d<fp_type>::value_1(grid_cfg_, i);
        split_1(x, time, low_, diag_, high_);
        heston_scheme::rhs_intermed_2(coefficients_, grid_cfg_, i, x, prev_solution, inter_solution_1, time, rhs_);
        solveru_ptr_->set_diagonals(low_, diag_, high_);
        solveru_ptr_->set_rhs(rhs_);
        solveru_ptr_->solve(vertical_boundary_pair, solution_v, time, x);
        inter_solution_2(i, solution_v);
    }

    // 2D container for intermediate solution Y_3:
    ccontainer_2d_t inter_solution_3(coefficients_->space_size_x_, coefficients_->space_size_y_, fp_type{});
    // 1D container for intermediate solution Y_3:
    solution_v.resize(coefficients_->space_size_x_);
    // containers for second split solver:
    low_.resize(coefficients_->space_size_x_);
    diag_.resize(coefficients_->space_size_x_);
    high_.resize(coefficients_->space_size_x_);
    rhs_.resize(coefficients_->space_size_x_);
    for (std::size_t j = 1; j < coefficients_->space_size_y_ - 1; ++j)
    {
        y = grid_2d<fp_type>::value_2(grid_cfg_, j);
        split_0(y, time, low_, diag_, high_);
        heston_scheme::rhs_intermed_3(coefficients_, grid_cfg_, j, y, prev_solution, inter_solution_1, inter_solution_2,
                                      time, rhs_);
        solvery_ptr_->set_diagonals(low_, diag_, high_);
        solvery_ptr_->set_rhs(rhs_);
        solvery_ptr_->solve(horizontal_boundary_pair, solution_v, time, y);
        inter_solution_3(j, solution_v);
    }

    // 1D container for final solution:
    solution_v.resize(coefficients_->space_size_y_);
    // containers for second split solver:
    low_.resize(coefficients_->space_size_y_);
    diag_.resize(coefficients_->space_size_y_);
    high_.resize(coefficients_->space_size_y_);
    rhs_.resize(coefficients_->space_size_y_);
    for (std::size_t i = 1; i < coefficients_->space_size_x_ - 1; ++i)
    {
        x = grid_2d<fp_type>::value_1(grid_cfg_, i);
        split_1(x, time, low_, diag_, high_);
        heston_scheme::rhs(coefficients_, grid_cfg_, i, x, prev_solution, inter_solution_3, time, rhs_);
        solveru_ptr_->set_diagonals(low_, diag_, high_);
        solveru_ptr_->set_rhs(rhs_);
        solveru_ptr_->solve(vertical_boundary_pair, solution_v, time, x);
        solution(i, solution_v);
    }
}

template <typename fp_type, typename solver, template <typename, typename> typename container, typename allocator>
void heat_modified_craig_sneyd_method<fp_type, solver, container, allocator>::solve(
    container_2d<by_enum::Row, fp_type, container, allocator> const &prev_solution,
    boundary_2d_pair<fp_type> const &horizontal_boundary_pair, boundary_2d_pair<fp_type> const &vertical_boundary_pair,
    fp_type const &time, std::function<fp_type(fp_type, fp_type)> const &heat_source,
    container_2d<by_enum::Row, fp_type, container, allocator> &solution)
{
}
} // namespace two_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_HEAT_MODIFIED_CRAIG_SNEYD_METHOD_HPP_
