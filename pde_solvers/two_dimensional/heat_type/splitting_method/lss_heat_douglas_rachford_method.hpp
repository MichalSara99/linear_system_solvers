#if !defined(_LSS_HEAT_DOUGLAS_RACHFORD_METHOD_HPP_)
#define _LSS_HEAT_DOUGLAS_RACHFORD_METHOD_HPP_

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
#include "pde_solvers/two_dimensional/heat_type/implicit_coefficients/lss_2d_general_svc_heston_equation_coefficients.hpp"

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
using lss_utility::pair_t;

template <template <typename, typename> typename container, typename fp_type, typename alloc>
using implicit_heston_scheme_function_t = std::function<void(
    general_svc_heston_equation_coefficients_ptr<fp_type> const &, grid_config_2d_ptr<fp_type> const &,
    std::size_t const &, fp_type const &, container_2d<by_enum::Row, fp_type, container, alloc> const &,
    container_2d<by_enum::Row, fp_type, container, alloc> const &,
    container_2d<by_enum::Row, fp_type, container, alloc> const &, fp_type const &, container<fp_type, alloc> &)>;

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class implicit_heston_scheme
{
    typedef container<fp_type, allocator> container_t;
    typedef container_2d<by_enum::Row, fp_type, container, allocator> rcontainer_2d_t;
    typedef implicit_heston_scheme_function_t<container, fp_type, allocator> scheme_function_t;

  public:
    static scheme_function_t const get_intermediate(std::pair<fp_type, fp_type> const &weights,
                                                    std::pair<fp_type, fp_type> const &weight_values,
                                                    bool is_homogeneus)
    {
        const fp_type zero = static_cast<fp_type>(0.0);
        const fp_type one = static_cast<fp_type>(1.0);

        auto scheme_fun_h = [=](general_svc_heston_equation_coefficients_ptr<fp_type> const &cfs,
                                grid_config_2d_ptr<fp_type> const &grid_cfg, std::size_t const &y_index,
                                fp_type const &y, rcontainer_2d_t const &input, rcontainer_2d_t const &inhom_input,
                                rcontainer_2d_t const &inhom_input_next, fp_type const &time, container_t &solution) {
            auto M = cfs->M_;
            auto M_tilde = cfs->M_tilde_;
            auto P = cfs->P_;
            auto P_tilde = cfs->P_tilde_;
            auto Z = cfs->Z_;
            auto W = cfs->W_;
            auto C = cfs->C_;

            auto const gamma = cfs->gamma_;
            auto const theta = cfs->theta_;
            auto const v_x = weight_values.first;
            auto const v_y = weight_values.second;
            auto const w_x = weights.first;
            auto const w_y = weights.second;

            const std::size_t N = solution.size() - 1;
            const fp_type wg_y = (y - v_y) <= zero ? one : w_y;
            fp_type x{};
            fp_type wg_x{};
            for (std::size_t t = 1; t < N; ++t)
            {
                x = grid_2d<fp_type>::value_1(grid_cfg, t);
                wg_x = (x - v_x) <= zero ? one : w_x;
                solution[t] =
                    (gamma * C(x, y) * input(t - 1, y_index - 1)) +
                    ((one - theta) * M(x, y, wg_x) * input(t - 1, y_index)) -
                    (gamma * C(x, y) * input(t - 1, y_index + 1)) + (M_tilde(x, y, wg_y) * input(t, y_index - 1)) +
                    ((one - W(x, y, wg_y) - (one - theta) * Z(x, y, wg_x)) * input(t, y_index)) +
                    (P_tilde(x, y, wg_y) * input(t, y_index + 1)) - (gamma * C(x, y) * input(t + 1, y_index - 1)) +
                    ((one - theta) * P(x, y, wg_x) * input(t + 1, y_index)) +
                    (gamma * C(x, y) * input(t + 1, y_index + 1));
            }
        };
        auto scheme_fun_nh = [=](general_svc_heston_equation_coefficients_ptr<fp_type> const &cfs,
                                 grid_config_2d_ptr<fp_type> const &grid_cfg, std::size_t const &y_index,
                                 fp_type const &y, rcontainer_2d_t const &input, rcontainer_2d_t const &inhom_input,
                                 rcontainer_2d_t const &inhom_input_next, fp_type const &time, container_t &solution) {
            auto M = cfs->M_;
            auto M_tilde = cfs->M_tilde_;
            auto P = cfs->P_;
            auto P_tilde = cfs->P_tilde_;
            auto Z = cfs->Z_;
            auto W = cfs->W_;
            auto C = cfs->C_;

            auto const gamma = cfs->gamma_;
            auto const theta = cfs->theta_;
            auto const rho = cfs->rho_;

            auto const v_x = weight_values.first;
            auto const v_y = weight_values.second;
            auto const w_x = weights.first;
            auto const w_y = weights.second;

            const std::size_t N = solution.size() - 1;
            const fp_type wg_y = (y - v_y) <= zero ? one : w_y;
            fp_type x{};
            fp_type wg_x{};
            for (std::size_t t = 1; t < N; ++t)
            {
                x = grid_2d<fp_type>::value_1(grid_cfg, t);
                wg_x = (x - v_x) <= zero ? one : w_x;
                solution[t] =
                    (gamma * C(x, y) * input(t - 1, y_index - 1)) +
                    ((one - theta) * M(x, y, wg_x) * input(t - 1, y_index)) -
                    (gamma * C(x, y) * input(t - 1, y_index + 1)) + (M_tilde(x, y, wg_y) * input(t, y_index - 1)) +
                    ((one - W(x, y, wg_y) - (one - theta) * Z(x, y, wg_x)) * input(t, y_index)) +
                    (P_tilde(x, y, wg_y) * input(t, y_index + 1)) - (gamma * C(x, y) * input(t + 1, y_index - 1)) +
                    ((one - theta) * P(x, y, wg_x) * input(t + 1, y_index)) +
                    (gamma * C(x, y) * input(t + 1, y_index + 1)) + (theta * rho * inhom_input_next(t, y_index)) +
                    ((one - theta) * rho * inhom_input(t, y_index));
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

    static scheme_function_t const get(std::pair<fp_type, fp_type> const &weights,
                                       std::pair<fp_type, fp_type> const &weight_values)
    {

        const fp_type zero = static_cast<fp_type>(0.0);
        const fp_type one = static_cast<fp_type>(1.0);

        auto scheme_fun = [=](general_svc_heston_equation_coefficients_ptr<fp_type> const &cfs,
                              grid_config_2d_ptr<fp_type> const &grid_cfg, std::size_t const &x_index, fp_type const &x,
                              rcontainer_2d_t const &input, rcontainer_2d_t const &inhom_input,
                              rcontainer_2d_t const &inhom_input_next, fp_type const &time, container_t &solution) {
            auto const M_tilde = cfs->M_tilde_;
            auto const P_tilde = cfs->P_tilde_;
            auto const W = cfs->W_;

            auto const theta = cfs->theta_;
            auto const v_y = weight_values.second;
            auto const w_y = weights.second;

            const std::size_t N = solution.size() - 1;
            fp_type y{};
            fp_type wg_y{};
            for (std::size_t t = 1; t < N; ++t)
            {
                y = grid_2d<fp_type>::value_2(grid_cfg, t);
                wg_y = (y - v_y) <= zero ? one : w_y;
                solution[t] = (-theta * M_tilde(x, y, wg_y) * input(x_index, t - 1)) +
                              (theta * W(x, y, wg_y) * input(x_index, t)) -
                              (theta * P_tilde(x, y, wg_y) * input(x_index, t + 1)) + inhom_input(x_index, t);
            }
        };

        return scheme_fun;
    }
};

/**
    heat_douglas_rachford_method object
 */
template <typename fp_type, typename solver, template <typename, typename> typename container = std::vector,
          typename allocator = std::allocator<fp_type>>
class heat_douglas_rachford_method : public heat_splitting_method<fp_type, container, allocator>
{
    typedef container_2d<by_enum::Row, fp_type, container, allocator> rcontainer_2d_t;
    typedef container_2d<by_enum::Column, fp_type, container, allocator> ccontainer_2d_t;
    typedef container<fp_type, allocator> container_t;

  private:
    // solvers:
    solver solvery_ptr_;
    solver solveru_ptr_;
    // scheme coefficients:
    general_svc_heston_equation_coefficients_ptr<fp_type> coefficients_;
    grid_config_2d_ptr<fp_type> grid_cfg_;

    explicit heat_douglas_rachford_method() = delete;

    void initialize()
    {
    }

    void split_0(fp_type const &y, fp_type const &v_x, fp_type const &w_x, container_t &low, container_t &diag,
                 container_t &high)
    {
        fp_type x{};
        fp_type w{};
        const fp_type one = static_cast<fp_type>(1.0);
        const fp_type zero = static_cast<fp_type>(0.0);
        for (std::size_t t = 0; t < low.size(); ++t)
        {
            x = grid_2d<fp_type>::value_1(grid_cfg_, t);
            w = (x - v_x) <= zero ? one : w_x;
            low[t] = (-coefficients_->theta_ * coefficients_->M_(x, y, w));
            diag[t] = (one + coefficients_->theta_ * coefficients_->Z_(x, y, w));
            high[t] = (-coefficients_->theta_ * coefficients_->P_(x, y, w));
        }
    }

    void split_1(fp_type const &x, fp_type const &v_y, fp_type const &w_y, container_t &low, container_t &diag,
                 container_t &high)
    {
        fp_type y{};
        fp_type w{};
        const fp_type one = static_cast<fp_type>(1.0);
        const fp_type zero = static_cast<fp_type>(0.0);
        for (std::size_t t = 0; t < low.size(); ++t)
        {
            y = grid_2d<fp_type>::value_2(grid_cfg_, t);
            w = (y - v_y) <= zero ? one : w_y;
            low[t] = (-coefficients_->theta_ * coefficients_->M_tilde_(x, y, w));
            diag[t] = (one + coefficients_->theta_ * coefficients_->W_(x, y, w));
            high[t] = (-coefficients_->theta_ * coefficients_->P_tilde_(x, y, w));
        }
    }

  public:
    explicit heat_douglas_rachford_method(solver const &solvery_ptr, solver const &solveru_ptr,
                                          general_svc_heston_equation_coefficients_ptr<fp_type> const &coefficients,
                                          grid_config_2d_ptr<fp_type> const &grid_config)
        : solvery_ptr_{solvery_ptr}, solveru_ptr_{solveru_ptr}, coefficients_{coefficients}, grid_cfg_{grid_config}
    {
        initialize();
    }

    ~heat_douglas_rachford_method()
    {
    }

    heat_douglas_rachford_method(heat_douglas_rachford_method const &) = delete;
    heat_douglas_rachford_method(heat_douglas_rachford_method &&) = delete;
    heat_douglas_rachford_method &operator=(heat_douglas_rachford_method const &) = delete;
    heat_douglas_rachford_method &operator=(heat_douglas_rachford_method &&) = delete;

    void solve(container_2d<by_enum::Row, fp_type, container, allocator> const &prev_solution,
               boundary_2d_pair<fp_type> const &horizontal_boundary_pair,
               boundary_2d_pair<fp_type> const &vertical_boundary_pair, fp_type const &time,
               std::pair<fp_type, fp_type> const &weights, std::pair<fp_type, fp_type> const &weight_values,
               container_2d<by_enum::Row, fp_type, container, allocator> &solution) override;

    void solve(container_2d<by_enum::Row, fp_type, container, allocator> const &prev_solution,
               boundary_2d_pair<fp_type> const &horizontal_boundary_pair,
               boundary_2d_pair<fp_type> const &vertical_boundary_pair, fp_type const &time,
               std::pair<fp_type, fp_type> const &weights, std::pair<fp_type, fp_type> const &weight_values,
               std::function<fp_type(fp_type, fp_type)> const &heat_source,
               container_2d<by_enum::Row, fp_type, container, allocator> &solution) override;
};

template <typename fp_type, typename solver, template <typename, typename> typename container, typename allocator>
void heat_douglas_rachford_method<fp_type, solver, container, allocator>::solve(
    container_2d<by_enum::Row, fp_type, container, allocator> const &prev_solution,
    boundary_2d_pair<fp_type> const &horizontal_boundary_pair, boundary_2d_pair<fp_type> const &vertical_boundary_pair,
    fp_type const &time, std::pair<fp_type, fp_type> const &weights, std::pair<fp_type, fp_type> const &weight_values,
    container_2d<by_enum::Row, fp_type, container, allocator> &solution)
{
    typedef implicit_heston_scheme<fp_type, container, allocator> heston_scheme;
    // extract weights:
    const fp_type w_x = weights.first;
    const fp_type x_val = weight_values.first;
    const fp_type w_y = weights.second;
    const fp_type y_val = weight_values.second;
    // 2D container for intermediate solution:
    ccontainer_2d_t inter_solution(coefficients_->space_size_x_, coefficients_->space_size_y_, fp_type{});
    // 1D container for intermediate solution:
    container_t solution_v(coefficients_->space_size_x_, fp_type{});
    // container for current source:
    rcontainer_2d_t curr_source(1, 1, fp_type{});
    // containers for first split solver:
    container_t low(coefficients_->space_size_x_, fp_type{});
    container_t diag(coefficients_->space_size_x_, fp_type{});
    container_t high(coefficients_->space_size_x_, fp_type{});
    container_t rhs(coefficients_->space_size_x_, fp_type{});
    // get the right-hand side of the scheme:
    auto scheme_y = heston_scheme::get_intermediate(weights, weight_values, true);
    fp_type y{};

    for (std::size_t j = 1; j < coefficients_->space_size_y_ - 1; ++j)
    {
        y = grid_2d<fp_type>::value_2(grid_cfg_, j);
        split_0(y, x_val, w_x, low, diag, high);
        scheme_y(coefficients_, grid_cfg_, j, y, prev_solution, curr_source, curr_source, time, rhs);
        solvery_ptr_->set_diagonals(low, diag, high);
        solvery_ptr_->set_rhs(rhs);
        solvery_ptr_->solve(horizontal_boundary_pair, solution_v, time, y);
        inter_solution(j, solution_v);
    }

    // 1D container for final solution:
    solution_v.resize(coefficients_->space_size_y_);
    // containers for second split solver:
    low.resize(coefficients_->space_size_y_);
    diag.resize(coefficients_->space_size_y_);
    high.resize(coefficients_->space_size_y_);
    rhs.resize(coefficients_->space_size_y_);
    // get the right-hand side of the scheme:
    auto scheme_u = heston_scheme::get(weights, weight_values);
    fp_type x{};

    for (std::size_t i = 1; i < coefficients_->space_size_x_ - 1; ++i)
    {
        x = grid_2d<fp_type>::value_1(grid_cfg_, i);
        split_1(x, y_val, w_y, low, diag, high);
        scheme_u(coefficients_, grid_cfg_, i, x, prev_solution, inter_solution, curr_source, time, rhs);
        solveru_ptr_->set_diagonals(low, diag, high);
        solveru_ptr_->set_rhs(rhs);
        solveru_ptr_->solve(vertical_boundary_pair, solution_v, time, x);
        solution(i, solution_v);
    }
}

template <typename fp_type, typename solver, template <typename, typename> typename container, typename allocator>
void heat_douglas_rachford_method<fp_type, solver, container, allocator>::solve(
    container_2d<by_enum::Row, fp_type, container, allocator> const &prev_solution,
    boundary_2d_pair<fp_type> const &horizontal_boundary_pair, boundary_2d_pair<fp_type> const &vertical_boundary_pair,
    fp_type const &time, std::pair<fp_type, fp_type> const &weights, std::pair<fp_type, fp_type> const &weight_values,
    std::function<fp_type(fp_type, fp_type)> const &heat_source,
    container_2d<by_enum::Row, fp_type, container, allocator> &solution)
{
}
} // namespace two_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_HEAT_DOUGLAS_RACHFORD_METHOD_HPP_
