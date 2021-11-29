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

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class implicit_heston_scheme
{
    typedef container<fp_type, allocator> container_t;
    typedef container_2d<by_enum::Row, fp_type, container, allocator> rcontainer_2d_t;

  public:
    static void rhs_intermed_1(general_svc_heston_equation_coefficients_ptr<fp_type> const &cfs,
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
                (gamma * C(x, y) * input(t - 1, y_index - 1)) + ((one - theta) * M(x, y, one) * input(t - 1, y_index)) -
                (gamma * C(x, y) * input(t - 1, y_index + 1)) + (M_tilde(x, y, one) * input(t, y_index - 1)) +
                ((one - W(x, y, one) - (one - theta) * Z(x, y, one)) * input(t, y_index)) +
                (P_tilde(x, y, one) * input(t, y_index + 1)) - (gamma * C(x, y) * input(t + 1, y_index - 1)) +
                ((one - theta) * P(x, y, one) * input(t + 1, y_index)) + (gamma * C(x, y) * input(t + 1, y_index + 1));
        }
    }

    static void rhs_intermed_1_source(general_svc_heston_equation_coefficients_ptr<fp_type> const &cfs,
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
                (gamma * C(x, y) * input(t - 1, y_index - 1)) + ((one - theta) * M(x, y, one) * input(t - 1, y_index)) -
                (gamma * C(x, y) * input(t - 1, y_index + 1)) + (M_tilde(x, y, one) * input(t, y_index - 1)) +
                ((one - W(x, y, one) - (one - theta) * Z(x, y, one)) * input(t, y_index)) +
                (P_tilde(x, y, one) * input(t, y_index + 1)) - (gamma * C(x, y) * input(t + 1, y_index - 1)) +
                ((one - theta) * P(x, y, one) * input(t + 1, y_index)) + (gamma * C(x, y) * input(t + 1, y_index + 1)) +
                (theta * rho * inhom_input_next(t, y_index)) + ((one - theta) * rho * inhom_input(t, y_index));
        }
    }

    static void rhs(general_svc_heston_equation_coefficients_ptr<fp_type> const &cfs,
                    grid_config_2d_ptr<fp_type> const &grid_cfg, std::size_t const &x_index, fp_type const &x,
                    rcontainer_2d_t const &input, rcontainer_2d_t const &inhom_input, fp_type const &time,
                    container_t &solution)
    {
        auto const one = static_cast<fp_type>(1.0);
        auto const &M_tilde = cfs->M_tilde_;
        auto const &P_tilde = cfs->P_tilde_;
        auto const &W = cfs->W_;

        auto const theta = cfs->theta_;

        const std::size_t N = solution.size() - 1;
        fp_type y{};
        for (std::size_t t = 1; t < N; ++t)
        {
            y = grid_2d<fp_type>::value_2(grid_cfg, t);
            solution[t] = (-theta * M_tilde(x, y, one) * input(x_index, t - 1)) +
                          (theta * W(x, y, one) * input(x_index, t)) -
                          (theta * P_tilde(x, y, one) * input(x_index, t + 1)) + inhom_input(x_index, t);
        }
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
    typedef implicit_heston_scheme<fp_type, container, allocator> heston_scheme;

  private:
    // constants:
    const fp_type cone_ = static_cast<fp_type>(1.0);
    // solvers:
    solver solvery_ptr_;
    solver solveru_ptr_;
    // scheme coefficients:
    general_svc_heston_equation_coefficients_ptr<fp_type> coefficients_;
    grid_config_2d_ptr<fp_type> grid_cfg_;
    // containers:
    container_t low_, diag_, high_, rhs_;

    explicit heat_douglas_rachford_method() = delete;

    void initialize(bool is_heat_source_set)
    {
    }

    void split_0(fp_type const &y, container_t &low, container_t &diag, container_t &high)
    {
        fp_type x{};
        for (std::size_t t = 0; t < low.size(); ++t)
        {
            x = grid_2d<fp_type>::value_1(grid_cfg_, t);
            low[t] = (-coefficients_->theta_ * coefficients_->M_(x, y, cone_));
            diag[t] = (cone_ + coefficients_->theta_ * coefficients_->Z_(x, y, cone_));
            high[t] = (-coefficients_->theta_ * coefficients_->P_(x, y, cone_));
        }
    }

    void split_1(fp_type const &x, container_t &low, container_t &diag, container_t &high)
    {
        fp_type y{};
        for (std::size_t t = 0; t < low.size(); ++t)
        {
            y = grid_2d<fp_type>::value_2(grid_cfg_, t);
            low[t] = (-coefficients_->theta_ * coefficients_->M_tilde_(x, y, cone_));
            diag[t] = (cone_ + coefficients_->theta_ * coefficients_->W_(x, y, cone_));
            high[t] = (-coefficients_->theta_ * coefficients_->P_tilde_(x, y, cone_));
        }
    }

  public:
    explicit heat_douglas_rachford_method(solver const &solvery_ptr, solver const &solveru_ptr,
                                          general_svc_heston_equation_coefficients_ptr<fp_type> const &coefficients,
                                          grid_config_2d_ptr<fp_type> const &grid_config, bool is_heat_source_set)
        : solvery_ptr_{solvery_ptr}, solveru_ptr_{solveru_ptr}, coefficients_{coefficients}, grid_cfg_{grid_config}
    {
        initialize(is_heat_source_set);
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
               container_2d<by_enum::Row, fp_type, container, allocator> &solution) override;

    void solve(container_2d<by_enum::Row, fp_type, container, allocator> const &prev_solution,
               boundary_2d_pair<fp_type> const &horizontal_boundary_pair,
               boundary_2d_pair<fp_type> const &vertical_boundary_pair, fp_type const &time,
               std::function<fp_type(fp_type, fp_type)> const &heat_source,
               container_2d<by_enum::Row, fp_type, container, allocator> &solution) override;
};

template <typename fp_type, typename solver, template <typename, typename> typename container, typename allocator>
void heat_douglas_rachford_method<fp_type, solver, container, allocator>::solve(
    container_2d<by_enum::Row, fp_type, container, allocator> const &prev_solution,
    boundary_2d_pair<fp_type> const &horizontal_boundary_pair, boundary_2d_pair<fp_type> const &vertical_boundary_pair,
    fp_type const &time, container_2d<by_enum::Row, fp_type, container, allocator> &solution)
{

    // 2D container for intermediate solution:
    ccontainer_2d_t inter_solution(coefficients_->space_size_x_, coefficients_->space_size_y_, fp_type{});
    // 1D container for intermediate solution:
    container_t solution_v(coefficients_->space_size_x_, fp_type{});
    low_.resize(coefficients_->space_size_x_);
    diag_.resize(coefficients_->space_size_x_);
    high_.resize(coefficients_->space_size_x_);
    rhs_.resize(coefficients_->space_size_x_);
    fp_type y{};
    for (std::size_t j = 1; j < coefficients_->space_size_y_ - 1; ++j)
    {
        y = grid_2d<fp_type>::value_2(grid_cfg_, j);
        split_0(y, low_, diag_, high_);
        heston_scheme::rhs_intermed_1(coefficients_, grid_cfg_, j, y, prev_solution, time, rhs_);
        solvery_ptr_->set_diagonals(low_, diag_, high_);
        solvery_ptr_->set_rhs(rhs_);
        solvery_ptr_->solve(horizontal_boundary_pair, solution_v, time, y);
        inter_solution(j, solution_v);
    }

    // 1D container for final solution:
    solution_v.resize(coefficients_->space_size_y_);
    // containers for second split solver:
    low_.resize(coefficients_->space_size_y_);
    diag_.resize(coefficients_->space_size_y_);
    high_.resize(coefficients_->space_size_y_);
    rhs_.resize(coefficients_->space_size_y_);
    fp_type x{};
    for (std::size_t i = 1; i < coefficients_->space_size_x_ - 1; ++i)
    {
        x = grid_2d<fp_type>::value_1(grid_cfg_, i);
        split_1(x, low_, diag_, high_);
        heston_scheme::rhs(coefficients_, grid_cfg_, i, x, prev_solution, inter_solution, time, rhs_);
        solveru_ptr_->set_diagonals(low_, diag_, high_);
        solveru_ptr_->set_rhs(rhs_);
        solveru_ptr_->solve(vertical_boundary_pair, solution_v, time, x);
        solution(i, solution_v);
    }
}

template <typename fp_type, typename solver, template <typename, typename> typename container, typename allocator>
void heat_douglas_rachford_method<fp_type, solver, container, allocator>::solve(
    container_2d<by_enum::Row, fp_type, container, allocator> const &prev_solution,
    boundary_2d_pair<fp_type> const &horizontal_boundary_pair, boundary_2d_pair<fp_type> const &vertical_boundary_pair,
    fp_type const &time, std::function<fp_type(fp_type, fp_type)> const &heat_source,
    container_2d<by_enum::Row, fp_type, container, allocator> &solution)
{
}
} // namespace two_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_HEAT_DOUGLAS_RACHFORD_METHOD_HPP_
