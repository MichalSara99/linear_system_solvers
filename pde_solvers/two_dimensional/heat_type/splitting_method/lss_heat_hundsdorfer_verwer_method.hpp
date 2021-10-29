#if !defined(_LSS_HEAT_HUNDSDORFER_VERWER_METHOD_HPP_)
#define _LSS_HEAT_HUNDSDORFER_VERWER_METHOD_HPP_

#include <functional>
#include <map>

#include "boundaries/lss_boundary.hpp"
#include "boundaries/lss_dirichlet_boundary.hpp"
#include "boundaries/lss_neumann_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "containers/lss_container_2d.hpp"
#include "containers/lss_container_3d.hpp"
#include "discretization/lss_discretization.hpp"
#include "lss_heat_splitting_method.hpp"
#include "pde_solvers/lss_heat_data_config.hpp"
#include "pde_solvers/lss_heat_solver_config.hpp"
#include "pde_solvers/lss_pde_discretization_config.hpp"
#include "pde_solvers/two_dimensional/heat_type/lss_2d_general_svc_heston_equation_implicit_coefficients.hpp"

namespace lss_pde_solvers
{

namespace two_dimensional
{
using lss_boundary::boundary_2d_pair;
using lss_boundary::boundary_2d_ptr;
using lss_boundary::dirichlet_boundary_2d;
using lss_boundary::neumann_boundary_2d;
using lss_containers::container_2d;
using lss_containers::container_3d;
using lss_enumerations::by_enum;
using lss_utility::coefficient_sevenlet_t;
using lss_utility::function_2d_sevenlet_t;
using lss_utility::pair_t;

template <template <typename, typename> typename container, typename fp_type, typename alloc>
using implicit_heston_scheme_hv_function_t =
    std::function<void(general_svc_heston_equation_implicit_coefficients_ptr<fp_type> const &, std::size_t const &,
                       fp_type const &, container_2d<by_enum::Row, fp_type, container, alloc> const &,
                       container_2d<by_enum::Row, fp_type, container, alloc> const &,
                       container_2d<by_enum::Row, fp_type, container, alloc> const &, fp_type const &,
                       container<fp_type, alloc> &)>;

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class implicit_heston_scheme_hv
{
    typedef container<fp_type, allocator> container_t;
    typedef container_2d<by_enum::Row, fp_type, container, allocator> rcontainer_2d_t;
    typedef implicit_heston_scheme_hv_function_t<container, fp_type, allocator> scheme_function_t;

  public:
    static scheme_function_t const get_intermediate_1(bool is_homogeneus)
    {
        const fp_type one = static_cast<fp_type>(1.0);

        auto scheme_fun_h = [=](general_svc_heston_equation_implicit_coefficients_ptr<fp_type> const &cfs,
                                std::size_t const &y_index, fp_type const &y, rcontainer_2d_t const &input,
                                rcontainer_2d_t const &inhom_input, rcontainer_2d_t const &inhom_input_next,
                                fp_type const &time, container_t &solution) {
            auto M = cfs->M_;
            auto M_tilde = cfs->M_tilde_;
            auto P = cfs->P_;
            auto P_tilde = cfs->P_tilde_;
            auto Z = cfs->Z_;
            auto W = cfs->W_;
            auto C = cfs->C_;

            auto const gamma = cfs->gamma_;
            auto const theta = cfs->theta_;

            auto const start_x = cfs->rangex_.lower();
            auto const h_1 = cfs->h_1_;

            const std::size_t N = solution.size() - 1;
            fp_type x{};
            for (std::size_t t = 1; t < N; ++t)
            {
                x = start_x + static_cast<fp_type>(t) * h_1;
                solution[t] =
                    (gamma * C(x, y) * input(t - 1, y_index - 1)) +
                    ((one - theta) * M(x, y, one) * input(t - 1, y_index)) -
                    (gamma * C(x, y) * input(t - 1, y_index + 1)) + (M_tilde(x, y, one) * input(t, y_index - 1)) +
                    ((one - W(x, y, one) - (one - theta) * Z(x, y, one)) * input(t, y_index)) +
                    (P_tilde(x, y, one) * input(t, y_index + 1)) - (gamma * C(x, y) * input(t + 1, y_index - 1)) +
                    ((one - theta) * P(x, y, one) * input(t + 1, y_index)) +
                    (gamma * C(x, y) * input(t + 1, y_index + 1));
            }
        };
        auto scheme_fun_nh = [=](general_svc_heston_equation_implicit_coefficients_ptr<fp_type> const &cfs,
                                 std::size_t const &y_index, fp_type const &y, rcontainer_2d_t const &input,
                                 rcontainer_2d_t const &inhom_input, rcontainer_2d_t const &inhom_input_next,
                                 fp_type const &time, container_t &solution) {
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

            auto const start_x = cfs->rangex_.lower();
            auto const h_1 = cfs->h_1_;

            const std::size_t N = solution.size() - 1;
            fp_type x{};
            for (std::size_t t = 1; t < N; ++t)
            {
                x = start_x + static_cast<fp_type>(t) * h_1;
                solution[t] =
                    (gamma * C(x, y) * input(t - 1, y_index - 1)) +
                    ((one - theta) * M(x, y, one) * input(t - 1, y_index)) -
                    (gamma * C(x, y) * input(t - 1, y_index + 1)) + (M_tilde(x, y, one) * input(t, y_index - 1)) +
                    ((one - W(x, y, one) - (one - theta) * Z(x, y, one)) * input(t, y_index)) +
                    (P_tilde(x, y, one) * input(t, y_index + 1)) - (gamma * C(x, y) * input(t + 1, y_index - 1)) +
                    ((one - theta) * P(x, y, one) * input(t + 1, y_index)) +
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

    static scheme_function_t const get_intermediate_2()
    {
        const fp_type one = static_cast<fp_type>(1.0);

        auto scheme_fun = [=](general_svc_heston_equation_implicit_coefficients_ptr<fp_type> const &cfs,
                              std::size_t const &x_index, fp_type const &x, rcontainer_2d_t const &input,
                              rcontainer_2d_t const &inhom_input, rcontainer_2d_t const &inhom_input_next,
                              fp_type const &time, container_t &solution) {
            auto const M_tilde = cfs->M_tilde_;
            auto const P_tilde = cfs->P_tilde_;
            auto const W = cfs->W_;

            auto const theta = cfs->theta_;

            auto const start_y = cfs->rangey_.lower();
            auto const h_2 = cfs->h_2_;

            const std::size_t N = solution.size() - 1;
            fp_type y{};
            for (std::size_t t = 1; t < N; ++t)
            {
                y = start_y + static_cast<fp_type>(t) * h_2;
                solution[t] = (-theta * M_tilde(x, y, one) * input(x_index, t - 1)) +
                              (theta * W(x, y, one) * input(x_index, t)) -
                              (theta * P_tilde(x, y, one) * input(x_index, t + 1)) + inhom_input(x_index, t);
            }
        };

        return scheme_fun;
    }

    static scheme_function_t const get_intermediate_3()
    {
        const fp_type one = static_cast<fp_type>(1.0);

        auto scheme_fun = [=](general_svc_heston_equation_implicit_coefficients_ptr<fp_type> const &cfs,
                              std::size_t const &y_index, fp_type const &y, rcontainer_2d_t const &input,
                              rcontainer_2d_t const &inhom_input, rcontainer_2d_t const &inhom_input_next,
                              fp_type const &time, container_t &solution) {
            auto M = cfs->M_;
            auto P = cfs->P_;
            auto Z = cfs->Z_;
            auto C = cfs->C_;
            auto W = cfs->W_;
            auto M_tilde = cfs->M_tilde_;
            auto P_tilde = cfs->P_tilde_;

            auto const gamma = cfs->gamma_;
            auto const theta = cfs->theta_;
            auto const zeta = cfs->zeta_;

            auto const start_x = cfs->rangex_.lower();
            auto const h_1 = cfs->h_1_;

            const std::size_t N = solution.size() - 1;
            fp_type x{};
            for (std::size_t t = 1; t < N; ++t)
            {
                x = start_x + static_cast<fp_type>(t) * h_1;
                solution[t] =
                    (-theta * M(x, y, one) * inhom_input(t - 1, y_index)) +
                    ((one + theta * Z(x, y, one)) * inhom_input(t, y_index)) -
                    (theta * P(x, y, one) * inhom_input(t + 1, y_index)) +
                    (theta * M(x, y, one) * input(t - 1, y_index)) - (theta * Z(x, y, one) * input(t, y_index)) +
                    (theta * P(x, y, one) * input(t + 1, y_index)) +
                    (zeta * gamma * C(x, y) *
                     ((inhom_input_next(t + 1, y_index + 1) - input(t + 1, y_index + 1)) -
                      (inhom_input_next(t + 1, y_index - 1) - input(t + 1, y_index - 1)) -
                      (inhom_input_next(t - 1, y_index + 1) - input(t - 1, y_index + 1)) +
                      (inhom_input_next(t - 1, y_index - 1) - input(t - 1, y_index - 1)))) +
                    (zeta * (M(x, y, one) * (inhom_input_next(t - 1, y_index) - input(t - 1, y_index)) -
                             (W(x, y, one) + Z(x, y, one)) * (inhom_input_next(t, y_index) - input(t, y_index)) +
                             P(x, y, one) * (inhom_input_next(t + 1, y_index) - input(t + 1, y_index)) +
                             M_tilde(x, y, one) * (inhom_input_next(t, y_index - 1) - input(t, y_index - 1)) +
                             P_tilde(x, y, one) * (inhom_input_next(t, y_index + 1) - input(t, y_index + 1))));
            }
        };

        return scheme_fun;
    }

    static scheme_function_t const get_intermediate_4()
    {
        const fp_type one = static_cast<fp_type>(1.0);

        auto scheme_fun = [=](general_svc_heston_equation_implicit_coefficients_ptr<fp_type> const &cfs,
                              std::size_t const &y_index, fp_type const &y, rcontainer_2d_t const &input,
                              rcontainer_2d_t const &inhom_input, rcontainer_2d_t const &inhom_input_next,
                              fp_type const &time, container_t &solution) {
            auto M = cfs->M_;
            auto P = cfs->P_;
            auto Z = cfs->Z_;

            auto const theta = cfs->theta_;

            auto const start_x = cfs->rangex_.lower();
            auto const h_1 = cfs->h_1_;

            const std::size_t N = solution.size() - 1;
            fp_type x{};
            for (std::size_t t = 1; t < N; ++t)
            {
                x = start_x + static_cast<fp_type>(t) * h_1;
                solution[t] = (-theta * M(x, y, one) * input(t - 1, y_index)) +
                              (theta * Z(x, y, one) * input(t, y_index)) -
                              (theta * P(x, y, one) * input(t + 1, y_index)) + inhom_input(t, y_index);
            }
        };

        return scheme_fun;
    }

    static scheme_function_t const get()
    {

        const fp_type one = static_cast<fp_type>(1.0);

        auto scheme_fun = [=](general_svc_heston_equation_implicit_coefficients_ptr<fp_type> const &cfs,
                              std::size_t const &x_index, fp_type const &x, rcontainer_2d_t const &input,
                              rcontainer_2d_t const &inhom_input, rcontainer_2d_t const &inhom_input_next,
                              fp_type const &time, container_t &solution) {
            auto const M_tilde = cfs->M_tilde_;
            auto const P_tilde = cfs->P_tilde_;
            auto const W = cfs->W_;

            auto const theta = cfs->theta_;

            auto const start_y = cfs->rangey_.lower();
            auto const h_2 = cfs->h_2_;

            const std::size_t N = solution.size() - 1;
            fp_type y{};
            for (std::size_t t = 1; t < N; ++t)
            {
                y = start_y + static_cast<fp_type>(t) * h_2;
                solution[t] = (-theta * M_tilde(x, y, one) * input(x_index, t - 1)) +
                              (theta * W(x, y, one) * input(x_index, t)) -
                              (theta * P_tilde(x, y, one) * input(x_index, t + 1)) + inhom_input(x_index, t);
            }
        };

        return scheme_fun;
    }
};

/**
    heat_hundsdorfer_verwer_method object
 */
template <typename fp_type, typename solver, template <typename, typename> typename container = std::vector,
          typename allocator = std::allocator<fp_type>>
class heat_hundsdorfer_verwer_method : public heat_splitting_method<fp_type, container, allocator>
{
    typedef container_2d<by_enum::Row, fp_type, container, allocator> rcontainer_2d_t;
    typedef container_2d<by_enum::Column, fp_type, container, allocator> ccontainer_2d_t;
    typedef container<fp_type, allocator> container_t;

  private:
    // solvers:
    solver solvery_ptr_;
    solver solveru_ptr_;
    // scheme coefficients:
    general_svc_heston_equation_implicit_coefficients_ptr<fp_type> coefficients_;

    explicit heat_hundsdorfer_verwer_method() = delete;

    void initialize()
    {
    }

    void split_0(fp_type const &y, container_t &low, container_t &diag, container_t &high)
    {
        fp_type x{};
        const fp_type start_x = coefficients_->rangex_.lower();
        const fp_type one = static_cast<fp_type>(1.0);
        for (std::size_t t = 0; t < low.size(); ++t)
        {
            x = start_x + static_cast<fp_type>(t) * coefficients_->h_1_;
            low[t] = (-coefficients_->theta_ * coefficients_->M_(x, y, one));
            diag[t] = (one + coefficients_->theta_ * coefficients_->Z_(x, y, one));
            high[t] = (-coefficients_->theta_ * coefficients_->P_(x, y, one));
        }
    }

    void split_1(fp_type const &x, container_t &low, container_t &diag, container_t &high)
    {
        fp_type y{};
        fp_type start_y = coefficients_->rangey_.lower();
        const fp_type one = static_cast<fp_type>(1.0);
        for (std::size_t t = 0; t < low.size(); ++t)
        {
            y = start_y + static_cast<fp_type>(t) * coefficients_->h_2_;
            low[t] = (-coefficients_->theta_ * coefficients_->M_tilde_(x, y, one));
            diag[t] = (one + coefficients_->theta_ * coefficients_->W_(x, y, one));
            high[t] = (-coefficients_->theta_ * coefficients_->P_tilde_(x, y, one));
        }
    }

  public:
    explicit heat_hundsdorfer_verwer_method(
        solver const &solvery_ptr, solver const &solveru_ptr,
        general_svc_heston_equation_implicit_coefficients_ptr<fp_type> const &coefficients)
        : solvery_ptr_{solvery_ptr}, solveru_ptr_{solveru_ptr}, coefficients_{coefficients}
    {
        initialize();
    }

    ~heat_hundsdorfer_verwer_method()
    {
    }

    heat_hundsdorfer_verwer_method(heat_hundsdorfer_verwer_method const &) = delete;
    heat_hundsdorfer_verwer_method(heat_hundsdorfer_verwer_method &&) = delete;
    heat_hundsdorfer_verwer_method &operator=(heat_hundsdorfer_verwer_method const &) = delete;
    heat_hundsdorfer_verwer_method &operator=(heat_hundsdorfer_verwer_method &&) = delete;

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
void heat_hundsdorfer_verwer_method<fp_type, solver, container, allocator>::solve(
    container_2d<by_enum::Row, fp_type, container, allocator> const &prev_solution,
    boundary_2d_pair<fp_type> const &horizontal_boundary_pair, boundary_2d_pair<fp_type> const &vertical_boundary_pair,
    fp_type const &time, std::pair<fp_type, fp_type> const &weights, std::pair<fp_type, fp_type> const &weight_values,
    container_2d<by_enum::Row, fp_type, container, allocator> &solution)
{
    typedef implicit_heston_scheme_hv<fp_type, container, allocator> heston_scheme;

    // 2D container for intermediate solution Y_1:
    ccontainer_2d_t inter_solution_1(coefficients_->space_size_x_, coefficients_->space_size_y_, fp_type{});
    // 1D container for intermediate solution Y_1:
    container_t solution_v(coefficients_->space_size_x_, fp_type{});
    // container for current source:
    rcontainer_2d_t curr_source(1, 1, fp_type{});
    // containers for first split solver:
    container_t low(coefficients_->space_size_x_, fp_type{});
    container_t diag(coefficients_->space_size_x_, fp_type{});
    container_t high(coefficients_->space_size_x_, fp_type{});
    container_t rhs(coefficients_->space_size_x_, fp_type{});
    // get the right-hand side of the scheme:
    auto scheme_y1 = heston_scheme::get_intermediate_1(true);
    fp_type y{};
    fp_type start_y{coefficients_->rangey_.lower()};
    for (std::size_t j = 1; j < coefficients_->space_size_y_ - 1; ++j)
    {
        y = start_y + static_cast<fp_type>(j) * coefficients_->h_2_;
        split_0(y, low, diag, high);
        scheme_y1(coefficients_, j, y, prev_solution, curr_source, curr_source, time, rhs);
        solvery_ptr_->set_diagonals(low, diag, high);
        solvery_ptr_->set_rhs(rhs);
        solvery_ptr_->solve(horizontal_boundary_pair, solution_v, time, y);
        inter_solution_1(j, solution_v);
    }

    // 2D container for intermediate solution Y_2:
    rcontainer_2d_t inter_solution_2(coefficients_->space_size_x_, coefficients_->space_size_y_, fp_type{});
    // 1D container for intermediate solution Y_2:
    solution_v.resize(coefficients_->space_size_y_);
    // containers for second split solver:
    low.resize(coefficients_->space_size_y_);
    diag.resize(coefficients_->space_size_y_);
    high.resize(coefficients_->space_size_y_);
    rhs.resize(coefficients_->space_size_y_);
    // get the right-hand side of the scheme:
    auto scheme_y2 = heston_scheme::get_intermediate_2();
    fp_type x{};
    fp_type start_x{coefficients_->rangex_.lower()};

    for (std::size_t i = 1; i < coefficients_->space_size_x_ - 1; ++i)
    {
        x = start_x + static_cast<fp_type>(i) * coefficients_->h_1_;
        split_1(x, low, diag, high);
        scheme_y2(coefficients_, i, x, prev_solution, inter_solution_1, curr_source, time, rhs);
        solveru_ptr_->set_diagonals(low, diag, high);
        solveru_ptr_->set_rhs(rhs);
        solveru_ptr_->solve(vertical_boundary_pair, solution_v, time, x);
        inter_solution_2(i, solution_v);
    }

    // 2D container for intermediate solution Y_3:
    ccontainer_2d_t inter_solution_3(coefficients_->space_size_x_, coefficients_->space_size_y_, fp_type{});
    // 1D container for intermediate solution Y_3:
    solution_v.resize(coefficients_->space_size_x_);
    // containers for second split solver:
    low.resize(coefficients_->space_size_x_);
    diag.resize(coefficients_->space_size_x_);
    high.resize(coefficients_->space_size_x_);
    rhs.resize(coefficients_->space_size_x_);
    // get the right-hand side of the scheme:
    auto scheme_y3 = heston_scheme::get_intermediate_3();
    for (std::size_t j = 1; j < coefficients_->space_size_y_ - 1; ++j)
    {
        y = start_y + static_cast<fp_type>(j) * coefficients_->h_2_;
        split_0(y, low, diag, high);
        scheme_y3(coefficients_, j, y, prev_solution, inter_solution_1, inter_solution_2, time, rhs);
        solvery_ptr_->set_diagonals(low, diag, high);
        solvery_ptr_->set_rhs(rhs);
        solvery_ptr_->solve(horizontal_boundary_pair, solution_v, time, y);
        inter_solution_3(j, solution_v);
    }

    // 1D container for final solution:
    solution_v.resize(coefficients_->space_size_y_);
    // containers for second split solver:
    low.resize(coefficients_->space_size_y_);
    diag.resize(coefficients_->space_size_y_);
    high.resize(coefficients_->space_size_y_);
    rhs.resize(coefficients_->space_size_y_);
    // get the right-hand side of the scheme:
    auto scheme_u = heston_scheme::get();
    for (std::size_t i = 1; i < coefficients_->space_size_x_ - 1; ++i)
    {
        x = start_x + static_cast<fp_type>(i) * coefficients_->h_1_;
        split_1(x, low, diag, high);
        scheme_u(coefficients_, i, x, prev_solution, inter_solution_3, curr_source, time, rhs);
        solveru_ptr_->set_diagonals(low, diag, high);
        solveru_ptr_->set_rhs(rhs);
        solveru_ptr_->solve(vertical_boundary_pair, solution_v, time, x);
        solution(i, solution_v);
    }
}

template <typename fp_type, typename solver, template <typename, typename> typename container, typename allocator>
void heat_hundsdorfer_verwer_method<fp_type, solver, container, allocator>::solve(
    container_2d<by_enum::Row, fp_type, container, allocator> const &prev_solution,
    boundary_2d_pair<fp_type> const &horizontal_boundary_pair, boundary_2d_pair<fp_type> const &vertical_boundary_pair,
    fp_type const &time, std::pair<fp_type, fp_type> const &weights, std::pair<fp_type, fp_type> const &weight_values,
    std::function<fp_type(fp_type, fp_type)> const &heat_source,
    container_2d<by_enum::Row, fp_type, container, allocator> &solution)
{
}
} // namespace two_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_HEAT_HUNDSDORFER_VERWER_METHOD_HPP_
