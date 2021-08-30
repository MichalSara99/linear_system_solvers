#if !defined(_LSS_HEAT_DOUGLAS_RACHFORD_METHOD_HPP_)
#define _LSS_HEAT_DOUGLAS_RACHFORD_METHOD_HPP_

#include <functional>
#include <map>

#include "boundaries/lss_boundary.hpp"
#include "boundaries/lss_dirichlet_boundary.hpp"
#include "boundaries/lss_neumann_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "containers/lss_container_2d.hpp"
#include "containers/lss_container_3d.hpp"
//#include "lss_general_svc_heat_equation_explicit_kernel.hpp"
//#include "lss_general_svc_heat_equation_implicit_kernel.hpp"
#include "discretization/lss_discretization.hpp"
#include "pde_solvers/lss_heat_data_config.hpp"
#include "pde_solvers/lss_heat_solver_config.hpp"
#include "pde_solvers/lss_pde_discretization_config.hpp"

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
using lss_utility::function_sevenlet_t;
using lss_utility::pair_t;

template <template <typename, typename> typename container, typename fp_type, typename alloc>
using implicit_heston_scheme_function_t = std::function<void(
    function_sevenlet_t<fp_type> const &, coefficient_sevenlet_t<fp_type> const &, std::size_t const &, fp_type const &,
    pair_t<fp_type> const &, container_2d<by_enum::Row, fp_type, container, alloc> const &,
    container_2d<by_enum::Row, fp_type, container, alloc> const &,
    container_2d<by_enum::Row, fp_type, container, alloc> const &, fp_type const &, container<fp_type, alloc> &)>;

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class implicit_heston_scheme
{
    typedef container<fp_type, allocator> container_t;
    typedef container_2d<by_enum::Row, fp_type, container, allocator> rcontainer_2d_t;
    typedef implicit_heston_scheme_function_t<container, fp_type, allocator> scheme_function_t;

  public:
    static scheme_function_t const get_intermediate(range<fp_type> const &rangex, fp_type const &theta,
                                                    bool is_homogeneus)
    {
        const fp_type two = static_cast<fp_type>(2.0);
        const fp_type one = static_cast<fp_type>(1.0);

        auto scheme_fun_h = [=](function_sevenlet_t<fp_type> const &funcs,
                                coefficient_sevenlet_t<fp_type> const &coefficients, std::size_t const &y_index,
                                fp_type const &y, pair_t<fp_type> const &steps, rcontainer_2d_t const &input,
                                rcontainer_2d_t const &inhom_input, rcontainer_2d_t const &inhom_input_next,
                                fp_type const &time, container_t &solution) {
            auto const &M = std::get<0>(funcs);
            auto const &M_tilde = std::get<1>(funcs);
            auto const &P = std::get<2>(funcs);
            auto const &P_tilde = std::get<3>(funcs);
            auto const &Z = std::get<4>(funcs);
            auto const &W = std::get<5>(funcs);
            auto const &C = std::get<6>(funcs);

            auto const &gamma = std::get<2>(coefficients);
            auto const &theta = std::get<6>(coefficients);

            auto const start_x = rangex.lower();
            auto const h_1 = steps.first;

            const std::size_t N = solution.size() - 1;
            fp_type x{};
            for (std::size_t t = 1; t < N; ++t)
            {
                x = start_x + static_cast<fp_type>(t) * h_1;
                solution[t] =
                    (gamma * C(x, y) * input(t - 1, y_index)) + ((one - theta) * M(x, y) * input(t - 1, y_index)) -
                    (gamma * C(x, y) * input(t - 1, y_index + 1)) + (M_tilde(x, y) * input(t, y_index - 1)) +
                    ((one - W(x, y) - (one - theta) * Z(x, y)) * input(t, y_index)) +
                    (P_tilde(x, y) * input(t, y_index + 1)) - (gamma * C(x, y) * input(t + 1, y_index - 1)) +
                    ((one - theta) * P(x, y) * input(t + 1, y_index)) + (gamma * C(x, y) * input(t + 1, y_index + 1));
            }
        };
        auto scheme_fun_nh = [=](function_sevenlet_t<fp_type> const &funcs,
                                 coefficient_sevenlet_t<fp_type> const &coefficients, std::size_t const &y_index,
                                 fp_type const &y, pair_t<fp_type> const &steps, rcontainer_2d_t const &input,
                                 rcontainer_2d_t const &inhom_input, rcontainer_2d_t const &inhom_input_next,
                                 fp_type const &time, container_t &solution) {
            auto const &M = std::get<0>(funcs);
            auto const &M_tilde = std::get<1>(funcs);
            auto const &P = std::get<2>(funcs);
            auto const &P_tilde = std::get<3>(funcs);
            auto const &Z = std::get<4>(funcs);
            auto const &W = std::get<5>(funcs);
            auto const &C = std::get<6>(funcs);

            auto const &gamma = std::get<2>(coefficients);
            auto const &rho = std::get<5>(coefficients);
            auto const &theta = std::get<6>(coefficients);

            auto const start_x = rangex.lower();
            auto const h_1 = steps.first;

            const std::size_t N = solution.size() - 1;
            fp_type x{};
            for (std::size_t t = 1; t < N; ++t)
            {
                x = start_x + static_cast<fp_type>(t) * h_1;
                solution[t] =
                    (gamma * C(x, y) * input(t - 1, y_index)) + ((one - theta) * M(x, y) * input(t - 1, y_index)) -
                    (gamma * C(x, y) * input(t - 1, y_index + 1)) + (M_tilde(x, y) * input(t, y_index - 1)) +
                    ((one - W(x, y) - (one - theta) * Z(x, y)) * input(t, y_index)) +
                    (P_tilde(x, y) * input(t, y_index + 1)) - (gamma * C(x, y) * input(t + 1, y_index - 1)) +
                    ((one - theta) * P(x, y) * input(t + 1, y_index)) + (gamma * C(x, y) * input(t + 1, y_index + 1)) +
                    (theta * rho * inhom_input_next(t, y_index)) + ((one - theta) * rho * inhom_input(t, y_index));
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

    static scheme_function_t const get(range<fp_type> const &rangey, fp_type const &theta)
    {
        const fp_type two = static_cast<fp_type>(2.0);
        const fp_type one = static_cast<fp_type>(1.0);

        auto scheme_fun = [=](function_sevenlet_t<fp_type> const &funcs,
                              coefficient_sevenlet_t<fp_type> const &coefficients, std::size_t const &x_index,
                              fp_type const &x, pair_t<fp_type> const &steps, rcontainer_2d_t const &input,
                              rcontainer_2d_t const &inhom_input, rcontainer_2d_t const &inhom_input_next,
                              fp_type const &time, container_t &solution) {
            auto const &M_tilde = std::get<1>(funcs);
            auto const &P_tilde = std::get<3>(funcs);
            auto const &W = std::get<5>(funcs);

            auto const &theta = std::get<6>(coefficients);

            auto const start_y = rangey.lower();
            auto const h_2 = steps.second;

            const std::size_t N = solution.size() - 1;
            fp_type y{};
            for (std::size_t t = 1; t < N; ++t)
            {
                y = start_y + static_cast<fp_type>(t) * h_2;
                solution[t] = (-theta * M_tilde(x, y) * input(x_index, t - 1)) + (theta * W(x, y) * input(x_index, t)) -
                              (theta * P_tilde(x, y) * input(x_index, t + 1)) + inhom_input(x_index, t);
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
class heat_douglas_rachford_method
{
    typedef container_2d<by_enum::Row, fp_type, container, allocator> rcontainer_2d_t;
    typedef container_2d<by_enum::Column, fp_type, container, allocator> ccontainer_2d_t;
    typedef container<fp_type, allocator> container_t;

  private:
    // solvers:
    solver solver0_ptr_;
    solver solver1_ptr_;
    // scheme coefficients:
    fp_type alpha_, beta_, gamma_, delta_, ni_, rho_, h_1_, h_2_;
    std::size_t space_size_x_, space_size_y_;
    range<fp_type> rangex_, rangey_;
    // theta variable:
    fp_type theta_;
    // functional coefficients:
    std::function<fp_type(fp_type, fp_type)> M_;
    std::function<fp_type(fp_type, fp_type)> M_tilde_;
    std::function<fp_type(fp_type, fp_type)> P_;
    std::function<fp_type(fp_type, fp_type)> P_tilde_;
    std::function<fp_type(fp_type, fp_type)> S_;
    std::function<fp_type(fp_type, fp_type)> Z_;
    std::function<fp_type(fp_type, fp_type)> W_;
    std::function<fp_type(fp_type, fp_type)> C_;
    function_sevenlet_t fun_sevenlet_;
    // constant coefficients:
    coefficient_sevenlet_t coeff_sevenlet_;
    // steps pair:
    std::pair<fp_type, fp_type> steps_;

    explicit heat_douglas_rachford_method() = delete;

    void initialize(pde_discretization_config_2d_ptr<fp_type> const &discretization_config)
    {
        // get space ranges:
        const auto &spaces = discretization_config->space_range();
        // across X:
        rangex_ = std::get<0>(spaces);
        // across Y:
        rangey_ = std::get<1>(spaces);
        // get space steps:
        const auto &hs = discretization_config->space_step();
        // across X:
        const fp_type h_1 = std::get<0>(hs);
        // across Y:
        const fp_type h_2 = std::get<1>(hs);
        // time step:
        const fp_type k = discretization_config->time_step();
        // size of spaces discretization:
        const auto &space_sizes = discretization_config->number_of_space_points();
        space_size_x_ = std::get<0>(space_sizes);
        space_size_y_ = std::get<1>(space_sizes);
        const fp_type one = static_cast<fp_type>(1.0);
        const fp_type two = static_cast<fp_type>(2.0);
        const fp_type half = static_cast<fp_type>(0.5);
        const fp_type quarter = static_cast<fp_type>(0.25);
        // calculate scheme coefficients:
        alpha_ = k / (h_1 * h_1);
        beta_ = k / (h_2 * h_2);
        gamma_ = quarter * k / (h_1 * h_2);
        delta_ = half * k / h_1;
        ni_ = half * k / h_2;
        rho_ = k;
        h_1_ = h_1;
        h_2_ = h_2;
        coeff_sevenlet_ = std::make_tuple(alpha_, beta_, gamma_, delta_, ni_, rho_, theta_);
        steps_ = std::make_pair(h_1_, h_2_);
    }

    void initialize_coefficients(heat_data_config_2d_ptr<fp_type> const &heat_data_config)
    {
        // save coefficients locally:
        auto const &a = heat_data_config->a_coefficient();
        auto const &b = heat_data_config->b_coefficient();
        auto const &c = heat_data_config->c_coefficient();
        auto const &d = heat_data_config->d_coefficient();
        auto const &e = heat_data_config->e_coefficient();
        auto const &f = heat_data_config->f_coefficient();

        const fp_type one = static_cast<fp_type>(1.0);
        const fp_type two = static_cast<fp_type>(2.0);
        const fp_type half = static_cast<fp_type>(0.5);

        M_ = [&](fp_type x, fp_type y) { return (alpha_ * a(x, y) - delta_ * d(x, y)); };
        M_tilde_ = [&](fp_type x, fp_type y) { return (beta_ * b(x, y) - ni_ * e(x, y)); };
        P_ = [&](fp_type x, fp_type y) { return (alpha_ * a(x, y) + delta_ * d(x, y)); };
        P_tilde_ = [&](fp_type x, fp_type y) { return (beta_ * b(x, y) + ni_ * e(x, y)); };
        S_ = [&](fp_type x, fp_type y) {
            return (one - two * (alpha_ * a(x, y) + beta_ * b(x, y) - half * rho_ * f(x, y)));
        };
        Z_ = [&](fp_type x, fp_type y) { return (two * alpha_ * a(x, y) - half * rho_ * f(x, y)); };
        W_ = [&](fp_type x, fp_type y) { return (two * beta_ * b(x, y) - half * rho_ * f(x, y)); };
        C_ = [&](fp_type x, fp_type y) { return c(x, y); };
        fun_sevenlet_ = std::make_tuple(M_, M_tilde_, P_, P_tilde_, Z_, W_, C_);
    }

    void split_0(fp_type const &y, container_t &low, container_t &diag, container_t &high)
    {
        fp_type x{};
        fp_type start_x = rangex_.lower();
        const fp_type one = static_cast<fp_type>(1.0);
        for (std::size_t t = 0; t < low.size(); ++t)
        {
            x = start_x + static_cast<fp_type>(t) * h_1_;
            low[t] = (-theta_ * M_(x, y));
            diag[t] = (one + theta_ * Z_(x, y));
            high[t] = (-theta_ * P_(x, y));
        }
    }

    void split_1(fp_type const &x, container_t &low, container_t &diag, container_t &high)
    {
        fp_type y{};
        fp_type start_y = rangey_.lower();
        const fp_type one = static_cast<fp_type>(1.0);
        for (std::size_t t = 0; t < low.size(); ++t)
        {
            y = start_y + static_cast<fp_type>(t) * h_2_;
            low[t] = (-theta_ * M_tilde_(x, y));
            diag[t] = (one + theta_ * W_(x, y));
            high[t] = (-theta_ * P_tilde_(x, y));
        }
    }

  public:
    explicit heat_douglas_rachford_method(solver const &solver0_ptr, solver const &solver1_ptr,
                                          heat_data_config_2d_ptr<fp_type> const &heat_data_config,
                                          pde_discretization_config_2d_ptr<fp_type> const &discretization_config,
                                          fp_type const &theta)
        : solver0_ptr_{solver0_ptr}, solver1_ptr_{solver1_ptr}, heat_data_cfg_{heat_data_config},
          discretization_cfg_{discretization_config}, theta_{theta}
    {
        initialize(discretization_config);
        initialize_coefficients(heat_data_config);
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
               container_2d<by_enum::Row, fp_type, container, allocator> &solution);

    void solve(container_2d<by_enum::Row, fp_type, container, allocator> const &prev_solution,
               boundary_2d_pair<fp_type> const &horizontal_boundary_pair,
               boundary_2d_pair<fp_type> const &vertical_boundary_pair, fp_type const &time,
               std::function<fp_type(fp_type, fp_type)> const &heat_source,
               container_2d<by_enum::Row, fp_type, container, allocator> &solution);
};

template <typename fp_type, typename solver, template <typename, typename> typename container, typename allocator>
void heat_douglas_rachford_method<fp_type, solver, container, allocator>::solve(
    container_2d<by_enum::Row, fp_type, container, allocator> const &prev_solution,
    boundary_2d_pair<fp_type> const &horizontal_boundary_pair, boundary_2d_pair<fp_type> const &vertical_boundary_pair,
    fp_type const &time, container_2d<by_enum::Row, fp_type, container, allocator> &solution)
{
    typedef implicit_heston_scheme<fp_type, container, allocator> heston_scheme;

    // make it a column solution
    ccontainer_2d_t csolution(solution);
    // 2D container for intermediate solution:
    rcontainer_2d_t inter_solution(space_size_y_, space_size_x_, fp_type{});
    // 1D container for intermediate solution:
    container_t solution_0(space_size_x_, fp_type{});
    // container for current source:
    rcontainer_2d_t curr_source(1, 1, fp_type{});
    // containers for first split solver:
    container_t low(space_size_x_, fp_type{});
    container_t diag(space_size_x_, fp_type{});
    container_t high(space_size_x_, fp_type{});
    container_t rhs(space_size_x_, fp_type{});
    // get the right-hand side of the scheme:
    auto scheme_0 = heston_scheme::get_intermediate(rangex_, theta_, true);
    fp_type y{};
    fp_type start_y{rangey_.lower()};
    for (std::size_t j = 1; j < space_size_y_ - 1; ++j)
    {
        y = start_y + static_cast<fp_type>(j) * h_2_;
        split_0(y, low, diag, high);
        scheme_0(fun_sevenlet_, coeff_sevenlet_, j, y, steps_, prev_solution, curr_source, curr_source, time, rhs);
        solver0_ptr_->set_diagonals(low, diag, high);
        solver0_ptr_->set_rhs(rhs);
        solver0_ptr_->solve(vertical_boundary_pair, solution_0, time, y);
        inter_solution(j, solution_0);
    }
    // 1D container for final solution:
    solution_0.resize(space_size_y_);
    // containers for second split solver:
    low.resize(space_size_y_);
    diag.resize(space_size_y_);
    high.resize(space_size_y_);
    rhs.resize(space_size_y_);
    // get the right-hand side of the scheme:
    auto scheme_1 = heston_scheme::get(rangey_, theta_);
    fp_type x{};
    fp_type start_x{rangex_.lower()};
    // this building of boundaries can be taken out of here - does not have to be done every time
    // auto const &lower = [&](fp_type x) {
    //    const std::size_t i = static_cast<std::size_t>((x - start_x) / h_1_);
    //    return solution(0, i);
    //};
    // auto const &upper = [&](fp_type x) {
    //    const std::size_t i = static_cast<std::size_t>((x - start_x) / h_1_);
    //    return solution(space_size_y_ - 1, i);
    //};
    // auto const &horizontal_boundary_pair = std::make_pair<dirichlet_boundary_2d<fp_type>>(lower, upper);
    for (std::size_t i = 0; i < space_size_x_; ++i)
    {
        x = start_x + static_cast<fp_type>(i) * h_1_;
        split_1(x, low, diag, high);
        scheme_1(fun_sevenlet_, coeff_sevenlet_, i, x, steps_, prev_solution, inter_solution, curr_source, time, rhs);
        solver1_ptr_->set_diagonals(low, diag, high);
        solver1_ptr_->set_rhs(rhs);
        solver1_ptr_->solve(horizontal_boundary_pair, solution_0, time, x);
        csolution(i, solution_0);
    }
    solution = csolution;
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
