#if !defined(_LSS_2D_GENERAL_SVC_HESTON_EQUATION_EXPLICIT_BOUNDARY_HPP_)
#define _LSS_2D_GENERAL_SVC_HESTON_EQUATION_EXPLICIT_BOUNDARY_HPP_

#include <vector>

#include "boundaries/lss_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include "containers/lss_container_2d.hpp"
#include "discretization/lss_discretization.hpp"
#include "lss_2d_general_svc_heston_equation_implicit_coefficients.hpp"
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
using lss_enumerations::by_enum;
using lss_utility::coefficient_sevenlet_t;
using lss_utility::function_2d_triplet_t;
using lss_utility::pair_t;
using lss_utility::range;

template <template <typename, typename> typename container, typename fp_type, typename alloc>
using explicit_heston_boundary_scheme_function_t =
    std::function<void(function_2d_triplet_t<fp_type> const &, coefficient_sevenlet_t<fp_type> const &,
                       boundary_2d_pair<fp_type> const &, std::size_t const &, fp_type const &, pair_t<fp_type> const &,
                       container_2d<by_enum::Row, fp_type, container, alloc> const &,
                       container_2d<by_enum::Row, fp_type, container, alloc> const &, fp_type const &,
                       container<fp_type, alloc> &)>;

/**
    explicit_heston_boundary_scheme object
 */
template <typename fp_type, template <typename, typename> typename container, typename allocator>
class explicit_heston_boundary_scheme
{
    typedef container<fp_type, allocator> container_t;
    typedef container_2d<by_enum::Row, fp_type, container, allocator> rcontainer_2d_t;
    typedef explicit_heston_boundary_scheme_function_t<container, fp_type, allocator> scheme_function_t;

  public:
    static scheme_function_t const get_vertical(range<fp_type> const &rangex, bool is_homogeneus)
    {
        const fp_type four = static_cast<fp_type>(4.0);
        const fp_type three = static_cast<fp_type>(3.0);
        const fp_type two = static_cast<fp_type>(2.0);
        const fp_type one = static_cast<fp_type>(1.0);

        auto scheme_fun_h = [=](function_2d_triplet_t<fp_type> const &funcs,
                                coefficient_sevenlet_t<fp_type> const &coefficients,
                                boundary_2d_pair<fp_type> const &horizontal_boundary_pair, std::size_t const &y_index,
                                fp_type const &y, pair_t<fp_type> const &steps, rcontainer_2d_t const &input,
                                rcontainer_2d_t const &inhom_input, fp_type const &time, container_t &solution) {
            auto const &D = std::get<0>(funcs);
            auto const &E = std::get<1>(funcs);
            auto const &F = std::get<2>(funcs);

            auto const &first_bnd = horizontal_boundary_pair.first;
            auto const &second_bnd = horizontal_boundary_pair.second;

            auto const &delta = std::get<3>(coefficients);
            auto const &ni = std::get<4>(coefficients);
            auto const &rho = std::get<5>(coefficients);

            auto const start_x = rangex.lower();
            auto const h_1 = steps.first;

            fp_type x{};
            if (auto const &ptr = std::dynamic_pointer_cast<dirichlet_boundary_2d<fp_type>>(first_bnd))
            {
                solution[0] = ptr->value(time, y);
            }

            const std::size_t N = solution.size() - 1;
            if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_2d<fp_type>>(second_bnd))
            {
                const fp_type delta_ = two * h_1 * ptr->value(time, y);
                x = start_x + static_cast<fp_type>(N) * h_1;
                solution[N] = ((one - three * ni * E(x, y) + rho * F(x, y)) * input(N, y_index)) +
                              (four * ni * E(x, y) * input(N, y_index + 1)) - (ni * E(x, y) * input(N, y_index + 2)) -
                              (delta * delta_ * D(x, y));
            }

            for (std::size_t t = 1; t < N; ++t)
            {
                x = start_x + static_cast<fp_type>(t) * h_1;
                solution[t] = (-delta * D(x, y) * input(t - 1, y_index)) +
                              ((one - three * ni * E(x, y) + rho * F(x, y)) * input(t, y_index)) +
                              (delta * D(x, y) * input(t + 1, y_index)) - (ni * E(x, y) * input(t, y_index + 2)) +
                              (four * ni * E(x, y) * input(t, y_index + 1));
            }
        };
        auto scheme_fun_nh = [=](function_2d_triplet_t<fp_type> const &funcs,
                                 coefficient_sevenlet_t<fp_type> const &coefficients,
                                 boundary_2d_pair<fp_type> const &horizontal_boundary_pair, std::size_t const &y_index,
                                 fp_type const &y, pair_t<fp_type> const &steps, rcontainer_2d_t const &input,
                                 rcontainer_2d_t const &inhom_input, fp_type const &time, container_t &solution) {
            auto const &D = std::get<0>(funcs);
            auto const &E = std::get<1>(funcs);
            auto const &F = std::get<2>(funcs);

            auto const &second_bnd = horizontal_boundary_pair.second;

            auto const &delta = std::get<3>(coefficients);
            auto const &ni = std::get<4>(coefficients);
            auto const &rho = std::get<5>(coefficients);
            auto const &theta = std::get<6>(coefficients);

            auto const start_x = rangex.lower();
            auto const h_1 = steps.first;

            fp_type x{};
            const std::size_t N = solution.size() - 1;
            if (auto const &ptr = std::dynamic_pointer_cast<neumann_boundary_2d<fp_type>>(second_bnd))
            {
                const fp_type delta_ = two * h_1 * ptr->value(time, y);
                x = start_x + static_cast<fp_type>(N) * h_1;
                solution[N] = ((one - three * ni * E(x, y) + rho * F(x, y)) * input(N, y_index)) +
                              (four * ni * E(x, y) * input(N, y_index + 1)) - (ni * E(x, y) * input(N, y_index + 2)) -
                              (delta * delta_ * D(x, y)) + (rho * inhom_input(N, y_index));
            }

            for (std::size_t t = 1; t < N; ++t)
            {
                x = start_x + static_cast<fp_type>(t) * h_1;
                solution[t] = (-delta * D(x, y) * input(t - 1, y_index)) +
                              ((one - three * ni * E(x, y) + rho * F(x, y)) * input(t, y_index)) +
                              (delta * D(x, y) * input(t + 1, y_index)) - (ni * E(x, y) * input(t, y_index + 2)) +
                              (four * ni * E(x, y) * input(t, y_index + 1)) + (rho * inhom_input(N, y_index));
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
    general_svc_heston_equation_explicit_boundary object
 */
template <typename fp_type, template <typename, typename> typename container = std::vector,
          typename allocator = std::allocator<fp_type>>
class general_svc_heston_equation_explicit_boundary
{
    typedef container_2d<by_enum::Row, fp_type, container, allocator> rcontainer_2d_t;
    typedef container_2d<by_enum::Column, fp_type, container, allocator> ccontainer_2d_t;
    typedef container<fp_type, allocator> container_t;

  private:
    general_svc_heston_equation_implicit_coefficients_ptr<fp_type> coefficients_;
    function_2d_triplet_t<fp_type> fun_triplet_;
    // constant coefficients:
    coefficient_sevenlet_t<fp_type> coeff_sevenlet_;
    // steps pair:
    std::pair<fp_type, fp_type> steps_;

    explicit general_svc_heston_equation_explicit_boundary() = delete;

    void initialize()
    {
        coeff_sevenlet_ =
            std::make_tuple(coefficients_->alpha_, coefficients_->beta_, coefficients_->gamma_, coefficients_->delta_,
                            coefficients_->ni_, coefficients_->rho_, coefficients_->theta_);
        steps_ = std::make_pair(coefficients_->h_1_, coefficients_->h_2_);
        fun_triplet_ = std::make_tuple(coefficients_->D_, coefficients_->E_, coefficients_->F_);
    }

  public:
    explicit general_svc_heston_equation_explicit_boundary(
        general_svc_heston_equation_implicit_coefficients_ptr<fp_type> const &coefficients)
        : coefficients_{coefficients}
    {
        initialize();
    }

    ~general_svc_heston_equation_explicit_boundary()
    {
    }

    general_svc_heston_equation_explicit_boundary(general_svc_heston_equation_explicit_boundary const &) = delete;
    general_svc_heston_equation_explicit_boundary(general_svc_heston_equation_explicit_boundary &&) = delete;
    general_svc_heston_equation_explicit_boundary &operator=(general_svc_heston_equation_explicit_boundary const &) =
        delete;
    general_svc_heston_equation_explicit_boundary &operator=(general_svc_heston_equation_explicit_boundary &&) = delete;

    void solve(container_2d<by_enum::Row, fp_type, container, allocator> const &prev_solution,
               boundary_2d_pair<fp_type> const &horizonatal_boundary_pair,
               boundary_2d_ptr<fp_type> const &vertical_upper_boundary_ptr, fp_type const &time,
               container_2d<by_enum::Row, fp_type, container, allocator> &solution);

    void solve(container_2d<by_enum::Row, fp_type, container, allocator> const &prev_solution,
               boundary_2d_pair<fp_type> const &horizonatal_boundary_pair, fp_type const &time,
               container_2d<by_enum::Row, fp_type, container, allocator> &solution);
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void general_svc_heston_equation_explicit_boundary<fp_type, container, allocator>::solve(
    container_2d<by_enum::Row, fp_type, container, allocator> const &prev_solution,
    boundary_2d_pair<fp_type> const &horizonatal_boundary_pair,
    boundary_2d_ptr<fp_type> const &vertical_upper_boundary_ptr, fp_type const &time,
    container_2d<by_enum::Row, fp_type, container, allocator> &solution)
{
    typedef explicit_heston_boundary_scheme<fp_type, container, allocator> heston_boundary_scheme;
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;

    ccontainer_2d_t csolution(solution);
    // 1D container for intermediate solution:
    container_t solution_v(coefficients_->space_size_x_, fp_type{});
    // container for current source:
    rcontainer_2d_t curr_source(1, 1, fp_type{});
    // get the right-hand side of the scheme:
    auto scheme = heston_boundary_scheme::get_vertical(coefficients_->rangex_, true);
    fp_type start_y{coefficients_->rangey_.lower()};

    scheme(fun_triplet_, coeff_sevenlet_, horizonatal_boundary_pair, 0, start_y, steps_, prev_solution, curr_source,
           time, solution_v);
    csolution(0, solution_v);

    auto const &upper_bnd_ptr = std::dynamic_pointer_cast<dirichlet_boundary_2d<fp_type>>(vertical_upper_boundary_ptr);
    auto const &upper_bnd = [=](fp_type s, fp_type t) { return upper_bnd_ptr->value(t, s); };
    d_1d::of_function(coefficients_->rangex_.lower(), coefficients_->h_1_, time, upper_bnd, solution_v);
    csolution(coefficients_->space_size_y_ - 1, solution_v);
    solution = csolution;
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void general_svc_heston_equation_explicit_boundary<fp_type, container, allocator>::solve(
    container_2d<by_enum::Row, fp_type, container, allocator> const &prev_solution,
    boundary_2d_pair<fp_type> const &horizonatal_boundary_pair, fp_type const &time,
    container_2d<by_enum::Row, fp_type, container, allocator> &solution)
{
    typedef explicit_heston_boundary_scheme<fp_type, container, allocator> heston_boundary_scheme;
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;

    // 1D container for intermediate solution:
    container_t solution_v(coefficients_->space_size_y_, fp_type{});
    // some constants:
    auto const &start_y = coefficients_->rangey_.lower();
    // populating lower horizontal:
    auto const &lower_bnd_ptr =
        std::dynamic_pointer_cast<dirichlet_boundary_2d<fp_type>>(horizonatal_boundary_pair.first);
    auto const &lower_bnd = [=](fp_type v, fp_type t) { return lower_bnd_ptr->value(t, v); };
    d_1d::of_function(start_y, coefficients_->h_2_, time, lower_bnd, solution_v);
    solution(0, solution_v);
    // populating upper horizontal:
    auto const lri = solution.rows() - 1;
    auto const two = static_cast<fp_type>(2.0);
    auto const three = static_cast<fp_type>(3.0);
    auto const four = static_cast<fp_type>(4.0);
    auto const &upper_bnd_ptr =
        std::dynamic_pointer_cast<neumann_boundary_2d<fp_type>>(horizonatal_boundary_pair.second);
    auto const &upper_bnd = [=](fp_type v, fp_type t) {
        const std::size_t j = static_cast<std::size_t>((v - start_y) / coefficients_->h_2_);
        auto const bnd_val = upper_bnd_ptr->value(t, v);
        return (((four * solution(lri - 1, j)) - solution(lri - 2, j) - (two * coefficients_->h_1_ * bnd_val)) / three);
    };
    d_1d::of_function(start_y, coefficients_->h_2_, time, upper_bnd, solution_v);
    solution(coefficients_->space_size_x_ - 1, solution_v);
}

} // namespace two_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_2D_GENERAL_SVC_HESTON_EQUATION_EXPLICIT_BOUNDARY_HPP_
