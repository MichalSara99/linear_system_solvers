#if !defined(_LSS_HEAT_BARAKAT_CLARK_SOLVER_METHOD_HPP_)
#define _LSS_HEAT_BARAKAT_CLARK_SOLVER_METHOD_HPP_

#include <future>

#include "boundaries/lss_boundary.hpp"
#include "boundaries/lss_dirichlet_boundary.hpp"
#include "boundaries/lss_neumann_boundary.hpp"
#include "boundaries/lss_robin_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "containers/lss_container_2d.hpp"
#include "discretization/lss_discretization.hpp"
#include "discretization/lss_grid.hpp"
#include "discretization/lss_grid_config.hpp"
#include "pde_solvers/lss_heat_data_config.hpp"
#include "pde_solvers/lss_heat_solver_config.hpp"
#include "pde_solvers/lss_pde_discretization_config.hpp"
#include "pde_solvers/one_dimensional/heat_type/explicit_coefficients/lss_heat_barakat_clark_coefficients.hpp"

namespace lss_pde_solvers
{

namespace one_dimensional
{

using lss_boundary::boundary_1d_pair;
using lss_boundary::boundary_1d_ptr;
using lss_boundary::dirichlet_boundary_1d;
using lss_boundary::neumann_boundary_1d;
using lss_boundary::robin_boundary_1d;
using lss_utility::sptr_t;

/**
 heat_barakat_clark_solver_method  object

*/
template <typename fp_type, template <typename, typename> typename container, typename allocator>
class heat_barakat_clark_solver_method
{
    typedef container<fp_type, allocator> container_t;
    typedef std::function<void(container_t &, container_t const &, fp_type, fp_type)> sweeper_fun;

  private:
    // constant coeffs:
    const fp_type cone_ = static_cast<fp_type>(1.0);
    const fp_type chalf_ = static_cast<fp_type>(0.5);
    const fp_type czero_ = static_cast<fp_type>(0.0);
    // scheme coefficients:
    heat_barakat_clark_coefficients_ptr<fp_type> coefficients_;
    grid_config_1d_ptr<fp_type> grid_cfg_;
    // sweepers:
    sweeper_fun up_sweeper_, down_sweeper_;
    // containers:
    container_t source_dummy_, source_, source_next_;

    explicit heat_barakat_clark_solver_method() = delete;

    void initialize(bool is_heat_sourse_set)
    {
        auto a = coefficients_->A_;
        auto b = coefficients_->B_;
        auto d = coefficients_->D_;
        auto K = coefficients_->K_;

        up_sweeper_ = [=](container_t &up_component, container_t const &rhs, fp_type time, fp_type rhs_coeff) {
            fp_type x{};
            for (std::size_t t = 1; t < up_component.size() - 1; ++t)
            {
                x = grid_1d<fp_type>::value(grid_cfg_, t);
                up_component[t] = b(time, x) * up_component[t] + d(time, x) * up_component[t + 1] +
                                  a(time, x) * up_component[t - 1] + K(time, x) * rhs_coeff * rhs[t];
            }
        };

        down_sweeper_ = [=](container_t &down_component, container_t const &rhs, fp_type time, fp_type rhs_coeff) {
            fp_type x{};
            for (std::size_t t = down_component.size() - 2; t >= 1; --t)
            {
                x = grid_1d<fp_type>::value(grid_cfg_, t);
                down_component[t] = b(time, x) * down_component[t] + d(time, x) * down_component[t + 1] +
                                    a(time, x) * down_component[t - 1] + K(time, x) * rhs_coeff * rhs[t];
            }
        };

        if (is_heat_sourse_set)
        {
            source_.resize(coefficients_->space_size_);
            source_next_.resize(coefficients_->space_size_);
        }
        else
        {
            source_dummy_.resize(coefficients_->space_size_);
        }
    }

  public:
    explicit heat_barakat_clark_solver_method(heat_barakat_clark_coefficients_ptr<fp_type> const &coefficients,
                                              grid_config_1d_ptr<fp_type> const &grid_config, bool is_heat_sourse_set)
        : coefficients_{coefficients}, grid_cfg_{grid_config}
    {
        initialize(is_heat_sourse_set);
    }

    ~heat_barakat_clark_solver_method()
    {
    }

    heat_barakat_clark_solver_method(heat_barakat_clark_solver_method const &) = delete;
    heat_barakat_clark_solver_method(heat_barakat_clark_solver_method &&) = delete;
    heat_barakat_clark_solver_method &operator=(heat_barakat_clark_solver_method const &) = delete;
    heat_barakat_clark_solver_method &operator=(heat_barakat_clark_solver_method &&) = delete;

    void solve(boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time,
               container<fp_type, allocator> &solution);

    void solve(boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, fp_type const &next_time,
               std::function<fp_type(fp_type, fp_type)> const &heat_source, container<fp_type, allocator> &solution);
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void heat_barakat_clark_solver_method<fp_type, container, allocator>::solve(
    boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, container<fp_type, allocator> &solution)
{
    // solution size:
    const std::size_t sol_size = solution.size();
    //  components of the solution:
    container_t cont_up(solution);
    container_t cont_down(solution);
    std::future<void> sweep_up =
        std::async(std::launch::async, up_sweeper_, std::ref(cont_up), source_dummy_, time, czero_);
    std::future<void> sweep_down =
        std::async(std::launch::async, down_sweeper_, std::ref(cont_down), source_dummy_, time, czero_);
    sweep_up.wait();
    sweep_down.wait();
    cont_up[0] = cont_down[0] = boundary_pair.first->value(time);
    cont_up[sol_size - 1] = cont_down[sol_size - 1] = boundary_pair.second->value(time);
    for (std::size_t t = 0; t < sol_size; ++t)
    {
        solution[t] = chalf_ * (cont_up[t] + cont_down[t]);
    }
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void heat_barakat_clark_solver_method<fp_type, container, allocator>::solve(
    boundary_1d_pair<fp_type> const &boundary_pair, fp_type const &time, fp_type const &next_time,
    std::function<fp_type(fp_type, fp_type)> const &heat_source, container<fp_type, allocator> &solution)
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    // solution size:
    const std::size_t sol_size = solution.size();
    //  components of the solution:
    container_t cont_up(solution);
    container_t cont_down(solution);
    d_1d::of_function(grid_cfg_, time, heat_source, source_);
    d_1d::of_function(grid_cfg_, next_time, heat_source, source_next_);
    std::future<void> sweep_up =
        std::async(std::launch::async, up_sweeper_, std::ref(cont_up), source_next_, time, cone_);
    std::future<void> sweep_down =
        std::async(std::launch::async, down_sweeper_, std::ref(cont_down), source_, time, cone_);
    sweep_up.wait();
    sweep_down.wait();
    cont_up[0] = cont_down[0] = boundary_pair.first->value(time);
    cont_up[sol_size - 1] = cont_down[sol_size - 1] = boundary_pair.second->value(time);
    for (std::size_t t = 0; t < sol_size; ++t)
    {
        solution[t] = chalf_ * (cont_up[t] + cont_down[t]);
    }
}
} // namespace one_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_HEAT_BARAKAT_CLARK_SOLVER_METHOD_HPP_
