#if !defined(_LSS_HEAT_SAULYEV_SOLVER_METHOD_HPP_)
#define _LSS_HEAT_SAULYEV_SOLVER_METHOD_HPP_

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
#include "pde_solvers/one_dimensional/heat_type/explicit_coefficients/lss_heat_saulyev_svc_coefficients.hpp"

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
 heat_saulyev_solver_method  object

*/
template <typename fp_type, template <typename, typename> typename container, typename allocator>
class heat_saulyev_solver_method
{
    typedef container<fp_type, allocator> container_t;
    typedef std::function<void(container_t &, container_t const &, fp_type)> sweeper_fun;

  private:
    // constant coeffs:
    const fp_type czero_ = static_cast<fp_type>(0.0);
    const fp_type cone_ = static_cast<fp_type>(1.0);
    // scheme coefficients:
    heat_saulyev_svc_coefficients_ptr<fp_type> coefficients_;
    grid_config_1d_ptr<fp_type> grid_cfg_;
    // sweepers:
    sweeper_fun up_sweeper_, down_sweeper_;
    // containers for source:
    container_t source_dummy_, source_, source_next_;

    explicit heat_saulyev_solver_method() = delete;

    void initialize(bool is_heat_sourse_set)
    {
        auto a = coefficients_->A_;
        auto b = coefficients_->B_;
        auto d = coefficients_->D_;
        auto K = coefficients_->K_;

        up_sweeper_ = [=](container_t &up_component, container_t const &rhs, fp_type rhs_coeff) {
            fp_type x{};
            for (std::size_t t = 1; t < up_component.size() - 1; ++t)
            {
                x = grid_1d<fp_type>::value(grid_cfg_, t);
                up_component[t] = b(x) * up_component[t] + d(x) * up_component[t + 1] + a(x) * up_component[t - 1] +
                                  K(x) * rhs_coeff * rhs[t];
            }
        };

        down_sweeper_ = [=](container_t &down_component, container_t const &rhs, fp_type rhs_coeff) {
            fp_type x{};
            for (std::size_t t = down_component.size() - 2; t >= 1; --t)
            {
                x = grid_1d<fp_type>::value(grid_cfg_, t);
                down_component[t] = b(x) * down_component[t] + d(x) * down_component[t + 1] +
                                    a(x) * down_component[t - 1] + K(x) * rhs_coeff * rhs[t];
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
    explicit heat_saulyev_solver_method(heat_saulyev_svc_coefficients_ptr<fp_type> const &coefficients,
                                        grid_config_1d_ptr<fp_type> const &grid_config, bool is_heat_sourse_set)
        : coefficients_{coefficients}, grid_cfg_{grid_config}
    {
        initialize(is_heat_sourse_set);
    }

    ~heat_saulyev_solver_method()
    {
    }

    heat_saulyev_solver_method(heat_saulyev_solver_method const &) = delete;
    heat_saulyev_solver_method(heat_saulyev_solver_method &&) = delete;
    heat_saulyev_solver_method &operator=(heat_saulyev_solver_method const &) = delete;
    heat_saulyev_solver_method &operator=(heat_saulyev_solver_method &&) = delete;

    void solve(boundary_1d_pair<fp_type> const &boundary_pair, std::size_t const &time_idx, fp_type const &time,
               container<fp_type, allocator> &solution);

    void solve(boundary_1d_pair<fp_type> const &boundary_pair, std::size_t const &time_idx, fp_type const &time,
               fp_type const &next_time, std::function<fp_type(fp_type, fp_type)> const &heat_source,
               container<fp_type, allocator> &solution);
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void heat_saulyev_solver_method<fp_type, container, allocator>::solve(boundary_1d_pair<fp_type> const &boundary_pair,
                                                                      std::size_t const &time_idx, fp_type const &time,
                                                                      container<fp_type, allocator> &solution)
{
    if (time_idx % 2 == 0)
    {
        down_sweeper_(solution, source_dummy_, czero_);
    }
    else
    {
        up_sweeper_(solution, source_dummy_, czero_);
    }
    solution[0] = boundary_pair.first->value(time);
    solution[coefficients_->space_size_ - 1] = boundary_pair.second->value(time);
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void heat_saulyev_solver_method<fp_type, container, allocator>::solve(
    boundary_1d_pair<fp_type> const &boundary_pair, std::size_t const &time_idx, fp_type const &time,
    fp_type const &next_time, std::function<fp_type(fp_type, fp_type)> const &heat_source,
    container<fp_type, allocator> &solution)
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    if (time_idx % 2 == 0)
    {
        d_1d::of_function(grid_cfg_, time, heat_source, source_);
        down_sweeper_(solution, source_, cone_);
    }
    else
    {
        d_1d::of_function(grid_cfg_, next_time, heat_source, source_next_);
        up_sweeper_(solution, source_next_, cone_);
    }
    solution[0] = boundary_pair.first->value(time);
    solution[coefficients_->space_size_ - 1] = boundary_pair.second->value(time);
}
} // namespace one_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_HEAT_SAULYEV_SOLVER_METHOD_HPP_
