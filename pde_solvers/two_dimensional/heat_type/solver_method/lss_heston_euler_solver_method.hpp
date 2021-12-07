#if !defined(_LSS_HESTON_EULER_SOLVER_METHOD_HPP_)
#define _LSS_HESTON_EULER_SOLVER_METHOD_HPP_

#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "containers/lss_container_2d.hpp"
#include "discretization/lss_discretization.hpp"
#include "discretization/lss_grid.hpp"
#include "discretization/lss_grid_config.hpp"
#include "pde_solvers/lss_pde_discretization_config.hpp"
#include "pde_solvers/two_dimensional/heat_type/explicit_coefficients/lss_heston_euler_coefficients.hpp"

namespace lss_pde_solvers
{
namespace two_dimensional
{

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class explicit_heston_scheme
{
    typedef container_2d<by_enum::Row, fp_type, container, allocator> rcontainer_2d_t;

  public:
    static void rhs(heston_euler_coefficients_ptr<fp_type> const &cfs, grid_config_2d_ptr<fp_type> const &grid_config,
                    rcontainer_2d_t const &input, fp_type const &time, rcontainer_2d_t &solution)
    {
        auto const one = static_cast<fp_type>(1.0);
        auto const &M = cfs->M_;
        auto const &M_tilde = cfs->M_tilde_;
        auto const &P = cfs->P_;
        auto const &P_tilde = cfs->P_tilde_;
        auto const &Z = cfs->Z_;
        auto const &W = cfs->W_;
        auto const &C = cfs->C_;
        auto const rows = input.rows() - 1;
        auto const cols = input.columns() - 1;

        fp_type x{}, y{}, val{};
        for (std::size_t r = 1; r < rows; ++r)
        {
            x = grid_2d<fp_type>::value_1(grid_config, r);
            for (std::size_t c = 1; c < cols; ++c)
            {
                y = grid_2d<fp_type>::value_2(grid_config, c);
                val = C(time, x, y) * input(r - 1, c - 1) + M(time, x, y) * input(r - 1, c) -
                      C(time, x, y) * input(r - 1, c + 1) + M_tilde(time, x, y) * input(r, c - 1) +
                      (one - Z(time, x, y) - W(time, x, y)) * input(r, c) + P_tilde(time, x, y) * input(r, c + 1) -
                      C(time, x, y) * input(r + 1, c - 1) + P(time, x, y) * input(r + 1, c) +
                      C(time, x, y) * input(r + 1, c + 1);
                solution(r, c, val);
            }
        }
    }

    static void rhs_source(heston_euler_coefficients_ptr<fp_type> const &cfs,
                           grid_config_2d_ptr<fp_type> const &grid_config, rcontainer_2d_t const &input,
                           fp_type const &time, rcontainer_2d_t const &inhom_input, rcontainer_2d_t &solution)
    {
        auto const one = static_cast<fp_type>(1.0);
        auto const &M = cfs->M_;
        auto const &M_tilde = cfs->M_tilde_;
        auto const &P = cfs->P_;
        auto const &P_tilde = cfs->P_tilde_;
        auto const &Z = cfs->Z_;
        auto const &W = cfs->W_;
        auto const &C = cfs->C_;
        auto const rho = cfs->rho_;
        auto const rows = input.rows() - 1; // without boundary
        auto const cols = input.columns() - 1;

        fp_type x{}, y{}, val{};
        for (std::size_t r = 1; r < rows; ++r)
        {
            x = grid_2d<fp_type>::value_1(grid_config, r);
            for (std::size_t c = 1; c < cols; ++c)
            {
                y = grid_2d<fp_type>::value_2(grid_config, c);
                val = C(time, x, y) * input(r - 1, c - 1) + M(time, x, y) * input(r - 1, c) -
                      C(time, x, y) * input(r - 1, c + 1) + M_tilde(time, x, y) * input(r, c - 1) +
                      (one - Z(time, x, y) - W(time, x, y)) * input(r, c) + P_tilde(time, x, y) * input(r, c + 1) -
                      C(time, x, y) * input(r + 1, c - 1) + P(time, x, y) * input(r + 1, c) +
                      C(time, x, y) * input(r + 1, c + 1) + rho * inhom_input(r, c);
                solution(r, c, val);
            }
        }
    }
};

/**
    heston_euler_solver_method object
*/
template <typename fp_type, template <typename, typename> typename container, typename allocator>
class heston_euler_solver_method
{
    typedef explicit_heston_scheme<fp_type, container, allocator> heston_scheme;
    typedef discretization<dimension_enum::Two, fp_type, container, allocator> d_2d;
    typedef container_2d<by_enum::Row, fp_type, container, allocator> rcontainer_2d_t;
    typedef sptr_t<container_2d<by_enum::Row, fp_type, container, allocator>> rcontainer_2d_ptr;

  private:
    // scheme coefficients:
    heston_euler_coefficients_ptr<fp_type> coefficients_;
    grid_config_2d_ptr<fp_type> grid_cfg_;
    rcontainer_2d_ptr source_;

    explicit heston_euler_solver_method() = delete;

    void initialize(bool is_heat_source_set)
    {
        if (is_heat_source_set)
        {
            source_ = std::make_shared<container_2d<by_enum::Row, fp_type, container, allocator>>(
                coefficients_->space_size_x_, coefficients_->space_size_y_);
        }
    }

  public:
    explicit heston_euler_solver_method(heston_euler_coefficients_ptr<fp_type> const &coefficients,
                                        grid_config_2d_ptr<fp_type> const &grid_config, bool is_heat_source_set)
        : coefficients_{coefficients}, grid_cfg_{grid_config}
    {
        initialize(is_heat_source_set);
    }

    ~heston_euler_solver_method()
    {
    }

    heston_euler_solver_method(heston_euler_solver_method const &) = delete;
    heston_euler_solver_method(heston_euler_solver_method &&) = delete;
    heston_euler_solver_method &operator=(heston_euler_solver_method const &) = delete;
    heston_euler_solver_method &operator=(heston_euler_solver_method &&) = delete;

    void solve(container_2d<by_enum::Row, fp_type, container, allocator> &prev_solution, fp_type const &time,
               container_2d<by_enum::Row, fp_type, container, allocator> &solution);

    void solve(container_2d<by_enum::Row, fp_type, container, allocator> &prev_solution, fp_type const &time,
               std::function<fp_type(fp_type, fp_type, fp_type)> const &heat_source,
               container_2d<by_enum::Row, fp_type, container, allocator> &solution);
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void heston_euler_solver_method<fp_type, container, allocator>::solve(
    container_2d<by_enum::Row, fp_type, container, allocator> &prev_solution, fp_type const &time,
    container_2d<by_enum::Row, fp_type, container, allocator> &solution)
{
    heston_scheme::rhs(coefficients_, grid_cfg_, prev_solution, time, solution);
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void heston_euler_solver_method<fp_type, container, allocator>::solve(
    container_2d<by_enum::Row, fp_type, container, allocator> &prev_solution, fp_type const &time,
    std::function<fp_type(fp_type, fp_type, fp_type)> const &heat_source,
    container_2d<by_enum::Row, fp_type, container, allocator> &solution)
{
    d_2d::of_function(grid_cfg_, time, heat_source, *source_);
    heston_scheme::rhs_source(coefficients_, grid_cfg_, prev_solution, time, *source_, solution);
}

} // namespace two_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_HESTON_EULER_SOLVER_METHOD_HPP_
