#if !defined(_LSS_1D_GENERAL_HESTON_EQUATION_BUILDER_HPP_)
#define _LSS_1D_GENERAL_HESTON_EQUATION_BUILDER_HPP_

#include <functional>
#include <map>

#include "boundaries/lss_boundary.hpp"
#include "common/lss_utility.hpp"
#include "discretization/lss_discretization.hpp"
#include "pde_solvers/two_dimensional/heat_type/lss_2d_general_heston_equation.hpp"

namespace lss_pde_solvers
{

namespace two_dimensional
{
using lss_boundary::boundary_2d_pair;
using lss_utility::sptr_t;

namespace implicit_solvers
{

/*!
============================================================================
Represents general variable coefficient Heston type equation

u_t = a(t,x,y)*u_xx + b(t,x,y)*u_yy + c(t,x,y)*u_xy + d(t,x,y)*u_x + e(t,x,y)*u_y +
        f(t,x,y)*u + F(t,x,y)

t > 0, x_1 < x < x_2, y_1 < y < y_2

with initial condition:

u(0,x,y) = G(x,y)

or terminal condition:

u(T,x,y) = G(x,y)

horizontal_boundary_pair = S = (S_1,S_2) boundary

             vol (Y)
        ________________
        |S_1,S_1,S_1,S_1|
        |               |
        |               |
S (X)   |               |
        |               |
        |               |
        |               |
        |               |
        |S_2,S_2,S_2,S_2|
        |_______________|

// ============================================================================
*/
template <typename fp_type, template <typename, typename> typename container = std::vector,
          typename allocator = std::allocator<fp_type>>
class general_heston_equation_builder
{
  private:
    heat_data_config_2d_ptr<fp_type> heat_data_config_;
    pde_discretization_config_2d_ptr<fp_type> discretization_config_;
    boundary_2d_ptr<fp_type> vertical_upper_boundary_;
    boundary_2d_pair<fp_type> horizontal_boundary_pair_;
    splitting_method_config_ptr<fp_type> splitting_method_config_;
    grid_config_hints_2d_ptr<fp_type> grid_hints_cfg_;
    heat_implicit_solver_config_ptr solver_config_;
    std::map<std::string, fp_type> solver_config_details_;

  public:
    general_heston_equation_builder &heat_data_config(const heat_data_config_2d_ptr<fp_type> &heat_data_config)
    {
        heat_data_config_ = heat_data_config;
        return *this;
    }

    general_heston_equation_builder &discretization_config(
        const pde_discretization_config_2d_ptr<fp_type> &discretization_config)
    {
        discretization_config_ = discretization_config;
        return *this;
    }

    general_heston_equation_builder &vertical_upper_boundary(const boundary_2d_ptr<fp_type> &boundary_ptr)
    {
        vertical_upper_boundary_ = boundary_ptr;
        return *this;
    }

    general_heston_equation_builder &horizontal_boundary(const boundary_2d_pair<fp_type> &boundary_pair)
    {
        horizontal_boundary_pair_ = boundary_pair;
        return *this;
    }

    general_heston_equation_builder &splitting_method_config(
        const splitting_method_config_ptr<fp_type> &splitting_config)
    {
        splitting_method_config_ = splitting_config;
        return *this;
    }

    general_heston_equation_builder &grid_hints(const grid_config_hints_2d_ptr<fp_type> &grid_hints)
    {
        grid_hints_cfg_ = grid_hints;
        return *this;
    }

    general_heston_equation_builder &solver_config(const heat_implicit_solver_config_ptr &solver_config)
    {
        solver_config_ = solver_config;
        return *this;
    }

    general_heston_equation_builder &solver_config_details(const std::map<std::string, fp_type> &solver_config_details)
    {
        solver_config_details_ = solver_config_details;
        return *this;
    }

    sptr_t<general_heston_equation<fp_type, container, allocator>> build()
    {
        return std::make_shared<general_heston_equation<fp_type, container, allocator>>(
            heat_data_config_, discretization_config_, vertical_upper_boundary_, horizontal_boundary_pair_,
            splitting_method_config_, grid_hints_cfg_, solver_config_, solver_config_details_);
    }
};

} // namespace implicit_solvers

namespace explicit_solvers
{
/*!
============================================================================
Represents general variable coefficient Heston type equation

u_t = a(t,x,y)*u_xx + b(t,x,y)*u_yy + c(t,x,y)*u_xy + d(t,x,y)*u_x +
e(t,x,y)*u_y + f(t,x,y)*u + F(t,x,y)

t > 0, x_1 < x < x_2, y_1 < y < y_2

with initial condition:

u(0,x,y) = G(x,y)

or terminal condition:

u(T,x,y) = G(x,y)

horizontal_boundary_pair = S = (S_1,S_2) boundary

         vol (Y)
    ________________
    |S_1,S_1,S_1,S_1|
    |               |
    |               |
S (X)   |               |
    |               |
    |               |
    |               |
    |               |
    |S_2,S_2,S_2,S_2|
    |_______________|

// ============================================================================
*/
template <typename fp_type, template <typename, typename> typename container = std::vector,
          typename allocator = std::allocator<fp_type>>
class general_heston_equation_builder
{
  private:
    heat_data_config_2d_ptr<fp_type> heat_data_config_;
    pde_discretization_config_2d_ptr<fp_type> discretization_config_;
    boundary_2d_ptr<fp_type> vertical_upper_boundary_;
    boundary_2d_pair<fp_type> horizontal_boundary_pair_;
    grid_config_hints_2d_ptr<fp_type> grid_hints_cfg_;
    heat_explicit_solver_config_ptr solver_config_;

  public:
    general_heston_equation_builder &heat_data_config(const heat_data_config_2d_ptr<fp_type> &heat_data_config)
    {
        heat_data_config_ = heat_data_config;
        return *this;
    }

    general_heston_equation_builder &discretization_config(
        const pde_discretization_config_2d_ptr<fp_type> &discretization_config)
    {
        discretization_config_ = discretization_config;
        return *this;
    }

    general_heston_equation_builder &vertical_upper_boundary(const boundary_2d_ptr<fp_type> &boundary_ptr)
    {
        vertical_upper_boundary_ = boundary_ptr;
        return *this;
    }

    general_heston_equation_builder &horizontal_boundary(const boundary_2d_pair<fp_type> &boundary_pair)
    {
        horizontal_boundary_pair_ = boundary_pair;
        return *this;
    }

    general_heston_equation_builder &grid_hints(const grid_config_hints_2d_ptr<fp_type> &grid_hints)
    {
        grid_hints_cfg_ = grid_hints;
        return *this;
    }

    general_heston_equation_builder &solver_config(const heat_explicit_solver_config_ptr &solver_config)
    {
        solver_config_ = solver_config;
        return *this;
    }

    sptr_t<general_heston_equation<fp_type, container, allocator>> build()
    {
        return std::make_shared<general_heston_equation<fp_type, container, allocator>>(
            heat_data_config_, discretization_config_, vertical_upper_boundary_, horizontal_boundary_pair_,
            grid_hints_cfg_, solver_config_);
    }
};
} // namespace explicit_solvers

} // namespace two_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_1D_GENERAL_HESTON_EQUATION_BUILDER_HPP_
