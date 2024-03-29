#if !defined(_LSS_1D_GENERAL_HEAT_EQUATION_BUILDER_HPP_)
#define _LSS_1D_GENERAL_HEAT_EQUATION_BUILDER_HPP_

#include <functional>
#include <map>

#include "boundaries/lss_boundary.hpp"
#include "common/lss_utility.hpp"
#include "discretization/lss_discretization.hpp"
#include "pde_solvers/one_dimensional/heat_type/lss_1d_general_heat_equation.hpp"

namespace lss_pde_solvers
{

namespace one_dimensional
{
using lss_boundary::boundary_1d_pair;
using lss_utility::sptr_t;

namespace implicit_solvers
{

/*!
============================================================================
Represents general variable coefficient 1D heat equation solver

u_t = a(t,x)*u_xx + b(t,x)*u_x + c(t,x)*u + F(t,x),
x_1 < x < x_2
t_1 < t < t_2

with initial condition:

u(t_1,x) = f(x)

or terminal condition:

u(t_2,x) = f(x)


// ============================================================================
*/
template <typename fp_type, template <typename, typename> typename container = std::vector,
          typename allocator = std::allocator<fp_type>>
class general_heat_equation_builder
{
  private:
    heat_data_config_1d_ptr<fp_type> heat_data_config_;
    pde_discretization_config_1d_ptr<fp_type> discretization_config_;
    boundary_1d_pair<fp_type> boundary_pair_;
    grid_config_hints_1d_ptr<fp_type> grid_hints_cfg_;
    heat_implicit_solver_config_ptr solver_config_;
    std::map<std::string, fp_type> solver_config_details_;

  public:
    general_heat_equation_builder &heat_data_config(const heat_data_config_1d_ptr<fp_type> &heat_data_config)
    {
        heat_data_config_ = heat_data_config;
        return *this;
    }

    general_heat_equation_builder &discretization_config(
        const pde_discretization_config_1d_ptr<fp_type> &discretization_config)
    {
        discretization_config_ = discretization_config;
        return *this;
    }

    general_heat_equation_builder &boundary_pair(const boundary_1d_pair<fp_type> &boundary_pair)
    {
        boundary_pair_ = boundary_pair;
        return *this;
    }

    general_heat_equation_builder &grid_hints(const grid_config_hints_1d_ptr<fp_type> &grid_hints)
    {
        grid_hints_cfg_ = grid_hints;
        return *this;
    }

    general_heat_equation_builder &solver_config(const heat_implicit_solver_config_ptr &solver_config)
    {
        solver_config_ = solver_config;
        return *this;
    }

    general_heat_equation_builder &solver_config_details(const std::map<std::string, fp_type> &solver_config_details)
    {
        solver_config_details_ = solver_config_details;
        return *this;
    }

    sptr_t<general_heat_equation<fp_type, container, allocator>> build()
    {
        return std::make_shared<general_heat_equation<fp_type, container, allocator>>(
            heat_data_config_, discretization_config_, boundary_pair_, grid_hints_cfg_, solver_config_,
            solver_config_details_);
    }
};

} // namespace implicit_solvers

namespace explicit_solvers
{

template <typename fp_type, template <typename, typename> typename container = std::vector,
          typename allocator = std::allocator<fp_type>>
class general_heat_equation_builder
{
  private:
    heat_data_config_1d_ptr<fp_type> heat_data_config_;
    pde_discretization_config_1d_ptr<fp_type> discretization_config_;
    boundary_1d_pair<fp_type> boundary_pair_;
    grid_config_hints_1d_ptr<fp_type> grid_hints_cfg_;
    heat_explicit_solver_config_ptr solver_config_;

  public:
    general_heat_equation_builder &heat_data_config(const heat_data_config_1d_ptr<fp_type> &heat_data_config)
    {
        heat_data_config_ = heat_data_config;
        return *this;
    }

    general_heat_equation_builder &discretization_config(
        const pde_discretization_config_1d_ptr<fp_type> &discretization_config)
    {
        discretization_config_ = discretization_config;
        return *this;
    }

    general_heat_equation_builder &boundary_pair(const boundary_1d_pair<fp_type> &boundary_pair)
    {
        boundary_pair_ = boundary_pair;
        return *this;
    }

    general_heat_equation_builder &grid_hints(const grid_config_hints_1d_ptr<fp_type> &grid_hints)
    {
        grid_hints_cfg_ = grid_hints;
        return *this;
    }

    general_heat_equation_builder &solver_config(const heat_explicit_solver_config_ptr &solver_config)
    {
        solver_config_ = solver_config;
        return *this;
    }

    sptr_t<general_heat_equation<fp_type, container, allocator>> build()
    {
        return std::make_shared<general_heat_equation<fp_type, container, allocator>>(
            heat_data_config_, discretization_config_, boundary_pair_, grid_hints_cfg_, solver_config_);
    }
};

} // namespace explicit_solvers

} // namespace one_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_1D_GENERAL_HEAT_EQUATION_BUILDER_HPP_
