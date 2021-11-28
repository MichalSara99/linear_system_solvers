#if !defined(_LSS_GENERAL_2ND_ODE_EQUATION_BUILDER_HPP_)
#define _LSS_GENERAL_2ND_ODE_EQUATION_BUILDER_HPP_

#include <map>

#include "boundaries/lss_boundary.hpp"
#include "common/lss_utility.hpp"
#include "ode_solvers/lss_ode_discretization_config.hpp"
#include "ode_solvers/second_degree/lss_general_ode_equation.hpp"

namespace lss_ode_solvers
{
using lss_boundary::boundary_2d_pair;
using lss_utility::sptr_t;

namespace implicit_solvers
{
template <typename fp_type, template <typename, typename> typename container = std::vector,
          typename allocator = std::allocator<fp_type>>
class general_2nd_ode_equation_builder
{
  private:
    ode_data_config_ptr<fp_type> ode_data_cfg_;
    ode_discretization_config_ptr<fp_type> ode_discretization_cfg_;
    boundary_1d_pair<fp_type> boundary_pair_;
    grid_config_hints_1d_ptr<fp_type> grid_hints_cfg_;
    ode_implicit_solver_config_ptr ode_solver_cfg_;
    std::map<std::string, fp_type> ode_solver_config_details_;

  public:
    general_2nd_ode_equation_builder &ode_data_config(ode_data_config_ptr<fp_type> const &ode_data_config)
    {
        ode_data_cfg_ = ode_data_config;
        return *this;
    }

    general_2nd_ode_equation_builder &discretization_config(
        ode_discretization_config_ptr<fp_type> const &discretization_config)
    {
        ode_discretization_cfg_ = discretization_config;
        return *this;
    }

    general_2nd_ode_equation_builder &boundary_pair(boundary_1d_pair<fp_type> const &boundary_pair)
    {
        boundary_pair_ = boundary_pair;
        return *this;
    }

    general_2nd_ode_equation_builder &grid_hints(grid_config_hints_1d_ptr<fp_type> const &grid_hints)
    {
        grid_hints_cfg_ = grid_hints;
        return *this;
    }

    general_2nd_ode_equation_builder &solver_config(ode_implicit_solver_config_ptr const &solver_config)
    {
        ode_solver_cfg_ = solver_config;
        return *this;
    }

    general_2nd_ode_equation_builder &solver_config_details(const std::map<std::string, fp_type> &solver_config_details)
    {
        ode_solver_config_details_ = solver_config_details;
        return *this;
    }

    sptr_t<general_ode_equation<fp_type, container, allocator>> build()
    {
        return std::make_shared<general_ode_equation<fp_type, container, allocator>>(
            ode_data_cfg_, ode_discretization_cfg_, boundary_pair_, grid_hints_cfg_, ode_solver_cfg_,
            ode_solver_config_details_);
    }
};

} // namespace implicit_solvers
} // namespace lss_ode_solvers

#endif ///_LSS_GENERAL_2ND_ODE_EQUATION_BUILDER_HPP_
