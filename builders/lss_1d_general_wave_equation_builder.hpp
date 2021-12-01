#if !defined(_LSS_1D_GENERAL_WAVE_EQUATION_BUILDER_HPP_)
#define _LSS_1D_GENERAL_WAVE_EQUATION_BUILDER_HPP_

#include <functional>
#include <map>

#include "boundaries/lss_boundary.hpp"
#include "common/lss_utility.hpp"
#include "discretization/lss_discretization.hpp"
#include "pde_solvers/lss_pde_discretization_config.hpp"
#include "pde_solvers/one_dimensional/wave_type/lss_1d_general_wave_equation.hpp"

namespace lss_pde_solvers
{

namespace one_dimensional
{
using lss_boundary::boundary_1d_pair;
using lss_boundary::boundary_1d_ptr;
using lss_utility::sptr_t;

namespace implicit_solvers
{

/*!
============================================================================
Represents general variable coefficient 1D wave equation solver

u_tt + a(t,x)*u_t = b(t,x)*u_xx + c(t,x)*u_x + d(t,x)*u + F(t,x),
x_1 < x < x_2
t_1 < t < t_2

with initial condition:

u(t_1,x) = f(x)

u_t(t_1,x) = g(x)

or terminal condition:

u(t_1,x) = f(x)

u_t(t_2,x) = g(x)

// ============================================================================
*/
template <typename fp_type, template <typename, typename> typename container = std::vector,
          typename allocator = std::allocator<fp_type>>
class general_wave_equation_builder
{
  private:
    wave_data_config_1d_ptr<fp_type> wave_data_config_;
    pde_discretization_config_1d_ptr<fp_type> discretization_config_;
    boundary_1d_pair<fp_type> boundary_pair_;
    grid_config_hints_1d_ptr<fp_type> grid_config_hints_;
    wave_implicit_solver_config_ptr solver_config_;
    std::map<std::string, fp_type> solver_config_details_;

  public:
    general_wave_equation_builder &wave_data_config(const wave_data_config_1d_ptr<fp_type> &wave_data_config)
    {
        wave_data_config_ = wave_data_config;
        return *this;
    }

    general_wave_equation_builder &discretization_config(
        const pde_discretization_config_1d_ptr<fp_type> &discretization_config)
    {
        discretization_config_ = discretization_config;
        return *this;
    }

    general_wave_equation_builder &boundary_pair(const boundary_1d_pair<fp_type> &boundary_pair)
    {
        boundary_pair_ = boundary_pair;
        return *this;
    }

    general_wave_equation_builder &grid_config_hints(const grid_config_hints_1d_ptr<fp_type> &grid_config_hints)
    {
        grid_config_hints_ = grid_config_hints;
        return *this;
    }

    general_wave_equation_builder &solver_config(const wave_implicit_solver_config_ptr &solver_config)
    {
        solver_config_ = solver_config;
        return *this;
    }

    general_wave_equation_builder &solver_config_details(const std::map<std::string, fp_type> &solver_config_details)
    {
        solver_config_details_ = solver_config_details;
        return *this;
    }

    sptr_t<general_wave_equation<fp_type, container, allocator>> build()
    {
        return std::make_shared<general_wave_equation<fp_type, container, allocator>>(
            wave_data_config_, discretization_config_, boundary_pair_, grid_config_hints_, solver_config_,
            solver_config_details_);
    }
};

} // namespace implicit_solvers

namespace explicit_solvers
{

template <typename fp_type, template <typename, typename> typename container = std::vector,
          typename allocator = std::allocator<fp_type>>
class general_wave_equation_builder
{
  private:
    wave_data_config_1d_ptr<fp_type> wave_data_config_;
    pde_discretization_config_1d_ptr<fp_type> discretization_config_;
    boundary_1d_pair<fp_type> boundary_pair_;
    grid_config_hints_1d_ptr<fp_type> grid_config_hints_;
    wave_explicit_solver_config_ptr solver_config_;

  public:
    general_wave_equation_builder &wave_data_config(const wave_data_config_1d_ptr<fp_type> &wave_data_config)
    {
        wave_data_config_ = wave_data_config;
        return *this;
    }

    general_wave_equation_builder &discretization_config(
        const pde_discretization_config_1d_ptr<fp_type> &discretization_config)
    {
        discretization_config_ = discretization_config;
        return *this;
    }

    general_wave_equation_builder &boundary_pair(const boundary_1d_pair<fp_type> &boundary_pair)
    {
        boundary_pair_ = boundary_pair;
        return *this;
    }

    general_wave_equation_builder &grid_config_hints(const grid_config_hints_1d_ptr<fp_type> &grid_config_hints)
    {
        grid_config_hints_ = grid_config_hints;
        return *this;
    }

    general_wave_equation_builder &solver_config(const wave_explicit_solver_config_ptr &solver_config)
    {
        solver_config_ = solver_config;
        return *this;
    }

    sptr_t<general_wave_equation<fp_type, container, allocator>> build()
    {
        return std::make_shared<general_wave_equation<fp_type, container, allocator>>(
            wave_data_config_, discretization_config_, boundary_pair_, grid_config_hints_, solver_config_);
    }
};

} // namespace explicit_solvers

} // namespace one_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_1D_GENERAL_WAVE_EQUATION_BUILDER_HPP_
