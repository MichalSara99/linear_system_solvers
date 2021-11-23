#if !defined(_LSS_GENERAL_ODE_EQUATION_HPP_)
#define _LSS_GENERAL_ODE_EQUATION_HPP_

#include <functional>
#include <map>

#include "boundaries/lss_boundary.hpp"
#include "common/lss_macros.hpp"
#include "containers/lss_container_2d.hpp"
//#include "lss_general_ode_equation_explicit_kernel.hpp"
#include "discretization/lss_discretization.hpp"
#include "discretization/lss_grid_config.hpp"
#include "lss_general_ode_equation_implicit_kernel.hpp"
#include "ode_solvers/lss_ode_data_config.hpp"
#include "ode_solvers/lss_ode_discretization_config.hpp"
#include "ode_solvers/lss_ode_solver_config.hpp"
#include "ode_solvers/transformation/lss_ode_data_transform.hpp"
#include "transformation/lss_boundary_transform.hpp"

namespace lss_ode_solvers
{

using lss_boundary::boundary_1d_pair;
using lss_boundary::boundary_1d_ptr;
using lss_grids::grid_config_1d;
using lss_grids::grid_config_hints_1d_ptr;
using lss_grids::grid_transform_config_1d;
using lss_grids::grid_transform_config_1d_ptr;
using lss_transformation::boundary_transform_1d;
using lss_transformation::boundary_transform_1d_ptr;

namespace implicit_solvers
{

/*!
============================================================================
Represents general 2.degree ODE

 u''(x) + a(x)*u'(x) + b(x)*u(x) = g(x),
 x_1 < x < x_2

// ============================================================================
*/
template <typename fp_type, template <typename, typename> typename container = std::vector,
          typename allocator = std::allocator<fp_type>>
class general_ode_equation
{

  private:
    ode_data_transform_ptr<fp_type> ode_data_trans_cfg_;
    ode_discretization_config_ptr<fp_type> ode_discretization_cfg_;
    boundary_transform_1d_ptr<fp_type> boundary_;
    grid_transform_config_1d_ptr<fp_type> grid_trans_cfg_; // this may be removed as it is not used later
    ode_implicit_solver_config_ptr ode_solver_cfg_;
    std::map<std::string, fp_type> ode_solver_config_details_;

    explicit general_ode_equation() = delete;

    void initialize(ode_data_config_ptr<fp_type> const &ode_data_cfg,
                    grid_config_hints_1d_ptr<fp_type> const &grid_config_hints,
                    boundary_1d_pair<fp_type> const &boundary_pair)
    {
        LSS_VERIFY(ode_data_cfg, "ode_data_config must not be null");
        LSS_VERIFY(ode_discretization_cfg_, "ode_discretization_config must not be null");
        LSS_VERIFY(std::get<0>(boundary_pair), "boundary_pair.first must not be null");
        LSS_VERIFY(std::get<1>(boundary_pair), "boundary_pair.second must not be null");
        LSS_VERIFY(ode_solver_cfg_, "ode_solver_config must not be null");
        if (!ode_solver_config_details_.empty())
        {
            auto const &it = ode_solver_config_details_.find("sor_omega");
            LSS_ASSERT(it != ode_solver_config_details_.end(), "sor_omega is not defined");
        }
        // make necessary transformations:
        // create grid_transform_config:
        grid_trans_cfg_ =
            std::make_shared<grid_transform_config_1d<fp_type>>(ode_discretization_cfg_, grid_config_hints);
        // transform original heat data:
        ode_data_trans_cfg_ = std::make_shared<ode_data_transform<fp_type>>(ode_data_cfg, grid_trans_cfg_);
        // transform original boundary:
        boundary_ = std::make_shared<boundary_transform_1d<fp_type>>(boundary_pair, grid_trans_cfg_);
    }

  public:
    explicit general_ode_equation(
        ode_data_config_ptr<fp_type> const &ode_data_config,
        ode_discretization_config_ptr<fp_type> const &ode_discretization_config,
        boundary_1d_pair<fp_type> const &boundary_pair, grid_config_hints_1d_ptr<fp_type> const &grid_config_hints,
        ode_implicit_solver_config_ptr const &ode_solver_config = dev_cusolver_qr_solver_config_ptr,
        std::map<std::string, fp_type> const &ode_solver_config_details = std::map<std::string, fp_type>())
        : ode_discretization_cfg_{ode_discretization_config}, ode_solver_cfg_{ode_solver_config},
          ode_solver_config_details_{ode_solver_config_details}
    {
        initialize(ode_data_config, grid_config_hints, boundary_pair);
    }

    ~general_ode_equation()
    {
    }

    general_ode_equation(general_ode_equation const &) = delete;
    general_ode_equation(general_ode_equation &&) = delete;
    general_ode_equation &operator=(general_ode_equation const &) = delete;
    general_ode_equation &operator=(general_ode_equation &&) = delete;

    /**
     * Get the final solution of the PDE
     *
     * \param solution - container for solution
     */
    void solve(container<fp_type, allocator> &solution);
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void general_ode_equation<fp_type, container, allocator>::solve(container<fp_type, allocator> &solution)
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container<fp_type, allocator> container_t;

    LSS_ASSERT(solution.size() > 0, "The input solution container must be initialized");
    // size of space discretization:
    const std::size_t space_size = ode_discretization_cfg_->number_of_space_points();
    // This is the proper size of the container:
    LSS_ASSERT(solution.size() == space_size, "The input solution container must have the correct size");
    // grid:
    auto const &grid_cfg = std::make_shared<grid_config_1d<fp_type>>(ode_discretization_cfg_);
    auto const &boundary_pair = boundary_->boundary_pair();
    const bool is_ode_nonhom_set = ode_data_trans_cfg_->is_nonhom_data_set();
    // get ode_nonhom:
    auto const &ode_nonhom = ode_data_trans_cfg_->nonhom_function();

    if (ode_solver_cfg_->memory_space() == memory_space_enum::Device)
    {
        if (ode_solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::CUDASolver)
        {
            typedef general_ode_equation_implicit_kernel<memory_space_enum::Device, tridiagonal_method_enum::CUDASolver,
                                                         fp_type, container, allocator>
                dev_cu_solver;

            dev_cu_solver solver(boundary_pair, ode_data_trans_cfg_, ode_discretization_cfg_, ode_solver_cfg_,
                                 grid_cfg);
            solver(solution, is_ode_nonhom_set, ode_nonhom);
        }
        else if (ode_solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::SORSolver)
        {
            typedef general_ode_equation_implicit_kernel<memory_space_enum::Device, tridiagonal_method_enum::SORSolver,
                                                         fp_type, container, allocator>
                dev_sor_solver;
            LSS_ASSERT(!ode_solver_config_details_.empty(), "ode_solver_config_details map must not be empty");
            fp_type omega_value = ode_solver_config_details_["sor_omega"];
            dev_sor_solver solver(boundary_pair, ode_data_trans_cfg_, ode_discretization_cfg_, ode_solver_cfg_,
                                  grid_cfg);
            solver(solution, is_ode_nonhom_set, ode_nonhom, omega_value);
        }
        else
        {
            throw std::exception("Not supported on Device");
        }
    }
    else if (ode_solver_cfg_->memory_space() == memory_space_enum::Host)
    {
        if (ode_solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::CUDASolver)
        {
            typedef general_ode_equation_implicit_kernel<memory_space_enum::Host, tridiagonal_method_enum::CUDASolver,
                                                         fp_type, container, allocator>
                host_cu_solver;
            host_cu_solver solver(boundary_pair, ode_data_trans_cfg_, ode_discretization_cfg_, ode_solver_cfg_,
                                  grid_cfg);
            solver(solution, is_ode_nonhom_set, ode_nonhom);
        }
        else if (ode_solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::SORSolver)
        {
            typedef general_ode_equation_implicit_kernel<memory_space_enum::Host, tridiagonal_method_enum::SORSolver,
                                                         fp_type, container, allocator>
                host_sor_solver;

            LSS_ASSERT(!ode_solver_config_details_.empty(), "ode_solver_config_details map must not be empty");
            fp_type omega_value = ode_solver_config_details_["sor_omega"];
            host_sor_solver solver(boundary_pair, ode_data_trans_cfg_, ode_discretization_cfg_, ode_solver_cfg_,
                                   grid_cfg);
            solver(solution, is_ode_nonhom_set, ode_nonhom, omega_value);
        }
        else if (ode_solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::DoubleSweepSolver)
        {
            typedef general_ode_equation_implicit_kernel<
                memory_space_enum::Host, tridiagonal_method_enum::DoubleSweepSolver, fp_type, container, allocator>
                host_dss_solver;
            host_dss_solver solver(boundary_pair, ode_data_trans_cfg_, ode_discretization_cfg_, ode_solver_cfg_,
                                   grid_cfg);
            solver(solution, is_ode_nonhom_set, ode_nonhom);
        }
        else if (ode_solver_cfg_->tridiagonal_method() == tridiagonal_method_enum::ThomasLUSolver)
        {
            typedef general_ode_equation_implicit_kernel<
                memory_space_enum::Host, tridiagonal_method_enum::ThomasLUSolver, fp_type, container, allocator>
                host_lus_solver;
            host_lus_solver solver(boundary_pair, ode_data_trans_cfg_, ode_discretization_cfg_, ode_solver_cfg_,
                                   grid_cfg);
            solver(solution, is_ode_nonhom_set, ode_nonhom);
        }
        else
        {
            throw std::exception("Not supported on Host");
        }
    }
    else
    {
        throw std::exception("Unreachable");
    }
}

} // namespace implicit_solvers

namespace explicit_solvers
{

} // namespace explicit_solvers

} // namespace lss_ode_solvers

#endif ///_LSS_GENERAL_ODE_EQUATION_HPP_
