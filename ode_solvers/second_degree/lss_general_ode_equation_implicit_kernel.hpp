#if !defined(_LSS_GENERAL_ODE_EQUATION_IMPLICIT_KERNEL_HPP_)
#define _LSS_GENERAL_ODE_EQUATION_IMPLICIT_KERNEL_HPP_

#include <vector>

#include "boundaries/lss_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include "discretization/lss_discretization.hpp"
#include "discretization/lss_grid_config.hpp"
#include "ode_solvers/lss_ode_discretization_config.hpp"
#include "ode_solvers/lss_ode_solver_config.hpp"
#include "ode_solvers/second_degree/implicit_coefficients/lss_general_ode_implicit_coefficients.hpp"
#include "ode_solvers/second_degree/solver_method/lss_general_ode_implicit_solver_method.hpp"
#include "sparse_solvers/tridiagonal/cuda_solver/lss_cuda_solver.hpp"
#include "sparse_solvers/tridiagonal/double_sweep_solver/lss_double_sweep_solver.hpp"
#include "sparse_solvers/tridiagonal/sor_solver/lss_sor_solver.hpp"
#include "sparse_solvers/tridiagonal/sor_solver_cuda/lss_sor_solver_cuda.hpp"
#include "sparse_solvers/tridiagonal/thomas_lu_solver/lss_thomas_lu_solver.hpp"

namespace lss_ode_solvers
{

using lss_boundary::boundary_1d_pair;
using lss_boundary::neumann_boundary_1d;
using lss_boundary::robin_boundary_1d;
using lss_cuda_solver::cuda_solver;
using lss_double_sweep_solver::double_sweep_solver;
using lss_enumerations::memory_space_enum;
using lss_enumerations::tridiagonal_method_enum;
using lss_grids::grid_config_1d_ptr;
using lss_sor_solver::sor_solver;
using lss_sor_solver_cuda::sor_solver_cuda;
using lss_thomas_lu_solver::thomas_lu_solver;
using lss_utility::diagonal_triplet_t;
using lss_utility::NaN;
using lss_utility::pair_t;
using lss_utility::range;

template <memory_space_enum memory_enum, tridiagonal_method_enum tridiagonal_method, typename fp_type,
          template <typename, typename> typename container = std::vector, typename allocator = std::allocator<fp_type>>
class general_ode_equation_implicit_kernel
{
};

// ===================================================================
// ============================== DEVICE =============================
// ===================================================================
template <typename fp_type, template <typename, typename> typename container, typename allocator>
class general_ode_equation_implicit_kernel<memory_space_enum::Device, tridiagonal_method_enum::CUDASolver, fp_type,
                                           container, allocator>
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container<fp_type, allocator> container_t;
    typedef cuda_solver<memory_space_enum::Device, fp_type, container, allocator> cusolver;
    typedef ode_implicit_solver_method<fp_type, sptr_t<cusolver>, container, allocator> solver_method;

  private:
    boundary_1d_pair<fp_type> boundary_pair_;
    ode_data_config_ptr<fp_type> ode_data_cfg_;
    ode_discretization_config_ptr<fp_type> discretization_cfg_;
    ode_implicit_solver_config_ptr solver_cfg_;
    grid_config_1d_ptr<fp_type> grid_cfg_;

  public:
    general_ode_equation_implicit_kernel(boundary_1d_pair<fp_type> const &boundary_pair,
                                         ode_data_config_ptr<fp_type> const &ode_data_config,
                                         ode_discretization_config_ptr<fp_type> const &discretization_config,
                                         ode_implicit_solver_config_ptr const &solver_config,
                                         grid_config_1d_ptr<fp_type> const &grid_config)
        : boundary_pair_{boundary_pair}, ode_data_cfg_{ode_data_config}, discretization_cfg_{discretization_config},
          solver_cfg_{solver_config}, grid_cfg_{grid_config}
    {
    }

    void operator()(container_t &solution, bool is_ode_nonhom_set, std::function<fp_type(fp_type)> const &ode_nonhom)
    {
        // get space range:
        const range<fp_type> space = discretization_cfg_->space_range();
        // size of space discretization:
        const std::size_t space_size = discretization_cfg_->number_of_space_points();
        // create a ode coefficient holder:
        auto const ode_coeff_holder =
            std::make_shared<general_ode_implicit_coefficients<fp_type>>(ode_data_cfg_, discretization_cfg_);
        // create and set up the solver:
        auto const &solver = std::make_shared<cusolver>(space, space_size);
        solver->set_factorization(solver_cfg_->tridiagonal_factorization());
        auto const &solver_method_ptr = std::make_shared<solver_method>(solver, ode_coeff_holder, grid_cfg_);
        if (is_ode_nonhom_set)
        {
            solver_method_ptr->solve(boundary_pair_, ode_nonhom, solution);
        }
        else
        {
            solver_method_ptr->solve(boundary_pair_, solution);
        }
    }
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class general_ode_equation_implicit_kernel<memory_space_enum::Device, tridiagonal_method_enum::SORSolver, fp_type,
                                           container, allocator>
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container<fp_type, allocator> container_t;
    typedef sor_solver_cuda<fp_type, container, allocator> sorcusolver;
    typedef ode_implicit_solver_method<fp_type, sptr_t<sorcusolver>, container, allocator> solver_method;

  private:
    boundary_1d_pair<fp_type> boundary_pair_;
    ode_data_config_ptr<fp_type> ode_data_cfg_;
    ode_discretization_config_ptr<fp_type> discretization_cfg_;
    ode_implicit_solver_config_ptr solver_cfg_;
    grid_config_1d_ptr<fp_type> grid_cfg_;

  public:
    general_ode_equation_implicit_kernel(boundary_1d_pair<fp_type> const &boundary_pair,
                                         ode_data_config_ptr<fp_type> const &ode_data_config,
                                         ode_discretization_config_ptr<fp_type> const &discretization_config,
                                         ode_implicit_solver_config_ptr const &solver_config,
                                         grid_config_1d_ptr<fp_type> const &grid_config)
        : boundary_pair_{boundary_pair}, ode_data_cfg_{ode_data_config}, discretization_cfg_{discretization_config},
          solver_cfg_{solver_config}, grid_cfg_{grid_config}
    {
    }

    void operator()(container_t &solution, bool is_ode_nonhom_set, std::function<fp_type(fp_type)> const &ode_nonhom,
                    fp_type omega_value)
    {
        // get space range:
        const range<fp_type> space = discretization_cfg_->space_range();
        // size of space discretization:
        const std::size_t space_size = discretization_cfg_->number_of_space_points();
        // create a ode coefficient holder:
        auto const ode_coeff_holder =
            std::make_shared<general_ode_implicit_coefficients<fp_type>>(ode_data_cfg_, discretization_cfg_);
        // create and set up the solver:
        auto const &solver = std::make_shared<sorcusolver>(space, space_size);
        solver->set_omega(omega_value);
        auto const &solver_method_ptr = std::make_shared<solver_method>(solver, ode_coeff_holder, grid_cfg_);
        if (is_ode_nonhom_set)
        {
            solver_method_ptr->solve(boundary_pair_, ode_nonhom, solution);
        }
        else
        {
            solver_method_ptr->solve(boundary_pair_, solution);
        }
    }
};

// ===================================================================
// ================================ HOST =============================
// ===================================================================
template <typename fp_type, template <typename, typename> typename container, typename allocator>
class general_ode_equation_implicit_kernel<memory_space_enum::Host, tridiagonal_method_enum::CUDASolver, fp_type,
                                           container, allocator>
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container<fp_type, allocator> container_t;
    typedef cuda_solver<memory_space_enum::Host, fp_type, container, allocator> cusolver;
    typedef ode_implicit_solver_method<fp_type, sptr_t<cusolver>, container, allocator> solver_method;

  private:
    boundary_1d_pair<fp_type> boundary_pair_;
    ode_data_config_ptr<fp_type> ode_data_cfg_;
    ode_discretization_config_ptr<fp_type> discretization_cfg_;
    ode_implicit_solver_config_ptr solver_cfg_;
    grid_config_1d_ptr<fp_type> grid_cfg_;

  public:
    general_ode_equation_implicit_kernel(boundary_1d_pair<fp_type> const &boundary_pair,
                                         ode_data_config_ptr<fp_type> const &ode_data_config,
                                         ode_discretization_config_ptr<fp_type> const &discretization_config,
                                         ode_implicit_solver_config_ptr const &solver_config,
                                         grid_config_1d_ptr<fp_type> const &grid_config)
        : boundary_pair_{boundary_pair}, ode_data_cfg_{ode_data_config}, discretization_cfg_{discretization_config},
          solver_cfg_{solver_config}, grid_cfg_{grid_config}
    {
    }

    void operator()(container_t &solution, bool is_ode_nonhom_set, std::function<fp_type(fp_type)> const &ode_nonhom)
    {
        // get space range:
        const range<fp_type> space = discretization_cfg_->space_range();
        // size of space discretization:
        const std::size_t space_size = discretization_cfg_->number_of_space_points();
        // create a ode coefficient holder:
        auto const ode_coeff_holder =
            std::make_shared<general_ode_implicit_coefficients<fp_type>>(ode_data_cfg_, discretization_cfg_);
        // create and set up the solver:
        auto const &solver = std::make_shared<cusolver>(space, space_size);
        solver->set_factorization(solver_cfg_->tridiagonal_factorization());
        auto const &solver_method_ptr = std::make_shared<solver_method>(solver, ode_coeff_holder, grid_cfg_);
        if (is_ode_nonhom_set)
        {
            solver_method_ptr->solve(boundary_pair_, ode_nonhom, solution);
        }
        else
        {
            solver_method_ptr->solve(boundary_pair_, solution);
        }
    }
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class general_ode_equation_implicit_kernel<memory_space_enum::Host, tridiagonal_method_enum::SORSolver, fp_type,
                                           container, allocator>
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container<fp_type, allocator> container_t;
    typedef sor_solver<fp_type, container, allocator> sorsolver;
    typedef ode_implicit_solver_method<fp_type, sptr_t<sorsolver>, container, allocator> solver_method;

  private:
    boundary_1d_pair<fp_type> boundary_pair_;
    ode_data_config_ptr<fp_type> ode_data_cfg_;
    ode_discretization_config_ptr<fp_type> discretization_cfg_;
    ode_implicit_solver_config_ptr solver_cfg_;
    grid_config_1d_ptr<fp_type> grid_cfg_;

  public:
    general_ode_equation_implicit_kernel(boundary_1d_pair<fp_type> const &boundary_pair,
                                         ode_data_config_ptr<fp_type> const &ode_data_config,
                                         ode_discretization_config_ptr<fp_type> const &discretization_config,
                                         ode_implicit_solver_config_ptr const &solver_config,
                                         grid_config_1d_ptr<fp_type> const &grid_config)
        : boundary_pair_{boundary_pair}, ode_data_cfg_{ode_data_config}, discretization_cfg_{discretization_config},
          solver_cfg_{solver_config}, grid_cfg_{grid_config}
    {
    }

    void operator()(container_t &solution, bool is_ode_nonhom_set, std::function<fp_type(fp_type)> const &ode_nonhom,
                    fp_type omega_value)
    {
        // get space range:
        const range<fp_type> space = discretization_cfg_->space_range();
        // size of space discretization:
        const std::size_t space_size = discretization_cfg_->number_of_space_points();
        // create a ode coefficient holder:
        auto const ode_coeff_holder =
            std::make_shared<general_ode_implicit_coefficients<fp_type>>(ode_data_cfg_, discretization_cfg_);
        // create and set up the solver:
        auto const &solver = std::make_shared<sorsolver>(space, space_size);
        solver->set_omega(omega_value);
        auto const &solver_method_ptr = std::make_shared<solver_method>(solver, ode_coeff_holder, grid_cfg_);
        if (is_ode_nonhom_set)
        {
            solver_method_ptr->solve(boundary_pair_, ode_nonhom, solution);
        }
        else
        {
            solver_method_ptr->solve(boundary_pair_, solution);
        }
    }
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class general_ode_equation_implicit_kernel<memory_space_enum::Host, tridiagonal_method_enum::DoubleSweepSolver, fp_type,
                                           container, allocator>
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container<fp_type, allocator> container_t;
    typedef double_sweep_solver<fp_type, container, allocator> ds_solver;
    typedef ode_implicit_solver_method<fp_type, sptr_t<ds_solver>, container, allocator> solver_method;

  private:
    boundary_1d_pair<fp_type> boundary_pair_;
    ode_data_config_ptr<fp_type> ode_data_cfg_;
    ode_discretization_config_ptr<fp_type> discretization_cfg_;
    ode_implicit_solver_config_ptr solver_cfg_;
    grid_config_1d_ptr<fp_type> grid_cfg_;

  public:
    general_ode_equation_implicit_kernel(boundary_1d_pair<fp_type> const &boundary_pair,
                                         ode_data_config_ptr<fp_type> const &ode_data_config,
                                         ode_discretization_config_ptr<fp_type> const &discretization_config,
                                         ode_implicit_solver_config_ptr const &solver_config,
                                         grid_config_1d_ptr<fp_type> const &grid_config)
        : boundary_pair_{boundary_pair}, ode_data_cfg_{ode_data_config}, discretization_cfg_{discretization_config},
          solver_cfg_{solver_config}, grid_cfg_{grid_config}
    {
    }

    void operator()(container_t &solution, bool is_ode_nonhom_set, std::function<fp_type(fp_type)> const &ode_nonhom)
    {
        // get space range:
        const range<fp_type> space = discretization_cfg_->space_range();
        // size of space discretization:
        const std::size_t space_size = discretization_cfg_->number_of_space_points();
        // create a ode coefficient holder:
        auto const ode_coeff_holder =
            std::make_shared<general_ode_implicit_coefficients<fp_type>>(ode_data_cfg_, discretization_cfg_);
        // create and set up the solver:
        auto const &solver = std::make_shared<ds_solver>(space, space_size);
        auto const &solver_method_ptr = std::make_shared<solver_method>(solver, ode_coeff_holder, grid_cfg_);
        if (is_ode_nonhom_set)
        {
            solver_method_ptr->solve(boundary_pair_, ode_nonhom, solution);
        }
        else
        {
            solver_method_ptr->solve(boundary_pair_, solution);
        }
    }
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class general_ode_equation_implicit_kernel<memory_space_enum::Host, tridiagonal_method_enum::ThomasLUSolver, fp_type,
                                           container, allocator>
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container<fp_type, allocator> container_t;
    typedef thomas_lu_solver<fp_type, container, allocator> tlu_solver;
    typedef ode_implicit_solver_method<fp_type, sptr_t<tlu_solver>, container, allocator> solver_method;

  private:
    boundary_1d_pair<fp_type> boundary_pair_;
    ode_data_config_ptr<fp_type> ode_data_cfg_;
    ode_discretization_config_ptr<fp_type> discretization_cfg_;
    ode_implicit_solver_config_ptr solver_cfg_;
    grid_config_1d_ptr<fp_type> grid_cfg_;

  public:
    general_ode_equation_implicit_kernel(boundary_1d_pair<fp_type> const &boundary_pair,
                                         ode_data_config_ptr<fp_type> const &ode_data_config,
                                         ode_discretization_config_ptr<fp_type> const &discretization_config,
                                         ode_implicit_solver_config_ptr const &solver_config,
                                         grid_config_1d_ptr<fp_type> const &grid_config)
        : boundary_pair_{boundary_pair}, ode_data_cfg_{ode_data_config}, discretization_cfg_{discretization_config},
          solver_cfg_{solver_config}, grid_cfg_{grid_config}
    {
    }

    void operator()(container_t &solution, bool is_ode_nonhom_set, std::function<fp_type(fp_type)> const &ode_nonhom)
    {
        // get space range:
        const range<fp_type> space = discretization_cfg_->space_range();
        // size of space discretization:
        const std::size_t space_size = discretization_cfg_->number_of_space_points();
        // create a ode coefficient holder:
        auto const ode_coeff_holder =
            std::make_shared<general_ode_implicit_coefficients<fp_type>>(ode_data_cfg_, discretization_cfg_);
        // create and set up the solver:
        auto const &solver = std::make_shared<tlu_solver>(space, space_size);
        auto const &solver_method_ptr = std::make_shared<solver_method>(solver, ode_coeff_holder, grid_cfg_);
        if (is_ode_nonhom_set)
        {
            solver_method_ptr->solve(boundary_pair_, ode_nonhom, solution);
        }
        else
        {
            solver_method_ptr->solve(boundary_pair_, solution);
        }
    }
};

} // namespace lss_ode_solvers

#endif ///_LSS_GENERAL_ODE_EQUATION_IMPLICIT_KERNEL_HPP_
