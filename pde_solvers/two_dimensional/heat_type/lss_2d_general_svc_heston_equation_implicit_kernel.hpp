#if !defined(_LSS_2D_GENERAL_SVC_HESTON_EQUATION_IMPLICIT_KERNEL_HPP_)
#define _LSS_2D_GENERAL_SVC_HESTON_EQUATION_IMPLICIT_KERNEL_HPP_

#include <vector>

#include "boundaries/lss_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include "containers/lss_container_2d.hpp"
#include "containers/lss_container_3d.hpp"
#include "discretization/lss_discretization.hpp"
#include "pde_solvers/lss_heat_data_config.hpp"
#include "pde_solvers/lss_heat_solver_config.hpp"
#include "pde_solvers/lss_pde_discretization_config.hpp"
#include "sparse_solvers/tridiagonal/cuda_solver/lss_cuda_solver.hpp"
#include "sparse_solvers/tridiagonal/double_sweep_solver/lss_double_sweep_solver.hpp"
#include "sparse_solvers/tridiagonal/sor_solver/lss_sor_solver.hpp"
#include "sparse_solvers/tridiagonal/sor_solver_cuda/lss_sor_solver_cuda.hpp"
#include "sparse_solvers/tridiagonal/thomas_lu_solver/lss_thomas_lu_solver.hpp"
#include "splitting_method/lss_heat_douglas_rachford_method.hpp"

namespace lss_pde_solvers
{
namespace two_dimensional
{

using lss_boundary::boundary_2d_pair;
using lss_boundary::boundary_2d_ptr;
using lss_boundary::neumann_boundary_1d;
using lss_boundary::robin_boundary_1d;
using lss_containers::container_2d;
using lss_containers::container_3d;
using lss_cuda_solver::cuda_solver;
using lss_double_sweep_solver::double_sweep_solver;
using lss_enumerations::by_enum;
using lss_enumerations::dimension_enum;
using lss_enumerations::implicit_pde_schemes_enum;
using lss_enumerations::memory_space_enum;
using lss_enumerations::traverse_direction_enum;
using lss_enumerations::tridiagonal_method_enum;
using lss_sor_solver::sor_solver;
using lss_sor_solver_cuda::sor_solver_cuda;
using lss_thomas_lu_solver::thomas_lu_solver;
using lss_utility::diagonal_triplet_t;
using lss_utility::function_triplet_t;
using lss_utility::NaN;
using lss_utility::pair_t;
using lss_utility::range;

template <memory_space_enum memory_enum, tridiagonal_method_enum tridiagonal_method, typename fp_type,
          template <typename, typename> typename container = std::vector, typename allocator = std::allocator<fp_type>>
class general_svc_heston_equation_implicit_kernel
{
};

// ===================================================================
// ============================== DEVICE =============================
// ===================================================================
template <typename fp_type, template <typename, typename> typename container, typename allocator>
class general_svc_heston_equation_implicit_kernel<memory_space_enum::Device, tridiagonal_method_enum::CUDASolver,
                                                  fp_type, container, allocator>
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container_2d<by_enum::Row, fp_type, container, allocator> rcontainer_2d_t;
    typedef cuda_solver<memory_space_enum::Device, fp_type, container, allocator> cusolver;
    typedef heat_douglas_rachford_method<fp_type, cusolver, container, allocator> douglas_rachford_method;
    // typedef heat_time_loop<fp_type, container, allocator> loop;

  private:
    boundary_2d_ptr<fp_type> boundary_hor_;
    boundary_2d_pair<fp_type> boundary_pair_ver_;
    heat_data_config_2d_ptr<fp_type> heat_data_cfg_;
    pde_discretization_config_1d_ptr<fp_type> discretization_cfg_;
    heat_implicit_solver_config_ptr solver_cfg_;

  public:
    general_svc_heston_equation_implicit_kernel(boundary_2d_ptr<fp_type> const &horizontal_upper_boundary_ptr,
                                                boundary_2d_pair<fp_type> const &vertical_boundary_pair,
                                                heat_data_config_2d_ptr<fp_type> const &heat_data_config,
                                                pde_discretization_config_1d_ptr<fp_type> const &discretization_config,
                                                heat_implicit_solver_config_ptr const &solver_config)
        : boundary_hor_{horizontal_upper_boundary_ptr}, boundary_pair_ver_{vertical_boundary_pair},
          heat_data_cfg_{heat_data_config}, discretization_cfg_{discretization_config}, solver_cfg_{solver_config}
    {
    }

    void operator()(rcontainer_2d_t &prev_solution, rcontainer_2d_t &next_solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type, fp_type)> const &heat_source)
    {
        // get space ranges:
        const auto &spaces = discretization_cfg_->space_range();
        // across X:
        const auto space_x = std::get<0>(spaces);
        // across Y:
        const auto space_y = std::get<1>(spaces);
        // get space steps:
        const auto &hs = discretization_cfg_->space_step();
        // across X:
        const fp_type h_1 = std::get<0>(hs);
        // across Y:
        const fp_type h_2 = std::get<1>(hs);
        // get time range:
        const range<fp_type> time = discretization_cfg_->time_range();
        // time step:
        const fp_type k = discretization_cfg_->time_step();
        // size of spaces discretization:
        const auto &space_sizes = discretization_cfg_->number_of_space_points();
        const std::size_t space_size_x = std::get<0>(space_sizes);
        const std::size_t space_size_y = std::get<1>(space_sizes);
        // last time index:
        const std::size_t last_time_idx = discretization_cfg_->number_of_time_points() - 1;
        // save traverse_direction
        const traverse_direction_enum traverse_dir = solver_cfg_->traverse_direction();
        // get propper theta accoring to clients chosen scheme:
        const fp_type half = static_cast<fp_type>(0.5);
        const fp_type one = static_cast<fp_type>(1.0);
        fp_type theta{};
        if (solver_cfg_->implicit_pde_scheme() == implicit_pde_schemes_enum::Euler)
        {
            theta = one;
        }
        else if (solver_cfg_->implicit_pde_scheme() == implicit_pde_schemes_enum::CrankNicolson)
        {
            theta = half;
        }
        else
        {
            throw std::exception("Unreachable");
        }
        // create and set up the main solvers:
        auto const &solver_0 = std::make_shared<cusolver>(space_x, space_size_x);
        solver_0->set_factorization(solver_cfg_->tridiagonal_factorization());
        auto const &solver_1 = std::make_shared<cusolver>(space_y, space_size_y);
        solver_1->set_factorization(solver_cfg_->tridiagonal_factorization());
        auto const &solver =
            std::make_shared<douglas_rachford_method>(solver_0, solver_1, heat_data_cfg_, discretization_cfg_, theta);
        // create and set up the horixontal boundary solver:

        if (is_heat_sourse_set)
        {
            // auto scheme_function =
            //    implicit_heat_scheme<fp_type, container, allocator>::get(solver_cfg_->implicit_pde_scheme(), false);
            //// create a container to carry discretized source heat
            // container_t source_curr(space_size, NaN<fp_type>());
            // container_t source_next(space_size, NaN<fp_type>());
            // loop::run(solver, scheme_function, boundary_pair_, fun_triplet_, space, time, last_time_idx, steps,
            //          traverse_dir, prev_solution, next_solution, rhs, heat_source, source_curr, source_next);
        }
        else
        {
            // loop::run(solver, boundary_pair_, fun_triplet_, space, time, last_time_idx, steps, traverse_dir,
            //          prev_solution, next_solution, rhs);
        }
    }

    void operator()(rcontainer_2d_t &prev_solution, rcontainer_2d_t &next_solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type, fp_type)> const &heat_source,
                    container_3d<fp_type, container, allocator> &solutions)
    {
    }
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class general_svc_heston_equation_implicit_kernel<memory_space_enum::Device, tridiagonal_method_enum::SORSolver,
                                                  fp_type, container, allocator>
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container_2d<by_enum::Row, fp_type, container, allocator> rcontainer_2d_t;
    typedef sor_solver_cuda<fp_type, container, allocator> sorcusolver;
    typedef heat_douglas_rachford_method<fp_type, sorcusolver, container, allocator> douglas_rachford_method;
    // typedef heat_time_loop<fp_type, container, allocator> loop;

  private:
    boundary_2d_ptr<fp_type> boundary_hor_;
    boundary_2d_pair<fp_type> boundary_pair_ver_;
    heat_data_config_2d_ptr<fp_type> heat_data_cfg_;
    pde_discretization_config_1d_ptr<fp_type> discretization_cfg_;
    heat_implicit_solver_config_ptr solver_cfg_;

  public:
    general_svc_heston_equation_implicit_kernel(boundary_2d_ptr<fp_type> const &horizontal_upper_boundary_ptr,
                                                boundary_2d_pair<fp_type> const &vertical_boundary_pair,
                                                heat_data_config_2d_ptr<fp_type> const &heat_data_config,
                                                pde_discretization_config_1d_ptr<fp_type> const &discretization_config,
                                                heat_implicit_solver_config_ptr const &solver_config)
        : boundary_hor_{horizontal_upper_boundary_ptr}, boundary_pair_ver_{vertical_boundary_pair},
          heat_data_cfg_{heat_data_config}, discretization_cfg_{discretization_config}, solver_cfg_{solver_config}
    {
    }

    void operator()(rcontainer_2d_t &prev_solution, rcontainer_2d_t &next_solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type, fp_type)> const &heat_source, fp_type omega_value)
    {
    }

    void operator()(rcontainer_2d_t &prev_solution, rcontainer_2d_t &next_solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type, fp_type)> const &heat_source, fp_type omega_value,
                    container_3d<fp_type, container, allocator> &solutions)
    {
    }
};

// ===================================================================
// ================================ HOST =============================
// ===================================================================
template <typename fp_type, template <typename, typename> typename container, typename allocator>
class general_svc_heston_equation_implicit_kernel<memory_space_enum::Host, tridiagonal_method_enum::CUDASolver, fp_type,
                                                  container, allocator>
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container_2d<by_enum::Row, fp_type, container, allocator> rcontainer_2d_t;
    typedef cuda_solver<memory_space_enum::Host, fp_type, container, allocator> cusolver;
    typedef heat_douglas_rachford_method<fp_type, cusolver, container, allocator> douglas_rachford_method;
    // typedef heat_time_loop<fp_type, container, allocator> loop;

  private:
    boundary_2d_ptr<fp_type> boundary_hor_;
    boundary_2d_pair<fp_type> boundary_pair_ver_;
    heat_data_config_2d_ptr<fp_type> heat_data_cfg_;
    pde_discretization_config_1d_ptr<fp_type> discretization_cfg_;
    heat_implicit_solver_config_ptr solver_cfg_;

  public:
    general_svc_heston_equation_implicit_kernel(boundary_2d_ptr<fp_type> const &horizontal_upper_boundary_ptr,
                                                boundary_2d_pair<fp_type> const &vertical_boundary_pair,
                                                heat_data_config_2d_ptr<fp_type> const &heat_data_config,
                                                pde_discretization_config_1d_ptr<fp_type> const &discretization_config,
                                                heat_implicit_solver_config_ptr const &solver_config)
        : boundary_hor_{horizontal_upper_boundary_ptr}, boundary_pair_ver_{vertical_boundary_pair},
          heat_data_cfg_{heat_data_config}, discretization_cfg_{discretization_config}, solver_cfg_{solver_config}
    {
    }

    void operator()(rcontainer_2d_t &prev_solution, rcontainer_2d_t &next_solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type, fp_type)> const &heat_source)
    {
    }

    void operator()(rcontainer_2d_t &prev_solution, rcontainer_2d_t &next_solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type)> const &heat_source,
                    container_3d<fp_type, container, allocator> &solutions)
    {
    }
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class general_svc_heston_equation_implicit_kernel<memory_space_enum::Host, tridiagonal_method_enum::SORSolver, fp_type,
                                                  container, allocator>
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container_2d<by_enum::Row, fp_type, container, allocator> rcontainer_2d_t;
    typedef sor_solver<fp_type, container, allocator> sorsolver;
    typedef heat_douglas_rachford_method<fp_type, sorsolver, container, allocator> douglas_rachford_method;
    // typedef heat_time_loop<fp_type, container, allocator> loop;

  private:
    boundary_2d_ptr<fp_type> boundary_hor_;
    boundary_2d_pair<fp_type> boundary_pair_ver_;
    heat_data_config_2d_ptr<fp_type> heat_data_cfg_;
    pde_discretization_config_1d_ptr<fp_type> discretization_cfg_;
    heat_implicit_solver_config_ptr solver_cfg_;

  public:
    general_svc_heston_equation_implicit_kernel(boundary_2d_ptr<fp_type> const &horizontal_upper_boundary_ptr,
                                                boundary_2d_pair<fp_type> const &vertical_boundary_pair,
                                                heat_data_config_2d_ptr<fp_type> const &heat_data_config,
                                                pde_discretization_config_1d_ptr<fp_type> const &discretization_config,
                                                heat_implicit_solver_config_ptr const &solver_config)
        : boundary_hor_{horizontal_upper_boundary_ptr}, boundary_pair_ver_{vertical_boundary_pair},
          heat_data_cfg_{heat_data_config}, discretization_cfg_{discretization_config}, solver_cfg_{solver_config}
    {
    }

    void operator()(rcontainer_2d_t &prev_solution, rcontainer_2d_t &next_solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type, fp_type)> const &heat_source, fp_type omega_value)
    {
    }

    void operator()(rcontainer_2d_t &prev_solution, rcontainer_2d_t &next_solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type, fp_type)> const &heat_source, fp_type omega_value,
                    container_3d<fp_type, container, allocator> &solutions)
    {
    }
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class general_svc_heston_equation_implicit_kernel<memory_space_enum::Host, tridiagonal_method_enum::DoubleSweepSolver,
                                                  fp_type, container, allocator>
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container_2d<by_enum::Row, fp_type, container, allocator> rcontainer_2d_t;
    typedef double_sweep_solver<fp_type, container, allocator> ds_solver;
    typedef heat_douglas_rachford_method<fp_type, ds_solver, container, allocator> douglas_rachford_method;
    // typedef heat_time_loop<fp_type, container, allocator> loop;

  private:
    boundary_2d_ptr<fp_type> boundary_hor_;
    boundary_2d_pair<fp_type> boundary_pair_ver_;
    heat_data_config_2d_ptr<fp_type> heat_data_cfg_;
    pde_discretization_config_1d_ptr<fp_type> discretization_cfg_;
    heat_implicit_solver_config_ptr solver_cfg_;

  public:
    general_svc_heston_equation_implicit_kernel(boundary_2d_ptr<fp_type> const &horizontal_upper_boundary_ptr,
                                                boundary_2d_pair<fp_type> const &vertical_boundary_pair,
                                                heat_data_config_2d_ptr<fp_type> const &heat_data_config,
                                                pde_discretization_config_1d_ptr<fp_type> const &discretization_config,
                                                heat_implicit_solver_config_ptr const &solver_config)
        : boundary_hor_{horizontal_upper_boundary_ptr}, boundary_pair_ver_{vertical_boundary_pair},
          heat_data_cfg_{heat_data_config}, discretization_cfg_{discretization_config}, solver_cfg_{solver_config}
    {
    }

    void operator()(rcontainer_2d_t &prev_solution, rcontainer_2d_t &next_solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type, fp_type)> const &heat_source)
    {
    }

    void operator()(rcontainer_2d_t &prev_solution, rcontainer_2d_t &next_solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type, fp_type)> const &heat_source,
                    container_3d<fp_type, container, allocator> &solutions)
    {
    }
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class general_svc_heston_equation_implicit_kernel<memory_space_enum::Host, tridiagonal_method_enum::ThomasLUSolver,
                                                  fp_type, container, allocator>
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container_2d<by_enum::Row, fp_type, container, allocator> rcontainer_2d_t;
    typedef thomas_lu_solver<fp_type, container, allocator> tlu_solver;
    typedef heat_douglas_rachford_method<fp_type, tlu_solver, container, allocator> douglas_rachford_method;
    // typedef heat_time_loop<fp_type, container, allocator> loop;

  private:
    boundary_2d_ptr<fp_type> boundary_hor_;
    boundary_2d_pair<fp_type> boundary_pair_ver_;
    heat_data_config_2d_ptr<fp_type> heat_data_cfg_;
    pde_discretization_config_1d_ptr<fp_type> discretization_cfg_;
    heat_implicit_solver_config_ptr solver_cfg_;

  public:
    general_svc_heston_equation_implicit_kernel(boundary_2d_ptr<fp_type> const &horizontal_upper_boundary_ptr,
                                                boundary_2d_pair<fp_type> const &vertical_boundary_pair,
                                                heat_data_config_2d_ptr<fp_type> const &heat_data_config,
                                                pde_discretization_config_1d_ptr<fp_type> const &discretization_config,
                                                heat_implicit_solver_config_ptr const &solver_config)
        : boundary_hor_{horizontal_upper_boundary_ptr}, boundary_pair_ver_{vertical_boundary_pair},
          heat_data_cfg_{heat_data_config}, discretization_cfg_{discretization_config}, solver_cfg_{solver_config}
    {
    }

    void operator()(rcontainer_2d_t &prev_solution, rcontainer_2d_t &next_solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type, fp_type)> const &heat_source)
    {
    }

    void operator()(rcontainer_2d_t &prev_solution, rcontainer_2d_t &next_solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type, fp_type)> const &heat_source,
                    container_3d<fp_type, container, allocator> &solutions)
    {
    }
};
} // namespace two_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_2D_GENERAL_SVC_HESTON_EQUATION_IMPLICIT_KERNEL_HPP_
