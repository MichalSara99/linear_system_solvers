#if !defined(_LSS_2D_GENERAL_SVC_HESTON_EQUATION_IMPLICIT_KERNEL_HPP_)
#define _LSS_2D_GENERAL_SVC_HESTON_EQUATION_IMPLICIT_KERNEL_HPP_

#include <vector>

#include "boundaries/lss_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include "containers/lss_container_2d.hpp"
#include "containers/lss_container_3d.hpp"
#include "discretization/lss_discretization.hpp"
#include "discretization/lss_grid.hpp"
#include "discretization/lss_grid_config.hpp"
//#include "lss_2d_general_svc_heston_equation_implicit_boundary.hpp"
#include "lss_2d_general_svc_heston_equation_explicit_boundary.hpp"
#include "pde_solvers/lss_heat_solver_config.hpp"
#include "pde_solvers/lss_pde_discretization_config.hpp"
#include "pde_solvers/lss_splitting_method_config.hpp"
#include "pde_solvers/lss_weighted_scheme_config.hpp"
#include "pde_solvers/transformation/lss_heat_data_transform.hpp"
#include "pde_solvers/two_dimensional/heat_type/implicit_coefficients/lss_2d_general_svc_heston_equation_coefficients.hpp"
#include "sparse_solvers/pentadiagonal/karawia_solver/lss_karawia_solver.hpp"
#include "sparse_solvers/tridiagonal/cuda_solver/lss_cuda_solver.hpp"
#include "sparse_solvers/tridiagonal/double_sweep_solver/lss_double_sweep_solver.hpp"
#include "sparse_solvers/tridiagonal/sor_solver/lss_sor_solver.hpp"
#include "sparse_solvers/tridiagonal/sor_solver_cuda/lss_sor_solver_cuda.hpp"
#include "sparse_solvers/tridiagonal/thomas_lu_solver/lss_thomas_lu_solver.hpp"
#include "splitting_method/lss_heat_craig_sneyd_method.hpp"
#include "splitting_method/lss_heat_douglas_rachford_method.hpp"
#include "splitting_method/lss_heat_hundsdorfer_verwer_method.hpp"
#include "splitting_method/lss_heat_modified_craig_sneyd_method.hpp"
#include "splitting_method/lss_heat_splitting_method.hpp"

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
using lss_grids::grid_config_1d_ptr;
using lss_grids::grid_config_2d_ptr;
using lss_sor_solver::sor_solver;
using lss_sor_solver_cuda::sor_solver_cuda;
using lss_thomas_lu_solver::thomas_lu_solver;
using lss_utility::diagonal_triplet_t;
using lss_utility::function_triplet_t;
using lss_utility::NaN;
using lss_utility::pair_t;
using lss_utility::range;
using lss_utility::sptr_t;

/**
    implicit_heston_solver_boundaries object
 */
template <typename fp_type, template <typename, typename> typename container, typename allocator>
class implicit_heston_solver_boundaries
{
    typedef container_2d<by_enum::Row, fp_type, container, allocator> container_2d_t;

  public:
    static boundary_2d_pair<fp_type> get_vertical(grid_config_1d_ptr<fp_type> const &grid_config_x,
                                                  container_2d_t const &next_solution)
    {
        auto const lci = next_solution.columns() - 1;
        auto lower = [=](fp_type t, fp_type x) -> fp_type {
            const std::size_t i = grid_config_x->index_of(x);
            return next_solution(i, 0);
        };
        auto upper = [=](fp_type t, fp_type x) -> fp_type {
            const std::size_t i = grid_config_x->index_of(x);
            return next_solution(i, lci);
        };
        auto vertical_low = std::make_shared<dirichlet_boundary_2d<fp_type>>(lower);
        auto vertical_high = std::make_shared<dirichlet_boundary_2d<fp_type>>(upper);
        return std::make_pair(vertical_low, vertical_high);
    }

    static boundary_2d_pair<fp_type> get_intermed_horizontal(grid_config_1d_ptr<fp_type> const &grid_config_y,
                                                             container_2d_t const &next_solution)
    {
        const std::size_t lri = next_solution.rows() - 1;
        auto lower = [=](fp_type t, fp_type y) -> fp_type {
            const std::size_t j = grid_config_y->index_of(y);
            return next_solution(0, j);
        };
        auto upper = [=](fp_type t, fp_type y) -> fp_type {
            const std::size_t j = grid_config_y->index_of(y);
            return next_solution(lri, j);
        };
        auto horizontal_low = std::make_shared<dirichlet_boundary_2d<fp_type>>(lower);
        auto horizontal_high = std::make_shared<dirichlet_boundary_2d<fp_type>>(upper);
        return std::make_pair(horizontal_low, horizontal_high);
    }
};

/**
 * implicit_heston_time_loop object
 */
template <typename fp_type, template <typename, typename> typename container, typename allocator>
class implicit_heston_time_loop
{
    typedef container<fp_type, allocator> container_t;
    typedef container_2d<by_enum::Row, fp_type, container, allocator> container_2d_t;

  public:
    template <typename solver, typename boundary_solver>
    static void run(solver const &solver_ptr, boundary_solver const &boundary_solver_ptr,
                    boundary_2d_pair<fp_type> const &horizontal_boundary_pair,
                    boundary_2d_ptr<fp_type> const &vertical_upper_boundary_ptr,
                    grid_config_2d_ptr<fp_type> const &grid_config, range<fp_type> const &time_range,
                    std::size_t const &last_time_idx, fp_type const time_step,
                    std::pair<fp_type, fp_type> const &weights, std::pair<fp_type, fp_type> const &weight_values,
                    traverse_direction_enum const &traverse_dir, container_2d_t &prev_solution,
                    container_2d_t &next_solution);

    //  time, last_time_idx, steps, traverse_dir, prev_solution, next_solution

    // static void run(function_quintuple_t<fp_type> const &fun_quintuple, boundary_1d_pair<fp_type> const
    // &boundary_pair,
    //                range<fp_type> const &space_range, range<fp_type> const &time_range,
    //                std::size_t const &last_time_idx, std::pair<fp_type, fp_type> const &steps,
    //                traverse_direction_enum const &traverse_dir, container_t &prev_solution_0,
    //                container_t &prev_solution_1, container_t &next_solution,
    //                std::function<fp_type(fp_type, fp_type)> const &wave_source, container_t &source);

    // static void run_with_stepping(function_quintuple_t<fp_type> const &fun_quintuple,
    //                              boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &space_range,
    //                              range<fp_type> const &time_range, std::size_t const &last_time_idx,
    //                              std::pair<fp_type, fp_type> const &steps, traverse_direction_enum const
    //                              &traverse_dir, container_t &prev_solution_0, container_t &prev_solution_1,
    //                              container_t &next_solution, container_2d_t &solutions);

    // static void run_with_stepping(function_quintuple_t<fp_type> const &fun_quintuple,
    //                              boundary_1d_pair<fp_type> const &boundary_pair, range<fp_type> const &space_range,
    //                              range<fp_type> const &time_range, std::size_t const &last_time_idx,
    //                              std::pair<fp_type, fp_type> const &steps, traverse_direction_enum const
    //                              &traverse_dir, container_t &prev_solution_0, container_t &prev_solution_1,
    //                              container_t &next_solution,
    //                              std::function<fp_type(fp_type, fp_type)> const &wave_source, container_t &source,
    //                              container_2d_t &solutions);
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
template <typename solver, typename boundary_solver>
void implicit_heston_time_loop<fp_type, container, allocator>::run(
    solver const &solver_ptr, boundary_solver const &boundary_solver_ptr,
    boundary_2d_pair<fp_type> const &horizontal_boundary_pair,
    boundary_2d_ptr<fp_type> const &vertical_upper_boundary_ptr, grid_config_2d_ptr<fp_type> const &grid_config,
    range<fp_type> const &time_range, std::size_t const &last_time_idx, fp_type const time_step,
    std::pair<fp_type, fp_type> const &weights, std::pair<fp_type, fp_type> const &weight_values,
    traverse_direction_enum const &traverse_dir, container_2d_t &prev_solution, container_2d_t &next_solution)
{

    typedef implicit_heston_solver_boundaries<fp_type, container, allocator> boundaries;

    const fp_type start_time = time_range.lower();
    const fp_type end_time = time_range.upper();
    const fp_type k = time_step;
    boundary_2d_pair<fp_type> ver_boundary_pair;
    boundary_2d_pair<fp_type> hor_inter_boundary_pair;

    fp_type time{};
    std::size_t time_idx{};

    if (traverse_dir == traverse_direction_enum::Forward)
    {
        time = start_time + k;
        time_idx = 1;
        while (time_idx <= last_time_idx)
        {
            boundary_solver_ptr->solve(prev_solution, horizontal_boundary_pair, vertical_upper_boundary_ptr, time,
                                       next_solution);
            ver_boundary_pair = boundaries::get_vertical(grid_config->grid_1(), next_solution);
            hor_inter_boundary_pair = boundaries::get_intermed_horizontal(grid_config->grid_2(), prev_solution);
            solver_ptr->solve(prev_solution, hor_inter_boundary_pair, ver_boundary_pair, time, weights, weight_values,
                              next_solution);
            boundary_solver_ptr->solve(prev_solution, horizontal_boundary_pair, time, next_solution);

            prev_solution = next_solution;
            time += k;
            time_idx++;
        }
    }
    else
    {
        time = end_time - k;
        time_idx = last_time_idx;
        do
        {
            time_idx--;
            boundary_solver_ptr->solve(prev_solution, horizontal_boundary_pair, vertical_upper_boundary_ptr, time,
                                       next_solution);
            ver_boundary_pair = boundaries::get_vertical(grid_config->grid_1(), next_solution);
            hor_inter_boundary_pair = boundaries::get_intermed_horizontal(grid_config->grid_2(), prev_solution);
            solver_ptr->solve(prev_solution, hor_inter_boundary_pair, ver_boundary_pair, time, weights, weight_values,
                              next_solution);
            boundary_solver_ptr->solve(prev_solution, horizontal_boundary_pair, time, next_solution);

            prev_solution = next_solution;
            time -= k;
        } while (time_idx > 0);
    }
}

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
    typedef container_3d<by_enum::Row, fp_type, container, allocator> rcontainer_3d_t;
    typedef cuda_solver<memory_space_enum::Device, fp_type, container, allocator> cusolver;
    typedef heat_douglas_rachford_method<fp_type, sptr_t<cusolver>, container, allocator> douglas_rachford_method;
    typedef heat_craig_sneyd_method<fp_type, sptr_t<cusolver>, container, allocator> craig_sneyd_method;
    typedef heat_modified_craig_sneyd_method<fp_type, sptr_t<cusolver>, container, allocator> m_craig_sneyd_method;
    typedef heat_hundsdorfer_verwer_method<fp_type, sptr_t<cusolver>, container, allocator> hundsdorfer_verwer_method;
    typedef general_svc_heston_equation_explicit_boundary<fp_type, container, allocator> explicit_boundary;
    typedef implicit_heston_time_loop<fp_type, container, allocator> loop;

  private:
    boundary_2d_ptr<fp_type> boundary_ver_;
    boundary_2d_pair<fp_type> boundary_pair_hor_;
    heat_data_transform_2d_ptr<fp_type> heat_data_cfg_;
    pde_discretization_config_2d_ptr<fp_type> discretization_cfg_;
    splitting_method_config_ptr<fp_type> splitting_cfg_;
    weighted_scheme_config_ptr<fp_type> weighted_scheme_cfg_;
    heat_implicit_solver_config_ptr solver_cfg_;
    grid_config_2d_ptr<fp_type> grid_cfg_;

  public:
    general_svc_heston_equation_implicit_kernel(boundary_2d_ptr<fp_type> const &vertical_upper_boundary_ptr,
                                                boundary_2d_pair<fp_type> const &horizontal_boundary_pair,
                                                heat_data_transform_2d_ptr<fp_type> const &heat_data_config,
                                                pde_discretization_config_2d_ptr<fp_type> const &discretization_config,
                                                splitting_method_config_ptr<fp_type> const &splitting_config,
                                                weighted_scheme_config_ptr<fp_type> const &weighted_scheme_config,
                                                heat_implicit_solver_config_ptr const &solver_config,
                                                grid_config_2d_ptr<fp_type> const &grid_config)
        : boundary_ver_{vertical_upper_boundary_ptr}, boundary_pair_hor_{horizontal_boundary_pair},
          heat_data_cfg_{heat_data_config}, discretization_cfg_{discretization_config},
          splitting_cfg_{splitting_config}, weighted_scheme_cfg_{weighted_scheme_config},
          solver_cfg_{solver_config}, grid_cfg_{grid_config}
    {
    }

    void operator()(rcontainer_2d_t &prev_solution, rcontainer_2d_t &next_solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type, fp_type)> const &heat_source)
    {
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
        const fp_type one = static_cast<fp_type>(1.0);
        const fp_type half = static_cast<fp_type>(0.5);
        const fp_type o_three = static_cast<fp_type>(0.3);
        const fp_type o_eight = static_cast<fp_type>(0.8);
        fp_type theta{};
        switch (solver_cfg_->implicit_pde_scheme())
        {
        case implicit_pde_schemes_enum::Euler:
            theta = one;
            break;
        case implicit_pde_schemes_enum::Theta_30:
            theta = o_three;
            break;
        case implicit_pde_schemes_enum::CrankNicolson:
            theta = half;
            break;
        case implicit_pde_schemes_enum::Theta_80:
            theta = o_eight;
            break;
        default:
            theta = half;
        }

        // create a Heston coefficient holder:
        auto const heston_coeff_holder = std::make_shared<general_svc_heston_equation_coefficients<fp_type>>(
            heat_data_cfg_, discretization_cfg_, splitting_cfg_, theta);
        heat_splitting_method_ptr<fp_type, container, allocator> splitting_ptr;
        auto solver_y = std::make_shared<cusolver>(space_size_x);
        solver_y->set_factorization(solver_cfg_->tridiagonal_factorization());
        auto solver_u = std::make_shared<cusolver>(space_size_y);
        solver_u->set_factorization(solver_cfg_->tridiagonal_factorization());
        // splitting method:
        if (splitting_cfg_->splitting_method() == splitting_method_enum::DouglasRachford)
        {
            // create and set up the main solvers:
            splitting_ptr =
                std::make_shared<douglas_rachford_method>(solver_y, solver_u, heston_coeff_holder, grid_cfg_);
        }
        else if (splitting_cfg_->splitting_method() == splitting_method_enum::CraigSneyd)
        {
            splitting_ptr = std::make_shared<craig_sneyd_method>(solver_y, solver_u, heston_coeff_holder, grid_cfg_);
        }
        else if (splitting_cfg_->splitting_method() == splitting_method_enum::ModifiedCraigSneyd)
        {
            splitting_ptr = std::make_shared<m_craig_sneyd_method>(solver_y, solver_u, heston_coeff_holder, grid_cfg_);
        }
        else if (splitting_cfg_->splitting_method() == splitting_method_enum::HundsdorferVerwer)
        {
            splitting_ptr =
                std::make_shared<hundsdorfer_verwer_method>(solver_y, solver_u, heston_coeff_holder, grid_cfg_);
        }
        else
        {
            throw std::exception("Unreachable");
        }
        // create and set up lower volatility boundary solver:
        auto boundary_solver = std::make_shared<explicit_boundary>(heston_coeff_holder, grid_cfg_);
        auto const &weights = std::make_pair(weighted_scheme_cfg_->weight_x(), weighted_scheme_cfg_->weight_y());
        auto const &weight_values =
            std::make_pair(weighted_scheme_cfg_->start_x_value(), weighted_scheme_cfg_->start_y_value());

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
            loop::run(splitting_ptr, boundary_solver, boundary_pair_hor_, boundary_ver_, grid_cfg_, time, last_time_idx,
                      k, weights, weight_values, traverse_dir, prev_solution, next_solution);
        }
    }

    void operator()(rcontainer_2d_t &prev_solution, rcontainer_2d_t &next_solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type, fp_type)> const &heat_source, rcontainer_3d_t &solutions)
    {
    }
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class general_svc_heston_equation_implicit_kernel<memory_space_enum::Device, tridiagonal_method_enum::SORSolver,
                                                  fp_type, container, allocator>
{

    typedef sor_solver_cuda<fp_type, container, allocator> sorcusolver;
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container_2d<by_enum::Row, fp_type, container, allocator> rcontainer_2d_t;
    typedef container_3d<by_enum::Row, fp_type, container, allocator> rcontainer_3d_t;
    typedef heat_douglas_rachford_method<fp_type, sptr_t<sorcusolver>, container, allocator> douglas_rachford_method;
    typedef heat_craig_sneyd_method<fp_type, sptr_t<sorcusolver>, container, allocator> craig_sneyd_method;
    typedef heat_modified_craig_sneyd_method<fp_type, sptr_t<sorcusolver>, container, allocator> m_craig_sneyd_method;
    typedef heat_hundsdorfer_verwer_method<fp_type, sptr_t<sorcusolver>, container, allocator>
        hundsdorfer_verwer_method;
    typedef general_svc_heston_equation_explicit_boundary<fp_type, container, allocator> explicit_boundary;
    typedef implicit_heston_time_loop<fp_type, container, allocator> loop;

  private:
    boundary_2d_ptr<fp_type> boundary_ver_;
    boundary_2d_pair<fp_type> boundary_pair_hor_;
    heat_data_transform_2d_ptr<fp_type> heat_data_cfg_;
    pde_discretization_config_2d_ptr<fp_type> discretization_cfg_;
    splitting_method_config_ptr<fp_type> splitting_cfg_;
    weighted_scheme_config_ptr<fp_type> weighted_scheme_cfg_;
    heat_implicit_solver_config_ptr solver_cfg_;
    grid_config_2d_ptr<fp_type> grid_cfg_;

  public:
    general_svc_heston_equation_implicit_kernel(boundary_2d_ptr<fp_type> const &vertical_upper_boundary_ptr,
                                                boundary_2d_pair<fp_type> const &horizontal_boundary_pair,
                                                heat_data_transform_2d_ptr<fp_type> const &heat_data_config,
                                                pde_discretization_config_2d_ptr<fp_type> const &discretization_config,
                                                splitting_method_config_ptr<fp_type> const &splitting_config,
                                                weighted_scheme_config_ptr<fp_type> const &weighted_scheme_config,
                                                heat_implicit_solver_config_ptr const &solver_config,
                                                grid_config_2d_ptr<fp_type> const &grid_config)
        : boundary_ver_{vertical_upper_boundary_ptr}, boundary_pair_hor_{horizontal_boundary_pair},
          heat_data_cfg_{heat_data_config}, discretization_cfg_{discretization_config},
          splitting_cfg_{splitting_config}, weighted_scheme_cfg_{weighted_scheme_config},
          solver_cfg_{solver_config}, grid_cfg_{grid_config}
    {
    }

    void operator()(rcontainer_2d_t &prev_solution, rcontainer_2d_t &next_solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type, fp_type)> const &heat_source, fp_type omega_value)
    {
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
        const fp_type one = static_cast<fp_type>(1.0);
        const fp_type half = static_cast<fp_type>(0.5);
        const fp_type o_three = static_cast<fp_type>(0.3);
        const fp_type o_eight = static_cast<fp_type>(0.8);
        fp_type theta{};
        switch (solver_cfg_->implicit_pde_scheme())
        {
        case implicit_pde_schemes_enum::Euler:
            theta = one;
            break;
        case implicit_pde_schemes_enum::Theta_30:
            theta = o_three;
            break;
        case implicit_pde_schemes_enum::CrankNicolson:
            theta = half;
            break;
        case implicit_pde_schemes_enum::Theta_80:
            theta = o_eight;
            break;
        default:
            theta = half;
        }

        // create a Heston coefficient holder:
        auto const heston_coeff_holder = std::make_shared<general_svc_heston_equation_coefficients<fp_type>>(
            heat_data_cfg_, discretization_cfg_, splitting_cfg_, theta);
        heat_splitting_method_ptr<fp_type, container, allocator> splitting_ptr;
        auto solver_y = std::make_shared<sorcusolver>(space_size_x);
        solver_y->set_omega(omega_value);
        auto solver_u = std::make_shared<sorcusolver>(space_size_y);
        solver_u->set_omega(omega_value);
        // splitting method:
        if (splitting_cfg_->splitting_method() == splitting_method_enum::DouglasRachford)
        {
            // create and set up the main solvers:
            splitting_ptr =
                std::make_shared<douglas_rachford_method>(solver_y, solver_u, heston_coeff_holder, grid_cfg_);
        }
        else if (splitting_cfg_->splitting_method() == splitting_method_enum::CraigSneyd)
        {
            splitting_ptr = std::make_shared<craig_sneyd_method>(solver_y, solver_u, heston_coeff_holder, grid_cfg_);
        }
        else if (splitting_cfg_->splitting_method() == splitting_method_enum::ModifiedCraigSneyd)
        {
            splitting_ptr = std::make_shared<m_craig_sneyd_method>(solver_y, solver_u, heston_coeff_holder, grid_cfg_);
        }
        else if (splitting_cfg_->splitting_method() == splitting_method_enum::HundsdorferVerwer)
        {
            splitting_ptr =
                std::make_shared<hundsdorfer_verwer_method>(solver_y, solver_u, heston_coeff_holder, grid_cfg_);
        }
        else
        {
            throw std::exception("Unreachable");
        }
        // create and set up lower volatility boundary solver:
        auto boundary_solver = std::make_shared<explicit_boundary>(heston_coeff_holder, grid_cfg_);
        auto const &weights = std::make_pair(weighted_scheme_cfg_->weight_x(), weighted_scheme_cfg_->weight_y());
        auto const &weight_values =
            std::make_pair(weighted_scheme_cfg_->start_x_value(), weighted_scheme_cfg_->start_y_value());

        if (is_heat_sourse_set)
        {
            // auto scheme_function =
            //    implicit_heat_scheme<fp_type, container,
            //    allocator>::get(solver_cfg_->implicit_pde_scheme(), false);
            //// create a container to carry discretized source heat
            // container_t source_curr(space_size, NaN<fp_type>());
            // container_t source_next(space_size, NaN<fp_type>());
            // loop::run(solver, scheme_function, boundary_pair_, fun_triplet_,
            // space, time, last_time_idx, steps,
            //          traverse_dir, prev_solution, next_solution, rhs,
            //          heat_source, source_curr, source_next);
        }
        else
        {
            loop::run(splitting_ptr, boundary_solver, boundary_pair_hor_, boundary_ver_, grid_cfg_, time, last_time_idx,
                      k, weights, weight_values, traverse_dir, prev_solution, next_solution);
        }
    }

    void operator()(rcontainer_2d_t &prev_solution, rcontainer_2d_t &next_solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type, fp_type)> const &heat_source, fp_type omega_value,
                    rcontainer_3d_t &solutions)
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
    typedef cuda_solver<memory_space_enum::Host, fp_type, container, allocator> cusolver;
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container_2d<by_enum::Row, fp_type, container, allocator> rcontainer_2d_t;
    typedef container_3d<by_enum::Row, fp_type, container, allocator> rcontainer_3d_t;
    typedef heat_douglas_rachford_method<fp_type, sptr_t<cusolver>, container, allocator> douglas_rachford_method;
    typedef heat_craig_sneyd_method<fp_type, sptr_t<cusolver>, container, allocator> craig_sneyd_method;
    typedef heat_modified_craig_sneyd_method<fp_type, sptr_t<cusolver>, container, allocator> m_craig_sneyd_method;
    typedef heat_hundsdorfer_verwer_method<fp_type, sptr_t<cusolver>, container, allocator> hundsdorfer_verwer_method;
    typedef general_svc_heston_equation_explicit_boundary<fp_type, container, allocator> explicit_boundary;
    typedef implicit_heston_time_loop<fp_type, container, allocator> loop;

  private:
    boundary_2d_ptr<fp_type> boundary_ver_;
    boundary_2d_pair<fp_type> boundary_pair_hor_;
    heat_data_transform_2d_ptr<fp_type> heat_data_cfg_;
    pde_discretization_config_2d_ptr<fp_type> discretization_cfg_;
    splitting_method_config_ptr<fp_type> splitting_cfg_;
    weighted_scheme_config_ptr<fp_type> weighted_scheme_cfg_;
    heat_implicit_solver_config_ptr solver_cfg_;
    grid_config_2d_ptr<fp_type> grid_cfg_;

  public:
    general_svc_heston_equation_implicit_kernel(boundary_2d_ptr<fp_type> const &vertical_upper_boundary_ptr,
                                                boundary_2d_pair<fp_type> const &horizontal_boundary_pair,
                                                heat_data_transform_2d_ptr<fp_type> const &heat_data_config,
                                                pde_discretization_config_2d_ptr<fp_type> const &discretization_config,
                                                splitting_method_config_ptr<fp_type> const &splitting_config,
                                                weighted_scheme_config_ptr<fp_type> const &weighted_scheme_config,
                                                heat_implicit_solver_config_ptr const &solver_config,
                                                grid_config_2d_ptr<fp_type> const &grid_config)
        : boundary_ver_{vertical_upper_boundary_ptr}, boundary_pair_hor_{horizontal_boundary_pair},
          heat_data_cfg_{heat_data_config}, discretization_cfg_{discretization_config},
          splitting_cfg_{splitting_config}, weighted_scheme_cfg_{weighted_scheme_config},
          solver_cfg_{solver_config}, grid_cfg_{grid_config}
    {
    }

    void operator()(rcontainer_2d_t &prev_solution, rcontainer_2d_t &next_solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type, fp_type)> const &heat_source)
    {
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
        const fp_type one = static_cast<fp_type>(1.0);
        const fp_type half = static_cast<fp_type>(0.5);
        const fp_type o_three = static_cast<fp_type>(0.3);
        const fp_type o_eight = static_cast<fp_type>(0.8);
        fp_type theta{};
        switch (solver_cfg_->implicit_pde_scheme())
        {
        case implicit_pde_schemes_enum::Euler:
            theta = one;
            break;
        case implicit_pde_schemes_enum::Theta_30:
            theta = o_three;
            break;
        case implicit_pde_schemes_enum::CrankNicolson:
            theta = half;
            break;
        case implicit_pde_schemes_enum::Theta_80:
            theta = o_eight;
            break;
        default:
            theta = half;
        }

        // create a Heston coefficient holder:
        auto const heston_coeff_holder = std::make_shared<general_svc_heston_equation_coefficients<fp_type>>(
            heat_data_cfg_, discretization_cfg_, splitting_cfg_, theta);
        heat_splitting_method_ptr<fp_type, container, allocator> splitting_ptr;
        auto solver_y = std::make_shared<cusolver>(space_size_x);
        solver_y->set_factorization(solver_cfg_->tridiagonal_factorization());
        auto solver_u = std::make_shared<cusolver>(space_size_y);
        solver_u->set_factorization(solver_cfg_->tridiagonal_factorization());
        // splitting method:
        if (splitting_cfg_->splitting_method() == splitting_method_enum::DouglasRachford)
        {
            // create and set up the main solvers:
            splitting_ptr =
                std::make_shared<douglas_rachford_method>(solver_y, solver_u, heston_coeff_holder, grid_cfg_);
        }
        else if (splitting_cfg_->splitting_method() == splitting_method_enum::CraigSneyd)
        {
            splitting_ptr = std::make_shared<craig_sneyd_method>(solver_y, solver_u, heston_coeff_holder, grid_cfg_);
        }
        else if (splitting_cfg_->splitting_method() == splitting_method_enum::ModifiedCraigSneyd)
        {
            splitting_ptr = std::make_shared<m_craig_sneyd_method>(solver_y, solver_u, heston_coeff_holder, grid_cfg_);
        }
        else if (splitting_cfg_->splitting_method() == splitting_method_enum::HundsdorferVerwer)
        {
            splitting_ptr =
                std::make_shared<hundsdorfer_verwer_method>(solver_y, solver_u, heston_coeff_holder, grid_cfg_);
        }
        else
        {
            throw std::exception("Unreachable");
        }
        // create and set up lower volatility boundary solver:
        auto boundary_solver = std::make_shared<explicit_boundary>(heston_coeff_holder, grid_cfg_);
        auto const &weights = std::make_pair(weighted_scheme_cfg_->weight_x(), weighted_scheme_cfg_->weight_y());
        auto const &weight_values =
            std::make_pair(weighted_scheme_cfg_->start_x_value(), weighted_scheme_cfg_->start_y_value());

        if (is_heat_sourse_set)
        {
            // auto scheme_function =
            //    implicit_heat_scheme<fp_type, container,
            //    allocator>::get(solver_cfg_->implicit_pde_scheme(), false);
            //// create a container to carry discretized source heat
            // container_t source_curr(space_size, NaN<fp_type>());
            // container_t source_next(space_size, NaN<fp_type>());
            // loop::run(solver, scheme_function, boundary_pair_, fun_triplet_,
            // space, time, last_time_idx, steps,
            //          traverse_dir, prev_solution, next_solution, rhs,
            //          heat_source, source_curr, source_next);
        }
        else
        {
            loop::run(splitting_ptr, boundary_solver, boundary_pair_hor_, boundary_ver_, grid_cfg_, time, last_time_idx,
                      k, weights, weight_values, traverse_dir, prev_solution, next_solution);
        }
    }

    void operator()(rcontainer_2d_t &prev_solution, rcontainer_2d_t &next_solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type)> const &heat_source, rcontainer_3d_t &solutions)
    {
    }
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class general_svc_heston_equation_implicit_kernel<memory_space_enum::Host, tridiagonal_method_enum::SORSolver, fp_type,
                                                  container, allocator>
{
    typedef sor_solver<fp_type, container, allocator> sorsolver;
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container_2d<by_enum::Row, fp_type, container, allocator> rcontainer_2d_t;
    typedef container_3d<by_enum::Row, fp_type, container, allocator> rcontainer_3d_t;
    typedef heat_douglas_rachford_method<fp_type, sptr_t<sorsolver>, container, allocator> douglas_rachford_method;
    typedef heat_craig_sneyd_method<fp_type, sptr_t<sorsolver>, container, allocator> craig_sneyd_method;
    typedef heat_modified_craig_sneyd_method<fp_type, sptr_t<sorsolver>, container, allocator> m_craig_sneyd_method;
    typedef heat_hundsdorfer_verwer_method<fp_type, sptr_t<sorsolver>, container, allocator> hundsdorfer_verwer_method;
    typedef general_svc_heston_equation_explicit_boundary<fp_type, container, allocator> explicit_boundary;
    typedef implicit_heston_time_loop<fp_type, container, allocator> loop;

  private:
    boundary_2d_ptr<fp_type> boundary_ver_;
    boundary_2d_pair<fp_type> boundary_pair_hor_;
    heat_data_transform_2d_ptr<fp_type> heat_data_cfg_;
    pde_discretization_config_2d_ptr<fp_type> discretization_cfg_;
    splitting_method_config_ptr<fp_type> splitting_cfg_;
    weighted_scheme_config_ptr<fp_type> weighted_scheme_cfg_;
    heat_implicit_solver_config_ptr solver_cfg_;
    grid_config_2d_ptr<fp_type> grid_cfg_;

  public:
    general_svc_heston_equation_implicit_kernel(boundary_2d_ptr<fp_type> const &vertical_upper_boundary_ptr,
                                                boundary_2d_pair<fp_type> const &horizontal_boundary_pair,
                                                heat_data_transform_2d_ptr<fp_type> const &heat_data_config,
                                                pde_discretization_config_2d_ptr<fp_type> const &discretization_config,
                                                splitting_method_config_ptr<fp_type> const &splitting_config,
                                                weighted_scheme_config_ptr<fp_type> const &weighted_scheme_config,
                                                heat_implicit_solver_config_ptr const &solver_config,
                                                grid_config_2d_ptr<fp_type> const &grid_config)
        : boundary_ver_{vertical_upper_boundary_ptr}, boundary_pair_hor_{horizontal_boundary_pair},
          heat_data_cfg_{heat_data_config}, discretization_cfg_{discretization_config},
          splitting_cfg_{splitting_config}, weighted_scheme_cfg_{weighted_scheme_config},
          solver_cfg_{solver_config}, grid_cfg_{grid_config}
    {
    }

    void operator()(rcontainer_2d_t &prev_solution, rcontainer_2d_t &next_solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type, fp_type)> const &heat_source, fp_type omega_value)
    {
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
        const fp_type one = static_cast<fp_type>(1.0);
        const fp_type half = static_cast<fp_type>(0.5);
        const fp_type o_three = static_cast<fp_type>(0.3);
        const fp_type o_eight = static_cast<fp_type>(0.8);
        fp_type theta{};
        switch (solver_cfg_->implicit_pde_scheme())
        {
        case implicit_pde_schemes_enum::Euler:
            theta = one;
            break;
        case implicit_pde_schemes_enum::Theta_30:
            theta = o_three;
            break;
        case implicit_pde_schemes_enum::CrankNicolson:
            theta = half;
            break;
        case implicit_pde_schemes_enum::Theta_80:
            theta = o_eight;
            break;
        default:
            theta = half;
        }

        // create a Heston coefficient holder:
        auto const heston_coeff_holder = std::make_shared<general_svc_heston_equation_coefficients<fp_type>>(
            heat_data_cfg_, discretization_cfg_, splitting_cfg_, theta);
        heat_splitting_method_ptr<fp_type, container, allocator> splitting_ptr;
        auto solver_y = std::make_shared<sorsolver>(space_size_x);
        solver_y->set_omega(omega_value);
        auto solver_u = std::make_shared<sorsolver>(space_size_y);
        solver_u->set_omega(omega_value);
        // splitting method:
        if (splitting_cfg_->splitting_method() == splitting_method_enum::DouglasRachford)
        {
            // create and set up the main solvers:
            splitting_ptr =
                std::make_shared<douglas_rachford_method>(solver_y, solver_u, heston_coeff_holder, grid_cfg_);
        }
        else if (splitting_cfg_->splitting_method() == splitting_method_enum::CraigSneyd)
        {
            splitting_ptr = std::make_shared<craig_sneyd_method>(solver_y, solver_u, heston_coeff_holder, grid_cfg_);
        }
        else if (splitting_cfg_->splitting_method() == splitting_method_enum::ModifiedCraigSneyd)
        {
            splitting_ptr = std::make_shared<m_craig_sneyd_method>(solver_y, solver_u, heston_coeff_holder, grid_cfg_);
        }
        else if (splitting_cfg_->splitting_method() == splitting_method_enum::HundsdorferVerwer)
        {
            splitting_ptr =
                std::make_shared<hundsdorfer_verwer_method>(solver_y, solver_u, heston_coeff_holder, grid_cfg_);
        }
        else
        {
            throw std::exception("Unreachable");
        }
        // create and set up lower volatility boundary solver:
        auto boundary_solver = std::make_shared<explicit_boundary>(heston_coeff_holder, grid_cfg_);
        auto const &weights = std::make_pair(weighted_scheme_cfg_->weight_x(), weighted_scheme_cfg_->weight_y());
        auto const &weight_values =
            std::make_pair(weighted_scheme_cfg_->start_x_value(), weighted_scheme_cfg_->start_y_value());

        if (is_heat_sourse_set)
        {
            // auto scheme_function =
            //    implicit_heat_scheme<fp_type, container,
            //    allocator>::get(solver_cfg_->implicit_pde_scheme(), false);
            //// create a container to carry discretized source heat
            // container_t source_curr(space_size, NaN<fp_type>());
            // container_t source_next(space_size, NaN<fp_type>());
            // loop::run(solver, scheme_function, boundary_pair_, fun_triplet_,
            // space, time, last_time_idx, steps,
            //          traverse_dir, prev_solution, next_solution, rhs,
            //          heat_source, source_curr, source_next);
        }
        else
        {
            loop::run(splitting_ptr, boundary_solver, boundary_pair_hor_, boundary_ver_, grid_cfg_, time, last_time_idx,
                      k, weights, weight_values, traverse_dir, prev_solution, next_solution);
        }
    }

    void operator()(rcontainer_2d_t &prev_solution, rcontainer_2d_t &next_solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type, fp_type)> const &heat_source, fp_type omega_value,
                    rcontainer_3d_t &solutions)
    {
    }
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class general_svc_heston_equation_implicit_kernel<memory_space_enum::Host, tridiagonal_method_enum::DoubleSweepSolver,
                                                  fp_type, container, allocator>
{
    typedef double_sweep_solver<fp_type, container, allocator> ds_solver;
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container_2d<by_enum::Row, fp_type, container, allocator> rcontainer_2d_t;
    typedef container_3d<by_enum::Row, fp_type, container, allocator> rcontainer_3d_t;
    typedef heat_douglas_rachford_method<fp_type, sptr_t<ds_solver>, container, allocator> douglas_rachford_method;
    typedef heat_craig_sneyd_method<fp_type, sptr_t<ds_solver>, container, allocator> craig_sneyd_method;
    typedef heat_modified_craig_sneyd_method<fp_type, sptr_t<ds_solver>, container, allocator> m_craig_sneyd_method;
    typedef heat_hundsdorfer_verwer_method<fp_type, sptr_t<ds_solver>, container, allocator> hundsdorfer_verwer_method;
    typedef general_svc_heston_equation_explicit_boundary<fp_type, container, allocator> explicit_boundary;
    typedef implicit_heston_time_loop<fp_type, container, allocator> loop;

  private:
    boundary_2d_ptr<fp_type> boundary_ver_;
    boundary_2d_pair<fp_type> boundary_pair_hor_;
    heat_data_transform_2d_ptr<fp_type> heat_data_cfg_;
    pde_discretization_config_2d_ptr<fp_type> discretization_cfg_;
    splitting_method_config_ptr<fp_type> splitting_cfg_;
    weighted_scheme_config_ptr<fp_type> weighted_scheme_cfg_;
    heat_implicit_solver_config_ptr solver_cfg_;
    grid_config_2d_ptr<fp_type> grid_cfg_;

  public:
    general_svc_heston_equation_implicit_kernel(boundary_2d_ptr<fp_type> const &vertical_upper_boundary_ptr,
                                                boundary_2d_pair<fp_type> const &horizontal_boundary_pair,
                                                heat_data_transform_2d_ptr<fp_type> const &heat_data_config,
                                                pde_discretization_config_2d_ptr<fp_type> const &discretization_config,
                                                splitting_method_config_ptr<fp_type> const &splitting_config,
                                                weighted_scheme_config_ptr<fp_type> const &weighted_scheme_config,
                                                heat_implicit_solver_config_ptr const &solver_config,
                                                grid_config_2d_ptr<fp_type> const &grid_config)
        : boundary_ver_{vertical_upper_boundary_ptr}, boundary_pair_hor_{horizontal_boundary_pair},
          heat_data_cfg_{heat_data_config}, discretization_cfg_{discretization_config},
          splitting_cfg_{splitting_config}, weighted_scheme_cfg_{weighted_scheme_config},
          solver_cfg_{solver_config}, grid_cfg_{grid_config}
    {
    }

    void operator()(rcontainer_2d_t &prev_solution, rcontainer_2d_t &next_solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type, fp_type)> const &heat_source)
    {
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
        const fp_type one = static_cast<fp_type>(1.0);
        const fp_type half = static_cast<fp_type>(0.5);
        const fp_type o_three = static_cast<fp_type>(0.3);
        const fp_type o_eight = static_cast<fp_type>(0.8);
        fp_type theta{};
        switch (solver_cfg_->implicit_pde_scheme())
        {
        case implicit_pde_schemes_enum::Euler:
            theta = one;
            break;
        case implicit_pde_schemes_enum::Theta_30:
            theta = o_three;
            break;
        case implicit_pde_schemes_enum::CrankNicolson:
            theta = half;
            break;
        case implicit_pde_schemes_enum::Theta_80:
            theta = o_eight;
            break;
        default:
            theta = half;
        }

        // create a Heston coefficient holder:
        auto const heston_coeff_holder = std::make_shared<general_svc_heston_equation_coefficients<fp_type>>(
            heat_data_cfg_, discretization_cfg_, splitting_cfg_, theta);
        heat_splitting_method_ptr<fp_type, container, allocator> splitting_ptr;
        // create and set up the main solvers:
        auto solver_y = std::make_shared<ds_solver>(space_size_x);
        auto solver_u = std::make_shared<ds_solver>(space_size_y);
        // splitting method:
        if (splitting_cfg_->splitting_method() == splitting_method_enum::DouglasRachford)
        {
            splitting_ptr =
                std::make_shared<douglas_rachford_method>(solver_y, solver_u, heston_coeff_holder, grid_cfg_);
        }
        else if (splitting_cfg_->splitting_method() == splitting_method_enum::CraigSneyd)
        {
            splitting_ptr = std::make_shared<craig_sneyd_method>(solver_y, solver_u, heston_coeff_holder, grid_cfg_);
        }
        else if (splitting_cfg_->splitting_method() == splitting_method_enum::ModifiedCraigSneyd)
        {
            splitting_ptr = std::make_shared<m_craig_sneyd_method>(solver_y, solver_u, heston_coeff_holder, grid_cfg_);
        }
        else if (splitting_cfg_->splitting_method() == splitting_method_enum::HundsdorferVerwer)
        {
            splitting_ptr =
                std::make_shared<hundsdorfer_verwer_method>(solver_y, solver_u, heston_coeff_holder, grid_cfg_);
        }
        else
        {
            throw std::exception("Unreachable");
        }
        // create and set up lower volatility boundary solver:
        auto boundary_solver = std::make_shared<explicit_boundary>(heston_coeff_holder, grid_cfg_);
        auto const &weights = std::make_pair(weighted_scheme_cfg_->weight_x(), weighted_scheme_cfg_->weight_y());
        auto const &weight_values =
            std::make_pair(weighted_scheme_cfg_->start_x_value(), weighted_scheme_cfg_->start_y_value());

        if (is_heat_sourse_set)
        {
            // auto scheme_function =
            //    implicit_heat_scheme<fp_type, container,
            //    allocator>::get(solver_cfg_->implicit_pde_scheme(), false);
            //// create a container to carry discretized source heat
            // container_t source_curr(space_size, NaN<fp_type>());
            // container_t source_next(space_size, NaN<fp_type>());
            // loop::run(solver, scheme_function, boundary_pair_, fun_triplet_,
            // space, time, last_time_idx, steps,
            //          traverse_dir, prev_solution, next_solution, rhs,
            //          heat_source, source_curr, source_next);
        }
        else
        {
            loop::run(splitting_ptr, boundary_solver, boundary_pair_hor_, boundary_ver_, grid_cfg_, time, last_time_idx,
                      k, weights, weight_values, traverse_dir, prev_solution, next_solution);
        }
    }

    void operator()(rcontainer_2d_t &prev_solution, rcontainer_2d_t &next_solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type, fp_type)> const &heat_source, rcontainer_3d_t &solutions)
    {
    }
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class general_svc_heston_equation_implicit_kernel<memory_space_enum::Host, tridiagonal_method_enum::ThomasLUSolver,
                                                  fp_type, container, allocator>
{
    typedef thomas_lu_solver<fp_type, container, allocator> tlu_solver;
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container_2d<by_enum::Row, fp_type, container, allocator> rcontainer_2d_t;
    typedef container_3d<by_enum::Row, fp_type, container, allocator> rcontainer_3d_t;
    typedef heat_douglas_rachford_method<fp_type, sptr_t<tlu_solver>, container, allocator> douglas_rachford_method;
    typedef heat_craig_sneyd_method<fp_type, sptr_t<tlu_solver>, container, allocator> craig_sneyd_method;
    typedef heat_modified_craig_sneyd_method<fp_type, sptr_t<tlu_solver>, container, allocator> m_craig_sneyd_method;
    typedef heat_hundsdorfer_verwer_method<fp_type, sptr_t<tlu_solver>, container, allocator> hundsdorfer_verwer_method;
    typedef general_svc_heston_equation_explicit_boundary<fp_type, container, allocator> explicit_boundary;
    typedef implicit_heston_time_loop<fp_type, container, allocator> loop;

  private:
    boundary_2d_ptr<fp_type> boundary_ver_;
    boundary_2d_pair<fp_type> boundary_pair_hor_;
    heat_data_transform_2d_ptr<fp_type> heat_data_cfg_;
    pde_discretization_config_2d_ptr<fp_type> discretization_cfg_;
    splitting_method_config_ptr<fp_type> splitting_cfg_;
    weighted_scheme_config_ptr<fp_type> weighted_scheme_cfg_;
    heat_implicit_solver_config_ptr solver_cfg_;
    grid_config_2d_ptr<fp_type> grid_cfg_;

  public:
    general_svc_heston_equation_implicit_kernel(boundary_2d_ptr<fp_type> const &vertical_upper_boundary_ptr,
                                                boundary_2d_pair<fp_type> const &horizontal_boundary_pair,
                                                heat_data_transform_2d_ptr<fp_type> const &heat_data_config,
                                                pde_discretization_config_2d_ptr<fp_type> const &discretization_config,
                                                splitting_method_config_ptr<fp_type> const &splitting_config,
                                                weighted_scheme_config_ptr<fp_type> const &weighted_scheme_config,
                                                heat_implicit_solver_config_ptr const &solver_config,
                                                grid_config_2d_ptr<fp_type> const &grid_config)
        : boundary_ver_{vertical_upper_boundary_ptr}, boundary_pair_hor_{horizontal_boundary_pair},
          heat_data_cfg_{heat_data_config}, discretization_cfg_{discretization_config},
          splitting_cfg_{splitting_config}, weighted_scheme_cfg_{weighted_scheme_config},
          solver_cfg_{solver_config}, grid_cfg_{grid_config}
    {
    }

    void operator()(rcontainer_2d_t &prev_solution, rcontainer_2d_t &next_solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type, fp_type)> const &heat_source)
    {
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
        const fp_type one = static_cast<fp_type>(1.0);
        const fp_type half = static_cast<fp_type>(0.5);
        const fp_type o_three = static_cast<fp_type>(0.3);
        const fp_type o_eight = static_cast<fp_type>(0.8);
        fp_type theta{};
        switch (solver_cfg_->implicit_pde_scheme())
        {
        case implicit_pde_schemes_enum::Euler:
            theta = one;
            break;
        case implicit_pde_schemes_enum::Theta_30:
            theta = o_three;
            break;
        case implicit_pde_schemes_enum::CrankNicolson:
            theta = half;
            break;
        case implicit_pde_schemes_enum::Theta_80:
            theta = o_eight;
            break;
        default:
            theta = half;
        }

        // create a Heston coefficient holder:
        auto const heston_coeff_holder = std::make_shared<general_svc_heston_equation_coefficients<fp_type>>(
            heat_data_cfg_, discretization_cfg_, splitting_cfg_, theta);
        heat_splitting_method_ptr<fp_type, container, allocator> splitting_ptr;
        // create and set up the main solvers:
        auto solver_y = std::make_shared<tlu_solver>(space_size_x);
        auto solver_u = std::make_shared<tlu_solver>(space_size_y);
        // splitting method:
        if (splitting_cfg_->splitting_method() == splitting_method_enum::DouglasRachford)
        {
            splitting_ptr =
                std::make_shared<douglas_rachford_method>(solver_y, solver_u, heston_coeff_holder, grid_cfg_);
        }
        else if (splitting_cfg_->splitting_method() == splitting_method_enum::CraigSneyd)
        {
            splitting_ptr = std::make_shared<craig_sneyd_method>(solver_y, solver_u, heston_coeff_holder, grid_cfg_);
        }
        else if (splitting_cfg_->splitting_method() == splitting_method_enum::ModifiedCraigSneyd)
        {
            splitting_ptr = std::make_shared<m_craig_sneyd_method>(solver_y, solver_u, heston_coeff_holder, grid_cfg_);
        }
        else if (splitting_cfg_->splitting_method() == splitting_method_enum::HundsdorferVerwer)
        {
            splitting_ptr =
                std::make_shared<hundsdorfer_verwer_method>(solver_y, solver_u, heston_coeff_holder, grid_cfg_);
        }
        else
        {
            throw std::exception("Unreachable");
        }
        // create and set up lower volatility boundary solver:
        auto boundary_solver = std::make_shared<explicit_boundary>(heston_coeff_holder, grid_cfg_);
        auto const &weights = std::make_pair(weighted_scheme_cfg_->weight_x(), weighted_scheme_cfg_->weight_y());
        auto const &weight_values =
            std::make_pair(weighted_scheme_cfg_->start_x_value(), weighted_scheme_cfg_->start_y_value());

        if (is_heat_sourse_set)
        {
            // auto scheme_function =
            //    implicit_heat_scheme<fp_type, container,
            //    allocator>::get(solver_cfg_->implicit_pde_scheme(), false);
            //// create a container to carry discretized source heat
            // container_t source_curr(space_size, NaN<fp_type>());
            // container_t source_next(space_size, NaN<fp_type>());
            // loop::run(solver, scheme_function, boundary_pair_, fun_triplet_,
            // space, time, last_time_idx, steps,
            //          traverse_dir, prev_solution, next_solution, rhs,
            //          heat_source, source_curr, source_next);
        }
        else
        {
            loop::run(splitting_ptr, boundary_solver, boundary_pair_hor_, boundary_ver_, grid_cfg_, time, last_time_idx,
                      k, weights, weight_values, traverse_dir, prev_solution, next_solution);
        }
    }

    void operator()(rcontainer_2d_t &prev_solution, rcontainer_2d_t &next_solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type, fp_type)> const &heat_source, rcontainer_3d_t &solutions)
    {
    }
};
} // namespace two_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_2D_GENERAL_SVC_HESTON_EQUATION_IMPLICIT_KERNEL_HPP_
