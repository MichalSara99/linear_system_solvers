#if !defined(_LSS_1D_GENERAL_HEAT_EQUATION_EXPLICIT_KERNEL_HPP_)
#define _LSS_1D_GENERAL_HEAT_EQUATION_EXPLICIT_KERNEL_HPP_

#include <vector>

#include "boundaries/lss_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include "containers/lss_container_2d.hpp"
#include "discretization/lss_discretization.hpp"
#include "discretization/lss_grid_config.hpp"
#include "explicit_schemes/lss_heat_barakat_clark_scheme.hpp"
#include "explicit_schemes/lss_heat_euler_cuda_scheme.hpp"
#include "explicit_schemes/lss_heat_euler_scheme.hpp"
#include "explicit_schemes/lss_heat_saulyev_scheme.hpp"
#include "implicit_coefficients/lss_1d_general_heat_equation_coefficients.hpp"
#include "pde_solvers/lss_heat_solver_config.hpp"
#include "pde_solvers/lss_pde_discretization_config.hpp"
#include "pde_solvers/transformation/lss_heat_data_transform.hpp"

namespace lss_pde_solvers
{
namespace one_dimensional
{

using lss_boundary::boundary_1d_pair;
using lss_containers::container_2d;
using lss_enumerations::dimension_enum;
using lss_enumerations::explicit_pde_schemes_enum;
using lss_enumerations::memory_space_enum;
using lss_enumerations::traverse_direction_enum;
using lss_utility::NaN;
using lss_utility::range;

template <memory_space_enum memory_enum, typename fp_type,
          template <typename, typename> typename container = std::vector, typename allocator = std::allocator<fp_type>>
class general_heat_equation_explicit_kernel
{
};

// ===================================================================
// ============================== DEVICE =============================
// ===================================================================
template <typename fp_type, template <typename, typename> typename container, typename allocator>
class general_heat_equation_explicit_kernel<memory_space_enum::Device, fp_type, container, allocator>
{
    typedef container<fp_type, allocator> container_t;

  private:
    boundary_1d_pair<fp_type> boundary_pair_;
    heat_data_transform_1d_ptr<fp_type> heat_data_cfg_;
    pde_discretization_config_1d_ptr<fp_type> discretization_cfg_;
    heat_explicit_solver_config_ptr solver_cfg_;
    grid_config_1d_ptr<fp_type> grid_cfg_;

  public:
    general_heat_equation_explicit_kernel(boundary_1d_pair<fp_type> const &boundary_pair,
                                          heat_data_transform_1d_ptr<fp_type> const &heat_data_config,
                                          pde_discretization_config_1d_ptr<fp_type> const &discretization_config,
                                          heat_explicit_solver_config_ptr const &solver_config,
                                          grid_config_1d_ptr<fp_type> const &grid_config)
        : boundary_pair_{boundary_pair}, heat_data_cfg_{heat_data_config}, discretization_cfg_{discretization_config},
          solver_cfg_{solver_config}, grid_cfg_{grid_config}
    {
    }

    void operator()(container_t &solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type)> const &heat_source)
    {
        // save traverse_direction
        const traverse_direction_enum traverse_dir = solver_cfg_->traverse_direction();
        // create a heat coefficient holder:
        auto const heat_coeff_holder = std::make_shared<general_heat_equation_coefficients<fp_type>>(
            heat_data_cfg_, discretization_cfg_, fp_type{});
        // Here make a dicision which explicit scheme to launch:
        if (solver_cfg_->explicit_pde_scheme() == explicit_pde_schemes_enum::Euler)
        {
            typedef heat_euler_cuda_scheme<fp_type, container, allocator> euler_cuda_scheme_t;
            euler_cuda_scheme_t euler_scheme(heat_coeff_holder, boundary_pair_, discretization_cfg_, grid_cfg_);
            euler_scheme(solution, is_heat_sourse_set, heat_source, traverse_dir);
        }
        else if (solver_cfg_->explicit_pde_scheme() == explicit_pde_schemes_enum::ADEBarakatClark)
        {
            throw std::exception("Not currently supported");
        }
        else if (solver_cfg_->explicit_pde_scheme() == explicit_pde_schemes_enum::ADESaulyev)
        {
            throw std::exception("Not currently supported");
        }
        else
        {
            throw std::exception("Unreachable");
        }
    }

    void operator()(container_t &solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type)> const &heat_source,
                    container_2d<by_enum::Row, fp_type, container, allocator> &solutions)
    {
        // save traverse_direction
        const traverse_direction_enum traverse_dir = solver_cfg_->traverse_direction();
        // create a heat coefficient holder:
        auto const heat_coeff_holder = std::make_shared<general_heat_equation_coefficients<fp_type>>(
            heat_data_cfg_, discretization_cfg_, fp_type{});
        // Here make a dicision which explicit scheme to launch:
        if (solver_cfg_->explicit_pde_scheme() == explicit_pde_schemes_enum::Euler)
        {
            typedef heat_euler_cuda_scheme<fp_type, container, allocator> euler_cuda_scheme_t;
            euler_cuda_scheme_t euler_scheme(heat_coeff_holder, boundary_pair_, discretization_cfg_, grid_cfg_);
            euler_scheme(solution, is_heat_sourse_set, heat_source, traverse_dir, solutions);
        }
        else if (solver_cfg_->explicit_pde_scheme() == explicit_pde_schemes_enum::ADEBarakatClark)
        {
            throw std::exception("Not currently supported");
        }
        else if (solver_cfg_->explicit_pde_scheme() == explicit_pde_schemes_enum::ADESaulyev)
        {
            throw std::exception("Not currently supported");
        }
        else
        {
            throw std::exception("Unreachable");
        }
    }
};

// ===================================================================
// ============================== HOST ===============================
// ===================================================================

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class general_heat_equation_explicit_kernel<memory_space_enum::Host, fp_type, container, allocator>
{
    typedef container<fp_type, allocator> container_t;

  private:
    boundary_1d_pair<fp_type> boundary_pair_;
    heat_data_transform_1d_ptr<fp_type> heat_data_cfg_;
    pde_discretization_config_1d_ptr<fp_type> discretization_cfg_;
    heat_explicit_solver_config_ptr solver_cfg_;
    grid_config_1d_ptr<fp_type> grid_cfg_;

  public:
    general_heat_equation_explicit_kernel(boundary_1d_pair<fp_type> const &boundary_pair,
                                          heat_data_transform_1d_ptr<fp_type> const &heat_data_config,
                                          pde_discretization_config_1d_ptr<fp_type> const &discretization_config,
                                          heat_explicit_solver_config_ptr const &solver_config,
                                          grid_config_1d_ptr<fp_type> const &grid_config)
        : boundary_pair_{boundary_pair}, heat_data_cfg_{heat_data_config}, discretization_cfg_{discretization_config},
          solver_cfg_{solver_config}, grid_cfg_{grid_config}
    {
    }

    void operator()(container_t &solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type)> const &heat_source)
    {
        // save traverse_direction
        const traverse_direction_enum traverse_dir = solver_cfg_->traverse_direction();
        // create a heat coefficient holder:
        auto const heat_coeff_holder = std::make_shared<general_heat_equation_coefficients<fp_type>>(
            heat_data_cfg_, discretization_cfg_, fp_type{});
        // Here make a dicision which explicit scheme to launch:
        if (solver_cfg_->explicit_pde_scheme() == explicit_pde_schemes_enum::Euler)
        {
            typedef heat_euler_scheme<fp_type, container, allocator> euler_scheme_t;
            euler_scheme_t euler_scheme(heat_coeff_holder, boundary_pair_, discretization_cfg_, grid_cfg_);
            euler_scheme(solution, is_heat_sourse_set, heat_source, traverse_dir);
        }
        else if (solver_cfg_->explicit_pde_scheme() == explicit_pde_schemes_enum::ADEBarakatClark)
        {
            typedef heat_barakat_clark_scheme<fp_type, container, allocator> barakat_clark_scheme_t;
            barakat_clark_scheme_t bc_scheme(heat_coeff_holder, boundary_pair_, discretization_cfg_, grid_cfg_);
            bc_scheme(solution, is_heat_sourse_set, heat_source, traverse_dir);
        }
        else if (solver_cfg_->explicit_pde_scheme() == explicit_pde_schemes_enum::ADESaulyev)
        {
            typedef heat_saulyev_scheme<fp_type, container, allocator> saulyev_scheme_t;
            saulyev_scheme_t s_scheme(heat_coeff_holder, boundary_pair_, discretization_cfg_, grid_cfg_);
            s_scheme(solution, is_heat_sourse_set, heat_source, traverse_dir);
        }
        else
        {
            throw std::exception("Unreachable");
        }
    }

    void operator()(container_t &solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type)> const &heat_source,
                    container_2d<by_enum::Row, fp_type, container, allocator> &solutions)
    {
        // save traverse_direction
        const traverse_direction_enum traverse_dir = solver_cfg_->traverse_direction();
        // create a heat coefficient holder:
        auto const heat_coeff_holder = std::make_shared<general_heat_equation_coefficients<fp_type>>(
            heat_data_cfg_, discretization_cfg_, fp_type{});
        // Here make a dicision which explicit scheme to launch:
        if (solver_cfg_->explicit_pde_scheme() == explicit_pde_schemes_enum::Euler)
        {
            typedef heat_euler_scheme<fp_type, container, allocator> euler_scheme_t;
            euler_scheme_t euler_scheme(heat_coeff_holder, boundary_pair_, discretization_cfg_, grid_cfg_);
            euler_scheme(solution, is_heat_sourse_set, heat_source, traverse_dir, solutions);
        }
        else if (solver_cfg_->explicit_pde_scheme() == explicit_pde_schemes_enum::ADEBarakatClark)
        {
            typedef heat_barakat_clark_scheme<fp_type, container, allocator> barakat_clark_scheme_t;
            barakat_clark_scheme_t bc_scheme(heat_coeff_holder, boundary_pair_, discretization_cfg_, grid_cfg_);
            bc_scheme(solution, is_heat_sourse_set, heat_source, traverse_dir, solutions);
        }
        else if (solver_cfg_->explicit_pde_scheme() == explicit_pde_schemes_enum::ADESaulyev)
        {
            typedef heat_saulyev_scheme<fp_type, container, allocator> saulyev_scheme_t;
            saulyev_scheme_t s_scheme(heat_coeff_holder, boundary_pair_, discretization_cfg_, grid_cfg_);
            s_scheme(solution, is_heat_sourse_set, heat_source, traverse_dir, solutions);
        }
        else
        {
            throw std::exception("Unreachable");
        }
    }
};

} // namespace one_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_1D_GENERAL_HEAT_EQUATION_EXPLICIT_KERNEL_HPP_
