#if !defined(_LSS_2D_GENERAL_HESTON_EQUATION_EXPLICIT_KERNEL_HPP_)
#define _LSS_2D_GENERAL_HESTON_EQUATION_EXPLICIT_KERNEL_HPP_

#include <vector>

#include "boundaries/lss_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include "containers/lss_container_2d.hpp"
#include "containers/lss_container_3d.hpp"
#include "discretization/lss_discretization.hpp"
#include "discretization/lss_grid.hpp"
#include "discretization/lss_grid_config.hpp"
#include "pde_solvers/lss_heat_solver_config.hpp"
#include "pde_solvers/lss_pde_discretization_config.hpp"
#include "pde_solvers/transformation/lss_heat_data_transform.hpp"
#include "pde_solvers/two_dimensional/heat_type/explicit_schemes/lss_heston_euler_cuda_solver_scheme.hpp"
#include "pde_solvers/two_dimensional/heat_type/explicit_schemes/lss_heston_euler_solver_scheme.hpp"
#include "pde_solvers/two_dimensional/heat_type/implicit_coefficients/lss_2d_general_heston_equation_coefficients.hpp"

namespace lss_pde_solvers
{
namespace two_dimensional
{

using lss_boundary::boundary_2d_pair;
using lss_boundary::boundary_2d_ptr;
using lss_containers::container_2d;
using lss_containers::container_3d;
using lss_enumerations::by_enum;
using lss_enumerations::dimension_enum;
using lss_enumerations::explicit_pde_schemes_enum;
using lss_enumerations::memory_space_enum;
using lss_grids::grid_config_1d_ptr;
using lss_grids::grid_config_2d_ptr;
using lss_utility::NaN;
using lss_utility::range;
using lss_utility::sptr_t;

template <memory_space_enum memory_enum, typename fp_type,
          template <typename, typename> typename container = std::vector, typename allocator = std::allocator<fp_type>>
class general_heston_equation_explicit_kernel
{
};

// ===================================================================
// ============================== DEVICE =============================
// ===================================================================
template <typename fp_type, template <typename, typename> typename container, typename allocator>
class general_heston_equation_explicit_kernel<memory_space_enum::Device, fp_type, container, allocator>
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container_2d<by_enum::Row, fp_type, container, allocator> rcontainer_2d_t;
    typedef container_3d<by_enum::Row, fp_type, container, allocator> rcontainer_3d_t;

  private:
    boundary_2d_ptr<fp_type> boundary_ver_;
    boundary_2d_pair<fp_type> boundary_pair_hor_;
    heat_data_transform_2d_ptr<fp_type> heat_data_cfg_;
    pde_discretization_config_2d_ptr<fp_type> discretization_cfg_;
    heat_explicit_solver_config_ptr solver_cfg_;
    grid_config_2d_ptr<fp_type> grid_cfg_;

  public:
    general_heston_equation_explicit_kernel(boundary_2d_ptr<fp_type> const &vertical_upper_boundary_ptr,
                                            boundary_2d_pair<fp_type> const &horizontal_boundary_pair,
                                            heat_data_transform_2d_ptr<fp_type> const &heat_data_config,
                                            pde_discretization_config_2d_ptr<fp_type> const &discretization_config,
                                            heat_explicit_solver_config_ptr const &solver_config,
                                            grid_config_2d_ptr<fp_type> const &grid_config)
        : boundary_ver_{vertical_upper_boundary_ptr}, boundary_pair_hor_{horizontal_boundary_pair},
          heat_data_cfg_{heat_data_config}, discretization_cfg_{discretization_config},
          solver_cfg_{solver_config}, grid_cfg_{grid_config}
    {
    }

    void operator()(rcontainer_2d_t &prev_solution, rcontainer_2d_t &next_solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type, fp_type)> const &heat_source)
    {
        // save traverse_direction
        const traverse_direction_enum traverse_dir = solver_cfg_->traverse_direction();
        // create a Heston coefficient holder:
        auto const heston_coeff_holder = std::make_shared<general_heston_equation_coefficients<fp_type>>(
            heat_data_cfg_, discretization_cfg_, nullptr, fp_type{0.0});
        // Here make a dicision which explicit scheme to launch:
        if (solver_cfg_->explicit_pde_scheme() == explicit_pde_schemes_enum::Euler)
        {

            typedef heston_euler_cuda_scheme<fp_type, container, allocator> heston_euler_cuda_scheme_t;
            heston_euler_cuda_scheme_t euler_scheme(heston_coeff_holder, boundary_ver_, boundary_pair_hor_,
                                                    discretization_cfg_, grid_cfg_);
            euler_scheme(prev_solution, next_solution, is_heat_sourse_set, heat_source, traverse_dir);
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

    void operator()(rcontainer_2d_t &prev_solution, rcontainer_2d_t &next_solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type, fp_type)> const &heat_source, rcontainer_3d_t &solutions)
    {
    }
};

// ===================================================================
// ================================ HOST =============================
// ===================================================================
template <typename fp_type, template <typename, typename> typename container, typename allocator>
class general_heston_equation_explicit_kernel<memory_space_enum::Host, fp_type, container, allocator>
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container_2d<by_enum::Row, fp_type, container, allocator> rcontainer_2d_t;
    typedef container_3d<by_enum::Row, fp_type, container, allocator> rcontainer_3d_t;

  private:
    boundary_2d_ptr<fp_type> boundary_ver_;
    boundary_2d_pair<fp_type> boundary_pair_hor_;
    heat_data_transform_2d_ptr<fp_type> heat_data_cfg_;
    pde_discretization_config_2d_ptr<fp_type> discretization_cfg_;
    heat_explicit_solver_config_ptr solver_cfg_;
    grid_config_2d_ptr<fp_type> grid_cfg_;

  public:
    general_heston_equation_explicit_kernel(boundary_2d_ptr<fp_type> const &vertical_upper_boundary_ptr,
                                            boundary_2d_pair<fp_type> const &horizontal_boundary_pair,
                                            heat_data_transform_2d_ptr<fp_type> const &heat_data_config,
                                            pde_discretization_config_2d_ptr<fp_type> const &discretization_config,
                                            heat_explicit_solver_config_ptr const &solver_config,
                                            grid_config_2d_ptr<fp_type> const &grid_config)
        : boundary_ver_{vertical_upper_boundary_ptr}, boundary_pair_hor_{horizontal_boundary_pair},
          heat_data_cfg_{heat_data_config}, discretization_cfg_{discretization_config},
          solver_cfg_{solver_config}, grid_cfg_{grid_config}
    {
    }

    void operator()(rcontainer_2d_t &prev_solution, rcontainer_2d_t &next_solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type, fp_type)> const &heat_source)
    {
        // save traverse_direction
        const traverse_direction_enum traverse_dir = solver_cfg_->traverse_direction();
        // create a Heston coefficient holder:
        auto const heston_coeff_holder = std::make_shared<general_heston_equation_coefficients<fp_type>>(
            heat_data_cfg_, discretization_cfg_, nullptr, fp_type{0.0});
        // Here make a dicision which explicit scheme to launch:
        if (solver_cfg_->explicit_pde_scheme() == explicit_pde_schemes_enum::Euler)
        {
            typedef heston_euler_scheme<fp_type, container, allocator> heston_euler_scheme_t;
            heston_euler_scheme_t euler_scheme(heston_coeff_holder, boundary_ver_, boundary_pair_hor_,
                                               discretization_cfg_, grid_cfg_);
            euler_scheme(prev_solution, next_solution, is_heat_sourse_set, heat_source, traverse_dir);
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

    void operator()(rcontainer_2d_t &prev_solution, rcontainer_2d_t &next_solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type, fp_type)> const &heat_source, rcontainer_3d_t &solutions)
    {
    }
};

} // namespace two_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_2D_GENERAL_HESTON_EQUATION_EXPLICIT_KERNEL_HPP_
