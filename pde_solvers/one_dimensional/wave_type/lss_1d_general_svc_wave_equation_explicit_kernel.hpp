#if !defined(_LSS_1D_GENERAL_SVC_WAVE_EQUATION_EXPLICIT_KERNEL_HPP_)
#define _LSS_1D_GENERAL_SVC_WAVE_EQUATION_EXPLICIT_KERNEL_HPP_

#include <vector>

#include "boundaries/lss_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include "containers/lss_container_2d.hpp"
#include "discretization/lss_discretization.hpp"
#include "explicit_schemes/lss_wave_euler_svc_cuda_scheme.hpp"
#include "explicit_schemes/lss_wave_euler_svc_scheme.hpp"
#include "pde_solvers/lss_pde_discretization_config.hpp"
#include "pde_solvers/lss_wave_solver_config.hpp"

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
using lss_utility::function_quintuple_t;
using lss_utility::NaN;
using lss_utility::range;

template <memory_space_enum memory_enum, typename fp_type,
          template <typename, typename> typename container = std::vector, typename allocator = std::allocator<fp_type>>
class general_svc_wave_equation_explicit_kernel
{
};

// ===================================================================
// ============================== DEVICE =============================
// ===================================================================
template <typename fp_type, template <typename, typename> typename container, typename allocator>
class general_svc_wave_equation_explicit_kernel<memory_space_enum::Device, fp_type, container, allocator>
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container<fp_type, allocator> container_t;

  private:
    function_quintuple_t<fp_type> fun_quintuple_;
    boundary_1d_pair<fp_type> boundary_pair_;
    pde_discretization_config_1d_ptr<fp_type> discretization_cfg_;
    wave_explicit_solver_config_ptr solver_cfg_;

  public:
    general_svc_wave_equation_explicit_kernel(function_quintuple_t<fp_type> const &fun_quintuple,
                                              boundary_1d_pair<fp_type> const &boundary_pair,
                                              pde_discretization_config_1d_ptr<fp_type> const &discretization_config,
                                              wave_explicit_solver_config_ptr const &solver_config)
        : fun_quintuple_{fun_quintuple}, boundary_pair_{boundary_pair}, discretization_cfg_{discretization_config},
          solver_cfg_{solver_config}
    {
    }

    void operator()(container_t &prev_solution_0, container_t &prev_solution_1, container_t &next_solution,
                    bool is_wave_sourse_set, std::function<fp_type(fp_type, fp_type)> const &wave_source)
    {
        // save traverse_direction
        const traverse_direction_enum traverse_dir = solver_cfg_->traverse_direction();

        // Here we have only Euler discretization available:

        typedef wave_euler_svc_cuda_scheme<fp_type, container, allocator> euler_cuda_scheme_t;
        euler_cuda_scheme_t euler_scheme(fun_quintuple_, boundary_pair_, discretization_cfg_);
        euler_scheme(prev_solution_0, prev_solution_1, next_solution, is_wave_sourse_set, wave_source, traverse_dir);
    }

    void operator()(container_t &prev_solution_0, container_t &prev_solution_1, container_t &next_solution,
                    bool is_wave_sourse_set, std::function<fp_type(fp_type, fp_type)> const &wave_source,
                    container_2d<by_enum::Row, fp_type, container, allocator> &solutions)
    {
        // save traverse_direction
        const traverse_direction_enum traverse_dir = solver_cfg_->traverse_direction();

        // Here we have only Euler discretization available:

        typedef wave_euler_svc_cuda_scheme<fp_type, container, allocator> euler_cuda_scheme_t;
        euler_cuda_scheme_t euler_scheme(fun_quintuple_, boundary_pair_, discretization_cfg_);
        euler_scheme(prev_solution_0, prev_solution_1, next_solution, is_wave_sourse_set, wave_source, traverse_dir,
                     solutions);
    }
};

// ===================================================================
// ============================== HOST ===============================
// ===================================================================

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class general_svc_wave_equation_explicit_kernel<memory_space_enum::Host, fp_type, container, allocator>
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container<fp_type, allocator> container_t;

  private:
    function_quintuple_t<fp_type> fun_quintuple_;
    boundary_1d_pair<fp_type> boundary_pair_;
    pde_discretization_config_1d_ptr<fp_type> discretization_cfg_;
    wave_explicit_solver_config_ptr solver_cfg_;

  public:
    class general_svc_wave_equation_explicit_kernel(
        function_quintuple_t<fp_type> const &fun_quintuple, boundary_1d_pair<fp_type> const &boundary_pair,
        pde_discretization_config_1d_ptr<fp_type> const &discretization_config,
        wave_explicit_solver_config_ptr const &solver_config)
        : fun_quintuple_{fun_quintuple}, boundary_pair_{boundary_pair}, discretization_cfg_{discretization_config},
          solver_cfg_{solver_config}
    {
    }

    void operator()(container_t &prev_solution_0, container_t &prev_solution_1, container_t &next_solution,
                    bool is_wave_sourse_set, std::function<fp_type(fp_type, fp_type)> const &wave_source)
    {
        // save traverse_direction
        const traverse_direction_enum traverse_dir = solver_cfg_->traverse_direction();
        // Here make a dicision which explicit scheme to launch:

        typedef wave_euler_svc_scheme<fp_type, container, allocator> euler_scheme_t;
        euler_scheme_t euler_scheme(fun_quintuple_, boundary_pair_, discretization_cfg_);
        euler_scheme(prev_solution_0, prev_solution_1, next_solution, is_wave_sourse_set, wave_source, traverse_dir);
    }

    void operator()(container_t &prev_solution_0, container_t &prev_solution_1, container_t &next_solution,
                    bool is_wave_sourse_set, std::function<fp_type(fp_type, fp_type)> const &wave_source,
                    container_2d<by_enum::Row, fp_type, container, allocator> &solutions)
    {
        // save traverse_direction
        const traverse_direction_enum traverse_dir = solver_cfg_->traverse_direction();
        // Here make a dicision which explicit scheme to launch:

        typedef wave_euler_svc_scheme<fp_type, container, allocator> euler_scheme_t;
        euler_scheme_t euler_scheme(fun_quintuple_, boundary_pair_, discretization_cfg_);
        euler_scheme(prev_solution_0, prev_solution_1, next_solution, is_wave_sourse_set, wave_source, traverse_dir,
                     solutions);
    }
};

} // namespace one_dimensional
} // namespace lss_pde_solvers
#endif ///_LSS_1D_GENERAL_SVC_WAVE_EQUATION_EXPLICIT_KERNEL_HPP_
