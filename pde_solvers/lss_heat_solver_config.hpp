#if !defined(_LSS_HEAT_SOLVER_CONFIG_HPP_)
#define _LSS_HEAT_SOLVER_CONFIG_HPP_

#include <map>

#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include "lss_pde_solver_config.hpp"

namespace lss_pde_solvers
{

using lss_enumerations::dimension_enum;
using lss_enumerations::explicit_pde_schemes_enum;
using lss_enumerations::factorization_enum;
using lss_enumerations::implicit_pde_schemes_enum;
using lss_enumerations::memory_space_enum;
using lss_enumerations::traverse_direction_enum;
using lss_enumerations::tridiagonal_method_enum;
using lss_utility::range;
using lss_utility::sptr_t;

/**
    heat_implicit_solver_config structure
 */
struct heat_implicit_solver_config : public pde_implicit_solver_config
{
  private:
    implicit_pde_schemes_enum implicit_pde_scheme_;

    explicit heat_implicit_solver_config() = delete;

  public:
    explicit heat_implicit_solver_config(memory_space_enum const &memory_space,
                                         traverse_direction_enum const &traverse_direction,
                                         tridiagonal_method_enum const &tridiagonal_method,
                                         factorization_enum const &tridiagonal_factorization,
                                         implicit_pde_schemes_enum const &implicit_pde_scheme)
        : pde_implicit_solver_config{memory_space, traverse_direction, tridiagonal_method, tridiagonal_factorization},
          implicit_pde_scheme_{implicit_pde_scheme}
    {
    }
    ~heat_implicit_solver_config()
    {
    }

    inline implicit_pde_schemes_enum implicit_pde_scheme() const
    {
        return implicit_pde_scheme_;
    }
};

/**
    heat_explicit_solver_config structure
 */
struct heat_explicit_solver_config : public pde_explicit_solver_config
{
  private:
    explicit_pde_schemes_enum explicit_pde_scheme_;

    explicit heat_explicit_solver_config() = delete;

  public:
    explicit heat_explicit_solver_config(memory_space_enum const &memory_space,
                                         traverse_direction_enum const &traverse_direction,
                                         explicit_pde_schemes_enum const &explicit_pde_scheme)
        : pde_explicit_solver_config{memory_space, traverse_direction}, explicit_pde_scheme_{explicit_pde_scheme}
    {
    }
    ~heat_explicit_solver_config()
    {
    }

    inline explicit_pde_schemes_enum explicit_pde_scheme() const
    {
        return explicit_pde_scheme_;
    }
};

using heat_implicit_solver_config_ptr = sptr_t<heat_implicit_solver_config>;

using heat_explicit_solver_config_ptr = sptr_t<heat_explicit_solver_config>;

namespace default_heat_solver_configs
{
// =================================================
// ===== some default implicit solver configs ======
// =================================================
// forward stepping:

auto dev_fwd_cusolver_qr_euler_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Device, traverse_direction_enum::Forward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::QRMethod, implicit_pde_schemes_enum::Euler);

auto dev_fwd_cusolver_qr_cn_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Device, traverse_direction_enum::Forward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::QRMethod, implicit_pde_schemes_enum::CrankNicolson);

auto dev_fwd_cusolver_lu_euler_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Device, traverse_direction_enum::Forward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::LUMethod, implicit_pde_schemes_enum::Euler);

auto dev_fwd_cusolver_lu_cn_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Device, traverse_direction_enum::Forward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::LUMethod, implicit_pde_schemes_enum::CrankNicolson);

auto dev_fwd_cusolver_euler_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Device, traverse_direction_enum::Forward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::None, implicit_pde_schemes_enum::Euler);

auto host_fwd_cusolver_qr_euler_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Host, traverse_direction_enum::Forward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::QRMethod, implicit_pde_schemes_enum::Euler);

auto host_fwd_cusolver_qr_cn_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Host, traverse_direction_enum::Forward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::QRMethod, implicit_pde_schemes_enum::CrankNicolson);

auto host_fwd_cusolver_lu_euler_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Host, traverse_direction_enum::Forward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::LUMethod, implicit_pde_schemes_enum::Euler);

auto host_fwd_cusolver_lu_cn_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Host, traverse_direction_enum::Forward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::LUMethod, implicit_pde_schemes_enum::CrankNicolson);

auto host_fwd_cusolver_euler_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Host, traverse_direction_enum::Forward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::None, implicit_pde_schemes_enum::Euler);

auto dev_fwd_sorsolver_euler_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Device, traverse_direction_enum::Forward, tridiagonal_method_enum::SORSolver,
    factorization_enum::None, implicit_pde_schemes_enum::Euler);

auto dev_fwd_sorsolver_cn_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Device, traverse_direction_enum::Forward, tridiagonal_method_enum::SORSolver,
    factorization_enum::None, implicit_pde_schemes_enum::CrankNicolson);

auto host_fwd_sorsolver_euler_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Host, traverse_direction_enum::Forward, tridiagonal_method_enum::SORSolver,
    factorization_enum::None, implicit_pde_schemes_enum::Euler);

auto host_fwd_sorsolver_cn_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Host, traverse_direction_enum::Forward, tridiagonal_method_enum::SORSolver,
    factorization_enum::None, implicit_pde_schemes_enum::CrankNicolson);

auto host_fwd_dssolver_euler_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Host, traverse_direction_enum::Forward, tridiagonal_method_enum::DoubleSweepSolver,
    factorization_enum::None, implicit_pde_schemes_enum::Euler);

auto host_fwd_dssolver_cn_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Host, traverse_direction_enum::Forward, tridiagonal_method_enum::DoubleSweepSolver,
    factorization_enum::None, implicit_pde_schemes_enum::CrankNicolson);

auto host_fwd_tlusolver_euler_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Host, traverse_direction_enum::Forward, tridiagonal_method_enum::ThomasLUSolver,
    factorization_enum::None, implicit_pde_schemes_enum::Euler);

auto host_fwd_tlusolver_cn_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Host, traverse_direction_enum::Forward, tridiagonal_method_enum::ThomasLUSolver,
    factorization_enum::None, implicit_pde_schemes_enum::CrankNicolson);

// backward stepping:

auto dev_bwd_cusolver_qr_euler_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Device, traverse_direction_enum::Backward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::QRMethod, implicit_pde_schemes_enum::Euler);

auto dev_bwd_cusolver_qr_cn_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Device, traverse_direction_enum::Backward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::QRMethod, implicit_pde_schemes_enum::CrankNicolson);

auto dev_bwd_cusolver_lu_euler_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Device, traverse_direction_enum::Backward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::LUMethod, implicit_pde_schemes_enum::Euler);

auto dev_bwd_cusolver_lu_cn_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Device, traverse_direction_enum::Backward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::LUMethod, implicit_pde_schemes_enum::CrankNicolson);

auto dev_bwd_cusolver_euler_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Device, traverse_direction_enum::Backward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::None, implicit_pde_schemes_enum::Euler);

auto host_bwd_cusolver_qr_euler_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Host, traverse_direction_enum::Backward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::QRMethod, implicit_pde_schemes_enum::Euler);

auto host_bwd_cusolver_qr_cn_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Host, traverse_direction_enum::Backward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::QRMethod, implicit_pde_schemes_enum::CrankNicolson);

auto host_bwd_cusolver_lu_euler_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Host, traverse_direction_enum::Backward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::LUMethod, implicit_pde_schemes_enum::Euler);

auto host_bwd_cusolver_lu_cn_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Host, traverse_direction_enum::Backward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::LUMethod, implicit_pde_schemes_enum::CrankNicolson);

auto host_bwd_cusolver_euler_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Host, traverse_direction_enum::Backward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::None, implicit_pde_schemes_enum::Euler);

auto dev_bwd_sorsolver_euler_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Device, traverse_direction_enum::Backward, tridiagonal_method_enum::SORSolver,
    factorization_enum::None, implicit_pde_schemes_enum::Euler);

auto dev_bwd_sorsolver_cn_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Device, traverse_direction_enum::Backward, tridiagonal_method_enum::SORSolver,
    factorization_enum::None, implicit_pde_schemes_enum::CrankNicolson);

auto host_bwd_sorsolver_euler_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Host, traverse_direction_enum::Backward, tridiagonal_method_enum::SORSolver,
    factorization_enum::None, implicit_pde_schemes_enum::Euler);

auto host_bwd_sorsolver_cn_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Host, traverse_direction_enum::Backward, tridiagonal_method_enum::SORSolver,
    factorization_enum::None, implicit_pde_schemes_enum::CrankNicolson);

auto host_bwd_dssolver_euler_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Host, traverse_direction_enum::Backward, tridiagonal_method_enum::DoubleSweepSolver,
    factorization_enum::None, implicit_pde_schemes_enum::Euler);

auto host_bwd_dssolver_cn_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Host, traverse_direction_enum::Backward, tridiagonal_method_enum::DoubleSweepSolver,
    factorization_enum::None, implicit_pde_schemes_enum::CrankNicolson);

auto host_bwd_tlusolver_euler_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Host, traverse_direction_enum::Backward, tridiagonal_method_enum::ThomasLUSolver,
    factorization_enum::None, implicit_pde_schemes_enum::Euler);

auto host_bwd_tlusolver_cn_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Host, traverse_direction_enum::Backward, tridiagonal_method_enum::ThomasLUSolver,
    factorization_enum::None, implicit_pde_schemes_enum::CrankNicolson);

auto host_bwd_tlusolver_o8_solver_config_ptr = std::make_shared<heat_implicit_solver_config>(
    memory_space_enum::Host, traverse_direction_enum::Backward, tridiagonal_method_enum::ThomasLUSolver,
    factorization_enum::None, implicit_pde_schemes_enum::Theta_80);

// =================================================
// ===== some default explicit solver configs ======
// =================================================
auto dev_expl_fwd_euler_solver_config_ptr = std::make_shared<heat_explicit_solver_config>(
    memory_space_enum::Device, traverse_direction_enum::Forward, explicit_pde_schemes_enum::Euler);

auto host_expl_fwd_euler_solver_config_ptr = std::make_shared<heat_explicit_solver_config>(
    memory_space_enum::Host, traverse_direction_enum::Forward, explicit_pde_schemes_enum::Euler);

auto dev_expl_bwd_euler_solver_config_ptr = std::make_shared<heat_explicit_solver_config>(
    memory_space_enum::Device, traverse_direction_enum::Backward, explicit_pde_schemes_enum::Euler);

auto host_expl_bwd_euler_solver_config_ptr = std::make_shared<heat_explicit_solver_config>(
    memory_space_enum::Host, traverse_direction_enum::Backward, explicit_pde_schemes_enum::Euler);

auto host_expl_fwd_bc_solver_config_ptr = std::make_shared<heat_explicit_solver_config>(
    memory_space_enum::Host, traverse_direction_enum::Forward, explicit_pde_schemes_enum::ADEBarakatClark);

auto host_expl_bwd_bc_solver_config_ptr = std::make_shared<heat_explicit_solver_config>(
    memory_space_enum::Host, traverse_direction_enum::Backward, explicit_pde_schemes_enum::ADEBarakatClark);

auto host_expl_fwd_s_solver_config_ptr = std::make_shared<heat_explicit_solver_config>(
    memory_space_enum::Host, traverse_direction_enum::Forward, explicit_pde_schemes_enum::ADESaulyev);

auto host_expl_bwd_s_solver_config_ptr = std::make_shared<heat_explicit_solver_config>(
    memory_space_enum::Host, traverse_direction_enum::Backward, explicit_pde_schemes_enum::ADESaulyev);
} // namespace default_heat_solver_configs
} // namespace lss_pde_solvers

#endif ///_LSS_HEAT_SOLVER_CONFIG_HPP_
