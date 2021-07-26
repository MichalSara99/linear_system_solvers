#if !defined(_LSS_SOLVER_CONFIG_HPP_)
#define _LSS_SOLVER_CONFIG_HPP_

#include <map>

#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"

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

// NOTE:
// check the dimension_enum template paramerter - it could be removed

template <dimension_enum dimension> struct implicit_solver_config
{
};

template <dimension_enum dimension> struct explicit_solver_config
{
};

/**
    1D implicit_solver_config structure
 */
template <> struct implicit_solver_config<dimension_enum::One>
{
  private:
    memory_space_enum memory_space_;
    traverse_direction_enum traverse_direction_;
    tridiagonal_method_enum tridiagonal_method_;
    factorization_enum tridiagonal_factorization_;
    implicit_pde_schemes_enum implicit_pde_scheme_;

    explicit implicit_solver_config() = delete;

    void initialize()
    {
        if (memory_space_ == memory_space_enum::Device)
        {
            LSS_VERIFY(!(tridiagonal_method_ == tridiagonal_method_enum::DoubleSweepSolver),
                       "No support for Double Sweep Solver on Device");
            LSS_VERIFY(!(tridiagonal_method_ == tridiagonal_method_enum::ThomasLUSolver),
                       "No support for Tomas LU Solver on Device");
        }

        if (tridiagonal_method_ == tridiagonal_method_enum::DoubleSweepSolver)
        {
            LSS_VERIFY(!(tridiagonal_factorization_ == factorization_enum::CholeskyMethod),
                       "No support for Cholesky Method factorization for Double Sweep Solver");
            LSS_VERIFY(!(tridiagonal_factorization_ == factorization_enum::LUMethod),
                       "No support for LU Method factorization for Double Sweep Solver");
            LSS_VERIFY(!(tridiagonal_factorization_ == factorization_enum::QRMethod),
                       "No support for QR Method factorization for Double Sweep "
                       "Solver");
        }

        if (tridiagonal_method_ == tridiagonal_method_enum::ThomasLUSolver)
        {
            LSS_VERIFY(!(tridiagonal_factorization_ == factorization_enum::CholeskyMethod),
                       "No support for Cholesky Method factorization for Thomas "
                       "LU Solver");
            LSS_VERIFY(!(tridiagonal_factorization_ == factorization_enum::LUMethod),
                       "No support for LU Method factorization for Thomas LU Solver");
            LSS_VERIFY(!(tridiagonal_factorization_ == factorization_enum::QRMethod),
                       "No support for QR Method factorization for Thomas LU "
                       "Solver");
        }

        if (tridiagonal_method_ == tridiagonal_method_enum::SORSolver)
        {
            LSS_VERIFY(!(tridiagonal_factorization_ == factorization_enum::CholeskyMethod),
                       "No support for Cholesky Method factorization for SOR"
                       " Solver");
            LSS_VERIFY(!(tridiagonal_factorization_ == factorization_enum::LUMethod),
                       "No support for LU Method factorization for SOR Solver");
            LSS_VERIFY(!(tridiagonal_factorization_ == factorization_enum::QRMethod),
                       "No support for QR Method factorization for SOR "
                       "Solver");
        }
    }

  public:
    explicit implicit_solver_config(memory_space_enum const &memory_space,
                                    traverse_direction_enum const &traverse_direction,
                                    tridiagonal_method_enum const &tridiagonal_method,
                                    factorization_enum const &tridiagonal_factorization,
                                    implicit_pde_schemes_enum const &implicit_pde_scheme)
        : memory_space_{memory_space}, traverse_direction_{traverse_direction}, tridiagonal_method_{tridiagonal_method},
          tridiagonal_factorization_{tridiagonal_factorization}, implicit_pde_scheme_{implicit_pde_scheme}
    {
        initialize();
    }
    ~implicit_solver_config()
    {
    }

    inline memory_space_enum memory_space() const
    {
        return memory_space_;
    }

    inline traverse_direction_enum traverse_direction() const
    {
        return traverse_direction_;
    }

    inline tridiagonal_method_enum tridiagonal_method() const
    {
        return tridiagonal_method_;
    }

    inline factorization_enum tridiagonal_factorization() const
    {
        return tridiagonal_factorization_;
    }

    inline implicit_pde_schemes_enum implicit_pde_scheme() const
    {
        return implicit_pde_scheme_;
    }
};

/**
    1D explicit_solver_config structure
 */
template <> struct explicit_solver_config<dimension_enum::One>
{
  private:
    memory_space_enum memory_space_;
    traverse_direction_enum traverse_direction_;
    explicit_pde_schemes_enum explicit_pde_scheme_;

    explicit explicit_solver_config() = delete;

  public:
    explicit explicit_solver_config(memory_space_enum const &memory_space,
                                    traverse_direction_enum const &traverse_direction,
                                    explicit_pde_schemes_enum const &explicit_pde_scheme)
        : memory_space_{memory_space}, traverse_direction_{traverse_direction}, explicit_pde_scheme_{
                                                                                    explicit_pde_scheme}
    {
    }
    ~explicit_solver_config()
    {
    }

    inline memory_space_enum memory_space() const
    {
        return memory_space_;
    }

    inline traverse_direction_enum traverse_direction() const
    {
        return traverse_direction_;
    }

    inline explicit_pde_schemes_enum explicit_pde_scheme() const
    {
        return explicit_pde_scheme_;
    }
};

/**
    2D implicit_solver_config structure
 */
template <> struct implicit_solver_config<dimension_enum::Two>
{
};

/**
    2D explicit_solver_config structure
 */
template <> struct explicit_solver_config<dimension_enum::Two>
{
};

using implicit_solver_config_1d = implicit_solver_config<dimension_enum::One>;

using implicit_solver_config_2d = implicit_solver_config<dimension_enum::Two>;

using explicit_solver_config_1d = explicit_solver_config<dimension_enum::One>;

using explicit_solver_config_2d = explicit_solver_config<dimension_enum::Two>;

using implicit_solver_config_1d_ptr = sptr_t<implicit_solver_config<dimension_enum::One>>;

using implicit_solver_config_2d_ptr = sptr_t<implicit_solver_config<dimension_enum::Two>>;

using explicit_solver_config_1d_ptr = sptr_t<explicit_solver_config<dimension_enum::One>>;

using explicit_solver_config_2d_ptr = sptr_t<explicit_solver_config<dimension_enum::Two>>;

// =================================================
// ===== some default implicit solver configs ======
// =================================================
// forward stepping:
auto dev_fwd_cusolver_qr_euler_solver_config_ptr = std::make_shared<implicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Device, traverse_direction_enum::Forward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::QRMethod, implicit_pde_schemes_enum::Euler);

auto dev_fwd_cusolver_qr_cn_solver_config_ptr = std::make_shared<implicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Device, traverse_direction_enum::Forward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::QRMethod, implicit_pde_schemes_enum::CrankNicolson);

auto dev_fwd_cusolver_lu_euler_solver_config_ptr = std::make_shared<implicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Device, traverse_direction_enum::Forward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::LUMethod, implicit_pde_schemes_enum::Euler);

auto dev_fwd_cusolver_lu_cn_solver_config_ptr = std::make_shared<implicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Device, traverse_direction_enum::Forward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::LUMethod, implicit_pde_schemes_enum::CrankNicolson);

auto dev_fwd_cusolver_euler_solver_config_ptr = std::make_shared<implicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Device, traverse_direction_enum::Forward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::None, implicit_pde_schemes_enum::Euler);

auto host_fwd_cusolver_qr_euler_solver_config_ptr = std::make_shared<implicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Host, traverse_direction_enum::Forward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::QRMethod, implicit_pde_schemes_enum::Euler);

auto host_fwd_cusolver_qr_cn_solver_config_ptr = std::make_shared<implicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Host, traverse_direction_enum::Forward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::QRMethod, implicit_pde_schemes_enum::CrankNicolson);

auto host_fwd_cusolver_lu_euler_solver_config_ptr = std::make_shared<implicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Host, traverse_direction_enum::Forward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::LUMethod, implicit_pde_schemes_enum::Euler);

auto host_fwd_cusolver_lu_cn_solver_config_ptr = std::make_shared<implicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Host, traverse_direction_enum::Forward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::LUMethod, implicit_pde_schemes_enum::CrankNicolson);

auto host_fwd_cusolver_euler_solver_config_ptr = std::make_shared<implicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Host, traverse_direction_enum::Forward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::None, implicit_pde_schemes_enum::Euler);

auto dev_fwd_sorsolver_euler_solver_config_ptr = std::make_shared<implicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Device, traverse_direction_enum::Forward, tridiagonal_method_enum::SORSolver,
    factorization_enum::None, implicit_pde_schemes_enum::Euler);

auto dev_fwd_sorsolver_cn_solver_config_ptr = std::make_shared<implicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Device, traverse_direction_enum::Forward, tridiagonal_method_enum::SORSolver,
    factorization_enum::None, implicit_pde_schemes_enum::CrankNicolson);

auto host_fwd_sorsolver_euler_solver_config_ptr = std::make_shared<implicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Host, traverse_direction_enum::Forward, tridiagonal_method_enum::SORSolver,
    factorization_enum::None, implicit_pde_schemes_enum::Euler);

auto host_fwd_sorsolver_cn_solver_config_ptr = std::make_shared<implicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Host, traverse_direction_enum::Forward, tridiagonal_method_enum::SORSolver,
    factorization_enum::None, implicit_pde_schemes_enum::CrankNicolson);

auto host_fwd_dssolver_euler_solver_config_ptr = std::make_shared<implicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Host, traverse_direction_enum::Forward, tridiagonal_method_enum::DoubleSweepSolver,
    factorization_enum::None, implicit_pde_schemes_enum::Euler);

auto host_fwd_dssolver_cn_solver_config_ptr = std::make_shared<implicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Host, traverse_direction_enum::Forward, tridiagonal_method_enum::DoubleSweepSolver,
    factorization_enum::None, implicit_pde_schemes_enum::CrankNicolson);

auto host_fwd_tlusolver_euler_solver_config_ptr = std::make_shared<implicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Host, traverse_direction_enum::Forward, tridiagonal_method_enum::ThomasLUSolver,
    factorization_enum::None, implicit_pde_schemes_enum::Euler);

auto host_fwd_tlusolver_cn_solver_config_ptr = std::make_shared<implicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Host, traverse_direction_enum::Forward, tridiagonal_method_enum::ThomasLUSolver,
    factorization_enum::None, implicit_pde_schemes_enum::CrankNicolson);
// backward stepping:
auto dev_bwd_cusolver_qr_euler_solver_config_ptr = std::make_shared<implicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Device, traverse_direction_enum::Backward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::QRMethod, implicit_pde_schemes_enum::Euler);

auto dev_bwd_cusolver_qr_cn_solver_config_ptr = std::make_shared<implicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Device, traverse_direction_enum::Backward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::QRMethod, implicit_pde_schemes_enum::CrankNicolson);

auto dev_bwd_cusolver_lu_euler_solver_config_ptr = std::make_shared<implicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Device, traverse_direction_enum::Backward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::LUMethod, implicit_pde_schemes_enum::Euler);

auto dev_bwd_cusolver_lu_cn_solver_config_ptr = std::make_shared<implicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Device, traverse_direction_enum::Backward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::LUMethod, implicit_pde_schemes_enum::CrankNicolson);

auto dev_bwd_cusolver_euler_solver_config_ptr = std::make_shared<implicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Device, traverse_direction_enum::Backward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::None, implicit_pde_schemes_enum::Euler);

auto host_bwd_cusolver_qr_euler_solver_config_ptr = std::make_shared<implicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Host, traverse_direction_enum::Backward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::QRMethod, implicit_pde_schemes_enum::Euler);

auto host_bwd_cusolver_qr_cn_solver_config_ptr = std::make_shared<implicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Host, traverse_direction_enum::Backward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::QRMethod, implicit_pde_schemes_enum::CrankNicolson);

auto host_bwd_cusolver_lu_euler_solver_config_ptr = std::make_shared<implicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Host, traverse_direction_enum::Backward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::LUMethod, implicit_pde_schemes_enum::Euler);

auto host_bwd_cusolver_lu_cn_solver_config_ptr = std::make_shared<implicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Host, traverse_direction_enum::Backward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::LUMethod, implicit_pde_schemes_enum::CrankNicolson);

auto host_bwd_cusolver_euler_solver_config_ptr = std::make_shared<implicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Host, traverse_direction_enum::Backward, tridiagonal_method_enum::CUDASolver,
    factorization_enum::None, implicit_pde_schemes_enum::Euler);

auto dev_bwd_sorsolver_euler_solver_config_ptr = std::make_shared<implicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Device, traverse_direction_enum::Backward, tridiagonal_method_enum::SORSolver,
    factorization_enum::None, implicit_pde_schemes_enum::Euler);

auto dev_bwd_sorsolver_cn_solver_config_ptr = std::make_shared<implicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Device, traverse_direction_enum::Backward, tridiagonal_method_enum::SORSolver,
    factorization_enum::None, implicit_pde_schemes_enum::CrankNicolson);

auto host_bwd_sorsolver_euler_solver_config_ptr = std::make_shared<implicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Host, traverse_direction_enum::Backward, tridiagonal_method_enum::SORSolver,
    factorization_enum::None, implicit_pde_schemes_enum::Euler);

auto host_bwd_sorsolver_cn_solver_config_ptr = std::make_shared<implicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Host, traverse_direction_enum::Backward, tridiagonal_method_enum::SORSolver,
    factorization_enum::None, implicit_pde_schemes_enum::CrankNicolson);

auto host_bwd_dssolver_euler_solver_config_ptr = std::make_shared<implicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Host, traverse_direction_enum::Backward, tridiagonal_method_enum::DoubleSweepSolver,
    factorization_enum::None, implicit_pde_schemes_enum::Euler);

auto host_bwd_dssolver_cn_solver_config_ptr = std::make_shared<implicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Host, traverse_direction_enum::Backward, tridiagonal_method_enum::DoubleSweepSolver,
    factorization_enum::None, implicit_pde_schemes_enum::CrankNicolson);

auto host_bwd_tlusolver_euler_solver_config_ptr = std::make_shared<implicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Host, traverse_direction_enum::Backward, tridiagonal_method_enum::ThomasLUSolver,
    factorization_enum::None, implicit_pde_schemes_enum::Euler);

auto host_bwd_tlusolver_cn_solver_config_ptr = std::make_shared<implicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Host, traverse_direction_enum::Backward, tridiagonal_method_enum::ThomasLUSolver,
    factorization_enum::None, implicit_pde_schemes_enum::CrankNicolson);

// =================================================
// ===== some default explicit solver configs ======
// =================================================
auto dev_expl_fwd_euler_solver_config_ptr = std::make_shared<explicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Device, traverse_direction_enum::Forward, explicit_pde_schemes_enum::Euler);

auto host_expl_fwd_euler_solver_config_ptr = std::make_shared<explicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Host, traverse_direction_enum::Forward, explicit_pde_schemes_enum::Euler);

auto dev_expl_bwd_euler_solver_config_ptr = std::make_shared<explicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Device, traverse_direction_enum::Backward, explicit_pde_schemes_enum::Euler);

auto host_expl_bwd_euler_solver_config_ptr = std::make_shared<explicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Host, traverse_direction_enum::Backward, explicit_pde_schemes_enum::Euler);

auto host_expl_fwd_bc_solver_config_ptr = std::make_shared<explicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Host, traverse_direction_enum::Forward, explicit_pde_schemes_enum::ADEBarakatClark);

auto host_expl_bwd_bc_solver_config_ptr = std::make_shared<explicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Host, traverse_direction_enum::Backward, explicit_pde_schemes_enum::ADEBarakatClark);

auto host_expl_fwd_s_solver_config_ptr = std::make_shared<explicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Host, traverse_direction_enum::Forward, explicit_pde_schemes_enum::ADESaulyev);

auto host_expl_bwd_s_solver_config_ptr = std::make_shared<explicit_solver_config<dimension_enum::One>>(
    memory_space_enum::Host, traverse_direction_enum::Backward, explicit_pde_schemes_enum::ADESaulyev);

} // namespace lss_pde_solvers

#endif ///_LSS_SOLVER_CONFIG_HPP_
