#if !defined(_LSS_ODE_SOLVER_CONFIG_HPP_)
#define _LSS_ODE_SOLVER_CONFIG_HPP_

#include <map>

#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"

namespace lss_ode_solvers
{

using lss_enumerations::factorization_enum;
using lss_enumerations::memory_space_enum;
using lss_enumerations::tridiagonal_method_enum;
using lss_utility::range;
using lss_utility::sptr_t;

/**
    ode_implicit_solver_config structure
 */
struct ode_implicit_solver_config
{
  private:
    memory_space_enum memory_space_;
    tridiagonal_method_enum tridiagonal_method_;
    factorization_enum tridiagonal_factorization_;

    explicit ode_implicit_solver_config() = delete;

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
    explicit ode_implicit_solver_config(memory_space_enum const &memory_space,
                                        tridiagonal_method_enum const &tridiagonal_method,
                                        factorization_enum const &tridiagonal_factorization)
        : memory_space_{memory_space}, tridiagonal_method_{tridiagonal_method}, tridiagonal_factorization_{
                                                                                    tridiagonal_factorization}
    {
        initialize();
    }
    ~ode_implicit_solver_config()
    {
    }

    inline memory_space_enum memory_space() const
    {
        return memory_space_;
    }

    inline tridiagonal_method_enum tridiagonal_method() const
    {
        return tridiagonal_method_;
    }

    inline factorization_enum tridiagonal_factorization() const
    {
        return tridiagonal_factorization_;
    }
};

using ode_implicit_solver_config_ptr = sptr_t<ode_implicit_solver_config>;

// =================================================
// ===== some default implicit solver configs ======
// =================================================
namespace default_ode_solver_configs
{
auto dev_cusolver_qr_solver_config_ptr = std::make_shared<ode_implicit_solver_config>(
    memory_space_enum::Device, tridiagonal_method_enum::CUDASolver, factorization_enum::QRMethod);

auto dev_cusolver_lu_solver_config_ptr = std::make_shared<ode_implicit_solver_config>(
    memory_space_enum::Device, tridiagonal_method_enum::CUDASolver, factorization_enum::LUMethod);

auto host_cusolver_qr_solver_config_ptr = std::make_shared<ode_implicit_solver_config>(
    memory_space_enum::Host, tridiagonal_method_enum::CUDASolver, factorization_enum::QRMethod);

auto host_cusolver_lu_solver_config_ptr = std::make_shared<ode_implicit_solver_config>(
    memory_space_enum::Host, tridiagonal_method_enum::CUDASolver, factorization_enum::LUMethod);

auto dev_sorsolver_solver_config_ptr = std::make_shared<ode_implicit_solver_config>(
    memory_space_enum::Device, tridiagonal_method_enum::SORSolver, factorization_enum::None);

auto host_sorsolver_solver_config_ptr = std::make_shared<ode_implicit_solver_config>(
    memory_space_enum::Host, tridiagonal_method_enum::SORSolver, factorization_enum::None);

auto host_dssolver_solver_config_ptr = std::make_shared<ode_implicit_solver_config>(
    memory_space_enum::Host, tridiagonal_method_enum::DoubleSweepSolver, factorization_enum::None);

auto host_tlusolver_solver_config_ptr = std::make_shared<ode_implicit_solver_config>(
    memory_space_enum::Host, tridiagonal_method_enum::ThomasLUSolver, factorization_enum::None);
} // namespace default_ode_solver_configs
} // namespace lss_ode_solvers

#endif ///_LSS_ODE_SOLVER_CONFIG_HPP_
