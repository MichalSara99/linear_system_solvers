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

/**
     pde_implicit_solver_config base class
 */
struct pde_implicit_solver_config
{
  private:
    memory_space_enum memory_space_;
    traverse_direction_enum traverse_direction_;
    tridiagonal_method_enum tridiagonal_method_;
    factorization_enum tridiagonal_factorization_;

    explicit pde_implicit_solver_config() = delete;

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
                       "No support for Cholesky Method factorization for Double Sweep "
                       "Solver");
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
    explicit pde_implicit_solver_config(memory_space_enum const &memory_space,
                                        traverse_direction_enum const &traverse_direction,
                                        tridiagonal_method_enum const &tridiagonal_method,
                                        factorization_enum const &tridiagonal_factorization)
        : memory_space_{memory_space}, traverse_direction_{traverse_direction}, tridiagonal_method_{tridiagonal_method},
          tridiagonal_factorization_{tridiagonal_factorization}
    {
        initialize();
    }
    virtual ~pde_implicit_solver_config()
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
};

/**
     pde_explicit_solver_config base class
 */
struct pde_explicit_solver_config
{
  private:
    memory_space_enum memory_space_;
    traverse_direction_enum traverse_direction_;

    explicit pde_explicit_solver_config() = delete;

  public:
    explicit pde_explicit_solver_config(memory_space_enum const &memory_space,
                                        traverse_direction_enum const &traverse_direction)
        : memory_space_{memory_space}, traverse_direction_{traverse_direction}
    {
    }
    ~pde_explicit_solver_config()
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
};

using pde_implicit_solver_config_ptr = sptr_t<pde_implicit_solver_config>;

using pde_explicit_solver_config_ptr = sptr_t<pde_explicit_solver_config>;

} // namespace lss_pde_solvers

#endif ///_LSS_SOLVER_CONFIG_HPP_
