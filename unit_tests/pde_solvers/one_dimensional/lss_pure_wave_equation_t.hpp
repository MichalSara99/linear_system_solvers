#if !defined(_LSS_PURE_WAVE_EQUATION_T_HPP_)
#define _LSS_PURE_WAVE_EQUATION_T_HPP_

#include "pde_solvers/one_dimensional/wave_type/lss_1d_general_svc_wave_equation.hpp"
#include <map>

#define PI 3.14159265359

// ///////////////////////////////////////////////////////////////////////////
//							PURE WAVE PROBLEMS
// ///////////////////////////////////////////////////////////////////////////

// ===========================================================================
// ========================== IMPLICIT SOLVERS ===============================
// ===========================================================================

// ===========================================================================
// =========== Wave problem with homogeneous boundary conditions =============
// ===========================================================================

// Dirichlet boundaries:

template <typename T> void testImplPureWaveEquationDirichletBCCUDASolverDeviceQRDetail()
{
    using lss_boundary::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_pde_solvers::pde_discretization_config_1d;
    using lss_pde_solvers::wave_coefficient_data_config_1d;
    using lss_pde_solvers::wave_data_config_1d;
    using lss_pde_solvers::wave_implicit_solver_config;
    using lss_pde_solvers::wave_initial_data_config_1d;
    using lss_pde_solvers::default_wave_solver_configs::dev_fwd_cusolver_qr_solver_config_ptr;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_wave_equation;
    using lss_utility::pi;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Wave equation: \n\n";
    std::cout << " Using CUDA solver with QR (DEVICE) method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_tt(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = sin(pi*x), x in <0,1> \n\n";
    std::cout << " U_x(x,0) = 0, x in <0,1> \n\n";
    std::cout << "============================================================\n";

    // typedef the general_svc_wave_equation
    typedef general_svc_wave_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(static_cast<T>(0.0), static_cast<T>(1.0));
    // time range
    range<T> time_range(static_cast<T>(0.0), static_cast<T>(0.8));
    // discretization config:
    auto const discretization_ptr = std::make_shared<pde_discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto b = [](T x) { return 1.0; };
    auto other = [](T x) { return 0.0; };
    auto const wave_coeffs_data_ptr = std::make_shared<wave_coefficient_data_config_1d<T>>(other, b, other, other);
    // initial condition:
    auto initial_condition = [](T x) { return std::sin(pi<T>() * x); };
    auto const wave_init_data_ptr = std::make_shared<wave_initial_data_config_1d<T>>(initial_condition, other);
    // wave data config:
    auto const wave_data_ptr = std::make_shared<wave_data_config_1d<T>>(wave_coeffs_data_ptr, wave_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // initialize pde solver
    pde_solver pdesolver(wave_data_ptr, discretization_ptr, boundary_pair, dev_fwd_cusolver_qr_solver_config_ptr);
    // prepare container for solution:
    std::vector<T> solution(Sd, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        const T var1 = std::sin(pi<T>() * x);
        const T var2 = std::cos(pi<T>() * t);
        return (var1 * var2);
    };

    T const h = discretization_ptr->space_step();
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        benchmark = exact(j * h, time_range.upper(), 20);
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

void testImplPureWaveEquationDirichletBCCUDASolverDeviceQR()
{
    std::cout << "============================================================\n";
    std::cout << " Implicit Pure Wave (CUDA QR DEVICE) Equation (Dirichlet BC)\n";
    std::cout << "============================================================\n";

    testImplPureWaveEquationDirichletBCCUDASolverDeviceQRDetail<double>();
    testImplPureWaveEquationDirichletBCCUDASolverDeviceQRDetail<float>();

    std::cout << "============================================================\n";
}

#endif //_LSS_PURE_WAVE_EQUATION_T_HPP_
