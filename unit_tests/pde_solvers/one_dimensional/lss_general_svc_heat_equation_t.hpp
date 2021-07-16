#if !defined(_LSS_GENERAL_SVC_HEAT_EQUATION_T_HPP_)
#define _LSS_GENERAL_SVC_HEAT_EQUATION_T_HPP_

#include "pde_solvers/one_dimensional/heat_type/lss_general_svc_heat_equation.hpp"
#include <map>

#define PI 3.14159

// ///////////////////////////////////////////////////////////////////////////
//							PURE HEAT PROBLEMS
// ///////////////////////////////////////////////////////////////////////////

// ===========================================================================
// ========================== IMPLICIT SOLVERS ===============================
// ===========================================================================

// ===========================================================================
// =========== Heat problem with homogeneous boundary conditions =============
// ===========================================================================

template <typename T> void testImplPureHeatEquationDirichletBCCUDASolverDeviceQREuler()
{
    using lss_boundary_1d::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_pde_solvers::dev_fwd_cusolver_qr_euler_solver_config_ptr;
    using lss_pde_solvers::discretization_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::implicit_solver_config;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Heat equation: \n\n";
    std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = x, x in <0,1> \n\n";
    std::cout << " Exact solution: \n";
    std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) "
                 "*sin(n*pi*x)/n}\n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_svc_heat_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(0.0, 1.0);
    // time range
    range<T> time_range(0.0, 0.1);
    // discretization config:
    auto const discretization_ptr = std::make_shared<discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [](T x) { return 1.0; };
    auto other = [](T x) { return 0.0; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, other, other);
    // initial condition:
    auto initial_condition = [](T x) { return x; };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(initial_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, dev_fwd_cusolver_qr_euler_solver_config_ptr);
    // prepare container for solution:
    // note: size is Sd+1 since we must include space point at x = 0
    std::vector<T> solution(Sd + 1, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        T const first = static_cast<T>(2.0) / PI;
        T sum{};
        T var1{};
        T var2{};
        for (std::size_t i = 1; i <= n; ++i)
        {
            var1 = std::pow(-1.0, i + 1) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
            var2 = std::sin(i * PI * x) / i;
            sum += (var1 * var2);
        }
        return (first * sum);
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

template <typename T> void testImplPureHeatEquationDirichletBCCUDASolverDeviceQRCrankNicolson()
{
    using lss_boundary_1d::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_pde_solvers::dev_fwd_cusolver_qr_cn_solver_config_ptr;
    using lss_pde_solvers::discretization_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::implicit_solver_config;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Heat equation: \n\n";
    std::cout << " Using CUDA solver algorithm with implicit CN method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = x, x in <0,1> \n\n";
    std::cout << " Exact solution: \n";
    std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) "
                 "*sin(n*pi*x)/n}\n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_svc_heat_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(0.0, 1.0);
    // time range
    range<T> time_range(0.0, 0.1);
    // discretization config:
    auto const discretization_ptr = std::make_shared<discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [](T x) { return 1.0; };
    auto other = [](T x) { return 0.0; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, other, other);
    // initial condition:
    auto initial_condition = [](T x) { return x; };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(initial_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, dev_fwd_cusolver_qr_cn_solver_config_ptr);
    // prepare container for solution:
    // note: size is Sd+1 since we must include space point at x = 0
    std::vector<T> solution(Sd + 1, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        T const first = static_cast<T>(2.0) / PI;
        T sum{};
        T var1{};
        T var2{};
        for (std::size_t i = 1; i <= n; ++i)
        {
            var1 = std::pow(-1.0, i + 1) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
            var2 = std::sin(i * PI * x) / i;
            sum += (var1 * var2);
        }
        return (first * sum);
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

void testImplPureHeatEquationDirichletBCCUDASolverDeviceQR()
{
    std::cout << "============================================================\n";
    std::cout << "======== Implicit Pure Heat Equation (Dirichlet BC) ========\n";
    std::cout << "============================================================\n";

    testImplPureHeatEquationDirichletBCCUDASolverDeviceQREuler<double>();
    testImplPureHeatEquationDirichletBCCUDASolverDeviceQREuler<float>();
    testImplPureHeatEquationDirichletBCCUDASolverDeviceQRCrankNicolson<double>();
    testImplPureHeatEquationDirichletBCCUDASolverDeviceQRCrankNicolson<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplPureHeatEquationDirichletBCSORSolverDeviceEuler()
{
    using lss_boundary_1d::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_pde_solvers::dev_fwd_sorsolver_euler_solver_config_ptr;
    using lss_pde_solvers::discretization_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::implicit_solver_config;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Heat equation: \n\n";
    std::cout << " Using CUDA SORsolver algorithm with implicit Euler method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = x, x in <0,1> \n\n";
    std::cout << " Exact solution: \n";
    std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) "
                 "*sin(n*pi*x)/n}\n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_svc_heat_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(0.0, 1.0);
    // time range
    range<T> time_range(0.0, 0.1);
    // discretization config:
    auto const discretization_ptr = std::make_shared<discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [](T x) { return 1.0; };
    auto other = [](T x) { return 0.0; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, other, other);
    // initial condition:
    auto initial_condition = [](T x) { return x; };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(initial_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // details:
    std::map<std::string, T> details;
    details["sor_omega"] = static_cast<T>(1.0);
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, dev_fwd_sorsolver_euler_solver_config_ptr,
                         details);
    // prepare container for solution:
    // note: size is Sd+1 since we must include space point at x = 0
    std::vector<T> solution(Sd + 1, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        T const first = static_cast<T>(2.0) / PI;
        T sum{};
        T var1{};
        T var2{};
        for (std::size_t i = 1; i <= n; ++i)
        {
            var1 = std::pow(-1.0, i + 1) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
            var2 = std::sin(i * PI * x) / i;
            sum += (var1 * var2);
        }
        return (first * sum);
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

template <typename T> void testImplPureHeatEquationDirichletBCSORSolverDeviceCrankNicolson()
{
    using lss_boundary_1d::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_pde_solvers::dev_fwd_sorsolver_cn_solver_config_ptr;
    using lss_pde_solvers::discretization_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::implicit_solver_config;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Heat equation: \n\n";
    std::cout << " Using CUDA SOR solver algorithm with implicit CN method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = x, x in <0,1> \n\n";
    std::cout << " Exact solution: \n";
    std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) "
                 "*sin(n*pi*x)/n}\n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_svc_heat_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(0.0, 1.0);
    // time range
    range<T> time_range(0.0, 0.1);
    // discretization config:
    auto const discretization_ptr = std::make_shared<discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [](T x) { return 1.0; };
    auto other = [](T x) { return 0.0; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, other, other);
    // initial condition:
    auto initial_condition = [](T x) { return x; };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(initial_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // details:
    std::map<std::string, T> details;
    details["sor_omega"] = static_cast<T>(1.0);
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, dev_fwd_sorsolver_cn_solver_config_ptr,
                         details);
    // prepare container for solution:
    // note: size is Sd+1 since we must include space point at x = 0
    std::vector<T> solution(Sd + 1, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        T const first = static_cast<T>(2.0) / PI;
        T sum{};
        T var1{};
        T var2{};
        for (std::size_t i = 1; i <= n; ++i)
        {
            var1 = std::pow(-1.0, i + 1) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
            var2 = std::sin(i * PI * x) / i;
            sum += (var1 * var2);
        }
        return (first * sum);
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

void testImplPureHeatEquationDirichletBCSORSolverDevice()
{
    std::cout << "============================================================\n";
    std::cout << "======== Implicit Pure Heat Equation (Dirichlet BC) ========\n";
    std::cout << "============================================================\n";

    testImplPureHeatEquationDirichletBCSORSolverDeviceEuler<double>();
    testImplPureHeatEquationDirichletBCSORSolverDeviceEuler<float>();
    testImplPureHeatEquationDirichletBCSORSolverDeviceCrankNicolson<double>();
    testImplPureHeatEquationDirichletBCSORSolverDeviceCrankNicolson<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplPureHeatEquationDirichletBCSORSolverHostEuler()
{
    using lss_boundary_1d::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_pde_solvers::discretization_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::host_fwd_sorsolver_euler_solver_config_ptr;
    using lss_pde_solvers::implicit_solver_config;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Heat equation: \n\n";
    std::cout << " Using CUDA SORsolver algorithm with implicit Euler method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = x, x in <0,1> \n\n";
    std::cout << " Exact solution: \n";
    std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) "
                 "*sin(n*pi*x)/n}\n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_svc_heat_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(0.0, 1.0);
    // time range
    range<T> time_range(0.0, 0.1);
    // discretization config:
    auto const discretization_ptr = std::make_shared<discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [](T x) { return 1.0; };
    auto other = [](T x) { return 0.0; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, other, other);
    // initial condition:
    auto initial_condition = [](T x) { return x; };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(initial_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // details:
    std::map<std::string, T> details;
    details["sor_omega"] = static_cast<T>(1.0);
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, host_fwd_sorsolver_euler_solver_config_ptr,
                         details);
    // prepare container for solution:
    // note: size is Sd+1 since we must include space point at x = 0
    std::vector<T> solution(Sd + 1, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        T const first = static_cast<T>(2.0) / PI;
        T sum{};
        T var1{};
        T var2{};
        for (std::size_t i = 1; i <= n; ++i)
        {
            var1 = std::pow(-1.0, i + 1) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
            var2 = std::sin(i * PI * x) / i;
            sum += (var1 * var2);
        }
        return (first * sum);
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

template <typename T> void testImplPureHeatEquationDirichletBCSORSolverHostCrankNicolson()
{
    using lss_boundary_1d::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_pde_solvers::discretization_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::host_fwd_sorsolver_cn_solver_config_ptr;
    using lss_pde_solvers::implicit_solver_config;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Heat equation: \n\n";
    std::cout << " Using CUDA SOR solver algorithm with implicit CN method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = x, x in <0,1> \n\n";
    std::cout << " Exact solution: \n";
    std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) "
                 "*sin(n*pi*x)/n}\n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_svc_heat_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(0.0, 1.0);
    // time range
    range<T> time_range(0.0, 0.1);
    // discretization config:
    auto const discretization_ptr = std::make_shared<discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [](T x) { return 1.0; };
    auto other = [](T x) { return 0.0; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, other, other);
    // initial condition:
    auto initial_condition = [](T x) { return x; };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(initial_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // details:
    std::map<std::string, T> details;
    details["sor_omega"] = static_cast<T>(1.0);
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, host_fwd_sorsolver_cn_solver_config_ptr,
                         details);
    // prepare container for solution:
    // note: size is Sd+1 since we must include space point at x = 0
    std::vector<T> solution(Sd + 1, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        T const first = static_cast<T>(2.0) / PI;
        T sum{};
        T var1{};
        T var2{};
        for (std::size_t i = 1; i <= n; ++i)
        {
            var1 = std::pow(-1.0, i + 1) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
            var2 = std::sin(i * PI * x) / i;
            sum += (var1 * var2);
        }
        return (first * sum);
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

void testImplPureHeatEquationDirichletBCSORSolverHost()
{
    std::cout << "============================================================\n";
    std::cout << "======== Implicit Pure Heat Equation (Dirichlet BC) ========\n";
    std::cout << "============================================================\n";

    testImplPureHeatEquationDirichletBCSORSolverHostEuler<double>();
    testImplPureHeatEquationDirichletBCSORSolverHostEuler<float>();
    testImplPureHeatEquationDirichletBCSORSolverHostCrankNicolson<double>();
    testImplPureHeatEquationDirichletBCSORSolverHostCrankNicolson<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplPureHeatEquationDirichletBCCUDASolverHostQREuler()
{
    using lss_boundary_1d::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_pde_solvers::discretization_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::host_fwd_cusolver_qr_euler_solver_config_ptr;
    using lss_pde_solvers::implicit_solver_config;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Heat equation: \n\n";
    std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = x, x in <0,1> \n\n";
    std::cout << " Exact solution: \n";
    std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) "
                 "*sin(n*pi*x)/n}\n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_svc_heat_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(0.0, 1.0);
    // time range
    range<T> time_range(0.0, 0.1);
    // discretization config:
    auto const discretization_ptr = std::make_shared<discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [](T x) { return 1.0; };
    auto other = [](T x) { return 0.0; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, other, other);
    // initial condition:
    auto initial_condition = [](T x) { return x; };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(initial_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair,
                         host_fwd_cusolver_qr_euler_solver_config_ptr);
    // prepare container for solution:
    // note: size is Sd+1 since we must include space point at x = 0
    std::vector<T> solution(Sd + 1, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        T const first = static_cast<T>(2.0) / PI;
        T sum{};
        T var1{};
        T var2{};
        for (std::size_t i = 1; i <= n; ++i)
        {
            var1 = std::pow(-1.0, i + 1) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
            var2 = std::sin(i * PI * x) / i;
            sum += (var1 * var2);
        }
        return (first * sum);
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

template <typename T> void testImplPureHeatEquationDirichletBCCUDASolverHostQRCrankNicolson()
{
    using lss_boundary_1d::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_pde_solvers::discretization_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::host_fwd_cusolver_qr_cn_solver_config_ptr;
    using lss_pde_solvers::implicit_solver_config;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Heat equation: \n\n";
    std::cout << " Using CUDA solver algorithm with implicit CN method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = x, x in <0,1> \n\n";
    std::cout << " Exact solution: \n";
    std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) "
                 "*sin(n*pi*x)/n}\n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_svc_heat_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(0.0, 1.0);
    // time range
    range<T> time_range(0.0, 0.1);
    // discretization config:
    auto const discretization_ptr = std::make_shared<discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [](T x) { return 1.0; };
    auto other = [](T x) { return 0.0; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, other, other);
    // initial condition:
    auto initial_condition = [](T x) { return x; };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(initial_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, host_fwd_cusolver_qr_cn_solver_config_ptr);
    // prepare container for solution:
    // note: size is Sd+1 since we must include space point at x = 0
    std::vector<T> solution(Sd + 1, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        T const first = static_cast<T>(2.0) / PI;
        T sum{};
        T var1{};
        T var2{};
        for (std::size_t i = 1; i <= n; ++i)
        {
            var1 = std::pow(-1.0, i + 1) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
            var2 = std::sin(i * PI * x) / i;
            sum += (var1 * var2);
        }
        return (first * sum);
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

void testImplPureHeatEquationDirichletBCCUDASolverHostQR()
{
    std::cout << "============================================================\n";
    std::cout << "======== Implicit Pure Heat Equation (Dirichlet BC) ========\n";
    std::cout << "============================================================\n";

    testImplPureHeatEquationDirichletBCCUDASolverHostQREuler<double>();
    testImplPureHeatEquationDirichletBCCUDASolverHostQREuler<float>();
    testImplPureHeatEquationDirichletBCCUDASolverHostQRCrankNicolson<double>();
    testImplPureHeatEquationDirichletBCCUDASolverHostQRCrankNicolson<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplPureHeatEquationDirichletBCDoubleSweepSolverEuler()
{
    using lss_boundary_1d::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_pde_solvers::discretization_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::host_fwd_dssolver_euler_solver_config_ptr;
    using lss_pde_solvers::implicit_solver_config;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Heat equation: \n\n";
    std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = x, x in <0,1> \n\n";
    std::cout << " Exact solution: \n";
    std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) "
                 "*sin(n*pi*x)/n}\n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_svc_heat_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(0.0, 1.0);
    // time range
    range<T> time_range(0.0, 0.1);
    // discretization config:
    auto const discretization_ptr = std::make_shared<discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [](T x) { return 1.0; };
    auto other = [](T x) { return 0.0; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, other, other);
    // initial condition:
    auto initial_condition = [](T x) { return x; };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(initial_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, host_fwd_dssolver_euler_solver_config_ptr);
    // prepare container for solution:
    // note: size is Sd+1 since we must include space point at x = 0
    std::vector<T> solution(Sd + 1, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        T const first = static_cast<T>(2.0) / PI;
        T sum{};
        T var1{};
        T var2{};
        for (std::size_t i = 1; i <= n; ++i)
        {
            var1 = std::pow(-1.0, i + 1) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
            var2 = std::sin(i * PI * x) / i;
            sum += (var1 * var2);
        }
        return (first * sum);
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

template <typename T> void testImplPureHeatEquationDirichletBCDoubleSweepSolverCrankNicolson()
{
    using lss_boundary_1d::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_pde_solvers::discretization_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::host_fwd_dssolver_cn_solver_config_ptr;
    using lss_pde_solvers::implicit_solver_config;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Heat equation: \n\n";
    std::cout << " Using CUDA solver algorithm with implicit CN method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = x, x in <0,1> \n\n";
    std::cout << " Exact solution: \n";
    std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) "
                 "*sin(n*pi*x)/n}\n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_svc_heat_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(0.0, 1.0);
    // time range
    range<T> time_range(0.0, 0.1);
    // discretization config:
    auto const discretization_ptr = std::make_shared<discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [](T x) { return 1.0; };
    auto other = [](T x) { return 0.0; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, other, other);
    // initial condition:
    auto initial_condition = [](T x) { return x; };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(initial_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, host_fwd_dssolver_cn_solver_config_ptr);
    // prepare container for solution:
    // note: size is Sd+1 since we must include space point at x = 0
    std::vector<T> solution(Sd + 1, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        T const first = static_cast<T>(2.0) / PI;
        T sum{};
        T var1{};
        T var2{};
        for (std::size_t i = 1; i <= n; ++i)
        {
            var1 = std::pow(-1.0, i + 1) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
            var2 = std::sin(i * PI * x) / i;
            sum += (var1 * var2);
        }
        return (first * sum);
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

void testImplPureHeatEquationDirichletBCDoubleSweepSolver()
{
    std::cout << "============================================================\n";
    std::cout << "======== Implicit Pure Heat Equation (Dirichlet BC) ========\n";
    std::cout << "============================================================\n";

    testImplPureHeatEquationDirichletBCDoubleSweepSolverEuler<double>();
    testImplPureHeatEquationDirichletBCDoubleSweepSolverEuler<float>();
    testImplPureHeatEquationDirichletBCDoubleSweepSolverCrankNicolson<double>();
    testImplPureHeatEquationDirichletBCDoubleSweepSolverCrankNicolson<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplPureHeatEquationDirichletBCThomasLUSolverEuler()
{
    using lss_boundary_1d::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_pde_solvers::discretization_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::host_fwd_tlusolver_euler_solver_config_ptr;
    using lss_pde_solvers::implicit_solver_config;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Heat equation: \n\n";
    std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = x, x in <0,1> \n\n";
    std::cout << " Exact solution: \n";
    std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) "
                 "*sin(n*pi*x)/n}\n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_svc_heat_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(0.0, 1.0);
    // time range
    range<T> time_range(0.0, 0.1);
    // discretization config:
    auto const discretization_ptr = std::make_shared<discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [](T x) { return 1.0; };
    auto other = [](T x) { return 0.0; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, other, other);
    // initial condition:
    auto initial_condition = [](T x) { return x; };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(initial_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, host_fwd_tlusolver_euler_solver_config_ptr);
    // prepare container for solution:
    // note: size is Sd+1 since we must include space point at x = 0
    std::vector<T> solution(Sd + 1, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        T const first = static_cast<T>(2.0) / PI;
        T sum{};
        T var1{};
        T var2{};
        for (std::size_t i = 1; i <= n; ++i)
        {
            var1 = std::pow(-1.0, i + 1) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
            var2 = std::sin(i * PI * x) / i;
            sum += (var1 * var2);
        }
        return (first * sum);
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

template <typename T> void testImplPureHeatEquationDirichletBCThomasLUSolverCrankNicolson()
{
    using lss_boundary_1d::dirichlet_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_pde_solvers::discretization_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::host_fwd_tlusolver_cn_solver_config_ptr;
    using lss_pde_solvers::implicit_solver_config;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Heat equation: \n\n";
    std::cout << " Using CUDA solver algorithm with implicit CN method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = x, x in <0,1> \n\n";
    std::cout << " Exact solution: \n";
    std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) "
                 "*sin(n*pi*x)/n}\n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_svc_heat_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(0.0, 1.0);
    // time range
    range<T> time_range(0.0, 0.1);
    // discretization config:
    auto const discretization_ptr = std::make_shared<discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [](T x) { return 1.0; };
    auto other = [](T x) { return 0.0; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, other, other);
    // initial condition:
    auto initial_condition = [](T x) { return x; };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(initial_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, host_fwd_tlusolver_cn_solver_config_ptr);
    // prepare container for solution:
    // note: size is Sd+1 since we must include space point at x = 0
    std::vector<T> solution(Sd + 1, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        T const first = static_cast<T>(2.0) / PI;
        T sum{};
        T var1{};
        T var2{};
        for (std::size_t i = 1; i <= n; ++i)
        {
            var1 = std::pow(-1.0, i + 1) * std::exp(-1.0 * (i * PI) * (i * PI) * t);
            var2 = std::sin(i * PI * x) / i;
            sum += (var1 * var2);
        }
        return (first * sum);
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

void testImplPureHeatEquationDirichletBCThomasLUSolver()
{
    std::cout << "============================================================\n";
    std::cout << "======== Implicit Pure Heat Equation (Dirichlet BC) ========\n";
    std::cout << "============================================================\n";

    testImplPureHeatEquationDirichletBCThomasLUSolverEuler<double>();
    testImplPureHeatEquationDirichletBCThomasLUSolverEuler<float>();
    testImplPureHeatEquationDirichletBCThomasLUSolverCrankNicolson<double>();
    testImplPureHeatEquationDirichletBCThomasLUSolverCrankNicolson<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplPureHeatEquationNeumannBCCUDASolverDeviceQREuler()
{
    using lss_boundary_1d::dirichlet_boundary_1d;
    using lss_boundary_1d::neumann_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_pde_solvers::dev_fwd_cusolver_qr_euler_solver_config_ptr;
    using lss_pde_solvers::discretization_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::implicit_solver_config;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Heat equation: \n\n";
    std::cout << " Using CUDA Solver algorithm with implicit Euler method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U_x(0,t) = 0 ,U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = cos((PI/2)*x), x in <0,1> \n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_svc_heat_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(0.0, 1.0);
    // time range
    range<T> time_range(0.0, 0.1);
    // discretization config:
    auto const discretization_ptr = std::make_shared<discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [](T x) { return 1.0; };
    auto other = [](T x) { return 0.0; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, other, other);
    // initial condition:
    auto initial_condition = [](T x) { return std::cos(x * PI / static_cast<T>(2.0)); };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(initial_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &neumann = [](T t) { return 0.0; };
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_left_ptr = std::make_shared<neumann_boundary_1d<T>>(neumann);
    auto const &boundary_right_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_left_ptr, boundary_right_ptr);
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, dev_fwd_cusolver_qr_euler_solver_config_ptr);
    // prepare container for solution:
    // note: size is Sd+1 since we must include space point at x = 0
    std::vector<T> solution(Sd + 1, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    // get exact solution:
    auto exact = [](T x, T t) {
        T const pipi = PI * PI;
        T const expon = (-pipi / static_cast<T>(4.0));
        T res = std::exp(expon * t) * std::cos(x * PI / static_cast<T>(2.0));
        return res;
    };

    T const h = discretization_ptr->space_step();
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        benchmark = exact(j * h, time_range.upper());
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

template <typename T> void testImplPureHeatEquationNeumannBCCUDASolverDeviceQRCrankNicolson()
{
    using lss_boundary_1d::dirichlet_boundary_1d;
    using lss_boundary_1d::neumann_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_pde_solvers::dev_fwd_cusolver_qr_cn_solver_config_ptr;
    using lss_pde_solvers::discretization_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::implicit_solver_config;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Heat equation: \n\n";
    std::cout << " Using CUDA solver algorithm with implicit CN method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U_x(0,t) = 0 ,U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = cos((PI/2)*x), x in <0,1> \n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_svc_heat_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(0.0, 1.0);
    // time range
    range<T> time_range(0.0, 0.1);
    // discretization config:
    auto const discretization_ptr = std::make_shared<discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [](T x) { return 1.0; };
    auto other = [](T x) { return 0.0; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, other, other);
    // initial condition:
    auto initial_condition = [](T x) { return std::cos(x * PI / static_cast<T>(2.0)); };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(initial_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &neumann = [](T t) { return 0.0; };
    auto const &dirichlet = [](T t) { return 0.0; };
    auto const &boundary_left_ptr = std::make_shared<neumann_boundary_1d<T>>(neumann);
    auto const &boundary_right_ptr = std::make_shared<dirichlet_boundary_1d<T>>(dirichlet);
    auto const &boundary_pair = std::make_pair(boundary_left_ptr, boundary_right_ptr);
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, dev_fwd_cusolver_qr_cn_solver_config_ptr);
    // prepare container for solution:
    // note: size is Sd+1 since we must include space point at x = 0
    std::vector<T> solution(Sd + 1, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    // get exact solution:
    auto exact = [](T x, T t) {
        T const pipi = PI * PI;
        T const expon = (-pipi / static_cast<T>(4.0));
        T res = std::exp(expon * t) * std::cos(x * PI / static_cast<T>(2.0));
        return res;
    };

    T const h = discretization_ptr->space_step();
    std::cout << "tp : FDM | Exact | Abs Diff\n";
    T benchmark{};
    for (std::size_t j = 0; j < solution.size(); ++j)
    {
        benchmark = exact(j * h, time_range.upper());
        std::cout << "t_" << j << ": " << solution[j] << " |  " << benchmark << " | " << (solution[j] - benchmark)
                  << '\n';
    }
}

void testImplPureHeatEquationNeumannBCCUDASolverDeviceQR()
{
    std::cout << "============================================================\n";
    std::cout << "======== Implicit Pure Heat Equation (Dirichlet BC) ========\n";
    std::cout << "============================================================\n";

    testImplPureHeatEquationNeumannBCCUDASolverDeviceQREuler<double>();
    testImplPureHeatEquationNeumannBCCUDASolverDeviceQREuler<float>();
    testImplPureHeatEquationNeumannBCCUDASolverDeviceQRCrankNicolson<double>();
    testImplPureHeatEquationNeumannBCCUDASolverDeviceQRCrankNicolson<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplPureHeatEquationNeumannBCThomasLUSolverEuler()
{
    using lss_boundary_1d::neumann_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_pde_solvers::discretization_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::host_fwd_tlusolver_euler_solver_config_ptr;
    using lss_pde_solvers::implicit_solver_config;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Heat equation: \n\n";
    std::cout << " Using Thomas LU solver with implicit Euler method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = x, x in <0,1> \n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_svc_heat_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(0.0, 1.0);
    // time range
    range<T> time_range(0.0, 0.1);
    // discretization config:
    auto const discretization_ptr = std::make_shared<discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [](T x) { return 1.0; };
    auto other = [](T x) { return 0.0; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, other, other);
    // initial condition:
    auto initial_condition = [](T x) { return x; };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(initial_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &neumann = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<neumann_boundary_1d<T>>(neumann);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, host_fwd_tlusolver_euler_solver_config_ptr);
    // prepare container for solution:
    // note: size is Sd+1 since we must include space point at x = 0
    std::vector<T> solution(Sd + 1, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        T const pipi = PI * PI;
        T const first = static_cast<T>(4.0) / pipi;
        T sum{};
        T var0{};
        T var1{};
        T var2{};
        for (std::size_t i = 1; i <= n; ++i)
        {
            var0 = static_cast<T>(2 * i - 1);
            var1 = std::exp(static_cast<T>(-1.0) * pipi * var0 * var0 * t);
            var2 = std::cos(var0 * PI * x) / (var0 * var0);
            sum += (var1 * var2);
        }
        return (static_cast<T>(0.5) - first * sum);
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

template <typename T> void testImplPureHeatEquationNeumannBCThomasLUSolverCrankNicolson()
{
    using lss_boundary_1d::neumann_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_pde_solvers::discretization_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::host_fwd_tlusolver_cn_solver_config_ptr;
    using lss_pde_solvers::implicit_solver_config;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Heat equation: \n\n";
    std::cout << " Using Thomas LU solver algorithm with implicit CN method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = x, x in <0,1> \n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_svc_heat_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 100;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(0.0, 1.0);
    // time range
    range<T> time_range(0.0, 0.1);
    // discretization config:
    auto const discretization_ptr = std::make_shared<discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [](T x) { return 1.0; };
    auto other = [](T x) { return 0.0; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, other, other);
    // initial condition:
    auto initial_condition = [](T x) { return x; };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(initial_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &neumann = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<neumann_boundary_1d<T>>(neumann);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, host_fwd_tlusolver_cn_solver_config_ptr);
    // prepare container for solution:
    // note: size is Sd+1 since we must include space point at x = 0
    std::vector<T> solution(Sd + 1, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        T const pipi = PI * PI;
        T const first = static_cast<T>(4.0) / pipi;
        T sum{};
        T var0{};
        T var1{};
        T var2{};
        for (std::size_t i = 1; i <= n; ++i)
        {
            var0 = static_cast<T>(2 * i - 1);
            var1 = std::exp(static_cast<T>(-1.0) * pipi * var0 * var0 * t);
            var2 = std::cos(var0 * PI * x) / (var0 * var0);
            sum += (var1 * var2);
        }
        return (static_cast<T>(0.5) - first * sum);
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

void testImplPureHeatEquationNeumannBCThomasLUSolver()
{
    std::cout << "============================================================\n";
    std::cout << "======== Implicit Pure Heat Equation (Neumann BC) ========\n";
    std::cout << "============================================================\n";

    testImplPureHeatEquationNeumannBCThomasLUSolverEuler<double>();
    testImplPureHeatEquationNeumannBCThomasLUSolverEuler<float>();
    testImplPureHeatEquationNeumannBCThomasLUSolverCrankNicolson<double>();
    testImplPureHeatEquationNeumannBCThomasLUSolverCrankNicolson<float>();

    std::cout << "============================================================\n";
}

template <typename T> void testImplPureHeatEquationNeumannBCDoubleSweepSolverEuler()
{
    using lss_boundary_1d::neumann_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_pde_solvers::discretization_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::host_fwd_dssolver_euler_solver_config_ptr;
    using lss_pde_solvers::implicit_solver_config;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Heat equation: \n\n";
    std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = x, x in <0,1> \n\n";
    std::cout << " Exact solution: \n";
    std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) "
                 "*sin(n*pi*x)/n}\n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_svc_heat_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 1000;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(0.0, 1.0);
    // time range
    range<T> time_range(0.0, 0.1);
    // discretization config:
    auto const discretization_ptr = std::make_shared<discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [](T x) { return 1.0; };
    auto other = [](T x) { return 0.0; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, other, other);
    // initial condition:
    auto initial_condition = [](T x) { return x; };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(initial_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &neumann = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<neumann_boundary_1d<T>>(neumann);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, host_fwd_dssolver_euler_solver_config_ptr);
    // prepare container for solution:
    // note: size is Sd+1 since we must include space point at x = 0
    std::vector<T> solution(Sd + 1, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        T const pipi = PI * PI;
        T const first = static_cast<T>(4.0) / pipi;
        T sum{};
        T var0{};
        T var1{};
        T var2{};
        for (std::size_t i = 1; i <= n; ++i)
        {
            var0 = static_cast<T>(2 * i - 1);
            var1 = std::exp(static_cast<T>(-1.0) * pipi * var0 * var0 * t);
            var2 = std::cos(var0 * PI * x) / (var0 * var0);
            sum += (var1 * var2);
        }
        return (static_cast<T>(0.5) - first * sum);
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

template <typename T> void testImplPureHeatEquationNeumannBCDoubleSweepSolverCrankNicolson()
{
    using lss_boundary_1d::neumann_boundary_1d;
    using lss_enumerations::implicit_pde_schemes_enum;
    using lss_pde_solvers::discretization_config_1d;
    using lss_pde_solvers::heat_coefficient_data_config_1d;
    using lss_pde_solvers::heat_data_config_1d;
    using lss_pde_solvers::heat_initial_data_config_1d;
    using lss_pde_solvers::host_fwd_dssolver_cn_solver_config_ptr;
    using lss_pde_solvers::implicit_solver_config;
    using lss_pde_solvers::one_dimensional::implicit_solvers::general_svc_heat_equation;
    using lss_utility::range;

    std::cout << "============================================================\n";
    std::cout << "Solving Boundary-value Heat equation: \n\n";
    std::cout << " Using CUDA solver algorithm with implicit CN method\n\n";
    std::cout << " Value type: " << typeid(T).name() << "\n\n";
    std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
    std::cout << " where\n\n";
    std::cout << " x in <0,1> and t > 0,\n";
    std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
    std::cout << " U(x,0) = x, x in <0,1> \n\n";
    std::cout << " Exact solution: \n";
    std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) "
                 "*sin(n*pi*x)/n}\n\n";
    std::cout << "============================================================\n";

    // typedef the Implicit1DHeatEquation
    typedef general_svc_heat_equation<T, std::vector, std::allocator<T>> pde_solver;

    // number of space subdivisions:
    std::size_t const Sd = 1000;
    // number of time subdivisions:
    std::size_t const Td = 100;
    // space range:
    range<T> space_range(0.0, 1.0);
    // time range
    range<T> time_range(0.0, 0.1);
    // discretization config:
    auto const discretization_ptr = std::make_shared<discretization_config_1d<T>>(space_range, Sd, time_range, Td);
    // coeffs:
    auto a = [](T x) { return 1.0; };
    auto other = [](T x) { return 0.0; };
    auto const heat_coeffs_data_ptr = std::make_shared<heat_coefficient_data_config_1d<T>>(a, other, other);
    // initial condition:
    auto initial_condition = [](T x) { return x; };
    auto const heat_init_data_ptr = std::make_shared<heat_initial_data_config_1d<T>>(initial_condition);
    // heat data config:
    auto const heat_data_ptr = std::make_shared<heat_data_config_1d<T>>(heat_coeffs_data_ptr, heat_init_data_ptr);
    // boundary conditions:
    auto const &neumann = [](T t) { return 0.0; };
    auto const &boundary_ptr = std::make_shared<neumann_boundary_1d<T>>(neumann);
    auto const &boundary_pair = std::make_pair(boundary_ptr, boundary_ptr);
    // initialize pde solver
    pde_solver pdesolver(heat_data_ptr, discretization_ptr, boundary_pair, host_fwd_dssolver_cn_solver_config_ptr);
    // prepare container for solution:
    // note: size is Sd+1 since we must include space point at x = 0
    std::vector<T> solution(Sd + 1, T{});
    // get the solution:
    pdesolver.solve(solution);
    // get exact solution:
    auto exact = [](T x, T t, std::size_t n) {
        T const pipi = PI * PI;
        T const first = static_cast<T>(4.0) / pipi;
        T sum{};
        T var0{};
        T var1{};
        T var2{};
        for (std::size_t i = 1; i <= n; ++i)
        {
            var0 = static_cast<T>(2 * i - 1);
            var1 = std::exp(static_cast<T>(-1.0) * pipi * var0 * var0 * t);
            var2 = std::cos(var0 * PI * x) / (var0 * var0);
            sum += (var1 * var2);
        }
        return (static_cast<T>(0.5) - first * sum);
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

void testImplPureHeatEquationNeumannBCDoubleSweepSolver()
{
    std::cout << "============================================================\n";
    std::cout << "======== Implicit Pure Heat Equation (Dirichlet BC) ========\n";
    std::cout << "============================================================\n";

    testImplPureHeatEquationNeumannBCDoubleSweepSolverEuler<double>();
    testImplPureHeatEquationNeumannBCDoubleSweepSolverEuler<float>();
    testImplPureHeatEquationNeumannBCDoubleSweepSolverCrankNicolson<double>();
    testImplPureHeatEquationNeumannBCDoubleSweepSolverCrankNicolson<float>();

    std::cout << "============================================================\n";
}

#endif //_LSS_GENERAL_SVC_HEAT_EQUATION_T_HPP_
