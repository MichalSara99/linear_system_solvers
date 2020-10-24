#pragma once
#if !defined(_LSS_ONE_DIM_HEAT_EQUATION_SOLVERS_T)
#define _LSS_ONE_DIM_HEAT_EQUATION_SOLVERS_T

#include"lss_types.h"
#include"lss_utility.h"
#include"lss_one_dim_heat_equation_solvers.h"


#define PI 3.14159

// ================================================================================================================
// =========================================== IMPLICIT SOLVERS ===================================================
// ================================================================================================================


// ================================================================================================================
// ==================================== Heat problem with homogeneous boundary conditions =========================
// ================================================================================================================

template<typename T>
void testImplHeatEquationDirichletBCDoubleSweepEuler() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_fdm_double_sweep_solver::FDMDoubleSweepSolver;
	using lss_one_dim_heat_equation_solvers::implicit_solvers::Implicit1DHeatEquation;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation: \n\n";
	std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
	std::cout << " Value type: " << typeid(T).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Implicit1DHeatEquation<T, 
				BoundaryConditionType::Dirichlet,
				FDMDoubleSweepSolver, 
				std::vector,
				std::allocator<T>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 100;
	// initial condition:
	auto initialCondition = [](T x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 0.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<T> solution(Sd + 1, T{});
	// initialize solver
	implicit_solver impl_solver(Range<T>(0.0,1.0),0.5,Sd,Td);
	// set boundary conditions:
	impl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	impl_solver.solve(solution,ImplicitPDESchemes::Euler);
	// get exact solution:
	auto exact = [](T x, T t, std::size_t n) {
		T const first = 2.0 / PI;
		T sum{};
		T var1{};
		T var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var1 = std::pow(-1.0, i + 1) * std::exp(-1.0*(i*PI)*(i*PI)*t);
			var2 = std::sin(i*PI*x) / i;
			sum += (var1*var2);
		}
		return (first * sum);
	};

	T const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	T benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.5, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}


template<typename T>
void testImplHeatEquationDirichletBCDoubleSweepCN() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_fdm_double_sweep_solver::FDMDoubleSweepSolver;
	using lss_one_dim_heat_equation_solvers::implicit_solvers::Implicit1DHeatEquation;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation: \n\n";
	std::cout << " Using Double Sweep algorithm with implicit Crank-Nicolson method\n\n";
	std::cout << " Value type: " << typeid(T).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Implicit1DHeatEquation<T,
		BoundaryConditionType::Dirichlet,
		FDMDoubleSweepSolver,
		std::vector,
		std::allocator<T>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 100;
	// initial condition:
	auto initialCondition = [](T x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 0.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<T> solution(Sd + 1, T{});
	// initialize solver
	implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.20, Sd, Td);
	// set boundary conditions:
	impl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	impl_solver.solve(solution);
	// get exact solution:
	auto exact = [](T x, T t, std::size_t n) {
		T const first = 2.0 / PI;
		T sum{};
		T var1{};
		T var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var1 = std::pow(-1.0, i + 1) * std::exp(-1.0*(i*PI)*(i*PI)*t);
			var2 = std::sin(i*PI*x) / i;
			sum += (var1*var2);
		}
		return (first * sum);
	};

	T const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	T benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.20, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}

void testImplHeatEquationDirichletBCDoubleSweep() {
	std::cout << "================================================================================\n";
	std::cout << "===================== Implicit Heat Equation (Dirichlet BC) ====================\n";
	std::cout << "================================================================================\n";

	testImplHeatEquationDirichletBCDoubleSweepEuler<double>();
	testImplHeatEquationDirichletBCDoubleSweepEuler<float>();
	testImplHeatEquationDirichletBCDoubleSweepCN<double>();
	testImplHeatEquationDirichletBCDoubleSweepCN<float>();

	std::cout << "================================================================================\n";
}


template<typename T>
void testImplHeatEquationDirichletBCThomasLUEuler() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_fdm_thomas_lu_solver::FDMThomasLUSolver;
	using lss_one_dim_heat_equation_solvers::implicit_solvers::Implicit1DHeatEquation;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation: \n\n";
	std::cout << " Using Thomas LU algorithm with implicit Euler method\n\n";
	std::cout << " Value type: " << typeid(T).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Implicit1DHeatEquation<T,
		BoundaryConditionType::Dirichlet,
		FDMThomasLUSolver,
		std::vector,
		std::allocator<T>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 100;
	// initial condition:
	auto initialCondition = [](T x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 0.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<T> solution(Sd + 1, T{});
	// initialize solver
	implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.5, Sd, Td);
	// set boundary conditions:
	impl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	impl_solver.solve(solution, ImplicitPDESchemes::Euler);
	// get exact solution:
	auto exact = [](T x, T t, std::size_t n) {
		T const first = 2.0 / PI;
		T sum{};
		T var1{};
		T var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var1 = std::pow(-1.0, i + 1) * std::exp(-1.0*(i*PI)*(i*PI)*t);
			var2 = std::sin(i*PI*x) / i;
			sum += (var1*var2);
		}
		return (first * sum);
	};

	T const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	T benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.5, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}


template<typename T>
void testImplHeatEquationDirichletBCThomasLUCN() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_fdm_thomas_lu_solver::FDMThomasLUSolver;
	using lss_one_dim_heat_equation_solvers::implicit_solvers::Implicit1DHeatEquation;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation: \n\n";
	std::cout << " Using Thomas LU algorithm with implicit Crank-Nicolson method\n\n";
	std::cout << " Value type: " << typeid(T).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Implicit1DHeatEquation<T,
		BoundaryConditionType::Dirichlet,
		FDMThomasLUSolver,
		std::vector,
		std::allocator<T>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 1000;
	// number of time subdivisions:
	std::size_t const Td = 1000;
	// initial condition:
	auto initialCondition = [](T x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 0.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<T> solution(Sd + 1, T{});
	// initialize solver
	implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.20, Sd, Td);
	// set boundary conditions:
	impl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	impl_solver.solve(solution);
	// get exact solution:
	auto exact = [](T x, T t, std::size_t n) {
		T const first = 2.0 / PI;
		T sum{};
		T var1{};
		T var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var1 = std::pow(-1.0, i + 1) * std::exp(-1.0*(i*PI)*(i*PI)*t);
			var2 = std::sin(i*PI*x) / i;
			sum += (var1*var2);
		}
		return (first * sum);
	};

	T const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	T benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.20, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}

void testImplHeatEquationDirichletBCThomasLU() {
	std::cout << "================================================================================\n";
	std::cout << "===================== Implicit Heat Equation (Dirichlet BC) ====================\n";
	std::cout << "================================================================================\n";

	testImplHeatEquationDirichletBCThomasLUEuler<double>();
	testImplHeatEquationDirichletBCThomasLUEuler<float>();
	testImplHeatEquationDirichletBCThomasLUCN<double>();
	testImplHeatEquationDirichletBCThomasLUCN<float>();

	std::cout << "================================================================================\n";
}


template<typename T>
void testImplHeatEquationRobinBCDoubleSweepEuler() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_fdm_double_sweep_solver::FDMDoubleSweepSolver;
	using lss_one_dim_heat_equation_solvers::implicit_solvers::Implicit1DHeatEquation;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation: \n\n";
	std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
	std::cout << " Value type: " << typeid(T).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = 0.5 - (4/(pi*pi))*sum_1^infty{ exp(-((2n-1)*pi)^2*t) *cos((2n-1)*pi*x)/(2n-1)^2}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Implicit1DHeatEquation<T,
		BoundaryConditionType::Robin,
		FDMDoubleSweepSolver,
		std::vector,
		std::allocator<T>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 400;
	// number of time subdivisions:
	std::size_t const Td = 150;
	// initial condition:
	auto initialCondition = [](T x) {return x; };
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<T> solution(Sd + 1, T{});
	// initialize solver
	implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.5, Sd, Td);
	// boundary conditions:
	// Robin boundaries are assumed to be of following form:
	//
	//				U_0 = leftLin * U_1 + leftConst
	//				U_{N-1} = rightLin * U_N + rightConst
	//
	// In our case discretizing the boundaries gives:
	// 
	//				(U_1 - U_-1)/2h = 0
	//				(U_N+1 - U_{N-1})/2h = 0
	//
	// Therefore we have:
	// 
	//				leftLin = 1.0, leftConst = 0.0
	//				rightLin = 1.0, rightConst = 0.0
	//
	// set boundary conditions:
	auto leftBoundary = std::make_pair(1.0, 0.0);
	auto rightBoundary = std::make_pair(1.0, 0.0);
	impl_solver.setBoundaryCondition(leftBoundary, rightBoundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	impl_solver.solve(solution, ImplicitPDESchemes::Euler);
	// get exact solution:
	auto exact = [](T x, T t, std::size_t n) {
		T const pipi = PI * PI;
		T const first = 4.0 / pipi;
		T sum{};
		T var0{};
		T var1{};
		T var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var0 = (2 * i - 1);
			var1 = std::exp(-1.0*pipi*var0*var0*t);
			var2 = std::cos(var0*PI*x) / (var0*var0);
			sum += (var1*var2);
		}
		return (0.5 - first * sum);
	};

	T const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	T benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.5, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}


template<typename T>
void testImplHeatEquationRobinBCDoubleSweepCN() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_fdm_double_sweep_solver::FDMDoubleSweepSolver;
	using lss_one_dim_heat_equation_solvers::implicit_solvers::Implicit1DHeatEquation;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation: \n\n";
	std::cout << " Using Double Sweep algorithm with implicit Cranc-Nicolson method\n\n";
	std::cout << " Value type: " << typeid(T).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = 0.5 - (4/(pi*pi))*sum_1^infty{ exp(-((2n-1)*pi)^2*t) *cos((2n-1)*pi*x)/(2n-1)^2}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Implicit1DHeatEquation<T,
		BoundaryConditionType::Robin,
		FDMDoubleSweepSolver,
		std::vector,
		std::allocator<T>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 1000;
	// number of time subdivisions:
	std::size_t const Td = 50;
	// initial condition:
	auto initialCondition = [](T x) {return x; };

	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<T> solution(Sd + 1, T{});
	// initialize solver
	implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.50, Sd, Td);
	// boundary conditions:
	// Robin boundaries are assumed to be of following form:
	//
	//				U_0 = leftLin * U_1 + leftConst
	//				U_{N-1} = rightLin * U_N + rightConst
	//
	// In our case discretizing the boundaries gives:
	// 
	//				(U_1 - U_0)/h = 0
	//				(U_N - U_{N-1})/h = 0
	//
	// Therefore we have:
	// 
	//				leftLin = 1.0, leftConst = 0.0
	//				rightLin = 1.0, rightConst = 0.0
	//
	// set boundary conditions:
	auto leftBoundary = std::make_pair(1.0, 0.0);
	auto rightBoundary = std::make_pair(1.0, 0.0);
	impl_solver.setBoundaryCondition(leftBoundary, rightBoundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	impl_solver.solve(solution);
	// get exact solution:
	auto exact = [](T x, T t, std::size_t n) {
		T const pipi = PI * PI;
		T const first = 4.0 / pipi;
		T sum{};
		T var0{};
		T var1{};
		T var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var0 = (2 * i - 1);
			var1 = std::exp(-1.0*pipi*var0*var0*t);
			var2 = std::cos(var0*PI*x) / (var0*var0);
			sum += (var1*var2);
		}
		return (0.5 - first * sum);
	};

	T const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	T benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.50, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}

void testImplHeatEquationRobinBCDoubleSweep() {
	std::cout << "================================================================================\n";
	std::cout << "===================== Implicit Heat Equation (Robin BC) ====================\n";
	std::cout << "================================================================================\n";

	testImplHeatEquationRobinBCDoubleSweepEuler<double>();
	testImplHeatEquationRobinBCDoubleSweepEuler<float>();
	testImplHeatEquationRobinBCDoubleSweepCN<double>();
	testImplHeatEquationRobinBCDoubleSweepCN<float>();

	std::cout << "================================================================================\n";
}


template<typename T>
void testImplHeatEquationRobinBCThomasLUEuler() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_fdm_thomas_lu_solver::FDMThomasLUSolver;
	using lss_one_dim_heat_equation_solvers::implicit_solvers::Implicit1DHeatEquation;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation: \n\n";
	std::cout << " Using Thomas LU algorithm with implicit Euler method\n\n";
	std::cout << " Value type: " << typeid(T).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = 0.5 - (4/(pi*pi))*sum_1^infty{ exp(-((2n-1)*pi)^2*t) *cos((2n-1)*pi*x)/(2n-1)^2}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Implicit1DHeatEquation<T,
		BoundaryConditionType::Robin,
		FDMThomasLUSolver,
		std::vector,
		std::allocator<T>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 100;
	// initial condition:
	auto initialCondition = [](T x) {return x; };
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<T> solution(Sd + 1, T{});
	// initialize solver
	implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.2, Sd, Td);
	// boundary conditions:
	// Robin boundaries are assumed to be of following form:
	//
	//				U_0 = leftLin * U_1 + leftConst
	//				U_{N-1} = rightLin * U_N + rightConst
	//
	// In our case discretizing the boundaries gives:
	// 
	//				(U_1 - U_0)/h = 0
	//				(U_N - U_{N-1})/h = 0
	//
	// Therefore we have:
	// 
	//				leftLin = 1.0, leftConst = 0.0
	//				rightLin = 1.0, rightConst = 0.0
	//
	// set boundary conditions:
	auto leftBoundary = std::make_pair(1.0, 0.0);
	auto rightBoundary = std::make_pair(1.0, 0.0);
	impl_solver.setBoundaryCondition(leftBoundary, rightBoundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	impl_solver.solve(solution, ImplicitPDESchemes::Euler);
	// get exact solution:
	auto exact = [](T x, T t, std::size_t n) {
		T const pipi = PI * PI;
		T const first = 4.0 / pipi;
		T sum{};
		T var0{};
		T var1{};
		T var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var0 = (2 * i - 1);
			var1 = std::exp(-1.0*pipi*var0*var0*t);
			var2 = std::cos(var0*PI*x) / (var0*var0);
			sum += (var1*var2);
		}
		return (0.5 - first * sum);
	};

	T const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	T benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.2, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}


template<typename T>
void testImplHeatEquationRobinBCThomasLUCN() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_fdm_thomas_lu_solver::FDMThomasLUSolver;
	using lss_one_dim_heat_equation_solvers::implicit_solvers::Implicit1DHeatEquation;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation: \n\n";
	std::cout << " Using Thomas LU algorithm with implicit Crank-Nicolson method\n\n";
	std::cout << " Value type: " << typeid(T).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = 0.5 - (4/(pi*pi))*sum_1^infty{ exp(-((2n-1)*pi)^2*t) *cos((2n-1)*pi*x)/(2n-1)^2}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Implicit1DHeatEquation<T,
		BoundaryConditionType::Robin,
		FDMThomasLUSolver,
		std::vector,
		std::allocator<T>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 1000;
	// number of time subdivisions:
	std::size_t const Td = 100;
	// initial condition:
	auto initialCondition = [](T x) {return x; };
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<T> solution(Sd + 1, T{});
	// initialize solver
	implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.20, Sd, Td);
	// Robin boundaries are assumed to be of following form:
	//
	//				U_0 = leftLin * U_1 + leftConst
	//				U_{N-1} = rightLin * U_N + rightConst
	//
	// In our case discretizing the boundaries gives:
	// 
	//				(U_1 - U_0)/h = 0
	//				(U_N - U_{N-1})/h = 0
	//
	// Therefore we have:
	// 
	//				leftLin = 1.0, leftConst = 0.0
	//				rightLin = 1.0, rightConst = 0.0
	//
	// set boundary conditions:
	auto leftBoundary = std::make_pair(1.0, 0.0);
	auto rightBoundary = std::make_pair(1.0, 0.0);
	impl_solver.setBoundaryCondition(leftBoundary, rightBoundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	impl_solver.solve(solution);
	// get exact solution:
	auto exact = [](T x, T t, std::size_t n) {
		T const pipi = PI * PI;
		T const first = 4.0 / pipi;
		T sum{};
		T var0{};
		T var1{};
		T var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var0 = (2 * i - 1);
			var1 = std::exp(-1.0*pipi*var0*var0*t);
			var2 = std::cos(var0*PI*x) / (var0*var0);
			sum += (var1*var2);
		}
		return (0.5 - first * sum);
	};

	T const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	T benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.20, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}

void testImplHeatEquationRobinBCThomasLU() {
	std::cout << "================================================================================\n";
	std::cout << "========================= Implicit Heat Equation (Robin BC) ====================\n";
	std::cout << "================================================================================\n";

	testImplHeatEquationRobinBCThomasLUEuler<double>();
	testImplHeatEquationRobinBCThomasLUEuler<float>();
	testImplHeatEquationRobinBCThomasLUCN<double>();
	testImplHeatEquationRobinBCThomasLUCN<float>();

	std::cout << "================================================================================\n";
}






// ================================================================================================================
// ================================= Heat problem with nonhomogeneous boundary conditions =========================
// ================================================================================================================

template<typename T>
void testImplNonHomHeatEquationDirichletBCDoubleSweepEuler() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_fdm_double_sweep_solver::FDMDoubleSweepSolver;
	using lss_one_dim_heat_equation_solvers::implicit_solvers::Implicit1DHeatEquation;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation with Non-hom. BC: \n\n";
	std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
	std::cout << " Value type: " << typeid(T).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = 0, U(1,t) = 100, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = 100*x + (198/pi)*sum_0^infty{ (-1)^(n)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Implicit1DHeatEquation<T,
		BoundaryConditionType::Dirichlet,
		FDMDoubleSweepSolver,
		std::vector,
		std::allocator<T>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 100;
	// initial condition:
	auto initialCondition = [](T x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 100.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<T> solution(Sd + 1, T{});
	// initialize solver
	implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.5, Sd, Td);
	// set boundary conditions:
	impl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	impl_solver.solve(solution, ImplicitPDESchemes::Euler);
	// get exact solution:
	auto exact = [](T x, T t, std::size_t n) {
		T const first = 198.0 / PI;
		T sum{};
		T var1{};
		T var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var1 = std::pow(-1.0, i ) * std::exp(-1.0*(i*PI)*(i*PI)*t);
			var2 = std::sin(i*PI*x) / i;
			sum += (var1*var2);
		}
		return (100*x + first * sum);
	};

	T const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	T benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.5, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}


template<typename T>
void testImplNonHomHeatEquationDirichletBCDoubleSweepCN() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_fdm_double_sweep_solver::FDMDoubleSweepSolver;
	using lss_one_dim_heat_equation_solvers::implicit_solvers::Implicit1DHeatEquation;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation with Non-hom. BC:: \n\n";
	std::cout << " Using Double Sweep algorithm with implicit Crank-Nicolson method\n\n";
	std::cout << " Value type: " << typeid(T).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = U(0,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = 100*x + (198/pi)*sum_0^infty{ (-1)^(n)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Implicit1DHeatEquation<T,
		BoundaryConditionType::Dirichlet,
		FDMDoubleSweepSolver,
		std::vector,
		std::allocator<T>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 100;
	// initial condition:
	auto initialCondition = [](T x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 100.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<T> solution(Sd + 1, T{});
	// initialize solver
	implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.20, Sd, Td);
	// set boundary conditions:
	impl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	impl_solver.solve(solution);
	// get exact solution:
	auto exact = [](T x, T t, std::size_t n) {
		T const first = 198.0 / PI;
		T sum{};
		T var1{};
		T var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var1 = std::pow(-1.0, i ) * std::exp(-1.0*(i*PI)*(i*PI)*t);
			var2 = std::sin(i*PI*x) / i;
			sum += (var1*var2);
		}
		return (100*x + first * sum);
	};

	T const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	T benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.20, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}

void testImplNonHomHeatEquationDirichletBCDoubleSweep() {
	std::cout << "================================================================================\n";
	std::cout << "=========== Implicit Heat Equation (non-homogenous Dirichlet BC) ===============\n";
	std::cout << "================================================================================\n";

	testImplNonHomHeatEquationDirichletBCDoubleSweepEuler<double>();
	testImplNonHomHeatEquationDirichletBCDoubleSweepEuler<float>();
	testImplNonHomHeatEquationDirichletBCDoubleSweepCN<double>();
	testImplNonHomHeatEquationDirichletBCDoubleSweepCN<float>();

	std::cout << "================================================================================\n";
}


template<typename T>
void testImplNonHomHeatEquationDirichletBCThomasLUEuler() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_fdm_thomas_lu_solver::FDMThomasLUSolver;
	using lss_one_dim_heat_equation_solvers::implicit_solvers::Implicit1DHeatEquation;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation with non-hom BC: \n\n";
	std::cout << " Using Thomas LU algorithm with implicit Euler method\n\n";
	std::cout << " Value type: " << typeid(T).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = U(0,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = 100.0*x + (198/pi)*sum_0^infty{ (-1)^(n)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Implicit1DHeatEquation<T,
		BoundaryConditionType::Dirichlet,
		FDMThomasLUSolver,
		std::vector,
		std::allocator<T>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 100;
	// initial condition:
	auto initialCondition = [](T x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 100.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<T> solution(Sd + 1, T{});
	// initialize solver
	implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.5, Sd, Td);
	// set boundary conditions:
	impl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	impl_solver.solve(solution, ImplicitPDESchemes::Euler);
	// get exact solution:
	auto exact = [](T x, T t, std::size_t n) {
		T const first = 198.0 / PI;
		T sum{};
		T var1{};
		T var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var1 = std::pow(-1.0, i) * std::exp(-1.0*(i*PI)*(i*PI)*t);
			var2 = std::sin(i*PI*x) / i;
			sum += (var1*var2);
		}
		return (100.0*x + first * sum);
	};

	T const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	T benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.5, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}


template<typename T>
void testImplNonHomHeatEquationDirichletBCThomasLUCN() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_fdm_thomas_lu_solver::FDMThomasLUSolver;
	using lss_one_dim_heat_equation_solvers::implicit_solvers::Implicit1DHeatEquation;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation with non-hom BC: \n\n";
	std::cout << " Using Thomas LU algorithm with implicit Crank-Nicolson method\n\n";
	std::cout << " Value type: " << typeid(T).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = 0, U(1,t) = 100, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = 100*x + (198/pi)*sum_0^infty{ (-1)^(n)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Implicit1DHeatEquation<T,
		BoundaryConditionType::Dirichlet,
		FDMThomasLUSolver,
		std::vector,
		std::allocator<T>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 100;
	// initial condition:
	auto initialCondition = [](T x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 100.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<T> solution(Sd + 1, T{});
	// initialize solver
	implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.20, Sd, Td);
	// set boundary conditions:
	impl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	impl_solver.solve(solution);
	// get exact solution:
	auto exact = [](T x, T t, std::size_t n) {
		T const first = 198.0 / PI;
		T sum{};
		T var1{};
		T var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var1 = std::pow(-1.0, i) * std::exp(-1.0*(i*PI)*(i*PI)*t);
			var2 = std::sin(i*PI*x) / i;
			sum += (var1*var2);
		}
		return (100.0*x + first * sum);
	};

	T const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	T benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.20, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}

void testImplNonHomHeatEquationDirichletBCThomasLU() {
	std::cout << "================================================================================\n";
	std::cout << "===== Implicit Heat Equation (with non-homogeneous Dirichlet BC) ===============\n";
	std::cout << "================================================================================\n";

	testImplNonHomHeatEquationDirichletBCThomasLUEuler<double>();
	testImplNonHomHeatEquationDirichletBCThomasLUEuler<float>();
	testImplNonHomHeatEquationDirichletBCThomasLUCN<double>();
	testImplNonHomHeatEquationDirichletBCThomasLUCN<float>();

	std::cout << "================================================================================\n";
}


// ===========================================================================================================
// ==================================== EPLICIT SOLVERS ======================================================
// ===========================================================================================================

// ================================================================================================================
// ==================================== Heat problem with homogeneous boundary conditions =========================
// ================================================================================================================

template<typename T>
void testExplHeatEquationDirichletBCEuler() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ExplicitPDESchemes;
	using lss_one_dim_heat_equation_solvers::explicit_solvers::Explicit1DHeatEquation;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation: \n\n";
	std::cout << " Using explicit Euler method\n\n";
	std::cout << " Value type: " << typeid(T).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Explicit1DHeatEquation<T,
		BoundaryConditionType::Dirichlet,
		std::vector,
		std::allocator<T>> explicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 10000;
	// initial condition:
	auto initialCondition = [](T x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 0.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<T> solution(Sd + 1, T{});
	// initialize solver
	explicit_solver expl_solver(Range<T>(0.0, 1.0), 0.5, Sd, Td);
	// set boundary conditions:
	expl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	expl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	expl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	expl_solver.solve(solution, ExplicitPDESchemes::Euler);
	// get exact solution:
	auto exact = [](T x, T t, std::size_t n) {
		T const first = 2.0 / PI;
		T sum{};
		T var1{};
		T var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var1 = std::pow(-1.0, i + 1) * std::exp(-1.0*(i*PI)*(i*PI)*t);
			var2 = std::sin(i*PI*x) / i;
			sum += (var1*var2);
		}
		return (first * sum);
	};

	T const h = expl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	T benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.5, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}


template<typename T>
void testExplHeatEquationDirichletBCADEBC() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ExplicitPDESchemes;
	using lss_one_dim_heat_equation_solvers::explicit_solvers::Explicit1DHeatEquation;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation: \n\n";
	std::cout << " Using explicit ADE Barakat Clark method\n\n";
	std::cout << " Value type: " << typeid(T).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Explicit1DHeatEquation<T,
		BoundaryConditionType::Dirichlet,
		std::vector,
		std::allocator<T>> explicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 10000;
	// initial condition:
	auto initialCondition = [](T x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 0.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<T> solution(Sd + 1, T{});
	// initialize solver
	explicit_solver expl_solver(Range<T>(0.0, 1.0), 0.50, Sd, Td);
	// set boundary conditions:
	expl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	expl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	expl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	expl_solver.solve(solution);
	// get exact solution:
	auto exact = [](T x, T t, std::size_t n) {
		T const first = 2.0 / PI;
		T sum{};
		T var1{};
		T var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var1 = std::pow(-1.0, i + 1) * std::exp(-1.0*(i*PI)*(i*PI)*t);
			var2 = std::sin(i*PI*x) / i;
			sum += (var1*var2);
		}
		return (first * sum);
	};

	T const h = expl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	T benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.50, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}

template<typename T>
void testExplHeatEquationDirichletBCADES() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ExplicitPDESchemes;
	using lss_one_dim_heat_equation_solvers::explicit_solvers::Explicit1DHeatEquation;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation: \n\n";
	std::cout << " Using explicit ADE Saulyev method\n\n";
	std::cout << " Value type: " << typeid(T).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Explicit1DHeatEquation<T,
		BoundaryConditionType::Dirichlet,
		std::vector,
		std::allocator<T>> explicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 10000;
	// initial condition:
	auto initialCondition = [](T x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 0.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<T> solution(Sd + 1, T{});
	// initialize solver
	explicit_solver expl_solver(Range<T>(0.0, 1.0), 0.50, Sd, Td);
	// set boundary conditions:
	expl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	expl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	expl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	expl_solver.solve(solution,ExplicitPDESchemes::ADESaulyev);
	// get exact solution:
	auto exact = [](T x, T t, std::size_t n) {
		T const first = 2.0 / PI;
		T sum{};
		T var1{};
		T var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var1 = std::pow(-1.0, i + 1) * std::exp(-1.0*(i*PI)*(i*PI)*t);
			var2 = std::sin(i*PI*x) / i;
			sum += (var1*var2);
		}
		return (first * sum);
	};

	T const h = expl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	T benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.50, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}

void testExplHeatEquationDirichletBC() {
	std::cout << "================================================================================\n";
	std::cout << "===================== Explicit Heat Equation (Dirichlet BC) ====================\n";
	std::cout << "================================================================================\n";

	testExplHeatEquationDirichletBCEuler<double>();
	testExplHeatEquationDirichletBCEuler<float>();
	testExplHeatEquationDirichletBCADEBC<double>();
	testExplHeatEquationDirichletBCADEBC<float>();
	testExplHeatEquationDirichletBCADES<double>();
	testExplHeatEquationDirichletBCADES<float>();

	std::cout << "================================================================================\n";
}


// ================================================================================================================
// ================================= Heat problem with nonhomogeneous boundary conditions =========================
// ================================================================================================================

template<typename T>
void testExplNonHomHeatEquationDirichletBCEuler() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ExplicitPDESchemes;
	using lss_one_dim_heat_equation_solvers::explicit_solvers::Explicit1DHeatEquation;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation with Non-hom BC: \n\n";
	std::cout << " Using explicit Euler method\n\n";
	std::cout << " Value type: " << typeid(T).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = 0, U(1,t) = 100, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = 100*x + (198/pi)*sum_0^infty{ (-1)^(n)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Explicit1DHeatEquation<T,
		BoundaryConditionType::Dirichlet,
		std::vector,
		std::allocator<T>> explicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 10000;
	// initial condition:
	auto initialCondition = [](T x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 100.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<T> solution(Sd + 1, T{});
	// initialize solver
	explicit_solver expl_solver(Range<T>(0.0, 1.0), 0.5, Sd, Td);
	// set boundary conditions:
	expl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	expl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	expl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	expl_solver.solve(solution, ExplicitPDESchemes::Euler);
	// get exact solution:
	auto exact = [](T x, T t, std::size_t n) {
		T const first = 198.0 / PI;
		T sum{};
		T var1{};
		T var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var1 = std::pow(-1.0, i ) * std::exp(-1.0*(i*PI)*(i*PI)*t);
			var2 = std::sin(i*PI*x) / i;
			sum += (var1*var2);
		}
		return (100.0*x + first * sum);
	};

	T const h = expl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	T benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.5, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}


template<typename T>
void testExplNonHomHeatEquationDirichletBCADEBC() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ExplicitPDESchemes;
	using lss_one_dim_heat_equation_solvers::explicit_solvers::Explicit1DHeatEquation;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation with non-hom BC: \n\n";
	std::cout << " Using explicit ADE Barakat Clark method\n\n";
	std::cout << " Value type: " << typeid(T).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = 0, U(1,t) = 100, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = 100*x +  (198/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Explicit1DHeatEquation<T,
		BoundaryConditionType::Dirichlet,
		std::vector,
		std::allocator<T>> explicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 10000;
	// initial condition:
	auto initialCondition = [](T x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 100.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<T> solution(Sd + 1, T{});
	// initialize solver
	explicit_solver expl_solver(Range<T>(0.0, 1.0), 0.50, Sd, Td);
	// set boundary conditions:
	expl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	expl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	expl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	expl_solver.solve(solution);
	// get exact solution:
	auto exact = [](T x, T t, std::size_t n) {
		T const first = 198.0 / PI;
		T sum{};
		T var1{};
		T var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var1 = std::pow(-1.0, i) * std::exp(-1.0*(i*PI)*(i*PI)*t);
			var2 = std::sin(i*PI*x) / i;
			sum += (var1*var2);
		}
		return (100.0*x + first * sum);
	};

	T const h = expl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	T benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.50, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}

template<typename T>
void testExplNonHomHeatEquationDirichletBCADES() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ExplicitPDESchemes;
	using lss_one_dim_heat_equation_solvers::explicit_solvers::Explicit1DHeatEquation;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation with non-hom BC: \n\n";
	std::cout << " Using explicit ADE Saulyev method\n\n";
	std::cout << " Value type: " << typeid(T).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = 0, U(1,t) = 100, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = 100.0*x + (198/pi)*sum_0^infty{ (-1)^(n)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Explicit1DHeatEquation<T,
		BoundaryConditionType::Dirichlet,
		std::vector,
		std::allocator<T>> explicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 10000;
	// initial condition:
	auto initialCondition = [](T x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 100.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<T> solution(Sd + 1, T{});
	// initialize solver
	explicit_solver expl_solver(Range<T>(0.0, 1.0), 0.50, Sd, Td);
	// set boundary conditions:
	expl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	expl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	expl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	expl_solver.solve(solution, ExplicitPDESchemes::ADESaulyev);
	// get exact solution:
	auto exact = [](T x, T t, std::size_t n) {
		T const first = 198.0 / PI;
		T sum{};
		T var1{};
		T var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var1 = std::pow(-1.0, i ) * std::exp(-1.0*(i*PI)*(i*PI)*t);
			var2 = std::sin(i*PI*x) / i;
			sum += (var1*var2);
		}
		return (100.0*x + first * sum);
	};

	T const h = expl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	T benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.50, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}

void testExplNonHomHeatEquationDirichletBC() {
	std::cout << "================================================================================\n";
	std::cout << "========= Explicit Heat Equation (with non-homogeneous Dirichlet BC) ===========\n";
	std::cout << "================================================================================\n";

	testExplNonHomHeatEquationDirichletBCEuler<double>();
	testExplNonHomHeatEquationDirichletBCEuler<float>();
	testExplNonHomHeatEquationDirichletBCADEBC<double>();
	testExplNonHomHeatEquationDirichletBCADEBC<float>();
	testExplNonHomHeatEquationDirichletBCADES<double>();
	testExplNonHomHeatEquationDirichletBCADES<float>();

	std::cout << "================================================================================\n";
}


// ================================================================================================================
// ============================== Heat problem with homogeneous Robin boundary conditions =========================
// ================================================================================================================

template<typename T>
void testExplHomHeatEquationRobinBCEuler() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_one_dim_heat_equation_solvers::explicit_solvers::Explicit1DHeatEquation;


	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation: \n\n";
	std::cout << " Using explicit Euler method\n\n";
	std::cout << " Value type: " << typeid(T).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = 0.5 - (4/(pi*pi))*sum_1^infty{ exp(-((2n-1)*pi)^2*t) *cos((2n-1)*pi*x)/(2n-1)^2}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Explicit1DHeatEquation<T,
		BoundaryConditionType::Robin,
		std::vector,
		std::allocator<T>> explicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 5000;
	// initial condition:
	auto initialCondition = [](T x) {return x; };
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<T> solution(Sd + 1, 0.0);
	// initialize solver
	explicit_solver expl_solver(Range<T>(0.0, 1.0), 0.2, Sd, Td);
	// boundary conditions:
	// Robin boundaries are assumed to be of following form:
	//
	//				U_0 = leftLin * U_1 + leftConst
	//				U_{N-1} = rightLin * U_N + rightConst
	//
	// In our case discretizing the boundaries gives:
	// 
	//				(U_1 - U_-1)/2h = 0
	//				(U_N+1 - U_{N-1})/2h = 0
	//
	// Therefore we have:
	// 
	//				leftLin = 1.0, leftConst = 0.0
	//				rightLin = 1.0, rightConst = 0.0
	//
	auto const h = expl_solver.spaceStep();
	auto leftBoundary = std::make_pair(1.0, 0.0);
	auto rightBoundary = std::make_pair(1.0, 0.0);
	// set boundary conditions:
	expl_solver.setBoundaryCondition(leftBoundary, rightBoundary);
	// set initial condition:
	expl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	expl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	expl_solver.solve(solution);
	// get exact solution:
	auto exact = [](T x, T t, std::size_t n) {
		T const pipi = PI * PI;
		T const first = 4.0 / pipi;
		T sum{};
		T var0{};
		T var1{};
		T var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var0 = (2 * i - 1);
			var1 = std::exp(-1.0*pipi*var0*var0*t);
			var2 = std::cos(var0*PI*x) / (var0*var0);
			sum += (var1*var2);
		}
		return (0.5 - first * sum);
	};

	std::cout << "tp : FDM | Exact | Abs Diff\n";
	T benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.2, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}


void testExplHomHeatEquationRobinBC() {
	std::cout << "================================================================================\n";
	std::cout << "========= Explicit Heat Equation (with homogeneous Robin BC) ===================\n";
	std::cout << "================================================================================\n";

	testExplHomHeatEquationRobinBCEuler<double>();
	testExplHomHeatEquationRobinBCEuler<float>();


	std::cout << "================================================================================\n";
}

#endif ///_LSS_ONE_DIM_HEAT_EQUATION_SOLVERS_T