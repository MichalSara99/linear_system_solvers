#pragma once
#if !defined(_LSS_ONE_DIM_ADVECTION_DIFFUSION_EQUATION_SOLVERS_T)
#define _LSS_ONE_DIM_ADVECTION_DIFFUSION_EQUATION_SOLVERS_T

#include"lss_types.h"
#include"lss_utility.h"
#include"lss_fdm_double_sweep_solver.h"
#include"lss_fdm_thomas_lu_solver.h"
#include"lss_one_dim_advection_diffusion_equation_solvers.h"


#define PI 3.14159

// ================================================================================================================
// =========================================== IMPLICIT SOLVERS ===================================================
// ================================================================================================================


// ================================================================================================================
// ======================= Advection Diffusion problem with homogeneous boundary conditions =======================
// ================================================================================================================

template<typename T>
void testImplAdvectionDiffEquationDirichletBCDoubleSweepEuler() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_fdm_double_sweep_solver::FDMDoubleSweepSolver;
	using lss_one_dim_advection_diffusion_equation_solvers::implicit_solvers::Implicit1DAdvectionDiffusionEquation;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Advection Diffusion equation: \n\n";
	std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
	std::cout << " Value type: " << typeid(T).name() << "\n\n";
	std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <-1,1> and t > 0,\n";
	std::cout << " U(-1,t) = U(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = 1, x in <-1,1> \n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DAdvectionDiffusionEquation
	typedef Implicit1DAdvectionDiffusionEquation<T,
		BoundaryConditionType::Dirichlet,
		FDMDoubleSweepSolver,
		std::vector,
		std::allocator<T>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 100;
	// initial condition:
	auto initialCondition = [](T x) {return 1.0; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 0.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<T> solution(Sd + 1, T{});
	// initialize solver
	implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.08, Sd, Td);
	// set boundary conditions:
	impl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// set convection term 
	impl_solver.setConvection(1.0);
	// get the solution:
	impl_solver.solve(solution, ImplicitPDESchemes::Euler);
	// get exact solution:
	auto exact = [](T x, T t, std::size_t n) {
		T const first = 2.0 / PI;
		T const exp_0p5x = std::exp(0.5*x);
		T const exp_m0p5 = std::exp(-0.5);
		T np_sqr{};
		T sum{};
		T num{}, den{}, var{};
		T lambda{};
		for (std::size_t i = 1; i <= n; ++i) {
			np_sqr = (i*i*PI*PI);
			lambda = 0.25 + np_sqr;
			num = (1.0 - std::pow(-1.0, i)*exp_m0p5)*exp_0p5x*std::exp(-1.0*lambda*t)*std::sin(i*PI*x);
			den = i * (1.0 + (0.25 / np_sqr));
			var = num / den;
			sum += var;
		}
		return (first * sum);
	};

	T const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	T benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.08, 30);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}


template<typename T>
void testImplAdvectionDiffEquationDirichletBCDoubleSweepCN() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_fdm_double_sweep_solver::FDMDoubleSweepSolver;
	using lss_one_dim_advection_diffusion_equation_solvers::implicit_solvers::Implicit1DAdvectionDiffusionEquation;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Advection Diffusion equation: \n\n";
	std::cout << " Using Double Sweep algorithm with implicit Crank-Nicolson method\n\n";
	std::cout << " Value type: " << typeid(T).name() << "\n\n";
	std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <-1,1> and t > 0,\n";
	std::cout << " U(-1,t) = U(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = -sin(pi*x), x in <-1,1> \n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DAdvectionDiffusionEquation
	typedef Implicit1DAdvectionDiffusionEquation<T,
		BoundaryConditionType::Dirichlet,
		FDMDoubleSweepSolver,
		std::vector,
		std::allocator<T>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 100;
	// initial condition:
	auto initialCondition = [](T x) {return 1.0; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 0.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<T> solution(Sd + 1, T{});
	// initialize solver
	implicit_solver impl_solver(Range<T>(0.0, 1.0), 0.09, Sd, Td);
	// set boundary conditions:
	impl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// set convection term 
	impl_solver.setConvection(1.0);
	// get the solution:
	impl_solver.solve(solution);
	// get exact solution:
	auto exact = [](T x, T t, std::size_t n) {
		T const first = 2.0 / PI;
		T const exp_0p5x = std::exp(0.5*x);
		T const exp_m0p5 = std::exp(-0.5);
		T np_sqr{};
		T sum{};
		T num{}, den{}, var{};
		T lambda{};
		for (std::size_t i = 1; i <= n; ++i) {
			np_sqr = (i*i*PI*PI);
			lambda = 0.25 + np_sqr;
			num = (1.0 - std::pow(-1.0, i)*exp_m0p5)*exp_0p5x*std::exp(-1.0*lambda*t)*std::sin(i*PI*x);
			den = i * (1.0 + (0.25 / np_sqr));
			var = num / den;
			sum += var;
		}
		return (first * sum);
	};

	T const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	T benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.09, 40);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}

void testImplAdvectionDiffEquationDirichletBCDoubleSweep() {
	std::cout << "================================================================================\n";
	std::cout << "=============== Implicit Advection Diffusion Equation (Dirichlet BC) ===========\n";
	std::cout << "================================================================================\n";

	testImplAdvectionDiffEquationDirichletBCDoubleSweepEuler<double>();
	testImplAdvectionDiffEquationDirichletBCDoubleSweepEuler<float>();
	testImplAdvectionDiffEquationDirichletBCDoubleSweepCN<double>();
	testImplAdvectionDiffEquationDirichletBCDoubleSweepCN<float>();

	std::cout << "================================================================================\n";
}




// ================================================================================================================
// =========================================== EXPLICIT SOLVERS ===================================================
// ================================================================================================================


// ================================================================================================================
// ======================= Advection Diffusion problem with homogeneous boundary conditions =======================
// ================================================================================================================

template<typename T>
void testExplAdvectionDiffEquationDirichletBCEuler() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ExplicitPDESchemes;
	using lss_one_dim_advection_diffusion_equation_solvers::explicit_solvers::Explicit1DAdvectionDiffusionEquation;;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Advection Diffusion equation: \n\n";
	std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
	std::cout << " Value type: " << typeid(T).name() << "\n\n";
	std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <-1,1> and t > 0,\n";
	std::cout << " U(-1,t) = U(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = 1, x in <-1,1> \n\n";
	std::cout << "===============================================================================\n";

	// typedef the Explicit1DAdvectionDiffusionEquation
	typedef Explicit1DAdvectionDiffusionEquation<T,
		BoundaryConditionType::Dirichlet,
		std::vector,
		std::allocator<T>> explicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 10000;
	// initial condition:
	auto initialCondition = [](T x) {return 1.0; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 0.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<T> solution(Sd + 1, T{});
	// initialize solver
	explicit_solver expl_solver(Range<T>(0.0, 1.0), 0.08, Sd, Td);
	// set boundary conditions:
	expl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	expl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	expl_solver.setThermalDiffusivity(1.0);
	// set convection term 
	expl_solver.setConvection(1.0);
	// get the solution:
	expl_solver.solve(solution, ExplicitPDESchemes::Euler);
	// get exact solution:
	auto exact = [](T x, T t, std::size_t n) {
		T const first = 2.0 / PI;
		T const exp_0p5x = std::exp(0.5*x);
		T const exp_m0p5 = std::exp(-0.5);
		T np_sqr{};
		T sum{};
		T num{}, den{}, var{};
		T lambda{};
		for (std::size_t i = 1; i <= n; ++i) {
			np_sqr = (i*i*PI*PI);
			lambda = 0.25 + np_sqr;
			num = (1.0 - std::pow(-1.0, i)*exp_m0p5)*exp_0p5x*std::exp(-1.0*lambda*t)*std::sin(i*PI*x);
			den = i * (1.0 + (0.25 / np_sqr));
			var = num / den;
			sum += var;
		}
		return (first * sum);
	};

	T const h = expl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	T benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.08, 30);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}

template<typename T>
void testExplAdvectionDiffEquationDirichletBCADEBC() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ExplicitPDESchemes;
	using lss_one_dim_advection_diffusion_equation_solvers::explicit_solvers::Explicit1DAdvectionDiffusionEquation;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Advection Diffusion equation: \n\n";
	std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
	std::cout << " Value type: " << typeid(T).name() << "\n\n";
	std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <-1,1> and t > 0,\n";
	std::cout << " U(-1,t) = U(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = 1, x in <-1,1> \n\n";
	std::cout << "===============================================================================\n";

	// typedef the Explicit1DAdvectionDiffusionEquation
	typedef Explicit1DAdvectionDiffusionEquation<T,
		BoundaryConditionType::Dirichlet,
		std::vector,
		std::allocator<T>> explicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 10000;
	// initial condition:
	auto initialCondition = [](T x) {return 1.0; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 0.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<T> solution(Sd + 1, T{});
	// initialize solver
	explicit_solver expl_solver(Range<T>(0.0, 1.0), 0.08, Sd, Td);
	// set boundary conditions:
	expl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	expl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	expl_solver.setThermalDiffusivity(1.0);
	// set convection term 
	expl_solver.setConvection(1.0);
	// get the solution:
	expl_solver.solve(solution);
	// get exact solution:
	auto exact = [](T x, T t, std::size_t n) {
		T const first = 2.0 / PI;
		T const exp_0p5x = std::exp(0.5*x);
		T const exp_m0p5 = std::exp(-0.5);
		T np_sqr{};
		T sum{};
		T num{}, den{}, var{};
		T lambda{};
		for (std::size_t i = 1; i <= n; ++i) {
			np_sqr = (i*i*PI*PI);
			lambda = 0.25 + np_sqr;
			num = (1.0 - std::pow(-1.0, i)*exp_m0p5)*exp_0p5x*std::exp(-1.0*lambda*t)*std::sin(i*PI*x);
			den = i * (1.0 + (0.25 / np_sqr));
			var = num / den;
			sum += var;
		}
		return (first * sum);
	};

	T const h = expl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	T benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.08, 30);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}

template<typename T>
void testExplAdvectionDiffEquationDirichletBCADES() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ExplicitPDESchemes;
	using lss_one_dim_advection_diffusion_equation_solvers::explicit_solvers::Explicit1DAdvectionDiffusionEquation;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Advection Diffusion equation: \n\n";
	std::cout << " Using Double Sweep algorithm with implicit Euler method\n\n";
	std::cout << " Value type: " << typeid(T).name() << "\n\n";
	std::cout << " U_t(x,t)  +  U_x(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <-1,1> and t > 0,\n";
	std::cout << " U(-1,t) = U(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = 1, x in <-1,1> \n\n";
	std::cout << "===============================================================================\n";

	// typedef the Explicit1DAdvectionDiffusionEquation
	typedef Explicit1DAdvectionDiffusionEquation<T,
		BoundaryConditionType::Dirichlet,
		std::vector,
		std::allocator<T>> explicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 10000;
	// initial condition:
	auto initialCondition = [](T x) {return 1.0; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 0.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<T> solution(Sd + 1, T{});
	// initialize solver
	explicit_solver expl_solver(Range<T>(0.0, 1.0), 0.08, Sd, Td);
	// set boundary conditions:
	expl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	expl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	expl_solver.setThermalDiffusivity(1.0);
	// set convection term 
	expl_solver.setConvection(1.0);
	// get the solution:
	expl_solver.solve(solution, ExplicitPDESchemes::ADESaulyev);
	// get exact solution:
	auto exact = [](T x, T t, std::size_t n) {
		T const first = 2.0 / PI;
		T const exp_0p5x = std::exp(0.5*x);
		T const exp_m0p5 = std::exp(-0.5);
		T np_sqr{};
		T sum{};
		T num{}, den{}, var{};
		T lambda{};
		for (std::size_t i = 1; i <= n; ++i) {
			np_sqr = (i*i*PI*PI);
			lambda = 0.25 + np_sqr;
			num = (1.0 - std::pow(-1.0, i)*exp_m0p5)*exp_0p5x*std::exp(-1.0*lambda*t)*std::sin(i*PI*x);
			den = i * (1.0 + (0.25 / np_sqr));
			var = num / den;
			sum += var;
		}
		return (first * sum);
	};

	T const h = expl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	T benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.08, 30);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}


void testExplAdvectionDiffEquationDirichletBC() {
	std::cout << "================================================================================\n";
	std::cout << "=============== Explicit Advection Diffusion Equation (Dirichlet BC) ===========\n";
	std::cout << "================================================================================\n";

	testExplAdvectionDiffEquationDirichletBCEuler<double>();
	testExplAdvectionDiffEquationDirichletBCEuler<float>();
	testExplAdvectionDiffEquationDirichletBCADEBC<double>();
	testExplAdvectionDiffEquationDirichletBCADEBC<float>();
	testExplAdvectionDiffEquationDirichletBCADES<double>();
	testExplAdvectionDiffEquationDirichletBCADES<float>();

	std::cout << "================================================================================\n";
}

















#endif ///_LSS_ONE_DIM_ADVECTION_DIFFUSION_EQUATION_SOLVERS_T