#pragma once
#if !defined(_LSS_ONE_DIM_HEAT_EQUATION_SOLVERS_CUDA_T)
#define _LSS_ONE_DIM_HEAT_EQUATION_SOLVERS_CUDA_T

#include"lss_types.h"
#include"lss_utility.h"
#include"lss_one_dim_heat_equation_solvers_cuda.h"

#define PI 3.14159

// ================================================================================================================
// =========================================== IMPLICIT SOLVERS ===================================================
// ================================================================================================================


// ================================================================================================================
// ==================================== Heat problem with homogeneous boundary conditions =========================
// ================================================================================================================

void testImplHeatEquationDoubleDirichletBCDeviceEuler() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_types::MemorySpace;
	using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
	using lss_one_dim_heat_equation_solvers_cuda::implicit_solvers::Implicit1DHeatEquationCUDA;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation: \n\n";
	std::cout << " Using CUDA solvers algorithm with implicit Euler method\n\n";
	std::cout << " Value type: " << typeid(double).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Implicit1DHeatEquationCUDA<double,
		BoundaryConditionType::Dirichlet,
		MemorySpace::Device,
		RealSparseSolverCUDA,
		std::vector,
		std::allocator<double>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 100;
	// initial condition:
	auto initialCondition = [](double x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 0.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<double> solution(Sd + 1, 0.0);
	// initialize solver
	implicit_solver impl_solver(Range<double>(0.0, 1.0), 0.2, Sd, Td);
	// set boundary conditions:
	impl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	impl_solver.solve(solution, ImplicitPDESchemes::Euler);
	// get exact solution:
	auto exact = [](double x, double t, std::size_t n) {
		double const first = 2.0 / PI;
		double sum{};
		double var1{};
		double var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var1 = std::pow(-1.0, i + 1) * std::exp(-1.0*(i*PI)*(i*PI)*t);
			var2 = std::sin(i*PI*x) / i;
			sum += (var1*var2);
		}
		return (first * sum);
	};

	double const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	double benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.2, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}


void testImplHeatEquationFloatDirichletBCDeviceEuler() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_types::MemorySpace;
	using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
	using lss_one_dim_heat_equation_solvers_cuda::implicit_solvers::Implicit1DHeatEquationCUDA;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation: \n\n";
	std::cout << " Using CUDA solvers algorithm with implicit Euler method\n\n";
	std::cout << " Value type: " << typeid(float).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Implicit1DHeatEquationCUDA<float,
		BoundaryConditionType::Dirichlet,
		MemorySpace::Device,
		RealSparseSolverCUDA,
		std::vector,
		std::allocator<float>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 100;
	// initial condition:
	auto initialCondition = [](float x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 0.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<float> solution(Sd + 1, 0.0);
	// initialize solver
	implicit_solver impl_solver(Range<float>(0.0, 1.0), 0.2, Sd, Td);
	// set boundary conditions:
	impl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	impl_solver.solve(solution, ImplicitPDESchemes::Euler);
	// get exact solution:
	auto exact = [](float x, float t, std::size_t n) {
		float const first = 2.0 / PI;
		float sum{};
		float var1{};
		float var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var1 = std::pow(-1.0, i + 1) * std::exp(-1.0*(i*PI)*(i*PI)*t);
			var2 = std::sin(i*PI*x) / i;
			sum += (var1*var2);
		}
		return (first * sum);
	};

	float const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	float benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.2, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}

void testImplHeatEquationDoubleDirichletBCDeviceCN() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_types::MemorySpace;
	using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
	using lss_one_dim_heat_equation_solvers_cuda::implicit_solvers::Implicit1DHeatEquationCUDA;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation: \n\n";
	std::cout << " Using CUDA solvers algorithm with implicit Clark-Nicolson method\n\n";
	std::cout << " Value type: " << typeid(double).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Implicit1DHeatEquationCUDA<double,
		BoundaryConditionType::Dirichlet,
		MemorySpace::Device,
		RealSparseSolverCUDA,
		std::vector,
		std::allocator<double>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 100;
	// initial condition:
	auto initialCondition = [](double x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 0.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<double> solution(Sd + 1, 0.0);
	// initialize solver
	implicit_solver impl_solver(Range<double>(0.0, 1.0), 0.2, Sd, Td);
	// set boundary conditions:
	impl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	impl_solver.solve(solution, ImplicitPDESchemes::CrankNicolson);
	// get exact solution:
	auto exact = [](double x, double t, std::size_t n) {
		double const first = 2.0 / PI;
		double sum{};
		double var1{};
		double var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var1 = std::pow(-1.0, i + 1) * std::exp(-1.0*(i*PI)*(i*PI)*t);
			var2 = std::sin(i*PI*x) / i;
			sum += (var1*var2);
		}
		return (first * sum);
	};

	double const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	double benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.2, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}

void testImplHeatEquationFloatDirichletBCDeviceCN() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_types::MemorySpace;
	using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
	using lss_one_dim_heat_equation_solvers_cuda::implicit_solvers::Implicit1DHeatEquationCUDA;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation: \n\n";
	std::cout << " Using CUDA solvers algorithm with implicit Clark-Nicolson method\n\n";
	std::cout << " Value type: " << typeid(float).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Implicit1DHeatEquationCUDA<float,
		BoundaryConditionType::Dirichlet,
		MemorySpace::Device,
		RealSparseSolverCUDA,
		std::vector,
		std::allocator<float>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 1000;
	// number of time subdivisions:
	std::size_t const Td = 1000;
	// initial condition:
	auto initialCondition = [](float x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 0.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<float> solution(Sd + 1, 0.0);
	// initialize solver
	implicit_solver impl_solver(Range<float>(0.0, 1.0), 0.2, Sd, Td);
	// set boundary conditions:
	impl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	impl_solver.solve(solution, ImplicitPDESchemes::CrankNicolson);
	// get exact solution:
	auto exact = [](float x, float t, std::size_t n) {
		float const first = 2.0 / PI;
		float sum{};
		float var1{};
		float var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var1 = std::pow(-1.0, i + 1) * std::exp(-1.0*(i*PI)*(i*PI)*t);
			var2 = std::sin(i*PI*x) / i;
			sum += (var1*var2);
		}
		return (first * sum);
	};

	float const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	float benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.2, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}


void testImplHeatEquationDirichletBCDevice() {
	std::cout << "================================================================================\n";
	std::cout << "===================== Implicit Heat Equation (Dirichlet BC) ====================\n";
	std::cout << "================================================================================\n";

	testImplHeatEquationDoubleDirichletBCDeviceEuler();
	testImplHeatEquationFloatDirichletBCDeviceEuler();
	testImplHeatEquationDoubleDirichletBCDeviceCN();
	testImplHeatEquationFloatDirichletBCDeviceCN();

	std::cout << "================================================================================\n";
}


void testImplHeatEquationDoubleDirichletBCHostEuler() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_types::MemorySpace;
	using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
	using lss_one_dim_heat_equation_solvers_cuda::implicit_solvers::Implicit1DHeatEquationCUDA;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation: \n\n";
	std::cout << " Using CUDA solvers algorithm with implicit Euler method\n\n";
	std::cout << " Value type: " << typeid(double).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Implicit1DHeatEquationCUDA<double,
		BoundaryConditionType::Dirichlet,
		MemorySpace::Host,
		RealSparseSolverCUDA,
		std::vector,
		std::allocator<double>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 100;
	// initial condition:
	auto initialCondition = [](double x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 0.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<double> solution(Sd + 1, 0.0);
	// initialize solver
	implicit_solver impl_solver(Range<double>(0.0, 1.0), 0.2, Sd, Td);
	// set boundary conditions:
	impl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	impl_solver.solve(solution, ImplicitPDESchemes::Euler);
	// get exact solution:
	auto exact = [](double x, double t, std::size_t n) {
		double const first = 2.0 / PI;
		double sum{};
		double var1{};
		double var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var1 = std::pow(-1.0, i + 1) * std::exp(-1.0*(i*PI)*(i*PI)*t);
			var2 = std::sin(i*PI*x) / i;
			sum += (var1*var2);
		}
		return (first * sum);
	};

	double const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	double benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.2, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}


void testImplHeatEquationFloatDirichletBCHostEuler() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_types::MemorySpace;
	using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
	using lss_one_dim_heat_equation_solvers_cuda::implicit_solvers::Implicit1DHeatEquationCUDA;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation: \n\n";
	std::cout << " Using CUDA solvers algorithm with implicit Euler method\n\n";
	std::cout << " Value type: " << typeid(float).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Implicit1DHeatEquationCUDA<float,
		BoundaryConditionType::Dirichlet,
		MemorySpace::Host,
		RealSparseSolverCUDA,
		std::vector,
		std::allocator<float>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 100;
	// initial condition:
	auto initialCondition = [](float x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 0.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<float> solution(Sd + 1, 0.0);
	// initialize solver
	implicit_solver impl_solver(Range<float>(0.0, 1.0), 0.2, Sd, Td);
	// set boundary conditions:
	impl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	impl_solver.solve(solution, ImplicitPDESchemes::Euler);
	// get exact solution:
	auto exact = [](float x, float t, std::size_t n) {
		float const first = 2.0 / PI;
		float sum{};
		float var1{};
		float var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var1 = std::pow(-1.0, i + 1) * std::exp(-1.0*(i*PI)*(i*PI)*t);
			var2 = std::sin(i*PI*x) / i;
			sum += (var1*var2);
		}
		return (first * sum);
	};

	float const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	float benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.2, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}

void testImplHeatEquationDoubleDirichletBCHostCN() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_types::MemorySpace;
	using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
	using lss_one_dim_heat_equation_solvers_cuda::implicit_solvers::Implicit1DHeatEquationCUDA;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation: \n\n";
	std::cout << " Using CUDA solvers algorithm with implicit Clark-Nicolson method\n\n";
	std::cout << " Value type: " << typeid(double).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Implicit1DHeatEquationCUDA<double,
		BoundaryConditionType::Dirichlet,
		MemorySpace::Host,
		RealSparseSolverCUDA,
		std::vector,
		std::allocator<double>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 100;
	// initial condition:
	auto initialCondition = [](double x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 0.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<double> solution(Sd + 1, 0.0);
	// initialize solver
	implicit_solver impl_solver(Range<double>(0.0, 1.0), 0.2, Sd, Td);
	// set boundary conditions:
	impl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	impl_solver.solve(solution, ImplicitPDESchemes::CrankNicolson);
	// get exact solution:
	auto exact = [](double x, double t, std::size_t n) {
		double const first = 2.0 / PI;
		double sum{};
		double var1{};
		double var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var1 = std::pow(-1.0, i + 1) * std::exp(-1.0*(i*PI)*(i*PI)*t);
			var2 = std::sin(i*PI*x) / i;
			sum += (var1*var2);
		}
		return (first * sum);
	};

	double const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	double benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.2, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}

void testImplHeatEquationFloatDirichletBCHostCN() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_types::MemorySpace;
	using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
	using lss_one_dim_heat_equation_solvers_cuda::implicit_solvers::Implicit1DHeatEquationCUDA;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation: \n\n";
	std::cout << " Using CUDA solvers algorithm with implicit Clark-Nicolson method\n\n";
	std::cout << " Value type: " << typeid(float).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Implicit1DHeatEquationCUDA<float,
		BoundaryConditionType::Dirichlet,
		MemorySpace::Host,
		RealSparseSolverCUDA,
		std::vector,
		std::allocator<float>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 1000;
	// number of time subdivisions:
	std::size_t const Td = 1000;
	// initial condition:
	auto initialCondition = [](float x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 0.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<float> solution(Sd + 1, 0.0);
	// initialize solver
	implicit_solver impl_solver(Range<float>(0.0, 1.0), 0.2, Sd, Td);
	// set boundary conditions:
	impl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	impl_solver.solve(solution, ImplicitPDESchemes::CrankNicolson);
	// get exact solution:
	auto exact = [](float x, float t, std::size_t n) {
		float const first = 2.0 / PI;
		float sum{};
		float var1{};
		float var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var1 = std::pow(-1.0, i + 1) * std::exp(-1.0*(i*PI)*(i*PI)*t);
			var2 = std::sin(i*PI*x) / i;
			sum += (var1*var2);
		}
		return (first * sum);
	};

	float const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	float benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.2, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}


void testImplHeatEquationDirichletBCHost() {
	std::cout << "================================================================================\n";
	std::cout << "===================== Implicit Heat Equation (Dirichlet BC) ====================\n";
	std::cout << "================================================================================\n";

	testImplHeatEquationDoubleDirichletBCHostEuler();
	testImplHeatEquationFloatDirichletBCHostEuler();
	testImplHeatEquationDoubleDirichletBCHostCN();
	testImplHeatEquationFloatDirichletBCHostCN();

	std::cout << "================================================================================\n";
}


void testImplHeatEquationDoubleRobinBCDeviceEuler() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_types::MemorySpace;
	using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
	using lss_one_dim_heat_equation_solvers_cuda::implicit_solvers::Implicit1DHeatEquationCUDA;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation: \n\n";
	std::cout << " Using CUDA solvers algorithm with implicit Euler method\n\n";
	std::cout << " Value type: " << typeid(double).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = 0.5 - (4/(pi*pi))*sum_1^infty{ exp(-((2n-1)*pi)^2*t) *cos((2n-1)*pi*x)/(2n-1)^2}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Implicit1DHeatEquationCUDA<double,
		BoundaryConditionType::Robin,
		MemorySpace::Device,
		RealSparseSolverCUDA,
		std::vector,
		std::allocator<double>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 100;
	// initial condition:
	auto initialCondition = [](double x) {return x; };
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<double> solution(Sd + 1, 0.0);
	// initialize solver
	implicit_solver impl_solver(Range<double>(0.0, 1.0), 0.2, Sd, Td);
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
	auto exact = [](double x, double t, std::size_t n) {
		double const pipi = PI * PI;
		double const first = 4.0 / pipi;
		double sum{};
		double var0{};
		double var1{};
		double var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var0 = (2 * i - 1);
			var1 = std::exp(-1.0*pipi*var0*var0*t);
			var2 = std::cos(var0*PI*x) / (var0*var0);
			sum += (var1*var2);
		}
		return (0.5 - first * sum);
	};

	double const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	double benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.2, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}


void testImplHeatEquationFloatRobinBCDeviceEuler() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_types::MemorySpace;
	using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
	using lss_one_dim_heat_equation_solvers_cuda::implicit_solvers::Implicit1DHeatEquationCUDA;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation: \n\n";
	std::cout << " Using CUDA solvers algorithm with implicit Euler method\n\n";
	std::cout << " Value type: " << typeid(float).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = 0.5 - (4/(pi*pi))*sum_1^infty{ exp(-((2n-1)*pi)^2*t) *cos((2n-1)*pi*x)/(2n-1)^2}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Implicit1DHeatEquationCUDA<float,
		BoundaryConditionType::Robin,
		MemorySpace::Device,
		RealSparseSolverCUDA,
		std::vector,
		std::allocator<float>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 100;
	// initial condition:
	auto initialCondition = [](float x) {return x; };
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<float> solution(Sd + 1, 0.0);
	// initialize solver
	implicit_solver impl_solver(Range<float>(0.0, 1.0), 0.2, Sd, Td);
	// set boundary conditions:
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
	auto exact = [](float x, float t, std::size_t n) {
		float const pipi = PI * PI;
		float const first = 4.0 / pipi;
		float sum{};
		float var0{};
		float var1{};
		float var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var0 = (2 * i - 1);
			var1 = std::exp(-1.0*pipi*var0*var0*t);
			var2 = std::cos(var0*PI*x) / (var0*var0);
			sum += (var1*var2);
		}
		return (0.5 - first * sum);
	};

	float const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	float benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.2, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}

void testImplHeatEquationDoubleRobinBCDeviceCN() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_types::MemorySpace;
	using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
	using lss_one_dim_heat_equation_solvers_cuda::implicit_solvers::Implicit1DHeatEquationCUDA;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation: \n\n";
	std::cout << " Using CUDA solvers algorithm with implicit Clark-Nicolson method\n\n";
	std::cout << " Value type: " << typeid(double).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = 0.5 - (4/(pi*pi))*sum_1^infty{ exp(-((2n-1)*pi)^2*t) *cos((2n-1)*pi*x)/(2n-1)^2}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Implicit1DHeatEquationCUDA<double,
		BoundaryConditionType::Robin,
		MemorySpace::Device,
		RealSparseSolverCUDA,
		std::vector,
		std::allocator<double>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 100;
	// initial condition:
	auto initialCondition = [](double x) {return x; };
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<double> solution(Sd + 1, 0.0);
	// initialize solver
	implicit_solver impl_solver(Range<double>(0.0, 1.0), 0.2, Sd, Td);
	// set boundary conditions:
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
	impl_solver.solve(solution, ImplicitPDESchemes::CrankNicolson);
	// get exact solution:
	auto exact = [](double x, double t, std::size_t n) {
		double const pipi = PI * PI;
		double const first = 4.0 / pipi;
		double sum{};
		double var0{};
		double var1{};
		double var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var0 = (2 * i - 1);
			var1 = std::exp(-1.0*pipi*var0*var0*t);
			var2 = std::cos(var0*PI*x) / (var0*var0);
			sum += (var1*var2);
		}
		return (0.5 - first * sum);
	};

	double const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	double benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.2, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}

void testImplHeatEquationFloatRobinBCDeviceCN() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_types::MemorySpace;
	using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
	using lss_one_dim_heat_equation_solvers_cuda::implicit_solvers::Implicit1DHeatEquationCUDA;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation: \n\n";
	std::cout << " Using CUDA solvers algorithm with implicit Clark-Nicolson method\n\n";
	std::cout << " Value type: " << typeid(float).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = 0.5 - (4/(pi*pi))*sum_1^infty{ exp(-((2n-1)*pi)^2*t) *cos((2n-1)*pi*x)/(2n-1)^2}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Implicit1DHeatEquationCUDA<float,
		BoundaryConditionType::Robin,
		MemorySpace::Device,
		RealSparseSolverCUDA,
		std::vector,
		std::allocator<float>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 1000;
	// number of time subdivisions:
	std::size_t const Td = 1000;
	// initial condition:
	auto initialCondition = [](float x) {return x; };
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<float> solution(Sd + 1, 0.0);
	// initialize solver
	implicit_solver impl_solver(Range<float>(0.0, 1.0), 0.2, Sd, Td);
	// set boundary conditions:
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
	impl_solver.solve(solution, ImplicitPDESchemes::CrankNicolson);
	// get exact solution:
	auto exact = [](float x, float t, std::size_t n) {
		float const pipi = PI * PI;
		float const first = 4.0 / pipi;
		float sum{};
		float var0{};
		float var1{};
		float var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var0 = (2 * i - 1);
			var1 = std::exp(-1.0*pipi*var0*var0*t);
			var2 = std::cos(var0*PI*x) / (var0*var0);
			sum += (var1*var2);
		}
		return (0.5 - first * sum);
	};

	float const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	float benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.2, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}


void testImplHeatEquationRobinBCDevice() {

	std::cout << "================================================================================\n";
	std::cout << "======================= Implicit Heat Equation (Robin BC) ======================\n";
	std::cout << "================================================================================\n";

	testImplHeatEquationDoubleRobinBCDeviceEuler();
	testImplHeatEquationFloatRobinBCDeviceEuler();
	testImplHeatEquationDoubleRobinBCDeviceCN();
	testImplHeatEquationFloatRobinBCDeviceCN();

	std::cout << "================================================================================\n";
}

// ================================================================================================================
// ============================ Heat problem with homogeneous boundary conditions and source ======================
// ================================================================================================================


void testImplHeatEquationSourceFloatDirichletBCDeviceEuler() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_types::MemorySpace;
	using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
	using lss_one_dim_heat_equation_solvers_cuda::implicit_solvers::Implicit1DHeatEquationCUDA;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation with source: \n\n";
	std::cout << " Using Euler algorithm on CUDA device with implicit Euler method\n\n";
	std::cout << " Value type: " << typeid(float).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = 1, x in <0,1> \n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquationCUDA
	typedef Implicit1DHeatEquationCUDA<float,
		BoundaryConditionType::Dirichlet,
		MemorySpace::Device,
		RealSparseSolverCUDA,
		std::vector,
		std::allocator<float>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 100;
	// initial condition:
	auto initialCondition = [](float x) {return 1.0f; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0f, 0.0f);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<float> solution(Sd + 1, 0.0f);
	// initialize solver
	implicit_solver impl_solver(Range<float>(0.0f, 1.0f), 0.5f, Sd, Td);
	// set boundary conditions:
	impl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// set heat source: 
	impl_solver.setHeatSource([](float x, float t) {return x; });
	// get the solution:
	impl_solver.solve(solution, ImplicitPDESchemes::Euler);
	// get exact solution:
	auto exact = [](float x, float t, std::size_t n) {
		float sum{};
		float q_n{};
		float f_n{};
		float lam_n{};
		float lam_2{};
		float var1{};
		for (std::size_t i = 1; i <= n; ++i) {
			q_n = (2.0 / (i*PI))*std::pow(-1.0, i + 1);
			f_n = (2.0 / (i*PI))*(1.0 - std::pow(-1.0, i));
			lam_n = i * PI;
			lam_2 = lam_n * lam_n;
			var1 = (q_n / lam_2 + (f_n - (q_n / lam_2))*std::exp(-1.0*lam_2*t))*std::sin(i*PI*x);
			sum += var1;
		}
		return sum;
	};

	float const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	float benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.5, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}

void testImplHeatEquationSourceDoubleDirichletBCDeviceEuler() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_types::MemorySpace;
	using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
	using lss_one_dim_heat_equation_solvers_cuda::implicit_solvers::Implicit1DHeatEquationCUDA;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation with source: \n\n";
	std::cout << " Using Euler algorithm on CUDA device with implicit Euler method\n\n";
	std::cout << " Value type: " << typeid(double).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = 1, x in <0,1> \n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquationCUDA
	typedef Implicit1DHeatEquationCUDA<double,
		BoundaryConditionType::Dirichlet,
		MemorySpace::Device,
		RealSparseSolverCUDA,
		std::vector,
		std::allocator<double>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 100;
	// initial condition:
	auto initialCondition = [](double x) {return 1.0f; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0f, 0.0f);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<double> solution(Sd + 1, 0.0f);
	// initialize solver
	implicit_solver impl_solver(Range<double>(0.0f, 1.0f), 0.5f, Sd, Td);
	// set boundary conditions:
	impl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// set heat source: 
	impl_solver.setHeatSource([](double x, double t) {return x; });
	// get the solution:
	impl_solver.solve(solution, ImplicitPDESchemes::Euler);
	// get exact solution:
	auto exact = [](double x, double t, std::size_t n) {
		double sum{};
		double q_n{};
		double f_n{};
		double lam_n{};
		double lam_2{};
		double var1{};
		for (std::size_t i = 1; i <= n; ++i) {
			q_n = (2.0 / (i*PI))*std::pow(-1.0, i + 1);
			f_n = (2.0 / (i*PI))*(1.0 - std::pow(-1.0, i));
			lam_n = i * PI;
			lam_2 = lam_n * lam_n;
			var1 = (q_n / lam_2 + (f_n - (q_n / lam_2))*std::exp(-1.0*lam_2*t))*std::sin(i*PI*x);
			sum += var1;
		}
		return sum;
	};

	double const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	double benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.5, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}


void testImplHeatEquationSourceFloatDirichletBCDeviceCN() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_types::MemorySpace;
	using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
	using lss_one_dim_heat_equation_solvers_cuda::implicit_solvers::Implicit1DHeatEquationCUDA;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation with source: \n\n";
	std::cout << " Using Euler algorithm on CUDA device with implicit Crank-Nicolson method\n\n";
	std::cout << " Value type: " << typeid(float).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = 1, x in <0,1> \n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquationCUDA
	typedef Implicit1DHeatEquationCUDA<float,
		BoundaryConditionType::Dirichlet,
		MemorySpace::Device,
		RealSparseSolverCUDA,
		std::vector,
		std::allocator<float>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 100;
	// initial condition:
	auto initialCondition = [](float x) {return 1.0f; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0f, 0.0f);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<float> solution(Sd + 1, 0.0f);
	// initialize solver
	implicit_solver impl_solver(Range<float>(0.0f, 1.0f), 0.5f, Sd, Td);
	// set boundary conditions:
	impl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// set heat source: 
	impl_solver.setHeatSource([](float x, float t) {return x; });
	// get the solution:
	impl_solver.solve(solution, ImplicitPDESchemes::CrankNicolson);
	// get exact solution:
	auto exact = [](float x, float t, std::size_t n) {
		float sum{};
		float q_n{};
		float f_n{};
		float lam_n{};
		float lam_2{};
		float var1{};
		for (std::size_t i = 1; i <= n; ++i) {
			q_n = (2.0 / (i*PI))*std::pow(-1.0, i + 1);
			f_n = (2.0 / (i*PI))*(1.0 - std::pow(-1.0, i));
			lam_n = i * PI;
			lam_2 = lam_n * lam_n;
			var1 = (q_n / lam_2 + (f_n - (q_n / lam_2))*std::exp(-1.0*lam_2*t))*std::sin(i*PI*x);
			sum += var1;
		}
		return sum;
	};

	float const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	float benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.5, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}

void testImplHeatEquationSourceDoubleDirichletBCDeviceCN() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_types::MemorySpace;
	using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
	using lss_one_dim_heat_equation_solvers_cuda::implicit_solvers::Implicit1DHeatEquationCUDA;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation with source: \n\n";
	std::cout << " Using Euler algorithm on CUDA device with implicit Crank-Nicolson method\n\n";
	std::cout << " Value type: " << typeid(double).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = 1, x in <0,1> \n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquationCUDA
	typedef Implicit1DHeatEquationCUDA<double,
		BoundaryConditionType::Dirichlet,
		MemorySpace::Device,
		RealSparseSolverCUDA,
		std::vector,
		std::allocator<double>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 100;
	// initial condition:
	auto initialCondition = [](double x) {return 1.0f; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0f, 0.0f);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<double> solution(Sd + 1, 0.0f);
	// initialize solver
	implicit_solver impl_solver(Range<double>(0.0f, 1.0f), 0.5f, Sd, Td);
	// set boundary conditions:
	impl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// set heat source: 
	impl_solver.setHeatSource([](double x, double t) {return x; });
	// get the solution:
	impl_solver.solve(solution, ImplicitPDESchemes::CrankNicolson);
	// get exact solution:
	auto exact = [](double x, double t, std::size_t n) {
		double sum{};
		double q_n{};
		double f_n{};
		double lam_n{};
		double lam_2{};
		double var1{};
		for (std::size_t i = 1; i <= n; ++i) {
			q_n = (2.0 / (i*PI))*std::pow(-1.0, i + 1);
			f_n = (2.0 / (i*PI))*(1.0 - std::pow(-1.0, i));
			lam_n = i * PI;
			lam_2 = lam_n * lam_n;
			var1 = (q_n / lam_2 + (f_n - (q_n / lam_2))*std::exp(-1.0*lam_2*t))*std::sin(i*PI*x);
			sum += var1;
		}
		return sum;
	};

	double const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	double benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.5, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}

void testImplHeatEquationSourceFloatDirichletBCHostEuler() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_types::MemorySpace;
	using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
	using lss_one_dim_heat_equation_solvers_cuda::implicit_solvers::Implicit1DHeatEquationCUDA;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation with source: \n\n";
	std::cout << " Using Euler algorithm on CUDA host with implicit Euler method\n\n";
	std::cout << " Value type: " << typeid(float).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = 1, x in <0,1> \n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquationCUDA
	typedef Implicit1DHeatEquationCUDA<float,
		BoundaryConditionType::Dirichlet,
		MemorySpace::Host,
		RealSparseSolverCUDA,
		std::vector,
		std::allocator<float>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 100;
	// initial condition:
	auto initialCondition = [](float x) {return 1.0f; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0f, 0.0f);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<float> solution(Sd + 1, 0.0f);
	// initialize solver
	implicit_solver impl_solver(Range<float>(0.0f, 1.0f), 0.5f, Sd, Td);
	// set boundary conditions:
	impl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// set heat source: 
	impl_solver.setHeatSource([](float x, float t) {return x; });
	// get the solution:
	impl_solver.solve(solution, ImplicitPDESchemes::Euler);
	// get exact solution:
	auto exact = [](float x, float t, std::size_t n) {
		float sum{};
		float q_n{};
		float f_n{};
		float lam_n{};
		float lam_2{};
		float var1{};
		for (std::size_t i = 1; i <= n; ++i) {
			q_n = (2.0 / (i*PI))*std::pow(-1.0, i + 1);
			f_n = (2.0 / (i*PI))*(1.0 - std::pow(-1.0, i));
			lam_n = i * PI;
			lam_2 = lam_n * lam_n;
			var1 = (q_n / lam_2 + (f_n - (q_n / lam_2))*std::exp(-1.0*lam_2*t))*std::sin(i*PI*x);
			sum += var1;
		}
		return sum;
	};

	float const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	float benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.5, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}

void testImplHeatEquationSourceDoubleDirichletBCHostEuler() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_types::MemorySpace;
	using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
	using lss_one_dim_heat_equation_solvers_cuda::implicit_solvers::Implicit1DHeatEquationCUDA;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation with source: \n\n";
	std::cout << " Using Euler algorithm on CUDA host with implicit Euler method\n\n";
	std::cout << " Value type: " << typeid(double).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = 1, x in <0,1> \n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquationCUDA
	typedef Implicit1DHeatEquationCUDA<double,
		BoundaryConditionType::Dirichlet,
		MemorySpace::Host,
		RealSparseSolverCUDA,
		std::vector,
		std::allocator<double>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 1000;
	// number of time subdivisions:
	std::size_t const Td = 1000;
	// initial condition:
	auto initialCondition = [](double x) {return 1.0f; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0f, 0.0f);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<double> solution(Sd + 1, 0.0f);
	// initialize solver
	implicit_solver impl_solver(Range<double>(0.0f, 1.0f), 0.5f, Sd, Td);
	// set boundary conditions:
	impl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// set heat source: 
	impl_solver.setHeatSource([](double x, double t) {return x; });
	// get the solution:
	impl_solver.solve(solution, ImplicitPDESchemes::Euler);
	// get exact solution:
	auto exact = [](double x, double t, std::size_t n) {
		double sum{};
		double q_n{};
		double f_n{};
		double lam_n{};
		double lam_2{};
		double var1{};
		for (std::size_t i = 1; i <= n; ++i) {
			q_n = (2.0 / (i*PI))*std::pow(-1.0, i + 1);
			f_n = (2.0 / (i*PI))*(1.0 - std::pow(-1.0, i));
			lam_n = i * PI;
			lam_2 = lam_n * lam_n;
			var1 = (q_n / lam_2 + (f_n - (q_n / lam_2))*std::exp(-1.0*lam_2*t))*std::sin(i*PI*x);
			sum += var1;
		}
		return sum;
	};

	double const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	double benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.5, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}


void testImplHeatEquationSourceFloatDirichletBCHostCN() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_types::MemorySpace;
	using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
	using lss_one_dim_heat_equation_solvers_cuda::implicit_solvers::Implicit1DHeatEquationCUDA;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation with source: \n\n";
	std::cout << " Using Euler algorithm on CUDA host with implicit Crank-Nicolson method\n\n";
	std::cout << " Value type: " << typeid(float).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = 1, x in <0,1> \n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquationCUDA
	typedef Implicit1DHeatEquationCUDA<float,
		BoundaryConditionType::Dirichlet,
		MemorySpace::Host,
		RealSparseSolverCUDA,
		std::vector,
		std::allocator<float>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 1000;
	// number of time subdivisions:
	std::size_t const Td = 1000;
	// initial condition:
	auto initialCondition = [](float x) {return 1.0f; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0f, 0.0f);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<float> solution(Sd + 1, 0.0f);
	// initialize solver
	implicit_solver impl_solver(Range<float>(0.0f, 1.0f), 0.5f, Sd, Td);
	// set boundary conditions:
	impl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// set heat source: 
	impl_solver.setHeatSource([](float x, float t) {return x; });
	// get the solution:
	impl_solver.solve(solution, ImplicitPDESchemes::CrankNicolson);
	// get exact solution:
	auto exact = [](float x, float t, std::size_t n) {
		float sum{};
		float q_n{};
		float f_n{};
		float lam_n{};
		float lam_2{};
		float var1{};
		for (std::size_t i = 1; i <= n; ++i) {
			q_n = (2.0 / (i*PI))*std::pow(-1.0, i + 1);
			f_n = (2.0 / (i*PI))*(1.0 - std::pow(-1.0, i));
			lam_n = i * PI;
			lam_2 = lam_n * lam_n;
			var1 = (q_n / lam_2 + (f_n - (q_n / lam_2))*std::exp(-1.0*lam_2*t))*std::sin(i*PI*x);
			sum += var1;
		}
		return sum;
	};

	float const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	float benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.5, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}

void testImplHeatEquationSourceDoubleDirichletBCHostCN() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_types::MemorySpace;
	using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
	using lss_one_dim_heat_equation_solvers_cuda::implicit_solvers::Implicit1DHeatEquationCUDA;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation with source: \n\n";
	std::cout << " Using Euler algorithm on CUDA host with implicit Crank-Nicolson method\n\n";
	std::cout << " Value type: " << typeid(double).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = 1, x in <0,1> \n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquationCUDA
	typedef Implicit1DHeatEquationCUDA<double,
		BoundaryConditionType::Dirichlet,
		MemorySpace::Host,
		RealSparseSolverCUDA,
		std::vector,
		std::allocator<double>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 1000;
	// number of time subdivisions:
	std::size_t const Td = 1000;
	// initial condition:
	auto initialCondition = [](double x) {return 1.0f; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0f, 0.0f);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<double> solution(Sd + 1, 0.0f);
	// initialize solver
	implicit_solver impl_solver(Range<double>(0.0f, 1.0f), 0.5f, Sd, Td);
	// set boundary conditions:
	impl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// set heat source: 
	impl_solver.setHeatSource([](double x, double t) {return x; });
	// get the solution:
	impl_solver.solve(solution, ImplicitPDESchemes::CrankNicolson);
	// get exact solution:
	auto exact = [](double x, double t, std::size_t n) {
		double sum{};
		double q_n{};
		double f_n{};
		double lam_n{};
		double lam_2{};
		double var1{};
		for (std::size_t i = 1; i <= n; ++i) {
			q_n = (2.0 / (i*PI))*std::pow(-1.0, i + 1);
			f_n = (2.0 / (i*PI))*(1.0 - std::pow(-1.0, i));
			lam_n = i * PI;
			lam_2 = lam_n * lam_n;
			var1 = (q_n / lam_2 + (f_n - (q_n / lam_2))*std::exp(-1.0*lam_2*t))*std::sin(i*PI*x);
			sum += var1;
		}
		return sum;
	};

	double const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	double benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.5, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}


void testImplHeatEquationSourceDirichletBC() {
	std::cout << "================================================================================\n";
	std::cout << "================= Implicit Heat Equation with source (Dirichlet BC) ============\n";
	std::cout << "================================================================================\n";

	testImplHeatEquationSourceFloatDirichletBCDeviceEuler();
	testImplHeatEquationSourceDoubleDirichletBCDeviceEuler();
	testImplHeatEquationSourceFloatDirichletBCDeviceCN();
	testImplHeatEquationSourceDoubleDirichletBCDeviceCN();
	testImplHeatEquationSourceFloatDirichletBCHostEuler();
	testImplHeatEquationSourceDoubleDirichletBCHostEuler();
	testImplHeatEquationSourceFloatDirichletBCHostCN();
	testImplHeatEquationSourceDoubleDirichletBCHostCN();


	std::cout << "================================================================================\n";
}





// ================================================================================================================
// ================================ Heat problem with non-homogeneous boundary conditions =========================
// ================================================================================================================

void testImplNonHomHeatEquationDoubleDirichletBCDeviceEuler() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_types::MemorySpace;
	using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
	using lss_one_dim_heat_equation_solvers_cuda::implicit_solvers::Implicit1DHeatEquationCUDA;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation with non-hom BC: \n\n";
	std::cout << " Using CUDA solvers algorithm with implicit Euler method\n\n";
	std::cout << " Value type: " << typeid(double).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = 0, U(1,t) = 100, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = 100*x + (198/pi)*sum_0^infty{ (-1)^(n)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Implicit1DHeatEquationCUDA<double,
		BoundaryConditionType::Dirichlet,
		MemorySpace::Device,
		RealSparseSolverCUDA,
		std::vector,
		std::allocator<double>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 1000;
	// number of time subdivisions:
	std::size_t const Td = 1000;
	// initial condition:
	auto initialCondition = [](double x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 100.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<double> solution(Sd + 1, 0.0);
	// initialize solver
	implicit_solver impl_solver(Range<double>(0.0, 1.0), 0.2, Sd, Td);
	// set boundary conditions:
	impl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	impl_solver.solve(solution, ImplicitPDESchemes::Euler);
	// get exact solution:
	auto exact = [](double x, double t, std::size_t n) {
		double const first = 198.0 / PI;
		double sum{};
		double var1{};
		double var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var1 = std::pow(-1.0, i) * std::exp(-1.0*(i*PI)*(i*PI)*t);
			var2 = std::sin(i*PI*x) / i;
			sum += (var1*var2);
		}
		return (100*x + first * sum);
	};

	double const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	double benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.2, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}


void testImplNonHomHeatEquationFloatDirichletBCDeviceEuler() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_types::MemorySpace;
	using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
	using lss_one_dim_heat_equation_solvers_cuda::implicit_solvers::Implicit1DHeatEquationCUDA;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation with non-hom BC: \n\n";
	std::cout << " Using CUDA solvers algorithm with implicit Euler method\n\n";
	std::cout << " Value type: " << typeid(float).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = 0, U(1,t) = 100, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = 100*x +  (198/pi)*sum_0^infty{ (-1)^(n)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Implicit1DHeatEquationCUDA<float,
		BoundaryConditionType::Dirichlet,
		MemorySpace::Device,
		RealSparseSolverCUDA,
		std::vector,
		std::allocator<float>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 1000;
	// number of time subdivisions:
	std::size_t const Td = 1000;
	// initial condition:
	auto initialCondition = [](float x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 100.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<float> solution(Sd + 1, 0.0);
	// initialize solver
	implicit_solver impl_solver(Range<float>(0.0, 1.0), 0.2, Sd, Td);
	// set boundary conditions:
	impl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	impl_solver.solve(solution, ImplicitPDESchemes::Euler);
	// get exact solution:
	auto exact = [](float x, float t, std::size_t n) {
		float const first = 198.0 / PI;
		float sum{};
		float var1{};
		float var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var1 = std::pow(-1.0, i ) * std::exp(-1.0*(i*PI)*(i*PI)*t);
			var2 = std::sin(i*PI*x) / i;
			sum += (var1*var2);
		}
		return (100*x + first * sum);
	};

	float const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	float benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.2, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}

void testImplNonHomHeatEquationDoubleDirichletBCDeviceCN() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_types::MemorySpace;
	using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
	using lss_one_dim_heat_equation_solvers_cuda::implicit_solvers::Implicit1DHeatEquationCUDA;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation with non-hom BC: \n\n";
	std::cout << " Using CUDA solvers algorithm with implicit Clark-Nicolson method\n\n";
	std::cout << " Value type: " << typeid(double).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = 0, U(1,t) = 100, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = 100*x +  (198/pi)*sum_0^infty{ (-1)^(n)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Implicit1DHeatEquationCUDA<double,
		BoundaryConditionType::Dirichlet,
		MemorySpace::Device,
		RealSparseSolverCUDA,
		std::vector,
		std::allocator<double>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 1000;
	// number of time subdivisions:
	std::size_t const Td = 1000;
	// initial condition:
	auto initialCondition = [](double x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 100.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<double> solution(Sd + 1, 0.0);
	// initialize solver
	implicit_solver impl_solver(Range<double>(0.0, 1.0), 0.2, Sd, Td);
	// set boundary conditions:
	impl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	impl_solver.solve(solution, ImplicitPDESchemes::CrankNicolson);
	// get exact solution:
	auto exact = [](double x, double t, std::size_t n) {
		double const first = 198.0 / PI;
		double sum{};
		double var1{};
		double var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var1 = std::pow(-1.0, i) * std::exp(-1.0*(i*PI)*(i*PI)*t);
			var2 = std::sin(i*PI*x) / i;
			sum += (var1*var2);
		}
		return (100.0*x + first * sum);
	};

	double const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	double benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.2, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}

void testImplNonHomHeatEquationFloatDirichletBCDeviceCN() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_types::MemorySpace;
	using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
	using lss_one_dim_heat_equation_solvers_cuda::implicit_solvers::Implicit1DHeatEquationCUDA;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation with non-hom BC: \n\n";
	std::cout << " Using CUDA solvers algorithm with implicit Clark-Nicolson method\n\n";
	std::cout << " Value type: " << typeid(float).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = 0, U(1,t) = 100, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = 100*x + (198/pi)*sum_0^infty{ (-1)^(n)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Implicit1DHeatEquationCUDA<float,
		BoundaryConditionType::Dirichlet,
		MemorySpace::Device,
		RealSparseSolverCUDA,
		std::vector,
		std::allocator<float>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 1000;
	// number of time subdivisions:
	std::size_t const Td = 1000;
	// initial condition:
	auto initialCondition = [](float x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 100.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<float> solution(Sd + 1, 0.0);
	// initialize solver
	implicit_solver impl_solver(Range<float>(0.0, 1.0), 0.2, Sd, Td);
	// set boundary conditions:
	impl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	impl_solver.solve(solution, ImplicitPDESchemes::CrankNicolson);
	// get exact solution:
	auto exact = [](float x, float t, std::size_t n) {
		float const first = 198.0 / PI;
		float sum{};
		float var1{};
		float var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var1 = std::pow(-1.0, i) * std::exp(-1.0*(i*PI)*(i*PI)*t);
			var2 = std::sin(i*PI*x) / i;
			sum += (var1*var2);
		}
		return (100.0*x + first * sum);
	};

	float const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	float benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.2, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}


void testImplNonHomHeatEquationDirichletBCDevice() {
	std::cout << "================================================================================\n";
	std::cout << "======== Implicit Heat Equation (with non-homogeneous Dirichlet BC) ============\n";
	std::cout << "================================================================================\n";

	testImplNonHomHeatEquationDoubleDirichletBCDeviceEuler();
	testImplNonHomHeatEquationFloatDirichletBCDeviceEuler();
	testImplNonHomHeatEquationDoubleDirichletBCDeviceCN();
	testImplNonHomHeatEquationFloatDirichletBCDeviceCN();

	std::cout << "================================================================================\n";
}

void testImplNonHomHeatEquationDoubleDirichletBCHostEuler() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_types::MemorySpace;
	using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
	using lss_one_dim_heat_equation_solvers_cuda::implicit_solvers::Implicit1DHeatEquationCUDA;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation with non-hom BC: \n\n";
	std::cout << " Using CUDA solvers algorithm with implicit Euler method\n\n";
	std::cout << " Value type: " << typeid(double).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = 0, U(1,t) = 100, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = 100*x + (198/pi)*sum_0^infty{ (-1)^(n)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Implicit1DHeatEquationCUDA<double,
		BoundaryConditionType::Dirichlet,
		MemorySpace::Host,
		RealSparseSolverCUDA,
		std::vector,
		std::allocator<double>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 100;
	// initial condition:
	auto initialCondition = [](double x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 100.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<double> solution(Sd + 1, 0.0);
	// initialize solver
	implicit_solver impl_solver(Range<double>(0.0, 1.0), 0.2, Sd, Td);
	// set boundary conditions:
	impl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	impl_solver.solve(solution, ImplicitPDESchemes::Euler);
	// get exact solution:
	auto exact = [](double x, double t, std::size_t n) {
		double const first = 198.0 / PI;
		double sum{};
		double var1{};
		double var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var1 = std::pow(-1.0, i) * std::exp(-1.0*(i*PI)*(i*PI)*t);
			var2 = std::sin(i*PI*x) / i;
			sum += (var1*var2);
		}
		return (100.0*x + first * sum);
	};

	double const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	double benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.2, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}


void testImplNonHomHeatEquationFloatDirichletBCHostEuler() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_types::MemorySpace;
	using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
	using lss_one_dim_heat_equation_solvers_cuda::implicit_solvers::Implicit1DHeatEquationCUDA;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation with non-hom BC: \n\n";
	std::cout << " Using CUDA solvers algorithm with implicit Euler method\n\n";
	std::cout << " Value type: " << typeid(float).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = 0, U(1,t) = 100, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = 100*x + (198/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Implicit1DHeatEquationCUDA<float,
		BoundaryConditionType::Dirichlet,
		MemorySpace::Host,
		RealSparseSolverCUDA,
		std::vector,
		std::allocator<float>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 100;
	// initial condition:
	auto initialCondition = [](float x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 100.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<float> solution(Sd + 1, 0.0);
	// initialize solver
	implicit_solver impl_solver(Range<float>(0.0, 1.0), 0.2, Sd, Td);
	// set boundary conditions:
	impl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	impl_solver.solve(solution, ImplicitPDESchemes::Euler);
	// get exact solution:
	auto exact = [](float x, float t, std::size_t n) {
		float const first = 198.0 / PI;
		float sum{};
		float var1{};
		float var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var1 = std::pow(-1.0, i ) * std::exp(-1.0*(i*PI)*(i*PI)*t);
			var2 = std::sin(i*PI*x) / i;
			sum += (var1*var2);
		}
		return (100.0*x + first * sum);
	};

	float const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	float benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.2, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}

void testImplNonHomHeatEquationDoubleDirichletBCHostCN() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_types::MemorySpace;
	using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
	using lss_one_dim_heat_equation_solvers_cuda::implicit_solvers::Implicit1DHeatEquationCUDA;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation with non-hom BC: \n\n";
	std::cout << " Using CUDA solvers algorithm with implicit Clark-Nicolson method\n\n";
	std::cout << " Value type: " << typeid(double).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = 0, U(1,t) = 100, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = 100.0*x + (198/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Implicit1DHeatEquationCUDA<double,
		BoundaryConditionType::Dirichlet,
		MemorySpace::Host,
		RealSparseSolverCUDA,
		std::vector,
		std::allocator<double>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 100;
	// initial condition:
	auto initialCondition = [](double x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 100.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<double> solution(Sd + 1, 0.0);
	// initialize solver
	implicit_solver impl_solver(Range<double>(0.0, 1.0), 0.2, Sd, Td);
	// set boundary conditions:
	impl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	impl_solver.solve(solution, ImplicitPDESchemes::CrankNicolson);
	// get exact solution:
	auto exact = [](double x, double t, std::size_t n) {
		double const first = 198.0 / PI;
		double sum{};
		double var1{};
		double var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var1 = std::pow(-1.0, i ) * std::exp(-1.0*(i*PI)*(i*PI)*t);
			var2 = std::sin(i*PI*x) / i;
			sum += (var1*var2);
		}
		return (100.0*x + first * sum);
	};

	double const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	double benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.2, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}

void testImplNonHomHeatEquationFloatDirichletBCHostCN() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_types::MemorySpace;
	using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
	using lss_one_dim_heat_equation_solvers_cuda::implicit_solvers::Implicit1DHeatEquationCUDA;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation with non-hom BC: \n\n";
	std::cout << " Using CUDA solvers algorithm with implicit Clark-Nicolson method\n\n";
	std::cout << " Value type: " << typeid(float).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = 0, U(1,t) = 100, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = 100.0*x + (198/pi)*sum_0^infty{ (-1)^(n)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Implicit1DHeatEquationCUDA<float,
		BoundaryConditionType::Dirichlet,
		MemorySpace::Host,
		RealSparseSolverCUDA,
		std::vector,
		std::allocator<float>> implicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 100;
	// initial condition:
	auto initialCondition = [](float x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 100.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<float> solution(Sd + 1, 0.0);
	// initialize solver
	implicit_solver impl_solver(Range<float>(0.0, 1.0), 0.2, Sd, Td);
	// set boundary conditions:
	impl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	impl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	impl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	impl_solver.solve(solution, ImplicitPDESchemes::CrankNicolson);
	// get exact solution:
	auto exact = [](float x, float t, std::size_t n) {
		float const first = 198.0 / PI;
		float sum{};
		float var1{};
		float var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var1 = std::pow(-1.0, i) * std::exp(-1.0*(i*PI)*(i*PI)*t);
			var2 = std::sin(i*PI*x) / i;
			sum += (var1*var2);
		}
		return (100.0*x + first * sum);
	};

	float const h = impl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	float benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.2, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}


void testImplNonHomHeatEquationDirichletBCHost() {
	std::cout << "================================================================================\n";
	std::cout << "========== Implicit Heat Equation (with non-homogeneous Dirichlet BC) ==========\n";
	std::cout << "================================================================================\n";

	testImplNonHomHeatEquationDoubleDirichletBCHostEuler();
	testImplNonHomHeatEquationFloatDirichletBCHostEuler();
	testImplNonHomHeatEquationDoubleDirichletBCHostCN();
	testImplNonHomHeatEquationFloatDirichletBCHostCN();

	std::cout << "================================================================================\n";
}



// ================================================================================================================
// =========================================== EXPLICIT SOLVERS ===================================================
// ================================================================================================================

// ================================================================================================================
// ========================== Heat problem with homogeneous Dirichlet boundary conditions =========================
// ================================================================================================================

void testExplHeatEquationDoubleDirichletBCDeviceEuler() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_one_dim_heat_equation_solvers_cuda::explicit_solvers::Explicit1DHeatEquationCUDA;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation: \n\n";
	std::cout << " Using CUDA  explicit Euler method\n\n";
	std::cout << " Value type: " << typeid(double).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Explicit1DHeatEquationCUDA<double,
		BoundaryConditionType::Dirichlet,
		std::vector,
		std::allocator<double>> explicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 5000;
	// initial condition:
	auto initialCondition = [](double x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 0.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<double> solution(Sd + 1, 0.0);
	// initialize solver
	explicit_solver expl_solver(Range<double>(0.0, 1.0), 0.2, Sd, Td);
	// set boundary conditions:
	expl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	expl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	expl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	expl_solver.solve(solution);
	// get exact solution:
	auto exact = [](double x, double t, std::size_t n) {
		double const first = 2.0 / PI;
		double sum{};
		double var1{};
		double var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var1 = std::pow(-1.0, i + 1) * std::exp(-1.0*(i*PI)*(i*PI)*t);
			var2 = std::sin(i*PI*x) / i;
			sum += (var1*var2);
		}
		return (first * sum);
	};

	double const h = expl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	double benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.2, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}


void testExplHeatEquationFloatDirichletBCDeviceEuler() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_one_dim_heat_equation_solvers_cuda::explicit_solvers::Explicit1DHeatEquationCUDA;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation: \n\n";
	std::cout << " Using CUDA  explicit Euler method\n\n";
	std::cout << " Value type: " << typeid(double).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = (2/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Explicit1DHeatEquationCUDA<float,
		BoundaryConditionType::Dirichlet,
		std::vector,
		std::allocator<float>> explicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 5000;
	// initial condition:
	auto initialCondition = [](float x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 0.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<float> solution(Sd + 1, 0.0);
	// initialize solver
	explicit_solver expl_solver(Range<float>(0.0, 1.0), 0.2, Sd, Td);
	// set boundary conditions:
	expl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	expl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	expl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	expl_solver.solve(solution);
	// get exact solution:
	auto exact = [](float x, float t, std::size_t n) {
		float const first = 2.0 / PI;
		float sum{};
		float var1{};
		float var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var1 = std::pow(-1.0, i + 1) * std::exp(-1.0*(i*PI)*(i*PI)*t);
			var2 = std::sin(i*PI*x) / i;
			sum += (var1*var2);
		}
		return (first * sum);
	};

	float const h = expl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	float benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.2, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}



void testExplHeatEquationDirichletBCDevice() {
	std::cout << "================================================================================\n";
	std::cout << "===================== Explicit Heat Equation (Dirichlet BC) ====================\n";
	std::cout << "================================================================================\n";

	testExplHeatEquationDoubleDirichletBCDeviceEuler();
	testExplHeatEquationFloatDirichletBCDeviceEuler();

	std::cout << "================================================================================\n";
}



// ================================================================================================================
// ========================== Heat problem with nonhomogeneous Dirichlet boundary conditions ======================
// ================================================================================================================

void testExplNonHomHeatEquationDoubleDirichletBCDeviceEuler() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_one_dim_heat_equation_solvers_cuda::explicit_solvers::Explicit1DHeatEquationCUDA;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation with non-hom BC: \n\n";
	std::cout << " Using CUDA implicit Euler method\n\n";
	std::cout << " Value type: " << typeid(double).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = 0, U(1,t) = 100, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = 100*x + (198/pi)*sum_0^infty{ (-1)^(n)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Explicit1DHeatEquationCUDA<double,
		BoundaryConditionType::Dirichlet,
		std::vector,
		std::allocator<double>> explicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 5000;
	// initial condition:
	auto initialCondition = [](double x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 100.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<double> solution(Sd + 1, 0.0);
	// initialize solver
	explicit_solver expl_solver(Range<double>(0.0, 1.0), 0.2, Sd, Td);
	// set boundary conditions:
	expl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	expl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	expl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	expl_solver.solve(solution);
	// get exact solution:
	auto exact = [](double x, double t, std::size_t n) {
		double const first = 198.0 / PI;
		double sum{};
		double var1{};
		double var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var1 = std::pow(-1.0, i) * std::exp(-1.0*(i*PI)*(i*PI)*t);
			var2 = std::sin(i*PI*x) / i;
			sum += (var1*var2);
		}
		return (100.0*x + first * sum);
	};

	double const h = expl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	double benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.2, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}


void testExplNonHomHeatEquationFloatDirichletBCDeviceEuler() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_one_dim_heat_equation_solvers_cuda::explicit_solvers::Explicit1DHeatEquationCUDA;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation with non-hom BC: \n\n";
	std::cout << " Using CUDA solvers algorithm with implicit Euler method\n\n";
	std::cout << " Value type: " << typeid(float).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = 0, U(1,t) = 100, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = 100*x + (198/pi)*sum_0^infty{ (-1)^(n+1)*exp(-(n*pi)^2*t) *sin(n*pi*x)/n}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Explicit1DHeatEquationCUDA<float,
		BoundaryConditionType::Dirichlet,
		std::vector,
		std::allocator<float>> explicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 5000;
	// initial condition:
	auto initialCondition = [](float x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 100.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<float> solution(Sd + 1, 0.0);
	// initialize solver
	explicit_solver expl_solver(Range<float>(0.0, 1.0), 0.2, Sd, Td);
	// set boundary conditions:
	expl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	expl_solver.setInitialCondition(initialCondition);
	// set thermal diffusivity (C^2 in PDE)
	expl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	expl_solver.solve(solution);
	// get exact solution:
	auto exact = [](float x, float t, std::size_t n) {
		float const first = 198.0 / PI;
		float sum{};
		float var1{};
		float var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var1 = std::pow(-1.0, i) * std::exp(-1.0*(i*PI)*(i*PI)*t);
			var2 = std::sin(i*PI*x) / i;
			sum += (var1*var2);
		}
		return (100.0*x + first * sum);
	};

	float const h = expl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	float benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.2, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}


void testExplNonHomHeatEquationDirichletBCDevice() {
	std::cout << "================================================================================\n";
	std::cout << "========= Explicit Heat Equation (with non-homogeneous Dirichlet BC) ===========\n";
	std::cout << "================================================================================\n";

	testExplNonHomHeatEquationDoubleDirichletBCDeviceEuler();
	testExplNonHomHeatEquationFloatDirichletBCDeviceEuler();

	std::cout << "================================================================================\n";
}

// ================================================================================================================
// ========================= Heat problem with homogeneous boundary conditions and source =========================
// ================================================================================================================


void testExplHeatEquationSourceFloatDirichletBCEuler() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ExplicitPDESchemes;
	using lss_one_dim_heat_equation_solvers_cuda::explicit_solvers::Explicit1DHeatEquationCUDA;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation with source: \n\n";
	std::cout << " Using explicit Euler method\n\n";
	std::cout << " Value type: " << typeid(float).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = 1, x in <0,1> \n\n";
	std::cout << "===============================================================================\n";

	// typedef the Explicit1DHeatEquationCUDA
	typedef Explicit1DHeatEquationCUDA<float,
		BoundaryConditionType::Dirichlet,
		std::vector,
		std::allocator<float>> explicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 10000;
	// initial condition:
	auto initialCondition = [](float x) {return 1.0; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 0.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<float> solution(Sd + 1,0.0f);
	// initialize solver
	explicit_solver expl_solver(Range<float>(0.0, 1.0), 0.5, Sd, Td);
	// set boundary conditions:
	expl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	expl_solver.setInitialCondition(initialCondition);
	// set heat source: 
	expl_solver.setHeatSource([](float x, float t) {return x; });
	// set thermal diffusivity (C^2 in PDE)
	expl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	expl_solver.solve(solution);
	// get exact solution:
	auto exact = [](float x, float t, std::size_t n) {
		float sum{};
		float q_n{};
		float f_n{};
		float lam_n{};
		float lam_2{};
		float var1{};
		for (std::size_t i = 1; i <= n; ++i) {
			q_n = (2.0 / (i*PI))*std::pow(-1.0, i + 1);
			f_n = (2.0 / (i*PI))*(1.0 - std::pow(-1.0, i));
			lam_n = i * PI;
			lam_2 = lam_n * lam_n;
			var1 = (q_n / lam_2 + (f_n - (q_n / lam_2))*std::exp(-1.0*lam_2*t))*std::sin(i*PI*x);
			sum += var1;
		}
		return sum;
	};

	float const h = expl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	float benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.5, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}


void testExplHeatEquationSourceDoubleDirichletBCEuler() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_types::ExplicitPDESchemes;
	using lss_one_dim_heat_equation_solvers_cuda::explicit_solvers::Explicit1DHeatEquationCUDA;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation with source: \n\n";
	std::cout << " Using explicit Euler method\n\n";
	std::cout << " Value type: " << typeid(double).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t) + x, \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U(0,t) = U(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = 1, x in <0,1> \n\n";
	std::cout << "===============================================================================\n";

	// typedef the Explicit1DHeatEquationCUDA
	typedef Explicit1DHeatEquationCUDA<double,
		BoundaryConditionType::Dirichlet,
		std::vector,
		std::allocator<double>> explicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 10000;
	// initial condition:
	auto initialCondition = [](double x) {return 1.0; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 0.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<double> solution(Sd + 1, 0.0f);
	// initialize solver
	explicit_solver expl_solver(Range<double>(0.0, 1.0), 0.5, Sd, Td);
	// set boundary conditions:
	expl_solver.setBoundaryCondition(boundary);
	// set initial condition:
	expl_solver.setInitialCondition(initialCondition);
	// set heat source: 
	expl_solver.setHeatSource([](double x, double t) {return x; });
	// set thermal diffusivity (C^2 in PDE)
	expl_solver.setThermalDiffusivity(1.0);
	// get the solution:
	expl_solver.solve(solution);
	// get exact solution:
	auto exact = [](double x, double t, std::size_t n) {
		double sum{};
		double q_n{};
		double f_n{};
		double lam_n{};
		double lam_2{};
		double var1{};
		for (std::size_t i = 1; i <= n; ++i) {
			q_n = (2.0 / (i*PI))*std::pow(-1.0, i + 1);
			f_n = (2.0 / (i*PI))*(1.0 - std::pow(-1.0, i));
			lam_n = i * PI;
			lam_2 = lam_n * lam_n;
			var1 = (q_n / lam_2 + (f_n - (q_n / lam_2))*std::exp(-1.0*lam_2*t))*std::sin(i*PI*x);
			sum += var1;
		}
		return sum;
	};

	double const h = expl_solver.spaceStep();
	std::cout << "tp : FDM | Exact | Abs Diff\n";
	double benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.5, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}


void testExplHeatEquationSourceDirichletBCEuler() {
	std::cout << "================================================================================\n";
	std::cout << "============== Explicit Heat Equation with source (Dirichlet BC) ===============\n";
	std::cout << "================================================================================\n";

	testExplHeatEquationSourceFloatDirichletBCEuler();
	testExplHeatEquationSourceDoubleDirichletBCEuler();

	std::cout << "================================================================================\n";
}



// ================================================================================================================
// ============================== Heat problem with homogeneous Robin boundary conditions =========================
// ================================================================================================================

void testExplHeatEquationDoubleRobinBCDeviceEuler() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_one_dim_heat_equation_solvers_cuda::explicit_solvers::Explicit1DHeatEquationCUDA;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation: \n\n";
	std::cout << " Using CUDA  explicit Euler method\n\n";
	std::cout << " Value type: " << typeid(double).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = 0.5 - (4/(pi*pi))*sum_1^infty{ exp(-((2n-1)*pi)^2*t) *cos((2n-1)*pi*x)/(2n-1)^2}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Explicit1DHeatEquationCUDA<double,
		BoundaryConditionType::Robin,
		std::vector,
		std::allocator<double>> explicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 5000;
	// initial condition:
	auto initialCondition = [](double x) {return x; };
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<double> solution(Sd + 1, 0.0);
	// initialize solver
	explicit_solver expl_solver(Range<double>(0.0, 1.0), 0.2, Sd, Td);
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
	auto exact = [](double x, double t, std::size_t n) {
		double const pipi = PI * PI;
		double const first = 4.0 / pipi;
		double sum{};
		double var0{};
		double var1{};
		double var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var0 = (2 * i - 1);
			var1 = std::exp(-1.0*pipi*var0*var0*t);
			var2 = std::cos(var0*PI*x) / (var0*var0);
			sum += (var1*var2);
		}
		return (0.5 - first * sum);
	};

	std::cout << "tp : FDM | Exact | Abs Diff\n";
	double benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.2, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}


void testExplHeatEquationFloatRobinBCDeviceEuler() {

	using lss_utility::Range;
	using lss_types::BoundaryConditionType;
	using lss_one_dim_heat_equation_solvers_cuda::explicit_solvers::Explicit1DHeatEquationCUDA;

	std::cout << "==============================================================================\n";
	std::cout << "Solving Boundary-value Heat equation: \n\n";
	std::cout << " Using CUDA  explicit Euler method\n\n";
	std::cout << " Value type: " << typeid(float).name() << "\n\n";
	std::cout << " U_t(x,t) = U_xx(x,t), \n\n";
	std::cout << " where\n\n";
	std::cout << " x in <0,1> and t > 0,\n";
	std::cout << " U_x(0,t) = U_x(1,t) = 0, t > 0 \n\n";
	std::cout << " U(x,0) = x, x in <0,1> \n\n";
	std::cout << " Exact solution: \n";
	std::cout << " U(x,t) = 0.5 - (4/(pi*pi))*sum_1^infty{ exp(-((2n-1)*pi)^2*t) *cos((2n-1)*pi*x)/(2n-1)^2}\n\n";
	std::cout << "===============================================================================\n";

	// typedef the Implicit1DHeatEquation
	typedef Explicit1DHeatEquationCUDA<float,
		BoundaryConditionType::Robin,
		std::vector,
		std::allocator<float>> explicit_solver;

	// number of space subdivisions:
	std::size_t const Sd = 100;
	// number of time subdivisions:
	std::size_t const Td = 5000;
	// initial condition:
	auto initialCondition = [](float x) {return x; };
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<float> solution(Sd + 1, 0.0);
	// initialize solver
	explicit_solver expl_solver(Range<float>(0.0, 1.0), 0.2, Sd, Td);
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
	auto exact = [](float x, float t, std::size_t n) {
		float const pipi = PI * PI;
		float const first = 4.0 / pipi;
		float sum{};
		float var0{};
		float var1{};
		float var2{};
		for (std::size_t i = 1; i <= n; ++i) {
			var0 = (2 * i - 1);
			var1 = std::exp(-1.0*pipi*var0*var0*t);
			var2 = std::cos(var0*PI*x) / (var0*var0);
			sum += (var1*var2);
		}
		return (0.5 - first * sum);
	};

	std::cout << "tp : FDM | Exact | Abs Diff\n";
	float benchmark{};
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		benchmark = exact(j * h, 0.2, 20);
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< benchmark << " | " << (solution[j] - benchmark) << '\n';
	}
}



void testExplHeatEquationRobinBCDevice() {
	std::cout << "================================================================================\n";
	std::cout << "======================= Explicit Heat Equation (Robin BC) ======================\n";
	std::cout << "================================================================================\n";

	testExplHeatEquationDoubleRobinBCDeviceEuler();
	testExplHeatEquationFloatRobinBCDeviceEuler();

	std::cout << "================================================================================\n";
}







#endif ///_LSS_ONE_DIM_HEAT_EQUATION_SOLVERS_CUDA_T