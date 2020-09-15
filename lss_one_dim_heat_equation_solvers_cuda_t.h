#pragma once
#if !defined(_LSS_ONE_DIM_HEAT_EQUATION_SOLVERS_CUDA_T)
#define _LSS_ONE_DIM_HEAT_EQUATION_SOLVERS_CUDA_T

#include"lss_types.h"
#include"lss_utility.h"
#include"lss_one_dim_heat_equation_solvers_cuda.h"

#define PI 3.14159

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
	std::cout << " U(0,t) = U(0,t) = 0, t > 0 \n\n";
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
	std::size_t const Td = 10000;
	// initial condition:
	auto initialCondition = [](double x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 0.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<double> solution(Sd + 1, 0.0);
	// initialize solver
	implicit_solver impl_solver(Range<double>(0.0, 1.0), 0.5, Sd, Td);
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
		benchmark = exact(j * h, 0.5, 20);
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
	std::cout << " U(0,t) = U(0,t) = 0, t > 0 \n\n";
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
	std::size_t const Td = 10000;
	// initial condition:
	auto initialCondition = [](float x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 0.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<float> solution(Sd + 1, 0.0);
	// initialize solver
	implicit_solver impl_solver(Range<float>(0.0, 1.0), 0.5, Sd, Td);
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
		benchmark = exact(j * h, 0.5, 20);
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
	std::cout << " U(0,t) = U(0,t) = 0, t > 0 \n\n";
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
	std::size_t const Td = 10000;
	// initial condition:
	auto initialCondition = [](double x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 0.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<double> solution(Sd + 1, 0.0);
	// initialize solver
	implicit_solver impl_solver(Range<double>(0.0, 1.0), 0.5, Sd, Td);
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
		benchmark = exact(j * h, 0.5, 20);
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
	std::cout << " U(0,t) = U(0,t) = 0, t > 0 \n\n";
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
	std::size_t const Td = 10000;
	// initial condition:
	auto initialCondition = [](float x) {return x; };
	// boundary conditions:
	auto boundary = std::make_pair(0.0, 0.0);
	// prepare container for solution:
	// note: size is Sd+1 since we must include space point at x = 0
	std::vector<float> solution(Sd + 1, 0.0);
	// initialize solver
	implicit_solver impl_solver(Range<float>(0.0, 1.0), 0.5, Sd, Td);
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
		benchmark = exact(j * h, 0.5, 20);
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













#endif ///_LSS_ONE_DIM_HEAT_EQUATION_SOLVERS_CUDA_T