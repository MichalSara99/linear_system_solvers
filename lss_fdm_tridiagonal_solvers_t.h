#pragma once
#if !defined(_LSS_FDM_TRIDIAGONAL_SOLVERS_T)
#define _LSS_FDM_TRIDIAGONAL_SOLVERS_T

#include<vector>

#include"lss_types.h"
#include"lss_fdm_double_sweep_solver.h"
#include"lss_fdm_thomas_lu_solver.h"


void testBVPDoubleDoubleSweepDirichletBC() {

	using lss_fdm_double_sweep_solver::FDMDoubleSweepSolver;
	using lss_types::BoundaryConditionType;

	/*
	
		Solve BVP:

				u''(t) = - 2,
		
		where
		
				t \in (0, 1) 
				u(0) = 0 ,  u(1) = 0

		
		Exact solution is 
			
				u(t) = t(1-t) 
		
	*/
	std::cout << "=================================\n";
	std::cout << "Solving Boundary-value problem: \n\n";
	std::cout << " u''(t) = -2, \n\n";
	std::cout << " where\n\n";
	std::cout << " t in <0,1>,\n";
	std::cout << " u(0) = u(1) = 0\n\n";
	std::cout << "Exact solution is:\n\n";
	std::cout << " u(t) = t(1-t)\n";
	std::cout << "=================================\n";

	// discretization:
	std::size_t N{ 20 };
	// step size:
	double h = 1.0 / static_cast<double>(N);
	// upper,mid, and lower diagonal:
	std::vector<double> upperDiag(N + 1, 1.0);
	std::vector<double> diagonal(N + 1, -2.0);
	std::vector<double> lowerDiag(N + 1, 1.0);

	// right-hand side:
	std::vector<double> rhs(N + 1, -2.0 * h * h );

	// boundary conditions:
	double left = 0.0;
	double right = 0.0;

	FDMDoubleSweepSolver<double, BoundaryConditionType::Dirichlet,std::vector,std::allocator<double>> dss{ N + 1 };
	dss.setDiagonals(std::move(lowerDiag), std::move(diagonal), std::move(upperDiag));
	dss.setBoundaryCondition(std::make_pair(left, right));
	dss.setRhs(std::move(rhs));
	//get the solution:
	auto solution = dss.solve();

	//exact value:
	auto exact = [](double x) { return x * (1.0 - x); };

	std::cout << "tp : FDM | Exact\n";
	for (std::size_t j = 0; j < solution.size(); ++j)
	{ 
		std::cout <<"t_"<< j << ": " << solution[j] << " |  "
			<< exact(j*h)<< '\n';
	}
}

void testBVPFloatDoubleSweepDirichletBC() {

	using lss_fdm_double_sweep_solver::FDMDoubleSweepSolver;
	using lss_types::BoundaryConditionType;

	/*

		Solve BVP:

				u''(t) = - 2,

		where

				t \in (0, 1)
				u(0) = 0 ,  u(1) = 0


		Exact solution is

				u(t) = t(1-t)

	*/
	std::cout << "=================================\n";
	std::cout << "Solving Boundary-value problem: \n\n";
	std::cout << " u''(t) = -2, \n\n";
	std::cout << " where\n\n";
	std::cout << " t in <0,1>,\n";
	std::cout << " u(0) = u(1) = 0\n\n";
	std::cout << "Exact solution is:\n\n";
	std::cout << " u(t) = t(1-t)\n";
	std::cout << "=================================\n";

	// discretization:
	std::size_t N{ 20 };
	// step size:
	float h = 1.0 / static_cast<float>(N);
	// upper,mid, and lower diagonal:
	std::vector<float> upperDiag(N + 1, 1.0f);
	std::vector<float> diagonal(N + 1, -2.0f);
	std::vector<float> lowerDiag(N + 1, 1.0f);

	// right-hand side:
	std::vector<float> rhs(N + 1, -2.0 * h * h);

	// boundary conditions:
	float left = 0.0f;
	float right = 0.0f;

	FDMDoubleSweepSolver<float, BoundaryConditionType::Dirichlet, std::vector, std::allocator<float>> dss{ N + 1 };
	dss.setDiagonals(std::move(lowerDiag), std::move(diagonal), std::move(upperDiag));
	dss.setBoundaryCondition(std::make_pair(left, right));
	dss.setRhs(std::move(rhs));
	//get the solution:
	auto solution = dss.solve();

	//exact value:
	auto exact = [](float x) { return x * (1.0 - x); };

	std::cout << "tp : FDM | Exact\n";
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< exact(j * h) << '\n';
	}
}



void testDoubleSweepDirichletBC() {
	std::cout << "==================================================\n";
	std::cout << "=========== Double Sweep (Dirichlet BC) ==========\n";
	std::cout << "==================================================\n";

	testBVPDoubleDoubleSweepDirichletBC();
	testBVPFloatDoubleSweepDirichletBC();

	std::cout << "==================================================\n";
}


void testBVPDoubleThomasLUSolverDirichletBC() {

	using lss_fdm_thomas_lu_solver::FDMThomasLUSolver;
	using lss_types::BoundaryConditionType;

	/*

		Solve BVP:

				u''(t) = - 2,

		where

				t \in (0, 1)
				u(0) = 0 ,  u(1) = 0


		Exact solution is

				u(t) = t(1-t)

	*/
	std::cout << "=================================\n";
	std::cout << "Solving Boundary-value problem: \n\n";
	std::cout << " u''(t) = -2, \n\n";
	std::cout << " where\n\n";
	std::cout << " t in <0,1>,\n";
	std::cout << " u(0) = u(1) = 0\n\n";
	std::cout << "Exact solution is:\n\n";
	std::cout << " u(t) = t(1-t)\n";
	std::cout << "=================================\n";

	// discretization:
	std::size_t N{ 20 };
	// step size:
	double h = 1.0 / static_cast<double>(N);
	// upper,mid, and lower diagonal:
	// must be of size N+1 because we need to add the t_0 point:
	std::vector<double> upperDiag(N + 1, 1.0);
	std::vector<double> diagonal(N + 1, -2.0);
	std::vector<double> lowerDiag(N + 1, 1.0);

	// right-hand side:
	std::vector<double> rhs(N + 1, -2.0 * h * h);

	// boundary conditions:
	double left = 0.0;
	double right = 0.0;

	FDMThomasLUSolver<double,BoundaryConditionType::Dirichlet,std::vector,std::allocator<double>> dss{ N + 1 };
	dss.setDiagonals(std::move(lowerDiag), std::move(diagonal), std::move(upperDiag));
	dss.setBoundaryCondition(std::make_pair(left, right));
	dss.setRhs(std::move(rhs));
	//get the solution:
	auto solution = dss.solve();

	//exact value:
	auto exact = [](double x) { return x * (1.0 - x); };

	std::cout << "tp : FDM | Exact\n";
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< exact(j * h) << '\n';
	}
}

void testBVPFloatThomasLUSolverDirichletBC() {

	using lss_fdm_thomas_lu_solver::FDMThomasLUSolver;
	using lss_types::BoundaryConditionType;
	/*

		Solve BVP:

				u''(t) = - 2,

		where

				t \in (0, 1)
				u(0) = 0 ,  u(1) = 0


		Exact solution is

				u(t) = t(1-t)

	*/
	std::cout << "=================================\n";
	std::cout << "Solving Boundary-value problem: \n\n";
	std::cout << " u''(t) = -2, \n\n";
	std::cout << " where\n\n";
	std::cout << " t in <0,1>,\n";
	std::cout << " u(0) = u(1) = 0\n\n";
	std::cout << "Exact solution is:\n\n";
	std::cout << " u(t) = t(1-t)\n";
	std::cout << "=================================\n";

	// discretization:
	std::size_t N{ 20 };
	// step size:
	float h = 1.0 / static_cast<float>(N);
	// upper,mid, and lower diagonal:
	std::vector<float> upperDiag(N + 1, 1.0f);
	std::vector<float> diagonal(N + 1, -2.0f);
	std::vector<float> lowerDiag(N + 1, 1.0f);

	// right-hand side:
	std::vector<float> rhs(N + 1, -2.0 * h * h);

	// boundary conditions:
	float left = 0.0f;
	float right = 0.0f;


	FDMThomasLUSolver<float, BoundaryConditionType::Dirichlet, std::vector, std::allocator<float>> dss{ N + 1 };
	dss.setDiagonals(std::move(lowerDiag), std::move(diagonal), std::move(upperDiag));
	dss.setBoundaryCondition(std::make_pair(left, right));
	dss.setRhs(std::move(rhs));
	//get the solution:
	auto solution = dss.solve();

	//exact value:
	auto exact = [](float x) { return x * (1.0 - x); };

	std::cout << "tp : FDM | Exact\n";
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< exact(j * h) << '\n';
	}
}

void testThomasLUSolverDirichletBC() {
	std::cout << "==================================================\n";
	std::cout << "========= Thomas LU Solver (Dirichlet BC) ========\n";
	std::cout << "==================================================\n";

	testBVPDoubleThomasLUSolverDirichletBC();
	testBVPFloatThomasLUSolverDirichletBC();



	std::cout << "==================================================\n";
}


void testBVPDoubleDoubleSweepDirichletBC1() {

	using lss_fdm_double_sweep_solver::FDMDoubleSweepSolver;
	using lss_types::BoundaryConditionType;

	/*

	Solve BVP:

	u''(t) = - 2,

	where

	t \in (0, 1)
	u(0) = 1 ,  u(1) = 1


	Exact solution is

	u(t) = -t*t + t + 1

	*/
	std::cout << "=================================\n";
	std::cout << "Solving Boundary-value problem: \n\n";
	std::cout << " u''(t) = -2, \n\n";
	std::cout << " where\n\n";
	std::cout << " t in <0,1>,\n";
	std::cout << " u(0) = u(1) = 1\n\n";
	std::cout << "Exact solution is:\n\n";
	std::cout << " u(t) = -t*t + t + 1\n";
	std::cout << "=================================\n";

	// discretization:
	std::size_t N{ 20 };
	// step size:
	double h = 1.0 / static_cast<double>(N);
	// upper,mid, and lower diagonal:
	std::vector<double> upperDiag(N + 1, 1.0);
	std::vector<double> diagonal(N + 1, -2.0);
	std::vector<double> lowerDiag(N + 1, 1.0);

	// right-hand side:
	std::vector<double> rhs(N + 1, -2.0 * h * h);

	// boundary conditions:
	double left = 1.0;
	double right = 1.0;

	FDMDoubleSweepSolver<double, BoundaryConditionType::Dirichlet, std::vector, std::allocator<double>> dss{ N + 1 };
	dss.setDiagonals(std::move(lowerDiag), std::move(diagonal), std::move(upperDiag));
	dss.setBoundaryCondition(std::make_pair(left, right));
	dss.setRhs(std::move(rhs));
	//get the solution:
	auto solution = dss.solve();

	//exact value:
	auto exact = [](double x) { return (-x * x + x + 1.0); };

	std::cout << "tp : FDM | Exact\n";
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< exact(j*h) << '\n';
	}
}

void testBVPFloatDoubleSweepDirichletBC1() {

	using lss_fdm_double_sweep_solver::FDMDoubleSweepSolver;
	using lss_types::BoundaryConditionType;

	/*

	Solve BVP:

	u''(t) = - 2,

	where

	t \in (0, 1)
	u(0) = 0 ,  u(1) = 0


	Exact solution is

	u(t) = -t*t + t + 1

	*/
	std::cout << "=================================\n";
	std::cout << "Solving Boundary-value problem: \n\n";
	std::cout << " u''(t) = -2, \n\n";
	std::cout << " where\n\n";

	std::cout << " t in <0,1>,\n";
	std::cout << " u(0) = u(1) = 1\n\n";
	std::cout << "Exact solution is:\n\n";
	std::cout << " u(t) = -t*t + t + 1\n";
	std::cout << "=================================\n";

	// discretization:
	std::size_t N{ 20 };
	// step size:
	float h = 1.0 / static_cast<float>(N);
	// upper,mid, and lower diagonal:
	std::vector<float> upperDiag(N + 1, 1.0f);
	std::vector<float> diagonal(N + 1, -2.0f);
	std::vector<float> lowerDiag(N + 1, 1.0f);

	// right-hand side:
	std::vector<float> rhs(N + 1, -2.0 * h * h);

	// boundary conditions:
	float left = 1.0f;
	float right = 1.0f;

	FDMDoubleSweepSolver<float, BoundaryConditionType::Dirichlet, std::vector, std::allocator<float>> dss{ N + 1 };
	dss.setDiagonals(std::move(lowerDiag), std::move(diagonal), std::move(upperDiag));
	dss.setBoundaryCondition(std::make_pair(left, right));
	dss.setRhs(std::move(rhs));
	//get the solution:
	auto solution = dss.solve();

	//exact value:
	auto exact = [](float x) { return (-x * x + x + 1.0); };

	std::cout << "tp : FDM | Exact\n";
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< exact(j * h) << '\n';
	}
}



void testDoubleSweepDirichletBC1() {
	std::cout << "==================================================\n";
	std::cout << "=========== Double Sweep (Dirichlet BC) ==========\n";
	std::cout << "==================================================\n";

	testBVPDoubleDoubleSweepDirichletBC1();
	testBVPFloatDoubleSweepDirichletBC1();

	std::cout << "==================================================\n";
}


void testBVPDoubleThomasLUSolverDirichletBC1() {

	using lss_fdm_thomas_lu_solver::FDMThomasLUSolver;
	using lss_types::BoundaryConditionType;

	/*

	Solve BVP:

	u''(t) = - 2,

	where

	t \in (0, 1)
	u(0) = 0 ,  u(1) = 0


	Exact solution is

	u(t) = -t*t + t + 1

	*/
	std::cout << "=================================\n";
	std::cout << "Solving Boundary-value problem: \n\n";
	std::cout << " u''(t) = -2, \n\n";
	std::cout << " where\n\n";
	std::cout << " t in <0,1>,\n";
	std::cout << " u(0) = u(1) = 1\n\n";
	std::cout << "Exact solution is:\n\n";
	std::cout << " u(t) = -t*t + t + 1\n";
	std::cout << "=================================\n";

	// discretization:
	std::size_t N{ 20 };
	// step size:
	double h = 1.0 / static_cast<double>(N);
	// upper,mid, and lower diagonal:
	// must be of size N+1 because we need to add the t_0 point:
	std::vector<double> upperDiag(N + 1, 1.0);
	std::vector<double> diagonal(N + 1, -2.0);
	std::vector<double> lowerDiag(N + 1, 1.0);

	// right-hand side:
	std::vector<double> rhs(N + 1, -2.0 * h * h);

	// boundary conditions:
	double left = 1.0;
	double right = 1.0;

	FDMThomasLUSolver<double, BoundaryConditionType::Dirichlet, std::vector, std::allocator<double>> dss{ N + 1 };
	dss.setDiagonals(std::move(lowerDiag), std::move(diagonal), std::move(upperDiag));
	dss.setBoundaryCondition(std::make_pair(left, right));
	dss.setRhs(std::move(rhs));
	//get the solution:
	auto solution = dss.solve();

	//exact value:
	auto exact = [](double x) { return (-x * x + x + 1.0); };

	std::cout << "tp : FDM | Exact\n";
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< exact(j * h) << '\n';
	}
}

void testBVPFloatThomasLUSolverDirichletBC1() {

	using lss_fdm_thomas_lu_solver::FDMThomasLUSolver;
	using lss_types::BoundaryConditionType;
	/*

	Solve BVP:

	u''(t) = - 2,

	where

	t \in (0, 1)
	u(0) = 0 ,  u(1) = 0


	Exact solution is

	u(t) = -t*t + t + 1

	*/
	std::cout << "=================================\n";
	std::cout << "Solving Boundary-value problem: \n\n";
	std::cout << " u''(t) = -2, \n\n";
	std::cout << " where\n\n";
	std::cout << " t in <0,1>,\n";
	std::cout << " u(0) = u(1) = 1\n\n";
	std::cout << "Exact solution is:\n\n";
	std::cout << " u(t) = -t*t + t + 1\n";
	std::cout << "=================================\n";

	// discretization:
	std::size_t N{ 20 };
	// step size:
	float h = 1.0 / static_cast<float>(N);
	// upper,mid, and lower diagonal:
	std::vector<float> upperDiag(N + 1, 1.0f);
	std::vector<float> diagonal(N + 1, -2.0f);
	std::vector<float> lowerDiag(N + 1, 1.0f);

	// right-hand side:
	std::vector<float> rhs(N + 1, -2.0 * h * h);

	// boundary conditions:
	float left = 1.0f;
	float right = 1.0f;


	FDMThomasLUSolver<float, BoundaryConditionType::Dirichlet, std::vector, std::allocator<float>> dss{ N + 1 };
	dss.setDiagonals(std::move(lowerDiag), std::move(diagonal), std::move(upperDiag));
	dss.setBoundaryCondition(std::make_pair(left, right));
	dss.setRhs(std::move(rhs));
	//get the solution:
	auto solution = dss.solve();

	//exact value:
	auto exact = [](float x) { return (-x * x + x + 1.0); };

	std::cout << "tp : FDM | Exact\n";
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< exact(j * h) << '\n';
	}
}

void testThomasLUSolverDirichletBC1() {
	std::cout << "==================================================\n";
	std::cout << "======== Thomas LU Solver (Dirichlet BC) =========\n";
	std::cout << "==================================================\n";

	testBVPDoubleThomasLUSolverDirichletBC1();
	testBVPFloatThomasLUSolverDirichletBC1();



	std::cout << "==================================================\n";
}


void testBVPDoubleDoubleSweepRobinBC() {

	using lss_fdm_double_sweep_solver::FDMDoubleSweepSolver;
	using lss_types::BoundaryConditionType;

	/*

	Solve BVP:

	u''(t) = - 2,

	where

	t \in (0, 1)
	u(0) = 1 ,  u'(1) + u(1) = 0


	Exact solution is

	u(t) = -t*t + t + 1

	*/
	std::cout << "=================================\n";
	std::cout << "Solving Boundary-value problem: \n\n";
	std::cout << " u''(t) = -2, \n\n";
	std::cout << " where\n\n";
	std::cout << " t in <0,1>,\n";
	std::cout << " u(0) = 1 \n";
	std::cout << " u'(1) + u(1) = 0\n\n";
	std::cout << "Exact solution is:\n\n";
	std::cout << " u(t) = -t*t + t + 1\n";
	std::cout << "=================================\n";

	// discretization:
	std::size_t N{ 100 };
	// step size:
	double h = 1.0 / static_cast<double>(N);
	// upper,mid, and lower diagonal:
	std::vector<double> upperDiag(N + 1, 1.0);
	std::vector<double> diagonal(N + 1, -2.0);
	std::vector<double> lowerDiag(N + 1, 1.0);

	// right-hand side:
	std::vector<double> rhs(N + 1, -2.0 * h * h);

	// boundary conditions:
	double leftLin = 0.0;
	double leftConst = 1.0;
	auto left = std::make_pair(leftLin, leftConst);

	double rightLin = (2.0 + h) / (2.0 - h);
	double rightConst = 0.0;
	auto right = std::make_pair(rightLin, rightConst);

	FDMDoubleSweepSolver<double, BoundaryConditionType::Robin, std::vector, std::allocator<double>> dss{ N + 1 };
	dss.setDiagonals(std::move(lowerDiag), std::move(diagonal), std::move(upperDiag));
	dss.setBoundaryCondition(left, right);
	dss.setRhs(std::move(rhs));
	//get the solution:
	auto solution = dss.solve();

	//exact value:
	auto exact = [](double x) { return (-x * x + x + 1.0); };

	std::cout << "tp : FDM | Exact | Abs Diff\n";
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< exact(j * h) << " | " << (solution[j] - exact(j * h)) << '\n';
	}
}

void testBVPFloatDoubleSweepRobinBC() {

	using lss_fdm_double_sweep_solver::FDMDoubleSweepSolver;
	using lss_types::BoundaryConditionType;

	/*

	Solve BVP:

	u''(t) = - 2,

	where

	t \in (0, 1)
	u(0) = 1 ,  u'(1) + u(1) = 0


	Exact solution is

	u(t) = -t*t + t + 1

	*/
	std::cout << "=================================\n";
	std::cout << "Solving Boundary-value problem: \n\n";
	std::cout << " u''(t) = -2, \n\n";
	std::cout << " where\n\n";
	std::cout << " t in <0,1>,\n";
	std::cout << " u(0) = 1 \n";
	std::cout << " u'(1) + u(1) = 0\n\n";
	std::cout << "Exact solution is:\n\n";
	std::cout << " u(t) = -t*t + t + 1\n";
	std::cout << "=================================\n";

	// discretization:
	std::size_t N{ 100 };
	// step size:
	float h = 1.0 / static_cast<float>(N);
	// upper,mid, and lower diagonal:
	std::vector<float> upperDiag(N + 1, 1.0f);
	std::vector<float> diagonal(N + 1, -2.0f);
	std::vector<float> lowerDiag(N + 1, 1.0f);

	// right-hand side:
	std::vector<float> rhs(N + 1, -2.0 * h * h);

	// boundary conditions:
	float leftLin = 0.0f;
	float leftConst = 1.0f;
	auto left = std::make_pair(leftLin, leftConst);

	float rightLin = (2.0 + h) / (2.0 - h);
	float rightConst = 0.0f;
	auto right = std::make_pair(rightLin, rightConst);

	FDMDoubleSweepSolver<float, BoundaryConditionType::Robin, std::vector, std::allocator<float>> dss{ N + 1 };
	dss.setDiagonals(std::move(lowerDiag), std::move(diagonal), std::move(upperDiag));
	dss.setBoundaryCondition(left, right);
	dss.setRhs(std::move(rhs));
	//get the solution:
	auto solution = dss.solve();

	//exact value:
	auto exact = [](float x) { return (-x * x + x + 1.0); };

	std::cout << "tp : FDM | Exact | Abs Diff\n";
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< exact(j * h) << " | " << (solution[j] - exact(j * h)) << '\n';
	}
}



void testDoubleSweepRobinBC() {
	std::cout << "==================================================\n";
	std::cout << "=========== Double Sweep (Robin BC) ==============\n";
	std::cout << "==================================================\n";

	testBVPDoubleDoubleSweepRobinBC();
	testBVPFloatDoubleSweepRobinBC();

	std::cout << "==================================================\n";
}


void testBVPDoubleThomasLUSolverRobinBC() {

	using lss_fdm_thomas_lu_solver::FDMThomasLUSolver;
	using lss_types::BoundaryConditionType;

	/*

	Solve BVP:

	u''(t) = - 2,

	where

	t \in (0, 1)
	u(0) = 1 ,  u'(1) + u(1) = 0


	Exact solution is

	u(t) = -t*t + t + 1

	*/
	std::cout << "=================================\n";
	std::cout << "Solving Boundary-value problem: \n\n";
	std::cout << " u''(t) = -2, \n\n";
	std::cout << " where\n\n";
	std::cout << " t in <0,1>,\n";
	std::cout << " u(0) = 1 \n";
	std::cout << " u'(1) + u(1) = 0\n\n";
	std::cout << "Exact solution is:\n\n";
	std::cout << " u(t) = -t*t + t + 1\n";
	std::cout << "=================================\n";

	// discretization:
	std::size_t N{ 100 };
	// step size:
	double h = 1.0 / static_cast<double>(N);
	// upper,mid, and lower diagonal:
	// must be of size N+1 because we need to add the t_0 point:
	std::vector<double> upperDiag(N + 1, 1.0);
	std::vector<double> diagonal(N + 1, -2.0);
	std::vector<double> lowerDiag(N + 1, 1.0);

	// right-hand side:
	std::vector<double> rhs(N + 1, -2.0 * h * h);

	// boundary conditions:
	double leftLin = 0.0;
	double leftConst = 1.0;
	auto left = std::make_pair(leftLin, leftConst);

	double rightLin = (2.0 + h) / (2.0 - h);
	double rightConst = 0.0;
	auto right = std::make_pair(rightLin, rightConst);

	FDMThomasLUSolver<double, BoundaryConditionType::Robin, std::vector, std::allocator<double>> dss{ N + 1 };
	dss.setDiagonals(std::move(lowerDiag), std::move(diagonal), std::move(upperDiag));
	dss.setBoundaryCondition(left, right);
	dss.setRhs(std::move(rhs));
	//get the solution:
	auto solution = dss.solve();

	//exact value:
	auto exact = [](float x) { return (-x * x + x + 1.0); };

	std::cout << "tp : FDM | Exact | Abs Diff\n";
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< exact(j * h) << " | " << (solution[j] - exact(j * h)) << '\n';
	}
}

void testBVPFloatThomasLUSolverRobinBC() {

	using lss_fdm_thomas_lu_solver::FDMThomasLUSolver;
	using lss_types::BoundaryConditionType;
	/*

	Solve BVP:

	u''(t) = - 2,

	where

	t \in (0, 1)
	u(0) = 1 ,  u'(1) + u(1) = 0


	Exact solution is

	u(t) = -t*t + t + 1

	*/
	std::cout << "=================================\n";
	std::cout << "Solving Boundary-value problem: \n\n";
	std::cout << " u''(t) = -2, \n\n";
	std::cout << " where\n\n";
	std::cout << " t in <0,1>,\n";
	std::cout << " u(0) = 1 \n";
	std::cout << " u'(1) + u(1) = 0\n\n";
	std::cout << "Exact solution is:\n\n";
	std::cout << " u(t) = -t*t + t + 1\n";
	std::cout << "=================================\n";

	// discretization:
	std::size_t N{ 100 };
	// step size:
	float h = 1.0 / static_cast<float>(N);
	// upper,mid, and lower diagonal:
	std::vector<float> upperDiag(N + 1, 1.0f);
	std::vector<float> diagonal(N + 1, -2.0f);
	std::vector<float> lowerDiag(N + 1, 1.0f);

	// right-hand side:
	std::vector<float> rhs(N + 1, -2.0 * h * h);

	// boundary conditions:
	float leftLin = 0.0f;
	float leftConst = 1.0f;
	auto left = std::make_pair(leftLin, leftConst);

	float rightLin = (2.0 + h) / (2.0 - h);
	float rightConst = 0.0f;
	auto right = std::make_pair(rightLin, rightConst);


	FDMThomasLUSolver<float, BoundaryConditionType::Robin, std::vector, std::allocator<float>> dss{ N + 1 };
	dss.setDiagonals(std::move(lowerDiag), std::move(diagonal), std::move(upperDiag));
	dss.setBoundaryCondition(left, right);
	dss.setRhs(std::move(rhs));
	//get the solution:
	auto solution = dss.solve();

	//exact value:
	auto exact = [](float x) { return (-x * x + x + 1.0); };

	std::cout << "tp : FDM | Exact | Abs Diff\n";
	for (std::size_t j = 0; j < solution.size(); ++j)
	{
		std::cout << "t_" << j << ": " << solution[j] << " |  "
			<< exact(j * h) <<" | "<<(solution[j] - exact(j * h)) << '\n';
	}
}

void testThomasLUSolverRobinBC() {
	std::cout << "==================================================\n";
	std::cout << "========= Thomas LU Solver (Robin BC) ========\n";
	std::cout << "==================================================\n";

	testBVPDoubleThomasLUSolverRobinBC();
	testBVPFloatThomasLUSolverRobinBC();



	std::cout << "==================================================\n";
}


#endif ///_LSS_FDM_TRIDIAGONAL_SOLVERS_T