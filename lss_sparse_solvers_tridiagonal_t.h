#pragma once
#if !defined(_LSS_SPARSE_SOLVERS_TRIDIAGONAL_T)
#define _LSS_SPARSE_SOLVERS_TRIDIAGONAL_T

#include"lss_sparse_solvers_tridiagonal.h"


void testBVPDoubleDoubleSweep() {

	using lss_sparse_solvers_tridiagonal::DoubleSweepSolver;

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

	DoubleSweepSolver<double> dss{ N + 1 };
	dss.setDiagonals(std::move(lowerDiag), std::move(diagonal), std::move(upperDiag));
	dss.setDirichletBC(right, left);
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

void testBVPFloatDoubleSweep() {

	using lss_sparse_solvers_tridiagonal::DoubleSweepSolver;

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

	DoubleSweepSolver<float> dss{ N + 1 };
	dss.setDiagonals(std::move(lowerDiag), std::move(diagonal), std::move(upperDiag));
	dss.setDirichletBC(right, left);
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



void testDoubleSweep() {
	std::cout << "==================================================\n";
	std::cout << "================== Double Sweep ==================\n";
	std::cout << "==================================================\n";

	testBVPDoubleDoubleSweep();
	testBVPFloatDoubleSweep();

	std::cout << "==================================================\n";
}


void testBVPDoubleThomasLUSolver() {

	using lss_sparse_solvers_tridiagonal::ThomasLUSolver;

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

	ThomasLUSolver<double> dss{ N + 1 };
	dss.setDiagonals(std::move(lowerDiag), std::move(diagonal), std::move(upperDiag));
	dss.setDirichletBC(right, left);
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

void testBVPFloatThomasLUSolver() {

	using lss_sparse_solvers_tridiagonal::ThomasLUSolver;

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

	ThomasLUSolver<float> dss{ N + 1 };
	dss.setDiagonals(std::move(lowerDiag), std::move(diagonal), std::move(upperDiag));
	dss.setDirichletBC(right, left);
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

void testThomasLUSolver() {
	std::cout << "==================================================\n";
	std::cout << "============== Thomas LU Solver ==================\n";
	std::cout << "==================================================\n";

	testBVPDoubleThomasLUSolver();
	testBVPFloatThomasLUSolver();

	std::cout << "==================================================\n";
}




#endif ///_LSS_SPARSE_SOLVERS_TRIDIAGONAL_T