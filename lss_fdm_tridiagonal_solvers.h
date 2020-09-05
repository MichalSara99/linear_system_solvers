#pragma once
#if !defined(_LSS_FDM_TRIDIAGONAL_SOLVERS)
#define _LSS_FDM_TRIDIAGONAL_SOLVERS

#include"lss_types.h"
#include"lss_fdm_double_sweep_solver.h"
#include"lss_fdm_thomas_lu_solver.h"

namespace lss_fdm_tridiagonal_solvers {

	using lss_types::BoundaryConditionType;
	using lss_fdm_double_sweep_solver::FDMDoubleSweepSolver;
	using lss_fdm_thomas_lu_solver::FDMThomasLUSolver;

	template<typename T,
			BoundaryConditionType BCType,
			template<typename, 
					BoundaryConditionType,
					template<typename,typename> typename Cont,
					typename> typename FDMSolver,
			template<typename,typename> typename Container,
			typename Alloc>
	class FDMTridiagonalSolver {
	};


	template<typename T,
		template<typename, 
				BoundaryConditionType, 
				template<typename, typename> typename Cont,
				typename> typename FDMSolver,
		template<typename, typename> typename Container,
		typename Alloc>
	class FDMTridiagonalSolver<T, BoundaryConditionType::Dirichlet,FDMSolver,Container,Alloc> {
	private:
		FDMSolver<T, BoundaryConditionType::Dirichlet, Container, Alloc> solver_;
	public:
		typedef T value_type;
		explicit FDMTridiagonalSolver() = delete;
		explicit FDMTridiagonalSolver(std::size_t discretizationSize)
			:solver_{ discretizationSize }{}

		FDMTridiagonalSolver(FDMTridiagonalSolver const &) = delete;
		FDMTridiagonalSolver& operator=(FDMTridiagonalSolver const &) = delete;
		FDMTridiagonalSolver(FDMTridiagonalSolver &&) = delete;
		FDMTridiagonalSolver& operator=(FDMTridiagonalSolver &&) = delete;

		~FDMTridiagonalSolver(){}

		void setDiagonals(Container<T, Alloc> lowerDiagonal,
			Container<T, Alloc> diagonal,
			Container<T, Alloc> upperDiagonal) {
			solver_.setDiagonals(std::move(lowerDiagonal),
				std::move(diagonal),
				std::move(upperDiagonal));
		}

		void setBoundaryCondition(std::pair<T, T> const &boundaryPair) {
			solver_.setBoundaryCondition(boundaryPair);
		}

		void setRhs(Container<T, Alloc> const &rhs) {
			solver_.setRhs(rhs);
		}

		void solve(Container<T, Alloc>& solution) {
			solver_.solve(solution);
		}

		Container<T, Alloc> const solve() {
			return solver_.solve();
		}

	};


	template<typename T,
		template<typename, 
				BoundaryConditionType,
				template<typename, typename> typename Cont,
				typename> typename FDMSolver,
		template<typename, typename> typename Container,
		typename Alloc>
	class FDMTridiagonalSolver<T, BoundaryConditionType::Robin, FDMSolver, Container, Alloc> {
	private:
		FDMSolver<T, BoundaryConditionType::Robin, Container, Alloc> solver_;
	public:
		typedef T value_type;
		explicit FDMTridiagonalSolver() = delete;
		explicit FDMTridiagonalSolver(std::size_t discretizationSize)
			:solver_{ discretizationSize } {}

		FDMTridiagonalSolver(FDMTridiagonalSolver const &) = delete;
		FDMTridiagonalSolver& operator=(FDMTridiagonalSolver const &) = delete;
		FDMTridiagonalSolver(FDMTridiagonalSolver &&) = delete;
		FDMTridiagonalSolver& operator=(FDMTridiagonalSolver &&) = delete;

		~FDMTridiagonalSolver() {}

		void setDiagonals(Container<T, Alloc> lowerDiagonal,
			Container<T, Alloc> diagonal,
			Container<T, Alloc> upperDiagonal) {
			solver_.setDiagonals(std::move(lowerDiagonal),
				std::move(diagonal),
				std::move(upperDiagonal));
		}

		void setBoundaryCondition(std::pair<T, T> const &left, std::pair<T, T> const &right) {
			solver_.setBoundaryCondition(left, right);
		}

		void setRhs(Container<T, Alloc> const &rhs) {
			solver_.setRhs(rhs);
		}

		void solve(Container<T, Alloc>& solution) {
			solver_.solve(solution);
		}

		Container<T, Alloc> const solve() {
			return solver_.solve();
		}

	};
}

#endif //_LSS_FDM_TRIDIAGONAL_SOLVERS