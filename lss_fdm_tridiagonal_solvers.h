#pragma once
#if !defined(_LSS_FDM_TRIDIAGONAL_SOLVERS)
#define _LSS_FDM_TRIDIAGONAL_SOLVERS

#include"lss_fdm_tridiagonal_solvers_policy.h"

namespace lss_fdm_tridiagonal_solvers {

	using lss_fdm_tridiagonal_solvers_policy::DoubleSweepSolver;
	using lss_fdm_tridiagonal_solvers_policy::FDMTridiagonalSolversPolicyBase;


	template<typename T,
			template<typename U,template<typename,typename> typename Container,typename Alloc>
			typename TridiagonalSparseSolver = DoubleSweepSolver,
			template<typename U,typename Alloc> typename Container = std::vector,
			typename U = T,
			typename Alloc = std::allocator<U>,
			typename = typename std::enable_if<std::is_floating_point<T>::value>::type,
			typename = typename std::enable_if<std::is_base_of<FDMTridiagonalSolversPolicyBase<T,Container,Alloc>,
				TridiagonalSparseSolver<T,Container,Alloc>>::value>::type>
	class FDMTridiagonalSparseSolver {
	private:
		TridiagonalSparseSolver<T, Container, Alloc> solver_;

	public:
		typedef T value_type;
		explicit FDMTridiagonalSparseSolver(std::size_t discretizationSize)
			:solver_(discretizationSize){}

		virtual ~FDMTridiagonalSparseSolver() {}

		FDMTridiagonalSparseSolver(FDMTridiagonalSparseSolver const&) = delete;
		FDMTridiagonalSparseSolver(FDMTridiagonalSparseSolver&&) = delete;

		FDMTridiagonalSparseSolver& operator=(FDMTridiagonalSparseSolver const&) = delete;
		FDMTridiagonalSparseSolver& operator=(FDMTridiagonalSparseSolver&&) = delete;

		void setDiagonals(Container<T, Alloc> lowerDiagonal,
			Container<T, Alloc> diagonal,
			Container<T, Alloc> upperDiagonal) {
			solver_.setDiagonals(std::move(lowerDiagonal),
				std::move(diagonal),
				std::move(upperDiagonal));
		}

		void setRhs(Container<T, Alloc> rhs) {
			solver_.setRhs(std::move(rhs));
		}

		inline void setDirichletBC(T right, T left) {
			solver_.setDirichletBC(std::move(right), std::move(left));
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