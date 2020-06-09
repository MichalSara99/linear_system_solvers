#pragma once
#if !defined(_LSS_SPARSE_SOLVERS_TRIDIAGONAL)
#define _LSS_SPARSE_SOLVERS_TRIDIAGONAL

#include<vector>
#include<type_traits>
#include"lss_macros.h"

namespace lss_sparse_solvers_tridiagonal {



	template<typename T,
			template<typename T,typename Allocator> typename Container = std::vector,
			typename Alloc = std::allocator<T>,
			typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
	class DoubleSweepSolver {
	private:
		std::size_t systemSize_;
		Container<T, Alloc>  a_, b_, c_, f_;
		Container<T, Alloc> L_, K_;
		T leftCondition_, rightCondition_;

		void kernel(Container<T, Alloc>& solution);

	public:
		typedef T value_type;
		explicit DoubleSweepSolver() = delete;
		explicit DoubleSweepSolver(std::size_t systemSize)
			:systemSize_{ systemSize },
			leftCondition_{}, rightCondition_{}{}

		virtual ~DoubleSweepSolver() {}


		void setDiagonals(Container<T, Alloc> lowerDiagonal,
			Container<T, Alloc> diagonal,
			Container<T, Alloc> upperDiagonal);

		void setRhs(Container<T, Alloc> rhs);

		inline void setDirichletBC(T right, T left) {
			leftCondition_ = left;
			rightCondition_ = right;
		}

		void solve(Container<T, Alloc>& solution);

		Container<T, Alloc> const solve();

	};

}


template<typename T,
	template<typename T, typename Alloc> typename Container,
	typename Alloc,
	typename U>
void lss_sparse_solvers_tridiagonal::DoubleSweepSolver<T, Container, Alloc,U>::
setDiagonals(Container<T, Alloc> lowerDiagonal,
	Container<T, Alloc> diagonal,
	Container<T, Alloc> upperDiagonal) {

	LSS_ASSERT(lowerDiagonal.size() == systemSize_,
		"Inncorect size for lowerDiagonal");
	LSS_ASSERT(diagonal.size() == systemSize_,
		"Inncorect size for diagonal");
	LSS_ASSERT(upperDiagonal.size() == systemSize_,
		"Inncorect size for upperDiagonal");
	a_ = std::move(lowerDiagonal);
	b_ = std::move(diagonal);
	c_ = std::move(upperDiagonal);
}

template<typename T,
	template<typename T, typename Alloc> typename Container,
	typename Alloc,
	typename U>
void lss_sparse_solvers_tridiagonal::DoubleSweepSolver<T, Container, Alloc,U>::
setRhs(Container<T, Alloc> rhs) {

	LSS_ASSERT(rhs.size() == systemSize_,
		"Inncorect size for right-hand side");
	f_ = std::move(rhs);
}


template<typename T,
	template<typename T,typename Alloc> typename Container,
	typename Alloc,
	typename U>
void lss_sparse_solvers_tridiagonal::DoubleSweepSolver<T,Container,Alloc,U>::
solve(Container<T, Alloc>& solution) {
	kernel(solution);
}


template<typename T,
	template<typename T, typename Alloc> typename Container,
	typename Alloc,
	typename U>
Container<T, Alloc> const lss_sparse_solvers_tridiagonal::DoubleSweepSolver<T, Container, Alloc,U>::
solve() {
	Container<T, Alloc> solution(systemSize_);
	kernel(solution);
	return solution;
}


template<typename T,
	template<typename T, typename Alloc> typename Container,
	typename Alloc,
	typename U>
void lss_sparse_solvers_tridiagonal::DoubleSweepSolver<T, Container, Alloc,U>::
kernel(Container<T, Alloc>& solution) {
	// clear coefficients:
	K_.clear();
	L_.clear();
	// resize coefficients:
	K_.resize(systemSize_);
	L_.resize(systemSize_);
	// init coefficients:
	K_[0] = leftCondition_;
	L_[0] = 0.0;

	T tmp{};
	for (std::size_t t = 1; t < systemSize_; ++t) {
		tmp = b_[t] + (a_[t] * L_[t - 1]);
		L_[t] = -1.0 * c_[t] / tmp;
		K_[t] = (f_[t] - (a_[t] * K_[t - 1])) / tmp;
	}

	f_[0] = leftCondition_;
	f_[systemSize_ - 1] = rightCondition_;

	for (std::size_t t = systemSize_ - 2; t >= 1; --t) {
		f_[t] = (L_[t] * f_[t + 1]) + K_[t];
	}
	std::copy(f_.begin(), f_.end(), solution.begin());
}

#endif ///_LSS_SPARSE_SOLVERS_TRIDIAGONAL