#pragma once
#if !defined(_LSS_FDM_TRIDIAGONAL_SOLVERS_POLICY)
#define _LSS_FDM_TRIDIAGONAL_SOLVERS_POLICY

#include<vector>
#include<type_traits>
#include"lss_macros.h"

namespace lss_fdm_tridiagonal_solvers_policy{


	template<typename T,
			template<typename T,typename Allocator> typename Container = std::vector,
			typename Alloc = std::allocator<T>>
	class FDMTridiagonalSolversPolicyBase{};


	// =====================================================================================
	// ============================= DoubleSweepSolver =====================================
	// =====================================================================================

	template<typename T,
			template<typename T,typename Allocator> typename Container = std::vector,
			typename Alloc = std::allocator<T>>
	class DoubleSweepSolver:public FDMTridiagonalSolversPolicyBase<T, Container,Alloc> {
	private:
		std::size_t discretizationSize_;
		Container<T, Alloc>  a_, b_, c_, f_;
		Container<T, Alloc> L_, K_;
		T leftCondition_, rightCondition_;

		void kernel(Container<T, Alloc>& solution);

	public:
		typedef T value_type;
		explicit DoubleSweepSolver() = delete;
		explicit DoubleSweepSolver(std::size_t discretizationSize)
			:discretizationSize_{ discretizationSize },
			leftCondition_{}, rightCondition_{}{}

		DoubleSweepSolver(DoubleSweepSolver const&) = delete;
		DoubleSweepSolver& operator=(DoubleSweepSolver const&) = delete;

		DoubleSweepSolver(DoubleSweepSolver&&) = delete;
		DoubleSweepSolver& operator=(DoubleSweepSolver &&) = delete;

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


	// =====================================================================================
	// ================================ ThomasLUSolver =====================================
	// =====================================================================================

	template<typename T,
		template<typename T, typename Allocator> typename Container = std::vector,
		typename Alloc = std::allocator<T>>
	class ThomasLUSolver:public FDMTridiagonalSolversPolicyBase<T, Container, Alloc> {
	private:
		std::size_t systemSize_, discretizationSize_;
		Container<T, Alloc>  a_, b_, c_, f_;
		Container<T, Alloc> beta_, gamma_;
		T leftCondition_, rightCondition_;


		void kernel(Container<T, Alloc>& solution);
		bool isDiagonallyDominant()const;

	public:
		typedef T value_type;
		explicit ThomasLUSolver() = delete;
		explicit ThomasLUSolver(std::size_t discretizationSize)
			:discretizationSize_{ discretizationSize},
			systemSize_{ discretizationSize - 2}, // because we subtract the boundary values which are known
			leftCondition_{},
			rightCondition_{}{}

		virtual ~ThomasLUSolver() {}

		ThomasLUSolver(ThomasLUSolver const&) = delete;
		ThomasLUSolver(ThomasLUSolver &&) = delete;
		ThomasLUSolver& operator=(ThomasLUSolver const&) = delete;
		ThomasLUSolver& operator=(ThomasLUSolver&&) = delete;


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


// =============================== ThomasLUSolver ========================================

template<typename T,
		template<typename T,typename Alloc> typename Container,
		typename Alloc>
void lss_fdm_tridiagonal_solvers_policy::ThomasLUSolver<T,Container,Alloc>::
kernel(Container<T, Alloc>& solution) {
	// clear the working containers:
	beta_.clear();
	gamma_.clear();

	// resize the working containers:
	beta_.resize(systemSize_);
	gamma_.resize(systemSize_);

	// init values for the working containers:
	beta_[0] = b_[0];
	gamma_[0] = c_[0] / beta_[0];

	for (std::size_t t = 1; t < systemSize_ - 1; ++t) {
		beta_[t] = b_[t] - (a_[t] * gamma_[t - 1]);
		gamma_[t] = c_[t] / beta_[t];
	}
	beta_[systemSize_ - 1] = b_[systemSize_ - 1] - (a_[systemSize_ - 1] * gamma_[systemSize_ - 2]);


	solution[1] = f_[0] / beta_[0];
	for (std::size_t t = 1; t < systemSize_; ++t) {
		solution[t + 1] = (f_[t] - (a_[t] * solution[t])) / beta_[t];
	}

	f_[systemSize_ - 1] = solution[systemSize_];
	for (long t = systemSize_ - 2; t >= 0; --t) {
		f_[t] = solution[t + 1] - (gamma_[t] * f_[t + 1]);
	}

	solution[0] = leftCondition_;
	solution[discretizationSize_ - 1] = rightCondition_;
	std::copy(f_.begin(), f_.end(), std::next(solution.begin()));
}


template<typename T,
		template<typename T,typename Alloc> typename Container,
		typename Alloc>
bool lss_fdm_tridiagonal_solvers_policy::ThomasLUSolver<T, Container, Alloc>::
isDiagonallyDominant()const {
	if (std::abs(b_[0]) < std::abs(c_[0])) return false;
	if (std::abs(b_[systemSize_ - 1]) < std::abs(c_[systemSize_ - 1]))return false;

	for (std::size_t t = 0; t < systemSize_ - 1; ++t)
		if (std::abs(b_[t]) < (std::abs(a_[t]) + std::abs(c_[t])))
			return false;
	return true;
}

template<typename T,
		template<typename T,typename Alloc> typename Container,
		typename Alloc>
void lss_fdm_tridiagonal_solvers_policy::ThomasLUSolver<T,Container,Alloc>::
setRhs(Container<T, Alloc> rhs) {
	LSS_ASSERT(rhs.size() == discretizationSize_,
		"Inncorect size for right-hand side");
	f_.clear();f_.resize(systemSize_);
	for (std::size_t t = 0; t < systemSize_; ++t)
		f_[t] = std::move(rhs[t + 1]);
}

template<typename T,
		template<typename T,typename Alloc> typename Container,
		typename Alloc>
void lss_fdm_tridiagonal_solvers_policy::ThomasLUSolver<T,Container,Alloc>::
setDiagonals(Container<T, Alloc> lowerDiagonal,
	Container<T, Alloc> diagonal,
	Container<T, Alloc> upperDiagonal) {

	LSS_ASSERT(lowerDiagonal.size() == discretizationSize_,
		"Inncorect size for lowerDiagonal");
	LSS_ASSERT(diagonal.size() == discretizationSize_,
		"Inncorect size for diagonal");
	LSS_ASSERT(upperDiagonal.size() == discretizationSize_,
		"Inncorect size for upperDiagonal");
	a_.clear(); a_.resize(systemSize_);
	b_.clear(); b_.resize(systemSize_);
	c_.clear(); c_.resize(systemSize_);
	for (std::size_t t = 0; t < systemSize_; ++t) {
		a_[t] = std::move(lowerDiagonal[t + 1]);
		b_[t] = std::move(diagonal[t + 1]);
		c_[t] = std::move(upperDiagonal[t + 1]);
	}

	LSS_ASSERT(isDiagonallyDominant() == true,
		"Tridiagonal matrix must be diagonally dominant.");
}


template<typename T,
	template<typename T, typename Alloc> typename Container,
	typename Alloc>
void lss_fdm_tridiagonal_solvers_policy::ThomasLUSolver<T,Container,Alloc>::
solve(Container<T, Alloc>& solution) {
	LSS_ASSERT(solution.size() == discretizationSize_,
		"Incorrect size of solution container");
	kernel(solution);
}

template<typename T,
	template<typename T, typename Alloc> typename Container,
	typename Alloc>
Container<T,Alloc> const lss_fdm_tridiagonal_solvers_policy::ThomasLUSolver<T, Container, Alloc>::
solve() {
	Container<T, Alloc> solution(discretizationSize_);
	kernel(solution);
	return solution;
}



// =============================== DoubleSweepSolver ========================================

template<typename T,
	template<typename T, typename Alloc> typename Container,
	typename Alloc>
void lss_fdm_tridiagonal_solvers_policy::DoubleSweepSolver<T, Container, Alloc>::
setDiagonals(Container<T, Alloc> lowerDiagonal,
	Container<T, Alloc> diagonal,
	Container<T, Alloc> upperDiagonal) {

	LSS_ASSERT(lowerDiagonal.size() == discretizationSize_,
		"Inncorect size for lowerDiagonal");
	LSS_ASSERT(diagonal.size() == discretizationSize_,
		"Inncorect size for diagonal");
	LSS_ASSERT(upperDiagonal.size() == discretizationSize_,
		"Inncorect size for upperDiagonal");
	a_ = std::move(lowerDiagonal);
	b_ = std::move(diagonal);
	c_ = std::move(upperDiagonal);
}

template<typename T,
	template<typename T, typename Alloc> typename Container,
	typename Alloc>
void lss_fdm_tridiagonal_solvers_policy::DoubleSweepSolver<T, Container, Alloc>::
setRhs(Container<T, Alloc> rhs) {

	LSS_ASSERT(rhs.size() == discretizationSize_,
		"Inncorect size for right-hand side");
	f_ = std::move(rhs);
}


template<typename T,
	template<typename T,typename Alloc> typename Container,
	typename Alloc>
void lss_fdm_tridiagonal_solvers_policy::DoubleSweepSolver<T,Container,Alloc>::
solve(Container<T, Alloc>& solution) {
	LSS_ASSERT(solution.size() == discretizationSize_,
		"Incorrect size of solution container");
	kernel(solution);
}


template<typename T,
	template<typename T, typename Alloc> typename Container,
	typename Alloc>
Container<T, Alloc> const lss_fdm_tridiagonal_solvers_policy::DoubleSweepSolver<T, Container, Alloc>::
solve() {
	Container<T, Alloc> solution(discretizationSize_);
	kernel(solution);
	return solution;
}


template<typename T,
	template<typename T, typename Alloc> typename Container,
	typename Alloc>
void lss_fdm_tridiagonal_solvers_policy::DoubleSweepSolver<T, Container, Alloc>::
kernel(Container<T, Alloc>& solution) {
	// clear coefficients:
	K_.clear();
	L_.clear();
	// resize coefficients:
	K_.resize(discretizationSize_);
	L_.resize(discretizationSize_);
	// init coefficients:
	K_[0] = leftCondition_;
	L_[0] = 0.0;

	T tmp{};
	for (std::size_t t = 1; t < discretizationSize_; ++t) {
		tmp = b_[t] + (a_[t] * L_[t - 1]);
		L_[t] = -1.0 * c_[t] / tmp;
		K_[t] = (f_[t] - (a_[t] * K_[t - 1])) / tmp;
	}

	f_[0] = leftCondition_;
	f_[discretizationSize_ - 1] = rightCondition_;

	for (std::size_t t = discretizationSize_ - 2; t >= 1; --t) {
		f_[t] = (L_[t] * f_[t + 1]) + K_[t];
	}
	std::copy(f_.begin(), f_.end(), solution.begin());
}

#endif ///_LSS_FDM_TRIDIAGONAL_SOLVERS_POLICY