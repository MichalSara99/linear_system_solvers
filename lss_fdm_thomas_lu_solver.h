#pragma once
#if !defined(_LSS_FDM_THOMAS_LU_SOLVER)
#define _LSS_FDM_THOMAS_LU_SOLVER

#include<vector>
#include<type_traits>
#include"lss_macros.h"
#include"lss_types.h"

namespace lss_fdm_thomas_lu_solver{

	using lss_types::BoundaryConditionType;


	// =====================================================================================
	// ============================= FDMThomasLUSolverBase ==================================
	// =====================================================================================

	template<typename T,
		template<typename T, typename Allocator> typename Container = std::vector,
		typename Alloc = std::allocator<T>>
	class FDMThomasLUSolverBase {
	protected:
		std::size_t systemSize_, discretizationSize_;
		Container<T, Alloc>  a_, b_, c_, f_;
		Container<T, Alloc> beta_, gamma_;

		virtual void kernel(Container<T, Alloc>& solution) = 0;
		virtual bool isDiagonallyDominant()const = 0;

	public:
		typedef T value_type;
		explicit FDMThomasLUSolverBase() = delete;
		explicit FDMThomasLUSolverBase(std::size_t discretizationSize)
			:discretizationSize_{ discretizationSize },
			systemSize_{ discretizationSize - 2 } {} // because we subtract the boundary values which are known{}

		virtual ~FDMThomasLUSolverBase() {}

		void setDiagonals(Container<T, Alloc> lowerDiagonal,
			Container<T, Alloc> diagonal,
			Container<T, Alloc> upperDiagonal);

		void setRhs(Container<T, Alloc> rhs);

		void solve(Container<T, Alloc>& solution);

		Container<T, Alloc> const solve();


	};


// =====================================================================================
// ============================= Concrete FDMThomasLUSolver ============================
// =====================================================================================


	template<typename T,
		BoundaryConditionType BCType,
		template<typename T, typename Allocator> typename Container,
		typename Alloc>
		class FDMThomasLUSolver {
		protected:
			void kernel(Container<T, Alloc>& solution) override {}
	};

	// Thomas LU Solver specialization for Dirichlet BC:
	template<typename T,
		template<typename T, typename Allocator> typename Container,
		typename Alloc>
		class FDMThomasLUSolver<T, BoundaryConditionType::Dirichlet,Container,Alloc>
	:public FDMThomasLUSolverBase<T,Container,Alloc> {
		private:
			std::pair<T, T> boundary_;
		public:
			typedef T value_type;
			explicit FDMThomasLUSolver() = delete;
			explicit FDMThomasLUSolver(std::size_t discretizationSize)
				:FDMThomasLUSolverBase<T, Container, Alloc>(discretizationSize){} 

			~FDMThomasLUSolver() {}

			FDMThomasLUSolver(FDMThomasLUSolver const&) = delete;
			FDMThomasLUSolver(FDMThomasLUSolver &&) = delete;
			FDMThomasLUSolver& operator=(FDMThomasLUSolver const&) = delete;
			FDMThomasLUSolver& operator=(FDMThomasLUSolver&&) = delete;

			inline void setBoundaryCondition(std::pair<T, T> const &boundaryPair)
			{ boundary_ = boundaryPair; }

		protected:
			bool isDiagonallyDominant()const override;
			void kernel(Container<T, Alloc>& solution) override;

	};

	// Thomas LU Solver specialization for Robin BC:
	template<typename T,
		template<typename T, typename Allocator> typename Container,
		typename Alloc>
		class FDMThomasLUSolver<T, BoundaryConditionType::Robin, Container, Alloc>
		:public FDMThomasLUSolverBase<T, Container, Alloc> {
		private:
			std::pair<T, T> left_;
			std::pair<T, T> right_;

		public:
			typedef T value_type;
			explicit FDMThomasLUSolver() = delete;
			explicit FDMThomasLUSolver(std::size_t discretizationSize)
				:FDMThomasLUSolverBase<T, Container, Alloc>(discretizationSize) {}

			~FDMThomasLUSolver() {}

			FDMThomasLUSolver(FDMThomasLUSolver const&) = delete;
			FDMThomasLUSolver(FDMThomasLUSolver &&) = delete;
			FDMThomasLUSolver& operator=(FDMThomasLUSolver const&) = delete;
			FDMThomasLUSolver& operator=(FDMThomasLUSolver&&) = delete;

			inline void setBoundaryCondition(std::pair<T, T> const &left, std::pair<T, T> const &right)
			{
				left_ = left;
				right_ = right;
			}

		protected:
			bool isDiagonallyDominant()const override;
			void kernel(Container<T, Alloc>& solution) override;

	};

}

// =============================== FDMThomasLUSolverBase implementation ========================================


template<typename T,
		template<typename T,typename Alloc> typename Container,
		typename Alloc>
void lss_fdm_thomas_lu_solver::FDMThomasLUSolverBase<T,Container,Alloc>::
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
void lss_fdm_thomas_lu_solver::FDMThomasLUSolverBase<T,Container,Alloc>::
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

	//LSS_ASSERT(isDiagonallyDominant() == true,
	//	"Tridiagonal matrix must be diagonally dominant.");
}


template<typename T,
	template<typename T, typename Alloc> typename Container,
	typename Alloc>
void lss_fdm_thomas_lu_solver::FDMThomasLUSolverBase<T,Container,Alloc>::
solve(Container<T, Alloc>& solution) {
	LSS_ASSERT(solution.size() == discretizationSize_,
		"Incorrect size of solution container");
	kernel(solution);
}

template<typename T,
	template<typename T, typename Alloc> typename Container,
	typename Alloc>
Container<T,Alloc> const lss_fdm_thomas_lu_solver::FDMThomasLUSolverBase<T, Container, Alloc>::
solve() {
	Container<T, Alloc> solution(discretizationSize_);
	kernel(solution);
	return solution;
}

// =============================== ThomasLUSolver implementation ========================================

template<typename T,
	template<typename T, typename Alloc> typename Container,
	typename Alloc>
	bool lss_fdm_thomas_lu_solver::FDMThomasLUSolver<T, lss_types::BoundaryConditionType::Dirichlet, Container, Alloc>::
	isDiagonallyDominant()const {
	if (std::abs(b_[0]) < std::abs(c_[0])) return false;
	if (std::abs(b_[systemSize_ - 1]) < std::abs(a_[systemSize_ - 1]))return false;

	for (std::size_t t = 0; t < systemSize_ - 1; ++t)
		if (std::abs(b_[t]) < (std::abs(a_[t]) + std::abs(c_[t])))
			return false;
	return true;
}

template<typename T,
	template<typename T, typename Alloc> typename Container,
	typename Alloc>
	void lss_fdm_thomas_lu_solver::FDMThomasLUSolver<T,lss_types::BoundaryConditionType::Dirichlet, Container, Alloc>::
	kernel(Container<T, Alloc>& solution) {

	// check the diagonal dominance:
	LSS_ASSERT(isDiagonallyDominant() == true,
		"Tridiagonal matrix must be diagonally dominant.");

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


	solution[1] = (f_[0] - a_[0]* boundary_.first) / beta_[0];
	for (std::size_t t = 1; t < systemSize_; ++t) {
		solution[t + 1] = (f_[t] - (a_[t] * solution[t])) / beta_[t];
	}
	solution[systemSize_] = ((f_[systemSize_-1] - c_[systemSize_ - 1]* boundary_.second) - (a_[systemSize_-1] * solution[systemSize_-1])) /
		beta_[systemSize_ - 1];

	f_[systemSize_ - 1] = solution[systemSize_];
	for (long t = systemSize_ - 2; t >= 0; --t) {
		f_[t] = solution[t + 1] - (gamma_[t] * f_[t + 1]);
	}
	//fill in the known boundary values:
	solution[0] = boundary_.first;
	solution[discretizationSize_ - 1] = boundary_.second;
	std::copy(f_.begin(), f_.end(), std::next(solution.begin()));
}


template<typename T,
	template<typename T, typename Alloc> typename Container,
	typename Alloc>
	bool lss_fdm_thomas_lu_solver::FDMThomasLUSolver<T, lss_types::BoundaryConditionType::Robin, Container, Alloc>::
	isDiagonallyDominant()const {
	auto alpha = left_.first;
	auto beta = right_.first;
	if (std::abs(alpha*a_[0]  + b_[0]) < std::abs(c_[0])) return false;
	if (std::abs(b_[systemSize_ - 1] + (c_[systemSize_ - 1]/beta)) < std::abs(a_[systemSize_ - 1]))return false;
	for (std::size_t t = 0; t < systemSize_ - 1; ++t)
		if (std::abs(b_[t]) < (std::abs(a_[t]) + std::abs(c_[t])))
			return false;
	return true;
}

template<typename T,
	template<typename T, typename Alloc> typename Container,
	typename Alloc>
	void lss_fdm_thomas_lu_solver::FDMThomasLUSolver<T, lss_types::BoundaryConditionType::Robin, Container, Alloc>::
	kernel(Container<T, Alloc>& solution) {

	// check the diagonal dominance:
	LSS_ASSERT(isDiagonallyDominant() == true,
		"Tridiagonal matrix must be diagonally dominant.");

	// clear the working containers:
	beta_.clear();
	gamma_.clear();

	// resize the working containers:
	beta_.resize(systemSize_);
	gamma_.resize(systemSize_);
	

	// init values for the working containers:
	beta_[0] = left_.first *a_[0] + b_[0];
	gamma_[0] = c_[0] / beta_[0];

	for (std::size_t t = 1; t < systemSize_ - 1; ++t) {
		beta_[t] = b_[t] - (a_[t] * gamma_[t - 1]);
		gamma_[t] = c_[t] / beta_[t];
	}
	beta_[systemSize_ - 1] = (b_[systemSize_ - 1] + (c_[systemSize_ - 1]/ right_.first)) - 
		(a_[systemSize_ - 1] * gamma_[systemSize_ - 2]);


	solution[1] = (f_[0] - a_[0] * left_.second) / beta_[0];
	for (std::size_t t = 1; t < systemSize_; ++t) {
		solution[t + 1] = (f_[t] - (a_[t] * solution[t])) / beta_[t];
	}
	solution[systemSize_] = ((f_[systemSize_ - 1] + (c_[systemSize_ - 1] * right_.second)/ right_.first) 
		- (a_[systemSize_ - 1] * solution[systemSize_ - 1])) /
		beta_[systemSize_ - 1];

	f_[systemSize_ - 1] = solution[systemSize_];
	for (long t = systemSize_ - 2; t >= 0; --t) {
		f_[t] = solution[t + 1] - (gamma_[t] * f_[t + 1]);
	}

	solution[0] = (left_.first*f_[0]) + left_.second;
	solution[discretizationSize_ - 1] = (f_[systemSize_ - 1] - right_.second) / right_.first;
	std::copy(f_.begin(), f_.end(), std::next(solution.begin()));
}

#endif ///_LSS_THOMAS_LU_SOLVER