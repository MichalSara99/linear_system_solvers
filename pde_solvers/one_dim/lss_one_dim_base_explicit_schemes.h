#pragma once
#if !defined(_LSS_ONE_DIM_BASE_EXPLICIT_SCHEMES)
#define _LSS_ONE_DIM_BASE_EXPLICIT_SCHEMES

#pragma warning( disable : 4244 )

#include<thread>
#include"common/lss_types.h"
#include"pde_solvers/one_dim/lss_one_dim_pde_utility.h"

namespace lss_one_dim_base_explicit_schemes {

	using lss_one_dim_pde_utility::Discretization;


	// ==============================================================================================================
	// ========================================== Explicit1DHeatSchemeBase  =========================================
	// ==============================================================================================================

	template<typename T,
			typename SchemeCoefficientHolder>
	class Explicit1DHeatSchemeBase :
		public Discretization<T, std::vector, std::allocator<T>> {
	protected:
		T spaceStart_;
		T terminalTime_;

		std::pair<T, T> deltas_;				// first = delta time, second = delta space
		SchemeCoefficientHolder coeffs_;		// scheme coefficients of PDE 
		std::vector<T> initialCondition_;
		std::function<T(T, T)> source_;
		bool isSourceSet_;

	public:
		explicit Explicit1DHeatSchemeBase() = delete;

		explicit Explicit1DHeatSchemeBase(T spaceStart,
									T terminalTime,
									std::pair<T, T> const& deltas,
									SchemeCoefficientHolder const& coeffs,
									std::vector<T> const& initialCondition,
									std::function<T(T, T)> const &source = nullptr,
									bool isSourceSet = false)
			:spaceStart_{ spaceStart },
			terminalTime_{ terminalTime },
			deltas_{ deltas },
			coeffs_{ coeffs },
			initialCondition_{ initialCondition },
			source_{ source },
			isSourceSet_{ isSourceSet } {}


		virtual ~Explicit1DHeatSchemeBase() = default;

		// stability check:
		virtual bool isStable()const = 0;

		// for Dirichlet BC
		virtual void operator()(std::pair<T, T> const &dirichletBCPair,
			std::vector<T> &solution)const = 0;
		// for Robin BC
		virtual void operator()(std::pair<T, T> const &leftRobinBCPair,
			std::pair<T, T> const &rightRobinBCPair,
			std::vector<T> &solution)const = 0;

	};


}


#endif //_LSS_ONE_DIM_BASE_EXPLICIT_SCHEMES