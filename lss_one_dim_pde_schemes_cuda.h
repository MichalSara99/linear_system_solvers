#pragma once
#if !defined(_LSS_ONE_DIM_PDE_SCHEMES_CUDA)
#define _LSS_ONE_DIM_PDE_SCHEMES_CUDA

#include<tuple>


namespace lss_one_dim_pde_schemes_cuda {


	template<typename T,
			template<typename,typename> typename Container,
			typename Alloc>
	class ExplicitEulerHeatEquationScheme{
	private:
		T lambda_;
		Container<T, Alloc> init_;

	public:
		typedef T value_type;
		explicit ExplicitEulerHeatEquationScheme() = delete;
		explicit ExplicitEulerHeatEquationScheme(T lambda,Container<T,Alloc> const &init)
			:lambda_{lambda},init_{init}{}

		~ExplicitEulerHeatEquationScheme(){}

		ExplicitEulerHeatEquationScheme(ExplicitEulerHeatEquationScheme const &) = delete;
		ExplicitEulerHeatEquationScheme(ExplicitEulerHeatEquationScheme &&) = delete;
		ExplicitEulerHeatEquationScheme& operator=(ExplicitEulerHeatEquationScheme const &) = delete;
		ExplicitEulerHeatEquationScheme& operator=(ExplicitEulerHeatEquationScheme &&) = delete;

		void operator()(std::pair<T, T> const &dirichletBCPair, Container<T,Alloc> &solution)const;

	};


	// ==============================================================================================================
	// ============================== ExplicitEulerHeatEquationScheme  implementation =================================
	// ==============================================================================================================

	template<typename T,
		template<typename, typename> typename Container,
		typename Alloc>
	void ExplicitEulerHeatEquationScheme<T,Container,Alloc>::
		operator()(std::pair<T, T> const &dirichletBCPair, Container<T, Alloc> &solution)const {

		/// here explicitEulerIterate will be used

	}



}

#endif ///_LSS_ONE_DIM_PDE_SCHEMES_CUDA