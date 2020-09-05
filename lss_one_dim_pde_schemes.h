#pragma once
#if !defined(_LSS_ONE_DIM_PDE_SCHEMES)
#define _LSS_ONE_DIM_PDE_SCHEMES

#include"lss_types.h"

namespace lss_one_dim_pde_schemes {

	using lss_types::ImplicitPDESchemes;


	template<typename T>
	class ImplicitHeatEquationSchemes {
		public:
			typedef std::function<void(T,std::vector<T> const&,std::vector<T> &)> SchemeFunction;

			static T const getTheta(ImplicitPDESchemes scheme) {
				double theta{};
				if (scheme == ImplicitPDESchemes::Euler)
					theta = 1.0;
				else
					theta = 0.5;
				return theta;
			}

			static SchemeFunction const getScheme(ImplicitPDESchemes scheme) {
				double theta{};
				if (scheme == ImplicitPDESchemes::Euler)
					theta = 1.0;
				else
					theta = 0.5;
				auto schemeFun = [=](T lambda, 
										std::vector<T> const& input,
										std::vector<T> &solution) {
					for (std::size_t t = 1; t < solution.size() - 1; ++t) {
						solution[t] = (lambda*(1.0 - theta)*input[t + 1])
							+ (1.0 - (2.0*lambda*(1.0 - theta)))*input[t]
							+ (lambda*(1.0 - theta)*input[t - 1]);
					}
				};
				return schemeFun;
			}
	};



}





#endif //_LSS_ONE_DIM_PDE_SCHEMES