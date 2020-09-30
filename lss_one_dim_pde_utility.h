#pragma once
#if !defined(_LSS_ONE_DIM_PDE_UTILITY)
#define _LSS_ONE_DIM_PDE_UTILITY

#include<functional>
#include<tuple>

namespace lss_one_dim_pde_utility {

	template<typename T,
			template<typename,typename> typename Container,
			typename Alloc>
	class Discretization {
	public:
		virtual ~Discretization(){}
		virtual void discretizeSpace(T const &step,
									std::pair<T, T> const &dirichletBC,
									Container<T,Alloc> & container)const = 0;

		virtual void discretizeInitialCondition(std::function<T(T)> const &init,
												Container<T,Alloc> &container)const = 0;
	};




}






#endif ///_LSS_ONE_DIM_PDE_UTILITY