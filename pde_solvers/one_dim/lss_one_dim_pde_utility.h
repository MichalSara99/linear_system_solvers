#pragma once
#if !defined(_LSS_ONE_DIM_PDE_UTILITY)
#define _LSS_ONE_DIM_PDE_UTILITY

#include<functional>
#include<tuple>
#include"common/lss_macros.h"

namespace lss_one_dim_pde_utility {

	template<typename T,
			template<typename,typename> typename Container,
			typename Alloc>
	class Discretization {
	public:
		virtual ~Discretization(){}
		void discretizeSpace(T const &step,
							T const &init,
							Container<T,Alloc> & container)const;

		void discretizeInitialCondition(std::function<T(T)> const &init,
										Container<T,Alloc> &container)const;

		void discretizeInSpace(T const &start,
								T const &step,
								T const &time,
								std::function<T(T, T)> const &fun,
								Container<T, Alloc> &output)const;
	};


	template<typename T,
			template<typename,typename> typename Container,
			typename Alloc>
	void Discretization<T, Container, Alloc>::discretizeSpace(T const &step,
																T const &init,
																Container<T, Alloc> &container)const {
		LSS_ASSERT(container.size() > 0, "The input container must be initialized.");
		container[0] = init;
		for (std::size_t t = 1; t < container.size(); ++t) {
			container[t] = container[t - 1] + step;
		}
	}

	template<typename T,
			template<typename, typename> typename Container,
			typename Alloc>
	void Discretization<T, Container, Alloc>::discretizeInitialCondition(std::function<T(T)> const &init,
																		Container<T, Alloc> &container)const {
		LSS_ASSERT(container.size() > 0, "The input container must be initialized.");
		for (std::size_t t = 0; t < container.size(); ++t) {
			container[t] = init(container[t]);
		}
	}

	template<typename T,
			template<typename, typename> typename Container,
			typename Alloc>
	void Discretization<T, Container, Alloc>::discretizeInSpace(T const &step,
																T const &init,	
																T const &time,
																std::function<T(T, T)> const &fun,
																Container<T, Alloc> &container)const {
		LSS_ASSERT(container.size() > 0, "The input container must be initialized.");
		container[0] = fun(init, time);
		for (std::size_t t = 1; t < container.size(); ++t) {
			container[t] = fun(container[t - 1] + step, time);
		}
	}



}






#endif ///_LSS_ONE_DIM_PDE_UTILITY