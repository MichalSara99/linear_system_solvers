#pragma once
#if !defined(_LSS_SPARSE_SOLVERS_TRIDIAGONAL)
#define _LSS_SPARSE_SOLVERS_TRIDIAGONAL

#include<vector>


namespace sparse_solvers_tridiagonal {



template<typename T,
		template<typename T,typename Allocator> typename Container = std::vector,
		typename Allocator = std::allocator<T>>
class DoubleSweepSolver {
private:
	Container<T, Allocator>  a_, b_, c_, f_;
	Container<T, Allocator> L_, K_;
	T leftCondition_, rightCondition_;

public:



};





}




#endif ///_LSS_SPARSE_SOLVERS_TRIDIAGONAL