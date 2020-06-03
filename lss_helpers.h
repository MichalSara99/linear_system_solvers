#pragma once
#if !defined(_LSS_HELPERS)
#define _LSS_HELPERS

#include<cusolverSp.h>


#include"lss_macros.h"

namespace lss_helpers {




	// ===============================================================================
	// ======================= RealSparseSolverCUDAHelpers ===========================
	// ===============================================================================

	class RealSparseSolverCUDAHelpers {
	private:
		cusolverSpHandle_t solverHandle_;
		cusparseMatDescr_t matDesc_;
	public:
		explicit RealSparseSolverCUDAHelpers()
			:solverHandle_{ NULL }, matDesc_{ NULL }{}

		inline void initialize() {
			LSS_ASSERT((cusolverSpCreate(&solverHandle_) == CUSOLVER_STATUS_SUCCESS),
				" Failed to initialize sparse solver handler");
			LSS_ASSERT(cusparseCreateMatDescr(&matDesc_) == CUSPARSE_STATUS_SUCCESS,
				" Failed to initialize matrix descriptor");
			LSS_ASSERT(cusparseSetMatIndexBase(matDesc_, CUSPARSE_INDEX_BASE_ZERO) == CUSOLVER_STATUS_SUCCESS,
				" Failure while setting indexing style");
			LSS_ASSERT(cusparseSetMatType(matDesc_, CUSPARSE_MATRIX_TYPE_GENERAL) == CUSOLVER_STATUS_SUCCESS,
				" Failure while setting matrix type");
		}
		inline cusolverSpHandle_t const& getSolverHandle() { return solverHandle_; }
		inline cusparseMatDescr_t const& getMatrixDescriptor() { return matDesc_; }

	};

}



#endif ///_LSS_HELPERS