#pragma once
#if !defined(_LSS_HELPERS)
#define _LSS_HELPERS

#include<cusolverSp.h>
#include<cusolverDn.h>
#include<cublas_v2.h>

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

		~RealSparseSolverCUDAHelpers(){}

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


	// ===============================================================================
	// ======================== RealDenseSolverCUDAHelpers ===========================
	// ===============================================================================


	class RealDenseSolverCUDAHelpers {
	private:
		cusolverDnHandle_t solverHandle_;
		cublasHandle_t cublasHandle_;

	public:
		explicit RealDenseSolverCUDAHelpers()
			:solverHandle_{NULL},cublasHandle_{NULL}{}

		~RealDenseSolverCUDAHelpers(){}

		inline void initialize() {
			LSS_ASSERT((cusolverDnCreate(&solverHandle_) == CUSOLVER_STATUS_SUCCESS),
				" Failed to initialize dense solver handler");
			LSS_ASSERT((cublasCreate_v2(&cublasHandle_) == CUBLAS_STATUS_SUCCESS),
				" Failed to initialize cublas handler");
		}

		inline cusolverDnHandle_t const& getDenseSolverHandle() { return solverHandle_; }
		inline cublasHandle_t const& getCublasHandle() { return cublasHandle_; }

	};


}



#endif ///_LSS_HELPERS