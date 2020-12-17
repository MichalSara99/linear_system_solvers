#pragma once
#if !defined(_LSS_SPARSE_SOLVERS_POLICY)
#define _LSS_SPARSE_SOLVERS_POLICY

#include"common/lss_macros.h"
#include<cusolverSp.h>

#include<type_traits>

namespace lss_sparse_solvers_policy {

	/* Base for sparse factorization on Host */

	template<typename T>
	struct SparseSolverHost{};


	/* Sparse QR factorization on Host */
	template<typename T>
	struct SparseSolverHostQR: public SparseSolverHost<T> {
	private:
		// for T = double
		static void _solve_impl(cusolverSpHandle_t handle,cusparseMatDescr_t matDesc,
			int const systemSize, int const nonZeroSize, T const* h_matVals, int const* h_rowCounts,
			int const* h_colIndices, T const* h_rhs,
			T tol, int reorder, T* h_solution, int* singularity, std::true_type);
		
		// for T = float
		static void _solve_impl(cusolverSpHandle_t handle, cusparseMatDescr_t matDesc,
			int const systemSize, int const nonZeroSize, T const* h_matVals, int const* h_rowCounts,
			int const* h_colIndices, T const* h_rhs,
			T tol, int reorder, T* h_solution, int* singularity, std::false_type);

	public:
		static void solve(cusolverSpHandle_t handle, cusparseMatDescr_t matDesc,
			int const systemSize, int const nonZeroSize, T const* h_matVals, int const* h_rowCounts,
			int const* h_colIndices, T const* h_rhs,
			T tol, int reorder, T* h_solution, int* singularity) {

			_solve_impl(handle, matDesc, systemSize,nonZeroSize, h_matVals, h_rowCounts, h_colIndices,
				h_rhs, tol, reorder, h_solution, singularity, std::is_same<T, double>());
		
		}
	};


	/* Sparse LU factorization on Host */

	template<typename T>
	struct SparseSolverHostLU:public SparseSolverHost<T> {
	private:
		// for T = double
		static void _solve_impl(cusolverSpHandle_t handle, cusparseMatDescr_t matDesc,
			int const systemSize, int const nonZeroSize, T const* h_matVals, int const* h_rowCounts,
			int const* h_colIndices, T const* h_rhs,
			T tol, int reorder, T* h_solution, int* singularity, std::true_type);

		// for T = float
		static void _solve_impl(cusolverSpHandle_t handle, cusparseMatDescr_t matDesc,
			int const systemSize, int const nonZeroSize, T const* h_matVals, int const* h_rowCounts,
			int const* h_colIndices, T const* h_rhs,
			T tol, int reorder, T* h_solution, int* singularity, std::false_type);

	public:
		static void solve(cusolverSpHandle_t handle, cusparseMatDescr_t matDesc,
			int const systemSize, int const nonZeroSize, T const* h_matVals, int const* h_rowCounts,
			int const* h_colIndices, T const* h_rhs,
			T tol, int reorder, T* h_solution, int* singularity) {

			_solve_impl(handle, matDesc, systemSize,nonZeroSize, h_matVals, h_rowCounts, h_colIndices,
				h_rhs, tol, reorder, h_solution, singularity, std::is_same<T, double>());

		}
	};

	/* Sparse Cholesky factorization on Host */

	template<typename T>
	struct SparseSolverHostCholesky:public SparseSolverHost<T> {
	private:
		// for T = double
		static void _solve_impl(cusolverSpHandle_t handle, cusparseMatDescr_t matDesc,
			int const systemSize, int const nonZeroSize, T const* h_matVals, int const* h_rowCounts,
			int const* h_colIndices, T const* h_rhs,
			T tol, int reorder, T* h_solution, int* singularity, std::true_type);

		// for T = float
		static void _solve_impl(cusolverSpHandle_t handle, cusparseMatDescr_t matDesc,
			int const systemSize, int const nonZeroSize, T const* h_matVals, int const* h_rowCounts,
			int const* h_colIndices, T const* h_rhs,
			T tol, int reorder, T* h_solution, int* singularity, std::false_type);

	public:
		static void solve(cusolverSpHandle_t handle, cusparseMatDescr_t matDesc,
			int const systemSize, int const nonZeroSize, T const* h_matVals, int const* h_rowCounts,
			int const* h_colIndices, T const* h_rhs,
			T tol, int reorder, T* h_solution, int* singularity) {

			_solve_impl(handle, matDesc, systemSize,nonZeroSize, h_matVals, h_rowCounts, h_colIndices,
				h_rhs, tol, reorder, h_solution, singularity, std::is_same<T, double>());

		}
	};


	/* Base for sparse factorization on Device */

	template<typename T>
	struct SparseSolverDevice {};

	/* Sparse QR factorization on Device */

	template<typename T>
	struct SparseSolverDeviceQR:public SparseSolverDevice<T> {
	private:
		// for T = double
		static void _solve_impl(cusolverSpHandle_t handle, cusparseMatDescr_t matDesc,
			int const systemSize, int const nonZeroSize, T const* d_matVals, int const* d_rowCounts,
			int const* d_colIndices, T const* d_rhs,
			T tol, int reorder, T* d_solution, int* singularity, std::true_type);

		// for T = float
		static void _solve_impl(cusolverSpHandle_t handle, cusparseMatDescr_t matDesc,
			int const systemSize, int const nonZeroSize, T const* d_matVals, int const* d_rowCounts,
			int const* d_colIndices, T const* d_rhs,
			T tol, int reorder, T* d_solution, int* singularity, std::false_type);

	public:
		static void solve(cusolverSpHandle_t handle, cusparseMatDescr_t matDesc,
			int const systemSize, int const nonZeroSize, T const* d_matVals, int const* d_rowCounts,
			int const* d_colIndices, T const* d_rhs,
			T tol, int reorder, T* d_solution, int* singularity) {

			_solve_impl(handle, matDesc, systemSize,nonZeroSize, d_matVals, d_rowCounts, d_colIndices,
				d_rhs, tol, reorder, d_solution, singularity, std::is_same<T, double>());

		}
	};

	/* Sparse Cholesky factorization on Device */

	template<typename T>
	struct SparseSolverDeviceCholesky:public SparseSolverDevice<T> {
	private:
		// for T = double
		static void _solve_impl(cusolverSpHandle_t handle, cusparseMatDescr_t matDesc,
			int const systemSize, int const nonZeroSize, T const* d_matVals, int const* d_rowCounts,
			int const* d_colIndices, T const* d_rhs,
			T tol, int reorder, T* d_solution, int* singularity, std::true_type);

		// for T = float
		static void _solve_impl(cusolverSpHandle_t handle, cusparseMatDescr_t matDesc,
			int const systemSize, int const nonZeroSize, T const* d_matVals, int const* d_rowCounts,
			int const* d_colIndices, T const* d_rhs,
			T tol, int reorder, T* d_solution, int* singularity, std::false_type);

	public:
		static void solve(cusolverSpHandle_t handle, cusparseMatDescr_t matDesc,
			int const systemSize, int const nonZeroSize, T const* d_matVals, int const* d_rowCounts,
			int const* d_colIndices, T const* d_rhs,
			T tol, int reorder, T* d_solution, int* singularity) {

			_solve_impl(handle, matDesc, systemSize,nonZeroSize, d_matVals, d_rowCounts, d_colIndices,
				d_rhs, tol, reorder, d_solution, singularity, std::is_same<T, double>());

		}
	};



}

/* Sparse QR factorization on HOST */

template<typename T>
/*static*/ void lss_sparse_solvers_policy::SparseSolverHostQR<T>::
_solve_impl(cusolverSpHandle_t handle, cusparseMatDescr_t matDesc,
	int const systemSize, int const nonZeroSize, T const* h_matVals, int const* h_rowCounts,
	int const* h_colIndices, T const* h_rhs,
	T tol, int reorder, T* h_solution, int* singularity, std::true_type) {


	CUSOLVER_STATUS(cusolverSpDcsrlsvqrHost(handle,
		systemSize,
		nonZeroSize,
		matDesc,
		h_matVals,
		h_rowCounts,
		h_colIndices,
		h_rhs,
		tol,
		reorder,
		h_solution,
		singularity));

}


template<typename T>
/*static*/ void lss_sparse_solvers_policy::SparseSolverHostQR<T>::
_solve_impl(cusolverSpHandle_t handle, cusparseMatDescr_t matDesc,
	int const systemSize, int const nonZeroSize, T const* h_matVals, int const* h_rowCounts,
	int const* h_colIndices, T const* h_rhs,
	T tol, int reorder, T* h_solution, int* singularity, std::false_type) {

	CUSOLVER_STATUS(cusolverSpScsrlsvqrHost(handle,
		systemSize,
		nonZeroSize,
		matDesc,
		h_matVals,
		h_rowCounts,
		h_colIndices,
		h_rhs,
		tol,
		reorder,
		h_solution,
		singularity));
}


/* Sparse LU factorization on HOST */

template<typename T>
/*static*/ void lss_sparse_solvers_policy::SparseSolverHostLU<T>::
_solve_impl(cusolverSpHandle_t handle, cusparseMatDescr_t matDesc,
	int const systemSize, int const nonZeroSize, T const* h_matVals, int const* h_rowCounts,
	int const* h_colIndices, T const* h_rhs,
	T tol, int reorder, T* h_solution, int* singularity, std::true_type) {

	CUSOLVER_STATUS(cusolverSpDcsrlsvluHost(handle,
		systemSize,
		nonZeroSize,
		matDesc,
		h_matVals,
		h_rowCounts,
		h_colIndices,
		h_rhs,
		tol,
		reorder,
		h_solution,
		singularity));

}


template<typename T>
/*static*/ void lss_sparse_solvers_policy::SparseSolverHostLU<T>::
_solve_impl(cusolverSpHandle_t handle, cusparseMatDescr_t matDesc,
	int const systemSize, int const nonZeroSize, T const* h_matVals, int const* h_rowCounts,
	int const* h_colIndices, T const* h_rhs,
	T tol, int reorder, T* h_solution, int* singularity, std::false_type) {

	CUSOLVER_STATUS(cusolverSpScsrlsvluHost(handle,
		systemSize,
		nonZeroSize,
		matDesc,
		h_matVals,
		h_rowCounts,
		h_colIndices,
		h_rhs,
		tol,
		reorder,
		h_solution,
		singularity));
}

/* Sparse Cholesky factorization on HOST */

template<typename T>
/*static*/ void lss_sparse_solvers_policy::SparseSolverHostCholesky<T>::
_solve_impl(cusolverSpHandle_t handle, cusparseMatDescr_t matDesc,
	int const systemSize, int const nonZeroSize, T const* h_matVals, int const* h_rowCounts,
	int const* h_colIndices, T const* h_rhs,
	T tol, int reorder, T* h_solution, int* singularity, std::true_type) {

	CUSOLVER_STATUS(cusolverSpDcsrlsvcholHost(handle,
		systemSize,
		nonZeroSize,
		matDesc,
		h_matVals,
		h_rowCounts,
		h_colIndices,
		h_rhs,
		tol,
		reorder,
		h_solution,
		singularity));

}


template<typename T>
/*static*/ void lss_sparse_solvers_policy::SparseSolverHostCholesky<T>::
_solve_impl(cusolverSpHandle_t handle, cusparseMatDescr_t matDesc,
	int const systemSize, int const nonZeroSize, T const* h_matVals, int const* h_rowCounts,
	int const* h_colIndices, T const* h_rhs,
	T tol, int reorder, T* h_solution, int* singularity, std::false_type) {

	CUSOLVER_STATUS(cusolverSpScsrlsvcholHost(handle,
		systemSize,
		nonZeroSize,
		matDesc,
		h_matVals,
		h_rowCounts,
		h_colIndices,
		h_rhs,
		tol,
		reorder,
		h_solution,
		singularity));
}


/* Sparse QR factorization on DEVICE */

template<typename T>
/*static*/ void lss_sparse_solvers_policy::SparseSolverDeviceQR<T>::
_solve_impl(cusolverSpHandle_t handle, cusparseMatDescr_t matDesc,
	int const systemSize,int const nonZeroSize, T const* d_matVals, int const* d_rowCounts,
	int const* d_colIndices, T const* d_rhs,
	T tol, int reorder, T* d_solution, int* singularity, std::true_type) {


	CUSOLVER_STATUS(cusolverSpDcsrlsvqr(handle,
		systemSize,
		nonZeroSize,
		matDesc,
		d_matVals,
		d_rowCounts,
		d_colIndices,
		d_rhs,
		tol,
		reorder,
		d_solution,
		singularity));

}


template<typename T>
/*static*/ void lss_sparse_solvers_policy::SparseSolverDeviceQR<T>::
_solve_impl(cusolverSpHandle_t handle, cusparseMatDescr_t matDesc,
	int const systemSize, int const nonZeroSize, T const* d_matVals, int const* d_rowCounts,
	int const* d_colIndices, T const* d_rhs,
	T tol, int reorder, T* d_solution, int* singularity, std::false_type) {

	CUSOLVER_STATUS(cusolverSpScsrlsvqr(handle,
		systemSize,
		nonZeroSize,
		matDesc,
		d_matVals,
		d_rowCounts,
		d_colIndices,
		d_rhs,
		tol,
		reorder,
		d_solution,
		singularity));
}


/* Sparse Cholesky factorization on DEVICE */

template<typename T>
/*static*/ void lss_sparse_solvers_policy::SparseSolverDeviceCholesky<T>::
_solve_impl(cusolverSpHandle_t handle, cusparseMatDescr_t matDesc,
	int const systemSize, int const nonZeroSize, T const* d_matVals, int const* d_rowCounts,
	int const* d_colIndices, T const* d_rhs,
	T tol, int reorder, T* d_solution, int* singularity, std::true_type) {


	CUSOLVER_STATUS(cusolverSpDcsrlsvchol(handle,
		systemSize,
		nonZeroSize,
		matDesc,
		d_matVals,
		d_rowCounts,
		d_colIndices,
		d_rhs,
		tol,
		reorder,
		d_solution,
		singularity));

}


template<typename T>
/*static*/ void lss_sparse_solvers_policy::SparseSolverDeviceCholesky<T>::
_solve_impl(cusolverSpHandle_t handle, cusparseMatDescr_t matDesc,
	int const systemSize, int const nonZeroSize, T const* d_matVals, int const* d_rowCounts,
	int const* d_colIndices, T const* d_rhs,
	T tol, int reorder, T* d_solution, int* singularity, std::false_type) {

	CUSOLVER_STATUS(cusolverSpScsrlsvchol(handle,
		systemSize,
		nonZeroSize,
		matDesc,
		d_matVals,
		d_rowCounts,
		d_colIndices,
		d_rhs,
		tol,
		reorder,
		d_solution,
		singularity));
}


#endif ///_LSS_SPARSE_SOLVERS_POLICY