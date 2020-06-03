#pragma once
#if !defined(_LSS_SPARSE_SPARSE_CUDA)
#define _LSS_SPARSE_SPARSE_CUDA

#include<type_traits>

#include<cuda_runtime.h>
#include<cusolverSp.h>

#include<thrust/host_vector.h>
#include<thrust/device_vector.h>

#include"lss_types.h"
#include"lss_utility.h"
#include"lss_helpers.h"


namespace lss_sparse_solvers_cuda {

	using lss_types::MemorySpace;
	using lss_types::SparseSolverFactorizationDevice;
	using lss_types::SparseSolverFactorizationHost;
	using lss_utility::FlatSparseMatrix;
	using lss_helpers::RealSparseSolverCUDAHelpers;



	template<MemorySpace MemSpace,
			typename T,
			typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
	class RealSparseSolverCUDA{};


	// ===============================================================================
	// ========= RealSparseSolverCUDA partial specialization for HOST ================
	// ===============================================================================

	template<typename T>
	class RealSparseSolverCUDA<MemorySpace::Host, T> {
	protected:
		int systemSize_;
		FlatSparseMatrix<T> matrixElements_;

		thrust::host_vector<T> h_matrixValues_;
		thrust::host_vector<T> h_vectorValues_; // of systemSize length
		thrust::host_vector<int> h_columnIndices_; 
		thrust::host_vector<int> h_rowCounts_; // of systemSize + 1 length

		// this one is for T = double
		void solve_impl(T const* h_matVals, int const* h_rowCounts, int const* h_colIndices, T const* h_rhs,
			T tol, int reorder, T* h_solution, int *singularity, SparseSolverFactorizationHost ssfh, std::true_type);
		// this one is for T = float
		void solve_impl(T const* h_matVals, int const* h_rowCounts, int const* h_colIndices, T const* h_rhs,
			T tol, int reorder, T* h_solution, int *singularity, SparseSolverFactorizationHost ssfh, std::false_type);

		void buildCSR();

	public:
		explicit RealSparseSolverCUDA(){}
		virtual ~RealSparseSolverCUDA(){}

		void initialize(int systemSize);

		inline int const nonZeroElements()const { return matrixElements_.size(); }

		inline void setRhs(std::vector<T> const &rhs) {
			for (std::size_t t = 0; t < h_vectorValues_.size(); ++t)
				h_vectorValues_[t] = rhs[t];
		}

		inline void setRhs(T const *rhs) {
			thrust::copy(rhs, rhs + systemSize_, h_vectorValues_.begin());
		}

		inline void setRhsValue(int idx, T value) {
			LSS_ASSERT(((idx >= 0) && (idx < systemSize_)), "idx is outside range");
			h_vectorValues_[idx] = value;
		}

		inline void setFlatSparseMatrix(FlatSparseMatrix<T> matrix) {
			matrixElements_ = std::move(matrix);
		}

		inline void setFlatSparseMatrixValue(int rowIdx, int colIdx, T value) {
			matrixElements_.emplace_back(rowIdx, colIdx, value);
		}

		inline void setFlatSparseMatrixValue(std::tuple<int, int, T> triplet) {
			matrixElements_.emplace_back(std::move(triplet));
		}

		void solve(T* solution, SparseSolverFactorizationHost ssfh);

		std::vector<T> const solve(SparseSolverFactorizationHost ssfh);

	};


	// ===============================================================================
	// ========= RealSparseSolverCUDA partial specialization for DEVICE ==============
	// ===============================================================================

	template<typename T>
	class RealSparseSolverCUDA<MemorySpace::Device, T> {
	protected:
		int systemSize_;
		FlatSparseMatrix<T> matrixElements_;

		thrust::host_vector<T> h_matrixValues_;
		thrust::host_vector<T> h_vectorValues_; // of systemSize length
		thrust::host_vector<int> h_columnIndices_;
		thrust::host_vector<int> h_rowCounts_; // of systemSize + 1 length

		// this one is for T = double
		void solve_impl(T const* d_matVals, int const * d_rowCounts,int const* d_colIndices, T const* d_rhs,
			T tol, int reorder, T* d_solution, int *singularity, SparseSolverFactorizationDevice ssfd, std::true_type);
		// this one is for T = float
		void solve_impl(T const* d_matVals, int const * d_rowCounts, int const* d_colIndices, T const* d_rhs,
			T tol, int reorder, T* d_solution, int *singularity, SparseSolverFactorizationDevice ssfd, std::false_type);

		void buildCSR();

	public:
		explicit RealSparseSolverCUDA() {}
		virtual ~RealSparseSolverCUDA() {}

		void initialize(int systemSize);

		inline int const nonZeroElements()const { return matrixElements_.size(); }

		inline void setRhs(std::vector<T> const& rhs) {
			for (std::size_t t = 0; t < h_vectorValues_.size(); ++t)
				h_vectorValues_[t] = rhs[t];
		}

		inline void setRhs(T const* rhs) {
			thrust::copy(rhs, rhs + systemSize_, h_vectorValues_.begin());
		}

		inline void setRhsValue(int idx, T value) {
			LSS_ASSERT(((idx >= 0) && (idx < systemSize_)), "idx is outside range");
			h_vectorValues_[idx] = value;
		}

		inline void setFlatSparseMatrix(FlatSparseMatrix<T> matrix) {
			matrixElements_ = std::move(matrix);
		}

		inline void setFlatSparseMatrixValue(int rowIdx, int colIdx, T value) {
			matrixElements_.emplace_back(rowIdx, colIdx, value);
		}

		inline void setFlatSparseMatrixValue(std::tuple<int, int, T> triplet) {
			matrixElements_.emplace_back(std::move(triplet));
		}

		void solve(T* solution, SparseSolverFactorizationDevice ssfd);

		std::vector<T> const solve(SparseSolverFactorizationDevice ssfd);

	};





}


template<typename T>
void lss_sparse_solvers_cuda::RealSparseSolverCUDA<lss_types::MemorySpace::Host, T>::
solve_impl(T const* h_matVals,  int const* h_rowCounts, int const* h_colIndices, T const* h_rhs,
	T tol, int reorder, T* h_solution, int *singularity, SparseSolverFactorizationHost ssfh, std::true_type) {
	
	int const nonZeroSize = nonZeroElements();

	RealSparseSolverCUDAHelpers helpers;
	helpers.initialize();

	switch (ssfh) {
	case SparseSolverFactorizationHost::QR:
		CUSOLVER_STATUS(cusolverSpDcsrlsvqrHost(helpers.getSolverHandle(),
			systemSize_, 
			nonZeroSize, 
			helpers.getMatrixDescriptor(),
			h_matVals, 
			h_rowCounts, 
			h_colIndices,
			h_rhs,
			tol,
			reorder,
			h_solution,
			singularity));
		break;
	case SparseSolverFactorizationHost::Cholesky:
		CUSOLVER_STATUS(cusolverSpDcsrlsvcholHost(helpers.getSolverHandle(),
			systemSize_,
			nonZeroSize,
			helpers.getMatrixDescriptor(),
			h_matVals,
			h_rowCounts,
			h_colIndices,
			h_rhs,
			tol,
			reorder,
			h_solution,
			singularity));
		break;
	case SparseSolverFactorizationHost::LU:
		CUSOLVER_STATUS(cusolverSpDcsrlsvluHost(helpers.getSolverHandle(),
			systemSize_,
			nonZeroSize,
			helpers.getMatrixDescriptor(),
			h_matVals,
			h_rowCounts,
			h_colIndices,
			h_rhs,
			tol,
			reorder,
			h_solution,
			singularity));
		break;
	default:
		CUSOLVER_STATUS(cusolverSpDcsrlsvqrHost(helpers.getSolverHandle(),
			systemSize_,
			nonZeroSize,
			helpers.getMatrixDescriptor(),
			h_matVals,
			h_rowCounts,
			h_colIndices,
			h_rhs,
			tol,
			reorder,
			h_solution,
			singularity));
		break;
	}


}

template<typename T>
void lss_sparse_solvers_cuda::RealSparseSolverCUDA<lss_types::MemorySpace::Host, T>::
solve_impl(T const* h_matVals, int const* h_rowCounts, int const* h_colIndices, T const* h_rhs,
	T tol, int reorder, T* h_solution, int* singularity, SparseSolverFactorizationHost ssfh, std::false_type) {

	int const nonZeroSize = nonZeroElements();

	RealSparseSolverCUDAHelpers helpers;
	helpers.initialize();

	switch (ssfh) {
	case SparseSolverFactorizationHost::QR:
		CUSOLVER_STATUS(cusolverSpScsrlsvqrHost(helpers.getSolverHandle(),
			systemSize_,
			nonZeroSize,
			helpers.getMatrixDescriptor(),
			h_matVals,
			h_rowCounts,
			h_colIndices,
			h_rhs,
			tol,
			reorder,
			h_solution,
			singularity));
		break;
	case SparseSolverFactorizationHost::Cholesky:
		CUSOLVER_STATUS(cusolverSpScsrlsvcholHost(helpers.getSolverHandle(),
			systemSize_,
			nonZeroSize,
			helpers.getMatrixDescriptor(),
			h_matVals,
			h_rowCounts,
			h_colIndices,
			h_rhs,
			tol,
			reorder,
			h_solution,
			singularity));
		break;
	case SparseSolverFactorizationHost::LU:
		CUSOLVER_STATUS(cusolverSpScsrlsvluHost(helpers.getSolverHandle(),
			systemSize_,
			nonZeroSize,
			helpers.getMatrixDescriptor(),
			h_matVals,
			h_rowCounts,
			h_colIndices,
			h_rhs,
			tol,
			reorder,
			h_solution,
			singularity));
		break;
	default:
		CUSOLVER_STATUS(cusolverSpScsrlsvqrHost(helpers.getSolverHandle(),
			systemSize_,
			nonZeroSize,
			helpers.getMatrixDescriptor(),
			h_matVals,
			h_rowCounts,
			h_colIndices,
			h_rhs,
			tol,
			reorder,
			h_solution,
			singularity));
		break;
	}

}


template<typename T>
void lss_sparse_solvers_cuda::RealSparseSolverCUDA<lss_types::MemorySpace::Device,T>::
solve_impl(T const* d_matVals, int const* d_rowCounts, int const* d_colIndices, T const* d_rhs,
	T tol, int reorder, T* d_solution, int *singularity, SparseSolverFactorizationDevice ssfd, std::true_type) {
	int const nonZeroSize = nonZeroElements();

	RealSparseSolverCUDAHelpers helpers;
	helpers.initialize();

	switch (ssfd) {
	case SparseSolverFactorizationDevice::QR:
		CUSOLVER_STATUS(cusolverSpDcsrlsvqr(helpers.getSolverHandle(),
			systemSize_,
			nonZeroSize,
			helpers.getMatrixDescriptor(),
			d_matVals,
			d_rowCounts,
			d_colIndices,
			d_rhs,
			tol,
			reorder,
			d_solution,
			singularity));
		break;
	case SparseSolverFactorizationDevice::Cholesky:
		CUSOLVER_STATUS(cusolverSpDcsrlsvchol(helpers.getSolverHandle(),
			systemSize_,
			nonZeroSize,
			helpers.getMatrixDescriptor(),
			d_matVals,
			d_rowCounts,
			d_colIndices,
			d_rhs,
			tol,
			reorder,
			d_solution,
			singularity));
		break;
	default:
		CUSOLVER_STATUS(cusolverSpDcsrlsvqr(helpers.getSolverHandle(),
			systemSize_,
			nonZeroSize,
			helpers.getMatrixDescriptor(),
			d_matVals,
			d_rowCounts,
			d_colIndices,
			d_rhs,
			tol,
			reorder,
			d_solution,
			singularity));
		break;
	}

}

template<typename T>
void lss_sparse_solvers_cuda::RealSparseSolverCUDA<lss_types::MemorySpace::Device,T>::
solve_impl(T const* d_matVals, int const* d_rowCounts, int const* d_colIndices, T const* d_rhs,
	T tol, int reorder, T* d_solution, int *singularity, SparseSolverFactorizationDevice ssfd, std::false_type) {

	RealSparseSolverCUDAHelpers helpers;
	helpers.initialize();

	switch (ssfd) {
	case SparseSolverFactorizationDevice::QR:
		CUSOLVER_STATUS(cusolverSpScsrlsvqr(helpers.getSolverHandle(),
			systemSize_,
			nonZeroSize,
			helpers.getMatrixDescriptor(),
			d_matVals,
			d_rowCounts,
			d_colIndices,
			d_rhs,
			tol,
			reorder,
			d_solution,
			singularity));
		break;
	case SparseSolverFactorizationDevice::Cholesky:
		CUSOLVER_STATUS(cusolverSpScsrlsvchol(helpers.getSolverHandle(),
			systemSize_,
			nonZeroSize,
			helpers.getMatrixDescriptor(),
			d_matVals,
			d_rowCounts,
			d_colIndices,
			d_rhs,
			tol,
			reorder,
			d_solution,
			singularity));
		break;
	default:
		CUSOLVER_STATUS(cusolverSpScsrlsvqr(helpers.getSolverHandle(),
			systemSize_,
			nonZeroSize,
			helpers.getMatrixDescriptor(),
			d_matVals,
			d_rowCounts,
			d_colIndices,
			d_rhs,
			tol,
			reorder,
			d_solution,
			singularity));
		break;
	}

}


template<typename T>
void lss_sparse_solvers_cuda::RealSparseSolverCUDA<lss_types::MemorySpace::Host,T>::initialize(int systemSize) {
	// set the size of the linear system:
	systemSize_ = systemSize;

	// clear the containers:
	matrixElements_.clear();
	h_matrixValues_.clear();
	h_vectorValues_.clear();
	h_columnIndices_.clear();
	h_rowCounts_.clear();

	// resize the containers to the correct size:
	h_vectorValues_.resize(systemSize_);
	h_rowCounts_.resize((systemSize_ + 1));
	matrixElements_.setColumns(systemSize_);
	matrixElements_.setRows(systemSize_);
}

template<typename T>
void lss_sparse_solvers_cuda::RealSparseSolverCUDA<lss_types::MemorySpace::Device, T>::initialize(int systemSize) {
	// set the size of the linear system:
	systemSize_ = systemSize;

	// clear the containers:
	matrixElements_.clear();
	h_matrixValues_.clear();
	h_vectorValues_.clear();
	h_columnIndices_.clear();
	h_rowCounts_.clear();

	// resize the containers to the correct size:
	h_vectorValues_.resize(systemSize_);
	h_rowCounts_.resize((systemSize_ + 1));
	matrixElements_.setColumns(systemSize_);
	matrixElements_.setRows(systemSize_);
}


template<typename T>
void lss_sparse_solvers_cuda::RealSparseSolverCUDA<lss_types::MemorySpace::Host, T>::buildCSR() {

	int const nonZeroSize = nonZeroElements();

	h_columnIndices_.resize(nonZeroSize);
	h_matrixValues_.resize(nonZeroSize);

	int nElement{ 0 };
	int nRowElement{ 0 };
	int lastRow{ 0 };
	h_rowCounts_[nRowElement++] = nElement;

	for (int i = 0; i < nonZeroSize; ++i) {

		if (lastRow < std::get<0>(matrixElements_.at(i))) {
			h_rowCounts_[nRowElement++] = i;
			lastRow++;
		}

		h_columnIndices_[i] = std::get<1>(matrixElements_.at(i));
		h_matrixValues_[i] = std::get<2>(matrixElements_.at(i));
	}
	h_rowCounts_[nRowElement] = nonZeroSize;

}

template<typename T>
void lss_sparse_solvers_cuda::RealSparseSolverCUDA<lss_types::MemorySpace::Device, T>::buildCSR() {

	int const nonZeroSize = nonZeroElements();

	h_columnIndices_.resize(nonZeroSize);
	h_matrixValues_.resize(nonZeroSize);

	int nElement{ 0 };
	int nRowElement{ 0 };
	int lastRow{ 0 };
	h_rowCounts_[nRowElement++] = nElement;

	for (int i = 0; i < nonZeroSize; ++i) {

		if (lastRow < std::get<0>(matrixElements_.at(i))) {
			h_rowCounts_[nRowElement++] = i;
			lastRow++;
		}

		h_columnIndices_[i] = std::get<1>(matrixElements_.at(i));
		h_matrixValues_[i] = std::get<2>(matrixElements_.at(i));
	}
	h_rowCounts_[nRowElement] = nonZeroSize;

}

template<typename T>
void lss_sparse_solvers_cuda::RealSparseSolverCUDA<lss_types::MemorySpace::Host,T>::
solve(T* solution, lss_types::SparseSolverFactorizationHost ssfh) {

	buildCSR();

	// get the non-zero size:
	int const nonZeroSize = nonZeroElements();

	// integer for holding index of row where singularity occurs:
	int singularIdx{ 0 };

	// prepare container for solution:
	thrust::host_vector<T> h_solution(systemSize_);

	// get the raw host pointers 
	T* h_matVals = thrust::raw_pointer_cast(h_matrixValues_.data());
	T* h_rhsVals = thrust::raw_pointer_cast(h_vectorValues_.data());
	T* h_sol = thrust::raw_pointer_cast(h_solution.data());
	int* h_col = thrust::raw_pointer_cast(h_columnIndices_.data());
	int* h_row = thrust::raw_pointer_cast(h_rowCounts_.data());

	// call the particular implementation:
	solve_impl(h_matVals, h_row, h_col, h_rhsVals, 0.0, 0, h_sol, &singularIdx, ssfh, std::is_same<T, double>());

	if (singularIdx >= 0) {
		std::cerr << "Sparse matrix is singular at row: " << singularIdx << "\n";
	}
	thrust::copy(h_solution.begin(), h_solution.end(), solution);
}

template<typename T>
void lss_sparse_solvers_cuda::RealSparseSolverCUDA<lss_types::MemorySpace::Device, T>::
solve(T* solution, lss_types::SparseSolverFactorizationDevice ssfd) {

	buildCSR();

	// get the non-zero size:
	int const nonZeroSize = nonZeroElements();

	// integer for holding index of row where singularity occurs:
	int singularIdx{ 0 };

	// prepare container for solution:
	thrust::device_vector<T> d_solution(systemSize_);
	
	// copy to the device constainers:
	thrust::device_vector<T> d_matrixValues = h_matrixValues_;
	thrust::device_vector<T> d_vectorValues = h_vectorValues_;
	thrust::device_vector<int> d_columnIndices = h_columnIndices_;
	thrust::device_vector<int> d_rowCounts = h_rowCounts_;


	// get the raw host pointers 
	T* d_matVals = thrust::raw_pointer_cast(d_matrixValues.data());
	T* d_rhsVals = thrust::raw_pointer_cast(d_vectorValues.data());
	T* d_sol = thrust::raw_pointer_cast(d_solution.data());
	int* d_col = thrust::raw_pointer_cast(d_columnIndices.data());
	int* d_row = thrust::raw_pointer_cast(d_rowCounts.data());

	// call the particular implementation:
	solve_impl(d_matVals, d_row, d_col, d_rhsVals, 0.0, 0, d_sol, &singularIdx, ssfd, std::is_same<T, double>());

	if (singularIdx >= 0) {
		std::cerr << "Sparse matrix is singular at row: " << singularIdx << "\n";
	}
	thrust::copy(d_solution.begin(), d_solution.end(), solution);

}

template<typename T>
std::vector<T> const lss_sparse_solvers_cuda::RealSparseSolverCUDA<lss_types::MemorySpace::Host, T>::
solve(lss_types::SparseSolverFactorizationHost ssfh) {

	buildCSR();

	// get the non-zero size:
	int const nonZeroSize = nonZeroElements();

	// integer for holding index of row where singularity occurs:
	int singularIdx{ 0 };

	// prepare container for solution:
	thrust::host_vector<T> h_solution(systemSize_);

	// get the raw host pointers 
	T* h_matVals = thrust::raw_pointer_cast(h_matrixValues_.data());
	T* h_rhsVals = thrust::raw_pointer_cast(h_vectorValues_.data());
	T* h_sol = thrust::raw_pointer_cast(h_solution.data());
	int* h_col = thrust::raw_pointer_cast(h_columnIndices_.data());
	int* h_row = thrust::raw_pointer_cast(h_rowCounts_.data());

	// call the particular implementation:
	solve_impl(h_matVals, h_row, h_col, h_rhsVals, 0.0, 0, h_sol, &singularIdx, ssfh, std::is_same<T, double>());

	if (singularIdx >= 0) {
		std::cerr << "Sparse matrix is singular at row: " << singularIdx << "\n";
	}


	std::vector<T> solution(systemSize_);
	for (std::size_t t = 0; t < h_solution.size(); ++t)
		solution[t] = h_solution[t];
	return solution;
}

template<typename T>
std::vector<T> const lss_sparse_solvers_cuda::RealSparseSolverCUDA<lss_types::MemorySpace::Device, T>::
solve(lss_types::SparseSolverFactorizationDevice ssfd) {

	buildCSR();

	// get the non-zero size:
	int const nonZeroSize = nonZeroElements();

	// integer for holding index of row where singularity occurs:
	int singularIdx{ 0 };

	// prepare container for solution:
	thrust::device_vector<T> d_solution(systemSize_);

	// copy to the device constainers:
	thrust::device_vector<T> d_matrixValues = h_matrixValues_;
	thrust::device_vector<T> d_vectorValues = h_vectorValues_;
	thrust::device_vector<int> d_columnIndices = h_columnIndices_;
	thrust::device_vector<int> d_rowCounts = h_rowCounts_;


	// get the raw host pointers 
	T* d_matVals = thrust::raw_pointer_cast(d_matrixValues.data());
	T* d_rhsVals = thrust::raw_pointer_cast(d_vectorValues.data());
	T* d_sol = thrust::raw_pointer_cast(d_solution.data());
	int* d_col = thrust::raw_pointer_cast(d_columnIndices.data());
	int* d_row = thrust::raw_pointer_cast(d_rowCounts.data());

	// call the particular implementation:
	solve_impl(d_matVals, d_row, d_col, d_rhsVals, 0.0, 0, d_sol, &singularIdx, ssfd, std::is_same<T, double>());

	if (singularIdx >= 0) {
		std::cerr << "Sparse matrix is singular at row: " << singularIdx << "\n";
	}

	thrust::host_vector<T> h_solution = d_solution;
	std::vector<T> solution(systemSize_);
	for (std::size_t t = 0; t < h_solution.size(); ++t)
		solution[t] = h_solution[t];

	return solution;
}





#endif ///_LSS_SPARSE_SPARSE_CUDA