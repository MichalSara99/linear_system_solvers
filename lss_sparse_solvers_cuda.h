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
#include"lss_sparse_solvers_policy.h"

namespace lss_sparse_solvers_cuda {

	using lss_types::MemorySpace;
	using lss_types::FlatMatrixSort;
	using lss_utility::FlatMatrix;
	using lss_helpers::RealSparseSolverCUDAHelpers;
	using lss_sparse_solvers_policy::SparseSolverDevice;
	using lss_sparse_solvers_policy::SparseSolverDeviceCholesky;
	using lss_sparse_solvers_policy::SparseSolverDeviceQR;
	using lss_sparse_solvers_policy::SparseSolverHost;
	using lss_sparse_solvers_policy::SparseSolverHostQR;
	using lss_sparse_solvers_policy::SparseSolverHostLU;
	using lss_sparse_solvers_policy::SparseSolverHostCholesky;



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
		FlatMatrix<T> matrixElements_;

		thrust::host_vector<T> h_matrixValues_;
		thrust::host_vector<T> h_vectorValues_; // of systemSize length
		thrust::host_vector<int> h_columnIndices_; 
		thrust::host_vector<int> h_rowCounts_; // of systemSize + 1 length

		void buildCSR();

	public:
		typedef T value_type;
		explicit RealSparseSolverCUDA()
			:systemSize_{ 0 } {}
		virtual ~RealSparseSolverCUDA(){}

		RealSparseSolverCUDA(RealSparseSolverCUDA const&) = delete;
		RealSparseSolverCUDA& operator=(RealSparseSolverCUDA const&) = delete;
		RealSparseSolverCUDA(RealSparseSolverCUDA &&) = delete;
		RealSparseSolverCUDA& operator=(RealSparseSolverCUDA &&) = delete;

		void initialize(int systemSize);

		inline int const nonZeroElements()const { return matrixElements_.size(); }

		template<template<typename T,typename Alloc> typename Container = std::vector,
			typename Alloc = std::allocator<T>>
		inline void setRhs(Container<T,Alloc> const &rhs) {
			LSS_ASSERT(rhs.size() == systemSize_,
				" rhs has incorrect size");
			thrust::copy(rhs.begin(), rhs.end(), h_vectorValues_.begin());
		}


		inline void setRhsValue(int idx, T value) {
			LSS_ASSERT(((idx >= 0) && (idx < systemSize_)), "idx is outside range");
			h_vectorValues_[idx] = value;
		}

		inline void setFlatSparseMatrix(FlatMatrix<T> matrix) {
			matrixElements_ = std::move(matrix);
		}

		inline void setFlatSparseMatrixValue(int rowIdx, int colIdx, T value) {
			matrixElements_.emplace_back(rowIdx, colIdx, value);
		}

		inline void setFlatSparseMatrixValue(std::tuple<int, int, T> triplet) {
			matrixElements_.emplace_back(std::move(triplet));
		}

		template<template<typename> typename SparseSolverHostPolicy = SparseSolverHostQR,
			template<typename T,typename Alloc> typename Container = std::vector,
			typename Alloc = std::allocator<T>,
				typename  = typename std::enable_if<std::is_base_of<SparseSolverHost<T>, SparseSolverHostPolicy<T>>::value>::type>
		void solve(Container<T,Alloc> &solution);

		template<template<typename T> typename  SparseSolverHostPolicy = SparseSolverHostQR,
			template<typename T,typename Alloc> typename Container = std::vector,
			typename Alloc = std::allocator<T>,
			typename = typename std::enable_if<std::is_base_of<SparseSolverHost<T>, SparseSolverHostPolicy<T>>::value>::type>
		Container<T,Alloc> const solve();

	};


	// ===============================================================================
	// ========= RealSparseSolverCUDA partial specialization for DEVICE ==============
	// ===============================================================================

	template<typename T>
	class RealSparseSolverCUDA<MemorySpace::Device, T> {
	protected:
		int systemSize_;
		FlatMatrix<T> matrixElements_;

		thrust::host_vector<T> h_matrixValues_;
		thrust::host_vector<T> h_vectorValues_; // of systemSize length
		thrust::host_vector<int> h_columnIndices_;
		thrust::host_vector<int> h_rowCounts_; // of systemSize + 1 length

		void buildCSR();

	public:
		typedef T value_type;
		explicit RealSparseSolverCUDA()
			:systemSize_{0} {}
		virtual ~RealSparseSolverCUDA() {}

		RealSparseSolverCUDA(RealSparseSolverCUDA const&) = delete;
		RealSparseSolverCUDA& operator=(RealSparseSolverCUDA const&) = delete;
		RealSparseSolverCUDA(RealSparseSolverCUDA &&) = delete;
		RealSparseSolverCUDA& operator=(RealSparseSolverCUDA &&) = delete;

		void initialize(int systemSize);

		inline int const nonZeroElements()const { return matrixElements_.size(); }

		template<template<typename T, typename Alloc> typename Container = std::vector,
			typename Alloc = std::allocator<T>>
		inline void setRhs(Container<T,Alloc> const& rhs) {
			LSS_ASSERT(rhs.size() == systemSize_,
				" rhs has incorrect size");
			thrust::copy(rhs.begin(), rhs.end(), h_vectorValues_.begin());
		}

		inline void setRhsValue(int idx, T value) {
			LSS_ASSERT(((idx >= 0) && (idx < systemSize_)), "idx is outside range");
			h_vectorValues_[idx] = value;
		}

		inline void setFlatSparseMatrix(FlatMatrix<T> matrix) {
			matrixElements_ = std::move(matrix);
		}

		inline void setFlatSparseMatrixValue(int rowIdx, int colIdx, T value) {
			matrixElements_.emplace_back(rowIdx, colIdx, value);
		}

		inline void setFlatSparseMatrixValue(std::tuple<int, int, T> triplet) {
			matrixElements_.emplace_back(std::move(triplet));
		}

		template<template<typename T> typename SparseSolverDevicePolicy = SparseSolverDeviceQR,
			template<typename T,typename Alloc> typename Container = std::vector,
			typename Alloc = std::allocator<T>,
			typename =typename std::enable_if<std::is_base_of<SparseSolverDevice<T>, SparseSolverDevicePolicy<T>>::value>::type>
		void solve(Container<T,Alloc> &solution);

		template<
			template<typename T> typename SparseSolverDevicePolicy = SparseSolverDeviceQR,
			template<typename T,typename Alloc> typename Container = std::vector,
			typename Alloc = std::allocator<T>,
			typename = typename std::enable_if<std::is_base_of<SparseSolverDevice<T>, SparseSolverDevicePolicy<T>>::value>::type>
		Container<T,Alloc> const solve();

	};

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

	// CUDA sparse solver is row-major:
	matrixElements_.sort(FlatMatrixSort::RowMajor);

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

	// CUDA sparse solver is row-major:
	matrixElements_.sort(FlatMatrixSort::RowMajor);

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
template<template<typename T> typename SparseSolverHostPolicy,
	template<typename T,typename Alloc> typename Container,
	typename Alloc,
	typename>
void lss_sparse_solvers_cuda::RealSparseSolverCUDA<lss_types::MemorySpace::Host,T>::
solve(Container<T,Alloc> &solution) {

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

	// create the helpers:
	RealSparseSolverCUDAHelpers helpers;
	helpers.initialize();
	// call the SparseSolverHostPolicy:
	SparseSolverHostPolicy<T>::solve(helpers.getSolverHandle(), helpers.getMatrixDescriptor(),
		systemSize_, nonZeroSize, h_matVals, h_row, h_col, h_rhsVals, 0.0, 0, h_sol, &singularIdx);

	if (singularIdx >= 0) {
		std::cerr << "Sparse matrix is singular at row: " << singularIdx << "\n";
	}
	thrust::copy(h_solution.begin(), h_solution.end(), solution.begin());
}

template<typename T>
template<
	template<typename T> typename SparseSolverDevicePolicy,
	template<typename T,typename Alloc> typename Container,
	typename Alloc,
	typename>
void lss_sparse_solvers_cuda::RealSparseSolverCUDA<lss_types::MemorySpace::Device, T>::
solve(Container<T,Alloc> &solution) {

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

	// create the helpers:
	RealSparseSolverCUDAHelpers helpers;
	helpers.initialize();
	// call the SparseSolverDevicePolicy:
	SparseSolverDevicePolicy<T>::solve(helpers.getSolverHandle(), helpers.getMatrixDescriptor(),
		systemSize_, nonZeroSize, d_matVals, d_row, d_col, d_rhsVals, 0.0, 0, d_sol, &singularIdx);

	if (singularIdx >= 0) {
		std::cerr << "Sparse matrix is singular at row: " << singularIdx << "\n";
	}
	thrust::copy(d_solution.begin(), d_solution.end(), solution.begin());

}

template<typename T>
template<
		template<typename T> typename SparseSolverHostPolicy,
		template<typename T,typename Alloc> typename Container,
		typename Alloc,
		typename>
Container<T,Alloc> const lss_sparse_solvers_cuda::RealSparseSolverCUDA<lss_types::MemorySpace::Host, T>::
solve() {

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

	// create the helpers:
	RealSparseSolverCUDAHelpers helpers;
	helpers.initialize();
	// call the SparseSolverHostPolicy:
	SparseSolverHostPolicy<T>::solve(helpers.getSolverHandle(), helpers.getMatrixDescriptor(),
		systemSize_, nonZeroSize, h_matVals, h_row, h_col, h_rhsVals, 0.0, 0, h_sol, &singularIdx);

	if (singularIdx >= 0) {
		std::cerr << "Sparse matrix is singular at row: " << singularIdx << "\n";
	}

	Container<T,Alloc> solution(h_solution.size());
	thrust::copy(h_solution.begin(), h_solution.end(), solution.begin());
	return solution;
}

template<typename T>
template<
	template<typename T> typename SparseSolverDevicePolicy,
	template<typename T, typename Alloc> typename Container,
	typename Alloc,
	typename>
Container<T,Alloc> const lss_sparse_solvers_cuda::RealSparseSolverCUDA<lss_types::MemorySpace::Device, T>::
solve() {

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

	// create the helpers:
	RealSparseSolverCUDAHelpers helpers;
	helpers.initialize();
	// call the SparseSolverDevicePolicy:
	SparseSolverDevicePolicy<T>::solve(helpers.getSolverHandle(), helpers.getMatrixDescriptor(),
		systemSize_, nonZeroSize, d_matVals, d_row, d_col, d_rhsVals, 0.0, 0, d_sol, &singularIdx);

	if (singularIdx >= 0) {
		std::cerr << "Sparse matrix is singular at row: " << singularIdx << "\n";
	}

	Container<T,Alloc> solution(systemSize_);
	thrust::copy(d_solution.begin(), d_solution.end(), solution.begin());

	return solution;
}





#endif ///_LSS_SPARSE_SPARSE_CUDA