#pragma once
#if !defined(_LSS_DENSE_SOLVERS_CUDA)
#define _LSS_DENSE_SOLVERS_CUDA


#include<type_traits>
#include<algorithm>
#include<thrust/host_vector.h>
#include<thrust/device_vector.h>


#include"lss_macros.h"
#include"lss_types.h"
#include"lss_helpers.h"
#include"lss_utility.h"
#include"lss_dense_solvers_policy.h"

#include<cusolverDn.h>

namespace lss_dense_solvers_cuda{

	using lss_utility::FlatMatrix;
	using lss_types::FlatMatrixSort;
	using lss_helpers::RealDenseSolverCUDAHelpers;
	using lss_dense_solvers_policy::DenseSolverQR;

	template<typename T,
		typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
	class RealDenseSolverCUDA {};


	template<typename T>
	class RealDenseSolverCUDA<T> {
	private:
		int matrixRows_;
		int matrixCols_;
		FlatMatrix<T> matrixElements_;

		thrust::host_vector<T> h_matrixValues_;
		thrust::host_vector<T> h_rhsValues_;

		void populate();

	public:
		explicit RealDenseSolverCUDA():
			matrixCols_{ 0 }, matrixRows_{0} {}
		virtual ~RealDenseSolverCUDA(){}

		void initialize(int matrixRows,int matrixColumns);

		inline void setRhs(std::vector<T> const& rhs) {
			LSS_ASSERT(rhs.size() == matrixRows_, 
				" Right-hand side vector of the system has incorrect size.");
			for (std::size_t t = 0; t < rhs.size(); ++t)
				h_rhsValues_[t] = rhs[t];
		}

		inline void setRhs(T* rhs) {
			thrust::copy(rhs, rhs + matrixRows_, h_rhsValues_.begin());
		}

		inline void setRhsValue(int idx, T value) {
			LSS_ASSERT(((idx >= 0) && (idx < matrixRows_)),
				"idx is outside range");
			h_rhsValues_[idx] = value;
		}

		inline void setFlatDenseMatrix(FlatMatrix<T> matrix) {
			LSS_ASSERT((matrix.rows()== matrixRows_) && (matrix.columns() == matrixCols_),
				" FlatMatrix has incorrect number of rows or columns");
			matrixElements_ = std::move(matrix);
		}

		inline void setFlatDenseMatrixValue(int rowIdx, int colIdx, T value) {
			LSS_ASSERT((rowIdx < matrixRows_) && (rowIdx >= 0),
				" row index is out of range");
			LSS_ASSERT((colIdx < matrixCols_) && (colIdx >= 0),
				" column index is out of range");
			matrixElements_.emplace_back(rowIdx, colIdx, value);
		}

		inline void setFlatDenseMatrixValue(std::tuple<int, int, T> triplet) {
			LSS_ASSERT((std::get<0>(triplet) < matrixRows_) && (std::get<0>(triplet) >= 0),
				" row index is out of range");
			LSS_ASSERT((std::get<1>(triplet) < matrixCols_) && (std::get<1>(triplet) >= 0),
				" column index is out of range");
			matrixElements_.emplace_back(std::move(triplet));
		}
	
		template<template<typename> typename DenseSolverPolicy = DenseSolverQR>
		void solve(T* solution);

		template<template<typename> typename DenseSolverPolicy = DenseSolverQR>
		std::vector<T> const solve();

	};



}


template<typename T>
void lss_dense_solvers_cuda::RealDenseSolverCUDA<T>::populate() {

	// CUDA Dense solver is column-major:
	matrixElements_.sort(FlatMatrixSort::ColumnMajor);

	for (std::size_t t = 0; t < matrixElements_.size(); ++t) {
		h_matrixValues_[t] = std::get<2>(matrixElements_.at(t));
	}
}

template<typename T>
void lss_dense_solvers_cuda::RealDenseSolverCUDA<T>::initialize(int matrixRows, int matrixColumns) {
	// set the sizes of the system components:
	matrixCols_ = matrixColumns;
	matrixRows_ = matrixRows;

	// clear the containers:
	matrixElements_.clear();
	h_matrixValues_.clear();
	h_rhsValues_.clear();

	// resize the containers to the correct size:
	h_matrixValues_.resize(matrixCols_ * matrixRows_);
	h_rhsValues_.resize(matrixRows_);
}



template<typename T>
template<template<typename> typename DenseSolverPolicy>
void lss_dense_solvers_cuda::RealDenseSolverCUDA<T>::solve(T* solution) {
	
	populate();

	// get the dimensions:
	int const lda = std::max(matrixElements_.rows(), matrixElements_.columns());
	int const m  = std::min(matrixElements_.rows(), matrixElements_.columns());
	int const ldb = h_rhsValues_.size();

	// step 1: create device containers:
	thrust::device_vector<T> d_matrixValues = h_matrixValues_;
	thrust::device_vector<T> d_rhsValues = h_rhsValues_;
	thrust::device_vector<T> d_solution(m);

	// step 2: cast to raw pointers:
	T* d_matVals = thrust::raw_pointer_cast(d_matrixValues.data());
	T* d_rhsVals = thrust::raw_pointer_cast(d_rhsValues.data());
	T* d_sol = thrust::raw_pointer_cast(d_solution.data());
	
	// create the helpers:
	RealDenseSolverCUDAHelpers helpers;
	helpers.initialize();

	// call the DenseSolverPolicy:
	DenseSolverPolicy<T>::solve(helpers.getDenseSolverHandle(), helpers.getCublasHandle(),
		m, d_matVals, lda, d_rhsVals, d_sol);

	thrust::copy(d_solution.begin(), d_solution.end(), solution);

}

template<typename T>
template<template<typename> typename DenseSolverPolicy>
std::vector<T> const lss_dense_solvers_cuda::RealDenseSolverCUDA<T>::solve() {

	populate();

	// get the dimensions:
	int const lda = std::max(matrixElements_.rows(), matrixElements_.columns());
	int const m = std::min(matrixElements_.rows(), matrixElements_.columns());
	int const ldb = h_rhsValues_.size();

	// prepare container for solution:
	thrust::host_vector<T> h_solution(m);

	// step 1: create device vectors:
	thrust::device_vector<T> d_matrixValues = h_matrixValues_;
	thrust::device_vector<T> d_rhsValues = h_rhsValues_;
	thrust::device_vector<T> d_solution = h_solution;

	// step 2: cast to raw pointers:
	T* d_matVals = thrust::raw_pointer_cast(d_matrixValues.data());
	T* d_rhsVals = thrust::raw_pointer_cast(d_rhsValues.data());
	T* d_sol = thrust::raw_pointer_cast(d_solution.data());

	// create the helpers:
	RealDenseSolverCUDAHelpers helpers;
	helpers.initialize();

	// call the DenseSolverPolicy:
	DenseSolverPolicy<T>::solve(helpers.getDenseSolverHandle(), helpers.getCublasHandle(),
		m, d_matVals, lda, d_rhsVals, d_sol);

	std::vector<T> solution(h_solution.size());
	thrust::copy(d_solution.begin(), d_solution.end(), solution.begin());

	return solution;
}








#endif ///_LSS_DENSE_SOLVERS_CUDA