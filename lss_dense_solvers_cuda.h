#pragma once
#if !defined(_LSS_DENSE_SOLVERS_CUDA)
#define _LSS_DENSE_SOLVERS_CUDA


#include<type_traits>

#include<thrust/host_vector.h>
#include<thrust/device_vector.h>


#include"lss_macros.h"
#include"lss_types.h"
#include"lss_utility.h"

#include<cusolverDn.h>

// Very good examples:
// https://github.com/NVIDIA/cuda-samples/blob/master/Samples/cuSolverDn_LinearSolver/cuSolverDn_LinearSolver.cpp

namespace lss_dense_solvers_cuda{

	using lss_utility::FlatMatrix;

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

	public:
		explicit RealDenseSolverCUDA() {}
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
	
	};



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












#endif ///_LSS_DENSE_SOLVERS_CUDA