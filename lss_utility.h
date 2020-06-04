#pragma once
#if !defined(_LSS_UTILITY)
#define _LSS_UTILITY

#include<vector>
#include<tuple>

namespace lss_utility {

	// ==========================================================================
	// =============================== FlatMatrix ===============================
	// ==========================================================================

	template<typename T>
	struct FlatMatrix {
	private:
		std::vector<std::tuple<int, int, T>> container_;
		int ncols_, nrows_;
	public:
		explicit FlatMatrix(int nrows, int ncols)
			:nrows_{ nrows }, ncols_{ ncols }{}

		explicit FlatMatrix()
			:FlatMatrix<T>(0, 0) {}

		virtual ~FlatMatrix() {}

		FlatMatrix(FlatMatrix<T> const& copy)
			:ncols_{ copy.ncols_ },
			nrows_{ copy.nrows_ },
			container_{ copy.container_ }{}

		FlatMatrix(FlatMatrix<T>&& other)noexcept
			:ncols_{ std::move(other.ncols_) },
			nrows_{ std::move(other.nrows_) },
			container_{ std::move(other.container_) }{}

		FlatMatrix<T>& operator=(FlatMatrix<T> const& copy) {
			if (this != &copy) {
				ncols_ = copy.ncols_;
				nrows_ = copy.nrows_;
				container_ = copy.container_;
			}
			return *this;
		}

		FlatMatrix<T>& operator=(FlatMatrix<T>&& other)noexcept {
			if (this != &other) {
				ncols_ = std::move(other.ncols_);
				nrows_ = std::move(other.nrows_);
				container_ = std::move(other.container_);
			}
			return *this;
		}


		inline void setRows(int nrows) { nrows_ = nrows; }
		inline void setColumns(int ncols) { ncols_ = ncols; }
		inline int const rows()const { return nrows_; }
		inline int const columns()const { return ncols_; }
		inline int const size()const { return container_.size(); }
		inline void clear() { container_.clear(); }

		inline void emplace_back(int rowIdx, int colIdx, T value) {
			LSS_ASSERT((rowIdx >= 0 && rowIdx < nrows_),
				" rowIdx is outside <0," << nrows_ << ")");
			LSS_ASSERT((colIdx >= 0 && colIdx < ncols_),
				" colIdx is outside <0," << ncols_ << ")");
			container_.emplace_back(std::make_tuple(rowIdx, colIdx, value));
		}

		inline void emplace_back(std::tuple<int, int, T> tuple) {
			LSS_ASSERT((std::get<0>(tuple) >= 0 && std::get<0>(tuple) < nrows_),
				" rowIdx is outside <0," << nrows_ << ")");
			LSS_ASSERT((std::get<1>(tuple) >= 0 && std::get<1>(tuple) < ncols_),
				" colIdx is outside <0," << ncols_ << ")");
			container_.emplace_back(std::move(tuple));
		}

		std::tuple<int, int, T> const& at(int idx)const {
			return container_.at(idx);
		}
	};


}




#endif ///_LSS_UTILITY