#pragma once
#if !defined(_LSS_UTILITY)
#define _LSS_UTILITY

#include <algorithm>
#include <memory>
#include <tuple>
#include <vector>

#include "lss_enumerations.h"

namespace lss_utility {

// ==========================================================================
// =========================== smart pointer aliases ========================
// ==========================================================================

template <typename T>
using sptr_t = std::shared_ptr<T>;

template <typename T>
using uptr_t = std::unique_ptr<T>;

// ==========================================================================
// ==================================== Swap ================================
// ==========================================================================

template <typename T>
void swap(T& a, T& b) {
  T t = a;
  a = b;
  b = t;
}

// ==========================================================================
// ==================================== Range ===============================
// ==========================================================================
template <typename T>
class Range {
 private:
  T l_, u_;

 public:
  explicit Range(T lower, T upper) : l_{lower}, u_{upper} {}
  explicit Range() : Range(T{}, T{}) {}

  ~Range() {}

  Range(Range const& copy) : l_{copy.l_}, u_{copy.u_} {}
  Range(Range&& other) noexcept
      : l_{std::move(other.l_)}, u_{std::move(other.u_)} {}

  Range& operator=(Range const& copy) {
    if (this != &copy) {
      l_ = copy.l_;
      u_ = copy.u_;
    }
    return *this;
  }

  Range& operator=(Range&& other) noexcept {
    if (this != &other) {
      l_ = std::move(other.l_);
      u_ = std::move(other.u_);
    }
    return *this;
  }

  inline T lower() const { return l_; }
  inline T upper() const { return u_; }
  inline T spread() const { return (u_ - l_); }
  inline T midPoint() const { return 0.5 * (l_ + u_); }
};

// ==========================================================================
// =============================== FlatMatrix ===============================
// ==========================================================================
using lss_enumerations::FlatMatrixSort;

template <typename T>
struct FlatMatrix {
 private:
  std::vector<std::tuple<std::size_t, std::size_t, T>> container_;
  std::size_t ncols_, nrows_;

 public:
  explicit FlatMatrix(std::size_t nrows, std::size_t ncols)
      : nrows_{nrows}, ncols_{ncols} {}

  explicit FlatMatrix() : FlatMatrix<T>(0, 0) {}

  virtual ~FlatMatrix() {}

  FlatMatrix(FlatMatrix<T> const& copy)
      : ncols_{copy.ncols_}, nrows_{copy.nrows_}, container_{copy.container_} {}

  FlatMatrix(FlatMatrix<T>&& other) noexcept
      : ncols_{std::move(other.ncols_)},
        nrows_{std::move(other.nrows_)},
        container_{std::move(other.container_)} {}

  FlatMatrix<T>& operator=(FlatMatrix<T> const& copy) {
    if (this != &copy) {
      ncols_ = copy.ncols_;
      nrows_ = copy.nrows_;
      container_ = copy.container_;
    }
    return *this;
  }

  FlatMatrix<T>& operator=(FlatMatrix<T>&& other) noexcept {
    if (this != &other) {
      ncols_ = std::move(other.ncols_);
      nrows_ = std::move(other.nrows_);
      container_ = std::move(other.container_);
    }
    return *this;
  }

  inline void setRows(std::size_t nrows) { nrows_ = nrows; }
  inline void setColumns(std::size_t ncols) { ncols_ = ncols; }
  inline std::size_t rows() const { return nrows_; }
  inline std::size_t columns() const { return ncols_; }
  inline std::size_t size() const { return container_.size(); }
  inline void clear() { container_.clear(); }

  inline void emplace_back(std::size_t rowIdx, std::size_t colIdx, T value) {
    LSS_ASSERT(rowIdx < nrows_, " rowIdx is outside <0," << nrows_ << ")");
    LSS_ASSERT(colIdx < ncols_, " colIdx is outside <0," << ncols_ << ")");
    container_.emplace_back(std::make_tuple(rowIdx, colIdx, value));
  }

  inline void emplace_back(std::tuple<std::size_t, std::size_t, T> tuple) {
    LSS_ASSERT(std::get<0>(tuple) < nrows_,
               " rowIdx is outside <0," << nrows_ << ")");
    LSS_ASSERT(std::get<1>(tuple) < ncols_,
               " colIdx is outside <0," << ncols_ << ")");
    container_.emplace_back(std::move(tuple));
  }

  void sort(FlatMatrixSort sort);

  std::tuple<std::size_t, std::size_t, T> const& at(std::size_t idx) const {
    return container_.at(idx);
  }
};

}  // namespace lss_utility

template <typename T>
void lss_utility::FlatMatrix<T>::sort(lss_types::FlatMatrixSort sort) {
  if (sort == lss_types::FlatMatrixSort::RowMajor) {
    std::sort(container_.begin(), container_.end(),
              [this](std::tuple<std::size_t, std::size_t, T> const& lhs,
                     std::tuple<std::size_t, std::size_t, T> const& rhs) {
                std::size_t const flatIdxLhs =
                    std::get<1>(lhs) + nrows_ * std::get<0>(lhs);
                std::size_t const flatIdxRhs =
                    std::get<1>(rhs) + nrows_ * std::get<0>(rhs);
                return (flatIdxLhs < flatIdxRhs);
              });
  } else {
    std::sort(container_.begin(), container_.end(),
              [this](std::tuple<std::size_t, std::size_t, T> const& lhs,
                     std::tuple<std::size_t, std::size_t, T> const& rhs) {
                std::size_t const flatIdxLhs =
                    std::get<0>(lhs) + ncols_ * std::get<1>(lhs);
                std::size_t const flatIdxRhs =
                    std::get<0>(rhs) + ncols_ * std::get<1>(rhs);
                return (flatIdxLhs < flatIdxRhs);
              });
  }
}

#endif  ///_LSS_UTILITY
