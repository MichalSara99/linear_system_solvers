#pragma once
#if !defined(_LSS_CONTAINERS)
#define _LSS_CONTAINERS

#include <algorithm>
#include <iterator>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

#include "lss_enumerations.h"

namespace lss_containers {

// =========================================================================
// ===================== container_2d container ============================
// =========================================================================
template <template <typename, typename> typename container, typename fp_type,
          typename alloc>
class container_2d {
 private:
  std::size_t rows_;
  std::size_t columns_;
  std::vector<container<fp_type, alloc>> data_;

  explicit container_2d() {}

 public:
  typedef fp_type value_type;
  typedef container<fp_type, alloc> element_type;

  explicit container_2d(std::size_t rows, std::size_t columns)
      : rows_{rows}, columns_{columns} {
    for (std::size_t r = 0; r < rows; ++r) {
      data_.emplace_back(
          container<fp_type, alloc>(static_cast<std::size_t>(columns)));
    }
  }

  explicit container_2d(std::size_t rows, std::size_t columns, fp_type value)
      : rows_{rows}, columns_{columns} {
    for (std::size_t r = 0; r < rows; ++r) {
      data_.emplace_back(
          container<fp_type, alloc>(static_cast<std::size_t>(columns), value));
    }
  }

  ~container_2d() {}

  container_2d(container_2d const& copy)
      : rows_{copy.rows_}, columns_{copy.columns_}, data_{copy.data_} {}

  container_2d(container_2d&& other) noexcept
      : rows_{std::move(other.rows_)},
        columns_{std::move(other.columns_)},
        data_{std::move(other.data_)} {}

  container_2d& operator=(container_2d const& copy) {
    if (this != &copy) {
      rows_ = copy.rows_;
      columns_ = copy.columns_;
      data_ = copy.data_;
    }
    return *this;
  }

  container_2d& operator=(container_2d&& other) noexcept {
    if (this != &other) {
      rows_ = std::move(other.rows_);
      columns_ = std::move(other.columns_);
      data_ = std::move(other.data_);
    }
    return *this;
  }

  std::size_t rows() const { return rows_; }
  std::size_t columns() const { return columns_; }
  std::size_t total_size() const { return rows_ * columns_; }

  // return value from container_2d at potision (row_idx,col_idx)
  fp_type operator()(std::size_t row_idx, std::size_t col_idx) const {
    LSS_ASSERT(row_idx < rows_, "Outside of row range");
    LSS_ASSERT(col_idx < columns_, "Outside of column range");
    return data_[row_idx][col_idx];
  }

  // return row container from container_2d at potision (row_idx)
  container<fp_type, alloc> operator()(std::size_t row_idx) const {
    LSS_ASSERT(row_idx < rows_, "Outside of row range");
    return data_[row_idx];
  }

  // return value from container_2d at potision (row_idx,col_idx)
  fp_type at(std::size_t row_idx, std::size_t col_idx) const {
    return operator()(row_idx, col_idx);
  }

  // return row container from container_2d at potision (row_idx)
  container<fp_type, alloc> at(std::size_t row_idx) const {
    return operator()(row_idx);
  }

  // place row_container at row position (row_idx)
  void operator()(std::size_t row_idx,
                  container<fp_type, alloc> const& row_container) {
    LSS_ASSERT(row_idx < rows_, "Outside of row range");
    LSS_ASSERT(row_container.size() == columns_, "Outside of column range");
    data_[row_idx] = std::move(row_container);
  }

  // place ro_container at row position starting at col_start_idx and ending at
  // col_end_idx
  void operator()(std::size_t row_idx,
                  container<fp_type, alloc> const& row_container,
                  std::size_t col_start_idx, std::size_t col_end_idx) {
    LSS_ASSERT(col_start_idx <= col_end_idx,
               "col_start_idx must be smaller or equal then col_end_idx");
    LSS_ASSERT(row_idx < rows_, "Outside of row range");
    LSS_ASSERT(col_start_idx < columns_,
               "col_start_idx is outside of column range");
    LSS_ASSERT(col_end_idx < columns_,
               "col_end_idx is outside of column range");
    const std::size_t len = col_end_idx - col_start_idx + 1;
    LSS_ASSERT(len <= row_container.size(),
               "Inserted length is bigger then the row_container");
    container<fp_type, alloc> cont = this->at(row_idx);
    std::size_t c = 0;
    for (std::size_t t = col_start_idx; t <= col_end_idx; ++t, ++c) {
      cont[t] = row_container[c];
    }
    this->operator()(row_idx, std::move(cont));
  }

  // place value at position (row_idx,col_idx)
  void operator()(std::size_t row_idx, std::size_t col_idx, fp_type value) {
    LSS_ASSERT(row_idx < rows_, "Outside of row range");
    LSS_ASSERT(col_idx < columns_, "Outside of column range");
    data_[row_idx][col_idx] = value;
  }

  // returns data as flat vector row-wise
  std::vector<fp_type, alloc> const data() const {
    std::vector<fp_type, alloc> d(rows_ * columns_);
    auto itr = d.begin();
    for (std::size_t r = 0; r < rows_; ++r) {
      auto row = at(r);
      std::copy(row.begin(), row.end(), itr);
      std::advance(itr, columns_);
    }
    return d;
  }
  template <typename in_itr>
  void set_data(in_itr first, in_itr last) {
    const int dist = std::distance(first, last);
    LSS_ASSERT((rows_ * columns_) == dist,
               "Source data is either too long or too short");
    for (std::size_t r = 0; r < rows_; ++r) {
      for (std::size_t c = 0; c < columns_; ++c) {
        this->operator()(r, c, *first);
        ++first;
      }
    }
  }
};

template <template <typename, typename> typename container, typename fp_type,
          typename alloc>
void copy(container_2d<container, fp_type, alloc>& dest,
          container_2d<container, fp_type, alloc> const& src) {
  LSS_ASSERT(dest.columns() == src.columns(),
             "dest and src must have same dimensions");
  LSS_ASSERT(dest.rows() == src.rows(),
             "dest and src must have same dimensions");
  for (std::size_t r = 0; r < dest.rows(); ++r) {
    dest(r, src.at(r));
  }
}

// concrete single-precision floating-point type
using matrix_float = container_2d<std::vector, float, std::allocator<float>>;
// concrete double-precision floating-point type
using matrix_double = container_2d<std::vector, double, std::allocator<double>>;

// ==========================================================================
// ============================== flat_matrix ===============================
// ==========================================================================
using lss_enumerations::flat_matrix_sort_enum;

template <typename T>
struct flat_matrix {
 private:
  std::vector<std::tuple<std::size_t, std::size_t, T>> container_;
  std::vector<std::size_t> column_cnt_;
  std::vector<T> diagonal_;
  std::size_t ncols_, nrows_;

  explicit flat_matrix() {}

 public:
  explicit flat_matrix(std::size_t nrows, std::size_t ncols)
      : nrows_{nrows}, ncols_{ncols} {
    column_cnt_.resize(nrows);
    diagonal_.resize(nrows);
  }

  virtual ~flat_matrix() {}

  flat_matrix(flat_matrix<T> const& copy)
      : ncols_{copy.ncols_},
        nrows_{copy.nrows_},
        container_{copy.container_},
        column_cnt_{copy.column_cnt_},
        diagonal_{copy.diagonal_} {}

  flat_matrix(flat_matrix<T>&& other) noexcept
      : ncols_{std::move(other.ncols_)},
        nrows_{std::move(other.nrows_)},
        container_{std::move(other.container_)},
        column_cnt_{std::move(other.column_cnt_)},
        diagonal_{std::move(other.diagonal_)} {}

  flat_matrix<T>& operator=(flat_matrix<T> const& copy) {
    if (this != &copy) {
      ncols_ = copy.ncols_;
      nrows_ = copy.nrows_;
      container_ = copy.container_;
      column_cnt_ = copy.column_cnt_;
      diagonal_ = copy.diagonal_;
    }
    return *this;
  }

  flat_matrix<T>& operator=(flat_matrix<T>&& other) noexcept {
    if (this != &other) {
      ncols_ = std::move(other.ncols_);
      nrows_ = std::move(other.nrows_);
      container_ = std::move(other.container_);
      column_cnt_ = std::move(other.column_cnt_);
      diagonal_ = std::move(other.diagonal_);
    }
    return *this;
  }

  inline std::size_t rows() const { return nrows_; }
  inline std::size_t columns() const { return ncols_; }
  inline std::size_t size() const { return container_.size(); }
  inline void clear() { container_.clear(); }
  inline T diagonal_at_row(std::size_t row_idx) const {
    return diagonal_[row_idx];
  }
  inline std::size_t non_zero_column_size(std::size_t row_idx) const {
    return column_cnt_[row_idx];
  }

  inline void emplace_back(std::size_t row_idx, std::size_t col_idx, T value) {
    LSS_ASSERT(row_idx < nrows_, " rowIdx is outside <0," << nrows_ << ")");
    LSS_ASSERT(col_idx < ncols_, " colIdx is outside <0," << ncols_ << ")");
    container_.emplace_back(std::make_tuple(row_idx, col_idx, value));
    column_cnt_[row_idx]++;
    if (row_idx == col_idx) diagonal_[row_idx] = value;
  }

  inline void emplace_back(std::tuple<std::size_t, std::size_t, T> tuple) {
    LSS_ASSERT(std::get<0>(tuple) < nrows_,
               " rowIdx is outside <0," << nrows_ << ")");
    LSS_ASSERT(std::get<1>(tuple) < ncols_,
               " colIdx is outside <0," << ncols_ << ")");
    container_.emplace_back(std::move(tuple));
    column_cnt_[std::get<0>(tuple)]++;
    if (std::get<1>(tuple) == std::get<0>(tuple))
      diagonal_[std::get<1>(tuple)] = std::get<2>(tuple);
  }

  void sort(flat_matrix_sort_enum sort);

  std::tuple<std::size_t, std::size_t, T> const& at(std::size_t idx) const {
    return container_.at(idx);
  }
};

}  // namespace lss_containers

template <typename T>
void lss_containers::flat_matrix<T>::sort(
    lss_enumerations::flat_matrix_sort_enum sort) {
  if (sort == lss_enumerations::flat_matrix_sort_enum::RowMajor) {
    std::sort(container_.begin(), container_.end(),
              [this](std::tuple<std::size_t, std::size_t, T> const& lhs,
                     std::tuple<std::size_t, std::size_t, T> const& rhs) {
                std::size_t const flat_idx_lhs =
                    std::get<1>(lhs) + nrows_ * std::get<0>(lhs);
                std::size_t const flat_idx_rhs =
                    std::get<1>(rhs) + nrows_ * std::get<0>(rhs);
                return (flat_idx_lhs < flat_idx_rhs);
              });
  } else {
    std::sort(container_.begin(), container_.end(),
              [this](std::tuple<std::size_t, std::size_t, T> const& lhs,
                     std::tuple<std::size_t, std::size_t, T> const& rhs) {
                std::size_t const flat_idx_lhs =
                    std::get<0>(lhs) + ncols_ * std::get<1>(lhs);
                std::size_t const flat_idx_rhs =
                    std::get<0>(rhs) + ncols_ * std::get<1>(rhs);
                return (flat_idx_lhs < flat_idx_rhs);
              });
  }
}

#endif  ///_LSS_CONTAINERS
