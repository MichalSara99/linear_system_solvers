#pragma once
#if !defined(_LSS_UTILITY)
#define _LSS_UTILITY

#include <algorithm>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

#include "lss_enumerations.h"

namespace lss_utility {

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

 protected:
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

  // place value at position (row_idx,col_idx)
  void operator()(std::size_t row_idx, std::size_t col_idx, fp_type value) {
    LSS_ASSERT(row_idx < rows_, "Outside of row range");
    LSS_ASSERT(col_idx < columns_, "Outside of column range");
    data_[row_idx][col_idx] = value;
  }
};
// =========================================================================
// ===================== PDE coefficient holder ============================
// =========================================================================
template <typename... coeff_type>
using coefficient_holder = std::tuple<coeff_type...>;

// =========================================================================
// ================================ pi =====================================
// =========================================================================
template <typename T>
static T const pi() {
  return (3.14159265358979);
}

// ==========================================================================
// ================================= NaN ====================================
// ==========================================================================

template <typename T>
static constexpr T NaN() {
  return std::numeric_limits<T>::quiet_NaN();
}

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
// ================================ norm_cdf ================================
// ==========================================================================

template <typename T>
double norm_cdf(T x) {
  T ind = T{};
  if (x <= 0.0) ind = 1.0;
  x = std::abs(x);
  T const cst = 1.0 / (std::sqrt(2.0 * pi<T>()));
  T const first = std::exp(-0.5 * x * x);
  T const second = 0.226 + 0.64 * x + 0.33 * std::sqrt(x * x + 3.0);
  T const res = 1.0 - ((first / second) * cst);
  return std::abs(ind - res);
}

// ==========================================================================
// =========================== black_scholes_exact ==========================
// ==========================================================================
template <typename T>
class black_scholes_exact {
 private:
  T time_;
  T strike_;
  T vol_;
  T maturity_;
  T rate_;

 protected:
  explicit black_scholes_exact(){};

 public:
  explicit black_scholes_exact(T time, T strike, T rate, T volatility,
                               T maturity)
      : time_{time},
        strike_{strike},
        rate_{rate},
        vol_{volatility},
        maturity_{maturity} {}

  ~black_scholes_exact() {}

  T call(T spot) const {
    T const tau = maturity_ - time_;
    T const s_tau = std::sqrt(tau);
    T const d_1 =
        (std::log(spot / strike_) + (rate_ + 0.5 * vol_ * vol_) * tau) /
        (vol_ * s_tau);
    T const d_2 = d_1 - vol_ * s_tau;
    T const result = norm_cdf(d_1) * spot -
                     (norm_cdf(d_2) * strike_ * std::exp(-rate_ * tau));
    return result;
  }

  T put(T spot) const {
    T const call_p = call(spot);
    T const tau = maturity_ - time_;
    return (strike_ * std::exp(-rate_ * tau) - spot + call_p);
  }
};

// ==========================================================================
// ==================================== range ===============================
// ==========================================================================
template <typename T>
class range {
 private:
  T l_, u_;

 public:
  explicit range(T lower, T upper) : l_{lower}, u_{upper} {}
  explicit range() : range(T{}, T{}) {}

  ~range() {}

  range(range const& copy) : l_{copy.l_}, u_{copy.u_} {}
  range(range&& other) noexcept
      : l_{std::move(other.l_)}, u_{std::move(other.u_)} {}

  range& operator=(range const& copy) {
    if (this != &copy) {
      l_ = copy.l_;
      u_ = copy.u_;
    }
    return *this;
  }

  range& operator=(range&& other) noexcept {
    if (this != &other) {
      l_ = std::move(other.l_);
      u_ = std::move(other.u_);
    }
    return *this;
  }

  inline T lower() const { return l_; }
  inline T upper() const { return u_; }
  inline T spread() const { return (u_ - l_); }
  inline T mid_point() const { return 0.5 * (l_ + u_); }
};

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

}  // namespace lss_utility

template <typename T>
void lss_utility::flat_matrix<T>::sort(
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

#endif  ///_LSS_UTILITY
