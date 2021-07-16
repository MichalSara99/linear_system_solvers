#pragma once
#if !defined(_LSS_FLAT_MATRIX_HPP_)
#define _LSS_FLAT_MATRIX_HPP_

#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "common/lss_utility.hpp"
#include <typeinfo>

namespace lss_containers
{

// ==========================================================================
// ============================== flat_matrix ===============================
// ==========================================================================
using lss_enumerations::flat_matrix_sort_enum;

template <typename fp_type> struct flat_matrix
{
  private:
    std::vector<std::tuple<std::size_t, std::size_t, fp_type>> container_;
    std::vector<std::size_t> column_cnt_;
    std::vector<fp_type> diagonal_;
    std::size_t ncols_, nrows_;

    explicit flat_matrix()
    {
    }

  public:
    explicit flat_matrix(std::size_t nrows, std::size_t ncols) : nrows_{nrows}, ncols_{ncols}
    {
        column_cnt_.resize(nrows);
        diagonal_.resize(nrows);
    }

    virtual ~flat_matrix()
    {
    }

    flat_matrix(flat_matrix<fp_type> const &copy)
        : ncols_{copy.ncols_}, nrows_{copy.nrows_}, container_{copy.container_},
          column_cnt_{copy.column_cnt_}, diagonal_{copy.diagonal_}
    {
    }

    flat_matrix(flat_matrix<fp_type> &&other) noexcept
        : ncols_{std::move(other.ncols_)}, nrows_{std::move(other.nrows_)}, container_{std::move(other.container_)},
          column_cnt_{std::move(other.column_cnt_)}, diagonal_{std::move(other.diagonal_)}
    {
    }

    flat_matrix<fp_type> &operator=(flat_matrix<fp_type> const &copy)
    {
        if (this != &copy)
        {
            ncols_ = copy.ncols_;
            nrows_ = copy.nrows_;
            container_ = copy.container_;
            column_cnt_ = copy.column_cnt_;
            diagonal_ = copy.diagonal_;
        }
        return *this;
    }

    flat_matrix<fp_type> &operator=(flat_matrix<fp_type> &&other) noexcept
    {
        if (this != &other)
        {
            ncols_ = std::move(other.ncols_);
            nrows_ = std::move(other.nrows_);
            container_ = std::move(other.container_);
            column_cnt_ = std::move(other.column_cnt_);
            diagonal_ = std::move(other.diagonal_);
        }
        return *this;
    }

    inline std::size_t rows() const
    {
        return nrows_;
    }
    inline std::size_t columns() const
    {
        return ncols_;
    }
    inline std::size_t size() const
    {
        return container_.size();
    }
    inline void clear()
    {
        container_.clear();
    }
    inline fp_type diagonal_at_row(std::size_t row_idx) const
    {
        return diagonal_[row_idx];
    }
    inline std::size_t non_zero_column_size(std::size_t row_idx) const
    {
        return column_cnt_[row_idx];
    }

    inline void emplace_back(std::size_t row_idx, std::size_t col_idx, fp_type value)
    {
        LSS_ASSERT(row_idx < nrows_, " rowIdx is outside <0," << nrows_ << ")");
        LSS_ASSERT(col_idx < ncols_, " colIdx is outside <0," << ncols_ << ")");
        container_.emplace_back(std::make_tuple(row_idx, col_idx, value));
        column_cnt_[row_idx]++;
        if (row_idx == col_idx)
            diagonal_[row_idx] = value;
    }

    inline void emplace_back(std::tuple<std::size_t, std::size_t, fp_type> tuple)
    {
        LSS_ASSERT(std::get<0>(tuple) < nrows_, " rowIdx is outside <0," << nrows_ << ")");
        LSS_ASSERT(std::get<1>(tuple) < ncols_, " colIdx is outside <0," << ncols_ << ")");
        container_.emplace_back(std::move(tuple));
        column_cnt_[std::get<0>(tuple)]++;
        if (std::get<1>(tuple) == std::get<0>(tuple))
            diagonal_[std::get<1>(tuple)] = std::get<2>(tuple);
    }

    void sort(flat_matrix_sort_enum sort);

    std::tuple<std::size_t, std::size_t, fp_type> const &at(std::size_t idx) const
    {
        return container_.at(idx);
    }
};

template <typename fp_type> void flat_matrix<fp_type>::sort(flat_matrix_sort_enum sort)
{
    if (sort == flat_matrix_sort_enum::RowMajor)
    {
        std::sort(container_.begin(), container_.end(),
                  [this](std::tuple<std::size_t, std::size_t, fp_type> const &lhs,
                         std::tuple<std::size_t, std::size_t, fp_type> const &rhs) {
                      std::size_t const flat_idx_lhs = std::get<1>(lhs) + nrows_ * std::get<0>(lhs);
                      std::size_t const flat_idx_rhs = std::get<1>(rhs) + nrows_ * std::get<0>(rhs);
                      return (flat_idx_lhs < flat_idx_rhs);
                  });
    }
    else
    {
        std::sort(container_.begin(), container_.end(),
                  [this](std::tuple<std::size_t, std::size_t, fp_type> const &lhs,
                         std::tuple<std::size_t, std::size_t, fp_type> const &rhs) {
                      std::size_t const flat_idx_lhs = std::get<0>(lhs) + ncols_ * std::get<1>(lhs);
                      std::size_t const flat_idx_rhs = std::get<0>(rhs) + ncols_ * std::get<1>(rhs);
                      return (flat_idx_lhs < flat_idx_rhs);
                  });
    }
}

} // namespace lss_containers

#endif ///_LSS_FLAT_MATRIX_HPP_
