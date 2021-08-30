#pragma once
#if !defined(_LSS_CONTAINER_2D_HPP_)
#define _LSS_CONTAINER_2D_HPP_

#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "common/lss_utility.hpp"
#include <typeinfo>
#include <vector>

namespace lss_containers
{

using lss_enumerations::by_enum;
using lss_utility::sptr_t;

template <by_enum by, typename fp_type, template <typename, typename> typename container, typename allocator>
class container_2d
{
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class container_2d<by_enum::Row, fp_type, container, allocator>
{
  private:
    std::size_t rows_;
    std::size_t columns_;
    std::vector<container<fp_type, allocator>> data_;

    explicit container_2d()
    {
    }

  public:
    typedef fp_type value_type;
    typedef container<fp_type, allocator> element_type;

    explicit container_2d(std::size_t rows, std::size_t columns) : rows_{rows}, columns_{columns}
    {
        for (std::size_t r = 0; r < rows; ++r)
        {
            data_.emplace_back(container<fp_type, allocator>(static_cast<std::size_t>(columns)));
        }
    }

    explicit container_2d(std::size_t rows, std::size_t columns, fp_type value) : rows_{rows}, columns_{columns}
    {

        for (std::size_t r = 0; r < rows; ++r)
        {
            data_.emplace_back(container<fp_type, allocator>(static_cast<std::size_t>(columns), value));
        }
    }

    ~container_2d()
    {
    }

    container_2d(container_2d const &copy) : rows_{copy.rows_}, columns_{copy.columns_}, data_{copy.data_}
    {
    }

    container_2d(container_2d &&other) noexcept
        : rows_{std::move(other.rows_)}, columns_{std::move(other.columns_)}, data_{std::move(other.data_)}
    {
    }

    container_2d(container_2d<by_enum::Column, fp_type, container, allocator> const &copy)
        : rows_{copy.rows()}, columns_{copy.columns()}
    {

        for (std::size_t r = 0; r < rows_; ++r)
        {
            data_.emplace_back(container<fp_type, allocator>(static_cast<std::size_t>(columns_)));
            for (std::size_t c = 0; c < columns_; ++c)
            {
                data_[r][c] = copy(r, c);
            }
        }
    }

    container_2d &operator=(container_2d const &copy)
    {
        if (this != &copy)
        {
            rows_ = copy.rows_;
            columns_ = copy.columns_;
            data_ = copy.data_;
        }
        return *this;
    }

    container_2d &operator=(container_2d &&other) noexcept
    {
        if (this != &other)
        {
            rows_ = std::move(other.rows_);
            columns_ = std::move(other.columns_);
            data_ = std::move(other.data_);
        }
        return *this;
    }

    container_2d &operator=(container_2d<by_enum::Column, fp_type, container, allocator> const &copy)
    {
        if (this != &copy)
        {
            data_.clear();
            rows_ = copy.rows();
            columns_ = copy.columns();
            for (std::size_t r = 0; r < rows_; ++r)
            {
                data_.emplace_back(container<fp_type, allocator>(static_cast<std::size_t>(columns_)));
                for (std::size_t c = 0; c < columns_; ++c)
                {
                    data_[r][c] = copy(r, c);
                }
            }
        }
        return *this;
    }

    std::size_t rows() const
    {
        return rows_;
    }
    std::size_t columns() const
    {
        return columns_;
    }
    std::size_t total_size() const
    {
        return rows_ * columns_;
    }

    // return value from container_2d at potision (row_idx,col_idx)
    fp_type operator()(std::size_t row_idx, std::size_t col_idx) const
    {
        LSS_ASSERT(row_idx < rows_, "Outside of row range");
        LSS_ASSERT(col_idx < columns_, "Outside of column range");
        return data_[row_idx][col_idx];
    }

    // return row container from container_2d at potision (row_idx)
    container<fp_type, allocator> operator()(std::size_t row_idx) const
    {
        LSS_ASSERT(row_idx < rows_, "Outside of row range");
        return data_[row_idx];
    }

    // return value from container_2d at potision (row_idx,col_idx)
    fp_type at(std::size_t row_idx, std::size_t col_idx) const
    {
        return operator()(row_idx, col_idx);
    }

    // return row container from container_2d at potision (row_idx)
    container<fp_type, allocator> at(std::size_t row_idx) const
    {
        return operator()(row_idx);
    }

    // place row container at position (row_idx)
    void operator()(std::size_t row_idx, container<fp_type, allocator> const &cont)
    {
        LSS_ASSERT(row_idx < rows_, "Outside of row range");
        LSS_ASSERT(cont.size() == columns_, "Outside of column range");
        data_[row_idx] = std::move(cont);
    }

    // place value at position (row_idx,col_idx)
    void operator()(std::size_t row_idx, std::size_t col_idx, fp_type value)
    {
        LSS_ASSERT(row_idx < rows_, "Outside of row range");
        LSS_ASSERT(col_idx < columns_, "Outside of column range");
        data_[row_idx][col_idx] = value;
    }

    // returns data as flat vector row-wise
    std::vector<fp_type, allocator> const data() const
    {
        std::vector<fp_type, allocator> d(rows_ * columns_);
        auto itr = d.begin();
        for (std::size_t r = 0; r < rows_; ++r)
        {
            auto row = at(r);
            std::copy(row.begin(), row.end(), itr);
            std::advance(itr, columns_);
        }
        return d;
    }
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
class container_2d<by_enum::Column, fp_type, container, allocator>
{
  private:
    std::size_t rows_;
    std::size_t columns_;
    std::vector<container<fp_type, allocator>> data_;

    explicit container_2d()
    {
    }

  public:
    typedef fp_type value_type;
    typedef container<fp_type, allocator> element_type;

    explicit container_2d(std::size_t rows, std::size_t columns) : rows_{rows}, columns_{columns}
    {
        for (std::size_t c = 0; c < columns; ++c)
        {
            data_.emplace_back(container<fp_type, allocator>(static_cast<std::size_t>(rows)));
        }
    }

    explicit container_2d(std::size_t rows, std::size_t columns, fp_type value) : rows_{rows}, columns_{columns}
    {
        for (std::size_t c = 0; c < columns; ++c)
        {
            data_.emplace_back(container<fp_type, allocator>(static_cast<std::size_t>(rows), value));
        }
    }

    ~container_2d()
    {
    }

    container_2d(container_2d const &copy) : rows_{copy.rows_}, columns_{copy.columns_}, data_{copy.data_}
    {
    }

    container_2d(container_2d &&other) noexcept
        : rows_{std::move(other.rows_)}, columns_{std::move(other.columns_)}, data_{std::move(other.data_)}
    {
    }

    container_2d(container_2d<by_enum::Row, fp_type, container, allocator> const &copy)
        : rows_{copy.rows()}, columns_{copy.columns()}
    {

        for (std::size_t c = 0; c < columns_; ++c)
        {
            data_.emplace_back(container<fp_type, allocator>(static_cast<std::size_t>(rows_)));
            for (std::size_t r = 0; r < rows_; ++r)
            {
                data_[c][r] = copy(r, c);
            }
        }
    }

    container_2d &operator=(container_2d const &copy)
    {
        if (this != &copy)
        {
            rows_ = copy.rows_;
            columns_ = copy.columns_;
            data_ = copy.data_;
        }
        return *this;
    }

    container_2d &operator=(container_2d &&other) noexcept
    {
        if (this != &other)
        {
            rows_ = std::move(other.rows_);
            columns_ = std::move(other.columns_);
            data_ = std::move(other.data_);
        }
        return *this;
    }

    container_2d &operator=(container_2d<by_enum::Row, fp_type, container, allocator> const &copy)
    {
        if (this != &copy)
        {
            data_.clear();
            rows_ = copy.rows();
            columns_ = copy.columns();

            for (std::size_t c = 0; c < columns_; ++c)
            {
                data_.emplace_back(container<fp_type, allocator>(static_cast<std::size_t>(rows_)));
                for (std::size_t r = 0; r < rows_; ++r)
                {
                    data_[c][r] = copy(r, c);
                }
            }
        }
        return *this;
    }

    std::size_t rows() const
    {
        return rows_;
    }
    std::size_t columns() const
    {
        return columns_;
    }
    std::size_t total_size() const
    {
        return rows_ * columns_;
    }

    // return value from container_2d at potision (row_idx,col_idx)
    fp_type operator()(std::size_t row_idx, std::size_t col_idx) const
    {
        LSS_ASSERT(row_idx < rows_, "Outside of row range");
        LSS_ASSERT(col_idx < columns_, "Outside of column range");
        return data_[col_idx][row_idx];
    }

    // return column container from container_2d at potision (col_idx)
    container<fp_type, allocator> operator()(std::size_t col_idx) const
    {
        LSS_ASSERT(col_idx < columns_, "Outside of column range");
        return data_[col_idx];
    }

    // return value from container_2d at potision (row_idx,col_idx)
    fp_type at(std::size_t row_idx, std::size_t col_idx) const
    {
        return operator()(row_idx, col_idx);
    }

    // return column container from container_2d at potision (col_idx)
    container<fp_type, allocator> at(std::size_t col_idx) const
    {
        return operator()(col_idx);
    }

    // place column container at position (col_idx)
    void operator()(std::size_t col_idx, container<fp_type, allocator> const &cont)
    {
        LSS_ASSERT(col_idx < columns_, "Outside of column range");
        LSS_ASSERT(cont.size() == rows_, "Outside of row range");
        data_[col_idx] = std::move(cont);
    }

    // place value at position (row_idx,col_idx)
    void operator()(std::size_t row_idx, std::size_t col_idx, fp_type value)
    {
        LSS_ASSERT(row_idx < rows_, "Outside of row range");
        LSS_ASSERT(col_idx < columns_, "Outside of column range");
        data_[col_idx][row_idx] = value;
    }

    // returns data as flat vector column-wise
    std::vector<fp_type, allocator> const data() const
    {
        std::vector<fp_type, allocator> d(rows_ * columns_);
        auto itr = d.begin();
        for (std::size_t c = 0; c < columns_; ++c)
        {
            auto col = at(c);
            std::copy(col.begin(), col.end(), itr);
            std::advance(itr, rows_);
        }
        return d;
    }
};

template <by_enum by, typename fp_type, template <typename, typename> typename container, typename allocator>
void copy(container_2d<by, fp_type, container, allocator> &dest,
          container_2d<by, fp_type, container, allocator> const &src)
{
    LSS_ASSERT(dest.columns() == src.columns(), "dest and src must have same dimensions");
    LSS_ASSERT(dest.rows() == src.rows(), "dest and src must have same dimensions");
    std::size_t size;
    if (by == by_enum::Row)
        size = dest.rows();
    else
        size = dest.columns();

    for (std::size_t t = 0; t < size; ++t)
    {
        dest(t, src.at(t));
    }
}

// concrete single-precision floating-point row-major matrix
using rmatrix_float = container_2d<by_enum::Row, float, std::vector, std::allocator<float>>;
// concrete double-precision floating-point row-major matrix
using rmatrix_double = container_2d<by_enum::Row, double, std::vector, std::allocator<double>>;
// concrete single-precision floating-point column-major matrix
using cmatrix_float = container_2d<by_enum::Column, float, std::vector, std::allocator<float>>;
// concrete double-precision floating-point column-major matrix
using cmatrix_double = container_2d<by_enum::Column, double, std::vector, std::allocator<double>>;

template <typename fp_type>
using rmatrix_default_ptr = sptr_t<container_2d<by_enum::Row, fp_type, std::vector, std::allocator<fp_type>>>;
template <typename fp_type>
using cmatrix_default_ptr = sptr_t<container_2d<by_enum::Column, fp_type, std::vector, std::allocator<fp_type>>>;

} // namespace lss_containers

#endif ///_LSS_CONTAINER_2D_HPP_
