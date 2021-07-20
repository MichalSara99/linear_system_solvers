#pragma once
#if !defined(_LSS_CONTAINER_2D_HPP_)
#define _LSS_CONTAINER_2D_HPP_

#include "common/lss_macros.hpp"
#include "common/lss_utility.hpp"
#include <typeinfo>
#include <vector>

namespace lss_containers
{

using lss_utility::sptr_t;

template <typename fp_type, template <typename, typename> typename container, typename allocator> class container_2d
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

    // place row_container at row position (row_idx)
    void operator()(std::size_t row_idx, container<fp_type, allocator> const &row_container)
    {
        LSS_ASSERT(row_idx < rows_, "Outside of row range");
        LSS_ASSERT(row_container.size() == columns_, "Outside of column range");
        data_[row_idx] = std::move(row_container);
    }

    // place ro_container at row position starting at col_start_idx and ending at
    // col_end_idx
    void operator()(std::size_t row_idx, container<fp_type, allocator> const &row_container, std::size_t col_start_idx,
                    std::size_t col_end_idx)
    {
        LSS_ASSERT(col_start_idx <= col_end_idx, "col_start_idx must be smaller or equal then col_end_idx");
        LSS_ASSERT(row_idx < rows_, "Outside of row range");
        LSS_ASSERT(col_start_idx < columns_, "col_start_idx is outside of column range");
        LSS_ASSERT(col_end_idx < columns_, "col_end_idx is outside of column range");
        const std::size_t len = col_end_idx - col_start_idx + 1;
        LSS_ASSERT(len <= row_container.size(), "Inserted length is bigger then the row_container");
        container<fp_type, allocator> cont = this->at(row_idx);
        std::size_t c = 0;
        for (std::size_t t = col_start_idx; t <= col_end_idx; ++t, ++c)
        {
            cont[t] = row_container[c];
        }
        this->operator()(row_idx, std::move(cont));
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
    template <typename in_itr> void set_data(in_itr first, in_itr last)
    {
        const std::size_t dist = std::distance(first, last);
        LSS_ASSERT((rows_ * columns_) == dist, "Source data is either too long or too short");
        for (std::size_t r = 0; r < rows_; ++r)
        {
            for (std::size_t c = 0; c < columns_; ++c)
            {
                this->operator()(r, c, *first);
                ++first;
            }
        }
    }
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void copy(container_2d<fp_type, container, allocator> &dest, container_2d<fp_type, container, allocator> const &src)
{
    LSS_ASSERT(dest.columns() == src.columns(), "dest and src must have same dimensions");
    LSS_ASSERT(dest.rows() == src.rows(), "dest and src must have same dimensions");
    for (std::size_t r = 0; r < dest.rows(); ++r)
    {
        dest(r, src.at(r));
    }
}

// concrete single-precision floating-point matrix
using matrix_float = container_2d<float, std::vector, std::allocator<float>>;
// concrete double-precision floating-point matrix
using matrix_double = container_2d<double, std::vector, std::allocator<double>>;

template <typename fp_type>
using matrix_default_ptr = sptr_t<container_2d<fp_type, std::vector, std::allocator<fp_type>>>;

} // namespace lss_containers

#endif ///_LSS_CONTAINER_2D_HPP_
