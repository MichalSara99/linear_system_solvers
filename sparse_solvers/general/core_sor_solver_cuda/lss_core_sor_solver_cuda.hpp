#pragma once
#if !defined(_LSS_CORE_SOR_SOLVER_CUDA_HPP_)
#define _LSS_CORE_SOR_SOLVER_CUDA_HPP_

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <numeric>

#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "common/lss_utility.hpp"
#include "containers/lss_flat_matrix.hpp"
#include "sparse_solvers/general/sor_solver_traits/lss_sor_solver_traits.hpp"

namespace lss_core_sor_solver
{

using lss_containers::flat_matrix;
using lss_containers::flat_matrix_sort_enum;
using lss_sor_solver_traits::sor_solver_cuda_traits;
using lss_utility::NaN;
using lss_utility::sptr_t;
using lss_utility::uptr_t;

template <typename fp_type> class sor_solver_core
{
  private:
    thrust::device_vector<fp_type> matrix_vals_;
    thrust::device_vector<std::size_t> non_zero_column_idx_;
    thrust::device_vector<fp_type> rhs_vals_;
    thrust::device_vector<fp_type> diagonal_vals_;
    thrust::device_vector<std::size_t> row_start_idx_;
    thrust::device_vector<std::size_t> row_end_idx_;
    std::size_t nrows_;

    explicit sor_solver_core()
    {
    }

  public:
    explicit sor_solver_core(thrust::device_vector<fp_type> const &matrix_values,
                             thrust::device_vector<std::size_t> const &non_zero_column_idx, std::size_t const &nrows,
                             thrust::device_vector<fp_type> const &rhs_values,
                             thrust::device_vector<fp_type> const &diagonal_values,
                             thrust::device_vector<std::size_t> const &row_start_idx,
                             thrust::device_vector<std::size_t> const &row_end_idx)
        : matrix_vals_{matrix_values}, non_zero_column_idx_{non_zero_column_idx}, nrows_{nrows}, rhs_vals_{rhs_values},
          diagonal_vals_{diagonal_values}, row_start_idx_{row_start_idx}, row_end_idx_{row_end_idx}
    {
    }

    ~sor_solver_core()
    {
    }

    sor_solver_core(sor_solver_core const &) = delete;
    sor_solver_core(sor_solver_core &&) = delete;
    sor_solver_core &operator=(sor_solver_core const &) = delete;
    sor_solver_core &operator=(sor_solver_core &&) = delete;

    void operator()(thrust::device_vector<fp_type> &solution, thrust::device_vector<fp_type> &next_solution,
                    thrust::device_vector<fp_type> &errors, fp_type omega);
};

template <typename fp_type, template <typename, typename> typename container = std::vector,
          typename allocator = std::allocator<fp_type>>
class core_sor_solver_cuda
{
  private:
    container<fp_type, allocator> b_;
    uptr_t<flat_matrix<fp_type>> matrix_data_ptr_;
    std::size_t system_size_;
    fp_type omega_;

    explicit core_sor_solver_cuda(){};

    core_sor_solver_cuda(core_sor_solver_cuda const &) = delete;
    core_sor_solver_cuda(core_sor_solver_cuda &&) = delete;
    core_sor_solver_cuda &operator=(core_sor_solver_cuda const &) = delete;
    core_sor_solver_cuda &operator=(core_sor_solver_cuda &&) = delete;

    template <template <typename> typename traits = sor_solver_cuda_traits>
    void kernel(container<fp_type, allocator> &solution);
    bool is_diagonally_dominant();

  public:
    typedef fp_type value_type;
    explicit core_sor_solver_cuda(std::size_t system_size) : system_size_{system_size}
    {
    }

    virtual ~core_sor_solver_cuda()
    {
    }

    void set_flat_sparse_matrix(flat_matrix<fp_type> flat_matrix);

    void set_rhs(container<fp_type, allocator> const &rhs);

    void set_omega(fp_type value);

    template <template <typename> typename traits = sor_solver_cuda_traits>
    void solve(container<fp_type, allocator> &solution);

    template <template <typename> typename traits = sor_solver_cuda_traits> container<fp_type, allocator> const solve();
};

template <typename fp_type> using core_sor_solver_cuda_ptr = sptr_t<core_sor_solver_cuda<fp_type>>;

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void core_sor_solver_cuda<fp_type, container, allocator>::set_omega(fp_type value)
{
    LSS_ASSERT((value > static_cast<fp_type>(0.0)) && (value < static_cast<fp_type>(2.0)),
               "relaxation parameter must be inside (0,2) range");
    omega_ = value;
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void core_sor_solver_cuda<fp_type, container, allocator>::set_rhs(container<fp_type, allocator> const &rhs)
{
    LSS_ASSERT(rhs.size() == system_size_, "Inncorect size for right-hand side");
    b_ = rhs;
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void core_sor_solver_cuda<fp_type, container, allocator>::set_flat_sparse_matrix(flat_matrix<fp_type> matrix)
{
    LSS_ASSERT(matrix.rows() == system_size_, "Inncorect number of rows for the flat_raw_matrix");
    LSS_ASSERT(matrix.columns() == system_size_, "Inncorect number of columns for the flat_raw_matrix");
    matrix_data_ptr_ = std::make_unique<flat_matrix<fp_type>>(std::move(matrix));
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
bool core_sor_solver_cuda<fp_type, container, allocator>::is_diagonally_dominant()
{
    LSS_VERIFY(matrix_data_ptr_, "flat_matrix has not been provided.");
    LSS_ASSERT(b_.size() == system_size_, "Incorrect size for right-hand side");
    // first make sure the matrix is row-major sorted
    matrix_data_ptr_->sort(flat_matrix_sort_enum::RowMajor);
    fp_type diag{};
    std::tuple<std::size_t, std::size_t, fp_type> non_diag{};
    fp_type sum{};
    std::size_t cols{};
    // for index of the flat_matrix element:
    std::size_t flt_idx{0};
    for (std::size_t r = 0; r < matrix_data_ptr_->rows(); ++r, flt_idx += cols)
    {
        sum = static_cast<fp_type>(0.0);
        diag = std::abs(matrix_data_ptr_->diagonal_at_row(r));
        cols = matrix_data_ptr_->non_zero_column_size(r);
        for (std::size_t c = flt_idx; c < cols + flt_idx; ++c)
        {
            non_diag = matrix_data_ptr_->at(c);
            if (std::get<0>(non_diag) != std::get<1>(non_diag))
                sum += std::abs(std::get<2>(non_diag));
        }
        if (diag < sum)
            return false;
    }
    return true;
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
template <template <typename fp_type> typename traits>
void core_sor_solver_cuda<fp_type, container, allocator>::kernel(container<fp_type, allocator> &solution)
{
    LSS_ASSERT(is_diagonally_dominant() == true, "flat_raw_matrix isd not diagonally dominant.");
    // set initial step:
    std::size_t step{0};
    // set iter_limit:
    std::size_t iter_limit = traits<fp_type>::iteration_limit();
    // set tolerance:
    fp_type const tol = traits<fp_type>::tolerance();
    // for error:
    fp_type error{};
    // make copies to pass to sor_solver_core:
    std::size_t total_size{matrix_data_ptr_->size()};
    thrust::device_vector<fp_type> rhs_values = b_;
    thrust::device_vector<fp_type> matrix_values(total_size);
    thrust::device_vector<std::size_t> non_zero_column_idx(total_size);
    std::tuple<std::size_t, std::size_t, fp_type> triplet{};
    for (std::size_t t = 0; t < total_size; ++t)
    {
        triplet = matrix_data_ptr_->at(t);
        non_zero_column_idx[t] = std::get<1>(triplet);
        matrix_values[t] = std::get<2>(triplet);
    }
    std::size_t row_size{matrix_data_ptr_->rows()};
    thrust::device_vector<fp_type> diagonal_values(row_size);
    thrust::device_vector<std::size_t> row_start_idx(row_size);
    thrust::device_vector<std::size_t> row_end_idx(row_size);
    std::size_t end_cnt{0};
    std::size_t non_zero{0};
    for (std::size_t r = 0; r < row_size; ++r)
    {
        diagonal_values[r] = matrix_data_ptr_->diagonal_at_row(r);
        row_start_idx[r] = end_cnt;
        non_zero = matrix_data_ptr_->non_zero_column_size(r);
        end_cnt += non_zero;
        row_end_idx[r] = end_cnt - 1;
    }
    // initialize sor_solver_core object:
    sor_solver_core<fp_type> core_solver(matrix_values, non_zero_column_idx, row_size, rhs_values, diagonal_values,
                                         row_start_idx, row_end_idx);

    thrust::device_vector<fp_type> sol(solution.begin(), solution.end());
    thrust::device_vector<fp_type> new_sol(solution.begin(), solution.end());
    thrust::device_vector<fp_type> errors(solution.begin(), solution.end());

    while (iter_limit > step)
    {
        core_solver(sol, new_sol, errors, omega_);
        error =
            thrust::reduce(errors.begin(), errors.end(), decltype(errors)::value_type(0.0), thrust::plus<fp_type>());
        if (error <= tol)
            break;
        sol = new_sol;
        step++;
    }
    thrust::copy(sol.begin(), sol.end(), solution.begin());
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
template <template <typename> typename traits>
void core_sor_solver_cuda<fp_type, container, allocator>::solve(container<fp_type, allocator> &solution)
{
    LSS_ASSERT(solution.size() == system_size_, "Incorrect size of solution container");
    kernel(solution);
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
template <template <typename> typename traits>
container<fp_type, allocator> const core_sor_solver_cuda<fp_type, container, allocator>::solve()
{
    container<fp_type, allocator> solution(system_size_);
    kernel(solution);
    return solution;
}

} // namespace lss_core_sor_solver

#endif ///_LSS_CORE_SOR_SOLVER_CUDA_HPP_
