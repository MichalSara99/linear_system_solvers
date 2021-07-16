#pragma once
#if !defined(_LSS_CORE_SOR_SOLVER_HPP_)
#define _LSS_CORE_SOR_SOLVER_HPP_

#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "common/lss_utility.hpp"
#include "containers/lss_flat_matrix.hpp"
#include "sparse_solvers/general/sor_solver_traits/lss_sor_solver_traits.hpp"

namespace lss_core_sor_solver
{

using lss_containers::flat_matrix;
using lss_containers::flat_matrix_sort_enum;
using lss_sor_solver_traits::sor_solver_traits;
using lss_utility::NaN;
using lss_utility::sptr_t;
using lss_utility::uptr_t;

template <typename fp_type, template <typename fp_type, typename allocator> typename container = std::vector,
          typename allocator = std::allocator<fp_type>>
class core_sor_solver
{
  private:
    container<fp_type, allocator> b_;
    uptr_t<flat_matrix<fp_type>> matrix_data_ptr_;
    std::size_t system_size_;
    fp_type omega_;

    explicit core_sor_solver(){};

    template <template <typename> typename traits = sor_solver_traits>
    void kernel(container<fp_type, allocator> &solution);
    bool is_diagonally_dominant();

  public:
    typedef fp_type value_type;
    explicit core_sor_solver(std::size_t system_size) : system_size_{system_size}
    {
    }

    virtual ~core_sor_solver()
    {
    }

    core_sor_solver(core_sor_solver const &) = delete;
    core_sor_solver(core_sor_solver &&) = delete;
    core_sor_solver &operator=(core_sor_solver const &) = delete;
    core_sor_solver &operator=(core_sor_solver &&) = delete;

    void set_flat_sparse_matrix(flat_matrix<fp_type> flat_matrix);

    void set_rhs(container<fp_type, allocator> const &rhs);

    void set_omega(fp_type value);

    template <template <typename> typename traits = sor_solver_traits>
    void solve(container<fp_type, allocator> &solution);

    template <template <typename> typename traits = sor_solver_traits> container<fp_type, allocator> const solve();
};

template <typename fp_type> using core_sor_solver_ptr = sptr_t<core_sor_solver<fp_type>>;

} // namespace lss_core_sor_solver

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void lss_core_sor_solver::core_sor_solver<fp_type, container, allocator>::set_omega(fp_type value)
{
    LSS_ASSERT((value > static_cast<fp_type>(0.0)) && (value < static_cast<fp_type>(2.0)),
               "relaxation parameter must be inside (0,2) range");
    omega_ = value;
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void lss_core_sor_solver::core_sor_solver<fp_type, container, allocator>::set_rhs(
    container<fp_type, allocator> const &rhs)
{
    LSS_ASSERT(rhs.size() == system_size_, "Inncorect size for right-hand side");
    b_ = rhs;
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void lss_core_sor_solver::core_sor_solver<fp_type, container, allocator>::set_flat_sparse_matrix(
    flat_matrix<fp_type> matrix)
{
    LSS_ASSERT(matrix.rows() == system_size_, "Inncorect number of rows for the flat_raw_matrix");
    LSS_ASSERT(matrix.columns() == system_size_, "Inncorect number of columns for the flat_raw_matrix");
    matrix_data_ptr_ = std::make_unique<flat_matrix<fp_type>>(std::move(matrix));
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
bool lss_core_sor_solver::core_sor_solver<fp_type, container, allocator>::is_diagonally_dominant()
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
template <template <typename> typename traits>
void lss_core_sor_solver::core_sor_solver<fp_type, container, allocator>::kernel(
    container<fp_type, allocator> &solution)
{
    LSS_ASSERT(is_diagonally_dominant() == true, "flat_raw_matrix isd not diagonally dominant.");
    // set initial step:
    std::size_t step{0};
    // set iter_limit:
    const std::size_t iter_limit = traits<fp_type>::iteration_limit();
    // set tolerance:
    const fp_type tol = traits<fp_type>::tolerance();
    // for error:
    fp_type error{};
    // for sigma_1,sigma_2:
    fp_type sigma_1{};
    fp_type sigma_2{};
    // for diagonal value:
    fp_type diag{};
    // for new solution:
    container<fp_type, allocator> x_new(solution);
    // for number of columns:
    std::size_t cols{};
    // for flat_matrix element:
    std::tuple<std::size_t, std::size_t, fp_type> mat_elm{};
    // for flat_matrix row and column index of the element:
    std::size_t mat_r{};
    std::size_t mat_c{};
    // for flat_matrix element value:
    fp_type mat_val{};
    // for index of the flat_matrix element:
    std::size_t flt_idx{0};
    const fp_type one = static_cast<fp_type>(1.0);

    while (iter_limit > step)
    {
        error = static_cast<fp_type>(0.0);
        flt_idx = 0;
        for (std::size_t r = 0; r < matrix_data_ptr_->rows(); ++r, flt_idx += cols)
        {
            sigma_1 = sigma_2 = static_cast<fp_type>(0.0);
            diag = matrix_data_ptr_->diagonal_at_row(r);
            cols = matrix_data_ptr_->non_zero_column_size(r);
            for (std::size_t c = flt_idx; c < cols + flt_idx; ++c)
            {
                mat_elm = matrix_data_ptr_->at(c);
                mat_r = std::get<0>(mat_elm);
                mat_c = std::get<1>(mat_elm);
                mat_val = std::get<2>(mat_elm);
                if (mat_c < mat_r)
                    sigma_1 += mat_val * x_new[mat_c];
                if (mat_c > mat_r)
                    sigma_2 += mat_val * solution[mat_c];
            }
            x_new[r] = (one - omega_) * solution[r] + ((omega_ / diag) * (b_[r] - sigma_1 - sigma_2));
            error += (x_new[r] - solution[r]) * (x_new[r] - solution[r]);
        }

        if (error <= tol)
            break;
        solution = x_new;
        step++;
    }
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
template <template <typename> typename traits>
void lss_core_sor_solver::core_sor_solver<fp_type, container, allocator>::solve(container<fp_type, allocator> &solution)
{
    LSS_ASSERT(solution.size() == system_size_, "Incorrect size of solution container");
    kernel(solution);
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
template <template <typename> typename traits>
container<fp_type, allocator> const lss_core_sor_solver::core_sor_solver<fp_type, container, allocator>::solve()
{
    container<fp_type, allocator> solution(system_size_);
    kernel(solution);
    return solution;
}

#endif ///_LSS_CORE_SOR_SOLVER_HPP_
