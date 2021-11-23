#pragma once
#if !defined(_LSS_THOMAS_LU_SOLVER_HPP_)
#define _LSS_THOMAS_LU_SOLVER_HPP_

#pragma warning(disable : 4244)

#include <tuple>
#include <type_traits>
#include <vector>

#include "boundaries/lss_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "common/lss_utility.hpp"
#include "lss_thomas_lu_boundary.hpp"

namespace lss_thomas_lu_solver
{

using lss_boundary::boundary_pair;
using lss_utility::range;
using lss_utility::sptr_t;

template <typename fp_type, template <typename, typename> typename container = std::vector,
          typename allocator = std::allocator<fp_type>>
class thomas_lu_solver
{

  private:
    std::size_t discretization_size_;
    container<fp_type, allocator> a_, b_, c_, f_;
    container<fp_type, allocator> beta_, gamma_;

    template <typename... fp_space_types>
    void kernel(boundary_pair<fp_type, fp_space_types...> const &boundary, container<fp_type, allocator> &solution,
                fp_type time, fp_space_types... space_args);

    explicit thomas_lu_solver() = delete;
    bool is_diagonally_dominant() const;

  public:
    typedef fp_type value_type;
    typedef container<fp_type, allocator> container_type;
    explicit thomas_lu_solver(std::size_t discretization_size) : discretization_size_{discretization_size}
    {
    }

    ~thomas_lu_solver()
    {
    }

    void set_diagonals(container<fp_type, allocator> lower_diagonal, container<fp_type, allocator> diagonal,
                       container<fp_type, allocator> upper_diagonal);

    void set_rhs(container<fp_type, allocator> const &rhs);

    void solve(boundary_pair<fp_type> const &boundary, container<fp_type, allocator> &solution)
    {
        LSS_ASSERT(solution.size() == discretization_size_, "Incorrect size of solution container");
        kernel(boundary, solution, fp_type{});
    }

    void solve(boundary_pair<fp_type> const &boundary, container<fp_type, allocator> &solution, fp_type at_time)
    {
        LSS_ASSERT(solution.size() == discretization_size_, "Incorrect size of solution container");
        kernel(boundary, solution, at_time);
    }

    void solve(boundary_pair<fp_type, fp_type> const &boundary, container<fp_type, allocator> &solution,
               fp_type at_time, fp_type space_arg)
    {
        LSS_ASSERT(solution.size() == discretization_size_, "Incorrect size of solution container");
        kernel<fp_type>(boundary, solution, at_time, space_arg);
    }
};

template <typename fp_type> using thomas_lu_solver_ptr = sptr_t<thomas_lu_solver<fp_type>>;

} // namespace lss_thomas_lu_solver

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void lss_thomas_lu_solver::thomas_lu_solver<fp_type, container, allocator>::set_diagonals(
    container<fp_type, allocator> lower_diagonal, container<fp_type, allocator> diagonal,
    container<fp_type, allocator> upper_diagonal)
{
    LSS_ASSERT(lower_diagonal.size() == discretization_size_, "Inncorect size for lowerDiagonal");
    LSS_ASSERT(diagonal.size() == discretization_size_, "Inncorect size for diagonal");
    LSS_ASSERT(upper_diagonal.size() == discretization_size_, "Inncorect size for upperDiagonal");
    a_ = std::move(lower_diagonal);
    b_ = std::move(diagonal);
    c_ = std::move(upper_diagonal);
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void lss_thomas_lu_solver::thomas_lu_solver<fp_type, container, allocator>::set_rhs(
    container<fp_type, allocator> const &rhs)
{
    LSS_ASSERT(rhs.size() == discretization_size_, "Inncorect size for right-hand side");
    f_ = rhs;
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
bool lss_thomas_lu_solver::thomas_lu_solver<fp_type, container, allocator>::is_diagonally_dominant() const
{
    // if (std::abs(b_[0]) < std::abs(c_[0]))
    //    return false;
    // if (std::abs(b_[system_size_ - 1]) < std::abs(a_[system_size_ - 1]))
    //    return false;

    // for (std::size_t t = 0; t < system_size_ - 1; ++t)
    //    if (std::abs(b_[t]) < (std::abs(a_[t]) + std::abs(c_[t])))
    //        return false;
    return true;
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
template <typename... fp_space_types>
void lss_thomas_lu_solver::thomas_lu_solver<fp_type, container, allocator>::kernel(
    boundary_pair<fp_type, fp_space_types...> const &boundary, container<fp_type, allocator> &solution, fp_type time,
    fp_space_types... space_args)
{
    // check the diagonal dominance:
    LSS_ASSERT(is_diagonally_dominant() == true, "Tridiagonal matrix must be diagonally dominant.");

    // clear the working containers:
    beta_.clear();
    gamma_.clear();

    // resize the working containers:
    beta_.resize(discretization_size_);
    gamma_.resize(discretization_size_);

    // get proper boundaries:
    const std::size_t N = discretization_size_ - 1;
    const fp_type one = static_cast<fp_type>(1.0);
    const auto &lowest_quad = std::make_tuple(a_[0], b_[0], c_[0], f_[0]);
    const auto &lower_quad = std::make_tuple(a_[1], b_[1], c_[1], f_[1]);
    const auto &higher_quad = std::make_tuple(a_[N - 1], b_[N - 1], c_[N - 1], f_[N - 1]);
    const auto &highest_quad = std::make_tuple(a_[N], b_[N], c_[N], f_[N]);
    const fp_type step = one / static_cast<fp_type>(N);
    thomas_lu_solver_boundary<fp_type> solver_boundary(lowest_quad, lower_quad, higher_quad, highest_quad,
                                                       discretization_size_, step);
    const auto &init_coeffs = solver_boundary.init_coefficients(boundary, time, space_args...);
    const std::size_t start_idx = solver_boundary.start_index();
    const auto &fin_coeffs = solver_boundary.final_coefficients(boundary, time, space_args...);
    const std::size_t end_idx = solver_boundary.end_index();
    const fp_type a = std::get<0>(fin_coeffs);
    const fp_type b = std::get<1>(fin_coeffs);
    const fp_type r = std::get<2>(fin_coeffs);

    // init values for the working containers:
    beta_[start_idx] = std::get<0>(init_coeffs);
    gamma_[start_idx] = std::get<1>(init_coeffs);

    for (std::size_t t = start_idx + 1; t < end_idx; ++t)
    {
        beta_[t] = b_[t] - (a_[t] * gamma_[t - 1]);
        gamma_[t] = c_[t] / beta_[t];
    }
    beta_[end_idx] = b - (a * gamma_[end_idx - 1]);

    solution[start_idx] = std::get<3>(init_coeffs);
    for (std::size_t t = start_idx + 1; t < end_idx; ++t)
    {
        solution[t] = (f_[t] - (a_[t] * solution[t - 1])) / beta_[t];
    }
    solution[end_idx] = (r - (a * solution[end_idx - 1])) / beta_[end_idx];

    f_[end_idx] = solution[end_idx];
    for (long long t = end_idx - 1; t >= start_idx && t >= 0; t--)
    {
        f_[t] = solution[t] - (gamma_[t] * f_[t + 1]);
    }
    // first copy to solution container:
    std::copy(f_.begin(), f_.end(), solution.begin());
    // fill in the boundary values:
    if (start_idx == 1)
        solution[0] = solver_boundary.lower_boundary(boundary, time, space_args...);
    if (end_idx == N - 1)
        solution[N] = solver_boundary.upper_boundary(boundary, time, space_args...);
}

#endif ///_LSS_THOMAS_LU_SOLVER_HPP_
