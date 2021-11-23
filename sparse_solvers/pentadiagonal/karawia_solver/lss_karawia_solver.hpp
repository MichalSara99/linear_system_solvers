#pragma once
#if !defined(_LSS_KARAWIA_SOLVER_HPP_)
#define _LSS_KARAWIA_SOLVER_HPP_

#pragma warning(disable : 4244)

#include <tuple>
#include <type_traits>
#include <vector>

#include "boundaries/lss_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "common/lss_utility.hpp"
#include "lss_karawia_boundary.hpp"

namespace lss_karawia_solver
{

using lss_boundary::boundary_pair;
using lss_utility::range;
using lss_utility::sptr_t;

template <typename fp_type, template <typename, typename> typename container = std::vector,
          typename allocator = std::allocator<fp_type>>
class karawia_solver
{

  private:
    std::size_t discretization_size_;
    container<fp_type, allocator> a_, b_, c_, d_, e_;
    container<fp_type, allocator> alpha_, beta_, gamma_, mu_, f_;

    template <typename... fp_space_types>
    void kernel(boundary_pair<fp_type, fp_space_types...> const &boundary,
                boundary_pair<fp_type, fp_space_types...> const &other_boundary,
                container<fp_type, allocator> &solution, fp_type time, fp_space_types... space_args);

    explicit karawia_solver() = delete;
    // bool is_diagonally_dominant() const;

  public:
    typedef fp_type value_type;
    typedef container<fp_type, allocator> container_type;
    explicit karawia_solver(std::size_t discretization_size) : discretization_size_{discretization_size}
    {
    }

    ~karawia_solver()
    {
    }

    void set_diagonals(container<fp_type, allocator> lowest_diagonal, container<fp_type, allocator> lower_diagonal,
                       container<fp_type, allocator> diagonal, container<fp_type, allocator> upper_diagonal,
                       container<fp_type, allocator> uppest_diagonal);

    void set_rhs(container<fp_type, allocator> const &rhs);

    void solve(boundary_pair<fp_type> const &boundary, boundary_pair<fp_type> const &other_boundary,
               container<fp_type, allocator> &solution)
    {
        LSS_ASSERT(solution.size() == discretization_size_, "Incorrect size of solution container");
        kernel(boundary, other_boundary, solution, fp_type{});
    }

    void solve(boundary_pair<fp_type> const &boundary, boundary_pair<fp_type> const &other_boundary,
               container<fp_type, allocator> &solution, fp_type at_time)
    {
        LSS_ASSERT(solution.size() == discretization_size_, "Incorrect size of solution container");
        kernel(boundary, other_boundary, solution, at_time);
    }

    void solve(boundary_pair<fp_type, fp_type> const &boundary, boundary_pair<fp_type, fp_type> const &other_boundary,
               container<fp_type, allocator> &solution, fp_type at_time, fp_type space_arg)
    {
        LSS_ASSERT(solution.size() == discretization_size_, "Incorrect size of solution container");
        kernel<fp_type>(boundary, other_boundary, solution, at_time, space_arg);
    }
};

template <typename fp_type> using karawia_solver_ptr = sptr_t<karawia_solver<fp_type>>;

} // namespace lss_karawia_solver

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void lss_karawia_solver::karawia_solver<fp_type, container, allocator>::set_diagonals(
    container<fp_type, allocator> lowest_diagonal, container<fp_type, allocator> lower_diagonal,
    container<fp_type, allocator> diagonal, container<fp_type, allocator> upper_diagonal,
    container<fp_type, allocator> uppest_diagonal)
{
    LSS_ASSERT(lowest_diagonal.size() == discretization_size_, "Inncorect size for lowerDiagonal");
    LSS_ASSERT(lower_diagonal.size() == discretization_size_, "Inncorect size for lowerDiagonal");
    LSS_ASSERT(diagonal.size() == discretization_size_, "Inncorect size for diagonal");
    LSS_ASSERT(upper_diagonal.size() == discretization_size_, "Inncorect size for upperDiagonal");
    LSS_ASSERT(uppest_diagonal.size() == discretization_size_, "Inncorect size for upperDiagonal");
    a_ = std::move(lowest_diagonal);
    b_ = std::move(lower_diagonal);
    c_ = std::move(diagonal);
    d_ = std::move(upper_diagonal);
    e_ = std::move(uppest_diagonal);
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void lss_karawia_solver::karawia_solver<fp_type, container, allocator>::set_rhs(
    container<fp_type, allocator> const &rhs)
{
    LSS_ASSERT(rhs.size() == discretization_size_, "Inncorect size for right-hand side");
    f_ = rhs;
}

// template <typename fp_type, template <typename, typename> typename container, typename allocator>
// bool lss_karawia_solver::karawia_solver<fp_type, container, allocator>::is_diagonally_dominant() const
//{
//    // if (std::abs(b_[0]) < std::abs(c_[0]))
//    //    return false;
//    // if (std::abs(b_[system_size_ - 1]) < std::abs(a_[system_size_ - 1]))
//    //    return false;
//
//    // for (std::size_t t = 0; t < system_size_ - 1; ++t)
//    //    if (std::abs(b_[t]) < (std::abs(a_[t]) + std::abs(c_[t])))
//    //        return false;
//    return true;
//}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
template <typename... fp_space_types>
void lss_karawia_solver::karawia_solver<fp_type, container, allocator>::kernel(
    boundary_pair<fp_type, fp_space_types...> const &boundary,
    boundary_pair<fp_type, fp_space_types...> const &other_boundary, container<fp_type, allocator> &solution,
    fp_type time, fp_space_types... space_args)
{
    // check the diagonal dominance:
    // LSS_ASSERT(is_diagonally_dominant() == true, "Tridiagonal matrix must be diagonally dominant.");

    // clear the working containers:
    alpha_.clear();
    beta_.clear();
    gamma_.clear();
    mu_.clear();

    // resize the working containers:
    alpha_.resize(discretization_size_);
    beta_.resize(discretization_size_);
    gamma_.resize(discretization_size_);
    mu_.resize(discretization_size_);

    // get proper boundaries:
    const std::size_t N = discretization_size_ - 1;
    const fp_type one = static_cast<fp_type>(1.0);
    const auto &lowest_sixta = std::make_tuple(a_[2], b_[2], c_[2], d_[2], e_[2], f_[2]);
    const auto &lower_sixta = std::make_tuple(a_[3], b_[3], c_[3], d_[3], e_[3], f_[3]);
    const auto &higher_sixta = std::make_tuple(a_[N - 3], b_[N - 3], c_[N - 3], d_[N - 3], e_[N - 3], f_[N - 3]);
    const auto &highest_sixta = std::make_tuple(a_[N - 2], b_[N - 2], c_[N - 2], d_[N - 2], e_[N - 2], f_[N - 2]);
    const fp_type step = one / static_cast<fp_type>(N);
    karawia_solver_boundary<fp_type> solver_boundary(lowest_sixta, lower_sixta, higher_sixta, highest_sixta,
                                                     discretization_size_, step);
    const auto &init_coeffs = solver_boundary.init_coefficients(boundary, other_boundary, time, space_args...);
    const std::size_t start_idx = solver_boundary.start_index();
    const auto &fin_coeffs = solver_boundary.final_coefficients(boundary, other_boundary, time, space_args...);
    const std::size_t end_idx = solver_boundary.end_index();

    // init values for the working containers:
    mu_[start_idx] = c_[start_idx];
    alpha_[start_idx] = d_[start_idx] / mu_[start_idx];
    beta_[start_idx] = e_[start_idx] / mu_[start_idx];
    f_[start_idx] = std::get<0>(init_coeffs) / mu_[start_idx];
    //
    const std::size_t next_idx = start_idx + 1;
    gamma_[next_idx] = b_[next_idx];
    mu_[next_idx] = c_[next_idx] - alpha_[start_idx] * gamma_[next_idx];
    alpha_[next_idx] = (d_[next_idx] - beta_[start_idx] * gamma_[next_idx]) / mu_[next_idx];
    beta_[next_idx] = e_[next_idx] / mu_[next_idx];
    f_[next_idx] = (std::get<1>(init_coeffs) - f_[start_idx] * gamma_[next_idx]) / mu_[next_idx];

    for (std::size_t t = next_idx + 1; t <= end_idx - 2; ++t)
    {
        gamma_[t] = b_[t] - alpha_[t - 2] * a_[t];
        mu_[t] = c_[t] - beta_[t - 2] * a_[t] - alpha_[t - 1] * gamma_[t];
        alpha_[t] = (d_[t] - beta_[t - 1] * gamma_[t]) / mu_[t];
        beta_[t] = e_[t] / mu_[t];
        f_[t] = (f_[t] - f_[t - 2] * a_[t] - f_[t - 1] * gamma_[t]) / mu_[t];
    }

    gamma_[end_idx - 1] = b_[end_idx - 1] - alpha_[end_idx - 3] * a_[end_idx - 1];
    mu_[end_idx - 1] =
        c_[end_idx - 1] - beta_[end_idx - 3] * a_[end_idx - 1] - alpha_[end_idx - 2] * gamma_[end_idx - 1];
    alpha_[end_idx - 1] = (d_[end_idx - 1] - beta_[end_idx - 2] * gamma_[end_idx - 1]) / mu_[end_idx - 1];
    //
    gamma_[end_idx] = b_[end_idx - 1] - alpha_[end_idx - 2] * a_[end_idx];
    mu_[end_idx] = c_[end_idx] - beta_[end_idx - 2] * a_[end_idx] - alpha_[end_idx - 1] * gamma_[end_idx];
    f_[end_idx - 1] =
        (std::get<0>(fin_coeffs) - f_[end_idx - 2] * a_[end_idx - 1] - f_[end_idx - 2] * gamma_[end_idx - 1]) /
        mu_[end_idx - 1];
    f_[end_idx] =
        (std::get<1>(fin_coeffs) - f_[end_idx - 1] * a_[end_idx] - f_[end_idx - 1] * gamma_[end_idx]) / mu_[end_idx];

    solution[end_idx] = f_[end_idx];
    solution[end_idx - 1] = f_[end_idx - 1] - alpha_[end_idx - 1] * solution[end_idx];
    for (std::size_t t = end_idx - 1; t-- > start_idx /*&& t >= 0*/; /*t--*/)
    {
        solution[t] = f_[t] - alpha_[t] * solution[t + 1] - beta_[t] * solution[t + 2];
    }

    // fill in the boundary values:
    solution[0] = solver_boundary.lower_boundary(boundary, time, space_args...);
    solution[1] = solver_boundary.lower_boundary(other_boundary, time, space_args...);
    solution[N - 1] = solver_boundary.upper_boundary(other_boundary, time, space_args...);
    solution[N] = solver_boundary.upper_boundary(boundary, time, space_args...);
}

#endif ///_LSS_KARAWIA_SOLVER_HPP_
