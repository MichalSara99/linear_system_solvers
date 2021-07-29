#pragma once
#if !defined(_LSS_CUDA_SOLVER_HPP_)
#define _LSS_CUDA_SOLVER_HPP_

#include "boundaries/lss_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "common/lss_utility.hpp"
#include "containers/lss_flat_matrix.hpp"
#include "lss_cuda_boundary.hpp"
#include "sparse_solvers/general/core_cuda_solver/lss_core_cuda_solver.hpp"
#include "sparse_solvers/general/core_cuda_solver/lss_core_cuda_solver_policy.hpp"

namespace lss_cuda_solver
{

using lss_boundary::boundary_pair;
using lss_core_cuda_solver::flat_matrix;
using lss_core_cuda_solver::real_sparse_solver_cuda;
using lss_enumerations::factorization_enum;
using lss_enumerations::memory_space_enum;
using lss_utility::range;
using lss_utility::sptr_t;

template <memory_space_enum memory_space, typename fp_type,
          template <typename, typename> typename container = std::vector, typename allocator = std::allocator<fp_type>>
class cuda_solver
{
  private:
    factorization_enum factorization_;
    std::size_t discretization_size_;
    range<fp_type> space_range_;
    container<fp_type, allocator> a_, b_, c_, f_;

    template <typename... fp_space_types>
    void kernel(boundary_pair<fp_type, fp_space_types...> const &boundary, container<fp_type, allocator> &solution,
                factorization_enum factorization, fp_type time, fp_space_types... space_args);

    explicit cuda_solver() = delete;

  public:
    typedef fp_type value_type;
    explicit cuda_solver(range<fp_type> const &space_range, std::size_t discretization_size)
        : space_range_{space_range}, discretization_size_{discretization_size}, factorization_{
                                                                                    factorization_enum::QRMethod}
    {
    }

    ~cuda_solver()
    {
    }

    cuda_solver(cuda_solver const &) = delete;
    cuda_solver(cuda_solver &&) = delete;
    cuda_solver &operator=(cuda_solver const &) = delete;
    cuda_solver &operator=(cuda_solver &&) = delete;

    void set_diagonals(container<fp_type, allocator> lower_diagonal, container<fp_type, allocator> diagonal,
                       container<fp_type, allocator> upper_diagonal);

    void set_rhs(container<fp_type, allocator> const &rhs);

    void set_factorization(factorization_enum factorization = factorization_enum::QRMethod);

    void solve(boundary_pair<fp_type> const &boundary, container<fp_type, allocator> &solution)
    {
        LSS_ASSERT(solution.size() == discretization_size_, "Incorrect size of solution container");
        kernel(boundary, solution, factorization_, fp_type{});
    }

    void solve(boundary_pair<fp_type> const &boundary, container<fp_type, allocator> &solution, fp_type at_time)
    {
        LSS_ASSERT(solution.size() == discretization_size_, "Incorrect size of solution container");
        kernel(boundary, solution, factorization_, at_time);
    }

    void solve(boundary_pair<fp_type, fp_type> const &boundary, container<fp_type, allocator> &solution,
               fp_type at_time, fp_type space_arg)
    {
        LSS_ASSERT(solution.size() == discretization_size_, "Incorrect size of solution container");
        kernel<fp_type>(boundary, solution, factorization_, at_time, space_arg);
    }
};

template <memory_space_enum memory_space, typename fp_type, template <typename, typename> typename container,
          typename allocator>
void cuda_solver<memory_space, fp_type, container, allocator>::set_diagonals(
    container<fp_type, allocator> lower_diagonal, container<fp_type, allocator> diagonal,
    container<fp_type, allocator> upper_diagonal)
{
    LSS_ASSERT(lower_diagonal.size() == discretization_size_, "Inncorect size for lower_diagonal");
    LSS_ASSERT(diagonal.size() == discretization_size_, "Inncorect size for diagonal");
    LSS_ASSERT(upper_diagonal.size() == discretization_size_, "Inncorect size for upper_diagonal");
    a_ = std::move(lower_diagonal);
    b_ = std::move(diagonal);
    c_ = std::move(upper_diagonal);
}

template <memory_space_enum memory_space, typename fp_type, template <typename, typename> typename container,
          typename allocator>
void cuda_solver<memory_space, fp_type, container, allocator>::set_rhs(container<fp_type, allocator> const &rhs)
{
    LSS_ASSERT(rhs.size() == discretization_size_, "Inncorect size for right-hand side");
    f_ = rhs;
}

template <memory_space_enum memory_space, typename fp_type, template <typename, typename> typename container,
          typename allocator>
void cuda_solver<memory_space, fp_type, container, allocator>::set_factorization(factorization_enum factorization)
{
    factorization_ = factorization;
}

template <memory_space_enum memory_space, typename fp_type, template <typename, typename> typename container,
          typename allocator>
template <typename... fp_space_types>
void cuda_solver<memory_space, fp_type, container, allocator>::kernel(
    boundary_pair<fp_type, fp_space_types...> const &boundary, container<fp_type, allocator> &solution,
    factorization_enum factorization, fp_type time, fp_space_types... space_args)
{
    // get proper boundaries:
    const std::size_t N = discretization_size_ - 1;
    const auto &lowest_quad = std::make_tuple(a_[0], b_[0], c_[0], f_[0]);
    const auto &lower_quad = std::make_tuple(a_[1], b_[1], c_[1], f_[1]);
    const auto &higher_quad = std::make_tuple(a_[N - 1], b_[N - 1], c_[N - 1], f_[N - 1]);
    const auto &highest_quad = std::make_tuple(a_[N], b_[N], c_[N], f_[N]);
    const fp_type step = space_range_.spread() / static_cast<fp_type>(N);
    cuda_boundary<fp_type> solver_boundary(lowest_quad, lower_quad, higher_quad, highest_quad, discretization_size_,
                                           step);
    const auto &init_coeffs = solver_boundary.init_coefficients(boundary, time, space_args...);
    const std::size_t start_idx = solver_boundary.start_index();
    const auto &fin_coeffs = solver_boundary.final_coefficients(boundary, time, space_args...);
    const std::size_t end_idx = solver_boundary.end_index();

    const std::size_t system_size = end_idx - start_idx + 1;
    flat_matrix<fp_type> mat(system_size, system_size);
    container<fp_type, allocator> rhs(system_size, fp_type{});

    rhs[0] = std::get<2>(init_coeffs);
    mat.emplace_back(0, 0, std::get<0>(init_coeffs));
    mat.emplace_back(0, 1, std::get<1>(init_coeffs));
    for (std::size_t t = 1; t < system_size - 1; ++t)
    {
        mat.emplace_back(t, t - 1, a_[t + start_idx]);
        mat.emplace_back(t, t, b_[t + start_idx]);
        mat.emplace_back(t, t + 1, c_[t + start_idx]);
        rhs[t] = f_[t + start_idx];
    }
    mat.emplace_back(system_size - 1, system_size - 2, std::get<0>(fin_coeffs));
    mat.emplace_back(system_size - 1, system_size - 1, std::get<1>(fin_coeffs));
    rhs[system_size - 1] = std::get<2>(fin_coeffs);

    // initialise the solver:
    real_sparse_solver_cuda<memory_space, fp_type, container, allocator> rss(system_size);
    rss.initialize(system_size);
    rss.set_flat_sparse_matrix(std::move(mat));
    rss.set_rhs(rhs);
    container<fp_type, allocator> sub_solution(system_size, fp_type{});
    rss.solve(sub_solution, factorization);

    std::copy(sub_solution.begin(), sub_solution.end(), std::next(solution.begin(), start_idx));
    // fill in the boundary values:
    if (start_idx == 1)
        solution[0] = solver_boundary.lower_boundary(boundary, time, space_args...);
    if (end_idx == N - 1)
        solution[N] = solver_boundary.upper_boundary(boundary, time, space_args...);
}

template <typename fp_type> using device_cuda_solver_ptr = sptr_t<cuda_solver<memory_space_enum::Device, fp_type>>;
template <typename fp_type> using host_cuda_solver_ptr = sptr_t<cuda_solver<memory_space_enum::Host, fp_type>>;

} // namespace lss_cuda_solver

#endif ///_LSS_CUDA_SOLVER_HPP_
