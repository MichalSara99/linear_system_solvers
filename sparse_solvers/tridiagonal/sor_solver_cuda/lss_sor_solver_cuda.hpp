#pragma once
#if !defined(_LSS_SOR_SOLVER_CUDA_HPP_)
#define _LSS_SOR_SOLVER_CUDA_HPP_

#include "boundaries/lss_boundary_1d.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "common/lss_utility.hpp"
#include "containers/lss_flat_matrix.hpp"
#include "lss_sor_cuda_boundary.hpp"
#include "sparse_solvers/general/core_sor_solver_cuda/lss_core_sor_solver_cuda.hpp"
#include "sparse_solvers/general/sor_solver_traits/lss_sor_solver_traits.hpp"

namespace lss_sor_solver_cuda
{

using lss_boundary_1d::boundary_1d_ptr;
using lss_containers::flat_matrix;
using lss_core_sor_solver::core_sor_solver_cuda;
using lss_sor_solver_traits::sor_solver_traits;
using lss_utility::range;
using lss_utility::sptr_t;

template <typename fp_type, template <typename, typename> typename container = std::vector,
          typename allocator = std::allocator<fp_type>>
class sor_solver_cuda
{
  private:
    boundary_1d_ptr<fp_type> low_;
    boundary_1d_ptr<fp_type> high_;
    std::size_t discretization_size_;
    range<fp_type> space_range_;
    container<fp_type, allocator> a_, b_, c_, f_;
    fp_type omega_;

    template <template <typename fp_type> typename traits = sor_solver_traits>
    void kernel(container<fp_type, allocator> &solution, fp_type time);
    explicit sor_solver_cuda() = delete;

  public:
    typedef fp_type value_type;
    explicit sor_solver_cuda(range<fp_type> const &space_range, std::size_t discretization_size)
        : discretization_size_{discretization_size}
    {
    }

    ~sor_solver_cuda()
    {
    }

    sor_solver_cuda(sor_solver_cuda const &) = delete;
    sor_solver_cuda(sor_solver_cuda &&) = delete;
    sor_solver_cuda &operator=(sor_solver_cuda const &) = delete;
    sor_solver_cuda &operator=(sor_solver_cuda &&) = delete;

    void set_diagonals(container<fp_type, allocator> lower_diagonal, container<fp_type, allocator> diagonal,
                       container<fp_type, allocator> upper_diagonal);

    void set_rhs(container<fp_type, allocator> const &rhs);

    void set_boundary(const boundary_1d_ptr<fp_type> &low, const boundary_1d_ptr<fp_type> &high);

    void set_omega(fp_type value)
    {
        omega_ = value;
    }

    template <template <typename fp_type> typename traits = sor_solver_traits>
    void solve(container<fp_type, allocator> &solution);

    template <template <typename fp_type> typename traits = sor_solver_traits>
    container<fp_type, allocator> const solve();

    template <template <typename fp_type> typename traits = sor_solver_traits>
    void solve(container<fp_type, allocator> &solution, fp_type at_time);

    template <template <typename fp_type> typename traits = sor_solver_traits>
    container<fp_type, allocator> const solve(fp_type at_time);
};

template <typename fp_type> using sor_solver_cuda_ptr = sptr_t<sor_solver_cuda<fp_type>>;

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void sor_solver_cuda<fp_type, container, allocator>::set_diagonals(container<fp_type, allocator> lower_diagonal,
                                                                   container<fp_type, allocator> diagonal,
                                                                   container<fp_type, allocator> upper_diagonal)
{
    LSS_ASSERT(lower_diagonal.size() == discretization_size_, "Inncorect size for lower_diagonal");
    LSS_ASSERT(diagonal.size() == discretization_size_, "Inncorect size for diagonal");
    LSS_ASSERT(upper_diagonal.size() == discretization_size_, "Inncorect size for upper_diagonal");
    a_ = std::move(lower_diagonal);
    b_ = std::move(diagonal);
    c_ = std::move(upper_diagonal);
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void sor_solver_cuda<fp_type, container, allocator>::set_rhs(container<fp_type, allocator> const &rhs)
{
    LSS_ASSERT(rhs.size() == discretization_size_, "Inncorect size for right-hand side");
    f_ = rhs;
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void sor_solver_cuda<fp_type, container, allocator>::set_boundary(const boundary_1d_ptr<fp_type> &low,
                                                                  const boundary_1d_ptr<fp_type> &high)
{
    LSS_VERIFY(low, "Low boundary must be set");
    LSS_VERIFY(high, "High boundary must be set");
    low_ = low;
    high_ = high;
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
template <template <typename fp_type> typename traits>
void sor_solver_cuda<fp_type, container, allocator>::solve(container<fp_type, allocator> &solution)
{
    LSS_ASSERT(solution.size() == discretization_size_, "Incorrect size of solution container");
    kernel(solution, fp_type{});
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
template <template <typename fp_type> typename traits>
container<fp_type, allocator> const sor_solver_cuda<fp_type, container, allocator>::solve()
{
    container<fp_type, allocator> solution(discretization_size_);
    kernel(solution, fp_type{});
    return solution;
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
template <template <typename fp_type> typename traits>
void sor_solver_cuda<fp_type, container, allocator>::solve(container<fp_type, allocator> &solution, fp_type at_time)
{
    LSS_ASSERT(solution.size() == discretization_size_, "Incorrect size of solution container");
    kernel(solution, at_time);
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
template <template <typename fp_type> typename traits>
container<fp_type, allocator> const sor_solver_cuda<fp_type, container, allocator>::solve(fp_type at_time)
{
    container<fp_type, allocator> solution(discretization_size_);
    kernel(solution, at_time);
    return solution;
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
template <template <typename fp_type> typename traits>
void sor_solver_cuda<fp_type, container, allocator>::kernel(container<fp_type, allocator> &solution, fp_type time)
{
    // get proper boundaries:
    const std::size_t N = discretization_size_ - 1;
    const auto &lowest_quad = std::make_tuple(a_[0], b_[0], c_[0], f_[0]);
    const auto &lower_quad = std::make_tuple(a_[1], b_[1], c_[1], f_[1]);
    const auto &higher_quad = std::make_tuple(a_[N - 1], b_[N - 1], c_[N - 1], f_[N - 1]);
    const auto &highest_quad = std::make_tuple(a_[N], b_[N], c_[N], f_[N]);
    const fp_type step = space_range_.spread() / static_cast<fp_type>(N);
    sor_cuda_boundary<fp_type> boundary(low_, high_, lowest_quad, lower_quad, higher_quad, highest_quad,
                                        discretization_size_, step);
    const auto &init_coeffs = boundary.init_coefficients(time);
    const std::size_t start_idx = boundary.start_index();
    const auto &fin_coeffs = boundary.final_coefficients(time);
    const std::size_t end_idx = boundary.end_index();

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
    core_sor_solver_cuda<fp_type, container, allocator> sor(system_size);
    sor.set_flat_sparse_matrix(std::move(mat));
    sor.set_rhs(rhs);
    sor.set_omega(omega_);
    container<fp_type, allocator> sub_solution(system_size, fp_type{});
    sor.solve(sub_solution);

    std::copy(sub_solution.begin(), sub_solution.end(), std::next(solution.begin(), start_idx));
    // fill in the boundary values:
    if (start_idx == 1)
        solution[0] = boundary.lower_boundary(time);
    if (end_idx == N - 1)
        solution[N] = boundary.upper_boundary(time);
}

} // namespace lss_sor_solver_cuda
#endif ///_LSS_SOR_SOLVER_CUDA_HPP_
