#pragma once
#if !defined(_LSS_DOUBLE_SWEEP_SOLVER_HPP_)
#define _LSS_DOUBLE_SWEEP_SOLVER_HPP_

#pragma warning(disable : 4244)

#include <tuple>
#include <type_traits>
#include <vector>

#include "boundaries/lss_boundary_1d.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "common/lss_utility.hpp"
#include "lss_double_sweep_boundary.hpp"

namespace lss_double_sweep_solver
{

using lss_boundary_1d::boundary_1d_ptr;
using lss_utility::range;
using lss_utility::sptr_t;

/**
 * fdm_double_sweep_solver
 */
template <typename fp_type, template <typename, typename> typename container = std::vector,
          typename allocator = std::allocator<fp_type>>
class double_sweep_solver
{
  private:
    boundary_1d_ptr<fp_type> low_;
    boundary_1d_ptr<fp_type> high_;
    std::size_t discretization_size_;
    range<fp_type> space_range_;
    container<fp_type, allocator> a_, b_, c_, f_;
    container<fp_type, allocator> L_, K_;

    void kernel(container<fp_type, allocator> &solution, fp_type mtime);
    explicit double_sweep_solver() = delete;

  public:
    typedef fp_type value_type;
    typedef container<fp_type, allocator> container_type;
    explicit double_sweep_solver(range<fp_type> const &space_range, std::size_t discretization_size)
        : space_range_{space_range}, discretization_size_{discretization_size}
    {
    }

    ~double_sweep_solver()
    {
    }

    void set_diagonals(container<fp_type, allocator> lower_diagonal, container<fp_type, allocator> diagonal,
                       container<fp_type, allocator> upper_diagonal);

    void set_rhs(container<fp_type, allocator> const &rhs);

    void set_boundary(const boundary_1d_ptr<fp_type> &low, const boundary_1d_ptr<fp_type> &high);

    void solve(container<fp_type, allocator> &solution);

    container<fp_type, allocator> const solve();

    void solve(container<fp_type, allocator> &solution, fp_type at_time);

    container<fp_type, allocator> const solve(fp_type at_time);
};

template <typename fp_type> using double_sweep_solver_ptr = sptr_t<double_sweep_solver<fp_type>>;

} // namespace lss_double_sweep_solver

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void lss_double_sweep_solver::double_sweep_solver<fp_type, container, allocator>::set_diagonals(
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
void lss_double_sweep_solver::double_sweep_solver<fp_type, container, allocator>::set_rhs(
    container<fp_type, allocator> const &rhs)
{
    LSS_ASSERT(rhs.size() == discretization_size_, "Inncorect size for right-hand side");
    f_ = rhs;
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void lss_double_sweep_solver::double_sweep_solver<fp_type, container, allocator>::set_boundary(
    const boundary_1d_ptr<fp_type> &low, const boundary_1d_ptr<fp_type> &high)
{
    LSS_VERIFY(low, "Low boundary must be set");
    LSS_VERIFY(high, "High boundary must be set");
    low_ = low;
    high_ = high;
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void lss_double_sweep_solver::double_sweep_solver<fp_type, container, allocator>::solve(
    container<fp_type, allocator> &solution)
{
    LSS_ASSERT(solution.size() == discretization_size_, "Incorrect size of solution container");
    kernel(solution, fp_type{});
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
container<fp_type, allocator> const lss_double_sweep_solver::double_sweep_solver<fp_type, container, allocator>::solve()
{
    container<fp_type, allocator> solution(discretization_size_);
    kernel(solution, fp_type{});
    return solution;
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void lss_double_sweep_solver::double_sweep_solver<fp_type, container, allocator>::solve(
    container<fp_type, allocator> &solution, fp_type at_time)
{
    LSS_ASSERT(solution.size() == discretization_size_, "Incorrect size of solution container");
    kernel(solution, at_time);
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
container<fp_type, allocator> const lss_double_sweep_solver::double_sweep_solver<fp_type, container, allocator>::solve(
    fp_type at_time)
{
    container<fp_type, allocator> solution(discretization_size_);
    kernel(solution, at_time);
    return solution;
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void lss_double_sweep_solver::double_sweep_solver<fp_type, container, allocator>::kernel(
    container<fp_type, allocator> &solution, fp_type time)
{
    // clear coefficients:
    K_.clear();
    L_.clear();
    // resize coefficients:
    K_.resize(discretization_size_);
    L_.resize(discretization_size_);
    // get proper boundaries:
    const std::size_t N = discretization_size_ - 1;
    const auto &low_quad = std::make_tuple(a_[0], b_[0], c_[0], f_[0]);
    const fp_type step = space_range_.spread() / static_cast<fp_type>(N);
    double_sweep_boundary<fp_type> boundary(low_, high_, low_quad, discretization_size_, step);
    // init coefficients:
    const auto pair = boundary.coefficients(time);
    const std::size_t start_index = boundary.start_index();
    const std::size_t end_index = boundary.end_index();

    L_[0] = std::get<1>(pair);
    K_[0] = std::get<0>(pair);

    fp_type tmp{};
    fp_type mone = static_cast<fp_type>(-1.0);
    for (std::size_t t = 1; t <= end_index; ++t)
    {
        tmp = b_[t] + (a_[t] * L_[t - 1]);
        L_[t] = mone * c_[t] / tmp;
        K_[t] = (f_[t] - (a_[t] * K_[t - 1])) / tmp;
    }

    f_[N] = boundary.upper_boundary(K_[N - 1], K_[N], L_[N - 1], L_[N], time);

    for (long long t = N - 1; t >= start_index && t >= 0; --t)
    {
        f_[t] = (L_[t] * f_[t + 1]) + K_[t];
    }
    if (start_index == 1)
        f_[0] = boundary.lower_boundary(time);

    std::copy(f_.begin(), f_.end(), solution.begin());
}

#endif ///_LSS_DOUBLE_SWEEP_SOLVER_HPP_
