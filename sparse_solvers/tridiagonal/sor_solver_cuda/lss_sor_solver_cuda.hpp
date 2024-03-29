#pragma once
#if !defined(_LSS_SOR_SOLVER_CUDA_HPP_)
#define _LSS_SOR_SOLVER_CUDA_HPP_

#include "boundaries/lss_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "common/lss_utility.hpp"
#include "containers/lss_flat_matrix.hpp"
#include "lss_sor_cuda_boundary.hpp"
#include "sparse_solvers/general/core_sor_solver_cuda/lss_core_sor_solver_cuda.hpp"
#include "sparse_solvers/general/sor_solver_traits/lss_sor_solver_traits.hpp"

namespace lss_sor_solver_cuda
{

using lss_boundary::boundary_pair;
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
    std::size_t discretization_size_;
    container<fp_type, allocator> a_, b_, c_, f_;
    fp_type omega_;
    sor_cuda_boundary_ptr<fp_type> sor_boundary_;

    template <typename... fp_space_types>
    void kernel(boundary_pair<fp_type, fp_space_types...> const &boundary, container<fp_type, allocator> &solution,
                fp_type time, fp_space_types... space_args);

    void initialize()
    {
        const fp_type one = static_cast<fp_type>(1.0);
        const fp_type step = one / static_cast<fp_type>(discretization_size_ - 1);
        sor_boundary_ = std::make_shared<sor_cuda_boundary<fp_type>>(discretization_size_, step);
    }

    explicit sor_solver_cuda() = delete;

  public:
    typedef fp_type value_type;
    explicit sor_solver_cuda(std::size_t discretization_size) : discretization_size_{discretization_size}
    {
        initialize();
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

    void set_omega(fp_type value)
    {
        omega_ = value;
    }

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
template <typename... fp_space_types>
void sor_solver_cuda<fp_type, container, allocator>::kernel(boundary_pair<fp_type, fp_space_types...> const &boundary,
                                                            container<fp_type, allocator> &solution, fp_type time,
                                                            fp_space_types... space_args)
{
    // get proper boundaries:
    const std::size_t N = discretization_size_ - 1;
    const auto &lowest_quad = std::make_tuple(a_[0], b_[0], c_[0], f_[0]);
    const auto &lower_quad = std::make_tuple(a_[1], b_[1], c_[1], f_[1]);
    const auto &higher_quad = std::make_tuple(a_[N - 1], b_[N - 1], c_[N - 1], f_[N - 1]);
    const auto &highest_quad = std::make_tuple(a_[N], b_[N], c_[N], f_[N]);

    sor_boundary_->set_lowest_quad(lowest_quad);
    sor_boundary_->set_lower_quad(lower_quad);
    sor_boundary_->set_higher_quad(higher_quad);
    sor_boundary_->set_highest_quad(highest_quad);

    const auto &init_coeffs = sor_boundary_->init_coefficients(boundary, time, space_args...);
    const std::size_t start_idx = sor_boundary_->start_index();
    const auto &fin_coeffs = sor_boundary_->final_coefficients(boundary, time, space_args...);
    const std::size_t end_idx = sor_boundary_->end_index();

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
        solution[0] = sor_boundary_->lower_boundary(boundary, time, space_args...);
    if (end_idx == N - 1)
        solution[N] = sor_boundary_->upper_boundary(boundary, time, space_args...);
}

} // namespace lss_sor_solver_cuda
#endif ///_LSS_SOR_SOLVER_CUDA_HPP_
