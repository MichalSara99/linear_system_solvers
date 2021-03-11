#pragma once
#if !defined(_LSS_2D_PDE_UTILITY)
#define _LSS_2D_PDE_UTILITY

#include <functional>
#include <tuple>

#include "common/lss_macros.h"
#include "common/lss_utility.h"

namespace lss_two_dim_pde_utility {

using lss_utility::coefficient_holder;
using lss_utility::container_2d;
using lss_utility::range;

// Two-dim PDE coefficient holder
template <typename type>
using two_dim_coefficient_holder =
    coefficient_holder<type, type, type, type, type, type>;

// Two-dim PDE coefficients (a,b,c,d,e,f)
template <typename fp_type>
using pde_coefficient_holder_const = two_dim_coefficient_holder<fp_type>;

// Two-dim PDE coefficients (a(x,y),b(x,y),c(x,y),d(x,y),e(x,y),f(x,y))
template <typename fp_type>
using pde_coefficient_holder_fun_2_arg =
    two_dim_coefficient_holder<std::function<fp_type(fp_type, fp_type)>>;

// Two-dim Dirichlet boundary:
template <typename fp_type>
struct dirichlet_boundary_2d {
 public:
  typedef std::function<fp_type(fp_type, fp_type)> fun_2d;
  std::pair<fun_2d, fun_2d> first_dim;
  std::pair<fun_2d, fun_2d> second_dim;

  explicit dirichlet_boundary_2d() {}
  explicit dirichlet_boundary_2d(std::pair<fun_2d, fun_2d> const &first_pair,
                                 std::pair<fun_2d, fun_2d> const &second_pair)
      : first_dim{first_pair}, second_dim{second_pair} {}
};

// TO BE CONSIDERED LATER:

// One-dim Robin boundary:
// template <typename fp_type>
// struct robin_boundary {
//  std::pair<fp_type, fp_type> left;
//  std::pair<fp_type, fp_type> right;
//
//  robin_boundary() {}
//  explicit robin_boundary(std::pair<fp_type, fp_type> const &left_boundary,
//                          std::pair<fp_type, fp_type> const &right_boundary)
//      : left{left_boundary}, right{right_boundary} {}
//};

template <typename fp_type>
struct heat_data_2d {
  // range for first and second space variable
  std::pair<range<fp_type>, range<fp_type>> space_range;
  // range for time variable
  range<fp_type> time_range;
  // Number of time subdivisions
  std::size_t time_division;
  // Number of space subdivisions
  std::pair<std::size_t, std::size_t> space_division;
  // Initial condition
  std::function<fp_type(fp_type, fp_type)> initial_condition;
  // terminal condition
  std::function<fp_type(fp_type, fp_type)> terminal_condition;
  // Independent source function
  std::function<fp_type(fp_type, fp_type, fp_type)> source_function;
  // Flag on source function
  bool is_source_function_set;

  explicit heat_data_2d(
      std::pair<range<fp_type>, range<fp_type>> const &space,
      range<fp_type> const &time,
      std::pair<std::size_t, std::size_t> const &space_subdivision,
      std::size_t const &time_subdivision,
      std::function<fp_type(fp_type, fp_type)> const &initial_condition,
      std::function<fp_type(fp_type, fp_type)> const &terminal_condition,
      std::function<fp_type(fp_type, fp_type, fp_type)> const &source,
      bool is_source_set)
      : space_range{space},
        time_range{time},
        space_division{space_subdivision},
        time_division{time_subdivision},
        initial_condition{initial_condition},
        terminal_condition{terminal_condition},
        source_function{source},
        is_source_function_set{is_source_set} {}

 protected:
  heat_data_2d() {}
};

template <typename fp_type, template <typename, typename> typename container,
          typename alloc>
class discretization_2d {
 public:
  static void discretize_space(fp_type const &step, fp_type const &init,
                               container<fp_type, alloc> &container);

  static void discretize_initial_condition(
      std::pair<fp_type, fp_type> const &inits,
      std::pair<fp_type, fp_type> const &steps,
      std::function<fp_type(fp_type, fp_type)> const &fun,
      container_2d<container, fp_type, alloc> &container);

  static void discretize_in_space(
      std::pair<fp_type, fp_type> const &starts,
      std::pair<fp_type, fp_type> const &steps, fp_type const &time,
      std::function<fp_type(fp_type, fp_type, fp_type)> const &fun,
      container_2d<container, fp_type, alloc> &output);
};

template <typename fp_type, template <typename, typename> typename container,
          typename alloc>
void discretization_2d<fp_type, container, alloc>::discretize_space(
    fp_type const &step, fp_type const &init,
    container<fp_type, alloc> &container) {
  LSS_ASSERT(container.size() > 0, "The input container must be initialized.");
  container[0] = init;
  for (std::size_t t = 1; t < container.size(); ++t) {
    container[t] = container[t - 1] + step;
  }
}

template <typename fp_type, template <typename, typename> typename container,
          typename alloc>
void discretization_2d<fp_type, container, alloc>::discretize_initial_condition(
    std::pair<fp_type, fp_type> const &inits,
    std::pair<fp_type, fp_type> const &steps,
    std::function<fp_type(fp_type, fp_type)> const &fun,
    container_2d<container, fp_type, alloc> &container) {
  LSS_ASSERT(container.rows() > 0, "The input container must be initialized.");
  LSS_ASSERT(container.columns() > 0,
             "The input container must be initialized.");
  auto const x_init = inits.first;
  auto const y_init = inits.second;
  auto const x_step = steps.first;
  auto const y_step = steps.second;

  auto const rows = container.rows();
  auto const cols = container.columns();
  for (std::size_t y_idx = 0; y_idx < rows; ++y_idx) {
    for (std::size_t x_idx = 0; x_idx < cols; ++x_idx) {
      container(y_idx, x_idx,
                fun(x_init + static_cast<fp_type>(x_idx) * x_step,
                    y_init + static_cast<fp_type>(y_idx) * y_step));
    }
  }
}

template <typename fp_type, template <typename, typename> typename container,
          typename alloc>
void discretization_2d<fp_type, container, alloc>::discretize_in_space(
    std::pair<fp_type, fp_type> const &inits,
    std::pair<fp_type, fp_type> const &steps, fp_type const &time,
    std::function<fp_type(fp_type, fp_type, fp_type)> const &fun,
    container_2d<container, fp_type, alloc> &container) {
  LSS_ASSERT(container.rows() > 0, "The input container must be initialized.");
  LSS_ASSERT(container.columns() > 0,
             "The input container must be initialized.");
  auto const x_init = inits.first;
  auto const y_init = inits.second;
  auto const x_step = steps.first;
  auto const y_step = steps.second;

  auto const rows = container.rows();
  auto const cols = container.columns();
  for (std::size_t y_idx = 0; y_idx < rows; ++y_idx) {
    for (std::size_t x_idx = 0; x_idx < cols; ++x_idx) {
      container(y_idx, x_idx,
                fun(x_init + static_cast<fp_type>(x_idx) * x_step,
                    y_init + static_cast<fp_type>(y_idx) * y_step, time));
    }
  }
}

// vector discretization:
template <typename fp_type>
using v_discretization_2d =
    discretization_2d<fp_type, std::vector, std::allocator<fp_type>>;

}  // namespace lss_two_dim_pde_utility

#endif  ///_LSS_2D_PDE_UTILITY
