#pragma once
#if !defined(_LSS_1D_SPACE_VARIABLE_HEAT_EXPLICIT_SCHEMES_POLICY)
#define _LSS_1D_SPACE_VARIABLE_HEAT_EXPLICIT_SCHEMES_POLICY

#pragma warning(disable : 4244)

#include <thread>

#include "common/lss_enumerations.h"
#include "pde_solvers/one_dim/lss_base_explicit_schemes.h"
#include "pde_solvers/one_dim/lss_pde_utility.h"

namespace lss_one_dim_space_variable_heat_explicit_schemes_policy {

using lss_enumerations::boundary_condition_enum;
using lss_one_dim_base_explicit_schemes::heat_scheme_base;
using lss_one_dim_pde_utility::dirichlet_boundary;
using lss_one_dim_pde_utility::discretization;
using lss_one_dim_pde_utility::pde_coefficient_holder_fun_1_arg;
using lss_one_dim_pde_utility::robin_boundary;

// ============================================================================
// ========================= heat_euler_scheme_policy =========================
// ============================================================================

template <typename fp_type, typename coefficient_holder,
          template <typename, typename> typename container, typename alloc>
struct heat_euler_scheme_forward_policy {
  // Dirichlet boundary without source:
  static void traverse(container<fp_type, alloc> &solution,
                       container<fp_type, alloc> const &init_solution,
                       dirichlet_boundary<fp_type> const &dirichlet_boundary,
                       std::pair<fp_type, fp_type> const &deltas,
                       coefficient_holder const &holder,
                       fp_type const &terminal_time);
  // Dirichlet boundary with source:
  static void traverse(container<fp_type, alloc> &solution,
                       container<fp_type, alloc> const &init_solution,
                       dirichlet_boundary<fp_type> const &dirichlet_boundary,
                       std::pair<fp_type, fp_type> const &deltas,
                       coefficient_holder const &holder,
                       fp_type const &terminal_time, fp_type const &space_start,
                       std::function<fp_type(fp_type, fp_type)> const &source);
  // Robin boundary without source:
  static void traverse(container<fp_type, alloc> &solution,
                       container<fp_type, alloc> const &init_solution,
                       robin_boundary<fp_type> const &robin_boundary,
                       std::pair<fp_type, fp_type> const &deltas,
                       coefficient_holder const &holder,
                       fp_type const &terminal_time);
  // Robin boundary with source:
  static void traverse(container<fp_type, alloc> &solution,
                       container<fp_type, alloc> const &init_solution,
                       robin_boundary<fp_type> const &robin_boundary,
                       std::pair<fp_type, fp_type> const &deltas,
                       coefficient_holder const &holder,
                       fp_type const &terminal_time, fp_type const &space_start,
                       std::function<fp_type(fp_type, fp_type)> const &source);
};

template <typename fp_type, typename coefficient_holder,
          template <typename, typename> typename container, typename alloc>
struct heat_euler_scheme_backward_policy {
  // Dirichlet boundary without source:
  static void traverse(container<fp_type, alloc> &solution,
                       container<fp_type, alloc> const &init_solution,
                       dirichlet_boundary<fp_type> const &dirichlet_boundary,
                       std::pair<fp_type, fp_type> const &deltas,
                       coefficient_holder const &holder,
                       fp_type const &terminal_time);
  // Dirichlet boundary with source:
  static void traverse(container<fp_type, alloc> &solution,
                       container<fp_type, alloc> const &init_solution,
                       dirichlet_boundary<fp_type> const &dirichlet_boundary,
                       std::pair<fp_type, fp_type> const &deltas,
                       coefficient_holder const &holder,
                       fp_type const &terminal_time, fp_type const &space_start,
                       std::function<fp_type(fp_type, fp_type)> const &source);
  // Robin boundary without source:
  static void traverse(container<fp_type, alloc> &solution,
                       container<fp_type, alloc> const &init_solution,
                       robin_boundary<fp_type> const &robin_boundary,
                       std::pair<fp_type, fp_type> const &deltas,
                       coefficient_holder const &holder,
                       fp_type const &terminal_time);
  // Robin boundary with source:
  static void traverse(container<fp_type, alloc> &solution,
                       container<fp_type, alloc> const &init_solution,
                       robin_boundary<fp_type> const &robin_boundary,
                       std::pair<fp_type, fp_type> const &deltas,
                       coefficient_holder const &holder,
                       fp_type const &terminal_time, fp_type const &space_start,
                       std::function<fp_type(fp_type, fp_type)> const &source);
};

// ============================================================================
// ============== ade_heat_bakarat_clark_scheme_policy ========================
// ============================================================================

template <typename fp_type, typename coefficient_holder,
          template <typename, typename> typename container, typename alloc>
struct ade_heat_bakarat_clark_scheme_forward_policy {
  // Dirichlet boundary without source:
  static void traverse(container<fp_type, alloc> &solution,
                       container<fp_type, alloc> const &init_solution,
                       dirichlet_boundary<fp_type> const &dirichlet_boundary,
                       std::pair<fp_type, fp_type> const &deltas,
                       coefficient_holder const &holder,
                       fp_type const &terminal_time);
  // Dirichlet boundary with source:
  static void traverse(container<fp_type, alloc> &solution,
                       container<fp_type, alloc> const &init_solution,
                       dirichlet_boundary<fp_type> const &dirichlet_boundary,
                       std::pair<fp_type, fp_type> const &deltas,
                       coefficient_holder const &holder,
                       fp_type const &terminal_time, fp_type const &space_start,
                       std::function<fp_type(fp_type, fp_type)> const &source);
};

template <typename fp_type, typename coefficient_holder,
          template <typename, typename> typename container, typename alloc>
struct ade_heat_bakarat_clark_scheme_backward_policy {
  // Dirichlet boundary without source:
  static void traverse(container<fp_type, alloc> &solution,
                       container<fp_type, alloc> const &init_solution,
                       dirichlet_boundary<fp_type> const &dirichlet_boundary,
                       std::pair<fp_type, fp_type> const &deltas,
                       coefficient_holder const &holder,
                       fp_type const &terminal_time);
  // Dirichlet boundary with source:
  static void traverse(container<fp_type, alloc> &solution,
                       container<fp_type, alloc> const &init_solution,
                       dirichlet_boundary<fp_type> const &dirichlet_boundary,
                       std::pair<fp_type, fp_type> const &deltas,
                       coefficient_holder const &holder,
                       fp_type const &terminal_time, fp_type const &space_start,
                       std::function<fp_type(fp_type, fp_type)> const &source);
};

// ============================================================================
// ================== ade_heat_saulyev_scheme_policy ==========================
// ============================================================================

template <typename fp_type, typename coefficient_holder,
          template <typename, typename> typename container, typename alloc>
struct ade_heat_saulyev_scheme_forward_policy {
  // Dirichlet boundary without source:
  static void traverse(container<fp_type, alloc> &solution,
                       container<fp_type, alloc> const &init_solution,
                       dirichlet_boundary<fp_type> const &dirichlet_boundary,
                       std::pair<fp_type, fp_type> const &deltas,
                       coefficient_holder const &holder,
                       fp_type const &terminal_time);
  // Dirichlet boundary with source:
  static void traverse(container<fp_type, alloc> &solution,
                       container<fp_type, alloc> const &init_solution,
                       dirichlet_boundary<fp_type> const &dirichlet_boundary,
                       std::pair<fp_type, fp_type> const &deltas,
                       coefficient_holder const &holder,
                       fp_type const &terminal_time, fp_type const &space_start,
                       std::function<fp_type(fp_type, fp_type)> const &source);
};

template <typename fp_type, typename coefficient_holder,
          template <typename, typename> typename container, typename alloc>
struct ade_heat_saulyev_scheme_backward_policy {
  // Dirichlet boundary without source:
  static void traverse(container<fp_type, alloc> &solution,
                       container<fp_type, alloc> const &init_solution,
                       dirichlet_boundary<fp_type> const &dirichlet_boundary,
                       std::pair<fp_type, fp_type> const &deltas,
                       coefficient_holder const &holder,
                       fp_type const &terminal_time);
  // Dirichlet boundary with source:
  static void traverse(container<fp_type, alloc> &solution,
                       container<fp_type, alloc> const &init_solution,
                       dirichlet_boundary<fp_type> const &dirichlet_boundary,
                       std::pair<fp_type, fp_type> const &deltas,
                       coefficient_holder const &holder,
                       fp_type const &terminal_time, fp_type const &space_start,
                       std::function<fp_type(fp_type, fp_type)> const &source);
};

}  // namespace lss_one_dim_space_variable_heat_explicit_schemes_policy

// ============================================================================
// ======================== IMPLEMENTATIONS ===================================

// ============================================================================
// =============== heat_euler_scheme_forward_policy ===========================
// ============================================================================

template <typename fp_type, typename coefficient_holder,
          template <typename, typename> typename container, typename alloc>
void lss_one_dim_space_variable_heat_explicit_schemes_policy::
    heat_euler_scheme_forward_policy<
        fp_type, coefficient_holder, container,
        alloc>::traverse(container<fp_type, alloc> &solution,
                         container<fp_type, alloc> const &init_solution,
                         dirichlet_boundary<fp_type> const &dirichlet_boundary,
                         std::pair<fp_type, fp_type> const &deltas,
                         coefficient_holder const &holder,
                         fp_type const &terminal_time) {
  // get delta time:
  fp_type const k = std::get<0>(deltas);
  // get delta space:
  fp_type const h = std::get<1>(deltas);
  // create first time point:
  fp_type time = k;
  // left space boundary:
  auto const &left = dirichlet_boundary.first;
  // right space boundary:
  auto const &right = dirichlet_boundary.second;
  // previous solution:
  container<fp_type, alloc> prev_sol(init_solution);
  // get coefficients:
  auto const &A = std::get<0>(holder);
  auto const &B = std::get<1>(holder);
  auto const &D = std::get<2>(holder);
  // size of the space vector:
  std::size_t const space_size = solution.size();

  while (time <= terminal_time) {
    solution[0] = left(time);
    solution[solution.size() - 1] = right(time);
    for (std::size_t t = 1; t < space_size - 1; ++t) {
      solution[t] = (1.0 - 2.0 * B(t * h)) * prev_sol[t] +
                    D(t * h) * prev_sol[t + 1] + A(t * h) * prev_sol[t - 1];
    }
    prev_sol = solution;
    time += k;
  }
}

template <typename fp_type, typename coefficient_holder,
          template <typename, typename> typename container, typename alloc>
void lss_one_dim_space_variable_heat_explicit_schemes_policy::
    heat_euler_scheme_forward_policy<fp_type, coefficient_holder, container,
                                     alloc>::
        traverse(container<fp_type, alloc> &solution,
                 container<fp_type, alloc> const &init_solution,
                 dirichlet_boundary<fp_type> const &dirichlet_boundary,
                 std::pair<fp_type, fp_type> const &deltas,
                 coefficient_holder const &holder, fp_type const &terminal_time,
                 fp_type const &space_start,
                 std::function<fp_type(fp_type, fp_type)> const &source) {
  // get delta time:
  fp_type const k = std::get<0>(deltas);
  // get delta space:
  fp_type const h = std::get<1>(deltas);
  // create first time point:
  fp_type time = k;
  // left space boundary:
  auto const &left = dirichlet_boundary.first;
  // right space boundary:
  auto const &right = dirichlet_boundary.second;
  // previous solution:
  container<fp_type, alloc> prev_sol(init_solution);
  // get coefficients:
  auto const &A = std::get<0>(holder);
  auto const &B = std::get<1>(holder);
  auto const &D = std::get<2>(holder);
  // size of the space vector:
  std::size_t const space_size = solution.size();

  // create a container to carry discretized source heat
  container<fp_type, alloc> source_curr(solution.size(), fp_type{});
  discretization<fp_type, container, alloc>::discretize_in_space(
      h, space_start, 0.0, source, source_curr);

  // loop for stepping in time:
  while (time <= terminal_time) {
    solution[0] = left(time);
    solution[solution.size() - 1] = right(time);
    for (std::size_t t = 1; t < space_size - 1; ++t) {
      solution[t] = solution[t] =
          (1.0 - 2.0 * B(t * h)) * prev_sol[t] + D(t * h) * prev_sol[t + 1] +
          A(t * h) * prev_sol[t - 1] + k * source_curr[t];
    }
    discretization<fp_type, container, alloc>::discretize_in_space(
        h, space_start, time, source, source_curr);
    prev_sol = solution;
    time += k;
  }
}

template <typename fp_type, typename coefficient_holder,
          template <typename, typename> typename container, typename alloc>
void lss_one_dim_space_variable_heat_explicit_schemes_policy::
    heat_euler_scheme_forward_policy<
        fp_type, coefficient_holder, container,
        alloc>::traverse(container<fp_type, alloc> &solution,
                         container<fp_type, alloc> const &init_solution,
                         robin_boundary<fp_type> const &robin_boundary,
                         std::pair<fp_type, fp_type> const &deltas,
                         coefficient_holder const &holder,
                         fp_type const &terminal_time) {
  // get delta time:
  fp_type const k = std::get<0>(deltas);
  // get delta space:
  fp_type const h = std::get<1>(deltas);
  // create first time point:
  fp_type time = k;
  // get coefficients:
  auto const &A = std::get<0>(holder);
  auto const &B = std::get<1>(holder);
  auto const &D = std::get<2>(holder);
  // left space boundary:
  fp_type const left_lin = robin_boundary.left.first;
  fp_type const left_const = robin_boundary.left.second;
  // right space boundary:
  fp_type const right_lin_ = robin_boundary.right.first;
  fp_type const right_const_ = robin_boundary.right.second;
  // conversion of right hand boundaries:
  fp_type const right_lin = 1.0 / right_lin_;
  fp_type const right_const = -1.0 * (right_const_ / right_lin_);
  // previous solution:
  container<fp_type, alloc> prev_sol(init_solution);
  // size of the space vector:
  std::size_t const space_size = solution.size();

  while (time <= terminal_time) {
    solution[0] = (D(0) + (A(0) * left_lin)) * prev_sol[1] +
                  (1.0 - 2.0 * B(0)) * prev_sol[0] + A(0) * left_const;
    solution[space_size - 1] =
        (A((space_size - 1) * h) + (D((space_size - 1) * h) * right_lin)) *
            prev_sol[space_size - 2] +
        (1.0 - 2.0 * B((space_size - 1) * h)) * prev_sol[space_size - 1] +
        D((space_size - 1) * h) * right_const;
    for (std::size_t t = 1; t < space_size - 1; ++t) {
      solution[t] = (1.0 - 2.0 * B(t * h)) * prev_sol[t] +
                    D(t * h) * prev_sol[t + 1] + A(t * h) * prev_sol[t - 1];
    }
    prev_sol = solution;
    time += k;
  }
}

template <typename fp_type, typename coefficient_holder,
          template <typename, typename> typename container, typename alloc>
void lss_one_dim_space_variable_heat_explicit_schemes_policy::
    heat_euler_scheme_forward_policy<fp_type, coefficient_holder, container,
                                     alloc>::
        traverse(container<fp_type, alloc> &solution,
                 container<fp_type, alloc> const &init_solution,
                 robin_boundary<fp_type> const &robin_boundary,
                 std::pair<fp_type, fp_type> const &deltas,
                 coefficient_holder const &holder, fp_type const &terminal_time,
                 fp_type const &space_start,
                 std::function<fp_type(fp_type, fp_type)> const &source) {
  // get delta time:
  fp_type const k = std::get<0>(deltas);
  // get delta space:
  fp_type const h = std::get<1>(deltas);
  // create first time point:
  fp_type time = k;
  // get coefficients:
  auto const &A = std::get<0>(holder);
  auto const &B = std::get<1>(holder);
  auto const &D = std::get<2>(holder);
  // left space boundary:
  fp_type const left_lin = robin_boundary.left.first;
  fp_type const left_const = robin_boundary.left.second;
  // right space boundary:
  fp_type const right_lin_ = robin_boundary.right.first;
  fp_type const right_const_ = robin_boundary.right.second;
  // conversion of right hand boundaries:
  fp_type const right_lin = 1.0 / right_lin_;
  fp_type const right_const = -1.0 * (right_const_ / right_lin_);
  // previous solution:
  container<fp_type, alloc> prev_sol(init_solution);
  // size of the space vector:
  std::size_t const space_size = solution.size();

  // create a container to carry discretized source heat
  container<fp_type, alloc> source_curr(solution.size(), fp_type{});
  discretization<fp_type, container, alloc>::discretize_in_space(
      h, space_start, 0.0, source, source_curr);

  // loop for stepping in time:
  while (time <= terminal_time) {
    solution[0] = (D(0) + (A(0) * left_lin)) * prev_sol[1] +
                  (1.0 - 2.0 * B(0)) * prev_sol[0] + A(0) * left_const +
                  k * source_curr[0];
    solution[space_size - 1] =
        (A((space_size - 1) * h) + (D((space_size - 1) * h) * right_lin)) *
            prev_sol[space_size - 2] +
        (1.0 - 2.0 * B((space_size - 1) * h)) * prev_sol[space_size - 1] +
        D((space_size - 1) * h) * right_const + k * source_curr[space_size - 1];
    for (std::size_t t = 1; t < space_size - 1; ++t) {
      solution[t] = (1.0 - 2.0 * B(t * h)) * prev_sol[t] +
                    D(t * h) * prev_sol[t + 1] + A(t * h) * prev_sol[t - 1] +
                    k * source_curr[t];
    }
    discretization<fp_type, container, alloc>::discretize_in_space(
        h, space_start, time, source, source_curr);
    prev_sol = solution;
    time += k;
  }
}

// ============================================================================
// =============== heat_euler_scheme_backward_policy ==========================
// ============================================================================

template <typename fp_type, typename coefficient_holder,
          template <typename, typename> typename container, typename alloc>
void lss_one_dim_space_variable_heat_explicit_schemes_policy::
    heat_euler_scheme_backward_policy<
        fp_type, coefficient_holder, container,
        alloc>::traverse(container<fp_type, alloc> &solution,
                         container<fp_type, alloc> const &init_solution,
                         dirichlet_boundary<fp_type> const &dirichlet_boundary,
                         std::pair<fp_type, fp_type> const &deltas,
                         coefficient_holder const &holder,
                         fp_type const &terminal_time) {
  // get delta time:
  fp_type const k = std::get<0>(deltas);
  // get delta space:
  fp_type const h = std::get<1>(deltas);
  // create last but one time point:
  fp_type time = terminal_time - k;
  // left space boundary:
  auto const &left = dirichlet_boundary.first;
  // right space boundary:
  auto const &right = dirichlet_boundary.second;
  // previous solution:
  container<fp_type, alloc> prev_sol(init_solution);
  // get coefficients:
  auto const &A = std::get<0>(holder);
  auto const &B = std::get<1>(holder);
  auto const &D = std::get<2>(holder);
  // size of the space vector:
  std::size_t const space_size = solution.size();

  while (time >= 0.0) {
    solution[0] = left(time);
    solution[solution.size() - 1] = right(time);
    for (std::size_t t = 1; t < space_size - 1; ++t) {
      solution[t] = (1.0 - 2.0 * B(t * h)) * prev_sol[t] +
                    D(t * h) * prev_sol[t + 1] + A(t * h) * prev_sol[t - 1];
    }
    prev_sol = solution;
    time -= k;
  }
}

template <typename fp_type, typename coefficient_holder,
          template <typename, typename> typename container, typename alloc>
void lss_one_dim_space_variable_heat_explicit_schemes_policy::
    heat_euler_scheme_backward_policy<fp_type, coefficient_holder, container,
                                      alloc>::
        traverse(container<fp_type, alloc> &solution,
                 container<fp_type, alloc> const &init_solution,
                 dirichlet_boundary<fp_type> const &dirichlet_boundary,
                 std::pair<fp_type, fp_type> const &deltas,
                 coefficient_holder const &holder, fp_type const &terminal_time,
                 fp_type const &space_start,
                 std::function<fp_type(fp_type, fp_type)> const &source) {
  // get delta time:
  fp_type const k = std::get<0>(deltas);
  // get delta space:
  fp_type const h = std::get<1>(deltas);
  // create last but one time point:
  fp_type time = terminal_time - k;
  // left space boundary:
  auto const &left = dirichlet_boundary.first;
  // right space boundary:
  auto const &right = dirichlet_boundary.second;
  // previous solution:
  container<fp_type, alloc> prev_sol(init_solution);
  // get coefficients:
  auto const &A = std::get<0>(holder);
  auto const &B = std::get<1>(holder);
  auto const &D = std::get<2>(holder);
  // size of the space vector:
  std::size_t const space_size = solution.size();

  // create a container to carry discretized source heat
  container<fp_type, alloc> source_curr(solution.size(), fp_type{});
  discretization<fp_type, container, alloc>::discretize_in_space(
      h, space_start, 0.0, source, source_curr);

  // loop for stepping in time:
  while (time >= 0.0) {
    solution[0] = left(time);
    solution[solution.size() - 1] = right(time);
    for (std::size_t t = 1; t < space_size - 1; ++t) {
      solution[t] = solution[t] =
          (1.0 - 2.0 * B(t * h)) * prev_sol[t] + D(t * h) * prev_sol[t + 1] +
          A(t * h) * prev_sol[t - 1] + k * source_curr[t];
    }
    discretization<fp_type, container, alloc>::discretize_in_space(
        h, space_start, time, source, source_curr);
    prev_sol = solution;
    time -= k;
  }
}

template <typename fp_type, typename coefficient_holder,
          template <typename, typename> typename container, typename alloc>
void lss_one_dim_space_variable_heat_explicit_schemes_policy::
    heat_euler_scheme_backward_policy<
        fp_type, coefficient_holder, container,
        alloc>::traverse(container<fp_type, alloc> &solution,
                         container<fp_type, alloc> const &init_solution,
                         robin_boundary<fp_type> const &robin_boundary,
                         std::pair<fp_type, fp_type> const &deltas,
                         coefficient_holder const &holder,
                         fp_type const &terminal_time) {
  // get delta time:
  fp_type const k = std::get<0>(deltas);
  // get delta space:
  fp_type const h = std::get<1>(deltas);
  // create last but one time point:
  fp_type time = terminal_time - k;
  // get coefficients:
  auto const &A = std::get<0>(holder);
  auto const &B = std::get<1>(holder);
  auto const &D = std::get<2>(holder);
  // left space boundary:
  fp_type const left_lin = robin_boundary.left.first;
  fp_type const left_const = robin_boundary.left.second;
  // right space boundary:
  fp_type const right_lin_ = robin_boundary.right.first;
  fp_type const right_const_ = robin_boundary.right.second;
  // conversion of right hand boundaries:
  fp_type const right_lin = 1.0 / right_lin_;
  fp_type const right_const = -1.0 * (right_const_ / right_lin_);
  // previous solution:
  container<fp_type, alloc> prev_sol(init_solution);
  // size of the space vector:
  std::size_t const space_size = solution.size();

  while (time >= 0.0) {
    solution[0] = (D(0) + (A(0) * left_lin)) * prev_sol[1] +
                  (1.0 - 2.0 * B(0)) * prev_sol[0] + A(0) * left_const;
    solution[space_size - 1] =
        (A((space_size - 1) * h) + (D((space_size - 1) * h) * right_lin)) *
            prev_sol[space_size - 2] +
        (1.0 - 2.0 * B((space_size - 1) * h)) * prev_sol[space_size - 1] +
        D((space_size - 1) * h) * right_const;
    for (std::size_t t = 1; t < space_size - 1; ++t) {
      solution[t] = (1.0 - 2.0 * B(t * h)) * prev_sol[t] +
                    D(t * h) * prev_sol[t + 1] + A(t * h) * prev_sol[t - 1];
    }
    prev_sol = solution;
    time -= k;
  }
}

template <typename fp_type, typename coefficient_holder,
          template <typename, typename> typename container, typename alloc>
void lss_one_dim_space_variable_heat_explicit_schemes_policy::
    heat_euler_scheme_backward_policy<fp_type, coefficient_holder, container,
                                      alloc>::
        traverse(container<fp_type, alloc> &solution,
                 container<fp_type, alloc> const &init_solution,
                 robin_boundary<fp_type> const &robin_boundary,
                 std::pair<fp_type, fp_type> const &deltas,
                 coefficient_holder const &holder, fp_type const &terminal_time,
                 fp_type const &space_start,
                 std::function<fp_type(fp_type, fp_type)> const &source) {
  // get delta time:
  fp_type const k = std::get<0>(deltas);
  // get delta space:
  fp_type const h = std::get<1>(deltas);
  // create last but one time point:
  fp_type time = terminal_time - k;
  // get coefficients:
  auto const &A = std::get<0>(holder);
  auto const &B = std::get<1>(holder);
  auto const &D = std::get<2>(holder);
  // left space boundary:
  fp_type const left_lin = robin_boundary.left.first;
  fp_type const left_const = robin_boundary.left.second;
  // right space boundary:
  fp_type const right_lin_ = robin_boundary.right.first;
  fp_type const right_const_ = robin_boundary.right.second;
  // conversion of right hand boundaries:
  fp_type const right_lin = 1.0 / right_lin_;
  fp_type const right_const = -1.0 * (right_const_ / right_lin_);
  // previous solution:
  container<fp_type, alloc> prev_sol(init_solution);
  // size of the space vector:
  std::size_t const space_size = solution.size();

  // create a container to carry discretized source heat
  container<fp_type, alloc> source_curr(solution.size(), fp_type{});
  discretization<fp_type, container, alloc>::discretize_in_space(
      h, space_start, 0.0, source, source_curr);

  // loop for stepping in time:
  while (time >= 0.0) {
    solution[0] = (D(0) + (A(0) * left_lin)) * prev_sol[1] +
                  (1.0 - 2.0 * B(0)) * prev_sol[0] + A(0) * left_const +
                  k * source_curr[0];
    solution[space_size - 1] =
        (A((space_size - 1) * h) + (D((space_size - 1) * h) * right_lin)) *
            prev_sol[space_size - 2] +
        (1.0 - 2.0 * B((space_size - 1) * h)) * prev_sol[space_size - 1] +
        D((space_size - 1) * h) * right_const + k * source_curr[space_size - 1];
    for (std::size_t t = 1; t < space_size - 1; ++t) {
      solution[t] = (1.0 - 2.0 * B(t * h)) * prev_sol[t] +
                    D(t * h) * prev_sol[t + 1] + A(t * h) * prev_sol[t - 1] +
                    k * source_curr[t];
    }
    discretization<fp_type, container, alloc>::discretize_in_space(
        h, space_start, time, source, source_curr);
    prev_sol = solution;
    time -= k;
  }
}

// ============================================================================
// ============= ade_heat_bakarat_clark_scheme_forward_policy =================
// ============================================================================

template <typename fp_type, typename coefficient_holder,
          template <typename, typename> typename container, typename alloc>
void lss_one_dim_space_variable_heat_explicit_schemes_policy::
    ade_heat_bakarat_clark_scheme_forward_policy<
        fp_type, coefficient_holder, container,
        alloc>::traverse(container<fp_type, alloc> &solution,
                         container<fp_type, alloc> const &init_solution,
                         dirichlet_boundary<fp_type> const &dirichlet_boundary,
                         std::pair<fp_type, fp_type> const &deltas,
                         coefficient_holder const &holder,
                         fp_type const &terminal_time) {
  // get delta time:
  fp_type const k = std::get<0>(deltas);
  // get delta space:
  fp_type const h = std::get<1>(deltas);
  // create first time point:
  fp_type time = k;
  // get coefficients:
  auto const &A = std::get<0>(holder);
  auto const &B = std::get<1>(holder);
  auto const &D = std::get<2>(holder);
  // calculate scheme coefficients:
  auto const &a = [&](fp_type x) { return (A(x) / (1.0 + B(x))); };
  auto const &b = [&](fp_type x) { return ((1.0 - B(x)) / (1.0 + B(x))); };
  auto const &d = [&](fp_type x) { return (D(x) / (1.0 + B(x))); };
  auto const &f = [&](fp_type x) { return (k / (1.0 + B(x))); };
  // left space boundary:
  auto const &left = dirichlet_boundary.first;
  // right space boundary:
  auto const &right = dirichlet_boundary.second;
  // conmponents of the solution:
  container<fp_type, alloc> com_1(init_solution);
  container<fp_type, alloc> com_2(init_solution);
  // size of the space vector:
  std::size_t const space_size = solution.size();
  // create a container to carry discretized source heat
  container<fp_type, alloc> source_curr(space_size, fp_type{});
  container<fp_type, alloc> source_next(space_size, fp_type{});
  // create upsweep anonymous function:
  auto up_sweep = [=](container<fp_type, alloc> &up_component,
                      container<fp_type, alloc> const &rhs, fp_type rhs_coeff) {
    for (std::size_t t = 1; t < space_size - 1; ++t) {
      up_component[t] =
          b(t * h) * up_component[t] + d(t * h) * up_component[t + 1] +
          a(t * h) * up_component[t - 1] + f(t * h) * rhs_coeff * rhs[t];
    }
  };
  // create downsweep anonymous function:
  auto down_sweep = [=](container<fp_type, alloc> &down_component,
                        container<fp_type, alloc> const &rhs,
                        fp_type rhs_coeff) {
    for (std::size_t t = space_size - 2; t >= 1; --t) {
      down_component[t] =
          b(t * h) * down_component[t] + d(t * h) * down_component[t + 1] +
          a(t * h) * down_component[t - 1] + f(t * h) * rhs_coeff * rhs[t];
    }
  };

  while (time <= terminal_time) {
    com_1[0] = com_2[0] = left(time);
    com_1[solution.size() - 1] = com_2[solution.size() - 1] = right(time);
    std::thread up_sweep_tr(std::move(up_sweep), std::ref(com_1), source_curr,
                            0.0);
    std::thread down_sweep_tr(std::move(down_sweep), std::ref(com_2),
                              source_curr, 0.0);
    up_sweep_tr.join();
    down_sweep_tr.join();
    for (std::size_t t = 0; t < space_size; ++t) {
      solution[t] = 0.5 * (com_1[t] + com_2[t]);
    }
    time += k;
  }
}

template <typename fp_type, typename coefficient_holder,
          template <typename, typename> typename container, typename alloc>
void lss_one_dim_space_variable_heat_explicit_schemes_policy::
    ade_heat_bakarat_clark_scheme_forward_policy<fp_type, coefficient_holder,
                                                 container, alloc>::
        traverse(container<fp_type, alloc> &solution,
                 container<fp_type, alloc> const &init_solution,
                 dirichlet_boundary<fp_type> const &dirichlet_boundary,
                 std::pair<fp_type, fp_type> const &deltas,
                 coefficient_holder const &holder, fp_type const &terminal_time,
                 fp_type const &space_start,
                 std::function<fp_type(fp_type, fp_type)> const &source) {
  // get delta time:
  fp_type const k = std::get<0>(deltas);
  // get delta space:
  fp_type const h = std::get<1>(deltas);
  // create first time point:
  fp_type time = k;
  // get coefficients:
  auto const &A = std::get<0>(holder);
  auto const &B = std::get<1>(holder);
  auto const &D = std::get<2>(holder);
  // calculate scheme coefficients:
  auto const &a = [&](fp_type x) { return (A(x) / (1.0 + B(x))); };
  auto const &b = [&](fp_type x) { return ((1.0 - B(x)) / (1.0 + B(x))); };
  auto const &d = [&](fp_type x) { return (D(x) / (1.0 + B(x))); };
  auto const &f = [&](fp_type x) { return (k / (1.0 + B(x))); };
  // left space boundary:
  auto const &left = dirichlet_boundary.first;
  // right space boundary:
  auto const &right = dirichlet_boundary.second;
  // conmponents of the solution:
  std::vector<fp_type> com_1(init_solution);
  std::vector<fp_type> com_2(init_solution);
  // size of the space vector:
  std::size_t const space_size = solution.size();
  // create a container to carry discretized source heat
  std::vector<fp_type> source_curr(space_size, fp_type{});
  std::vector<fp_type> source_next(space_size, fp_type{});
  // create upsweep anonymous function:
  auto up_sweep = [=](std::vector<fp_type> &up_component,
                      std::vector<fp_type> const &rhs, fp_type rhs_coeff) {
    for (std::size_t t = 1; t < space_size - 1; ++t) {
      up_component[t] =
          b(t * h) * up_component[t] + d(t * h) * up_component[t + 1] +
          a(t * h) * up_component[t - 1] + f(t * h) * rhs_coeff * rhs[t];
    }
  };
  // create downsweep anonymous function:
  auto down_sweep = [=](std::vector<fp_type> &down_component,
                        std::vector<fp_type> const &rhs, fp_type rhs_coeff) {
    for (std::size_t t = space_size - 2; t >= 1; --t) {
      down_component[t] =
          b(t * h) * down_component[t] + d(t * h) * down_component[t + 1] +
          a(t * h) * down_component[t - 1] + f(t * h) * rhs_coeff * rhs[t];
    }
  };

  discretization<fp_type, container, alloc>::discretize_in_space(
      h, space_start, 0.0, source, source_curr);
  discretization<fp_type, container, alloc>::discretize_in_space(
      h, space_start, time, source, source_next);
  // loop for stepping in time:
  while (time <= terminal_time) {
    com_1[0] = com_2[0] = left(time);
    com_1[solution.size() - 1] = com_2[solution.size() - 1] = right(time);
    std::thread up_sweep_tr(std::move(up_sweep), std::ref(com_1), source_next,
                            1.0);
    std::thread down_sweep_tr(std::move(down_sweep), std::ref(com_2),
                              source_curr, 1.0);
    up_sweep_tr.join();
    down_sweep_tr.join();
    for (std::size_t t = 0; t < space_size; ++t) {
      solution[t] = 0.5 * (com_1[t] + com_2[t]);
    }
    discretization<fp_type, container, alloc>::discretize_in_space(
        h, space_start, time, source, source_curr);
    discretization<fp_type, container, alloc>::discretize_in_space(
        h, space_start, 2.0 * time, source, source_next);
    time += k;
  }
}

// ============================================================================
// ============= ade_heat_bakarat_clark_scheme_backward_policy ================
// ============================================================================

template <typename fp_type, typename coefficient_holder,
          template <typename, typename> typename container, typename alloc>
void lss_one_dim_space_variable_heat_explicit_schemes_policy::
    ade_heat_bakarat_clark_scheme_backward_policy<
        fp_type, coefficient_holder, container,
        alloc>::traverse(container<fp_type, alloc> &solution,
                         container<fp_type, alloc> const &init_solution,
                         dirichlet_boundary<fp_type> const &dirichlet_boundary,
                         std::pair<fp_type, fp_type> const &deltas,
                         coefficient_holder const &holder,
                         fp_type const &terminal_time) {
  // get delta time:
  fp_type const k = std::get<0>(deltas);
  // get delta space:
  fp_type const h = std::get<1>(deltas);
  // create last but one time point:
  fp_type time = terminal_time - k;
  // get coefficients:
  auto const &A = std::get<0>(holder);
  auto const &B = std::get<1>(holder);
  auto const &D = std::get<2>(holder);
  // calculate scheme coefficients:
  auto const &a = [&](fp_type x) { return (A(x) / (1.0 + B(x))); };
  auto const &b = [&](fp_type x) { return ((1.0 - B(x)) / (1.0 + B(x))); };
  auto const &d = [&](fp_type x) { return (D(x) / (1.0 + B(x))); };
  auto const &f = [&](fp_type x) { return (k / (1.0 + B(x))); };
  // left space boundary:
  auto const &left = dirichlet_boundary.first;
  // right space boundary:
  auto const &right = dirichlet_boundary.second;
  // conmponents of the solution:
  container<fp_type, alloc> com_1(init_solution);
  container<fp_type, alloc> com_2(init_solution);
  // size of the space vector:
  std::size_t const space_size = solution.size();
  // create a container to carry discretized source heat
  container<fp_type, alloc> source_curr(space_size, fp_type{});
  container<fp_type, alloc> source_next(space_size, fp_type{});
  // create upsweep anonymous function:
  auto up_sweep = [=](container<fp_type, alloc> &up_component,
                      container<fp_type, alloc> const &rhs, fp_type rhs_coeff) {
    for (std::size_t t = 1; t < space_size - 1; ++t) {
      up_component[t] =
          b(t * h) * up_component[t] + d(t * h) * up_component[t + 1] +
          a(t * h) * up_component[t - 1] + f(t * h) * rhs_coeff * rhs[t];
    }
  };
  // create downsweep anonymous function:
  auto down_sweep = [=](container<fp_type, alloc> &down_component,
                        container<fp_type, alloc> const &rhs,
                        fp_type rhs_coeff) {
    for (std::size_t t = space_size - 2; t >= 1; --t) {
      down_component[t] =
          b(t * h) * down_component[t] + d(t * h) * down_component[t + 1] +
          a(t * h) * down_component[t - 1] + f(t * h) * rhs_coeff * rhs[t];
    }
  };

  while (time >= 0.0) {
    com_1[0] = com_2[0] = left(time);
    com_1[solution.size() - 1] = com_2[solution.size() - 1] = right(time);
    std::thread up_sweep_tr(std::move(up_sweep), std::ref(com_1), source_curr,
                            0.0);
    std::thread down_sweep_tr(std::move(down_sweep), std::ref(com_2),
                              source_curr, 0.0);
    up_sweep_tr.join();
    down_sweep_tr.join();
    for (std::size_t t = 0; t < space_size; ++t) {
      solution[t] = 0.5 * (com_1[t] + com_2[t]);
    }
    time -= k;
  }
}

template <typename fp_type, typename coefficient_holder,
          template <typename, typename> typename container, typename alloc>
void lss_one_dim_space_variable_heat_explicit_schemes_policy::
    ade_heat_bakarat_clark_scheme_backward_policy<fp_type, coefficient_holder,
                                                  container, alloc>::
        traverse(container<fp_type, alloc> &solution,
                 container<fp_type, alloc> const &init_solution,
                 dirichlet_boundary<fp_type> const &dirichlet_boundary,
                 std::pair<fp_type, fp_type> const &deltas,
                 coefficient_holder const &holder, fp_type const &terminal_time,
                 fp_type const &space_start,
                 std::function<fp_type(fp_type, fp_type)> const &source) {
  // get delta time:
  fp_type const k = std::get<0>(deltas);
  // get delta space:
  fp_type const h = std::get<1>(deltas);
  // create last but one time point:
  fp_type time = terminal_time - k;
  // get coefficients:
  auto const &A = std::get<0>(holder);
  auto const &B = std::get<1>(holder);
  auto const &D = std::get<2>(holder);
  // calculate scheme coefficients:
  auto const &a = [&](fp_type x) { return (A(x) / (1.0 + B(x))); };
  auto const &b = [&](fp_type x) { return ((1.0 - B(x)) / (1.0 + B(x))); };
  auto const &d = [&](fp_type x) { return (D(x) / (1.0 + B(x))); };
  auto const &f = [&](fp_type x) { return (k / (1.0 + B(x))); };
  // left space boundary:
  auto const &left = dirichlet_boundary.first;
  // right space boundary:
  auto const &right = dirichlet_boundary.second;
  // conmponents of the solution:
  std::vector<fp_type> com_1(init_solution);
  std::vector<fp_type> com_2(init_solution);
  // size of the space vector:
  std::size_t const space_size = solution.size();
  // create a container to carry discretized source heat
  std::vector<fp_type> source_curr(space_size, fp_type{});
  std::vector<fp_type> source_next(space_size, fp_type{});
  // create upsweep anonymous function:
  auto up_sweep = [=](std::vector<fp_type> &up_component,
                      std::vector<fp_type> const &rhs, fp_type rhs_coeff) {
    for (std::size_t t = 1; t < space_size - 1; ++t) {
      up_component[t] =
          b(t * h) * up_component[t] + d(t * h) * up_component[t + 1] +
          a(t * h) * up_component[t - 1] + f(t * h) * rhs_coeff * rhs[t];
    }
  };
  // create downsweep anonymous function:
  auto down_sweep = [=](std::vector<fp_type> &down_component,
                        std::vector<fp_type> const &rhs, fp_type rhs_coeff) {
    for (std::size_t t = space_size - 2; t >= 1; --t) {
      down_component[t] =
          b(t * h) * down_component[t] + d(t * h) * down_component[t + 1] +
          a(t * h) * down_component[t - 1] + f(t * h) * rhs_coeff * rhs[t];
    }
  };

  discretization<fp_type, container, alloc>::discretize_in_space(
      h, space_start, 0.0, source, source_curr);
  discretization<fp_type, container, alloc>::discretize_in_space(
      h, space_start, time, source, source_next);
  // loop for stepping in time:
  while (time >= 0.0) {
    com_1[0] = com_2[0] = left(time);
    com_1[solution.size() - 1] = com_2[solution.size() - 1] = right(time);
    std::thread up_sweep_tr(std::move(up_sweep), std::ref(com_1), source_next,
                            1.0);
    std::thread down_sweep_tr(std::move(down_sweep), std::ref(com_2),
                              source_curr, 1.0);
    up_sweep_tr.join();
    down_sweep_tr.join();
    for (std::size_t t = 0; t < space_size; ++t) {
      solution[t] = 0.5 * (com_1[t] + com_2[t]);
    }
    discretization<fp_type, container, alloc>::discretize_in_space(
        h, space_start, time, source, source_curr);
    discretization<fp_type, container, alloc>::discretize_in_space(
        h, space_start, 2.0 * time, source, source_next);
    time -= k;
  }
}

// ============================================================================
// ================== ade_heat_saulyev_scheme_forward_policy ==================
// ============================================================================

template <typename fp_type, typename coefficient_holder,
          template <typename, typename> typename container, typename alloc>
void lss_one_dim_space_variable_heat_explicit_schemes_policy::
    ade_heat_saulyev_scheme_forward_policy<
        fp_type, coefficient_holder, container,
        alloc>::traverse(container<fp_type, alloc> &solution,
                         container<fp_type, alloc> const &init_solution,
                         dirichlet_boundary<fp_type> const &dirichlet_boundary,
                         std::pair<fp_type, fp_type> const &deltas,
                         coefficient_holder const &holder,
                         fp_type const &terminal_time) {
  // get delta time:
  fp_type const k = std::get<0>(deltas);
  // get delta space:
  fp_type const h = std::get<1>(deltas);
  // create first time point:
  fp_type time = k;
  // get coefficients:
  auto const &A = std::get<0>(holder);
  auto const &B = std::get<1>(holder);
  auto const &D = std::get<2>(holder);
  // calculate scheme coefficients:
  auto const &a = [&](fp_type x) { return (A(x) / (1.0 + B(x))); };
  auto const &b = [&](fp_type x) { return ((1.0 - B(x)) / (1.0 + B(x))); };
  auto const &d = [&](fp_type x) { return (D(x) / (1.0 + B(x))); };
  auto const &f = [&](fp_type x) { return (k / (1.0 + B(x))); };
  // left space boundary:
  auto const &left = dirichlet_boundary.first;
  // right space boundary:
  auto const &right = dirichlet_boundary.second;
  // get the initial condition :
  solution = init_solution;
  // size of the space vector:
  std::size_t const space_size = solution.size();
  // create a container to carry discretized source heat
  container<fp_type, alloc> source_curr(space_size, fp_type{});
  container<fp_type, alloc> source_next(space_size, fp_type{});
  // create upsweep anonymous function:
  auto up_sweep = [=](container<fp_type, alloc> &up_component,
                      container<fp_type, alloc> const &rhs, fp_type rhs_coeff) {
    for (std::size_t t = 1; t < space_size - 1; ++t) {
      up_component[t] =
          b(t * h) * up_component[t] + d(t * h) * up_component[t + 1] +
          a(t * h) * up_component[t - 1] + f(t * h) * rhs_coeff * rhs[t];
    }
  };
  // create downsweep anonymous function:
  auto down_sweep = [=](container<fp_type, alloc> &down_component,
                        container<fp_type, alloc> const &rhs,
                        fp_type rhs_coeff) {
    for (std::size_t t = space_size - 2; t >= 1; --t) {
      down_component[t] =
          b(t * h) * down_component[t] + d(t * h) * down_component[t + 1] +
          a(t * h) * down_component[t - 1] + f(t * h) * rhs_coeff * rhs[t];
    }
  };

  std::size_t t = 1;
  while (time <= terminal_time) {
    solution[0] = left(time);
    solution[solution.size() - 1] = right(time);
    if (t % 2 == 0)
      down_sweep(solution, source_curr, 0.0);
    else
      up_sweep(solution, source_curr, 0.0);
    ++t;
    time += k;
  }
}

template <typename fp_type, typename coefficient_holder,
          template <typename, typename> typename container, typename alloc>
void lss_one_dim_space_variable_heat_explicit_schemes_policy::
    ade_heat_saulyev_scheme_forward_policy<fp_type, coefficient_holder,
                                           container, alloc>::
        traverse(container<fp_type, alloc> &solution,
                 container<fp_type, alloc> const &init_solution,
                 dirichlet_boundary<fp_type> const &dirichlet_boundary,
                 std::pair<fp_type, fp_type> const &deltas,
                 coefficient_holder const &holder, fp_type const &terminal_time,
                 fp_type const &space_start,
                 std::function<fp_type(fp_type, fp_type)> const &source) {
  // get delta time:
  fp_type const k = std::get<0>(deltas);
  // get delta space:
  fp_type const h = std::get<1>(deltas);
  // create first time point:
  fp_type time = k;
  // get coefficients:
  auto const &A = std::get<0>(holder);
  auto const &B = std::get<1>(holder);
  auto const &D = std::get<2>(holder);
  // calculate scheme coefficients:
  auto const &a = [&](fp_type x) { return (A(x) / (1.0 + B(x))); };
  auto const &b = [&](fp_type x) { return ((1.0 - B(x)) / (1.0 + B(x))); };
  auto const &d = [&](fp_type x) { return (D(x) / (1.0 + B(x))); };
  auto const &f = [&](fp_type x) { return (k / (1.0 + B(x))); };
  // left space boundary:
  auto const &left = dirichlet_boundary.first;
  // right space boundary:
  auto const &right = dirichlet_boundary.second;
  // get the initial condition :
  solution = init_solution;
  // size of the space vector:
  std::size_t const space_size = solution.size();
  // create a container to carry discretized source heat
  std::vector<fp_type> source_curr(space_size, fp_type{});
  std::vector<fp_type> source_next(space_size, fp_type{});
  // create upsweep anonymous function:
  auto up_sweep = [=](std::vector<fp_type> &up_component,
                      std::vector<fp_type> const &rhs, fp_type rhs_coeff) {
    for (std::size_t t = 1; t < space_size - 1; ++t) {
      up_component[t] =
          b(t * h) * up_component[t] + d(t * h) * up_component[t + 1] +
          a(t * h) * up_component[t - 1] + f(t * h) * rhs_coeff * rhs[t];
    }
  };
  // create downsweep anonymous function:
  auto down_sweep = [=](std::vector<fp_type> &down_component,
                        std::vector<fp_type> const &rhs, fp_type rhs_coeff) {
    for (std::size_t t = space_size - 2; t >= 1; --t) {
      down_component[t] =
          b(t * h) * down_component[t] + d(t * h) * down_component[t + 1] +
          a(t * h) * down_component[t - 1] + f(t * h) * rhs_coeff * rhs[t];
    }
  };

  discretization<fp_type, container, alloc>::discretize_in_space(
      h, space_start, 0.0, source, source_curr);
  discretization<fp_type, container, alloc>::discretize_in_space(
      h, space_start, time, source, source_next);
  // loop for stepping in time:
  std::size_t t = 1;
  while (time <= terminal_time) {
    solution[0] = left(time);
    solution[solution.size() - 1] = right(time);
    if (t % 2 == 0)
      down_sweep(solution, source_curr, 1.0);
    else
      up_sweep(solution, source_next, 1.0);
    ++t;
    discretization<fp_type, container, alloc>::discretize_in_space(
        h, space_start, time, source, source_curr);
    discretization<fp_type, container, alloc>::discretize_in_space(
        h, space_start, 2.0 * time, source, source_next);
    time += k;
  }
}

// ============================================================================
// ================== ade_heat_saulyev_scheme_backward_policy =================
// ============================================================================

template <typename fp_type, typename coefficient_holder,
          template <typename, typename> typename container, typename alloc>
void lss_one_dim_space_variable_heat_explicit_schemes_policy::
    ade_heat_saulyev_scheme_backward_policy<
        fp_type, coefficient_holder, container,
        alloc>::traverse(container<fp_type, alloc> &solution,
                         container<fp_type, alloc> const &init_solution,
                         dirichlet_boundary<fp_type> const &dirichlet_boundary,
                         std::pair<fp_type, fp_type> const &deltas,
                         coefficient_holder const &holder,
                         fp_type const &terminal_time) {
  // get delta time:
  fp_type const k = std::get<0>(deltas);
  // get delta space:
  fp_type const h = std::get<1>(deltas);
  // create last but one time point:
  fp_type time = terminal_time - k;
  // get coefficients:
  auto const &A = std::get<0>(holder);
  auto const &B = std::get<1>(holder);
  auto const &D = std::get<2>(holder);
  // calculate scheme coefficients:
  auto const &a = [&](fp_type x) { return (A(x) / (1.0 + B(x))); };
  auto const &b = [&](fp_type x) { return ((1.0 - B(x)) / (1.0 + B(x))); };
  auto const &d = [&](fp_type x) { return (D(x) / (1.0 + B(x))); };
  auto const &f = [&](fp_type x) { return (k / (1.0 + B(x))); };
  // left space boundary:
  auto const &left = dirichlet_boundary.first;
  // right space boundary:
  auto const &right = dirichlet_boundary.second;
  // get the initial condition :
  solution = init_solution;
  // size of the space vector:
  std::size_t const space_size = solution.size();
  // create a container to carry discretized source heat
  container<fp_type, alloc> source_curr(space_size, fp_type{});
  container<fp_type, alloc> source_next(space_size, fp_type{});
  // create upsweep anonymous function:
  auto up_sweep = [=](container<fp_type, alloc> &up_component,
                      container<fp_type, alloc> const &rhs, fp_type rhs_coeff) {
    for (std::size_t t = 1; t < space_size - 1; ++t) {
      up_component[t] =
          b(t * h) * up_component[t] + d(t * h) * up_component[t + 1] +
          a(t * h) * up_component[t - 1] + f(t * h) * rhs_coeff * rhs[t];
    }
  };
  // create downsweep anonymous function:
  auto down_sweep = [=](container<fp_type, alloc> &down_component,
                        container<fp_type, alloc> const &rhs,
                        fp_type rhs_coeff) {
    for (std::size_t t = space_size - 2; t >= 1; --t) {
      down_component[t] =
          b(t * h) * down_component[t] + d(t * h) * down_component[t + 1] +
          a(t * h) * down_component[t - 1] + f(t * h) * rhs_coeff * rhs[t];
    }
  };

  std::size_t t = 1;
  while (time >= 0.0) {
    solution[0] = left(time);
    solution[solution.size() - 1] = right(time);
    if (t % 2 == 0)
      down_sweep(solution, source_curr, 0.0);
    else
      up_sweep(solution, source_curr, 0.0);
    ++t;
    time -= k;
  }
}

template <typename fp_type, typename coefficient_holder,
          template <typename, typename> typename container, typename alloc>
void lss_one_dim_space_variable_heat_explicit_schemes_policy::
    ade_heat_saulyev_scheme_backward_policy<fp_type, coefficient_holder,
                                            container, alloc>::
        traverse(container<fp_type, alloc> &solution,
                 container<fp_type, alloc> const &init_solution,
                 dirichlet_boundary<fp_type> const &dirichlet_boundary,
                 std::pair<fp_type, fp_type> const &deltas,
                 coefficient_holder const &holder, fp_type const &terminal_time,
                 fp_type const &space_start,
                 std::function<fp_type(fp_type, fp_type)> const &source) {
  // get delta time:
  fp_type const k = std::get<0>(deltas);
  // get delta space:
  fp_type const h = std::get<1>(deltas);
  // create last but one time point:
  fp_type time = terminal_time - k;
  // get coefficients:
  auto const &A = std::get<0>(holder);
  auto const &B = std::get<1>(holder);
  auto const &D = std::get<2>(holder);
  // calculate scheme coefficients:
  auto const &a = [&](fp_type x) { return (A(x) / (1.0 + B(x))); };
  auto const &b = [&](fp_type x) { return ((1.0 - B(x)) / (1.0 + B(x))); };
  auto const &d = [&](fp_type x) { return (D(x) / (1.0 + B(x))); };
  auto const &f = [&](fp_type x) { return (k / (1.0 + B(x))); };
  // left space boundary:
  auto const &left = dirichlet_boundary.first;
  // right space boundary:
  auto const &right = dirichlet_boundary.second;
  // get the initial condition :
  solution = init_solution;
  // size of the space vector:
  std::size_t const space_size = solution.size();
  // create a container to carry discretized source heat
  std::vector<fp_type> source_curr(space_size, fp_type{});
  std::vector<fp_type> source_next(space_size, fp_type{});
  // create upsweep anonymous function:
  auto up_sweep = [=](std::vector<fp_type> &up_component,
                      std::vector<fp_type> const &rhs, fp_type rhs_coeff) {
    for (std::size_t t = 1; t < space_size - 1; ++t) {
      up_component[t] =
          b(t * h) * up_component[t] + d(t * h) * up_component[t + 1] +
          a(t * h) * up_component[t - 1] + f(t * h) * rhs_coeff * rhs[t];
    }
  };
  // create downsweep anonymous function:
  auto down_sweep = [=](std::vector<fp_type> &down_component,
                        std::vector<fp_type> const &rhs, fp_type rhs_coeff) {
    for (std::size_t t = space_size - 2; t >= 1; --t) {
      down_component[t] =
          b(t * h) * down_component[t] + d(t * h) * down_component[t + 1] +
          a(t * h) * down_component[t - 1] + f(t * h) * rhs_coeff * rhs[t];
    }
  };

  discretization<fp_type, container, alloc>::discretize_in_space(
      h, space_start, 0.0, source, source_curr);
  discretization<fp_type, container, alloc>::discretize_in_space(
      h, space_start, time, source, source_next);
  // loop for stepping in time:
  std::size_t t = 1;
  while (time >= 0.0) {
    solution[0] = left(time);
    solution[solution.size() - 1] = right(time);
    if (t % 2 == 0)
      down_sweep(solution, source_curr, 1.0);
    else
      up_sweep(solution, source_next, 1.0);
    ++t;
    discretization<fp_type, container, alloc>::discretize_in_space(
        h, space_start, time, source, source_curr);
    discretization<fp_type, container, alloc>::discretize_in_space(
        h, space_start, 2.0 * time, source, source_next);
    time -= k;
  }
}

#endif  //_LSS_1D_SPACE_VARIABLE_HEAT_EXPLICIT_SCHEMES_POLICY
