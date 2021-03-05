#pragma once
#if !defined(_LSS_1D_SPACE_VARIABLE_HEAT_EXPLICIT_SCHEMES_CUDA_POLICY)
#define _LSS_1D_SPACE_VARIABLE_HEAT_EXPLICIT_SCHEMES_CUDA_POLICY

#pragma warning(disable : 4244)

#include <thread>

#include "common/lss_enumerations.h"
#include "pde_solvers/one_dim/lss_base_explicit_schemes.h"
#include "pde_solvers/one_dim/lss_pde_utility.h"

namespace lss_one_dim_space_variable_heat_explicit_schemes_cuda_policy {

using lss_enumerations::boundary_condition_enum;
using lss_one_dim_base_explicit_schemes::heat_scheme_base;
using lss_one_dim_pde_utility::dirichlet_boundary;
using lss_one_dim_pde_utility::discretization;
using lss_one_dim_pde_utility::pde_coefficient_holder_fun_1_arg;
using lss_one_dim_pde_utility::robin_boundary;

// ============================================================================
// ========================= heat_euler_scheme_policy =========================
// ============================================================================

template <typename fp_type, typename coefficient_holder>
struct heat_euler_scheme_forward_policy {
  // Dirichlet boundary without source:
  static void traverse(fp_type *solution, fp_type const *init_solution,
                       unsigned long long size,
                       dirichlet_boundary<fp_type> const &dirichlet_boundary,
                       std::pair<fp_type, fp_type> const &deltas,
                       coefficient_holder const &holder,
                       fp_type const &terminal_time,
                       fp_type const &space_start);
  // Dirichlet boundary with source:
  static void traverse(fp_type *solution, fp_type const *init_solution,
                       unsigned long long size,
                       dirichlet_boundary<fp_type> const &dirichlet_boundary,
                       std::pair<fp_type, fp_type> const &deltas,
                       coefficient_holder const &holder,
                       fp_type const &terminal_time, fp_type const &space_start,
                       std::function<fp_type(fp_type, fp_type)> const &source);
  // Robin boundary without source:
  static void traverse(fp_type *solution, fp_type const *init_solution,
                       unsigned long long size,
                       robin_boundary<fp_type> const &robin_boundary,
                       std::pair<fp_type, fp_type> const &deltas,
                       coefficient_holder const &holder,
                       fp_type const &terminal_time,
                       fp_type const &space_start);
  // Robin boundary with source:
  static void traverse(fp_type *solution, fp_type const *init_solution,
                       unsigned long long size,
                       robin_boundary<fp_type> const &robin_boundary,
                       std::pair<fp_type, fp_type> const &deltas,
                       coefficient_holder const &holder,
                       fp_type const &terminal_time, fp_type const &space_start,
                       std::function<fp_type(fp_type, fp_type)> const &source);
};

template <typename fp_type, typename coefficient_holder>
struct heat_euler_scheme_backward_policy {
  // Dirichlet boundary without source:
  static void traverse(fp_type *solution, fp_type const *init_solution,
                       unsigned long long size,
                       dirichlet_boundary<fp_type> const &dirichlet_boundary,
                       std::pair<fp_type, fp_type> const &deltas,
                       coefficient_holder const &holder,
                       fp_type const &terminal_time,
                       fp_type const &space_start);
  // Dirichlet boundary with source:
  static void traverse(fp_type *solution, fp_type const *init_solution,
                       unsigned long long size,
                       dirichlet_boundary<fp_type> const &dirichlet_boundary,
                       std::pair<fp_type, fp_type> const &deltas,
                       coefficient_holder const &holder,
                       fp_type const &terminal_time, fp_type const &space_start,
                       std::function<fp_type(fp_type, fp_type)> const &source);
  // Robin boundary without source:
  static void traverse(fp_type *solution, fp_type const *init_solution,
                       unsigned long long size,
                       robin_boundary<fp_type> const &robin_boundary,
                       std::pair<fp_type, fp_type> const &deltas,
                       coefficient_holder const &holder,
                       fp_type const &terminal_time,
                       fp_type const &space_start);
  // Robin boundary with source:
  static void traverse(fp_type *solution, fp_type const *init_solution,
                       unsigned long long size,
                       robin_boundary<fp_type> const &robin_boundary,
                       std::pair<fp_type, fp_type> const &deltas,
                       coefficient_holder const &holder,
                       fp_type const &terminal_time, fp_type const &space_start,
                       std::function<fp_type(fp_type, fp_type)> const &source);
};

}  // namespace lss_one_dim_space_variable_heat_explicit_schemes_cuda_policy

#endif  //_LSS_1D_SPACE_VARIABLE_HEAT_EXPLICIT_SCHEMES_CUDA_POLICY
