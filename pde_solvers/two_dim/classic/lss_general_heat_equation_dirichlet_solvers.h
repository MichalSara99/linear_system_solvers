#pragma once
#if !defined(_LSS_2D_GENERAL_HEAT_EQUATION_DIRICHLET_SOLVERS)
#define _LSS_2D_GENERAL_HEAT_EQUATION_DIRICHLET_SOLVERS

#include <functional>

#include "common/lss_enumerations.h"
#include "common/lss_macros.h"
#include "common/lss_utility.h"
//#include "lss_heat_explicit_schemes.h"
#include "lss_general_heat_equation_solvers_base.h"
#include "lss_heat_implicit_schemes.h"
#include "pde_solvers/one_dim/lss_pde_utility.h"
#include "pde_solvers/two_dim/lss_pde_utility.h"

namespace lss_two_dim_classic_pde_solvers {

using lss_enumerations::boundary_condition_enum;
using lss_enumerations::explicit_pde_schemes_enum;
using lss_enumerations::implicit_pde_schemes_enum;
using lss_one_dim_pde_utility::discretization;
using lss_two_dim_pde_utility::discretization_2d;
using lss_two_dim_pde_utility::heat_data_2d;
using lss_two_dim_pde_utility::pde_coefficient_holder_const;
// using lss_one_dim_pde_utility::robin_boundary;
using lss_two_dim_pde_utility::dirichlet_boundary_2d;
using lss_utility::container_2d;
using lss_utility::copy;
using lss_utility::range;
using lss_utility::sptr_t;
using lss_utility::uptr_t;

namespace implicit_solvers {

// ============================================================================
// ======= general_heat_equation Dirichlet Specialisation Template ============
// ============================================================================

/*!
   ============================================================================
        Represents solver for general Dirichlet 2D heat equation

        u_t = a*u_xx + b*u_yy + c*u_xy + d*u_x + e*u_y + f*u + F(x,y,t),
        t > 0, x_1 < x < x_2, y_1 < y < y_2

        with initial condition:

        u(x,y,0) = f(x,y)

        and Dirichlet boundaries:

        u(x_1,y,t) = A_1(y,t)
        u(x_2,y,t) = A_2(y,t)
        u(x,y_1,t) = B_1(x,t)
        u(x,y_2,t) = B_2(x,t)

   ============================================================================
 */
template <typename fp_type,
          template <typename, boundary_condition_enum,
                    template <typename, typename> typename cont, typename>
          typename fdm_solver,
          template <typename, typename> typename container, typename alloc>
class general_heat_equation<fp_type, boundary_condition_enum::Dirichlet,
                            fdm_solver, container, alloc> {
 private:
  typedef fdm_solver<fp_type, boundary_condition_enum::Dirichlet, container,
                     alloc>
      fdm_solver_t;
  typedef heat_data_2d<fp_type> heat_data_2d_t;

  // unique ptrs to solvers may be dropped if threads are used
  uptr_t<fdm_solver_t> solver_fst_ptr_;  // first finite-difference solver
  uptr_t<fdm_solver_t> solver_sec_ptr_;  // second finite-difference solver
  uptr_t<heat_data_2d_t> dataPtr_;       // heat data
  sptr_t<dirichlet_boundary_2d<fp_type>> boundary_;  // boundaries
  pde_coefficient_holder_const<fp_type> coeffs_;  // coefficients a, b, c in PDE

 public:
  typedef fp_type value_type;
  explicit general_heat_equation() = delete;

  /*!
    Constructor for general Dirichlet 2D heat equation

   \param space_range
   \param terminal_time
   \param space_discretization
   \param time_discretization
   */
  explicit general_heat_equation(
      std::pair<range<fp_type>, range<fp_type>> const &space_range,
      fp_type terminal_time,
      std::pair<std::size_t, std::size_t> const &space_discretization,
      std::size_t const &time_discretization)
      : solver_fst_ptr_{std::make_unique<fdm_solver_t>(
            space_discretization.first + 1)},
        solver_sec_ptr_{
            std::make_unique<fdm_solver_t>(space_discretization.second + 1)},
        dataPtr_{std::make_unique<heat_data_2d_t>(
            space_range, range<fp_type>(fp_type{}, terminal_time),
            space_discretization, time_discretization, nullptr, nullptr,
            nullptr, false)} {}

  ~general_heat_equation() {}

  general_heat_equation(general_heat_equation const &) = delete;
  general_heat_equation(general_heat_equation &&) = delete;
  general_heat_equation &operator=(general_heat_equation const &) = delete;
  general_heat_equation &operator=(general_heat_equation &&) = delete;

  /*!
  Represents spacial steps

  \returns Pair of spacial steps
  */
  inline std::pair<fp_type, fp_type> space_step() const {
    return std::make_pair<fp_type, fp_type>(
        (dataPtr_->space_range.first.spread()) /
            static_cast<fp_type>(dataPtr_->space_division.first),
        (dataPtr_->space_range.second.spread()) /
            static_cast<fp_type>(dataPtr_->space_division.second));
  }

  /*!
  Represents temporal step

  \returns Time step
  */
  inline fp_type time_step() const {
    return ((dataPtr_->time_range.upper()) /
            static_cast<fp_type>(dataPtr_->time_division));
  }

  /*!
  Sets Dirichlet boundaries
  first_pair: (u(x_1,y,t) = A_1(y,t),u(x_2,y,t) = A_2(y,t))
  second_pair: (u(x,y_1,t) = B_1(x,t),u(x,y_2,t) = B_2(x,t))

  \param dirichlet_boundary
  */
  inline void set_boundary_condition(
      sptr_t<dirichlet_boundary_2d<fp_type>> const &dirichlet_boundary) {
    boundary_ = dirichlet_boundary;
  }

  /*!
  Sets initial condition:
  u(x,y,0) = f(x,y)

  \param initial_condition
  */
  inline void set_initial_condition(
      std::function<fp_type(fp_type, fp_type)> const &initial_condition) {
    dataPtr_->initial_condition = initial_condition;
  }

  /*!
  Sets heat source: F(x,y,t)

  \param heat_source
  */
  inline void set_heat_source(
      std::function<fp_type(fp_type, fp_type, fp_type)> const &heat_source) {
    dataPtr_->is_source_function_set = true;
    dataPtr_->source_function = heat_source;
  }

  /*!
  Sets pair of 2.nd order coefficients: a,b

  \param values
  */
  inline void set_2_order_coefficients(
      std::pair<fp_type, fp_type> const &values) {
    std::get<0>(coeffs_) = values.first;   // a
    std::get<1>(coeffs_) = values.second;  // b
  }

  /*!
  Sets mixed order coefficient: c

  \param value
  */
  inline void set_mixed_order_coefficient(fp_type value) {
    std::get<2>(coeffs_) = value;  // c
  }

  /*!
  Sets pair of 1.st order coefficients: d,e

  \param values
  */
  inline void set_1_order_coefficients(
      std::pair<fp_type, fp_type> const &values) {
    std::get<3>(coeffs_) = values.first;   // d
    std::get<4>(coeffs_) = values.second;  // e
  }

  /*!
  Sets pair of 0th order coefficient: f

  \param value
  */
  inline void set_0_order_coefficient(fp_type value) {
    std::get<5>(coeffs_) = value;  // f
  }

  /*!
    Solves the corresponding equation
    \param solution
    \param scheme
   */
  void solve(container_2d<container, fp_type, alloc> &solution,
             implicit_pde_schemes_enum scheme =
                 implicit_pde_schemes_enum::CrankNicolson);
};
}  // namespace implicit_solvers

// ========================= IMPLEMENTATIONS ==================================

// ============================================================================
// ============== general_heat_equation (Dirichlet) implementation ============
// ============================================================================

template <typename fp_type,
          template <typename, boundary_condition_enum,
                    template <typename, typename> typename cont, typename>
          typename fdm_solver,
          template <typename, typename> typename container, typename alloc>
void implicit_solvers::general_heat_equation<
    fp_type, boundary_condition_enum::Dirichlet, fdm_solver, container,
    alloc>::solve(container_2d<container, fp_type, alloc> &solution,
                  implicit_pde_schemes_enum scheme) {
  LSS_VERIFY(boundary_, "Dirichlet Boundary must not be null");
  LSS_VERIFY(dataPtr_->initial_condition, "Initial condition must be set.");
  LSS_ASSERT(solution.rows() > 0,
             "The input solution container must be initialized.");
  LSS_ASSERT(solution.columns() > 0,
             "The input solution container must be initialized.");
  // convinient typedefs:
  typedef container_2d<container, fp_type, alloc> matrix_t;
  typedef container<fp_type, alloc> vector_t;
  typedef discretization<fp_type, container, alloc> d1d_t;
  typedef discretization_2d<fp_type, container, alloc> d2d_t;

  // get correct theta according to the scheme:
  fp_type const theta =
      lss_two_dim_heat_implicit_schemes::heat_equation_schemes<
          fp_type, container, alloc>::get_theta(scheme);
  // get space steps:
  auto const &h = space_step();
  fp_type const h_1 = h.first;   // x step
  fp_type const h_2 = h.second;  // y step
  // get time step:
  fp_type const k = time_step();
  // get space ranges:
  auto const &space_range = dataPtr_->space_range;
  // get source heat function:
  auto const &heat_source = dataPtr_->source_function;
  // space divisions:
  auto const &space_divison = dataPtr_->space_division;
  // dirichlet boundary functions for x axis:
  // u(a,y,t) = F_1(y,t)
  // u(b,y,t) = F_2(y,t)
  auto const &first_dir_start = boundary_->first_dim.first;
  auto const &first_dir_end = boundary_->first_dim.second;
  // dirichlet boundary functions for y axis:
  // u(x,c,t) = G_1(x,t)
  // u(x,d,t) = G_2(x,t)
  auto const &second_dir_start = boundary_->second_dim.first;
  auto const &second_dir_end = boundary_->second_dim.second;

  std::size_t const space_size_1 =
      space_divison.first;  // columns = x axis -> rows
  std::size_t const space_size_2 =
      space_divison.second;  // rows = y axis -> columns
  // calculate scheme const coefficients:
  fp_type const alpha = (std::get<0>(coeffs_) * k) / (h_1 * h_1);
  fp_type const beta = (std::get<1>(coeffs_) * k) / (h_2 * h_2);
  fp_type const gamma =
      (std::get<2>(coeffs_) * k / (static_cast<fp_type>(4.0) * h_1 * h_2));
  fp_type const delta =
      (std::get<3>(coeffs_) * k / (static_cast<fp_type>(2.0) * h_1));
  fp_type const ni =
      (std::get<4>(coeffs_) * k / (static_cast<fp_type>(2.0) * h_2));
  fp_type const rho = (std::get<5>(coeffs_) * k);

  // x space range
  auto const x_range = space_range.first;
  // y space range
  auto const y_range = space_range.second;
  // x init:
  auto const x_init = x_range.lower();
  // y init:
  auto const y_init = y_range.lower();
  // inits pair:
  auto const inits = std::make_pair(x_init, y_init);

  // create container to carry mesh in space and then previous solution:
  matrix_t prev_sol(solution);
  // use the mesh in space to get values of initial condition
  d2d_t::discretize_initial_condition(inits, h, dataPtr_->initial_condition,
                                      prev_sol);

  // prepare containers for diagonal vectors for solver_fst_ptr_:
  vector_t low_fst(space_size_1 + 1, -1.0 * (alpha - delta) * theta);
  vector_t diag_fst(space_size_1 + 1,
                    (1.0 + (2.0 * alpha - 0.5 * rho) * theta));
  vector_t up_fst(space_size_1 + 1, -1.0 * (alpha + delta) * theta);

  vector_t rhs_fst(space_size_1 + 1, fp_type{});
  vector_t intermed_lower(space_size_1 + 1, fp_type{});
  vector_t intermed_upper(space_size_1 + 1, fp_type{});
  // prepare containers for diagonal vectors for solver_sec_ptr_:
  vector_t low_sec(space_size_2 + 1, -1.0 * (beta - ni) * theta);
  vector_t diag_sec(space_size_2 + 1, (1.0 + (2.0 * beta - 0.5 * rho) * theta));
  vector_t up_sec(space_size_2 + 1, -1.0 * (beta + ni) * theta);
  vector_t rhs_sec(space_size_2 + 1, fp_type{});
  vector_t next_lower(space_size_2 + 1, fp_type{});
  vector_t next_upper(space_size_2 + 1, fp_type{});
  //// create container to carry intermediate solution (Y matrix):
  matrix_t intermed_sol(space_size_2 + 1, space_size_1 + 1, fp_type{});
  //// create container to carry final solution (U matrix):
  matrix_t next_sol(solution);
  // create first time point:
  fp_type time = k;
  // store terminal time:
  fp_type const last_time = dataPtr_->time_range.upper();
  // set properties of solver_fst_ptr_:
  solver_fst_ptr_->set_diagonals(std::move(low_fst), std::move(diag_fst),
                                 std::move(up_fst));
  // set properties of solver_sec_ptr_:
  solver_sec_ptr_->set_diagonals(std::move(low_sec), std::move(diag_sec),
                                 std::move(up_sec));
  // differentiate between inhomogeneous and homogeneous PDE:
  if ((dataPtr_->is_source_function_set)) {
    // wrap the scheme coefficients:
    const auto scheme_coeffs =
        std::make_tuple(alpha, beta, gamma, delta, ni, rho, k);
    // get the correct scheme:
    auto scheme_funcs =
        lss_two_dim_heat_implicit_schemes::heat_equation_schemes<
            fp_type, container, alloc>::get_inhom_scheme(scheme);
    auto scheme_fun_0 = scheme_funcs.first;
    auto scheme_fun_1 = scheme_funcs.second;
    // save y_init for dirichlet boundaries:
    auto y_val = fp_type{};
    auto x_val = fp_type{};
    // create a container to carry discretized source heat
    matrix_t source_curr(solution.rows(), solution.columns(), fp_type{});
    matrix_t source_next(solution.rows(), solution.columns(), fp_type{});
    // discretize current source and next source:
    d2d_t::discretize_in_space(inits, h, 0.0, heat_source, source_curr);
    d2d_t::discretize_in_space(inits, h, time, heat_source, source_next);

    // loop for stepping in time:
    while (time <= last_time) {
      // lower Y axis Dirichlet boundary:
      d1d_t::discretize_in_space(h_1, x_init, time, second_dir_start,
                                 intermed_lower);
      d1d_t::discretize_in_space(h_1, x_init, time, second_dir_end,
                                 intermed_upper);

      intermed_sol(0, intermed_lower);
      intermed_sol(space_size_2, intermed_upper);

      for (std::size_t sol_idx = 1; sol_idx < space_size_2; ++sol_idx) {
        y_val = y_init + static_cast<fp_type>(sol_idx) * h_2;
        scheme_fun_0(scheme_coeffs, prev_sol, source_curr, source_next, rhs_fst,
                     sol_idx);
        solver_fst_ptr_->set_boundary_condition(std::make_pair(
            first_dir_start(y_val, time), first_dir_end(y_val, time)));
        solver_fst_ptr_->set_rhs(rhs_fst);
        // just trying to reuse rhs_fst here inm solve:
        solver_fst_ptr_->solve(rhs_fst);
        intermed_sol(sol_idx, rhs_fst);
      }
      // here follows loop accross space_size_2 using intermed_sol:
      // lower Y axis Dirichlet boundary:
      d1d_t::discretize_in_space(h_2, y_init, time, first_dir_start,
                                 next_lower);
      d1d_t::discretize_in_space(h_2, y_init, time, first_dir_end, next_upper);
      next_sol(0, next_lower);
      next_sol(space_size_1, next_upper);
      for (std::size_t sol_idx = 1; sol_idx < space_size_1; ++sol_idx) {
        x_val = x_init + static_cast<fp_type>(sol_idx) * h_1;
        scheme_fun_1(scheme_coeffs, prev_sol, intermed_sol, intermed_sol,
                     rhs_sec, sol_idx);
        solver_sec_ptr_->set_boundary_condition(std::make_pair(
            second_dir_start(x_val, time), second_dir_end(x_val, time)));
        solver_sec_ptr_->set_rhs(rhs_sec);
        // just trying to reuse rhs_sec here inm solve:
        solver_sec_ptr_->solve(rhs_sec);
        next_sol(sol_idx, rhs_sec);
      }
      prev_sol = next_sol;

      d2d_t::discretize_in_space(inits, h, time, heat_source, source_curr);
      d2d_t::discretize_in_space(inits, h, 2.0 * time, heat_source,
                                 source_next);

      time += k;
    }
  } else {
    // wrap the scheme coefficients:
    const auto scheme_coeffs =
        std::make_tuple(alpha, beta, gamma, delta, ni, rho, fp_type{});
    //  get the correct scheme:
    auto scheme_funcs =
        lss_two_dim_heat_implicit_schemes::heat_equation_schemes<
            fp_type, container, alloc>::get_scheme(scheme);
    auto scheme_fun_0 = scheme_funcs.first;
    auto scheme_fun_1 = scheme_funcs.second;
    // save y_init for dirichlet boundaries:
    auto y_val = fp_type{};
    auto x_val = fp_type{};
    // loop for stepping in time:
    while (time <= last_time) {
      // lower Y axis Dirichlet boundary:
      d1d_t::discretize_in_space(h_1, x_init, time, second_dir_start,
                                 intermed_lower);
      d1d_t::discretize_in_space(h_1, x_init, time, second_dir_end,
                                 intermed_upper);

      intermed_sol(0, intermed_lower);
      intermed_sol(space_size_2, intermed_upper);

      for (std::size_t sol_idx = 1; sol_idx < space_size_2; ++sol_idx) {
        y_val = y_init + static_cast<fp_type>(sol_idx) * h_2;
        scheme_fun_0(scheme_coeffs, prev_sol, intermed_sol, intermed_sol,
                     rhs_fst, sol_idx);
        solver_fst_ptr_->set_boundary_condition(std::make_pair(
            first_dir_start(y_val, time), first_dir_end(y_val, time)));
        solver_fst_ptr_->set_rhs(rhs_fst);
        // just trying to reuse rhs_fst here inm solve:
        solver_fst_ptr_->solve(rhs_fst);
        intermed_sol(sol_idx, rhs_fst);
      }
      // here follows loop accross space_size_2 using intermed_sol:
      // lower Y axis Dirichlet boundary:
      d1d_t::discretize_in_space(h_2, y_init, time, first_dir_start,
                                 next_lower);
      d1d_t::discretize_in_space(h_2, y_init, time, first_dir_end, next_upper);
      next_sol(0, next_lower);
      next_sol(space_size_1, next_upper);
      for (std::size_t sol_idx = 1; sol_idx < space_size_1; ++sol_idx) {
        x_val = x_init + static_cast<fp_type>(sol_idx) * h_1;
        scheme_fun_1(scheme_coeffs, prev_sol, intermed_sol, intermed_sol,
                     rhs_sec, sol_idx);
        solver_sec_ptr_->set_boundary_condition(std::make_pair(
            second_dir_start(x_val, time), second_dir_end(x_val, time)));
        solver_sec_ptr_->set_rhs(rhs_sec);
        // just trying to reuse rhs_sec here inm solve:
        solver_sec_ptr_->solve(rhs_sec);
        next_sol(sol_idx, rhs_sec);
      }
      prev_sol = next_sol;
      time += k;
    }
  }
  copy(solution, prev_sol);
}

}  // namespace lss_two_dim_classic_pde_solvers

#endif  //_LSS_2D_GENERAL_HEAT_EQUATION_DIRICHLET_SOLVERS
