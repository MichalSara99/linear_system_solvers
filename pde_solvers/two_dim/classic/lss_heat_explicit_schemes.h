#pragma once
#if !defined(_LSS_2D_HEAT_EXPLICIT_SCHEMES)
#define _LSS_2D_HEAT_EXPLICIT_SCHEMES

#pragma warning(disable : 4244)

#include <future>

#include "common/lss_containers.h"
#include "common/lss_enumerations.h"
#include "common/lss_utility.h"
#include "pde_solvers/two_dim/lss_base_explicit_schemes.h"
#include "pde_solvers/two_dim/lss_pde_utility.h"

namespace lss_two_dim_heat_explicit_schemes {

using lss_containers::container_2d;
using lss_containers::copy;
using lss_enumerations::boundary_condition_enum;
using lss_two_dim_base_explicit_schemes::heat_scheme_base;
using lss_two_dim_pde_utility::dirichlet_boundary_2d;
using lss_two_dim_pde_utility::discretization_2d;
using lss_two_dim_pde_utility::pde_coefficient_holder_const;
using lss_two_dim_pde_utility::robin_boundary_2d;
using lss_utility::sptr_t;

// ============================================================================
// ======================= heat_euler_scheme ==================================
// ============================================================================

template <template <typename, typename> typename container, typename fp_type,
          typename alloc>
class heat_euler_scheme
    : public heat_scheme_base<container, fp_type, alloc,
                              pde_coefficient_holder_const<fp_type>> {
 private:
  typedef container_2d<container, fp_type, alloc> matrix_t;
  heat_euler_scheme() = default;

 public:
  explicit heat_euler_scheme(
      fp_type time, fp_type time_delta,
      std::pair<fp_type, fp_type> const &spatial_inits,
      std::pair<fp_type, fp_type> const &spatial_deltas,
      pde_coefficient_holder_const<fp_type> const &coeffs,
      sptr_t<matrix_t> const &initial_condition,
      std::function<fp_type(fp_type, fp_type, fp_type)> const &source = nullptr,
      bool is_source_set = false)
      : heat_scheme_base<container, fp_type, alloc,
                         pde_coefficient_holder_const<fp_type>>(
            time, time_delta, spatial_inits, spatial_deltas, coeffs,
            initial_condition, source, is_source_set) {}

  ~heat_euler_scheme() {}

  heat_euler_scheme(heat_euler_scheme const &) = delete;
  heat_euler_scheme(heat_euler_scheme &&) = delete;
  heat_euler_scheme &operator=(heat_euler_scheme const &) = delete;
  heat_euler_scheme &operator=(heat_euler_scheme &&) = delete;

  // stability check:
  bool is_stable() const override;

  // for Dirichlet BC
  void operator()(
      sptr_t<dirichlet_boundary_2d<fp_type>> const &dirichlet_boundary,
      matrix_t &solution) const override;
  // for Robin BC
  void operator()(sptr_t<robin_boundary_2d<fp_type>> const &robin_boundary,
                  matrix_t &solution) const override;
};

// ============================================================================
// ================== ade_heat_bakarat_clark_scheme  ==========================
// ============================================================================

template <template <typename, typename> typename container, typename fp_type,
          typename alloc>
class ade_heat_bakarat_clark_scheme
    : public heat_scheme_base<container, fp_type, alloc,
                              pde_coefficient_holder_const<fp_type>> {
 private:
  typedef container_2d<container, fp_type, alloc> matrix_t;
  ade_heat_bakarat_clark_scheme() = default;

 public:
  explicit ade_heat_bakarat_clark_scheme(
      fp_type time, fp_type time_delta,
      std::pair<fp_type, fp_type> const &spatial_inits,
      std::pair<fp_type, fp_type> const &spatial_deltas,
      pde_coefficient_holder_const<fp_type> const &coeffs,
      sptr_t<matrix_t> const &initial_condition,
      std::function<fp_type(fp_type, fp_type, fp_type)> const &source = nullptr,
      bool is_source_set = false)
      : heat_scheme_base<container, fp_type, alloc,
                         pde_coefficient_holder_const<fp_type>>(
            time, time_delta, spatial_inits, spatial_deltas, coeffs,
            initial_condition, source, is_source_set) {}

  ~ade_heat_bakarat_clark_scheme() {}

  ade_heat_bakarat_clark_scheme(ade_heat_bakarat_clark_scheme const &) = delete;
  ade_heat_bakarat_clark_scheme(ade_heat_bakarat_clark_scheme &&) = delete;
  ade_heat_bakarat_clark_scheme &operator=(
      ade_heat_bakarat_clark_scheme const &) = delete;
  ade_heat_bakarat_clark_scheme &operator=(ade_heat_bakarat_clark_scheme &&) =
      delete;

  // stability check:
  bool is_stable() const override { return true; };

  // for Dirichlet BC
  void operator()(
      sptr_t<dirichlet_boundary_2d<fp_type>> const &dirichlet_boundary,
      matrix_t &solution) const override;
  // for Robin BC
  void operator()(sptr_t<robin_boundary_2d<fp_type>> const &robin_boundary,
                  matrix_t &solution) const override;
};

// ============================================================================
// ==================== ade_heat_saulyev_scheme ===============================
// ============================================================================

template <template <typename, typename> typename container, typename fp_type,
          typename alloc>
class ade_heat_saulyev_scheme
    : public heat_scheme_base<container, fp_type, alloc,
                              pde_coefficient_holder_const<fp_type>> {
 private:
  typedef container_2d<container, fp_type, alloc> matrix_t;
  ade_heat_saulyev_scheme() = default;

 public:
  explicit ade_heat_saulyev_scheme(
      fp_type time, fp_type time_delta,
      std::pair<fp_type, fp_type> const &spatial_inits,
      std::pair<fp_type, fp_type> const &spatial_deltas,
      pde_coefficient_holder_const<fp_type> const &coeffs,
      sptr_t<matrix_t> const &initial_condition,
      std::function<fp_type(fp_type, fp_type, fp_type)> const &source = nullptr,
      bool is_source_set = false)
      : heat_scheme_base<container, fp_type, alloc,
                         pde_coefficient_holder_const<fp_type>>(
            time, time_delta, spatial_inits, spatial_deltas, coeffs,
            initial_condition, source, is_source_set) {}

  ~ade_heat_saulyev_scheme() {}

  ade_heat_saulyev_scheme(ade_heat_saulyev_scheme const &) = delete;
  ade_heat_saulyev_scheme(ade_heat_saulyev_scheme &&) = delete;
  ade_heat_saulyev_scheme &operator=(ade_heat_saulyev_scheme const &) = delete;
  ade_heat_saulyev_scheme &operator=(ade_heat_saulyev_scheme &&) = delete;

  // stability check:
  bool is_stable() const override { return true; };

  // for Dirichlet BC
  void operator()(
      sptr_t<dirichlet_boundary_2d<fp_type>> const &dirichlet_boundary,
      matrix_t &solution) const override;
  // for Robin BC
  void operator()(sptr_t<robin_boundary_2d<fp_type>> const &robin_boundary,
                  matrix_t &solution) const override;
};

}  // namespace lss_two_dim_heat_explicit_schemes

// ============================================================================
// =========================== IMPLEMENTATIONS ================================

// TODO: this needs to be checked for first and mixed derivatives !!!
template <template <typename, typename> typename container, typename fp_type,
          typename alloc>
bool lss_two_dim_heat_explicit_schemes::heat_euler_scheme<
    container, fp_type, alloc>::is_stable() const {
  fp_type const alpha = std::get<0>(coeffs_);
  fp_type const beta = std::get<1>(coeffs_);
  fp_type const gamma = std::get<2>(coeffs_);
  fp_type const delta = std::get<3>(coeffs_);
  fp_type const ni = std::get<4>(coeffs_);

  fp_type const secon_ord = static_cast<fp_type>(2.0) * (alpha + beta);
  fp_type const first_ord = static_cast<fp_type>(2.0) * (delta + ni);

  return ((secon_ord <= 1.0) && (first_ord <= 1.0) && ((2.0 * gamma) <= 1.0));
}

template <template <typename, typename> typename container, typename fp_type,
          typename alloc>
void lss_two_dim_heat_explicit_schemes::
    heat_euler_scheme<container, fp_type, alloc>::operator()(
        sptr_t<dirichlet_boundary_2d<fp_type>> const &dirichlet_boundary,
        matrix_t &solution) const {
  LSS_ASSERT(is_stable() == true, "This discretization is not stable.");
  LSS_ASSERT(((solution.rows() > 0) && (solution.columns() > 0)),
             "The input solution container must be initialized.");
  LSS_ASSERT(
      ((solution.rows() == initial_condition_->rows()) &&
       (solution.columns() == initial_condition_->columns())),
      "Entered solution vector size differs from initialCondition vector.");
  typedef discretization_2d<fp_type, container, alloc> d_2d;

  // get delta time:
  fp_type const k = time_delta_;
  // spacial deltas:
  auto const &h = spatial_deltas_;
  // spacial inits:
  auto const &inits = spatial_inits_;
  // create first time point:
  fp_type time = k;
  // calculate scheme coefficients:
  fp_type two = static_cast<fp_type>(2.0);
  fp_type const alpha = std::get<0>(coeffs_);
  fp_type const beta = std::get<1>(coeffs_);
  fp_type const gamma = std::get<2>(coeffs_);
  fp_type const delta = std::get<3>(coeffs_);
  fp_type const ni = std::get<4>(coeffs_);
  fp_type const rho = std::get<5>(coeffs_) * two;
  // intermediate constants:
  fp_type const A =
      static_cast<fp_type>(1.0) - two * alpha - two * beta + two * gamma + rho;
  fp_type const B_P = alpha + delta - gamma;
  fp_type const B_M = alpha - delta - gamma;
  fp_type const C_P = beta + ni - gamma;
  fp_type const C_M = beta - ni - gamma;
  // conmponents of the solution:
  // copy initial condition to solution:
  copy(solution, *initial_condition_);
  matrix_t next_solution(*initial_condition_);
  // size of the space vector:
  std::size_t const row_size = initial_condition_->rows();
  std::size_t const column_size = initial_condition_->columns();
  // create a container to carry discretized source heat
  matrix_t source_curr(*initial_condition_);
  // create kernel anonymous function:
  auto kernel = [=](matrix_t &next_sol, const matrix_t &prev_sol,
                    std::size_t row_idx, matrix_t const &rhs,
                    fp_type rhs_coeff) {
    fp_type val{};
    for (std::size_t c = 1; c < column_size - 1; ++c) {
      val = A * prev_sol(row_idx, c) + B_P * prev_sol(row_idx + 1, c) +
            B_M * prev_sol(row_idx - 1, c) + C_P * prev_sol(row_idx, c + 1) +
            C_M * prev_sol(row_idx, c - 1) +
            gamma * prev_sol(row_idx + 1, c + 1) +
            gamma * prev_sol(row_idx - 1, c - 1) +
            k * rhs_coeff * rhs(row_idx, c);
      next_sol(row_idx, c, val);
    }
  };

  std::vector<std::future<void>> futures;
  if (!is_source_set_) {
    // loop for stepping in time:
    while (time <= time_) {
      dirichlet_boundary->fill(inits, h, time, solution);
      dirichlet_boundary->fill(inits, h, time, next_solution);
      for (std::size_t r = 1; r < row_size - 1; ++r) {
        futures.emplace_back(std::async(std::launch::async, kernel,
                                        std::ref(next_solution), solution, r,
                                        source_curr, 0.0));
      }

      for (auto &future : futures) {
        future.get();
      }
      futures.clear();
      copy(solution, next_solution);
      time += k;
    }
  } else {
    d_2d::discretize_in_space(inits, h, 0.0, source_, source_curr);
    // loop for stepping in time:
    while (time <= time_) {
      dirichlet_boundary->fill(inits, h, time, solution);
      dirichlet_boundary->fill(inits, h, time, next_solution);
      for (std::size_t r = 1; r < row_size - 1; ++r) {
        futures.emplace_back(std::async(std::launch::async, kernel,
                                        std::ref(next_solution), solution, r,
                                        source_curr, 1.0));
      }

      for (auto &future : futures) {
        future.get();
      }
      futures.clear();
      copy(solution, next_solution);
      d_2d::discretize_in_space(inits, h, time, source_, source_curr);
      time += k;
    }
  }
}

template <template <typename, typename> typename container, typename fp_type,
          typename alloc>
void lss_two_dim_heat_explicit_schemes::heat_euler_scheme<
    container, fp_type,
    alloc>::operator()(sptr_t<robin_boundary_2d<fp_type>> const &robin_boundary,
                       matrix_t &solution) const {
  throw std::exception("Not yet implemented. TODO");
}

template <template <typename, typename> typename container, typename fp_type,
          typename alloc>
void lss_two_dim_heat_explicit_schemes::
    ade_heat_bakarat_clark_scheme<container, fp_type, alloc>::operator()(
        sptr_t<dirichlet_boundary_2d<fp_type>> const &dirichlet_boundary,
        matrix_t &solution) const {
  LSS_ASSERT(((solution.rows() > 0) && (solution.columns() > 0)),
             "The input solution container must be initialized.");
  LSS_ASSERT(
      ((solution.rows() == initial_condition_->rows()) &&
       (solution.columns() == initial_condition_->columns())),
      "Entered solution vector size differs from initialCondition vector.");
  typedef discretization_2d<fp_type, container, alloc> d_2d;

  // get delta time:
  fp_type const k = time_delta_;
  // spacial deltas:
  auto const &h = spatial_deltas_;
  // spacial inits:
  auto const &inits = spatial_inits_;
  // create first time point:
  fp_type time = k;
  // calculate scheme coefficients:
  fp_type const alpha = std::get<0>(coeffs_);
  fp_type const beta = std::get<1>(coeffs_);
  fp_type const gamma = std::get<2>(coeffs_);
  fp_type const delta = std::get<3>(coeffs_);
  fp_type const ni = std::get<4>(coeffs_);
  fp_type const rho = std::get<5>(coeffs_);
  // intermediate constants:
  fp_type const A = static_cast<fp_type>(1.0) + alpha + beta - gamma - rho;
  fp_type const B_A =
      ((static_cast<fp_type>(1.0) + gamma + rho - alpha - beta) / A);
  fp_type const CP_A = ((alpha + delta - gamma) / A);
  fp_type const CM_A = ((alpha - delta - gamma) / A);
  fp_type const DP_A = ((beta + ni - gamma) / A);
  fp_type const DM_A = ((beta - ni - gamma) / A);
  fp_type const E_A = gamma / A;
  fp_type const F_A = k / A;
  // conmponents of the solution:
  matrix_t com_1(*initial_condition_);
  matrix_t com_2(*initial_condition_);
  // size of the space vector:
  std::size_t const row_size = initial_condition_->rows();
  std::size_t const column_size = initial_condition_->columns();
  // create a container to carry discretized source heat
  matrix_t source_curr(*initial_condition_);
  matrix_t source_next(*initial_condition_);
  // create upsweep anonymous function:
  auto up_sweep = [=](matrix_t &up_component, matrix_t const &rhs,
                      fp_type rhs_coeff) {
    fp_type val{};
    for (std::size_t r = 1; r < row_size - 1; ++r) {
      for (std::size_t c = 1; c < column_size - 1; ++c) {
        val = B_A * up_component(r, c) + CP_A * up_component(r + 1, c) +
              DP_A * up_component(r, c + 1) + E_A * up_component(r + 1, c + 1) +
              CM_A * up_component(r - 1, c) + DM_A * up_component(r, c - 1) +
              E_A * up_component(r - 1, c - 1) + F_A * rhs_coeff * rhs(r, c);
        up_component(r, c, val);
      }
    }
  };
  // create downsweep anonymous function:
  auto down_sweep = [=](matrix_t &down_component, matrix_t const &rhs,
                        fp_type rhs_coeff) {
    fp_type val{};
    for (std::size_t r = row_size - 2; r >= 1; --r) {
      for (std::size_t c = column_size - 2; c >= 1; --c) {
        val = B_A * down_component(r, c) + CM_A * down_component(r - 1, c) +
              DM_A * down_component(r, c - 1) +
              E_A * down_component(r - 1, c - 1) +
              CP_A * down_component(r + 1, c) +
              DP_A * down_component(r, c + 1) +
              E_A * down_component(r + 1, c + 1) + F_A * rhs_coeff * rhs(r, c);
        down_component(r, c, val);
      }
    }
  };

  if (!is_source_set_) {
    // loop for stepping in time:
    while (time <= time_) {
      dirichlet_boundary->fill(inits, h, time, com_1);
      dirichlet_boundary->fill(inits, h, time, com_2);
      std::thread up_sweep_tr(std::move(up_sweep), std::ref(com_1), source_curr,
                              0.0);
      std::thread down_sweep_tr(std::move(down_sweep), std::ref(com_2),
                                source_curr, 0.0);
      up_sweep_tr.join();
      down_sweep_tr.join();
      for (std::size_t r = 0; r < row_size; ++r) {
        for (std::size_t c = 0; c < column_size; ++c) {
          solution(r, c, (0.5 * (com_1(r, c) + com_2(r, c))));
        }
      }
      time += k;
    }
  } else {
    d_2d::discretize_in_space(inits, h, 0.0, source_, source_curr);
    d_2d::discretize_in_space(inits, h, time, source_, source_next);
    // loop for stepping in time:
    while (time <= time_) {
      dirichlet_boundary->fill(inits, h, time, com_1);
      dirichlet_boundary->fill(inits, h, time, com_2);
      std::thread up_sweep_tr(std::move(up_sweep), std::ref(com_1), source_next,
                              1.0);
      std::thread down_sweep_tr(std::move(down_sweep), std::ref(com_2),
                                source_curr, 1.0);
      up_sweep_tr.join();
      down_sweep_tr.join();
      for (std::size_t r = 0; r < row_size; ++r) {
        for (std::size_t c = 0; c < column_size; ++c) {
          solution(r, c, (0.5 * (com_1(r, c) + com_2(r, c))));
        }
      }
      d_2d::discretize_in_space(inits, h, time, source_, source_curr);
      d_2d::discretize_in_space(inits, h, 2.0 * time, source_, source_next);
      time += k;
    }
  }
}

template <template <typename, typename> typename container, typename fp_type,
          typename alloc>
void lss_two_dim_heat_explicit_schemes::ade_heat_bakarat_clark_scheme<
    container, fp_type,
    alloc>::operator()(sptr_t<robin_boundary_2d<fp_type>> const &robin_boundary,
                       matrix_t &solution) const {
  throw new std::exception("Not available.");
}

template <template <typename, typename> typename container, typename fp_type,
          typename alloc>
void lss_two_dim_heat_explicit_schemes::
    ade_heat_saulyev_scheme<container, fp_type, alloc>::operator()(
        sptr_t<dirichlet_boundary_2d<fp_type>> const &dirichlet_boundary,
        matrix_t &solution) const {
  LSS_ASSERT(((solution.rows() > 0) && (solution.columns() > 0)),
             "The input solution container must be initialized.");
  LSS_ASSERT(
      ((solution.rows() == initial_condition_->rows()) &&
       (solution.columns() == initial_condition_->columns())),
      "Entered solution vector size differs from initialCondition vector.");
  typedef discretization_2d<fp_type, container, alloc> d_2d;

  // get delta time:
  fp_type const k = time_delta_;
  // spacial deltas:
  auto const &h = spatial_deltas_;
  // spacial inits:
  auto const &inits = spatial_inits_;
  // create first time point:
  fp_type time = k;
  // calculate scheme coefficients:
  fp_type const alpha = std::get<0>(coeffs_);
  fp_type const beta = std::get<1>(coeffs_);
  fp_type const gamma = std::get<2>(coeffs_);
  fp_type const delta = std::get<3>(coeffs_);
  fp_type const ni = std::get<4>(coeffs_);
  fp_type const rho = std::get<5>(coeffs_);
  // intermediate constants:
  fp_type const A = static_cast<fp_type>(1.0) + alpha + beta - gamma - rho;
  fp_type const B_A =
      ((static_cast<fp_type>(1.0) + gamma + rho - alpha - beta) / A);
  fp_type const CP_A = ((alpha + delta - gamma) / A);
  fp_type const CM_A = ((alpha - delta - gamma) / A);
  fp_type const DP_A = ((beta + ni - gamma) / A);
  fp_type const DM_A = ((beta - ni - gamma) / A);
  fp_type const E_A = gamma / A;
  fp_type const F_A = k / A;
  // size of the space vector:
  std::size_t const row_size = initial_condition_->rows();
  std::size_t const column_size = initial_condition_->columns();
  // copy initial condition to solution:
  copy(solution, *initial_condition_);
  // create a container to carry discretized source heat
  matrix_t source_curr(*initial_condition_);
  matrix_t source_next(*initial_condition_);
  // create upsweep anonymous function:
  auto up_sweep = [=](matrix_t &up_component, matrix_t const &rhs,
                      fp_type rhs_coeff) {
    fp_type val{};
    for (std::size_t r = 1; r < row_size - 1; ++r) {
      for (std::size_t c = 1; c < column_size - 1; ++c) {
        val = B_A * up_component(r, c) + CP_A * up_component(r + 1, c) +
              DP_A * up_component(r, c + 1) + E_A * up_component(r + 1, c + 1) +
              CM_A * up_component(r - 1, c) + DM_A * up_component(r, c - 1) +
              E_A * up_component(r - 1, c - 1) + F_A * rhs_coeff * rhs(r, c);
        up_component(r, c, val);
      }
    }
  };
  // create downsweep anonymous function:
  auto down_sweep = [=](matrix_t &down_component, matrix_t const &rhs,
                        fp_type rhs_coeff) {
    fp_type val{};
    for (std::size_t r = row_size - 2; r >= 1; --r) {
      for (std::size_t c = column_size - 2; c >= 1; --c) {
        val = B_A * down_component(r, c) + CM_A * down_component(r - 1, c) +
              DM_A * down_component(r, c - 1) +
              E_A * down_component(r - 1, c - 1) +
              CP_A * down_component(r + 1, c) +
              DP_A * down_component(r, c + 1) +
              E_A * down_component(r + 1, c + 1) + F_A * rhs_coeff * rhs(r, c);
        down_component(r, c, val);
      }
    }
  };

  if (!is_source_set_) {
    // loop for stepping in time:
    std::size_t t = 1;
    while (time <= time_) {
      dirichlet_boundary->fill(inits, h, time, solution);
      if (t % 2 == 0)
        down_sweep(solution, source_curr, 0.0);
      else
        up_sweep(solution, source_curr, 0.0);
      ++t;
      time += k;
    }
  } else {
    d_2d::discretize_in_space(inits, h, 0.0, source_, source_curr);
    d_2d::discretize_in_space(inits, h, time, source_, source_next);
    // loop for stepping in time:
    std::size_t t = 1;
    while (time <= time_) {
      dirichlet_boundary->fill(inits, h, time, solution);
      if (t % 2 == 0)
        down_sweep(solution, source_curr, 1.0);
      else
        up_sweep(solution, source_next, 1.0);
      ++t;
      d_2d::discretize_in_space(inits, h, time, source_, source_curr);
      d_2d::discretize_in_space(inits, h, 2.0 * time, source_, source_next);
      time += k;
    }
  }
}

template <template <typename, typename> typename container, typename fp_type,
          typename alloc>
void lss_two_dim_heat_explicit_schemes::ade_heat_saulyev_scheme<
    container, fp_type,
    alloc>::operator()(sptr_t<robin_boundary_2d<fp_type>> const &robin_boundary,
                       matrix_t &solution) const {
  throw new std::exception("Not available.");
}

#endif  //_LSS_2D_HEAT_EXPLICIT_SCHEMES
