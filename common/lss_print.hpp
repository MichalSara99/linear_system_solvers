#pragma once
#if !defined(_LSS_PRINT_HPP_)
#define _LSS_PRINT_HPP_

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <tuple>

#include "common/lss_enumerations.hpp"
#include "containers/lss_container_2d.hpp"
#include "discretization/lss_grid.hpp"
#include "discretization/lss_grid_config.hpp"
#include "discretization/lss_grid_config_hints.hpp"
#include "discretization/lss_grid_transform_config.hpp"
#include "ode_solvers/lss_ode_discretization_config.hpp"
#include "pde_solvers/lss_pde_discretization_config.hpp"

namespace lss_print
{

using lss_containers::container_2d;
using lss_enumerations::by_enum;
using lss_enumerations::grid_enum;

using lss_grids::grid_config_1d;
using lss_grids::grid_config_2d;
using lss_grids::grid_config_hints_1d_ptr;
using lss_grids::grid_config_hints_2d_ptr;
using lss_grids::grid_transform_config_1d;
using lss_grids::grid_transform_config_2d;
using lss_ode_solvers::ode_discretization_config_ptr;
using lss_pde_solvers::pde_discretization_config_1d_ptr;
using lss_pde_solvers::pde_discretization_config_2d_ptr;

/**
 * Prints contents of the container using passed discretization
 *
 * \param ode_discretization_cfg - ODE discretization configuration
 * \param cont - container to be printed
 * \param out - stream to print the contents to
 */
template <typename fp_type, template <typename, typename> typename container, typename allocator>
void print(ode_discretization_config_ptr<fp_type> const &ode_discretization_cfg,
           container<fp_type, allocator> const &cont, std::ostream &out = std::cout)
{
    const std::size_t space_size = ode_discretization_cfg->number_of_space_points();
    LSS_ASSERT(cont.size() == space_size, "Container size differs from passed discretization");
    const fp_type h = ode_discretization_cfg->space_step();
    const auto &space = ode_discretization_cfg->space_range();
    const fp_type space_start = space.lower();

    out << "SPACE_POINTS\n";
    fp_type m{};
    for (std::size_t t = 0; t < space_size - 1; ++t)
    {
        m = static_cast<fp_type>(t);
        out << (space_start + m * h);
        out << ",";
    }
    m = static_cast<fp_type>(space_size - 1);
    out << (space_start + m * h) << "\nVALUES\n";
    for (std::size_t t = 0; t < space_size - 1; ++t)
    {
        out << cont[t];
        out << ",";
    }
    out << cont[space_size - 1];
}

/**
 * Prints contents of the container using passed discretization
 *
 * \param pde_discretization_cfg - 1D PDE discretization configuration
 * \param grid_hints_cfg - 1D grid hints configuration
 * \param cont - container to be printed
 * \param out - stream to print the contents to
 */
template <typename fp_type, template <typename, typename> typename container, typename allocator>
void print(pde_discretization_config_1d_ptr<fp_type> const &pde_discretization_cfg,
           grid_config_hints_1d_ptr<fp_type> const &grid_hints_cfg, container<fp_type, allocator> const &cont,
           std::ostream &out = std::cout)
{
    const std::size_t space_size = pde_discretization_cfg->number_of_space_points();
    LSS_ASSERT(cont.size() == space_size, "Container size differs from passed discretization");
    // create grid_config:
    auto const &grid_cfg = std::make_shared<grid_config_1d<fp_type>>(pde_discretization_cfg);
    // create grid_transform_config:
    auto const &grid_trans_cfg =
        std::make_shared<grid_transform_config_1d<fp_type>>(pde_discretization_cfg, grid_hints_cfg);
    fp_type zeta{};
    out << "SPACE_POINTS\n";
    for (std::size_t t = 0; t < space_size - 1; ++t)
    {
        zeta = grid_1d<fp_type>::value(grid_cfg, t);
        out << grid_1d<fp_type>::transformed_value(grid_trans_cfg, zeta);
        out << ",";
    }
    zeta = grid_1d<fp_type>::value(grid_cfg, space_size - 1);
    out << grid_1d<fp_type>::transformed_value(grid_trans_cfg, zeta) << "\nVALUES\n";
    for (std::size_t t = 0; t < space_size - 1; ++t)
    {
        out << cont[t];
        out << ",";
    }
    out << cont[space_size - 1];
}

/**
 * Prints contents of the container using passed discretization
 *
 * \param pde_discretization_cfg - 2D PDE discretization configuration
 * \param grid_hints_cfg - 2D grid hints configuration
 * \param cont - 2D row-major container to be printed
 * \param out - stream to print the contents to
 */
template <typename fp_type, template <typename, typename> typename container, typename allocator>
void print(pde_discretization_config_2d_ptr<fp_type> const &pde_discretization_cfg,
           grid_config_hints_2d_ptr<fp_type> const &grid_hints_cfg,
           container_2d<by_enum::Row, fp_type, container, allocator> const &cont, std::ostream &out = std::cout)
{
    const auto &space_sizes = pde_discretization_cfg->number_of_space_points();
    LSS_ASSERT((cont.columns() == std::get<1>(space_sizes)) && (cont.rows() == std::get<0>(space_sizes)),
               "The input cont container must have the correct size");
    // create grid_config:
    auto const &grid_cfg = std::make_shared<grid_config_2d<fp_type>>(pde_discretization_cfg);
    // create grid_transform_config:
    auto const &grid_trans_cfg =
        std::make_shared<grid_transform_config_2d<fp_type>>(pde_discretization_cfg, grid_hints_cfg);
    fp_type zeta{}, eta{};
    out << "SPACE_POINTS_X\n";
    for (std::size_t t = 0; t < space_sizes.first - 1; ++t)
    {
        zeta = grid_2d<fp_type>::value_1(grid_cfg, t);
        out << grid_2d<fp_type>::transformed_value_1(grid_trans_cfg, zeta);
        out << ",";
    }
    zeta = grid_2d<fp_type>::value_1(grid_cfg, space_sizes.first - 1);
    out << grid_2d<fp_type>::transformed_value_1(grid_trans_cfg, zeta) << "\nSPACE_POINTS_Y\n";
    for (std::size_t t = 0; t < space_sizes.second - 1; ++t)
    {
        eta = grid_2d<fp_type>::value_2(grid_cfg, t);
        out << grid_2d<fp_type>::transformed_value_2(grid_trans_cfg, eta);
        out << ",";
    }
    eta = grid_2d<fp_type>::value_2(grid_cfg, space_sizes.second - 1);
    out << grid_2d<fp_type>::transformed_value_2(grid_trans_cfg, eta) << "\nVALUES\n";
    for (std::size_t r = 0; r < cont.rows(); ++r)
    {
        for (std::size_t c = 0; c < cont.columns() - 1; ++c)
        {
            out << cont(r, c);
            out << ",";
        }
        out << cont(r, cont.columns() - 1);
        out << "\n";
    }
}

/**
 * Prints contents of the 2D container using passed discretization
 *
 * \param pde_discretization_cfg - 1D PDE discretization configuration
 * \param grid_hints_cfg - 1D grid hints configuration
 * \param cont_2d - 2D row-major container to be printed
 * \param out - stream to print the contents to
 */
template <typename fp_type, template <typename, typename> typename container, typename allocator>
void print(pde_discretization_config_1d_ptr<fp_type> const &pde_discretization_cfg,
           grid_config_hints_1d_ptr<fp_type> const &grid_hints_cfg,
           container_2d<by_enum::Row, fp_type, container, allocator> const &cont_2d, std::ostream &out = std::cout)
{
    const std::size_t space_size = pde_discretization_cfg->number_of_space_points();
    const std::size_t time_size = pde_discretization_cfg->number_of_time_points();
    LSS_ASSERT((cont_2d.columns() == space_size) && (cont_2d.rows() == time_size),
               "Container size differs from passed discretization");

    // create grid_config:
    auto const &grid_cfg = std::make_shared<grid_config_1d<fp_type>>(pde_discretization_cfg);
    // create grid_transform_config:
    auto const &grid_trans_cfg =
        std::make_shared<grid_transform_config_1d<fp_type>>(pde_discretization_cfg, grid_hints_cfg);

    const fp_type k = pde_discretization_cfg->time_step();
    const auto &time = pde_discretization_cfg->time_range();
    const fp_type time_start = time.lower();

    out << "SPACE_POINTS\n";
    fp_type m{}, zeta{};
    for (std::size_t t = 0; t < space_size - 1; ++t)
    {
        zeta = grid_1d<fp_type>::value(grid_cfg, t);
        out << grid_1d<fp_type>::transformed_value(grid_trans_cfg, zeta);
        out << ",";
    }
    zeta = grid_1d<fp_type>::value(grid_cfg, space_size - 1);
    out << grid_1d<fp_type>::transformed_value(grid_trans_cfg, zeta) << "\nTIME_POINTS\n";
    for (std::size_t t = 0; t < time_size - 1; ++t)
    {
        m = static_cast<fp_type>(t);
        out << (time_start + m * k);
        out << ",";
    }
    m = static_cast<fp_type>(time_size - 1);
    out << (time_start + m * k) << "\nVALUES\n";
    for (std::size_t t = 0; t < time_size; ++t)
    {
        for (std::size_t i = 0; i < space_size - 1; ++i)
        {
            out << cont_2d(t, i);
            out << ",";
        }
        out << cont_2d(t, space_size - 1);
        out << "\n";
    }
}

} // namespace lss_print

#endif ///_LSS_PRINT_HPP_
