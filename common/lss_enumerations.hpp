#pragma once
#if !defined(_LSS_ENUMERATIONS_HPP_)
#define _LSS_ENUMERATIONS_HPP_

#include <tuple>
#include <vector>

#include "lss_macros.hpp"

namespace lss_enumerations
{
/**
    by_enum
 */
enum class by_enum
{
    Row,
    Column,
};

/**
    factorization_enum
 */
enum class factorization_enum
{
    QRMethod,
    LUMethod,
    CholeskyMethod,
    None,
};

/**
    splitting_method_enum
 */
enum class splitting_method_enum
{
    DouglasRachford,
    CraigSneyd,
    ModifiedCraigSneyd,
    HundsdorferVerwer,
};

/**
    tridiagonal_method_enum
 */
enum class tridiagonal_method_enum
{
    CUDASolver,
    DoubleSweepSolver,
    SORSolver,
    ThomasLUSolver,
};

/**
    dimension_enum
 */
enum class dimension_enum
{
    One,
    Two,
};

/**
    traverse_direction_enum
 */
enum class traverse_direction_enum
{
    Forward,
    Backward,
};

/**
    memory_space_enum
 */
enum class memory_space_enum
{
    Host,
    Device
};

/**
    flat_matrix_sort_enum
 */
enum class flat_matrix_sort_enum
{
    RowMajor,
    ColumnMajor
};

/**
    implicit_pde_schemes_enum
 */
enum class implicit_pde_schemes_enum
{
    Euler,
    Theta_30,
    CrankNicolson,
    Theta_80,
};

/**
    explicit_pde_schemes_enum
 */
enum class explicit_pde_schemes_enum
{
    Euler,
    ADEBarakatClark,
    ADESaulyev
};

/**
    dirichlet_side_enum
 */
enum class dirichlet_side_enum
{
    Left,
    Up,
    Right,
    Bottom
};

} // namespace lss_enumerations

#endif ///_LSS_ENUMERATIONS_HPP_
