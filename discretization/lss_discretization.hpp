#if !defined(_LSS_DISCRETIZATION_HPP_)
#define _LSS_DISCRETIZATION_HPP_

#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include "containers/lss_container_2d.hpp"
#include "lss_grid.hpp"
#include "lss_grid_config.hpp"
#include <functional>

using lss_containers::container_2d;
using lss_enumerations::by_enum;
using lss_enumerations::dimension_enum;
using lss_grids::grid_1d;
using lss_grids::grid_2d;
using lss_grids::grid_config_1d_ptr;
using lss_grids::grid_config_2d_ptr;

template <dimension_enum dimension, typename fp_type, template <typename, typename> typename container,
          typename allocator>
struct discretization
{
};

/**
    1D discretization structure
 */
template <typename fp_type, template <typename, typename> typename container, typename allocator>
struct discretization<dimension_enum::One, fp_type, container, allocator>
{
  public:
    /**
     * Discretize 1D space
     *
     * \param grid_config - 1D grid config object
     * \param container_x - 1D container for output
     */
    static void of_space(grid_config_1d_ptr<fp_type> const &grid_config, container<fp_type, allocator> &container_x);

    /**
     * Discretize function F(x) where x = first dim variable
     *
     * \param grid_config - 1D grid config object
     * \param fun - function F(x)
     * \param container_fun - 1D container for output
     */
    static void of_function(grid_config_1d_ptr<fp_type> const &grid_config, std::function<fp_type(fp_type)> const &fun,
                            container<fp_type, allocator> &container_fun);

    /**
     * Discretize function F(t,x) where t = time, x = first dim variable
     *
     * \param grid_config - 1D grid config object
     * \param time - time valraible t
     * \param fun - function F(t,x)
     * \param container_fun_t = 1D container for output
     */
    static void of_function(grid_config_1d_ptr<fp_type> const &grid_config, fp_type const &time,
                            std::function<fp_type(fp_type, fp_type)> const &fun,
                            container<fp_type, allocator> &container_fun_t);
};

/**
 * Discretize 1D space
 *
 * \param grid_config - 1D grid config object
 * \param container_x - 1D container for output
 */
template <typename fp_type, template <typename, typename> typename container, typename allocator>
void discretization<dimension_enum::One, fp_type, container, allocator>::of_space(
    grid_config_1d_ptr<fp_type> const &grid_config, container<fp_type, allocator> &container_x)
{
    LSS_ASSERT(container_x.size() > 0, "The input container must be initialized.");
    for (std::size_t t = 0; t < container_x.size(); ++t)
    {
        container_x[t] = grid_1d<fp_type>::value(grid_config, t);
    }
}

/**
 * Discretize function F(x) where x = first dim variable
 *
 * \param grid_config - 1D grid config object
 * \param fun - function F(x)
 * \param container_fun - 1D container for output
 */
template <typename fp_type, template <typename, typename> typename container, typename allocator>
void discretization<dimension_enum::One, fp_type, container, allocator>::of_function(
    grid_config_1d_ptr<fp_type> const &grid_config, std::function<fp_type(fp_type)> const &fun,
    container<fp_type, allocator> &container_fun)
{
    LSS_ASSERT(container_fun.size() > 0, "The input container must be initialized.");
    for (std::size_t t = 0; t < container_fun.size(); ++t)
    {
        container_fun[t] = fun(grid_1d<fp_type>::value(grid_config, t));
    }
}

/**
 * Discretize function F(t,x) where t = time, x = first dim variable
 *
 * \param grid_config - 1D grid config object
 * \param time - time valraible t
 * \param fun - function F(t,x)
 * \param container_fun_t - 1D container for output
 */
template <typename fp_type, template <typename, typename> typename container, typename allocator>
void discretization<dimension_enum::One, fp_type, container, allocator>::of_function(
    grid_config_1d_ptr<fp_type> const &grid_config, fp_type const &time,
    std::function<fp_type(fp_type, fp_type)> const &fun, container<fp_type, allocator> &container_fun_t)
{
    LSS_ASSERT(container_fun_t.size() > 0, "The input container must be initialized.");
    for (std::size_t t = 0; t < container_fun_t.size(); ++t)
    {
        container_fun_t[t] = fun(time, grid_1d<fp_type>::value(grid_config, t));
    }
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
using discretization_1d = discretization<dimension_enum::One, fp_type, container, allocator>;

/**
    2D discretization structure
 */
template <typename fp_type, template <typename, typename> typename container, typename allocator>
struct discretization<dimension_enum::Two, fp_type, container, allocator>
{
  public:
    /**
     * Discretize function F(x,y) where x=first dim variable,
     *  y = second dim variable
     *
     * \param grid_config - 2D grid config object
     * \param fun - function F(x,y)
     * \param container_fun - 2D container for output
     */
    static void of_function(grid_config_2d_ptr<fp_type> const &grid_config,
                            std::function<fp_type(fp_type, fp_type)> const &fun,
                            container_2d<by_enum::Row, fp_type, container, allocator> &container_fun);

    /**
     * Discretize function F(t,x,y) where t=time, x=first dim variable,
     *  y = second dim variable
     *
     * \param grid_config - 2D grid config object
     * \param time  - time valraible t
     * \param fun - function F(t,x,y)
     * \param container_fun_t - 2D container for output
     */
    static void of_function(grid_config_2d_ptr<fp_type> const &grid_config, fp_type const &time,
                            std::function<fp_type(fp_type, fp_type, fp_type)> const &fun,
                            container_2d<by_enum::Row, fp_type, container, allocator> &container_fun_t);
    /**
     * Discretize function F(t,x,y) where t=time, x=first dim variable,
     *  y = second dim variable
     *
     * \param grid_config - 2D grid config object
     * \param time - time valraible t
     * \param fun - function F(t,x,y)
     * \param rows - number of rows of the output
     * \param cols - number of columns of the output
     * \param cont - 1D container for output (row-wise)
     */
    static void of_function(grid_config_2d_ptr<fp_type> const &grid_config, fp_type const &time,
                            std::function<fp_type(fp_type, fp_type, fp_type)> const &fun, std::size_t const &rows,
                            std::size_t const &cols, container<fp_type, allocator> &cont);
};

/**
 * Discretize function F(x,y) where x=first dim variable,
 *  y = second dim variable
 *
 * \param grid_config - 2D grid config object
 * \param fun - function F(x,y)
 * \param container_fun - 2D container for output
 */
template <typename fp_type, template <typename, typename> typename container, typename allocator>
void discretization<dimension_enum::Two, fp_type, container, allocator>::of_function(
    grid_config_2d_ptr<fp_type> const &grid_config, std::function<fp_type(fp_type, fp_type)> const &fun,
    container_2d<by_enum::Row, fp_type, container, allocator> &container_fun)
{
    LSS_ASSERT(container_fun.rows() > 0, "The input container must be initialized.");
    LSS_ASSERT(container_fun.columns() > 0, "The input container must be initialized.");
    fp_type value{};
    for (std::size_t r = 0; r < container_fun.rows(); ++r)
    {
        for (std::size_t c = 0; c < container_fun.columns(); ++c)
        {
            value = fun(grid_2d<fp_type>::value_1(grid_config, r), grid_2d<fp_type>::value_2(grid_config, c));
            container_fun(r, c, value);
        }
    }
}

/**
 * Discretize function F(t,x,y) where t=time, x=first dim variable,
 *  y = second dim variable
 *
 * \param grid_config - 2D grid config object
 * \param time  - time valraible t
 * \param fun - function F(t,x,y)
 * \param container_fun_t - 2D container for output
 */
template <typename fp_type, template <typename, typename> typename container, typename allocator>
void discretization<dimension_enum::Two, fp_type, container, allocator>::of_function(
    grid_config_2d_ptr<fp_type> const &grid_config, fp_type const &time,
    std::function<fp_type(fp_type, fp_type, fp_type)> const &fun,
    container_2d<by_enum::Row, fp_type, container, allocator> &container_fun_t)
{
    LSS_ASSERT(container_fun_t.rows() > 0, "The input container must be initialized.");
    LSS_ASSERT(container_fun_t.columns() > 0, "The input container must be initialized.");
    fp_type value{};
    for (std::size_t r = 0; r < container_fun_t.rows(); ++r)
    {
        for (std::size_t c = 0; c < container_fun_t.columns(); ++c)
        {
            value = fun(time, grid_2d<fp_type>::value_1(grid_config, r), grid_2d<fp_type>::value_2(grid_config, c));
            container_fun_t(r, c, value);
        }
    }
}

/**
 * Discretize function F(t,x,y) where t=time, x=first dim variable,
 *  y = second dim variable
 *
 * \param grid_config - 2D grid config object
 * \param time  - time valraible t
 * \param fun - function F(t,x,y)
 * \param rows - number of rows of the putput
 * \param columns - number of columns of the output
 * \param cont - 1D container for output (row-wise)
 */
template <typename fp_type, template <typename, typename> typename container, typename allocator>
void discretization<dimension_enum::Two, fp_type, container, allocator>::of_function(
    grid_config_2d_ptr<fp_type> const &grid_config, fp_type const &time,
    std::function<fp_type(fp_type, fp_type, fp_type)> const &fun, std::size_t const &rows, std::size_t const &cols,
    container<fp_type, allocator> &cont)
{
    LSS_ASSERT(cont.size() > 0, "The input container must be initialized.");
    fp_type value{};
    for (std::size_t r = 0; r < rows; ++r)
    {
        for (std::size_t c = 0; c < cols; ++c)
        {
            value = fun(time, grid_2d<fp_type>::value_1(grid_config, r), grid_2d<fp_type>::value_2(grid_config, c));
            cont[c + r * cols] = value;
        }
    }
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
using discretization_2d = discretization<dimension_enum::Two, fp_type, container, allocator>;

#endif ///_LSS_DISCRETIZATION_HPP_
