#if !defined(_LSS_DISCRETIZATION_HPP_)
#define _LSS_DISCRETIZATION_HPP_

#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include <functional>

namespace lss_pde_solvers
{

using lss_enumerations::dimension_enum;

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
    static void of_space(fp_type const &init_x, fp_type const &step_x, container<fp_type, allocator> &container_x);

    static void of_function(fp_type const &init_x, fp_type const &step_x, std::function<fp_type(fp_type)> const &fun,
                            container<fp_type, allocator> &container_fun);

    static void of_function(fp_type const &init_x, fp_type const &step_x, fp_type const &time,
                            std::function<fp_type(fp_type, fp_type)> const &fun,
                            container<fp_type, allocator> &container_fun_t);
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void discretization<dimension_enum::One, fp_type, container, allocator>::of_space(
    fp_type const &init_x, fp_type const &step_x, container<fp_type, allocator> &container_x)
{
    LSS_ASSERT(container_x.size() > 0, "The input container must be initialized.");
    container_x[0] = init_x;
    for (std::size_t t = 1; t < container_x.size(); ++t)
    {
        container_x[t] = container_x[t - 1] + step_x;
    }
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void discretization<dimension_enum::One, fp_type, container, allocator>::of_function(
    fp_type const &init_x, fp_type const &step_x, std::function<fp_type(fp_type)> const &fun,
    container<fp_type, allocator> &container_fun)
{
    LSS_ASSERT(container_fun.size() > 0, "The input container must be initialized.");
    for (std::size_t t = 0; t < container_fun.size(); ++t)
    {
        container_fun[t] = fun(init_x + static_cast<fp_type>(t) * step_x);
    }
}

template <typename fp_type, template <typename, typename> typename container, typename allocator>
void discretization<dimension_enum::One, fp_type, container, allocator>::of_function(
    fp_type const &init_x, fp_type const &step_x, fp_type const &time,
    std::function<fp_type(fp_type, fp_type)> const &fun, container<fp_type, allocator> &container_fun_t)
{
    LSS_ASSERT(container_fun_t.size() > 0, "The input container must be initialized.");
    for (std::size_t t = 0; t < container_fun_t.size(); ++t)
    {
        container_fun_t[t] = fun(init_x + static_cast<fp_type>(t) * step_x, time);
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
};

} // namespace lss_pde_solvers

#endif ///_LSS_DISCRETIZATION_HPP_
