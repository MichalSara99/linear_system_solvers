#if !defined(_LSS_HEAT_SPLITTING_METHOD_HPP_)
#define _LSS_HEAT_SPLITTING_METHOD_HPP_

#include <functional>
#include <map>

#include "boundaries/lss_boundary.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "common/lss_utility.hpp"
#include "containers/lss_container_2d.hpp"

namespace lss_pde_solvers
{

namespace two_dimensional
{
using lss_boundary::boundary_2d_pair;
using lss_boundary::boundary_2d_ptr;
using lss_containers::container_2d;
using lss_enumerations::by_enum;
using lss_utility::sptr_t;

/**
    heat_splitting_method object
 */
template <typename fp_type, template <typename, typename> typename container = std::vector,
          typename allocator = std::allocator<fp_type>>
class heat_splitting_method
{
  public:
    explicit heat_splitting_method()
    {
    }

    ~heat_splitting_method()
    {
    }

    heat_splitting_method(heat_splitting_method const &) = delete;
    heat_splitting_method(heat_splitting_method &&) = delete;
    heat_splitting_method &operator=(heat_splitting_method const &) = delete;
    heat_splitting_method &operator=(heat_splitting_method &&) = delete;

    virtual void solve(container_2d<by_enum::Row, fp_type, container, allocator> const &prev_solution,
                       boundary_2d_pair<fp_type> const &horizontal_boundary_pair,
                       boundary_2d_pair<fp_type> const &vertical_boundary_pair, fp_type const &time,
                       std::pair<fp_type, fp_type> const &weights, std::pair<fp_type, fp_type> const &weight_values,
                       container_2d<by_enum::Row, fp_type, container, allocator> &solution) = 0;

    virtual void solve(container_2d<by_enum::Row, fp_type, container, allocator> const &prev_solution,
                       boundary_2d_pair<fp_type> const &horizontal_boundary_pair,
                       boundary_2d_pair<fp_type> const &vertical_boundary_pair, fp_type const &time,
                       std::pair<fp_type, fp_type> const &weights, std::pair<fp_type, fp_type> const &weight_values,
                       std::function<fp_type(fp_type, fp_type)> const &heat_source,
                       container_2d<by_enum::Row, fp_type, container, allocator> &solution) = 0;
};

template <typename fp_type, template <typename, typename> typename container, typename allocator>
using heat_splitting_method_ptr = sptr_t<heat_splitting_method<fp_type, container, allocator>>;

} // namespace two_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_HEAT_SPLITTING_METHOD_HPP_
