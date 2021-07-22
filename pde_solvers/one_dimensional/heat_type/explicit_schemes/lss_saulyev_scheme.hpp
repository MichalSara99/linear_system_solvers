#if !defined(_LSS_SAULYEV_SCHEME_HPP_)
#define _LSS_SAULYEV_SCHEME_HPP_

#include "boundaries/lss_boundary_1d.hpp"
#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include "containers/lss_container_2d.hpp"
#include "pde_solvers/lss_discretization.hpp"
#include "pde_solvers/lss_discretization_config.hpp"

namespace lss_pde_solvers
{

namespace one_dimensional
{

using lss_containers::container_2d;
using lss_enumerations::traverse_direction_enum;
using lss_utility::range;

template <typename fp_type>
using function_triplet =
    std::tuple<std::function<fp_type(fp_type)>, std::function<fp_type(fp_type)>, std::function<fp_type(fp_type)>>;

template <typename fp_type, template <typename, typename> typename container, typename allocator> class saulyev_scheme
{
    typedef discretization<dimension_enum::One, fp_type, container, allocator> d_1d;
    typedef container<fp_type, allocator> container_t;

  private:
    function_triplet<fp_type> fun_triplet_;
    boundary_1d_pair<fp_type> boundary_pair_;
    discretization_config_1d_ptr<fp_type> discretization_cfg_;

    void initialize()
    {
        auto const &first = boundary_pair_.first;
        if (std::dynamic_pointer_cast<neumann_boundary_1d>(first))
        {
            throw std::exception("Neumann boundary type is not supported for this scheme");
        }
        if (std::dynamic_pointer_cast<robin_boundary_1d>(first))
        {
            throw std::exception("Robin boundary type is not supported for this scheme");
        }
        auto const &second = boundary_pair_.second;
        if (std::dynamic_pointer_cast<neumann_boundary_1d>(second))
        {
            throw std::exception("Neumann boundary type is not supported for this scheme");
        }
        if (std::dynamic_pointer_cast<robin_boundary_1d>(second))
        {
            throw std::exception("Robin boundary type is not supported for this scheme");
        }
    }

    explicit saulyev_scheme() = delete;

  public:
    saulyev_scheme(function_triplet<fp_type> const &fun_triplet, boundary_1d_pair<fp_type> const &boundary_pair,
                   discretization_config_1d_ptr<fp_type> const &discretization_config)
        : fun_triplet_{fun_triplet}, boundary_pair_{boundary_pair}, discretization_cfg_{discretization_config}
    {
        initialize();
    }

    ~saulyev_scheme()
    {
    }

    void operator()(container_t &prev_solution, container_t &next_solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type)> const &heat_source, traverse_direction_enum traverse_dir)
    {
        if (traverse_dir == traverse_direction_enum::Forward)
        {
        }
        else if (traverse_dir == traverse_direction_enum::Backward)
        {
        }
        else
        {
            throw std::exception("Unreachable");
        }
    }

    void operator()(container_t &prev_solution, container_t &next_solution, bool is_heat_sourse_set,
                    std::function<fp_type(fp_type, fp_type)> const &heat_source, traverse_direction_enum traverse_dir,
                    container_2d<fp_type, container, allocator> &solutions)
    {
        if (traverse_dir == traverse_direction_enum::Forward)
        {
        }
        else if (traverse_dir == traverse_direction_enum::Backward)
        {
        }
        else
        {
            throw std::exception("Unreachable");
        }
    }
};

} // namespace one_dimensional

} // namespace lss_pde_solvers

#endif ///_LSS_SAULYEV_SCHEME_HPP_
