#if !defined(_LSS_SPLITTING_METHOD_CONFIG_BUILDER_HPP_)
#define _LSS_SPLITTING_METHOD_CONFIG_BUILDER_HPP_

#include <functional>

#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "common/lss_utility.hpp"
#include "pde_solvers/lss_splitting_method_config.hpp"

namespace lss_pde_solvers
{

using lss_enumerations::dimension_enum;
using lss_enumerations::splitting_method_enum;
using lss_utility::range;
using lss_utility::sptr_t;

// ============================================================================
// ==================== splitting_method_config_builder =======================
// ============================================================================

/**
   splitting_method_config_builder structure
 */
template <typename fp_type> struct splitting_method_config_builder
{
  private:
    splitting_method_enum splitting_method_;
    fp_type weighting_value_;

  public:
    splitting_method_config_builder &splitting_method(splitting_method_enum splitting_method)
    {
        splitting_method_ = splitting_method;
        return *this;
    }

    splitting_method_config_builder &weighting_value(fp_type value)
    {
        weighting_value_ = value;
        return *this;
    }

    splitting_method_config_ptr<fp_type> build()
    {
        return std::make_shared<splitting_method_config<fp_type>>(splitting_method_, weighting_value_);
    }
};

template <typename fp_type>
using splitting_method_config_builder_ptr = sptr_t<splitting_method_config_builder<fp_type>>;

} // namespace lss_pde_solvers

#endif ///_LSS_SPLITTING_METHOD_CONFIG_BUILDER_HPP_
