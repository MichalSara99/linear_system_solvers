#if !defined(_LSS_SPLITTING_METHOD_CONFIG_HPP_)
#define _LSS_SPLITTING_METHOD_CONFIG_HPP_

#include <map>

#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"

namespace lss_pde_solvers
{

using lss_enumerations::splitting_method_enum;
using lss_utility::sptr_t;

/**
    splitting_method_config structure
 */
template <typename fp_type> struct splitting_method_config
{
  private:
    splitting_method_enum splitting_method_;
    fp_type weighting_value_;

    explicit splitting_method_config() = delete;

  public:
    explicit splitting_method_config(splitting_method_enum splittitng_method, fp_type weighting_value = fp_type(0.5))
        : splitting_method_{splittitng_method}, weighting_value_{weighting_value}
    {
    }
    ~splitting_method_config()
    {
    }

    inline splitting_method_enum splitting_method() const
    {
        return splitting_method_;
    }

    inline fp_type weighting_value() const
    {
        return weighting_value_;
    }
};

template <typename fp_type> using splitting_method_config_ptr = sptr_t<splitting_method_config<fp_type>>;

} // namespace lss_pde_solvers

#endif ///_LSS_SPLITTING_METHOD_CONFIG_HPP_
