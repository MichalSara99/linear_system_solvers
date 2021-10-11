#if !defined(_LSS_WEIGHTED_SCHEME_CONFIG_HPP_)
#define _LSS_WEIGHTED_SCHEME_CONFIG_HPP_

#include <map>

#include "common/lss_enumerations.hpp"
#include "common/lss_macros.hpp"
#include "common/lss_utility.hpp"

namespace lss_pde_solvers
{

using lss_utility::sptr_t;

/**
    weighted_scheme_config structure
 */
template <typename fp_type> struct weighted_scheme_config
{
  private:
    fp_type start_x_val_;
    fp_type start_y_val_;
    fp_type weight_x_dir_;
    fp_type weight_y_dir_;

    void initialize()
    {
        const fp_type zero = static_cast<fp_type>(0.0);
        const fp_type one = static_cast<fp_type>(1.0);
        LSS_ASSERT((weight_x_dir_ >= zero) && (weight_x_dir_ <= one), "Weight weight_x must be in range 0 to 1");
        LSS_ASSERT((weight_y_dir_ >= zero) && (weight_y_dir_ <= one), "Weight weight_y must be in range 0 to 1");
    }

  public:
    explicit weighted_scheme_config(fp_type start_x_val = fp_type(0.0), fp_type start_y_val = fp_type(0.0),
                                    fp_type weight_x = fp_type(1.0), fp_type weight_y = fp_type(1.0))
        : start_x_val_{start_x_val}, start_y_val_{start_y_val}, weight_x_dir_{weight_x}, weight_y_dir_{weight_y}
    {
        initialize();
    }
    ~weighted_scheme_config()
    {
    }

    inline fp_type start_x_value() const
    {
        return start_x_val_;
    }

    inline fp_type start_y_value() const
    {
        return start_y_val_;
    }

    inline fp_type weight_x() const
    {
        return weight_x_dir_;
    }

    inline fp_type weight_y() const
    {
        return weight_y_dir_;
    }
};

template <typename fp_type> using weighted_scheme_config_ptr = sptr_t<weighted_scheme_config<fp_type>>;

} // namespace lss_pde_solvers

#endif ///_LSS_WEIGHTED_SCHEME_CONFIG_HPP_
