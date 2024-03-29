#pragma once
#if !defined(_LSS_UTILITY_HPP_)
#define _LSS_UTILITY_HPP_

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <tuple>

namespace lss_utility
{

// =========================================================================
// ================================ pi =====================================
// =========================================================================
template <typename T> static T const pi()
{
    return (static_cast<T>(3.14159265358979));
}

// ==========================================================================
// ================================= NaN ====================================
// ==========================================================================

template <typename T> static constexpr T NaN()
{
    return std::numeric_limits<T>::quiet_NaN();
}

// ==========================================================================
// =========================== smart pointer aliases ========================
// ==========================================================================

template <typename T> using sptr_t = std::shared_ptr<T>;

template <typename T> using uptr_t = std::unique_ptr<T>;

// ==========================================================================
// ==================================== Swap ================================
// ==========================================================================

template <typename T> void swap(T &a, T &b)
{
    T t = a;
    a = b;
    b = t;
}

// ==========================================================================
// ================================ norm_cdf ================================
// ==========================================================================

template <typename T> double norm_cdf(T x)
{
    T ind = T{};
    if (x <= 0.0)
        ind = 1.0;
    x = std::abs(x);
    T const cst = 1.0 / (std::sqrt(2.0 * pi<T>()));
    T const first = std::exp(-0.5 * x * x);
    T const second = 0.226 + 0.64 * x + 0.33 * std::sqrt(x * x + 3.0);
    T const res = 1.0 - ((first / second) * cst);
    return std::abs(ind - res);
}

// ==========================================================================
// =========================== black_scholes_exact ==========================
// ==========================================================================
template <typename T> class black_scholes_exact
{
  private:
    T time_;
    T strike_;
    T vol_;
    T maturity_;
    T rate_;

  protected:
    explicit black_scholes_exact(){};

  public:
    explicit black_scholes_exact(T time, T strike, T rate, T volatility, T maturity)
        : time_{time}, strike_{strike}, rate_{rate}, vol_{volatility}, maturity_{maturity}
    {
    }

    ~black_scholes_exact()
    {
    }

    T call(T spot) const
    {
        T const tau = maturity_ - time_;
        T const s_tau = std::sqrt(tau);
        T const d_1 = (std::log(spot / strike_) + (rate_ + 0.5 * vol_ * vol_) * tau) / (vol_ * s_tau);
        T const d_2 = d_1 - vol_ * s_tau;
        T const result = norm_cdf(d_1) * spot - (norm_cdf(d_2) * strike_ * std::exp(-rate_ * tau));
        return result;
    }

    T call(T spot, T time_to_maturity) const
    {
        T const tau = time_to_maturity;
        T const s_tau = std::sqrt(tau);
        T const d_1 = (std::log(spot / strike_) + (rate_ + 0.5 * vol_ * vol_) * tau) / (vol_ * s_tau);
        T const d_2 = d_1 - vol_ * s_tau;
        T const result = norm_cdf(d_1) * spot - (norm_cdf(d_2) * strike_ * std::exp(-rate_ * tau));
        return result;
    }

    T put(T spot) const
    {
        T const call_p = call(spot);
        T const tau = maturity_ - time_;
        return (strike_ * std::exp(-rate_ * tau) - spot + call_p);
    }

    T put(T spot, T time_to_maturity) const
    {
        T const call_p = call(spot, time_to_maturity);
        T const tau = time_to_maturity;
        return (strike_ * std::exp(-rate_ * tau) - spot + call_p);
    }
};

// ==========================================================================
// ==================================== range ===============================
// ==========================================================================
template <typename T> class range
{
  private:
    T l_, u_;

  public:
    explicit range(T lower, T upper) : l_{lower}, u_{upper}
    {
    }
    explicit range() : range(T{}, T{})
    {
    }

    ~range()
    {
    }

    range(range const &copy) : l_{copy.l_}, u_{copy.u_}
    {
    }
    range(range &&other) noexcept : l_{std::move(other.l_)}, u_{std::move(other.u_)}
    {
    }

    range &operator=(range const &copy)
    {
        if (this != &copy)
        {
            l_ = copy.l_;
            u_ = copy.u_;
        }
        return *this;
    }

    range &operator=(range &&other) noexcept
    {
        if (this != &other)
        {
            l_ = std::move(other.l_);
            u_ = std::move(other.u_);
        }
        return *this;
    }

    inline T lower() const
    {
        return l_;
    }
    inline T upper() const
    {
        return u_;
    }
    inline T spread() const
    {
        return (u_ - l_);
    }
    inline T mid_point() const
    {
        return 0.5 * (l_ + u_);
    }
};

template <typename fp_type> using range_ptr = sptr_t<range<fp_type>>;

} // namespace lss_utility

#endif ///_LSS_UTILITY_HPP_
