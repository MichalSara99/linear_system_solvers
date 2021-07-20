#pragma once
#if !defined(_LSS_CONTAINER_2D_T_HPP_)
#define _LSS_CONTAINER_2D_T_HPP_

#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include "containers/lss_container_2d.hpp"

template <typename T> void testContainer2d1()
{
    using lss_containers::container_2d;
    using lss_utility::sptr_t;

    std::cout << "===================================\n";
    std::cout << "Creating following container:\n";
    std::cout << "     [10, -1, 2, 0] \n";
    std::cout << " A = [-1, 11, -1, 3] \n";
    std::cout << "     [2, -1, 10, -1] \n";
    std::cout << "	 [0, 3, -1, 8]\n";
    std::cout << "\n";

    container_2d<T, std::vector, std::allocator<T>> cont_2d(4, 4);

    cont_2d(0, 0, 10.0);
    cont_2d(0, 1, -1.0);
    cont_2d(0, 2, 2.0);
    cont_2d(0, 3, 0.0);

    cont_2d(1, 0, -1.0);
    cont_2d(1, 1, 11.0);
    cont_2d(1, 2, -1.0);
    cont_2d(1, 3, 3.0);

    cont_2d(2, 0, 2.0);
    cont_2d(2, 1, -1.0);
    cont_2d(2, 2, 10.0);
    cont_2d(2, 3, -1.0);

    cont_2d(3, 0, 0.0);
    cont_2d(3, 1, 3.0);
    cont_2d(3, 2, -1.0);
    cont_2d(3, 3, 8.0);

    for (std::size_t r = 0; r < cont_2d.rows(); ++r)
    {
        for (std::size_t c = 0; c < cont_2d.columns(); ++c)
        {
            std::cout << cont_2d(r, c) << " ";
        }
        std::cout << "\n";
    }

    std::cout << "flat data: \n";
    auto const data = cont_2d.data();
    for (auto const t : data)
    {
        std::cout << t << ", ";
    }
    std::cout << "\n";
    std::cout << "set_data: \n";
    cont_2d.set_data(data.begin(), data.end());
    for (std::size_t r = 0; r < cont_2d.rows(); ++r)
    {
        for (std::size_t c = 0; c < cont_2d.columns(); ++c)
        {
            std::cout << cont_2d(r, c) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

template <typename T> void testContainer2d2()
{
    using lss_containers::container_2d;
    using lss_utility::sptr_t;

    std::cout << "===================================\n";
    std::cout << "Creating following container:\n";
    std::cout << "     [10, -1, 2, 0] \n";
    std::cout << " A = [-1, 11, -1, 3] \n";
    std::cout << "     [2, -1, 10, -1] \n";
    std::cout << "	 [0, 3, -1, 8]\n";
    std::cout << "\n";

    container_2d<T, std::vector, std::allocator<T>> cont_2d(4, 4);

    cont_2d(0, {10, -1, 2, 0});
    cont_2d(1, {-1, 11, -1, 3.0});
    cont_2d(2, {2.0, -1.0, 10.0, -1.0});
    cont_2d(3, {0.0, 3.0, -1.0, 8.0});

    for (std::size_t r = 0; r < cont_2d.rows(); ++r)
    {
        for (std::size_t c = 0; c < cont_2d.columns(); ++c)
        {
            std::cout << cont_2d(r, c) << " ";
        }
        std::cout << "\n";
    }

    std::cout << "flat data: \n";
    auto const data = cont_2d.data();
    for (auto const t : data)
    {
        std::cout << t << ", ";
    }
    std::cout << "\n";
    std::cout << "set_data: \n";
    cont_2d.set_data(data.begin(), data.end());
    for (std::size_t r = 0; r < cont_2d.rows(); ++r)
    {
        for (std::size_t c = 0; c < cont_2d.columns(); ++c)
        {
            std::cout << cont_2d(r, c) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

template <typename T> void testContainer2d3()
{
    using lss_containers::container_2d;
    using lss_utility::sptr_t;

    std::cout << "===================================\n";
    std::cout << "Creating following container:\n";
    std::cout << "     [10, -1, 2, 0] \n";
    std::cout << " A = [-1, 11, -1, 3] \n";
    std::cout << "     [2, -1, 10, -1] \n";
    std::cout << "	 [0, 3, -1, 8]\n";
    std::cout << "\n";

    container_2d<T, std::vector, std::allocator<T>> cont_2d(4, 4);

    cont_2d(0, {10, -1, 2, 0});
    cont_2d(1, {-1, 11, -1, 3.0});
    cont_2d(2, {2.0, -1.0, 10.0, -1.0});
    cont_2d(3, {0.0, 3.0, -1.0, 8.0});

    auto const &fourth = cont_2d(3);

    for (std::size_t c = 0; c < cont_2d.columns(); ++c)
    {
        std::cout << fourth[c] << " ";
    }
    std::cout << "\n";
    std::cout << "flat data: \n";
    auto data = cont_2d.data();
    for (auto const t : data)
    {
        std::cout << t << ", ";
    }
    std::cout << "\n";

    std::cout << "\n";
    std::cout << "set_data: \n";
    cont_2d.set_data(data.begin(), data.end());
    for (std::size_t r = 0; r < cont_2d.rows(); ++r)
    {
        for (std::size_t c = 0; c < cont_2d.columns(); ++c)
        {
            std::cout << cont_2d(r, c) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void testContainer2d()
{
    std::cout << "============================================================\n";
    std::cout << "=================== Testing Container2d ====================\n";
    std::cout << "============================================================\n";

    testContainer2d1<float>();
    testContainer2d1<double>();
    testContainer2d2<float>();
    testContainer2d2<double>();
    testContainer2d3<float>();
    testContainer2d3<double>();

    std::cout << "============================================================\n";
}

template <typename T> void testContainer2dCopy()
{
    using lss_containers::container_2d;
    using lss_utility::sptr_t;

    std::cout << "===================================\n";
    std::cout << "Creating following container:\n";
    std::cout << "     [10, -1, 2, 0] \n";
    std::cout << " A = [-1, 11, -1, 3] \n";
    std::cout << "     [2, -1, 10, -1] \n";
    std::cout << "	 [0, 3, -1, 8]\n";
    std::cout << "\n";

    container_2d<T, std::vector, std::allocator<T>> cont_2d(4, 4);

    cont_2d(0, 0, 10.0);
    cont_2d(0, 1, -1.0);
    cont_2d(0, 2, 2.0);
    cont_2d(0, 3, 0.0);

    cont_2d(1, 0, -1.0);
    cont_2d(1, 1, 11.0);
    cont_2d(1, 2, -1.0);
    cont_2d(1, 3, 3.0);

    cont_2d(2, 0, 2.0);
    cont_2d(2, 1, -1.0);
    cont_2d(2, 2, 10.0);
    cont_2d(2, 3, -1.0);

    cont_2d(3, 0, 0.0);
    cont_2d(3, 1, 3.0);
    cont_2d(3, 2, -1.0);
    cont_2d(3, 3, 8.0);

    container_2d<T, std::vector, std::allocator<T>> copy_cont_2d(4, 4);

    copy(copy_cont_2d, cont_2d);

    for (std::size_t r = 0; r < cont_2d.rows(); ++r)
    {
        for (std::size_t c = 0; c < cont_2d.columns(); ++c)
        {
            std::cout << cont_2d(r, c) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n\ncopy:\n";
    for (std::size_t r = 0; r < copy_cont_2d.rows(); ++r)
    {
        for (std::size_t c = 0; c < copy_cont_2d.columns(); ++c)
        {
            std::cout << copy_cont_2d(r, c) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
    std::cout << "flat data: \n";
    auto const data = cont_2d.data();
    for (auto const t : data)
    {
        std::cout << t << ", ";
    }
    std::cout << "\n";
    std::cout << "set_data: \n";
    auto raw_data = data.data();
    cont_2d.set_data(raw_data, raw_data + cont_2d.total_size());
    for (std::size_t r = 0; r < cont_2d.rows(); ++r)
    {
        for (std::size_t c = 0; c < cont_2d.columns(); ++c)
        {
            std::cout << cont_2d(r, c) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void testCopyContainer2d()
{
    std::cout << "============================================================\n";
    std::cout << "=========== Testing Copying Container2d ====================\n";
    std::cout << "============================================================\n";

    testContainer2dCopy<float>();
    testContainer2dCopy<double>();

    std::cout << "============================================================\n";
}

template <typename T> void testContainer2dPartialCopyRow()
{
    using lss_containers::container_2d;
    using lss_utility::sptr_t;

    std::cout << "===================================\n";
    std::cout << "Creating following container:\n";
    std::cout << "     [10, -1, 2, 0] \n";
    std::cout << " A = [-1, 11, -1, 3] \n";
    std::cout << "     [2, -1, 10, -1] \n";
    std::cout << "	 [0, 3, -1, 8]\n";
    std::cout << "\n";

    container_2d<T, std::vector, std::allocator<T>> cont_2d(4, 4);

    cont_2d(0, 0, 10.0);
    cont_2d(0, 1, -1.0);
    cont_2d(0, 2, 2.0);
    cont_2d(0, 3, 0.0);

    cont_2d(1, 0, -1.0);
    cont_2d(1, 1, 11.0);
    cont_2d(1, 2, -1.0);
    cont_2d(1, 3, 3.0);

    cont_2d(2, 0, 2.0);
    cont_2d(2, 1, -1.0);
    cont_2d(2, 2, 10.0);
    cont_2d(2, 3, -1.0);

    cont_2d(3, 0, 0.0);
    cont_2d(3, 1, 3.0);
    cont_2d(3, 2, -1.0);
    cont_2d(3, 3, 8.0);

    std::vector<T> small_row(2, T(3.1415));
    std::vector<T> another_small_row(12, std::exp(static_cast<T>(1.0)));

    cont_2d(1, small_row, 1, 2);
    cont_2d(2, another_small_row, 1, 3);

    for (std::size_t r = 0; r < cont_2d.rows(); ++r)
    {
        for (std::size_t c = 0; c < cont_2d.columns(); ++c)
        {
            std::cout << cont_2d(r, c) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
    std::cout << "flat data: \n";
    auto const data = cont_2d.data();
    for (auto const t : data)
    {
        std::cout << t << ", ";
    }
    std::cout << "\n";
}

void testPartialCopyRowContainer2d()
{
    std::cout << "============================================================\n";
    std::cout << "=========== Testing Copying Partial Row Container2d ========\n";
    std::cout << "============================================================\n";

    testContainer2dPartialCopyRow<float>();
    testContainer2dPartialCopyRow<double>();

    std::cout << "============================================================\n";
}

#endif ///_LSS_CONTAINER_2D_T_HPP_
