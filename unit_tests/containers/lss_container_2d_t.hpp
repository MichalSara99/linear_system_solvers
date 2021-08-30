#pragma once
#if !defined(_LSS_CONTAINER_2D_T_HPP_)
#define _LSS_CONTAINER_2D_T_HPP_

#include "common/lss_enumerations.hpp"
#include "common/lss_utility.hpp"
#include "containers/lss_container_2d.hpp"

template <typename T> void testContainer2d1()
{
    using lss_containers::container_2d;
    using lss_enumerations::by_enum;
    using lss_utility::sptr_t;

    typedef container_2d<by_enum::Row, T, std::vector, std::allocator<T>> rcontainer_2d_t;
    typedef container_2d<by_enum::Column, T, std::vector, std::allocator<T>> ccontainer_2d_t;

    std::cout << "===================================\n";
    std::cout << "Creating following container:\n";
    std::cout << "     [10, -1, 2, 0] \n";
    std::cout << " A = [-1, 11, -1, 3] \n";
    std::cout << "     [2, -1, 10, -1] \n";
    std::cout << "	 [0, 3, -1, 8]\n";
    std::cout << "\n";

    rcontainer_2d_t rcont_2d(4, 4);

    rcont_2d(0, 0, 10.0);
    rcont_2d(0, 1, -1.0);
    rcont_2d(0, 2, 2.0);
    rcont_2d(0, 3, 0.0);

    rcont_2d(1, 0, -1.0);
    rcont_2d(1, 1, 11.0);
    rcont_2d(1, 2, -1.0);
    rcont_2d(1, 3, 3.0);

    rcont_2d(2, 0, 2.0);
    rcont_2d(2, 1, -1.0);
    rcont_2d(2, 2, 10.0);
    rcont_2d(2, 3, -1.0);

    rcont_2d(3, 0, 0.0);
    rcont_2d(3, 1, 3.0);
    rcont_2d(3, 2, -1.0);
    rcont_2d(3, 3, 8.0);

    for (std::size_t r = 0; r < rcont_2d.rows(); ++r)
    {
        for (std::size_t c = 0; c < rcont_2d.columns(); ++c)
        {
            std::cout << rcont_2d(r, c) << " ";
        }
        std::cout << "\n";
    }

    std::cout << "flat data: \n";
    auto const data = rcont_2d.data();
    for (auto const t : data)
    {
        std::cout << t << ", ";
    }

    std::cout << "\n\ncopy row-wise container to column-wise container:\n";
    ccontainer_2d_t ccont_2d(rcont_2d);
    for (std::size_t r = 0; r < ccont_2d.rows(); ++r)
    {
        for (std::size_t c = 0; c < ccont_2d.columns(); ++c)
        {
            std::cout << ccont_2d(r, c) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

template <typename T> void testContainer2d2()
{
    using lss_containers::container_2d;
    using lss_enumerations::by_enum;
    using lss_utility::sptr_t;

    typedef container_2d<by_enum::Row, T, std::vector, std::allocator<T>> rcontainer_2d_t;
    typedef container_2d<by_enum::Column, T, std::vector, std::allocator<T>> ccontainer_2d_t;

    std::cout << "===================================\n";
    std::cout << "Creating following container:\n";
    std::cout << "     [10, -1, 2, 0] \n";
    std::cout << " A = [-1, 11, -1, 3] \n";
    std::cout << "     [2, -1, 10, -1] \n";
    std::cout << "	 [0, 3, -1, 8]\n";
    std::cout << "\n";

    rcontainer_2d_t rcont_2d(4, 4);

    rcont_2d(0, {10, -1, 2, 0});
    rcont_2d(1, {-1, 11, -1, 3.0});
    rcont_2d(2, {2.0, -1.0, 10.0, -1.0});
    rcont_2d(3, {0.0, 3.0, -1.0, 8.0});

    for (std::size_t r = 0; r < rcont_2d.rows(); ++r)
    {
        for (std::size_t c = 0; c < rcont_2d.columns(); ++c)
        {
            std::cout << rcont_2d(r, c) << " ";
        }
        std::cout << "\n";
    }

    std::cout << "flat data: \n";
    auto const data = rcont_2d.data();
    for (auto const t : data)
    {
        std::cout << t << ", ";
    }
    std::cout << "\n\ncopy row-wise container to column-wise container:\n";
    ccontainer_2d_t ccont_2d(rcont_2d);
    for (std::size_t r = 0; r < ccont_2d.rows(); ++r)
    {
        for (std::size_t c = 0; c < ccont_2d.columns(); ++c)
        {
            std::cout << ccont_2d(r, c) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

template <typename T> void testContainer2d3()
{
    using lss_containers::container_2d;
    using lss_enumerations::by_enum;
    using lss_utility::sptr_t;

    typedef container_2d<by_enum::Row, T, std::vector, std::allocator<T>> rcontainer_2d_t;
    typedef container_2d<by_enum::Column, T, std::vector, std::allocator<T>> ccontainer_2d_t;

    std::cout << "===================================\n";
    std::cout << "Creating following container:\n";
    std::cout << "     [10, -1, 2, 0] \n";
    std::cout << " A = [-1, 11, -1, 3] \n";
    std::cout << "     [2, -1, 10, -1] \n";
    std::cout << "	 [0, 3, -1, 8]\n";
    std::cout << "\n";

    ccontainer_2d_t ccont_2d(4, 4);

    ccont_2d(0, {10, -1, 2, 0});
    ccont_2d(1, {-1, 11, -1, 3.0});
    ccont_2d(2, {2.0, -1.0, 10.0, -1.0});
    ccont_2d(3, {0.0, 3.0, -1.0, 8.0});

    auto const &fourth = ccont_2d(3);

    for (std::size_t c = 0; c < ccont_2d.columns(); ++c)
    {
        std::cout << fourth[c] << " ";
    }
    std::cout << "\n";
    std::cout << "flat data: \n";
    auto data = ccont_2d.data();
    for (auto const t : data)
    {
        std::cout << t << ", ";
    }

    std::cout << "\n\ncopy column-wise container to row-wise container:\n";
    rcontainer_2d_t rcont_2d(ccont_2d);
    for (std::size_t r = 0; r < rcont_2d.rows(); ++r)
    {
        for (std::size_t c = 0; c < rcont_2d.columns(); ++c)
        {
            std::cout << rcont_2d(r, c) << " ";
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
    using lss_enumerations::by_enum;
    using lss_utility::sptr_t;

    typedef container_2d<by_enum::Row, T, std::vector, std::allocator<T>> rcontainer_2d_t;
    typedef container_2d<by_enum::Column, T, std::vector, std::allocator<T>> ccontainer_2d_t;

    std::cout << "===================================\n";
    std::cout << "Creating following container:\n";
    std::cout << "     [10, -1, 2, 0] \n";
    std::cout << " A = [-1, 11, -1, 3] \n";
    std::cout << "     [2, -1, 10, -1] \n";
    std::cout << "	 [0, 3, -1, 8]\n";
    std::cout << "\n";

    ccontainer_2d_t ccont_2d(4, 4);

    ccont_2d(0, 0, 10.0);
    ccont_2d(0, 1, -1.0);
    ccont_2d(0, 2, 2.0);
    ccont_2d(0, 3, 0.0);

    ccont_2d(1, 0, -1.0);
    ccont_2d(1, 1, 11.0);
    ccont_2d(1, 2, -1.0);
    ccont_2d(1, 3, 3.0);

    ccont_2d(2, 0, 2.0);
    ccont_2d(2, 1, -1.0);
    ccont_2d(2, 2, 10.0);
    ccont_2d(2, 3, -1.0);

    ccont_2d(3, 0, 0.0);
    ccont_2d(3, 1, 3.0);
    ccont_2d(3, 2, -1.0);
    ccont_2d(3, 3, 8.0);

    ccontainer_2d_t copy_ccont_2d(4, 4);

    copy(copy_ccont_2d, ccont_2d);

    for (std::size_t r = 0; r < ccont_2d.rows(); ++r)
    {
        for (std::size_t c = 0; c < ccont_2d.columns(); ++c)
        {
            std::cout << ccont_2d(r, c) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n\ncopy:\n";
    for (std::size_t r = 0; r < copy_ccont_2d.rows(); ++r)
    {
        for (std::size_t c = 0; c < copy_ccont_2d.columns(); ++c)
        {
            std::cout << copy_ccont_2d(r, c) << " ";
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

template <typename T> void testContainer2d4()
{
    using lss_containers::container_2d;
    using lss_enumerations::by_enum;
    using lss_utility::sptr_t;

    typedef container_2d<by_enum::Row, T, std::vector, std::allocator<T>> rcontainer_2d_t;
    typedef container_2d<by_enum::Column, T, std::vector, std::allocator<T>> ccontainer_2d_t;

    std::cout << "===================================\n";
    std::cout << "Creating following container:\n";
    std::cout << "     [10, -1, 2] \n";
    std::cout << " A = [-1, 11, -1] \n";
    std::cout << "     [2, -1, 10] \n";
    std::cout << "	 [0, 3, -1]\n";
    std::cout << "\n";

    rcontainer_2d_t rcont_2d(4, 3);

    rcont_2d(0, 0, 10.0);
    rcont_2d(0, 1, -1.0);
    rcont_2d(0, 2, 2.0);

    rcont_2d(1, 0, -1.0);
    rcont_2d(1, 1, 11.0);
    rcont_2d(1, 2, -1.0);

    rcont_2d(2, 0, 2.0);
    rcont_2d(2, 1, -1.0);
    rcont_2d(2, 2, 10.0);

    rcont_2d(3, 0, 0.0);
    rcont_2d(3, 1, 3.0);
    rcont_2d(3, 2, -1.0);

    for (std::size_t r = 0; r < rcont_2d.rows(); ++r)
    {
        for (std::size_t c = 0; c < rcont_2d.columns(); ++c)
        {
            std::cout << rcont_2d(r, c) << " ";
        }
        std::cout << "\n";
    }

    std::cout << "flat data: \n";
    auto const data = rcont_2d.data();
    for (auto const t : data)
    {
        std::cout << t << ", ";
    }

    std::cout << "\n\ncopy row-wise container to column-wise container:\n";
    ccontainer_2d_t ccont_2d(rcont_2d);
    for (std::size_t r = 0; r < ccont_2d.rows(); ++r)
    {
        for (std::size_t c = 0; c < ccont_2d.columns(); ++c)
        {
            std::cout << ccont_2d(r, c) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void testNotSymetricalContainer2d()
{
    std::cout << "============================================================\n";
    std::cout << "=========== Testing non-diagonal Container2d ===============\n";
    std::cout << "============================================================\n";

    testContainer2d4<float>();
    testContainer2d4<double>();

    std::cout << "============================================================\n";
}

#endif ///_LSS_CONTAINER_2D_T_HPP_
