#pragma once
#if !defined(_LSS_MACROS_HPP_)
#define _LSS_MACROS_HPP_

#include <cuda_runtime.h>
#include <cusolverSp.h>

#include <iostream>

#define CUDA_ERROR(value)                                                                                              \
    {                                                                                                                  \
        cudaError_t error = value;                                                                                     \
        if (error != cudaSuccess)                                                                                      \
        {                                                                                                              \
            std::cerr << "File: " << __FILE__ << "\nLine: " << __LINE__ << "\nWhat: \n" << cudaGetErrorString(error);  \
        }                                                                                                              \
    }

#define CUSOLVER_STATUS(value)                                                                                         \
    {                                                                                                                  \
        cusolverStatus_t status = value;                                                                               \
        if (status != CUSOLVER_STATUS_SUCCESS)                                                                         \
        {                                                                                                              \
            std::cerr << "File: " << __FILE__ << "\nLine: " << __LINE__ << "\n";                                       \
        }                                                                                                              \
    }

#define CUBLAS_STATUS(value)                                                                                           \
    {                                                                                                                  \
        cublasStatus_t status = value;                                                                                 \
        if (status != CUBLAS_STATUS_SUCCESS)                                                                           \
        {                                                                                                              \
            std::cerr << "File: " << __FILE__ << "\nLine: " << __LINE__ << "\n";                                       \
        }                                                                                                              \
    }

#define LSS_ASSERT(condition, message)                                                                                 \
    {                                                                                                                  \
        do                                                                                                             \
        {                                                                                                              \
            if (!(condition))                                                                                          \
            {                                                                                                          \
                std::cerr << "Assertion `" #condition "` failed in " << __FILE__ << " line " << __LINE__ << ": "       \
                          << message << std::endl;                                                                     \
                std::terminate();                                                                                      \
            }                                                                                                          \
        } while (false);                                                                                               \
    }

#define LSS_VERIFY(variable, message)                                                                                  \
    {                                                                                                                  \
        LSS_ASSERT(variable, message);                                                                                 \
    }

#endif ///_LSS_MACROS_HPP_
