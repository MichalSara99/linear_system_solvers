#pragma once
#if !defined(_LSS_PDE_CUDA_KERNELS)
#define _LSS_PDE_CUDA_KERNELS

#include<device_launch_parameters.h>

namespace lss_pde_cuda_kernels {


	template<typename T>
	__global__
		void explicitEulerIterate1D(T* prev, T* next, T lambda, unsigned long long size) {
		unsigned long long const t_id = blockDim.x * blockIdx.x + threadIdx.x;
		if (t_id >= size)return;
		if (t_id == 0)return;
		if (t_id == (size - 1))return;
		next[t_id] = lambda * prev[t_id + 1] +
			(1.0 - 2.0*lambda)*prev[t_id] +
			lambda * prev[t_id - 1];
	}






}


#endif ///_LSS_PDE_CUDA_KERNELS