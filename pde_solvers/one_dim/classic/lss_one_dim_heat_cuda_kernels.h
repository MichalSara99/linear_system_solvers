#pragma once
#if !defined(_LSS_ONE_DIM_HEAT_CUDA_KERNELS)
#define _LSS_ONE_DIM_HEAT_CUDA_KERNELS

#include<device_launch_parameters.h>

#define THREADS_PER_BLOCK 256



namespace lss_one_dim_heat_cuda_kernels {


	template<typename T>
	__global__
	void fillDirichletBC1D(T *solution,T left,T right, unsigned long long size) {
		unsigned long long const tid = blockDim.x*blockIdx.x + threadIdx.x;
		if (tid >= size)return;
		if (tid == 0)
			solution[tid] = left;
		if (tid == (size - 1))
			solution[tid] = right;
	}

	template<typename T>
	__global__
	void fillRobinBC1D(T *solution, T lambda,T gamma,T delta,
						T leftLinear, T leftConst, T rightLinear, T rightConst, unsigned long long size) {
		unsigned long long const tid = blockIdx.x*blockDim.x + threadIdx.x;
		if (tid >= size)return;
		if (tid == 0)
			solution[tid] = (leftLinear * (lambda - gamma) + (lambda + gamma)) * solution[tid + 1] +
			(1.0 - (2.0*lambda - delta))*solution[tid] +
			(lambda - gamma)* leftConst;
		if (tid == (size - 1))
			solution[tid] = (rightLinear * (lambda + gamma) + (lambda - gamma)) * solution[tid - 1] +
			(1.0 - (2.0*lambda - delta))*solution[tid] +
			(lambda + gamma) * rightConst;
	}

	template<typename T>
	__global__
		void fillRobinBC1D(T *solution, T sourceLeft, T sourceRight, T lambda, T gamma, T delta, T timeStep,
							T leftLinear, T leftConst, T rightLinear, T rightConst, unsigned long long size) {
		unsigned long long const tid = blockIdx.x*blockDim.x + threadIdx.x;
		if (tid >= size)return;
		if (tid == 0)
			solution[tid] = (leftLinear * (lambda - gamma) + (lambda + gamma)) * solution[tid + 1] +
			(1.0 - (2.0*lambda - delta))*solution[tid] +
			(lambda - gamma)* leftConst +
			timeStep * sourceLeft;
		if (tid == (size - 1))
			solution[tid] = (rightLinear * (lambda + gamma) + (lambda - gamma)) * solution[tid - 1] +
			(1.0 - (2.0*lambda - delta))*solution[tid] +
			(lambda + gamma) * rightConst +
			timeStep * sourceRight;
	}


	// Euler 1D kernel without source:
	template<typename T>
	__global__
	void explicitEulerIterate1D(T* prev, T* next, T lambda,T gamma,T delta, unsigned long long size) {
		unsigned long long const tid = blockDim.x * blockIdx.x + threadIdx.x;
		if (tid >= size)return;
		if (tid == 0)return;
		if (tid == (size - 1))return;
		next[tid] = (lambda + gamma) * prev[tid + 1] +
			(1.0 - (2.0*lambda - delta))*prev[tid] +
			(lambda - gamma)* prev[tid - 1];
	}

	// Euler 1D kernel with source and timeStep:
	template<typename T>
	__global__
		void explicitEulerIterate1D(T* prev, T* next, T* source, T lambda, T gamma,T delta, T timeStep, unsigned long long size) {
		unsigned long long const tid = blockDim.x * blockIdx.x + threadIdx.x;
		if (tid >= size)return;
		if (tid == 0)return;
		if (tid == (size - 1))return;
		next[tid] = (lambda + gamma) * prev[tid + 1] +
			(1.0 - (2.0*lambda - delta))*prev[tid] +
			(lambda - gamma)* prev[tid - 1] +
			timeStep * source[tid];
	}







}


#endif ///_LSS_ONE_DIM_HEAT_CUDA_KERNELS