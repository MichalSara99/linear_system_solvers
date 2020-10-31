#include<device_launch_parameters.h>
#include"lss_pde_cuda_kernels.h"
#include"lss_utility.h"
#include"lss_one_dim_pde_schemes_cuda.h"




namespace lss_one_dim_pde_schemes_cuda {

	using lss_pde_cuda_kernels::explicitEulerIterate1D;
	using lss_pde_cuda_kernels::fillDirichletBC1D;
	using lss_pde_cuda_kernels::fillRobinBC1D;
	using lss_utility::swap;

	void ExplicitEulerLoopSP::operator()(float const *input, std::pair<float, float> const &boundaryPair,
										unsigned long long const size, float *solution)const {
		// prepare pointers on device:
		float *d_prev = NULL;
		float *d_next = NULL;
		// allocate block of memory on device:
		cudaMalloc((void**)&d_prev, size * sizeof(float));
		cudaMalloc((void**)&d_next, size * sizeof(float));
		// copy contents of input to d_prev:
		cudaMemcpy(d_prev, input, size * sizeof(float),
			cudaMemcpyKind::cudaMemcpyHostToDevice);

		unsigned int const threadsPerBlock = THREADS_PER_BLOCK;
		unsigned int const blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

		float time = timeStep_;
		float k = timeStep_;
		float left = boundaryPair.first;
		float right = boundaryPair.second;
		while (time <= terminalT_) {
			// populate new solution in d_next:
			explicitEulerIterate1D<float><<<threadsPerBlock, blocksPerGrid>>>(d_prev, d_next, lambda_,gamma_, size);
			// fill in the dirichlet boundaries in d_next:
			fillDirichletBC1D<float><<<threadsPerBlock, blocksPerGrid>>>(d_next, left, right, size);
			// swap the two pointers:
			swap(d_prev, d_next);
			time += k;
		}
		// copy the contents of d_next to the solution pointer:
		cudaMemcpy(solution, d_prev, size * sizeof(float),
			cudaMemcpyKind::cudaMemcpyDeviceToHost);
		cudaFree(d_prev);
		cudaFree(d_next);
	}

	void ExplicitEulerLoopDP::operator()(double const *input, std::pair<double, double> const &boundaryPair,
										unsigned long long const size, double *solution)const {
		// prepare pointers on device:
		double *d_prev = NULL;
		double *d_next = NULL;
		// allocate block of memory on device:
		cudaMalloc((void**)&d_prev, size * sizeof(double));
		cudaMalloc((void**)&d_next, size * sizeof(double));
		// copy contents of input to d_prev:
		cudaMemcpy(d_prev, input, size * sizeof(double),
			cudaMemcpyKind::cudaMemcpyHostToDevice);

		unsigned int const threadsPerBlock = THREADS_PER_BLOCK;
		unsigned int const blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

		double time = timeStep_;
		double k = timeStep_;
		double left = boundaryPair.first;
		double right = boundaryPair.second;
		while (time <= terminalT_) {
			// populate new solution in d_next:
			explicitEulerIterate1D<double><<<threadsPerBlock, blocksPerGrid>>>(d_prev, d_next, lambda_,gamma_, size);
			// fill in the dirichlet boundaries in d_next:
			fillDirichletBC1D<double><<<threadsPerBlock, blocksPerGrid>>>(d_next, left, right, size);
			// swap the two pointers:
			swap(d_prev, d_next);
			time += k;
		}
		// copy the contents of d_next to the solution pointer:
		cudaMemcpy(solution, d_prev, size * sizeof(double),
			cudaMemcpyKind::cudaMemcpyDeviceToHost);
		cudaFree(d_prev);
		cudaFree(d_next);
	}
	


	void ExplicitEulerLoopSP::operator()(float const *input, std::pair<float, float> const &leftPair,
		std::pair<float, float> const &rightPair, unsigned long long const size, float *solution)const {

		// prepare pointers on device:
		float *d_prev = NULL;
		float *d_next = NULL;
		// allocate block of memory on device:
		cudaMalloc((void**)&d_prev, size * sizeof(float));
		cudaMalloc((void**)&d_next, size * sizeof(float));
		// copy contents of input to d_prev:
		cudaMemcpy(d_prev, input, size * sizeof(float),
			cudaMemcpyKind::cudaMemcpyHostToDevice);

		unsigned int const threadsPerBlock = THREADS_PER_BLOCK;
		unsigned int const blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

		float time = timeStep_;
		float k = timeStep_;
		float leftLinear = leftPair.first;
		float leftConst = leftPair.second;
		float rightLinear = rightPair.first;
		float rightConst = rightPair.second;
		while (time <= terminalT_) {
			// populate new solution in d_next:
			explicitEulerIterate1D<float> << <threadsPerBlock, blocksPerGrid >> >(d_prev, d_next, lambda_,gamma_, size);
			// fill in the dirichlet boundaries in d_next:
			fillRobinBC1D<float><<<threadsPerBlock, blocksPerGrid>>>(d_next, lambda_,gamma_, leftLinear, leftConst, rightLinear, rightConst, size);
			// swap the two pointers:
			swap(d_prev, d_next);
			time += k;
		}
		// copy the contents of d_next to the solution pointer:
		cudaMemcpy(solution, d_prev, size * sizeof(float),
			cudaMemcpyKind::cudaMemcpyDeviceToHost);
		cudaFree(d_prev);
		cudaFree(d_next);
	}


	void ExplicitEulerLoopDP::operator()(double const *input, std::pair<double, double> const &leftPair,
		std::pair<double, double> const &rightPair, unsigned long long const size, double *solution)const {
		// prepare pointers on device:
		double *d_prev = NULL;
		double *d_next = NULL;
		// allocate block of memory on device:
		cudaMalloc((void**)&d_prev, size * sizeof(double));
		cudaMalloc((void**)&d_next, size * sizeof(double));
		// copy contents of input to d_prev:
		cudaMemcpy(d_prev, input, size * sizeof(double),
			cudaMemcpyKind::cudaMemcpyHostToDevice);

		unsigned int const threadsPerBlock = THREADS_PER_BLOCK;
		unsigned int const blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

		double time = timeStep_;
		double k = timeStep_;
		double leftLinear = leftPair.first;
		double leftConst = leftPair.second;
		double rightLinear = rightPair.first;
		double rightConst = rightPair.second;
		while (time <= terminalT_) {
			// populate new solution in d_next:
			explicitEulerIterate1D<double><<<threadsPerBlock, blocksPerGrid>>>(d_prev, d_next, lambda_,gamma_, size);
			// fill in the dirichlet boundaries in d_next:
			fillRobinBC1D<double><<<threadsPerBlock, blocksPerGrid>>>(d_next, lambda_,gamma_, leftLinear, leftConst, rightLinear, rightConst, size);
			// swap the two pointers:
			swap(d_prev, d_next);
			time += k;
		}
		// copy the contents of d_next to the solution pointer:
		cudaMemcpy(solution, d_prev, size * sizeof(double),
			cudaMemcpyKind::cudaMemcpyDeviceToHost);
		cudaFree(d_prev);
		cudaFree(d_next);

	}


}
