#include<device_launch_parameters.h>
#include"lss_pde_cuda_kernels.h"
#include<pde_solvers/one_dim/lss_one_dim_pde_schemes_cuda.h>




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
		// unpack the deltas and PDE coefficients:
		float const k = std::get<0>(deltas_);
		float const h = std::get<1>(deltas_);
		float const A = std::get<0>(coeffs_);
		float const B = std::get<1>(coeffs_);
		float const C = std::get<2>(coeffs_);
		// calculate scheme coefficients:
		float const lambda = (A*k) / (h*h);
		float const gamma = (B*k)/(2.0f*h);
		float const delta = (C*k);
		// store bc:
		float const left = boundaryPair.first;
		float const right = boundaryPair.second;

		float time = k;

		if (isSourceSet_) {
			// prepare a pointer for source on device:
			float *d_source = NULL;
			// allocate block memory on device:
			cudaMalloc((void**)&d_source, size * sizeof(float));
			// create vector on host:
			std::vector<float> h_source(size, 0.0f);
			// source is zero:
			while (time <= terminalT_) {
				// discretize source function on host:
				discretizeInSpace(h,spaceStart_, time, source_, h_source);
				// copy h_source contents to d_source (host => device ):
				cudaMemcpy(d_source, h_source.data(), size * sizeof(float),
					cudaMemcpyKind::cudaMemcpyHostToDevice);
				// populate new solution in d_next:
				explicitEulerIterate1D<float> << <threadsPerBlock, blocksPerGrid >> > (d_prev, d_next, d_source, lambda, gamma, delta, k, size);
				// fill in the dirichlet boundaries in d_next:
				fillDirichletBC1D<float> << <threadsPerBlock, blocksPerGrid >> >(d_next, left, right, size);
				// swap the two pointers:
				swap(d_prev, d_next);
				time += k;
			}
			// free allocated memory blocks on device:
			cudaFree(d_source);
		}
		else {
			// source is zero:
			while (time <= terminalT_) {
				// populate new solution in d_next:
				explicitEulerIterate1D<float> << <threadsPerBlock, blocksPerGrid >> >(d_prev, d_next, lambda, gamma, delta, size);
				// fill in the dirichlet boundaries in d_next:
				fillDirichletBC1D<float> << <threadsPerBlock, blocksPerGrid >> >(d_next, left, right, size);
				// swap the two pointers:
				swap(d_prev, d_next);
				time += k;
			}
		}
		// copy the contents of d_next to the solution pointer:
		cudaMemcpy(solution, d_prev, size * sizeof(float),
			cudaMemcpyKind::cudaMemcpyDeviceToHost);
		// free allocated memory blocks on device:
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
		// unpack the deltas and PDE coefficients:
		double const k = std::get<0>(deltas_);
		double const h = std::get<1>(deltas_);
		double const A = std::get<0>(coeffs_);
		double const B = std::get<1>(coeffs_);
		double const C = std::get<2>(coeffs_);
		// calculate scheme coefficients:
		double const lambda = (A*k) / (h*h);
		double const gamma = (B*k) / (2.0f*h);
		double const delta = (C*k);
		// store bc:
		double const left = boundaryPair.first;
		double const right = boundaryPair.second;

		double time = k;

		if (isSourceSet_) {
			// prepare a pointer for source on device:
			double *d_source = NULL;
			// allocate block memory on device:
			cudaMalloc((void**)&d_source, size * sizeof(double));
			// create vector on host:
			std::vector<double> h_source(size, 0.0);
			// source is zero:
			while (time <= terminalT_) {
				// discretize source function on host:
				discretizeInSpace(h, spaceStart_, time, source_, h_source);
				// copy h_source contents to d_source (host => device ):
				cudaMemcpy(d_source, h_source.data(), size * sizeof(double),
					cudaMemcpyKind::cudaMemcpyHostToDevice);
				// populate new solution in d_next:
				explicitEulerIterate1D<double> << <threadsPerBlock, blocksPerGrid >> >(d_prev, d_next, d_source, lambda, gamma, delta, k, size);
				// fill in the dirichlet boundaries in d_next:
				fillDirichletBC1D<double> << <threadsPerBlock, blocksPerGrid >> >(d_next, left, right, size);
				// swap the two pointers:
				swap(d_prev, d_next);
				time += k;
			}
			// free allocated memory blocks on device:
			cudaFree(d_source);
		}
		else {
			while (time <= terminalT_) {
				// populate new solution in d_next:
				explicitEulerIterate1D<double> << <threadsPerBlock, blocksPerGrid >> >(d_prev, d_next, lambda, gamma, delta, size);
				// fill in the dirichlet boundaries in d_next:
				fillDirichletBC1D<double> << <threadsPerBlock, blocksPerGrid >> >(d_next, left, right, size);
				// swap the two pointers:
				swap(d_prev, d_next);
				time += k;
			}
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
		// unpack the deltas and PDE coefficients:
		float const k = std::get<0>(deltas_);
		float const h = std::get<1>(deltas_);
		float const A = std::get<0>(coeffs_);
		float const B = std::get<1>(coeffs_);
		float const C = std::get<2>(coeffs_);
		// calculate scheme coefficients:
		float const lambda = (A*k) / (h*h);
		float const gamma = (B*k) / (2.0f*h);
		float const delta = (C*k);
		// store bc:
		float const leftLinear = leftPair.first;
		float const leftConst = leftPair.second;
		float const rightLinear = rightPair.first;
		float const rightConst = rightPair.second;

		float time = k;

		if (isSourceSet_) {
			// prepare a pointer for source on device:
			float *d_source = NULL;
			// allocate block memory on device:
			cudaMalloc((void**)&d_source, size * sizeof(float));
			// create vector on host:
			std::vector<float> h_source(size, 0.0);
			// source is zero:
			while (time <= terminalT_) {
				// discretize source function on host:
				discretizeInSpace(h, spaceStart_, time, source_, h_source);
				// copy h_source contents to d_source (host => device ):
				cudaMemcpy(d_source, h_source.data(), size * sizeof(float),
					cudaMemcpyKind::cudaMemcpyHostToDevice);
				// populate new solution in d_next:
				explicitEulerIterate1D<float> << <threadsPerBlock, blocksPerGrid >> >(d_prev, d_next, d_source,
																						lambda, gamma, delta, k, size);
				// fill in the dirichlet boundaries in d_next:
				fillRobinBC1D<float> << <threadsPerBlock, blocksPerGrid >> >(d_next, h_source.front(), h_source.back(),
																				lambda, gamma, delta,k,
																				leftLinear, leftConst,
																				rightLinear,rightConst, size);
				// swap the two pointers:
				swap(d_prev, d_next);
				time += k;
			}
			// free allocated memory blocks on device:
			cudaFree(d_source);
		}
		else {
			while (time <= terminalT_) {
				// populate new solution in d_next:
				explicitEulerIterate1D<float> << <threadsPerBlock, blocksPerGrid >> >(d_prev, d_next,
																						lambda, gamma, delta, size);
				// fill in the dirichlet boundaries in d_next:
				fillRobinBC1D<float> << <threadsPerBlock, blocksPerGrid >> >(d_next, lambda, gamma, delta,
																			leftLinear, leftConst,
																			rightLinear, rightConst, size);
				// swap the two pointers:
				swap(d_prev, d_next);
				time += k;
			}
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
		// unpack the deltas and PDE coefficients:
		double const k = std::get<0>(deltas_);
		double const h = std::get<1>(deltas_);
		double const A = std::get<0>(coeffs_);
		double const B = std::get<1>(coeffs_);
		double const C = std::get<2>(coeffs_);
		// calculate scheme coefficients:
		double const lambda = (A*k) / (h*h);
		double const gamma = (B*k) / (2.0f*h);
		double const delta = (C*k);
		// store bc:
		double const leftLinear = leftPair.first;
		double const leftConst = leftPair.second;
		double const rightLinear = rightPair.first;
		double const rightConst = rightPair.second;

		double time = k;

		if (isSourceSet_) {
			// prepare a pointer for source on device:
			double *d_source = NULL;
			// allocate block memory on device:
			cudaMalloc((void**)&d_source, size * sizeof(double));
			// create vector on host:
			std::vector<double> h_source(size, 0.0);
			// source is zero:
			while (time <= terminalT_) {
				// discretize source function on host:
				discretizeInSpace(h, spaceStart_, time, source_, h_source);
				// copy h_source contents to d_source (host => device ):
				cudaMemcpy(d_source, h_source.data(), size * sizeof(double),
					cudaMemcpyKind::cudaMemcpyHostToDevice);
				// populate new solution in d_next:
				explicitEulerIterate1D<double> << <threadsPerBlock, blocksPerGrid >> >(d_prev, d_next, d_source,
																						lambda, gamma, delta, k, size);
				// fill in the dirichlet boundaries in d_next:
				fillRobinBC1D<double> << <threadsPerBlock, blocksPerGrid >> >(d_next, h_source.front(), h_source.back(),
																			lambda, gamma, delta,k,
																			leftLinear, leftConst,
																			rightLinear, rightConst, size);
				// swap the two pointers:
				swap(d_prev, d_next);
				time += k;
			}
			// free allocated memory blocks on device:
			cudaFree(d_source);
		}
		else {
			while (time <= terminalT_) {
				// populate new solution in d_next:
				explicitEulerIterate1D<double> << <threadsPerBlock, blocksPerGrid >> >(d_prev, d_next, 
																						lambda, gamma, delta, size);
				// fill in the dirichlet boundaries in d_next:
				fillRobinBC1D<double> << <threadsPerBlock, blocksPerGrid >> >(d_next, lambda, gamma, delta, 
																				leftLinear, leftConst,
																				rightLinear, rightConst, size);
				// swap the two pointers:
				swap(d_prev, d_next);
				time += k;
			}
		}
		// copy the contents of d_next to the solution pointer:
		cudaMemcpy(solution, d_prev, size * sizeof(double),
			cudaMemcpyKind::cudaMemcpyDeviceToHost);
		cudaFree(d_prev);
		cudaFree(d_next);

	}


}
