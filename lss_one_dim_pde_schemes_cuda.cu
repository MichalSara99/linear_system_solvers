#include<device_launch_parameters.h>
#include"lss_pde_cuda_kernels.h"
#include"lss_one_dim_pde_schemes_cuda.h"




namespace lss_one_dim_pde_schemes_cuda {

	using lss_pde_cuda_kernels::explicitEulerIterate1D;


	void ExplicitEulerLoopSP::operator()(float const *input, float *solution, unsigned long long size)const {
		float time = timeStep_;
		float k = timeStep_;

		while (time <= terminalT_) {

			// explicitEulerIterate1D(input, solution, lambda_, size);


			time += k;
		}
	}

	void ExplicitEulerLoopDP::operator()(double const *input, double *solution, unsigned long long size)const {
		double time = timeStep_;
		double k = timeStep_;
		while (time <= terminalT_) {




			time += k;
		}
	}
	



}
