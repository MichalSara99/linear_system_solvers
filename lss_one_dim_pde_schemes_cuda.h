#pragma once
#if !defined(_LSS_ONE_DIM_PDE_SCHEMES_CUDA)
#define _LSS_ONE_DIM_PDE_SCHEMES_CUDA


#include<tuple>
#include"lss_macros.h"

namespace lss_one_dim_pde_schemes_cuda {


	class ExplicitEulerLoopSP {
	private:
		float timeStep_;
		float lambda_, terminalT_;
	public:
		~ExplicitEulerLoopSP(){}
		explicit ExplicitEulerLoopSP() = delete;
		explicit ExplicitEulerLoopSP(float timeStep,
									float lambda,
									float terminalTime):
			timeStep_{ timeStep },
			lambda_{ lambda },
			terminalT_{ terminalTime } {}

		void operator()(float const *input,float *solution,unsigned long long size)const;
	};
	
	class ExplicitEulerLoopDP {
	private:
		double timeStep_;
		double lambda_, terminalT_;
	public:
		~ExplicitEulerLoopDP() {}
		explicit ExplicitEulerLoopDP() = delete;
		explicit ExplicitEulerLoopDP(double timeStep,
									double lambda,
									double terminalTime):
			timeStep_{timeStep},
			lambda_{lambda},
			terminalT_{ terminalTime } {}

		void operator()(double const *input,double *solution, unsigned long long size)const;
	};


	template<typename T,
			template<typename,typename> typename Container,
			typename Alloc>
	class ExplicitEulerHeatEquationScheme{};


	template<template<typename, typename> typename Container,
			typename Alloc>
		class ExplicitEulerHeatEquationScheme<float,Container,Alloc> {
		private:
			float lambda_;
			float timeStep_;
			float terminalT_;
			Container<float, Alloc> init_;

		public:
			typedef float value_type;
			explicit ExplicitEulerHeatEquationScheme() = delete;
			explicit ExplicitEulerHeatEquationScheme(float lambda,
													float timeStep,
													float terminalTime,
													Container<float, Alloc> const &init)
				:lambda_{ lambda },
				timeStep_{timeStep},
				terminalT_{ terminalTime },
				init_{init} {}

			~ExplicitEulerHeatEquationScheme() {}

			ExplicitEulerHeatEquationScheme(ExplicitEulerHeatEquationScheme const &) = delete;
			ExplicitEulerHeatEquationScheme(ExplicitEulerHeatEquationScheme &&) = delete;
			ExplicitEulerHeatEquationScheme& operator=(ExplicitEulerHeatEquationScheme const &) = delete;
			ExplicitEulerHeatEquationScheme& operator=(ExplicitEulerHeatEquationScheme &&) = delete;

			void operator()(std::pair<float, float> const &dirichletBCPair, Container<float, Alloc> &solution)const;
	};


	template<template<typename, typename> typename Container,
			typename Alloc>
		class ExplicitEulerHeatEquationScheme<double, Container, Alloc> {
		private:
			double lambda_;
			double timeStep_;
			double terminalT_;
			Container<double, Alloc> init_;

		public:
			typedef double value_type;
			explicit ExplicitEulerHeatEquationScheme() = delete;
			explicit ExplicitEulerHeatEquationScheme(double lambda,
													double timeStep,
													double terminalTime,
													Container<double, Alloc> const &init)
				:lambda_{ lambda },
				timeStep_{timeStep},
				terminalT_{ terminalTime },
				init_{ init } {}

			~ExplicitEulerHeatEquationScheme() {}

			ExplicitEulerHeatEquationScheme(ExplicitEulerHeatEquationScheme const &) = delete;
			ExplicitEulerHeatEquationScheme(ExplicitEulerHeatEquationScheme &&) = delete;
			ExplicitEulerHeatEquationScheme& operator=(ExplicitEulerHeatEquationScheme const &) = delete;
			ExplicitEulerHeatEquationScheme& operator=(ExplicitEulerHeatEquationScheme &&) = delete;

			void operator()(std::pair<double, double> const &dirichletBCPair, Container<double, Alloc> &solution)const;
	};




	// ==============================================================================================================
	// ============================== ExplicitEulerHeatEquationScheme  implementation ===============================
	// ==============================================================================================================

	template<template<typename, typename> typename Container,
			typename Alloc>
	void ExplicitEulerHeatEquationScheme<float,Container,Alloc>::
		operator()(std::pair<float, float> const &dirichletBCPair, Container<float, Alloc> &solution)const {
		LSS_ASSERT(init.size() == solution.size(),
			"Initial and final solution must have the same size");
		// get the size of the vector:
		std::size_t const size = solution.size();
		// create prev pointer:
		float *prev = (float*)malloc(size * sizeof(float));
		std::copy(init_.begin(), init_.end(), prev);
		// create next pointer:
		float *next = (float*)malloc(size * sizeof(float));
		// launch the Euler loop:
		ExplicitEulerLoopSP loop{ timeStep_,lambda_,terminalT_ };
		loop(prev, next, size);
		// next point to the solution
		std::copy(next, next + size, std::back_inserter(solution));
		free(prev);
		free(next);
	}

	template<template<typename, typename> typename Container,
		typename Alloc>
		void ExplicitEulerHeatEquationScheme<double, Container, Alloc>::
		operator()(std::pair<double, double> const &dirichletBCPair, Container<double, Alloc> &solution)const {
		LSS_ASSERT(init.size() == solution.size(),
			"Initial and final solution must have the same size");
		// get the size of the vector:
		std::size_t const size = solution.size();
		// create prev pointer:
		double *prev = (double*)malloc(size * sizeof(double));
		std::copy(init_.begin(), init_.end(), prev);
		// create next pointer:
		double *next = (double*)malloc(size * sizeof(double));
		// launch the Euler loop:
		ExplicitEulerLoopDP loop{ timeStep_,lambda_,terminalT_ };
		loop(prev, next, size);
		// next point to the solution
		std::copy(next, next + size, std::back_inserter(solution));
		free(prev);
		free(next);
	}



}

#endif ///_LSS_ONE_DIM_PDE_SCHEMES_CUDA