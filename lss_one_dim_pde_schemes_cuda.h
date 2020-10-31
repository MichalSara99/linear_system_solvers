#pragma once
#if !defined(_LSS_ONE_DIM_PDE_SCHEMES_CUDA)
#define _LSS_ONE_DIM_PDE_SCHEMES_CUDA


#include<tuple>
#include"lss_macros.h"

namespace lss_one_dim_pde_schemes_cuda {



	class ExplicitEulerLoopSP {
	private:
		float timeStep_;
		float lambda_;
		float gamma_;
		float terminalT_;
	public:
		~ExplicitEulerLoopSP(){}
		explicit ExplicitEulerLoopSP() = delete;
		explicit ExplicitEulerLoopSP(float timeStep,
									float lambda,
									float gamma,
									float terminalTime):
			timeStep_{ timeStep },
			lambda_{ lambda },
			gamma_{gamma},
			terminalT_{ terminalTime } {}


		void operator()(float const *input, std::pair<float, float> const &dirichletBC,
						unsigned long long const size, float *solution)const;
		void operator()(float const *input, std::pair<float, float> const &leftRobinBC, 
						std::pair<float, float> const &rightRobinBC, unsigned long long const size, float *solution)const;
	};
	
	class ExplicitEulerLoopDP {
	private:
		double timeStep_;
		double lambda_;
		double gamma_;
		double terminalT_;
	public:
		~ExplicitEulerLoopDP() {}
		explicit ExplicitEulerLoopDP() = delete;
		explicit ExplicitEulerLoopDP(double timeStep,
									double lambda,
									double gamma,
									double terminalTime):
			timeStep_{timeStep},
			lambda_{lambda},
			gamma_{gamma},
			terminalT_{ terminalTime } {}

		void operator()(double const *input, std::pair<double, double> const &dirichletBC,
						unsigned long long const size, double *solution)const;
		void operator()(double const *input, std::pair<double, double> const &leftRobinBC,
						std::pair<double, double> const &rightRobinBC, unsigned long long const size, double *solution)const;
	};

	// =======================================================================================================================
	// ================================== ExplicitEulerHeatEquationScheme general template ===================================
	// =======================================================================================================================

	template<typename T,
			template<typename,typename> typename Container,
			typename Alloc>
	class ExplicitEulerHeatEquationScheme{};


	// =======================================================================================================================
	// =========== Single-Precision Floating-Point ExplicitEulerHeatEquationScheme partial specialization template ===========
	// =======================================================================================================================

	template<template<typename, typename> typename Container,
			typename Alloc>
		class ExplicitEulerHeatEquationScheme<float, Container,Alloc> {
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

			void operator()(std::pair<float, float> const &boundaryPair, Container<float, Alloc> &solution)const;
			void operator()(std::pair<float, float> const &leftPair, std::pair<float, float> const &rightPair,
				Container<float, Alloc> &solution)const;
			
	};


	// =======================================================================================================================
	// =========== Double-Precision Floating-Point ExplicitEulerHeatEquationScheme partial specialization template ===========
	// =======================================================================================================================


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

			void operator()(std::pair<double, double> const &boundaryPair, Container<double, Alloc> &solution)const;
			void operator()(std::pair<double, double> const &leftPair, std::pair<double, double> const &rightPair,
				Container<double, Alloc> &solution)const;
		
	};


	// =======================================================================================================================
	// ================================== ExplicitAdvectionDiffusionEquationScheme general template ==========================
	// =======================================================================================================================

	template<typename T,
		template<typename, typename> typename Container,
		typename Alloc>
	class ExplicitAdvectionDiffusionEquationScheme {};

	// =======================================================================================================================
	// ======= Single-Precision Floating-Point ExplicitAdvectionDiffusionEquationScheme partial specialization template ======
	// =======================================================================================================================

	template<template<typename, typename> typename Container,
		typename Alloc>
		class ExplicitAdvectionDiffusionEquationScheme<float, Container, Alloc> {
		private:
			float lambda_;
			float gamma_;
			float timeStep_;
			float terminalT_;
			Container<float, Alloc> init_;

		public:
			typedef float value_type;
			explicit ExplicitAdvectionDiffusionEquationScheme() = delete;
			explicit ExplicitAdvectionDiffusionEquationScheme(float lambda,
															float gamma,
															float timeStep,
															float terminalTime,
															Container<float, Alloc> const &init)
				:lambda_{ lambda },
				gamma_{ gamma },
				timeStep_{ timeStep },
				terminalT_{ terminalTime },
				init_{ init } {}

			~ExplicitAdvectionDiffusionEquationScheme() {}

			ExplicitAdvectionDiffusionEquationScheme(ExplicitAdvectionDiffusionEquationScheme const &) = delete;
			ExplicitAdvectionDiffusionEquationScheme(ExplicitAdvectionDiffusionEquationScheme &&) = delete;
			ExplicitAdvectionDiffusionEquationScheme& operator=(ExplicitAdvectionDiffusionEquationScheme const &) = delete;
			ExplicitAdvectionDiffusionEquationScheme& operator=(ExplicitAdvectionDiffusionEquationScheme &&) = delete;

			void operator()(std::pair<float, float> const &boundaryPair, Container<float, Alloc> &solution)const;
			void operator()(std::pair<float, float> const &leftPair, std::pair<float, float> const &rightPair,
				Container<float, Alloc> &solution)const;

	};


	// =======================================================================================================================
	// ======= Double-Precision Floating-Point ExplicitAdvectionDiffusionEquationScheme partial specialization template ======
	// =======================================================================================================================


	template<template<typename, typename> typename Container,
		typename Alloc>
		class ExplicitAdvectionDiffusionEquationScheme<double, Container, Alloc> {
		private:
			double lambda_;
			double gamma_;
			double timeStep_;
			double terminalT_;
			Container<double, Alloc> init_;

		public:
			typedef double value_type;
			explicit ExplicitAdvectionDiffusionEquationScheme() = delete;
			explicit ExplicitAdvectionDiffusionEquationScheme(double lambda,
															double gamma,
															double timeStep,
															double terminalTime,
															Container<double, Alloc> const &init)
				:lambda_{ lambda },
				gamma_{gamma},
				timeStep_{ timeStep },
				terminalT_{ terminalTime },
				init_{ init } {}

			~ExplicitAdvectionDiffusionEquationScheme() {}

			ExplicitAdvectionDiffusionEquationScheme(ExplicitAdvectionDiffusionEquationScheme const &) = delete;
			ExplicitAdvectionDiffusionEquationScheme(ExplicitAdvectionDiffusionEquationScheme &&) = delete;
			ExplicitAdvectionDiffusionEquationScheme& operator=(ExplicitAdvectionDiffusionEquationScheme const &) = delete;
			ExplicitAdvectionDiffusionEquationScheme& operator=(ExplicitAdvectionDiffusionEquationScheme &&) = delete;

			void operator()(std::pair<double, double> const &boundaryPair, Container<double, Alloc> &solution)const;
			void operator()(std::pair<double, double> const &leftPair, std::pair<double, double> const &rightPair,
				Container<double, Alloc> &solution)const;

	};






	// ========================================= IMPLEMENTATIONS ====================================================

	// ==============================================================================================================
	// ============================== ExplicitEulerHeatEquationScheme  implementation ===============================
	// ==============================================================================================================

	template<template<typename, typename> typename Container,
			typename Alloc>
	void ExplicitEulerHeatEquationScheme<float, Container,Alloc>::
		operator()(std::pair<float, float> const &boundaryPair, Container<float, Alloc> &solution)const {
		LSS_ASSERT(init_.size() == solution.size(),
			"Initial and final solution must have the same size");
		// get the size of the vector:
		std::size_t const size = solution.size();
		// create prev pointer:
		float *prev = (float*)malloc(size * sizeof(float));
		std::copy(init_.begin(), init_.end(), prev);
		// create next pointer:
		float *next = (float*)malloc(size * sizeof(float));
		// launch the Euler loop:
		ExplicitEulerLoopSP loop{ timeStep_,lambda_,0.0f,terminalT_ };
		loop(prev, boundaryPair, size, next);
		// next point to the solution
		std::copy(next, next + size, solution.begin());
		free(prev);
		free(next);
	}

	template<template<typename, typename> typename Container,
		typename Alloc>
		void ExplicitEulerHeatEquationScheme<double, Container, Alloc>::
		operator()(std::pair<double, double> const &boundaryPair, Container<double, Alloc> &solution)const {
		LSS_ASSERT(init_.size() == solution.size(),
			"Initial and final solution must have the same size");
		// get the size of the vector:
		std::size_t const size = solution.size();
		// create prev pointer:
		double *prev = (double*)malloc(size * sizeof(double));
		std::copy(init_.begin(), init_.end(), prev);
		
		// create next pointer:
		double *next = (double*)malloc(size * sizeof(double));
		// launch the Euler loop:
		ExplicitEulerLoopDP loop{ timeStep_,lambda_,0.0,terminalT_ };
		loop(prev, boundaryPair, size, next);
		// next point to the solution
		std::copy(next, next + size, solution.begin());
		free(prev);
		free(next);
	}

	template<template<typename, typename> typename Container,
		typename Alloc>
		void ExplicitEulerHeatEquationScheme<float, Container, Alloc>::
		operator()(std::pair<float, float> const &leftPair, std::pair<float, float> const &rightPair,
					Container<float, Alloc> &solution)const {
		LSS_ASSERT(init_.size() == solution.size(),
			"Initial and final solution must have the same size");
		// get the size of the vector:
		std::size_t const size = solution.size();
		// create prev pointer:
		float *prev = (float*)malloc(size * sizeof(float));
		std::copy(init_.begin(), init_.end(), prev);
		// create next pointer:
		float *next = (float*)malloc(size * sizeof(float));
		// launch the Euler loop:
		ExplicitEulerLoopSP loop{ timeStep_,lambda_,0.0f,terminalT_ };
		loop(prev, leftPair, rightPair, size, next);
		// next point to the solution
		std::copy(next, next + size, solution.begin());
		free(prev);
		free(next);
	}

	template<template<typename, typename> typename Container,
		typename Alloc>
		void ExplicitEulerHeatEquationScheme<double, Container, Alloc>::
		operator()(std::pair<double, double> const &leftPair, std::pair<double, double> const &rightPair,
			Container<double, Alloc> &solution)const {
		LSS_ASSERT(init_.size() == solution.size(),
			"Initial and final solution must have the same size");
		// get the size of the vector:
		std::size_t const size = solution.size();
		// create prev pointer:
		double *prev = (double*)malloc(size * sizeof(double));
		std::copy(init_.begin(), init_.end(), prev);

		// create next pointer:
		double *next = (double*)malloc(size * sizeof(double));
		// launch the Euler loop:
		ExplicitEulerLoopDP loop{ timeStep_,lambda_,0.0,terminalT_ };
		loop(prev, leftPair, rightPair, size, next);
		// next point to the solution
		std::copy(next, next + size, solution.begin());
		free(prev);
		free(next);
	}


	// ==============================================================================================================
	// ===================== ExplicitAdvectionDiffusionEquationScheme  implementation ===============================
	// ==============================================================================================================


	template<template<typename, typename> typename Container,
		typename Alloc>
	void ExplicitAdvectionDiffusionEquationScheme<float, Container, Alloc>::
		operator()(std::pair<float, float> const &boundaryPair, Container<float, Alloc> &solution)const {
		LSS_ASSERT(init_.size() == solution.size(),
			"Initial and final solution must have the same size");
		// get the size of the vector:
		std::size_t const size = solution.size();
		// create prev pointer:
		float *prev = (float*)malloc(size * sizeof(float));
		std::copy(init_.begin(), init_.end(), prev);
		// create next pointer:
		float *next = (float*)malloc(size * sizeof(float));
		// launch the Euler loop:
		ExplicitEulerLoopSP loop{ timeStep_,lambda_,gamma_,terminalT_ };
		loop(prev, boundaryPair, size, next);
		// next point to the solution
		std::copy(next, next + size, solution.begin());
		free(prev);
		free(next);
	}

	template<template<typename, typename> typename Container,
		typename Alloc>
	void ExplicitAdvectionDiffusionEquationScheme<double, Container, Alloc>::
		operator()(std::pair<double, double> const &boundaryPair, Container<double, Alloc> &solution)const {
		LSS_ASSERT(init_.size() == solution.size(),
			"Initial and final solution must have the same size");
		// get the size of the vector:
		std::size_t const size = solution.size();
		// create prev pointer:
		double *prev = (double*)malloc(size * sizeof(double));
		std::copy(init_.begin(), init_.end(), prev);

		// create next pointer:
		double *next = (double*)malloc(size * sizeof(double));
		// launch the Euler loop:
		ExplicitEulerLoopDP loop{ timeStep_,lambda_,gamma_,terminalT_ };
		loop(prev, boundaryPair, size, next);
		// next point to the solution
		std::copy(next, next + size, solution.begin());
		free(prev);
		free(next);
	}

	template<template<typename, typename> typename Container,
		typename Alloc>
	void ExplicitAdvectionDiffusionEquationScheme<float, Container, Alloc>::
		operator()(std::pair<float, float> const &leftPair, std::pair<float, float> const &rightPair,
			Container<float, Alloc> &solution)const {
		LSS_ASSERT(init_.size() == solution.size(),
			"Initial and final solution must have the same size");
		// get the size of the vector:
		std::size_t const size = solution.size();
		// create prev pointer:
		float *prev = (float*)malloc(size * sizeof(float));
		std::copy(init_.begin(), init_.end(), prev);
		// create next pointer:
		float *next = (float*)malloc(size * sizeof(float));
		// launch the Euler loop:
		ExplicitEulerLoopSP loop{ timeStep_,lambda_,gamma_,terminalT_ };
		loop(prev, leftPair, rightPair, size, next);
		// next point to the solution
		std::copy(next, next + size, solution.begin());
		free(prev);
		free(next);
	}

	template<template<typename, typename> typename Container,
		typename Alloc>
	void ExplicitAdvectionDiffusionEquationScheme<double, Container, Alloc>::
		operator()(std::pair<double, double> const &leftPair, std::pair<double, double> const &rightPair,
			Container<double, Alloc> &solution)const {
		LSS_ASSERT(init_.size() == solution.size(),
			"Initial and final solution must have the same size");
		// get the size of the vector:
		std::size_t const size = solution.size();
		// create prev pointer:
		double *prev = (double*)malloc(size * sizeof(double));
		std::copy(init_.begin(), init_.end(), prev);

		// create next pointer:
		double *next = (double*)malloc(size * sizeof(double));
		// launch the Euler loop:
		ExplicitEulerLoopDP loop{ timeStep_,lambda_,gamma_,terminalT_ };
		loop(prev, leftPair, rightPair, size, next);
		// next point to the solution
		std::copy(next, next + size, solution.begin());
		free(prev);
		free(next);
	}





}

#endif ///_LSS_ONE_DIM_PDE_SCHEMES_CUDA