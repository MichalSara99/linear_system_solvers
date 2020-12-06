#pragma once
#if !defined(_LSS_ONE_DIM_PDE_SCHEMES_CUDA)
#define _LSS_ONE_DIM_PDE_SCHEMES_CUDA


#include<tuple>
#include"lss_macros.h"
#include"lss_one_dim_pde_utility.h"

namespace lss_one_dim_pde_schemes_cuda {

	using lss_one_dim_pde_utility::Discretization;


	class ExplicitEulerLoopSP:
		public Discretization<float,std::vector,std::allocator<float>> {
	private:
		float spaceStart_;
		float spaceStep_;
		float timeStep_;
		float lambda_;
		float gamma_;
		float terminalT_;
		std::function<float(float, float)> source_;
		bool isSourceSet_;

	public:
		~ExplicitEulerLoopSP(){}
		explicit ExplicitEulerLoopSP() = delete;
		explicit ExplicitEulerLoopSP(float spaceStart,
									float spaceStep,
									float timeStep,
									float lambda,
									float gamma,
									float terminalTime,
									std::function<float(float, float)> const &source,
									bool isSourceSet = false):
			spaceStart_{ spaceStart },
			spaceStep_{ spaceStep },
			timeStep_{ timeStep },
			lambda_{ lambda },
			gamma_{gamma},
			terminalT_{ terminalTime },
			source_{source},
			isSourceSet_{ isSourceSet } {}


		void operator()(float const *input, std::pair<float, float> const &dirichletBC,
						unsigned long long const size, float *solution)const;
		void operator()(float const *input, std::pair<float, float> const &leftRobinBC, 
						std::pair<float, float> const &rightRobinBC, unsigned long long const size, float *solution)const;
	};
	
	class ExplicitEulerLoopDP:
		public Discretization<double, std::vector, std::allocator<double>> {
	private:
		double spaceStart_;
		double spaceStep_;
		double timeStep_;
		double lambda_;
		double gamma_;
		double terminalT_;
		std::function<double(double, double)> source_;
		bool isSourceSet_;

	public:
		~ExplicitEulerLoopDP() {}
		explicit ExplicitEulerLoopDP() = delete;
		explicit ExplicitEulerLoopDP(double spaceStart,
									double spaceStep,
									double timeStep,
									double lambda,
									double gamma,
									double terminalTime,
									std::function<double(double, double)> const &source,
									bool isSourceSet = false):
			spaceStart_{ spaceStart },
			spaceStep_{ spaceStep },
			timeStep_{timeStep},
			lambda_{lambda},
			gamma_{gamma},
			terminalT_{ terminalTime },
			source_{source},
			isSourceSet_{ isSourceSet } {}

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
			float spaceStart_;
			float spaceStep_;
			float timeStep_;
			float terminalT_;
			float diffusivity_;
			Container<float, Alloc> init_;
			bool isSourceSet_;
			std::function<float(float, float)> source_;


		public:
			typedef float value_type;
			explicit ExplicitEulerHeatEquationScheme() = delete;
			explicit ExplicitEulerHeatEquationScheme(float spaceStart,
													float spaceStep,
													float timeStep,
													float terminalTime,
													float diffusivity,
													Container<float, Alloc> const &init,
													bool isSourceSet,
													std::function<float(float, float)> const &source = nullptr)
				:spaceStart_{ spaceStart },
				spaceStep_{ spaceStep },
				timeStep_{timeStep},
				terminalT_{ terminalTime },
				diffusivity_{ diffusivity },
				init_{init},
				isSourceSet_{ isSourceSet },
				source_{ source } {}

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
			double spaceStart_;
			double spaceStep_;
			double timeStep_;
			double terminalT_;
			double diffusivity_;
			Container<double, Alloc> init_;
			bool isSourceSet_;
			std::function<double(double, double)> source_;

		public:
			typedef double value_type;
			explicit ExplicitEulerHeatEquationScheme() = delete;
			explicit ExplicitEulerHeatEquationScheme(double spaceStart,
													double spaceStep,
													double timeStep,
													double terminalTime,
													double diffusivity,
													Container<double, Alloc> const &init,
													bool isSourceSet,
													std::function<double(double, double)> const &source = nullptr)
				:spaceStart_{ spaceStart },
				spaceStep_{ spaceStep },
				timeStep_{timeStep},
				terminalT_{ terminalTime },
				diffusivity_{ diffusivity },
				init_{ init },				
				isSourceSet_ {isSourceSet},
				source_{ source } {}

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
			float spaceStart_;
			float spaceStep_;
			float timeStep_;
			float terminalT_;
			float diffusivity_;
			float convection_;
			Container<float, Alloc> init_;
			bool isSourceSet_;
			std::function<float(float, float)> source_;

		public:
			typedef float value_type;
			explicit ExplicitAdvectionDiffusionEquationScheme() = delete;
			explicit ExplicitAdvectionDiffusionEquationScheme(float spaceStart,
															float spaceStep,
															float timeStep,
															float terminalTime,
															float diffusivity,
															float convection,
															Container<float, Alloc> const &init,
															bool isSourceSet,
															std::function<float(float, float)> const &source = nullptr)
				:spaceStart_{ spaceStart },
				spaceStep_{ spaceStep },
				timeStep_{ timeStep },
				terminalT_{ terminalTime },
				diffusivity_{ diffusivity },
				convection_{ convection },
				init_{ init },
				isSourceSet_{ isSourceSet },
				source_{ source } {}

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
			double spaceStart_;
			double spaceStep_;
			double timeStep_;
			double terminalT_;
			double diffusivity_;
			double convection_;
			Container<double, Alloc> init_;
			bool isSourceSet_;
			std::function<double(double, double)> source_;

		public:
			typedef double value_type;
			explicit ExplicitAdvectionDiffusionEquationScheme() = delete;
			explicit ExplicitAdvectionDiffusionEquationScheme(double spaceStart,
															double spaceStep,
															double timeStep,
															double terminalTime,
															double diffusivity,
															double convection,
															Container<double, Alloc> const &init,
															bool isSourceSet,
															std::function<double(double, double)> const &source = nullptr)
				:spaceStart_{ spaceStart },
				spaceStep_{ spaceStep },
				timeStep_{ timeStep },
				terminalT_{ terminalTime },
				diffusivity_{ diffusivity },
				convection_{ convection },
				init_{ init },
				isSourceSet_{ isSourceSet },
				source_{ source } {}

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
		// construct lambda:
		float const lambda = (diffusivity_*timeStep_) / (spaceStep_*spaceStep_);
		// create prev pointer:
		float *prev = (float*)malloc(size * sizeof(float));
		std::copy(init_.begin(), init_.end(), prev);
		// create next pointer:
		float *next = (float*)malloc(size * sizeof(float));
		// launch the Euler loop:
		ExplicitEulerLoopSP loop{spaceStart_, spaceStep_,timeStep_,lambda,0.0f,terminalT_,source_,isSourceSet_ };
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
		// construct lambda:
		double const lambda = (diffusivity_*timeStep_) / (spaceStep_*spaceStep_);
		// create prev pointer:
		double *prev = (double*)malloc(size * sizeof(double));
		std::copy(init_.begin(), init_.end(), prev);
		
		// create next pointer:
		double *next = (double*)malloc(size * sizeof(double));
		// launch the Euler loop:
		ExplicitEulerLoopDP loop{ spaceStart_, spaceStep_, timeStep_,lambda,0.0,terminalT_,source_,isSourceSet_ };
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
		// construct lambda:
		float const lambda = (diffusivity_*timeStep_) / (spaceStep_*spaceStep_);
		// create prev pointer:
		float *prev = (float*)malloc(size * sizeof(float));
		std::copy(init_.begin(), init_.end(), prev);
		// create next pointer:
		float *next = (float*)malloc(size * sizeof(float));
		// launch the Euler loop:
		ExplicitEulerLoopSP loop{ spaceStart_, spaceStep_, timeStep_,lambda,0.0f,terminalT_,source_,isSourceSet_ };
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
		// construct lambda:
		double const lambda = (diffusivity_*timeStep_) / (spaceStep_*spaceStep_);
		// create prev pointer:
		double *prev = (double*)malloc(size * sizeof(double));
		std::copy(init_.begin(), init_.end(), prev);

		// create next pointer:
		double *next = (double*)malloc(size * sizeof(double));
		// launch the Euler loop:
		ExplicitEulerLoopDP loop{ spaceStart_, spaceStep_, timeStep_,lambda,0.0,terminalT_,source_,isSourceSet_ };
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
		// construct lambda:
		float const lambda = (diffusivity_*timeStep_) / (spaceStep_*spaceStep_);
		// calculate gamma:
		float const gamma = (convection_ *  timeStep_) / (2.0f*spaceStep_);
		// create prev pointer:
		float *prev = (float*)malloc(size * sizeof(float));
		std::copy(init_.begin(), init_.end(), prev);
		// create next pointer:
		float *next = (float*)malloc(size * sizeof(float));
		// launch the Euler loop:
		ExplicitEulerLoopSP loop{ spaceStart_, spaceStep_, timeStep_,lambda,gamma,terminalT_,source_,isSourceSet_ };
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
		// construct lambda:
		double const lambda = (diffusivity_*timeStep_) / (spaceStep_*spaceStep_);
		// calculate gamma:
		double const gamma = (convection_ *  timeStep_) / (2.0*spaceStep_);
		// create prev pointer:
		double *prev = (double*)malloc(size * sizeof(double));
		std::copy(init_.begin(), init_.end(), prev);

		// create next pointer:
		double *next = (double*)malloc(size * sizeof(double));
		// launch the Euler loop:
		ExplicitEulerLoopDP loop{ spaceStart_, spaceStep_, timeStep_,lambda,gamma,terminalT_,source_,isSourceSet_ };
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
		// construct lambda:
		float const lambda = (diffusivity_*timeStep_) / (spaceStep_*spaceStep_);
		// calculate gamma:
		float const gamma = (convection_ *  timeStep_) / (2.0f*spaceStep_);
		// create prev pointer:
		float *prev = (float*)malloc(size * sizeof(float));
		std::copy(init_.begin(), init_.end(), prev);
		// create next pointer:
		float *next = (float*)malloc(size * sizeof(float));
		// launch the Euler loop:
		ExplicitEulerLoopSP loop{ spaceStart_, spaceStep_, timeStep_,lambda,gamma,terminalT_,source_,isSourceSet_ };
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
		// construct lambda:
		double const lambda = (diffusivity_*timeStep_) / (spaceStep_*spaceStep_);
		// calculate gamma:
		double const gamma = (convection_ *  timeStep_) / (2.0*spaceStep_);
		// create prev pointer:
		double *prev = (double*)malloc(size * sizeof(double));
		std::copy(init_.begin(), init_.end(), prev);

		// create next pointer:
		double *next = (double*)malloc(size * sizeof(double));
		// launch the Euler loop:
		ExplicitEulerLoopDP loop{ spaceStart_, spaceStep_, timeStep_,lambda,gamma,terminalT_,source_,isSourceSet_ };
		loop(prev, leftPair, rightPair, size, next);
		// next point to the solution
		std::copy(next, next + size, solution.begin());
		free(prev);
		free(next);
	}





}

#endif ///_LSS_ONE_DIM_PDE_SCHEMES_CUDA