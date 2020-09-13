#pragma once
#if !defined(_LSS_ONE_DIM_PDE_SCHEMES)
#define _LSS_ONE_DIM_PDE_SCHEMES

#include<thread>
#include"lss_types.h"

namespace lss_one_dim_pde_schemes {

	using lss_types::ImplicitPDESchemes;
	using lss_types::ExplicitPDESchemes;


	// ==============================================================================================================
	// ========================================= ImplicitHeatEquationSchemes  =======================================
	// ==============================================================================================================

	template<typename T>
	class ImplicitHeatEquationSchemes {
		public:
			typedef std::function<void(T,std::vector<T> const&,std::vector<T> &)> SchemeFunction;

			static T const getTheta(ImplicitPDESchemes scheme) {
				double theta{};
				if (scheme == ImplicitPDESchemes::Euler)
					theta = 1.0;
				else
					theta = 0.5;
				return theta;
			}

			static SchemeFunction const getScheme(ImplicitPDESchemes scheme) {
				double theta{};
				if (scheme == ImplicitPDESchemes::Euler)
					theta = 1.0;
				else
					theta = 0.5;
				auto schemeFun = [=](T lambda, 
										std::vector<T> const& input,
										std::vector<T> &solution) {
					for (std::size_t t = 1; t < solution.size() - 1; ++t) {
						solution[t] = (lambda*(1.0 - theta)*input[t + 1])
							+ (1.0 - (2.0*lambda*(1.0 - theta)))*input[t]
							+ (lambda*(1.0 - theta)*input[t - 1]);
					}
				};
				return schemeFun;
			}
	};

	// ==============================================================================================================
	// ============================================= ExplicitEulerScheme  ===========================================
	// ==============================================================================================================

	template<typename T>
	class ExplicitEulerScheme {
	private:
		std::vector<T> initialCondition_;
		T spaceStep_;
		T timeStep_;
		T terminalTime_;
		T thermalDiffusivity_;

	public:
		explicit ExplicitEulerScheme() = delete;
		explicit ExplicitEulerScheme(std::vector<T> const& initialCondition,
									T spaceStep,
									T timeStep,
									T terminalTime,
									T thermalDiffusivity)
			:initialCondition_{ initialCondition },
			spaceStep_{ spaceStep },
			timeStep_{ timeStep },
			terminalTime_{ terminalTime },
			thermalDiffusivity_{ thermalDiffusivity } {}

		~ExplicitEulerScheme(){}

		ExplicitEulerScheme(ExplicitEulerScheme const &) = delete;
		ExplicitEulerScheme(ExplicitEulerScheme &&) = delete;
		ExplicitEulerScheme& operator=(ExplicitEulerScheme const&) = delete;
		ExplicitEulerScheme& operator=(ExplicitEulerScheme &&) = delete;

		// stability check:
		bool inline isStable()const { return ((2.0*thermalDiffusivity_*timeStep_ / (spaceStep_*spaceStep_)) <= 1.0); }

		// for Dirichlet BC
		void operator()(std::pair<T,T> const &dirichletBCPair, std::vector<T> &solution)const;

	};


	// ==============================================================================================================
	// ============================================= ADEBakaratClarkScheme  =========================================
	// ==============================================================================================================


	template<typename T>
	class ADEBakaratClarkScheme {
	private:
		std::vector<T> initialCondition_;
		T spaceStep_;
		T timeStep_;
		T terminalTime_;
		T thermalDiffusivity_;
	
	public:
		explicit ADEBakaratClarkScheme() = delete;
		explicit ADEBakaratClarkScheme(std::vector<T> const& initialCondition,
										T spaceStep,
										T timeStep,
										T terminalTime,
										T thermalDiffusivity)
			:initialCondition_{ initialCondition },
			spaceStep_{ spaceStep },
			timeStep_{ timeStep },
			terminalTime_{ terminalTime },
			thermalDiffusivity_{ thermalDiffusivity } {}

		~ADEBakaratClarkScheme() {}

		ADEBakaratClarkScheme(ADEBakaratClarkScheme const &) = delete;
		ADEBakaratClarkScheme(ADEBakaratClarkScheme &&) = delete;
		ADEBakaratClarkScheme& operator=(ADEBakaratClarkScheme const&) = delete;
		ADEBakaratClarkScheme& operator=(ADEBakaratClarkScheme &&) = delete;

		// for Dirichlet BC
		void operator()(std::pair<T, T> const &dirichletBCPair,std::vector<T> &solution)const;

	};

}



template<typename T>
void lss_one_dim_pde_schemes::ExplicitEulerScheme<T>::operator()(std::pair<T, T> const &dirichletBCPair, 
	std::vector<T> &solution)const {
	LSS_ASSERT(solution.size() > 0,
		"The input solution container must be initialized.");
	LSS_ASSERT(solution.size() == initialCondition_.size(),
		"Entered solution vector size differs from initialCondition vector.");
	LSS_ASSERT(isStable() == true, 
		"This discretization is not stable.");
	// create first time point:
	T time = timeStep_;
	// calculate lambda:
	T const lambda = (thermalDiffusivity_ *  timeStep_) / (spaceStep_*spaceStep_);
	// set up coefficients:
	T const a = 1.0 - 2.0*lambda;
	T const b = lambda;
	// previous solution:
	std::vector<T> prevSol = initialCondition_;
	// left space boundary:
	T const left = dirichletBCPair.first;
	// right space boundary:
	T const right = dirichletBCPair.second;
	// size of the space vector:
	std::size_t const spaceSize = solution.size();
	// loop for stepping in time:
	while (time <= terminalTime_) {
		solution[0] = left;
		solution[solution.size() - 1] = right;
		for (std::size_t t = 1; t < spaceSize - 1; ++t) {
			solution[t] = a * prevSol[t] + b * (prevSol[t + 1] + prevSol[t - 1]);
		}
		prevSol = solution;
		time += timeStep_;
	}
}


template<typename T>
void lss_one_dim_pde_schemes::ADEBakaratClarkScheme<T>::operator()(std::pair<T, T> const &dirichletBCPair,
	std::vector<T> &solution)const {
	LSS_ASSERT(solution.size() > 0,
		"The input solution container must be initialized.");
	LSS_ASSERT(solution.size() == initialCondition_.size(),
		"Entered solution vector size differs from initialCondition vector.");
	// create first time point:
	T time = timeStep_;
	// calculate lambda:
	T const lambda = (thermalDiffusivity_ *  timeStep_) / (spaceStep_*spaceStep_);
	// set up coefficients:
	T const divisor = 1.0 + lambda;
	T const a = (1.0 - lambda) / divisor;
	T const b = lambda / divisor;
	// left space boundary:
	T const left = dirichletBCPair.first;
	// right space boundary:
	T const right = dirichletBCPair.second;
	// conmponents of the solution:
	std::vector<T> com1(initialCondition_);
	std::vector<T> com2(initialCondition_);
	// size of the space vector:
	std::size_t const spaceSize = solution.size();
	// create upsweep anonymous function:
	auto upSweep = [=](std::vector<T>& upComponent) {
		for (std::size_t t = 1; t < spaceSize - 1; ++t) {
			upComponent[t] = a * upComponent[t] + b * (upComponent[t + 1] + upComponent[t - 1]);
		}
	};
	// create downsweep anonymous function:
	auto downSweep = [=](std::vector<T>& downComponent) {
		for (std::size_t t = spaceSize - 2; t >= 1; --t) {
			downComponent[t] = a * downComponent[t] + b * (downComponent[t + 1] + downComponent[t - 1]);
		}
	};
	// loop for stepping in time:
	while (time <= terminalTime_) {
		com1[0] = com2[0] = left;
		com1[solution.size() - 1] = com2[solution.size() - 1] = right;
		std::thread upSweepTr(std::move(upSweep), std::ref(com1));
		std::thread downSweepTr(std::move(downSweep), std::ref(com2));
		upSweepTr.join();
		downSweepTr.join();
		for (std::size_t t = 0; t < spaceSize; ++t) {
			solution[t] = 0.5*(com1[t] + com2[t]);
		}
		time += timeStep_;
	}
}





#endif //_LSS_ONE_DIM_PDE_SCHEMES