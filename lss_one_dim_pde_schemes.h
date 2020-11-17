#pragma once
#if !defined(_LSS_ONE_DIM_PDE_SCHEMES)
#define _LSS_ONE_DIM_PDE_SCHEMES

#include<thread>
#include"lss_types.h"
#include"lss_one_dim_pde_utility.h"

namespace lss_one_dim_pde_schemes {

	using lss_types::ImplicitPDESchemes;
	using lss_types::ExplicitPDESchemes;
	using lss_types::BoundaryConditionType;
	using lss_one_dim_pde_utility::Discretization;

	template<typename T>
	using  SchemeFunction = std::function<void(T,T, std::vector<T> const&, std::vector<T> &)>;
	template<typename T>
	using InhomSchemeFunction = std::function<void(T, T,T, std::vector<T> const &, std::vector<T> const &,
													std::vector<T> const &, std::vector<T>&)>;
	template<typename T>
	using  SchemeFunctionCUDA = std::function<void(T,T, std::vector<T> const&, std::vector<T> &,
													std::pair<T, T> const &, std::pair<T, T> const &)>;


	// ==============================================================================================================
	// ========================================= ImplicitHeatEquationSchemes  =======================================
	// ==============================================================================================================

	template<typename T>
	class ImplicitHeatEquationSchemes {
		public:

			static T const getTheta(ImplicitPDESchemes scheme) {
				double theta{};
				if (scheme == ImplicitPDESchemes::Euler)
					theta = 1.0;
				else
					theta = 0.5;
				return theta;
			}

			static SchemeFunction<T> const getScheme(ImplicitPDESchemes scheme) {
				double theta{};
				if (scheme == ImplicitPDESchemes::Euler)
					theta = 1.0;
				else
					theta = 0.5;
				auto schemeFun = [=](T lambda, T gamma,
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


			static InhomSchemeFunction<T> const getInhomScheme(ImplicitPDESchemes scheme) {
				double theta{};
				if (scheme == ImplicitPDESchemes::Euler)
					theta = 1.0;
				else
					theta = 0.5;
				auto schemeFun = [=](T lambda, T gamma,T timeStep,
									std::vector<T> const& input,
									std::vector<T> const& inhomInput,
									std::vector<T> const& inhomInputNext,
									std::vector<T> &solution) {
					for (std::size_t t = 1; t < solution.size() - 1; ++t) {
						solution[t] = (lambda*(1.0 - theta)*input[t + 1])
							+ (1.0 - (2.0*lambda*(1.0 - theta)))*input[t]
							+ (lambda*(1.0 - theta)*input[t - 1])
							+ timeStep * (theta*inhomInputNext[t] +
							(1.0 - theta)*inhomInput[t]);
					}
				};
				return schemeFun;
			}

			static SchemeFunctionCUDA<T> const getSchemeCUDA(BoundaryConditionType bcType,
														ImplicitPDESchemes scheme) {
				double theta{};
				if (scheme == ImplicitPDESchemes::Euler)
					theta = 1.0;
				else
					theta = 0.5;

				auto schemeFunDirichlet = [=](T lambda,T gamma,
					std::vector<T> const& input,
					std::vector<T> &solution,
					std::pair<T,T> const &boundaryPair0,
					std::pair<T,T> const &boundaryPair1) {
					
					T const left = boundaryPair0.first;
					T const right = boundaryPair0.second;
					std::size_t const lastIdx = solution.size() - 1;

					solution[0] = (lambda*(1.0 - theta)*input[1])
						+ (1.0 - (2.0*lambda*(1.0 - theta)))*input[0]
						+ (lambda*left);

					for (std::size_t t = 1; t < lastIdx; ++t) {
						solution[t] = (lambda*(1.0 - theta)*input[t + 1])
							+ (1.0 - (2.0*lambda*(1.0 - theta)))*input[t]
							+ (lambda*(1.0 - theta)*input[t - 1]);
					}
					
					solution[lastIdx] = (lambda*right)
						+ (1.0 - (2.0*lambda*(1.0 - theta)))*input[lastIdx]
						+ (lambda*(1.0 - theta)*input[lastIdx - 1]);
				};

				auto schemeFunRobin = [=](T lambda,T gamma,
					std::vector<T> const& input,
					std::vector<T> &solution,
					std::pair<T, T> const &boundaryPair0,
					std::pair<T, T> const &boundaryPair1) {

					T const leftLinear = boundaryPair0.first;
					T const leftConst = boundaryPair0.second;
					T const rightLinear = boundaryPair1.first;
					T const rightConst = boundaryPair1.second;
					std::size_t const lastIdx = solution.size() - 1;

					solution[0] = (lambda*(1.0 - theta)*(1.0 + leftLinear)*input[1])
						+ (1.0 - (2.0*lambda*(1.0 - theta)))*input[0]
						+ (lambda*leftConst);

					for (std::size_t t = 1; t < lastIdx; ++t) {
						solution[t] = (lambda*(1.0 - theta)*input[t + 1])
							+ (1.0 - (2.0*lambda*(1.0 - theta)))*input[t]
							+ (lambda*(1.0 - theta)*input[t - 1]);
					}

					solution[lastIdx] = (lambda*(1.0 - theta)*(1.0 + rightLinear)*input[lastIdx - 1])
						+ (1.0 - (2.0*lambda*(1.0 - theta)))*input[lastIdx]
						+ (lambda*rightConst);
				};

				if (bcType == BoundaryConditionType::Dirichlet)
					return schemeFunDirichlet;
				else
					return schemeFunRobin;
			}
	};

	// ==============================================================================================================
	// =============================== ImplicitAdvectionDiffusionEquationSchemes  ===================================
	// ==============================================================================================================

	template<typename T>
	class ImplicitAdvectionDiffusionEquationSchemes{
	public:
		static T const getTheta(ImplicitPDESchemes scheme) {
			double theta{};
			if (scheme == ImplicitPDESchemes::Euler)
				theta = 1.0;
			else
				theta = 0.5;
			return theta;
		}

		static SchemeFunction<T> const getScheme(ImplicitPDESchemes scheme) {
			double theta{};
			if (scheme == ImplicitPDESchemes::Euler)
				theta = 1.0;
			else
				theta = 0.5;
			auto schemeFun = [=](T lambda,T gamma,
				std::vector<T> const& input,
				std::vector<T> &solution) {
				for (std::size_t t = 1; t < solution.size() - 1; ++t) {
					solution[t] = ((lambda - gamma)*(1.0 - theta)*input[t + 1])
						+ (1.0 - (2.0*lambda*(1.0 - theta)))*input[t]
						+ ((lambda + gamma)*(1.0 - theta)*input[t - 1]);
				}
			};
			return schemeFun;
		}

		static SchemeFunctionCUDA<T> const getSchemeCUDA(BoundaryConditionType bcType,
			ImplicitPDESchemes scheme) {
			double theta{};
			if (scheme == ImplicitPDESchemes::Euler)
				theta = 1.0;
			else
				theta = 0.5;

			auto schemeFunDirichlet = [=](T lambda, T gamma,
				std::vector<T> const& input,
				std::vector<T> &solution,
				std::pair<T, T> const &boundaryPair0,
				std::pair<T, T> const &boundaryPair1) {

				T const left = boundaryPair0.first;
				T const right = boundaryPair0.second;
				std::size_t const lastIdx = solution.size() - 1;

				solution[0] = ((lambda - gamma)*(1.0 - theta)*input[1])
					+ (1.0 - (2.0*lambda*(1.0 - theta)))*input[0]
					+ ((lambda + gamma)*left);

				for (std::size_t t = 1; t < lastIdx; ++t) {
					solution[t] = ((lambda - gamma)*(1.0 - theta)*input[t + 1])
						+ (1.0 - (2.0*lambda*(1.0 - theta)))*input[t]
						+ ((lambda + gamma)*(1.0 - theta)*input[t - 1]);
				}

				solution[lastIdx] = ((lambda - gamma)*right)
					+ (1.0 - (2.0*lambda*(1.0 - theta)))*input[lastIdx]
					+ ((lambda + gamma)*(1.0 - theta)*input[lastIdx - 1]);
			};

			
			auto schemeFunRobin = [=](T lambda, T gamma,
				std::vector<T> const& input,
				std::vector<T> &solution,
				std::pair<T, T> const &boundaryPair0,
				std::pair<T, T> const &boundaryPair1) {

				T const leftLinear = boundaryPair0.first;
				T const leftConst = boundaryPair0.second;
				T const rightLinear = boundaryPair1.first;
				T const rightConst = boundaryPair1.second;
				std::size_t const lastIdx = solution.size() - 1;

				solution[0] = (((lambda - gamma)*(1.0 - theta) + leftLinear * (lambda + gamma))*input[1])
					+ (1.0 - (2.0*lambda*(1.0 - theta)))*input[0]
					+ ((lambda + gamma)*leftConst);

				for (std::size_t t = 1; t < lastIdx; ++t) {
					solution[t] = ((lambda-gamma)*(1.0 - theta)*input[t + 1])
						+ (1.0 - (2.0*lambda*(1.0 - theta)))*input[t]
						+ ((lambda + gamma)*(1.0 - theta)*input[t - 1]);
				}

				solution[lastIdx] = (((lambda + gamma)*(1.0 - theta) + rightLinear * (lambda - gamma))*input[lastIdx - 1])
					+ (1.0 - (2.0*lambda*(1.0 - theta)))*input[lastIdx]
					+ ((lambda - gamma)*rightConst);
			};

			if (bcType == BoundaryConditionType::Dirichlet)
				return schemeFunDirichlet;
			else
				return schemeFunRobin;
		}





	};



	// ==============================================================================================================
	// ============================================= ExplicitSchemeBase  ============================================
	// ==============================================================================================================

	template<typename T>
	class ExplicitSchemeBase:
		public Discretization<T,std::vector,std::allocator<T>> {
	protected:
		std::vector<T> initialCondition_;
		T spaceStep_;
		T timeStep_;
		T terminalTime_;
		T thermalDiffusivity_;

	public:
		explicit ExplicitSchemeBase() = delete;
		explicit ExplicitSchemeBase(std::vector<T> const& initialCondition,
									T spaceStep,
									T timeStep,
									T terminalTime,
									T thermalDiffusivity)
			:initialCondition_{ initialCondition },
			spaceStep_{ spaceStep },
			timeStep_{ timeStep },
			terminalTime_{ terminalTime },
			thermalDiffusivity_{ thermalDiffusivity } {}

		virtual ~ExplicitSchemeBase() {}

		// stability check:
		virtual bool inline isStable()const = 0;

		// for Dirichlet BC
		virtual void operator()(std::pair<T, T> const &dirichletBCPair,
								std::vector<T> &solution)const=0;
		// for Robin BC
		virtual void operator()(std::pair<T, T> const &leftRobinBCPair,
								std::pair<T, T> const &rightRobinBCPair,
								std::vector<T> &solution)const=0;

	};

	// ==============================================================================================================
	// ========================================= ExplicitHeatEulerScheme  ===========================================
	// ==============================================================================================================

	template<typename T>
	class ExplicitHeatEulerScheme :public ExplicitSchemeBase<T> {
	private:
		std::function<T(T, T)> source_;
		bool isSourceSet_;
	public:
		explicit ExplicitHeatEulerScheme() = delete;
		explicit ExplicitHeatEulerScheme(std::vector<T> const& initialCondition,
									T spaceStep,
									T timeStep,
									T terminalTime,
									T thermalDiffusivity,
									bool isSourceSet = false,
									std::function<T(T, T)> const &source = nullptr)
			:ExplicitSchemeBase<T>(initialCondition,
									spaceStep,
									timeStep,
									terminalTime,
									thermalDiffusivity),
			isSourceSet_{ isSourceSet },
			source_{ source }{}

		~ExplicitHeatEulerScheme(){}

		ExplicitHeatEulerScheme(ExplicitHeatEulerScheme const &) = delete;
		ExplicitHeatEulerScheme(ExplicitHeatEulerScheme &&) = delete;
		ExplicitHeatEulerScheme& operator=(ExplicitHeatEulerScheme const&) = delete;
		ExplicitHeatEulerScheme& operator=(ExplicitHeatEulerScheme &&) = delete;

		// stability check:
		bool inline isStable()const override{ return ((2.0*thermalDiffusivity_*timeStep_ / (spaceStep_*spaceStep_)) <= 1.0); }

		// for Dirichlet BC
		void operator()(std::pair<T,T> const &dirichletBCPair, 
						std::vector<T> &solution)const override;
		// for Robin BC
		void operator()(std::pair<T, T> const &leftRobinBCPair,
						std::pair<T, T> const &rightRobinBCPair,
						std::vector<T> &solution)const override;

	};


	// ==============================================================================================================
	// ========================================= ADEHeatBakaratClarkScheme  =========================================
	// ==============================================================================================================


	template<typename T>
	class ADEHeatBakaratClarkScheme:public ExplicitSchemeBase<T> {	
	private:
		std::function<T(T, T)> source_;
		bool isSourceSet_;
	public:
		explicit ADEHeatBakaratClarkScheme() = delete;
		explicit ADEHeatBakaratClarkScheme(std::vector<T> const& initialCondition,
										T spaceStep,
										T timeStep,
										T terminalTime,
										T thermalDiffusivity,
										bool isSourceSet = false,
										std::function<T(T, T)> const &source = nullptr)
			:ExplicitSchemeBase<T>(initialCondition,
									spaceStep,
									timeStep,
									terminalTime,
									thermalDiffusivity),
			isSourceSet_{ isSourceSet },
			source_{ source }{}

		~ADEHeatBakaratClarkScheme() {}

		ADEHeatBakaratClarkScheme(ADEHeatBakaratClarkScheme const &) = delete;
		ADEHeatBakaratClarkScheme(ADEHeatBakaratClarkScheme &&) = delete;
		ADEHeatBakaratClarkScheme& operator=(ADEHeatBakaratClarkScheme const&) = delete;
		ADEHeatBakaratClarkScheme& operator=(ADEHeatBakaratClarkScheme &&) = delete;

		// stability check:
		bool inline isStable()const override { return true; };

		// for Dirichlet BC
		void operator()(std::pair<T, T> const &dirichletBCPair,
						std::vector<T> &solution)const override;
		// for Robin BC
		void operator()(std::pair<T, T> const &leftRobinBCPair,
						std::pair<T, T> const &rightRobinBCPair,
						std::vector<T> &solution)const override;

	};

	// ==============================================================================================================
	// ========================================= ADEHeatSaulyevScheme  ==============================================
	// ==============================================================================================================

	template<typename T>
	class ADEHeatSaulyevScheme:public ExplicitSchemeBase<T> {
	private:
		std::function<T(T, T)> source_;
		bool isSourceSet_;
	public:
		explicit ADEHeatSaulyevScheme() = delete;
		explicit ADEHeatSaulyevScheme(std::vector<T> const &initialCondition,
									T spaceStep,
									T timeStep,
									T terminalTime,
									T thermalDiffusivity,
									bool isSourceSet = false,
									std::function<T(T, T)> const &source = nullptr)
			:ExplicitSchemeBase<T>(initialCondition,
									spaceStep,
									timeStep,
									terminalTime,
									thermalDiffusivity),
			isSourceSet_{ isSourceSet },
			source_{ source }{}

		~ADEHeatSaulyevScheme(){}

		ADEHeatSaulyevScheme(ADEHeatSaulyevScheme const &) = delete;
		ADEHeatSaulyevScheme(ADEHeatSaulyevScheme &&) = delete;
		ADEHeatSaulyevScheme& operator=(ADEHeatSaulyevScheme const &) = delete;
		ADEHeatSaulyevScheme& operator=(ADEHeatSaulyevScheme &&) = delete;

		// stability check:
		bool inline isStable()const override { return true; };

		// for Dirichlet BC
		void operator()(std::pair<T, T> const &dirichletBCPair, 
						std::vector<T> &solution) const override;
		// for Robin BC
		void operator()(std::pair<T, T> const &leftRobinBCPair, 
						std::pair<T, T> const &rightRobinBCPair,
						std::vector<T> &solution)const override;

	};

	// ==============================================================================================================
	// ========================================= ExplicitAdvectionDiffusionEulerScheme  =============================
	// ==============================================================================================================

	template<typename T>
	class ExplicitAdvectionDiffusionEulerScheme :public ExplicitSchemeBase<T> {
	private:
		T convection_;
	public:
		explicit ExplicitAdvectionDiffusionEulerScheme() = delete;
		explicit ExplicitAdvectionDiffusionEulerScheme(std::vector<T> const& initialCondition,
														T spaceStep,
														T timeStep,
														T terminalTime,
														T thermalDiffusivity,
														T convection)
			:ExplicitSchemeBase<T>(initialCondition,
									spaceStep,
									timeStep,
									terminalTime,
									thermalDiffusivity),
			convection_{convection}{}

		~ExplicitAdvectionDiffusionEulerScheme() {}

		ExplicitAdvectionDiffusionEulerScheme(ExplicitAdvectionDiffusionEulerScheme const &) = delete;
		ExplicitAdvectionDiffusionEulerScheme(ExplicitAdvectionDiffusionEulerScheme &&) = delete;
		ExplicitAdvectionDiffusionEulerScheme& operator=(ExplicitAdvectionDiffusionEulerScheme const&) = delete;
		ExplicitAdvectionDiffusionEulerScheme& operator=(ExplicitAdvectionDiffusionEulerScheme &&) = delete;

		// stability check:
		bool inline isStable()const override
		{
			return ((2.0*thermalDiffusivity_*timeStep_ / (spaceStep_*spaceStep_)) <= 1.0)
				&& (convection_*(timeStep_ / spaceStep_) <= 1.0);
		}

		// for Dirichlet BC
		void operator()(std::pair<T, T> const &dirichletBCPair,
						std::vector<T> &solution)const override;
		// for Robin BC
		void operator()(std::pair<T, T> const &leftRobinBCPair,
						std::pair<T, T> const &rightRobinBCPair,
						std::vector<T> &solution)const override;

	};

	// ==============================================================================================================
	// ================================== ADEAdvectionDiffusionBakaratClarkScheme  ==================================
	// ==============================================================================================================


	template<typename T>
	class ADEAdvectionDiffusionBakaratClarkScheme :public ExplicitSchemeBase<T> {
	private:
		T convection_;
	public:
		explicit ADEAdvectionDiffusionBakaratClarkScheme() = delete;
		explicit ADEAdvectionDiffusionBakaratClarkScheme(std::vector<T> const& initialCondition,
														T spaceStep,
														T timeStep,
														T terminalTime,
														T thermalDiffusivity,
														T convection)
			:ExplicitSchemeBase<T>(initialCondition,
				spaceStep,
				timeStep,
				terminalTime,
				thermalDiffusivity),
			convection_{ convection }{}

		~ADEAdvectionDiffusionBakaratClarkScheme() {}

		ADEAdvectionDiffusionBakaratClarkScheme(ADEAdvectionDiffusionBakaratClarkScheme const &) = delete;
		ADEAdvectionDiffusionBakaratClarkScheme(ADEAdvectionDiffusionBakaratClarkScheme &&) = delete;
		ADEAdvectionDiffusionBakaratClarkScheme& operator=(ADEAdvectionDiffusionBakaratClarkScheme const&) = delete;
		ADEAdvectionDiffusionBakaratClarkScheme& operator=(ADEAdvectionDiffusionBakaratClarkScheme &&) = delete;

		// stability check:
		bool inline isStable()const override { return true; };

		// for Dirichlet BC
		void operator()(std::pair<T, T> const &dirichletBCPair,
						std::vector<T> &solution)const override;
		// for Robin BC
		void operator()(std::pair<T, T> const &leftRobinBCPair,
						std::pair<T, T> const &rightRobinBCPair,
						std::vector<T> &solution)const override;

	};

	// ==============================================================================================================
	// ========================================= ADEAdvectionDiffusionSaulyevScheme  ================================
	// ==============================================================================================================

	template<typename T>
	class ADEAdvectionDiffusionSaulyevScheme :public ExplicitSchemeBase<T> {
	private:
		T convection_;
	public:
		explicit ADEAdvectionDiffusionSaulyevScheme() = delete;
		explicit ADEAdvectionDiffusionSaulyevScheme(std::vector<T> const &initialCondition,
													T spaceStep,
													T timeStep,
													T terminalTime,
													T thermalDiffusivity,
													T convection)
			:ExplicitSchemeBase<T>(initialCondition,
									spaceStep,
									timeStep,
									terminalTime,
									thermalDiffusivity),
			convection_{ convection }{}

		~ADEAdvectionDiffusionSaulyevScheme() {}

		ADEAdvectionDiffusionSaulyevScheme(ADEAdvectionDiffusionSaulyevScheme const &) = delete;
		ADEAdvectionDiffusionSaulyevScheme(ADEAdvectionDiffusionSaulyevScheme &&) = delete;
		ADEAdvectionDiffusionSaulyevScheme& operator=(ADEAdvectionDiffusionSaulyevScheme const &) = delete;
		ADEAdvectionDiffusionSaulyevScheme& operator=(ADEAdvectionDiffusionSaulyevScheme &&) = delete;

		// stability check:
		bool inline isStable()const override { return true; };

		// for Dirichlet BC
		void operator()(std::pair<T, T> const &dirichletBCPair,
						std::vector<T> &solution) const override;
		// for Robin BC
		void operator()(std::pair<T, T> const &leftRobinBCPair,
						std::pair<T, T> const &rightRobinBCPair,
						std::vector<T> &solution)const override;

	};

}

// ====================================== IMPLEMENTATIONS =======================================================

template<typename T>
void lss_one_dim_pde_schemes::ExplicitHeatEulerScheme<T>::operator()(std::pair<T, T> const &dirichletBCPair, 
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
	if (!isSourceSet_) {
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
	else {
		// create a container to carry discretized source heat
		std::vector<T> sourceCurr(solution.size(), T{});
		discretizeInSpace(spaceStep_, left, 0.0, source_, sourceCurr);
		// loop for stepping in time:
		while (time <= terminalTime_) {
			solution[0] = left;
			solution[solution.size() - 1] = right;
			for (std::size_t t = 1; t < spaceSize - 1; ++t) {
				solution[t] = a * prevSol[t] + b * (prevSol[t + 1] + prevSol[t - 1]) +
					timeStep_ * sourceCurr[t];
			}
			discretizeInSpace(spaceStep_, left, time, source_, sourceCurr);
			prevSol = solution;
			time += timeStep_;
		}
	}

}

template<typename T>
void lss_one_dim_pde_schemes::ExplicitHeatEulerScheme<T>::operator()(std::pair<T, T> const &leftRobinBCPair,
																	std::pair<T, T> const &rightRobinBCPair,
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
	// left space boundary:
	T const leftLin = leftRobinBCPair.first;
	T const leftConst = leftRobinBCPair.second;
	// right space boundary:
	T const rightLin_ = rightRobinBCPair.first;
	T const rightConst_ = rightRobinBCPair.second;
	// conversion of right hand boundaries:
	T const rightLin = 1.0 / rightLin_;
	T const rightConst = -1.0*(rightConst_ / rightLin_);
	// set up coefficients:
	T const a = 1.0 - 2.0*lambda;
	T const b = lambda;
	T const c = 1.0 + leftLin;
	T const d = 1.0 + rightLin;
	// previous solution:
	std::vector<T> prevSol = initialCondition_;
	// size of the space vector:
	std::size_t const spaceSize = solution.size();
	// loop for stepping in time:
	while (time <= terminalTime_) {
		solution[0] = b * c*prevSol[1] + a * prevSol[0] + b * leftConst;
		solution[solution.size() - 1] = b * d*prevSol[solution.size() - 2] + a * prevSol[solution.size() - 1] + b * rightConst;
		for (std::size_t t = 1; t < spaceSize - 1; ++t) {
			solution[t] = a * prevSol[t] + b * (prevSol[t + 1] + prevSol[t - 1]);
		}
		prevSol = solution;
		time += timeStep_;
	}
}




template<typename T>
void lss_one_dim_pde_schemes::ADEHeatBakaratClarkScheme<T>::operator()(std::pair<T, T> const &dirichletBCPair,
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
	T const c = timeStep_ / divisor;
	// left space boundary:
	T const left = dirichletBCPair.first;
	// right space boundary:
	T const right = dirichletBCPair.second;
	// conmponents of the solution:
	std::vector<T> com1(initialCondition_);
	std::vector<T> com2(initialCondition_);
	// size of the space vector:
	std::size_t const spaceSize = solution.size();
	// create a container to carry discretized source heat
	std::vector<T> sourceCurr(spaceSize, T{});
	std::vector<T> sourceNext(spaceSize, T{});
	// create upsweep anonymous function:
	auto upSweep = [=](std::vector<T>& upComponent,std::vector<T> const &rhs,T rhsCoeff) {
		for (std::size_t t = 1; t < spaceSize - 1; ++t) {
			upComponent[t] = a * upComponent[t] + b * (upComponent[t + 1] + upComponent[t - 1]) + c * rhsCoeff *rhs[t];
		}
	};
	// create downsweep anonymous function:
	auto downSweep = [=](std::vector<T>& downComponent, std::vector<T> const &rhs, T rhsCoeff) {
		for (std::size_t t = spaceSize - 2; t >= 1; --t) {
			downComponent[t] = a * downComponent[t] + b * (downComponent[t + 1] + downComponent[t - 1]) + c * rhsCoeff *rhs[t];
		}
	};

	if (!isSourceSet_) {
		// loop for stepping in time:
		while (time <= terminalTime_) {
			com1[0] = com2[0] = left;
			com1[solution.size() - 1] = com2[solution.size() - 1] = right;
			std::thread upSweepTr(std::move(upSweep), std::ref(com1), sourceCurr,0.0);
			std::thread downSweepTr(std::move(downSweep), std::ref(com2), sourceCurr, 0.0);
			upSweepTr.join();
			downSweepTr.join();
			for (std::size_t t = 0; t < spaceSize; ++t) {
				solution[t] = 0.5*(com1[t] + com2[t]);
			}
			time += timeStep_;
		}
	}
	else {
		discretizeInSpace(spaceStep_, left, 0.0, source_, sourceCurr);
		discretizeInSpace(spaceStep_, left, time, source_, sourceNext);
		// loop for stepping in time:
		while (time <= terminalTime_) {
			com1[0] = com2[0] = left;
			com1[solution.size() - 1] = com2[solution.size() - 1] = right;
			std::thread upSweepTr(std::move(upSweep), std::ref(com1), sourceNext,1.0);
			std::thread downSweepTr(std::move(downSweep), std::ref(com2), sourceCurr,1.0);
			upSweepTr.join();
			downSweepTr.join();
			for (std::size_t t = 0; t < spaceSize; ++t) {
				solution[t] = 0.5*(com1[t] + com2[t]);
			}
			discretizeInSpace(spaceStep_, left, time, source_, sourceCurr);
			discretizeInSpace(spaceStep_, left, 2.0*time, source_, sourceNext);
			time += timeStep_;
		}
	}

}

template<typename T>
void lss_one_dim_pde_schemes::ADEHeatBakaratClarkScheme<T>::operator()(std::pair<T, T> const &leftRobinBCPair,
																		std::pair<T,T> const &rightRobinBCPair,
																		std::vector<T> &solution)const {
	throw new std::exception("Not available.");
}


template<typename T>
void lss_one_dim_pde_schemes::ADEHeatSaulyevScheme<T>::operator()(std::pair<T, T> const &dirichletBCPair,
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
	T const c = timeStep_ / divisor;
	// left space boundary:
	T const left = dirichletBCPair.first;
	// right space boundary:
	T const right = dirichletBCPair.second;
	// get the initial condition :
	solution = initialCondition_;
	// size of the space vector:
	std::size_t const spaceSize = solution.size();
	// create a container to carry discretized source heat
	std::vector<T> sourceCurr(spaceSize, T{});
	std::vector<T> sourceNext(spaceSize, T{});
	// create upsweep anonymous function:
	auto upSweep = [=](std::vector<T>& upComponent, std::vector<T> const &rhs, T rhsCoeff) {
		for (std::size_t t = 1; t < spaceSize - 1; ++t) {
			upComponent[t] = a * upComponent[t] + b * (upComponent[t + 1] + upComponent[t - 1]) + c * rhsCoeff *rhs[t];
		}
	};
	// create downsweep anonymous function:
	auto downSweep = [=](std::vector<T>& downComponent, std::vector<T> const &rhs, T rhsCoeff) {
		for (std::size_t t = spaceSize - 2; t >= 1; --t) {
			downComponent[t] = a * downComponent[t] + b * (downComponent[t + 1] + downComponent[t - 1]) + c * rhsCoeff *rhs[t];
		}
	};

	if (!isSourceSet_) {
		// loop for stepping in time:
		std::size_t t = 1;
		while (time <= terminalTime_) {
			solution[0] = left;
			solution[solution.size() - 1] = right;
			if (t % 2 == 0)
				downSweep(solution,sourceCurr,0.0);
			else
				upSweep(solution, sourceCurr, 0.0);
			++t;
			time += timeStep_;
		}
	}
	else {
		discretizeInSpace(spaceStep_, left, 0.0, source_, sourceCurr);
		discretizeInSpace(spaceStep_, left, time, source_, sourceNext);
		// loop for stepping in time:
		std::size_t t = 1;
		while (time <= terminalTime_) {
			solution[0] = left;
			solution[solution.size() - 1] = right;
			if (t % 2 == 0)
				downSweep(solution, sourceCurr, 1.0);
			else
				upSweep(solution, sourceNext, 1.0);
			++t;
			discretizeInSpace(spaceStep_, left, time, source_, sourceCurr);
			discretizeInSpace(spaceStep_, left, 2.0*time, source_, sourceNext);
			time += timeStep_;
		}
	}

}

template<typename T>
void lss_one_dim_pde_schemes::ADEHeatSaulyevScheme<T>::operator()(std::pair<T, T> const &leftRobinBCPair,
																std::pair<T,T> const &rightRobinBCPair,	
																std::vector<T> &solution)const {
	throw new std::exception("Not available.");
}


template<typename T>
void lss_one_dim_pde_schemes::ExplicitAdvectionDiffusionEulerScheme<T>::operator()(std::pair<T, T> const &dirichletBCPair,
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
	// calculate gamma:
	T const gamma = (convection_ *  timeStep_) / (2.0*spaceStep_);
	// set up coefficients:
	T const a = 1.0 - 2.0*lambda;
	T const b = lambda + gamma;
	T const c = lambda - gamma;
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
			solution[t] = a * prevSol[t] + b * prevSol[t - 1] + c * prevSol[t + 1];
		}
		prevSol = solution;
		time += timeStep_;
	}
}

template<typename T>
void lss_one_dim_pde_schemes::ExplicitAdvectionDiffusionEulerScheme<T>::operator()(std::pair<T, T> const &leftRobinBCPair,
																				std::pair<T, T> const &rightRobinBCPair,
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
	// calculate gamma:
	T const gamma = (convection_ *  timeStep_) / (2.0*spaceStep_);
	// left space boundary:
	T const leftLin = leftRobinBCPair.first;
	T const leftConst = leftRobinBCPair.second;
	// right space boundary:
	T const rightLin_ = rightRobinBCPair.first;
	T const rightConst_ = rightRobinBCPair.second;
	// conversion of right hand boundaries:
	T const rightLin = 1.0 / rightLin_;
	T const rightConst = -1.0*(rightConst_ / rightLin_);
	// set up coefficients:
	T const a = 1.0 - 2.0*lambda;
	T const b = lambda + gamma;
	T const c = lambda - gamma;
	T const alpha = (c + leftLin * b);
	T const beta = (b + rightLin * c);
	// previous solution:
	std::vector<T> prevSol = initialCondition_;
	// size of the space vector:
	std::size_t const spaceSize = solution.size();
	// loop for stepping in time:
	while (time <= terminalTime_) {
		solution[0] = alpha * prevSol[1] + a * prevSol[0] + b * leftConst;
		solution[solution.size() - 1] = beta * prevSol[solution.size() - 2] + a * prevSol[solution.size() - 1] + c * rightConst;
		for (std::size_t t = 1; t < spaceSize - 1; ++t) {
			solution[t] = a * prevSol[t] + c * prevSol[t + 1] + b * prevSol[t - 1];
		}
		prevSol = solution;
		time += timeStep_;
	}
}



template<typename T>
void lss_one_dim_pde_schemes::ADEAdvectionDiffusionBakaratClarkScheme<T>::operator()(std::pair<T, T> const &dirichletBCPair,
																					std::vector<T> &solution)const {
	LSS_ASSERT(solution.size() > 0,
		"The input solution container must be initialized.");
	LSS_ASSERT(solution.size() == initialCondition_.size(),
		"Entered solution vector size differs from initialCondition vector.");
	// create first time point:
	T time = timeStep_;
	// calculate lambda:
	T const lambda = (thermalDiffusivity_ *  timeStep_) / (spaceStep_*spaceStep_);
	// calculate gamma:
	T const gamma = (convection_ *  timeStep_) / (2.0*spaceStep_);
	// set up coefficients:
	T const divisor = 1.0 + lambda;
	T const a = (1.0 - lambda) / divisor;
	T const b = (lambda - gamma) / divisor;
	T const c = (lambda + gamma) / divisor;
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
			upComponent[t] = a * upComponent[t] + b * upComponent[t + 1] + c * upComponent[t - 1];
		}
	};
	// create downsweep anonymous function:
	auto downSweep = [=](std::vector<T>& downComponent) {
		for (std::size_t t = spaceSize - 2; t >= 1; --t) {
			downComponent[t] = a * downComponent[t] + b * downComponent[t + 1] + c * downComponent[t - 1];
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

template<typename T>
void lss_one_dim_pde_schemes::ADEAdvectionDiffusionBakaratClarkScheme<T>::operator()(std::pair<T, T> const &leftRobinBCPair,
																					std::pair<T, T> const &rightRobinBCPair,
																					std::vector<T> &solution)const {
	throw new std::exception("Not available.");
}


template<typename T>
void lss_one_dim_pde_schemes::ADEAdvectionDiffusionSaulyevScheme<T>::operator()(std::pair<T, T> const &dirichletBCPair,
	std::vector<T> &solution)const {
	LSS_ASSERT(solution.size() > 0,
		"The input solution container must be initialized.");
	LSS_ASSERT(solution.size() == initialCondition_.size(),
		"Entered solution vector size differs from initialCondition vector.");
	// create first time point:
	T time = timeStep_;
	// calculate lambda:
	T const lambda = (thermalDiffusivity_ *  timeStep_) / (spaceStep_*spaceStep_);
	// calculate gamma:
	T const gamma = (convection_ *  timeStep_) / (2.0*spaceStep_);
	// set up coefficients:
	T const divisor = 1.0 + lambda;
	T const a = (1.0 - lambda) / divisor;
	T const b = (lambda - gamma) / divisor;
	T const c = (lambda + gamma) / divisor;
	// left space boundary:
	T const left = dirichletBCPair.first;
	// right space boundary:
	T const right = dirichletBCPair.second;
	// get the initial condition :
	solution = initialCondition_;
	// size of the space vector:
	std::size_t const spaceSize = solution.size();
	// create upsweep anonymous function:
	auto upSweep = [=](std::vector<T>& upComponent) {
		for (std::size_t t = 1; t < spaceSize - 1; ++t) {
			upComponent[t] = a * upComponent[t] + b * upComponent[t + 1] + c * upComponent[t - 1];
		}
	};
	// create downsweep anonymous function:
	auto downSweep = [=](std::vector<T>& downComponent) {
		for (std::size_t t = spaceSize - 2; t >= 1; --t) {
			downComponent[t] = a * downComponent[t] + b * downComponent[t + 1] + c * downComponent[t - 1];
		}
	};
	// loop for stepping in time:
	std::size_t t = 1;
	while (time <= terminalTime_) {
		solution[0] = left;
		solution[solution.size() - 1] = right;
		if (t % 2 == 0)
			downSweep(solution);
		else
			upSweep(solution);
		++t;
		time += timeStep_;
	}
}

template<typename T>
void lss_one_dim_pde_schemes::ADEAdvectionDiffusionSaulyevScheme<T>::operator()(std::pair<T, T> const &leftRobinBCPair,
																				std::pair<T, T> const &rightRobinBCPair,
																				std::vector<T> &solution)const {
	throw new std::exception("Not available.");
}

#endif //_LSS_ONE_DIM_PDE_SCHEMES