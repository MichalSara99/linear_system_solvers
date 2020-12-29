#pragma once
#if !defined(_LSS_ONE_DIM_HEAT_SCHEMES)
#define _LSS_ONE_DIM_HEAT_SCHEMES

#pragma warning( disable : 4244 )

#include<thread>
#include"common/lss_types.h"
#include"pde_solvers/one_dim/lss_one_dim_pde_utility.h"

namespace lss_one_dim_heat_schemes {

	using lss_types::ImplicitPDESchemes;
	using lss_types::ExplicitPDESchemes;
	using lss_types::BoundaryConditionType;
	using lss_one_dim_pde_utility::Discretization;

	template<typename T>
	using SchemeFunction = std::function<void(std::tuple<T, T, T, T> const&, std::vector<T> const &, std::vector<T> const &,
																				std::vector<T> const &, std::vector<T>&)>;
	template<typename T>
	using SchemeFunctionCUDA = std::function<void(std::tuple<T, T, T, T> const&,
													std::vector<T> const &, std::vector<T> const &,
													std::vector<T> const &, std::vector<T>&,
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
				auto schemeFun = [=](std::tuple<T,T,T,T> const& coeffs,
					std::vector<T> const& input,
					std::vector<T> const& inhomInput,
					std::vector<T> const& inhomInputNext,
					std::vector<T> &solution) {

					// inhomInput not used
					// inhomInputNext not used

					T const lambda = std::get<0>(coeffs);
					T const gamma = std::get<1>(coeffs);
					T const delta = std::get<2>(coeffs);

					for (std::size_t t = 1; t < solution.size() - 1; ++t) {
						solution[t] = ((lambda + gamma)*(1.0 - theta)*input[t + 1])
							+ ((1.0 - (2.0*lambda - delta)*(1.0 - theta))*input[t])
							+ ((lambda - gamma)*(1.0 - theta)*input[t - 1]);
					}
				};
				return schemeFun;
			}


			static SchemeFunction<T> const getInhomScheme(ImplicitPDESchemes scheme) {
				double theta{};
				if (scheme == ImplicitPDESchemes::Euler)
					theta = 1.0;
				else
					theta = 0.5;
				auto schemeFun = [=](std::tuple<T,T,T,T> const &coeffs,
					std::vector<T> const& input,
					std::vector<T> const& inhomInput,
					std::vector<T> const& inhomInputNext,
					std::vector<T> &solution) {

					T const lambda = std::get<0>(coeffs);
					T const gamma = std::get<1>(coeffs);
					T const delta = std::get<2>(coeffs);
					T const k = std::get<3>(coeffs);

					for (std::size_t t = 1; t < solution.size() - 1; ++t) {
						solution[t] = ((lambda + gamma)*(1.0 - theta)*input[t + 1])
							+ (1.0 - ((2.0*lambda - delta)*(1.0 - theta)))*input[t]
							+ ((lambda - gamma)*(1.0 - theta)*input[t - 1])
							+ k * (theta*inhomInputNext[t] +
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

				auto schemeFunDirichlet = [=](std::tuple<T,T,T,T> const &coeffs,
					std::vector<T> const& input,
					std::vector<T> const& inhomInput,
					std::vector<T> const& inhomInputNext,
					std::vector<T> &solution,
					std::pair<T, T> const &boundaryPair0,
					std::pair<T, T> const &boundaryPair1) {

					// inhomInput			= not used here
					// inhomInputNext		= not uesd here
					// boundaryPair1		= not used here 

					T const left = boundaryPair0.first;
					T const right = boundaryPair0.second;
					std::size_t const lastIdx = solution.size() - 1;

					T const lambda = std::get<0>(coeffs);
					T const gamma = std::get<1>(coeffs);
					T const delta = std::get<2>(coeffs);

					solution[0] = ((lambda + gamma)*(1.0 - theta)*input[1])
						+ (1.0 - (2.0*lambda - delta)*(1.0 - theta))*input[0]
						+ ((lambda - gamma)*left);

					for (std::size_t t = 1; t < lastIdx; ++t) {
						solution[t] = ((lambda + gamma)*(1.0 - theta)*input[t + 1])
							+ (1.0 - (2.0*lambda - delta)*(1.0 - theta))*input[t]
							+ ((lambda - gamma)*(1.0 - theta)*input[t - 1]);
					}

					solution[lastIdx] = ((lambda + gamma)*right)
						+ (1.0 - (2.0*lambda - delta)*(1.0 - theta))*input[lastIdx]
						+ ((lambda - gamma)*(1.0 - theta)*input[lastIdx - 1]);

				};

				auto schemeFunRobin = [=](std::tuple<T, T, T, T> const &coeffs,
					std::vector<T> const& input,
					std::vector<T> const& inhomInput,
					std::vector<T> const& inhomInputNext,
					std::vector<T> &solution,
					std::pair<T, T> const &boundaryPair0,
					std::pair<T, T> const &boundaryPair1) {

					// inhomInput			= not used here
					// inhomInputNext		= not uesd here

					T const leftLinear = boundaryPair0.first;
					T const leftConst = boundaryPair0.second;
					T const rightLinear = boundaryPair1.first;
					T const rightConst = boundaryPair1.second;
					std::size_t const lastIdx = solution.size() - 1;

					T const lambda = std::get<0>(coeffs);
					T const gamma = std::get<1>(coeffs);
					T const delta = std::get<2>(coeffs);

					solution[0] = (((lambda + gamma) + (lambda-gamma)*leftLinear)*(1.0 - theta)*input[1])
						+ (1.0 - (2.0*lambda - delta)*(1.0 - theta))*input[0]
						+ ((lambda - gamma)*leftConst);

					for (std::size_t t = 1; t < lastIdx; ++t) {
						solution[t] = ((lambda + gamma)*(1.0 - theta)*input[t + 1])
							+ (1.0 - (2.0*lambda - delta)*(1.0 - theta))*input[t]
							+ ((lambda - gamma)*(1.0 - theta)*input[t - 1]);
					}

					solution[lastIdx] = (((lambda - gamma) + (lambda + gamma)*rightLinear)*(1.0 - theta)*input[lastIdx - 1])
						+ (1.0 - (2.0*lambda - delta)*(1.0 - theta))*input[lastIdx]
						+ ((lambda + gamma)*rightConst);
				};

				if (bcType == BoundaryConditionType::Dirichlet)
					return schemeFunDirichlet;
				else
					return schemeFunRobin;
			}

			static SchemeFunctionCUDA<T> const getInhomSchemeCUDA(BoundaryConditionType bcType,
				ImplicitPDESchemes scheme) {
				double theta{};
				if (scheme == ImplicitPDESchemes::Euler)
					theta = 1.0;
				else
					theta = 0.5;

				auto schemeFunDirichlet = [=](std::tuple<T, T, T, T> const &coeffs,
					std::vector<T> const& input,
					std::vector<T> const& inhomInput,
					std::vector<T> const& inhomInputNext,
					std::vector<T> &solution,
					std::pair<T, T> const &boundaryPair0,
					std::pair<T, T> const &boundaryPair1) {

					// boundaryPair1		= not used here 

					T const left = boundaryPair0.first;
					T const right = boundaryPair0.second;
					std::size_t const lastIdx = solution.size() - 1;

					T const lambda = std::get<0>(coeffs);
					T const gamma = std::get<1>(coeffs);
					T const delta = std::get<2>(coeffs);
					T const k = std::get<3>(coeffs);

					solution[0] = ((lambda + gamma)*(1.0 - theta)*input[1])
						+ (1.0 - (2.0*lambda - delta)*(1.0 - theta))*input[0]
						+ ((lambda - gamma)*left) +
						k * (theta*inhomInputNext[0] +
						(1.0 - theta)*inhomInput[0]);

					for (std::size_t t = 1; t < lastIdx; ++t) {
						solution[t] = ((lambda + gamma)*(1.0 - theta)*input[t + 1])
							+ (1.0 - (2.0*lambda - delta)*(1.0 - theta))*input[t]
							+ ((lambda - gamma)*(1.0 - theta)*input[t - 1])+
							k * (theta*inhomInputNext[t] +
							(1.0 - theta)*inhomInput[t]);
					}

					solution[lastIdx] = ((lambda + gamma)*right)
						+ (1.0 - (2.0*lambda - delta)*(1.0 - theta))*input[lastIdx]
						+ ((lambda - gamma)*(1.0 - theta)*input[lastIdx - 1]) +
						k * (theta*inhomInputNext[lastIdx] +
						(1.0 - theta)*inhomInput[lastIdx]);
				};

				auto schemeFunRobin = [=](std::tuple<T, T, T, T> const &coeffs,
					std::vector<T> const& input,
					std::vector<T> const& inhomInput,
					std::vector<T> const& inhomInputNext,
					std::vector<T> &solution,
					std::pair<T, T> const &boundaryPair0,
					std::pair<T, T> const &boundaryPair1) {

					T const leftLinear = boundaryPair0.first;
					T const leftConst = boundaryPair0.second;
					T const rightLinear = boundaryPair1.first;
					T const rightConst = boundaryPair1.second;
					std::size_t const lastIdx = solution.size() - 1;

					T const lambda = std::get<0>(coeffs);
					T const gamma = std::get<1>(coeffs);
					T const delta = std::get<2>(coeffs);
					T const k = std::get<3>(coeffs);

					solution[0] = (((lambda + gamma) + (lambda - gamma)*leftLinear)*(1.0 - theta)*input[1])
						+ (1.0 - (2.0*lambda - delta)*(1.0 - theta))*input[0]
						+ ((lambda - gamma)*leftConst) +
						k * (theta*inhomInputNext[0] +
						(1.0 - theta)*inhomInput[0]);

					for (std::size_t t = 1; t < lastIdx; ++t) {
						solution[t] = ((lambda + gamma)*(1.0 - theta)*input[t + 1])
							+ (1.0 - (2.0*lambda - delta)*(1.0 - theta))*input[t]
							+ ((lambda - gamma)*(1.0 - theta)*input[t - 1]) +
							k * (theta*inhomInputNext[t] +
							(1.0 - theta)*inhomInput[t]);
					}

					solution[lastIdx] = (((lambda - gamma) + (lambda + gamma)*rightLinear)*(1.0 - theta)*input[lastIdx - 1])
						+ (1.0 - (2.0*lambda - delta)*(1.0 - theta))*input[lastIdx]
						+ ((lambda + gamma)*rightConst) +
						k * (theta*inhomInputNext[lastIdx] +
						(1.0 - theta)*inhomInput[lastIdx]);
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
		T spaceStart_;
		T terminalTime_;

		std::pair<T, T> deltas_;				// first = delta time, second = delta space
		std::tuple<T, T, T> coeffs_;			// coefficients of PDE 
		std::vector<T> initialCondition_;
		std::function<T(T, T)> source_;
		bool isSourceSet_;

	public:
		explicit ExplicitSchemeBase() = delete;

		explicit ExplicitSchemeBase(T spaceStart,
									T terminalTime,
									std::pair<T, T> const& deltas,
									std::tuple<T, T, T> const& coeffs,
									std::vector<T> const& initialCondition,
									std::function<T(T, T)> const &source = nullptr,
									bool isSourceSet = false)
			:spaceStart_{ spaceStart },
			terminalTime_{ terminalTime },
			deltas_{ deltas },
			coeffs_{ coeffs },
			initialCondition_{ initialCondition },
			source_{ source },
			isSourceSet_{ isSourceSet } {}


		virtual ~ExplicitSchemeBase() {}

		// stability check:
		virtual bool isStable()const = 0;

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
	public:
		explicit ExplicitHeatEulerScheme() = delete;
		explicit ExplicitHeatEulerScheme(T spaceStart,
										T terminalTime,
										std::pair<T, T> const& deltas,
										std::tuple<T, T, T> const& coeffs,
										std::vector<T> const& initialCondition,
										std::function<T(T, T)> const &source = nullptr,
										bool isSourceSet = false)
			:ExplicitSchemeBase<T>(spaceStart,
									terminalTime,
									deltas,
									coeffs,
									initialCondition,
									source,
									isSourceSet) {}

		~ExplicitHeatEulerScheme(){}

		ExplicitHeatEulerScheme(ExplicitHeatEulerScheme const &) = delete;
		ExplicitHeatEulerScheme(ExplicitHeatEulerScheme &&) = delete;
		ExplicitHeatEulerScheme& operator=(ExplicitHeatEulerScheme const&) = delete;
		ExplicitHeatEulerScheme& operator=(ExplicitHeatEulerScheme &&) = delete;

		// stability check:
		bool isStable()const override;

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
	public:
		explicit ADEHeatBakaratClarkScheme() = delete;
		explicit ADEHeatBakaratClarkScheme(T spaceStart,
											T terminalTime,
											std::pair<T, T> const& deltas,
											std::tuple<T, T, T> const& coeffs,
											std::vector<T> const& initialCondition,
											std::function<T(T, T)> const &source = nullptr,
											bool isSourceSet = false)
			:ExplicitSchemeBase<T>(spaceStart,
									terminalTime,
									deltas,
									coeffs,
									initialCondition,
									source,
									isSourceSet) {}

		~ADEHeatBakaratClarkScheme() {}

		ADEHeatBakaratClarkScheme(ADEHeatBakaratClarkScheme const &) = delete;
		ADEHeatBakaratClarkScheme(ADEHeatBakaratClarkScheme &&) = delete;
		ADEHeatBakaratClarkScheme& operator=(ADEHeatBakaratClarkScheme const&) = delete;
		ADEHeatBakaratClarkScheme& operator=(ADEHeatBakaratClarkScheme &&) = delete;

		// stability check:
		bool isStable()const override { return true; };

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
	public:
		explicit ADEHeatSaulyevScheme() = delete;
		explicit ADEHeatSaulyevScheme(T spaceStart,
										T terminalTime,
										std::pair<T, T> const& deltas,
										std::tuple<T, T, T> const& coeffs,
										std::vector<T> const& initialCondition,
										std::function<T(T, T)> const &source = nullptr,
										bool isSourceSet = false)
			:ExplicitSchemeBase<T>(spaceStart,
									terminalTime,
									deltas,
									coeffs,
									initialCondition,
									source,
									isSourceSet){}

		~ADEHeatSaulyevScheme(){}

		ADEHeatSaulyevScheme(ADEHeatSaulyevScheme const &) = delete;
		ADEHeatSaulyevScheme(ADEHeatSaulyevScheme &&) = delete;
		ADEHeatSaulyevScheme& operator=(ADEHeatSaulyevScheme const &) = delete;
		ADEHeatSaulyevScheme& operator=(ADEHeatSaulyevScheme &&) = delete;

		// stability check:
		bool isStable()const override { return true; };

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
bool lss_one_dim_heat_schemes::ExplicitHeatEulerScheme<T>::isStable()const {
	T const A = std::get<0>(coeffs_);
	T const B = std::get<1>(coeffs_);
	T const k = std::get<0>(deltas_);
	T const h = std::get<1>(deltas_);

	return (((2.0*A*k / (h*h)) <= 1.0) && (B*(k / h) <= 1.0));
}

template<typename T>
void lss_one_dim_heat_schemes::ExplicitHeatEulerScheme<T>::operator()(std::pair<T, T> const &dirichletBCPair,
																	std::vector<T> &solution)const {
	LSS_ASSERT(solution.size() > 0,
		"The input solution container must be initialized.");
	LSS_ASSERT(solution.size() == initialCondition_.size(),
		"Entered solution vector size differs from initialCondition vector.");
	LSS_ASSERT(isStable() == true, 
		"This discretization is not stable.");
	// get delta time:
	T const k = std::get<0>(deltas_);
	// get delta space:
	T const h = std::get<1>(deltas_);
	// create first time point:
	T time = k;
	// get coefficients:
	T const A = std::get<0>(coeffs_);
	T const B = std::get<1>(coeffs_);
	T const C = std::get<2>(coeffs_);
	// calculate scheme coefficients:
	T const lambda = (A *  k) / (h*h);
	T const gamma = (B *  k) / (2.0 * h);
	T const delta = C * k;
	// set up coefficients:
	T const a = 1.0 - (2.0*lambda - delta);
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
	if (!isSourceSet_) {
		// loop for stepping in time:
		while (time <= terminalTime_) {
			solution[0] = left;
			solution[solution.size() - 1] = right;
			for (std::size_t t = 1; t < spaceSize - 1; ++t) {
				solution[t] = a * prevSol[t] + b * prevSol[t + 1] + c * prevSol[t - 1];
			}
			prevSol = solution;
			time += k;
		}
	}
	else {
		// create a container to carry discretized source heat
		std::vector<T> sourceCurr(solution.size(), T{});
		discretizeInSpace(h, spaceStart_, 0.0, source_, sourceCurr);
		// loop for stepping in time:
		while (time <= terminalTime_) {
			solution[0] = left;
			solution[solution.size() - 1] = right;
			for (std::size_t t = 1; t < spaceSize - 1; ++t) {
				solution[t] = a * prevSol[t] + b * prevSol[t + 1] + c * prevSol[t - 1] +
					k * sourceCurr[t];
			}
			discretizeInSpace(h, spaceStart_, time, source_, sourceCurr);
			prevSol = solution;
			time += k;
		}
	}

}

template<typename T>
void lss_one_dim_heat_schemes::ExplicitHeatEulerScheme<T>::operator()(std::pair<T, T> const &leftRobinBCPair,
																	std::pair<T, T> const &rightRobinBCPair,
																	std::vector<T> &solution)const {
	LSS_ASSERT(solution.size() > 0,
		"The input solution container must be initialized.");
	LSS_ASSERT(solution.size() == initialCondition_.size(),
		"Entered solution vector size differs from initialCondition vector.");
	LSS_ASSERT(isStable() == true,
		"This discretization is not stable.");
	// get delta time:
	T const k = std::get<0>(deltas_);
	// get delta space:
	T const h = std::get<1>(deltas_);
	// create first time point:
	T time = k;
	// get coefficients:
	T const A = std::get<0>(coeffs_);
	T const B = std::get<1>(coeffs_);
	T const C = std::get<2>(coeffs_);
	// calculate scheme coefficients:
	T const lambda = (A *  k) / (h*h);
	T const gamma = (B *  k) / (2.0 * h);
	T const delta = C * k;
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
	T const a = 1.0 - (2.0*lambda - delta);
	T const b = lambda + gamma;
	T const c = lambda - gamma;
	// previous solution:
	std::vector<T> prevSol = initialCondition_;
	// size of the space vector:
	std::size_t const spaceSize = solution.size();
	if (!isSourceSet_) {
		// loop for stepping in time:
		while (time <= terminalTime_) {
			solution[0] = (b + (c * leftLin))*prevSol[1] + a * prevSol[0] + c * leftConst;
			solution[solution.size() - 1] = (c + (b * rightLin))*prevSol[solution.size() - 2] + a * prevSol[solution.size() - 1] + b * rightConst;
			for (std::size_t t = 1; t < spaceSize - 1; ++t) {
				solution[t] = a * prevSol[t] + b * prevSol[t + 1] + c * prevSol[t - 1];
			}
			prevSol = solution;
			time += k;
		}
	}
	else {
		// create a container to carry discretized source heat
		std::vector<T> sourceCurr(solution.size(), T{});
		discretizeInSpace(h, spaceStart_, 0.0, source_, sourceCurr);
		// loop for stepping in time:
		// loop for stepping in time:
		while (time <= terminalTime_) {
			solution[0] = (b + (c * leftLin))*prevSol[1] + a * prevSol[0] + c * leftConst;
			solution[solution.size() - 1] = (c + (b * rightLin))*prevSol[solution.size() - 2] + a * prevSol[solution.size() - 1] + b * rightConst;
			for (std::size_t t = 1; t < spaceSize - 1; ++t) {
				solution[t] = a * prevSol[t] + b * prevSol[t + 1] + c * prevSol[t - 1] +
					k * sourceCurr[t];
			}
			discretizeInSpace(h, spaceStart_, time, source_, sourceCurr);
			prevSol = solution;
			time += k;
		}


	}
}




template<typename T>
void lss_one_dim_heat_schemes::ADEHeatBakaratClarkScheme<T>::operator()(std::pair<T, T> const &dirichletBCPair,
																		std::vector<T> &solution)const {
	LSS_ASSERT(solution.size() > 0,
		"The input solution container must be initialized.");
	LSS_ASSERT(solution.size() == initialCondition_.size(),
		"Entered solution vector size differs from initialCondition vector.");
	// get delta time:
	T const k = std::get<0>(deltas_);
	// get delta space:
	T const h = std::get<1>(deltas_);
	// create first time point:
	T time = k;
	// get coefficients:
	T const A = std::get<0>(coeffs_);
	T const B = std::get<1>(coeffs_);
	T const C = std::get<2>(coeffs_);
	// calculate scheme coefficients:
	T const lambda = (A *  k) / (h*h);
	T const gamma = (B *  k) / (2.0 * h);
	T const delta = C * k / 2.0;
	// set up coefficients:
	T const divisor = 1.0 + lambda - delta;
	T const a = (1.0 - lambda + delta) / divisor;
	T const b = (lambda + gamma) / divisor;
	T const c = (lambda - gamma) / divisor;
	T const d = k / divisor;
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
			upComponent[t] = a * upComponent[t] + b * upComponent[t + 1] + c * upComponent[t - 1] + d * rhsCoeff *rhs[t];
		}
	};
	// create downsweep anonymous function:
	auto downSweep = [=](std::vector<T>& downComponent, std::vector<T> const &rhs, T rhsCoeff) {
		for (std::size_t t = spaceSize - 2; t >= 1; --t) {
			downComponent[t] = a * downComponent[t] + b * downComponent[t + 1] + c * downComponent[t - 1] + d * rhsCoeff *rhs[t];
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
			time += k;
		}
	}
	else {
		discretizeInSpace(h, spaceStart_, 0.0, source_, sourceCurr);
		discretizeInSpace(h, spaceStart_, time, source_, sourceNext);
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
			discretizeInSpace(h, spaceStart_, time, source_, sourceCurr);
			discretizeInSpace(h, spaceStart_, 2.0*time, source_, sourceNext);
			time += k;
		}
	}

}

template<typename T>
void lss_one_dim_heat_schemes::ADEHeatBakaratClarkScheme<T>::operator()(std::pair<T, T> const &leftRobinBCPair,
																		std::pair<T,T> const &rightRobinBCPair,
																		std::vector<T> &solution)const {
	throw new std::exception("Not available.");
}


template<typename T>
void lss_one_dim_heat_schemes::ADEHeatSaulyevScheme<T>::operator()(std::pair<T, T> const &dirichletBCPair,
																std::vector<T> &solution)const {
	LSS_ASSERT(solution.size() > 0,
		"The input solution container must be initialized.");
	LSS_ASSERT(solution.size() == initialCondition_.size(),
		"Entered solution vector size differs from initialCondition vector.");
	// get delta time:
	T const k = std::get<0>(deltas_);
	// get delta space:
	T const h = std::get<1>(deltas_);
	// create first time point:
	T time = k;
	// get coefficients:
	T const A = std::get<0>(coeffs_);
	T const B = std::get<1>(coeffs_);
	T const C = std::get<2>(coeffs_);
	// calculate scheme coefficients:
	T const lambda = (A *  k) / (h*h);
	T const gamma = (B *  k) / (2.0 * h);
	T const delta = C * k / 2.0;
	// set up coefficients:
	T const divisor = 1.0 + lambda - delta;
	T const a = (1.0 - lambda + delta) / divisor;
	T const b = (lambda + gamma) / divisor;
	T const c = (lambda - gamma) / divisor;
	T const d = k / divisor;
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
			upComponent[t] = a * upComponent[t] + b * upComponent[t + 1] + c * upComponent[t - 1] + d * rhsCoeff *rhs[t];
		}
	};
	// create downsweep anonymous function:
	auto downSweep = [=](std::vector<T>& downComponent, std::vector<T> const &rhs, T rhsCoeff) {
		for (std::size_t t = spaceSize - 2; t >= 1; --t) {
			downComponent[t] = a * downComponent[t] + b * downComponent[t + 1] + c*downComponent[t - 1] + d * rhsCoeff *rhs[t];
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
			time += k;
		}
	}
	else {
		discretizeInSpace(h, spaceStart_, 0.0, source_, sourceCurr);
		discretizeInSpace(h, spaceStart_, time, source_, sourceNext);
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
			discretizeInSpace(h, spaceStart_, time, source_, sourceCurr);
			discretizeInSpace(h, spaceStart_, 2.0*time, source_, sourceNext);
			time += k;
		}
	}

}

template<typename T>
void lss_one_dim_heat_schemes::ADEHeatSaulyevScheme<T>::operator()(std::pair<T, T> const &leftRobinBCPair,
																std::pair<T,T> const &rightRobinBCPair,	
																std::vector<T> &solution)const {
	throw new std::exception("Not available.");
}


#endif //_LSS_ONE_DIM_HEAT_SCHEMES