#pragma once
#if !defined(_LSS_ONE_DIM_SPACE_VARIABLE_GENERAL_HEAT_EQUATION_SOLVERS)
#define _LSS_ONE_DIM_SPACE_VARIABLE_GENERAL_HEAT_EQUATION_SOLVERS

#include<functional>
#include"common/lss_types.h"
#include"common/lss_utility.h"
#include"common/lss_macros.h"
#include"pde_solvers/one_dim/lss_one_dim_pde_utility.h"
#include"lss_one_dim_space_variable_heat_implicit_schemes.h"
#include"lss_one_dim_space_variable_heat_explicit_schemes.h"


namespace lss_one_dim_space_variable_general_heat_equation_solvers {

	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_types::ExplicitPDESchemes;
	using lss_one_dim_space_variable_heat_implicit_schemes::ImplicitSpaceVariableHeatEquationSchemes;
	using lss_one_dim_space_variable_heat_explicit_schemes::ExplicitHeatEulerScheme;
	using lss_one_dim_space_variable_heat_explicit_schemes::ADEHeatBakaratClarkScheme;
	using lss_one_dim_space_variable_heat_explicit_schemes::ADEHeatSaulyevScheme;
	using lss_utility::Range;
	using lss_one_dim_pde_utility::Discretization;

	// Alias for PDE coefficients (a(x),b(x),c(x)) 
	template<typename T>
	using PDECoefficientHolder = std::tuple<std::function<T(T)>, std::function<T(T)>, std::function<T(T)>>;


	namespace implicit_solvers {

		// ==============================================================================================================
		// ======================= Implicit1DSpaceVariableGeneralHeatEquation General Template ==========================
		// ==============================================================================================================


		template<typename T,
			BoundaryConditionType BType,
			template<typename,
					BoundaryConditionType,
					template<typename, typename> typename Cont,
					typename> typename FDMSolver,
			template<typename, typename> typename Container,
			typename Alloc>
		class Implicit1DSpaceVariableGeneralHeatEquation {};


		// ==============================================================================================================
		// ============ Implicit1DSpaceVariableGeneralHeatEquation Dirichlet Specialisation Template ====================
		// ==============================================================================================================
		// 	
		//	u_t = a(x)*u_xx + b(x)*u_x + c(x)*u + F(x,t), t > 0, x_1 < x < x_2
		// 
		//	with initial condition
		//  u(x,0) = f(x)
		//
		//	and Dirichlet boundaries:
		//  u(x_1,t) = A
		//	u(x_2,t) = B
		//	
		//===============================================================================================================

		template<typename T,
				template<typename,
						BoundaryConditionType,
						template<typename, typename> typename Cont,
						typename> typename FDMSolver,
				template<typename, typename> typename Container,
				typename Alloc>
		class Implicit1DSpaceVariableGeneralHeatEquation<T,BoundaryConditionType::Dirichlet, FDMSolver,Container,Alloc>
		:public Discretization<T,Container,Alloc>{
		private:
			FDMSolver<T, BoundaryConditionType::Dirichlet, Container, Alloc> fdmSolver_;// finite-difference solver
			Range<T> spacer_;															// space range
			T terminalT_;																// terminal time
			std::size_t timeN_;															// number of time subdivisions
			std::size_t spaceN_;														// number of space subdivisions
			std::function<T(T)> init_;													// init condition
			std::function<T(T, T)> source_;												// heat source F(x,t)
			std::pair<T, T> boundary_;													// boundaries
			PDECoefficientHolder coeffs_;												// coefficients a(x), b(x), c(x) in PDE
			bool isSourceSet_;

		public:
			typedef T value_type;
			explicit Implicit1DSpaceVariableGeneralHeatEquation() = delete;
			explicit Implicit1DSpaceVariableGeneralHeatEquation(Range<T> const &spaceRange,
											T terminalTime,
											std::size_t const &spaceDiscretization,
											std::size_t const &timeDiscretization)
				:fdmSolver_{ spaceDiscretization + 1 },
				spacer_{ spaceRange }, 
				terminalT_{ terminalTime },
				timeN_{ timeDiscretization },
				spaceN_{ spaceDiscretization },
				source_{ nullptr },
				isSourceSet_{false} {}

			~Implicit1DSpaceVariableGeneralHeatEquation(){}

			Implicit1DSpaceVariableGeneralHeatEquation(Implicit1DSpaceVariableGeneralHeatEquation const &) = delete;
			Implicit1DSpaceVariableGeneralHeatEquation(Implicit1DSpaceVariableGeneralHeatEquation &&) = delete;
			Implicit1DSpaceVariableGeneralHeatEquation& operator=(Implicit1DSpaceVariableGeneralHeatEquation const &) = delete;
			Implicit1DSpaceVariableGeneralHeatEquation& operator=(Implicit1DSpaceVariableGeneralHeatEquation &&) = delete;

			inline T spaceStep()const { return (spacer_.spread() / static_cast<T>(spaceN_)); }
			inline T timeStep()const { return (terminalT_ / static_cast<T>(timeN_)); }

			inline void setBoundaryCondition(std::pair<T, T> const &boundaryPair) { 
				boundary_ = boundaryPair;
				fdmSolver_.setBoundaryCondition(boundaryPair);
			}
			inline void setInitialCondition(std::function<T(T)> const &initialCondition) {
				init_ = initialCondition; 
			}
			inline void setHeatSource(std::function<T(T, T)> const &heatSource) {
				isSourceSet_ = true;
				source_ = heatSource;
			}
			inline void set2OrderCoefficient(std::function<T(T)> const &a) {
				std::get<0>(coeffs_) = a;
			}
			inline void set1OrderCoefficient(std::function<T(T)> const &b) {
				std::get<1>(coeffs_) = b;
			}
			inline void set0OrderCoefficient(std::function<T(T)> const &c) {
				std::get<2>(coeffs_) = c;
			}

			void solve(Container<T,Alloc> &solution,
				ImplicitPDESchemes scheme = ImplicitPDESchemes::CrankNicolson);
		};

		// ==============================================================================================================
		// ============== Implicit1DSpaceVariableGeneralHeatEquation Robin Specialisation Template ======================
		// ==============================================================================================================
		// 	
		//	u_t = a(x)*u_xx + b(x)*u_x + c(x)*u + F(x,t), t > 0, x_1 < x < x_2
		// 
		//	with initial condition
		//  u(x,0) = f(x)
		//
		//	and Robin boundaries:
		//  d_1*u_x(x_1,t) + f_1*u(x_1,t) = A
		//	d_2*u_x(x_2,t) + f_2*u(x_2,t) = B
		//
		//	It is assumed the Robin boundaries are discretised in following way:
		//	d_1*(U_1 - U_0)/h + f_1*(U_0 + U_1)/2 = A
		//	d_2*(U_N - U_N-1)/h + f_2*(U_N-1 + U_N)/2 = B
		//
		//	And therefore can be rewritten in form:
		//
		//	U_0 = ((2*d_1 + f_1*h)/(2*d_1 - f_1*h)) * U_1 + (2*h/(f_1*h - 2*d_1))*A	
		//	U_N-1 = ((2*d_2 + f_2*h)/(2*d_2 - f_2*h)) * U_N + (2*h/(f_2*h - 2*d_2))*B
		//
		//	or
		//
		//	U_0 = alpha * U_1 + phi,	
		//	U_N-1 = beta * U_N + psi,	
		//
		//===============================================================================================================

		template<typename T,
				template<typename,
						BoundaryConditionType,
						template<typename, typename> typename Cont,
						typename> typename FDMSolver,
				template<typename, typename> typename Container,
				typename Alloc>
		class Implicit1DSpaceVariableGeneralHeatEquation<T, BoundaryConditionType::Robin, FDMSolver, Container, Alloc>:
			public Discretization<T, Container, Alloc> {
		private:
			FDMSolver<T, BoundaryConditionType::Robin, Container, Alloc> fdmSolver_;	// finite-difference solver
			Range<T> spacer_;															// space range
			T terminalT_;																// terminal time
			std::size_t timeN_;															// number of time subdivisions
			std::size_t spaceN_;														// number of space subdivisions
			std::function<T(T, T)> source_;												// heat source F(x,t)
			std::function<T(T)> init_;													// initi condition
			std::pair<T, T> left_;														// left boundary pair
			std::pair<T, T> right_;														// right boundary pair
			PDECoefficientHolder coeffs_;												// coefficients a(x), b(x), c(x) in PDE
			bool isSourceSet_;

		public:
			typedef T value_type;
			explicit Implicit1DSpaceVariableGeneralHeatEquation() = delete;
			explicit Implicit1DSpaceVariableGeneralHeatEquation(Range<T> const &spaceRange,
											T terminalTime,
											std::size_t const &spaceDiscretization,
											std::size_t const &timeDiscretization)
				:fdmSolver_{ spaceDiscretization + 1 },
				spacer_{ spaceRange },
				terminalT_{ terminalTime },
				timeN_{ timeDiscretization },
				spaceN_{ spaceDiscretization },
				source_{ nullptr },
				isSourceSet_{ false } {}

			~Implicit1DSpaceVariableGeneralHeatEquation() {}

			Implicit1DSpaceVariableGeneralHeatEquation(Implicit1DSpaceVariableGeneralHeatEquation const &) = delete;
			Implicit1DSpaceVariableGeneralHeatEquation(Implicit1DSpaceVariableGeneralHeatEquation &&) = delete;
			Implicit1DSpaceVariableGeneralHeatEquation& operator=(Implicit1DSpaceVariableGeneralHeatEquation const &) = delete;
			Implicit1DSpaceVariableGeneralHeatEquation& operator=(Implicit1DSpaceVariableGeneralHeatEquation &&) = delete;

			inline T spaceStep()const { return (spacer_.spread() / static_cast<T>(spaceN_)); }
			inline T timeStep()const { return (terminalT_ / static_cast<T>(timeN_)); }

			inline void setBoundaryCondition(std::pair<T, T> const &left,std::pair<T,T> const &right) { 
				left_ = left;
				right_ = right;
				fdmSolver_.setBoundaryCondition(left, right);
			}

			inline void setInitialCondition(std::function<T(T)> const &initialCondition) { 
				init_ = initialCondition; 
			}
			inline void setHeatSource(std::function<T(T, T)> const &heatSource) {
				isSourceSet_ = true;
				source_ = heatSource;
			}
			inline void set2OrderCoefficient(std::function<T(T)> const &a) {
				std::get<0>(coeffs_) = a;
			}
			inline void set1OrderCoefficient(std::function<T(T)> const &b) {
				std::get<1>(coeffs_) = b;
			}
			inline void set0OrderCoefficient(std::function<T(T)> const &c) {
				std::get<2>(coeffs_) = c;
			}

			void solve(Container<T, Alloc> &solution,
				ImplicitPDESchemes scheme = ImplicitPDESchemes::CrankNicolson);

		};

	}






	namespace explicit_solvers {

		// ==============================================================================================================
		// =========================== Explicit1DSpaceVariableGeneralHeatEquation General Template ======================
		// ==============================================================================================================

		template<typename T,
				BoundaryConditionType BType,
				template<typename, typename> typename Container,
				typename Alloc>
		class Explicit1DSpaceVariableGeneralHeatEquation {};


		// ==============================================================================================================
		// ================ Explicit1DSpaceVariableGeneralHeatEquation Dirichlet Specialisation Template ================
		// ==============================================================================================================
		// 	
		//	u_t = a(x)*u_xx + b(x)*u_x + c(x)*u + F(x,t), t > 0, x_1 < x < x_2
		// 
		//	with initial condition
		//  u(x,0) = f(x)
		//
		//	and Dirichlet boundaries:
		//  u(x_1,t) = A
		//	u(x_2,t) = B
		//	
		//===============================================================================================================

		template<typename T,
			template<typename, typename> typename Container,
			typename Alloc>
		class Explicit1DSpaceVariableGeneralHeatEquation<T,BoundaryConditionType::Dirichlet,Container,Alloc>:
		public Discretization<T,Container,Alloc>{
		private:
			Range<T> spacer_;											// space range
			T terminalT_;												// terminal time
			std::size_t timeN_;											// number of time subdivisions
			std::size_t spaceN_;										// number of space subdivisions
			std::function<T(T)> init_;									// initi condition
			std::function<T(T, T)> source_;								// heat source	F(x,t)
			std::pair<T, T> boundary_;									// boundaries
			PDECoefficientHolder coeffs_;								// coefficients a(x), b(x), c(x) in PDE
			bool isSourceSet_;

		public:
			typedef T value_type;
			explicit Explicit1DSpaceVariableGeneralHeatEquation() = delete;
			explicit Explicit1DSpaceVariableGeneralHeatEquation(Range<T> const &spaceRange,
											T terminalTime,
											std::size_t const &spaceDiscretization,
											std::size_t const &timeDiscretization)
				:spacer_{ spaceRange },
				terminalT_{ terminalTime },
				timeN_{ timeDiscretization },
				spaceN_{ spaceDiscretization },
				source_{nullptr },
				isSourceSet_{ false } {}

			~Explicit1DSpaceVariableGeneralHeatEquation(){}

			Explicit1DSpaceVariableGeneralHeatEquation(Explicit1DSpaceVariableGeneralHeatEquation const &) = delete;
			Explicit1DSpaceVariableGeneralHeatEquation(Explicit1DSpaceVariableGeneralHeatEquation &&) = delete;
			Explicit1DSpaceVariableGeneralHeatEquation& operator=(Explicit1DSpaceVariableGeneralHeatEquation const &) = delete;
			Explicit1DSpaceVariableGeneralHeatEquation& operator=(Explicit1DSpaceVariableGeneralHeatEquation &&) = delete;

			inline T spaceStep()const { return (spacer_.spread() / static_cast<T>(spaceN_)); }
			inline T timeStep()const { return (terminalT_ / static_cast<T>(timeN_)); }

			inline void setBoundaryCondition(std::pair<T, T> const &boundaryPair) {
				boundary_ = boundaryPair;
			}
			inline void setInitialCondition(std::function<T(T)> const &initialCondition) {
				init_ = initialCondition;
			}
			inline void setHeatSource(std::function<T(T, T)> const &heatSource) {
				isSourceSet_ = true;
				source_ = heatSource;
			}
			inline void set2OrderCoefficient(std::function<T(T)> const &a) {
				std::get<0>(coeffs_) = a;
			}
			inline void set1OrderCoefficient(std::function<T(T)> const &b) {
				std::get<1>(coeffs_) = b;
			}
			inline void set0OrderCoefficient(std::function<T(T)> const &c) {
				std::get<2>(coeffs_) = c;
			}

			void solve(Container<T, Alloc> &solution,
				ExplicitPDESchemes scheme = ExplicitPDESchemes::ADEBarakatClark);
		};


		// ==============================================================================================================
		// ================== Explicit1DSpaceVariableGeneralHeatEquation Robin Specialisation Template ==================
		// ==============================================================================================================
		// 	
		//	u_t = a(x)*u_xx + b(x)*u_x + c(x)*u + F(x,t), t > 0, x_1 < x < x_2
		// 
		//	with initial condition
		//  u(x,0) = f(x)
		//
		//	and Robin boundaries:
		//  d_1*u_x(x_1,t) + f_1*u(x_1,t) = A
		//	d_2*u_x(x_2,t) + f_2*u(x_2,t) = B
		//
		//	It is assumed the Robin boundaries are discretised in following way:
		//	d_1*(U_1 - U_0)/h + f_1*(U_0 + U_1)/2 = A
		//	d_2*(U_N - U_N-1)/h + f_2*(U_N-1 + U_N)/2 = B
		//
		//	And therefore can be rewritten in form:
		//
		//	U_0 = ((2*d_1 + f_1*h)/(2*d_1 - f_1*h)) * U_1 + (2*h/(f_1*h - 2*d_1))*A	
		//	U_N-1 = ((2*d_2 + f_2*h)/(2*d_2 - f_2*h)) * U_N + (2*h/(f_2*h - 2*d_2))*B
		//
		//	or
		//
		//	U_0 = alpha * U_1 + phi,	
		//	U_N-1 = beta * U_N + psi,	
		//
		//===============================================================================================================

		template<typename T,
			template<typename, typename> typename Container,
			typename Alloc>
		class Explicit1DSpaceVariableGeneralHeatEquation<T, BoundaryConditionType::Robin, Container, Alloc>:
			public Discretization<T,Container,Alloc>{
		private:
			Range<T> spacer_;											// space range
			T terminalT_;												// terminal time
			std::size_t timeN_;											// number of time subdivisions
			std::size_t spaceN_;										// number of space subdivisions
			std::function<T(T, T)> source_;								// heat source F(x,t)
			std::function<T(T)> init_;									// initi condition
			std::pair<T, T> left_;										// left boundary pair
			std::pair<T, T> right_;										// right boundary pair
			PDECoefficientHolder coeffs_;								// coefficients a(x,t), b(x,t), c(x,t) in PDE
			bool isSourceSet_;

		public:
			typedef T value_type;
			explicit Explicit1DSpaceVariableGeneralHeatEquation() = delete;
			explicit Explicit1DSpaceVariableGeneralHeatEquation(Range<T> const &spaceRange,
											T terminalTime,
											std::size_t const &spaceDiscretization,
											std::size_t const &timeDiscretization)
				:spacer_{ spaceRange },
				terminalT_{ terminalTime },
				timeN_{ timeDiscretization },
				spaceN_{ spaceDiscretization },
				source_{ nullptr },
				isSourceSet_{ false } {}

			~Explicit1DSpaceVariableGeneralHeatEquation() {}

			Explicit1DSpaceVariableGeneralHeatEquation(Explicit1DSpaceVariableGeneralHeatEquation const &) = delete;
			Explicit1DSpaceVariableGeneralHeatEquation(Explicit1DSpaceVariableGeneralHeatEquation &&) = delete;
			Explicit1DSpaceVariableGeneralHeatEquation& operator=(Explicit1DSpaceVariableGeneralHeatEquation const &) = delete;
			Explicit1DSpaceVariableGeneralHeatEquation& operator=(Explicit1DSpaceVariableGeneralHeatEquation &&) = delete;

			inline T spaceStep()const { return (spacer_.spread() / static_cast<T>(spaceN_)); }
			inline T timeStep()const { return (terminalT_ / static_cast<T>(timeN_)); }

			inline void setBoundaryCondition(std::pair<T, T> const &left,std::pair<T,T> const &right) {
				left_ = left;
				right_ = right;
			}
			inline void setInitialCondition(std::function<T(T)> const &initialCondition) {
				init_ = initialCondition;
			}
			inline void setHeatSource(std::function<T(T, T)> const &heatSource) {
				isSourceSet_ = true;
				source_ = heatSource;
			}
			inline void set2OrderCoefficient(std::function<T(T)> const &a) {
				std::get<0>(coeffs_) = a;
			}
			inline void set1OrderCoefficient(std::function<T(T)> const &b) {
				std::get<1>(coeffs_) = b;
			}
			inline void set0OrderCoefficient(std::function<T(T)> const &c) {
				std::get<2>(coeffs_) = c;
			}

			void solve(Container<T, Alloc> &solution);

		};


	}


	// ====================================== IMPLEMENTATIONS =======================================================

	// ==============================================================================================================
	// =================== Implicit1DSpaceVariableGeneralHeatEquation (Dirichlet) implementation ====================
	// ==============================================================================================================

	template<typename T,
			template<typename,
					BoundaryConditionType,
					template<typename, typename> typename Cont,
					typename> typename FDMSolver,
			template<typename, typename> typename Container,
			typename Alloc>
	void implicit_solvers::Implicit1DSpaceVariableGeneralHeatEquation<T, BoundaryConditionType::Dirichlet, FDMSolver, Container, Alloc>::
		solve(Container<T,Alloc> &solution, ImplicitPDESchemes scheme) {

		LSS_ASSERT(solution.size() > 0, "The input solution container must be initialized.");
		// get correct theta according to the scheme:
		T const theta = ImplicitSpaceVariableHeatEquationSchemes<T>::getTheta(scheme);
		// get space step:
		T const h = spaceStep();
		// get time step:
		T const k = timeStep();
		// calculate scheme const coefficients:
		T const lambda = k / (h*h);
		T const gamma = k / (2.0*h);
		T const delta = 0.5*k;
		// save scheme variable coefficients:
		auto const &a = std::get<0>(coeffs_);
		auto const &b = std::get<1>(coeffs_);
		auto const &c = std::get<2>(coeffs_);
		// create container to carry mesh in space and then previous solution:
		Container<T, Alloc> prevSol(spaceN_ + 1, T{});
		// populate the container with mesh in space
		discretizeSpace(h, spacer_.lower(), prevSol);
		// use the mesh in space to get values of initial condition
		discretizeInitialCondition(init_, prevSol);
		// since coefficients are different in space :
		Container<T, Alloc> low(spaceN_ + 1, T{});
		Container<T, Alloc> diag(spaceN_ + 1, T{});
		Container<T, Alloc> up(spaceN_ + 1, T{});
		// prepare space variable coefficients:
		auto const &A = [&](T x) {return (lambda*a(x) - gamma * b(x)); };
		auto const &B = [&](T x) {return (lambda*a(x) - delta * c(x)); };
		auto const &D = [&](T x) {return (lambda*a(x) + gamma * b(x)); };
		for (std::size_t t = 0; t < low.size(); ++t) {
			low[t] = -1.0*A(t*h)*theta;
			diag[t] = (1.0 + 2.0*B(t*h) *theta);
			up[t] = -1.0*D(t*h)*theta;
		}
		Container<T, Alloc> rhs(spaceN_ + 1, T{});
		// create container to carry new solution:
		Container<T, Alloc> nextSol(spaceN_ + 1, T{});
		// create first time point:
		T time = k;
		// store terminal time:
		T const lastTime = terminalT_;
		// set properties of FDMSolver:
		fdmSolver_.setDiagonals(std::move(low), std::move(diag), std::move(up));
		// differentiate between inhomogeneous and homogeneous PDE:
		if (isSourceSet_) {
			// wrap the scheme coefficients:
			const auto schemeCoeffs = std::make_tuple(A, B, D,h, k);
			// get the correct scheme:
			auto schemeFun = ImplicitSpaceVariableHeatEquationSchemes<T>::getInhomScheme(scheme);
			// create a container to carry discretized source heat
			Container<T, Alloc> sourceCurr(spaceN_ + 1, T{});
			Container<T, Alloc> sourceNext(spaceN_ + 1, T{});
			discretizeInSpace(h, spacer_.lower(), 0.0, source_, sourceCurr);
			discretizeInSpace(h, spacer_.lower(), time, source_, sourceNext);
			// loop for stepping in time:
			while (time <= lastTime) {
				schemeFun(schemeCoeffs, prevSol, sourceCurr, sourceNext, rhs);
				fdmSolver_.setRhs(rhs);
				fdmSolver_.solve(nextSol);
				prevSol = nextSol;
				discretizeInSpace(h, spacer_.lower(), time, source_, sourceCurr);
				discretizeInSpace(h, spacer_.lower(), 2.0*time, source_, sourceNext);
				time += k;
			}
		}
		else {
			// wrap the scheme coefficients:
			const auto schemeCoeffs = std::make_tuple(A, B, D, h, T{});
			// get the correct scheme:
			auto schemeFun = ImplicitSpaceVariableHeatEquationSchemes<T>::getScheme(scheme);
			// loop for stepping in time:
			while (time <= lastTime) {
				schemeFun(schemeCoeffs, prevSol, Container<T, Alloc>(), Container<T, Alloc>(), rhs);
				fdmSolver_.setRhs(rhs);
				fdmSolver_.solve(nextSol);
				prevSol = nextSol;
				time += k;
			}
		}
		// copy into solution vector
		std::copy(prevSol.begin(), prevSol.end(), solution.begin());
	}


	// ==============================================================================================================
	// ==================== Implicit1DSpaceVariableGeneralHeatEquation (Robin) implementation =======================
	// ==============================================================================================================


	template<typename T,
			template<typename,
					BoundaryConditionType,
					template<typename, typename> typename Cont,
					typename> typename FDMSolver,
			template<typename, typename> typename Container,
			typename Alloc>
	void implicit_solvers::Implicit1DSpaceVariableGeneralHeatEquation<T, BoundaryConditionType::Robin, FDMSolver, Container, Alloc>::
		solve(Container<T, Alloc> &solution, ImplicitPDESchemes scheme) {

		LSS_ASSERT(solution.size() > 0, "The input solution container must be initialized.");
		// get correct theta according to the scheme:
		T const theta = ImplicitSpaceVariableHeatEquationSchemes<T>::getTheta(scheme);
		// get space step:
		T const h = spaceStep();
		// get time step:
		T const k = timeStep();
		// calculate scheme const coefficients:
		T const lambda = k / (h*h);
		T const gamma = k / (2.0*h);
		T const delta = 0.5*k;
		// save scheme variable coefficients:
		auto const &a = std::get<0>(coeffs_);
		auto const &b = std::get<1>(coeffs_);
		auto const &c = std::get<2>(coeffs_);
		// create container to carry mesh in space and then previous solution:
		Container<T, Alloc> prevSol(spaceN_ + 1, T{});
		// populate the container with mesh in space
		discretizeSpace(h,spacer_.lower(), prevSol);
		// use the mesh in space to get values of initial condition
		discretizeInitialCondition(init_, prevSol);
		// since coefficients are different in space :
		Container<T, Alloc> low(spaceN_ + 1, T{});
		Container<T, Alloc> diag(spaceN_ + 1, T{});
		Container<T, Alloc> up(spaceN_ + 1, T{});
		// prepare space variable coefficients:
		auto const &A = [&](T x) {return (lambda*a(x) - gamma * b(x)); };
		auto const &B = [&](T x) {return (lambda*a(x) - delta * c(x)); };
		auto const &D = [&](T x) {return (lambda*a(x) + gamma * b(x)); };
		for (std::size_t t = 0; t < low.size(); ++t) {
			low[t] = -1.0*A(t*h)*theta;
			diag[t] = (1.0 + 2.0*B(t*h) *theta);
			up[t] = -1.0*D(t*h)*theta;
		}
		Container<T, Alloc> rhs(spaceN_ + 1, T{});
		// create container to carry new solution:
		Container<T, Alloc> nextSol(spaceN_ + 1, T{});
		// create first time point:
		T time = k;
		// store terminal time:
		T const lastTime = terminalT_;
		// set properties of FDMSolver:
		fdmSolver_.setDiagonals(std::move(low), std::move(diag), std::move(up));
		// differentiate between inhomogeneous and homogeneous PDE:
		if (isSourceSet_) {
			// wrap the scheme coefficients:
			const auto schemeCoeffs = std::make_tuple(A, B, D, h, k);
			// get the correct scheme:
			auto schemeFun = ImplicitSpaceVariableHeatEquationSchemes<T>::getInhomScheme(scheme);
			// create a container to carry discretized source heat
			Container<T, Alloc> sourceCurr(spaceN_ + 1, T{});
			Container<T, Alloc> sourceNext(spaceN_ + 1, T{});
			discretizeInSpace(h, spacer_.lower(), 0.0, source_, sourceCurr);
			discretizeInSpace(h, spacer_.lower(), time, source_, sourceNext);
			// loop for stepping in time:
			while (time <= lastTime) {
				schemeFun(schemeCoeffs, prevSol, sourceCurr, sourceNext, rhs);
				fdmSolver_.setRhs(rhs);
				fdmSolver_.solve(nextSol);
				prevSol = nextSol;
				discretizeInSpace(h, spacer_.lower(), time, source_, sourceCurr);
				discretizeInSpace(h, spacer_.lower(), 2.0*time, source_, sourceNext);
				time += k;
			}
		}
		else {
			// wrap the scheme coefficients:
			const auto schemeCoeffs = std::make_tuple(A, B, D, h, T{});
			// get the correct scheme:
			auto schemeFun = ImplicitSpaceVariableHeatEquationSchemes<T>::getScheme(scheme);
			// loop for stepping in time:
			while (time <= lastTime) {
				schemeFun(schemeCoeffs, prevSol, Container<T, Alloc>(), Container<T, Alloc>(), rhs);
				fdmSolver_.setRhs(rhs);
				fdmSolver_.solve(nextSol);
				prevSol = nextSol;
				time += k;
			}
		}
		// copy into solution vector
		std::copy(prevSol.begin(), prevSol.end(), solution.begin());
	}


	// ==============================================================================================================
	// ====================== Explicit1DSpaceVariableGeneralHeatEquation (Dirichlet) implementation =================
	// ==============================================================================================================

	
	template<typename T,
			template<typename, typename> typename Container,
			typename Alloc>
		void explicit_solvers::Explicit1DSpaceVariableGeneralHeatEquation<T, BoundaryConditionType::Dirichlet, Container, Alloc>::
		solve(Container<T, Alloc> &solution, ExplicitPDESchemes scheme) {

		LSS_ASSERT(solution.size() > 0, "The input solution container must be initialized.");
		// get space step:
		T const h = spaceStep();
		// get time step:
		T const k = timeStep();
		// calculate scheme const coefficients:
		T const lambda = k / (h*h);
		T const gamma = k / (2.0*h);
		T const delta = 0.5*k;
		// save scheme variable coefficients:
		auto const &a = std::get<0>(coeffs_);
		auto const &b = std::get<1>(coeffs_);
		auto const &c = std::get<2>(coeffs_);
		// prepare space variable coefficients:
		auto const &A = [&](T x) {return (lambda*a(x) - gamma * b(x)); };
		auto const &B = [&](T x) {return (lambda*a(x) - delta * c(x)); };
		auto const &D = [&](T x) {return (lambda*a(x) + gamma * b(x)); };
		// wrap up the scheme coefficients:
		auto schemeCoeffs = std::make_tuple(A, B, D);
		// create container to carry mesh in space and then previous solution:
		Container<T, Alloc> initCondition(spaceN_ + 1, T{});
		// populate the container with mesh in space
		discretizeSpace(h, boundary_.first, initCondition);
		// use the mesh in space to get values of initial condition
		discretizeInitialCondition(init_, initCondition);
		// get the correct scheme:
		if (scheme == ExplicitPDESchemes::Euler) {
			ExplicitHeatEulerScheme<T> euler{ spacer_.lower(),terminalT_,
											std::make_pair(k,h),
											coeffs_,schemeCoeffs,
											initCondition,source_,isSourceSet_ };
			euler(boundary_, solution);
		}
		else if(scheme == ExplicitPDESchemes::ADEBarakatClark) {
			ADEHeatBakaratClarkScheme<T> adebc{ spacer_.lower(),terminalT_,
												std::make_pair(k,h),
												schemeCoeffs,initCondition,
												source_,isSourceSet_ };
			adebc(boundary_, solution);
		}
		else {
			ADEHeatSaulyevScheme<T> ades{ spacer_.lower(),terminalT_,
											std::make_pair(k,h),
											schemeCoeffs,initCondition,
											source_,isSourceSet_ };
			ades(boundary_, solution);
		}
	}


	// ==============================================================================================================
	// ================== Explicit1DSpaceVariableGeneralHeatEquation (Robin) implementation =========================
	// ==============================================================================================================

	template<typename T,
			template<typename, typename> typename Container,
			typename Alloc>
		void explicit_solvers::Explicit1DSpaceVariableGeneralHeatEquation<T, BoundaryConditionType::Robin, Container, Alloc>::
		solve(Container<T, Alloc> &solution) {

		LSS_ASSERT(solution.size() > 0, "The input solution container must be initialized.");
		// get space step:
		T const h = spaceStep();
		// get time step:
		T const k = timeStep();
		// calculate scheme const coefficients:
		T const lambda = k / (h*h);
		T const gamma = k / (2.0*h);
		T const delta = 0.5*k;
		// save scheme variable coefficients:
		auto const &a = std::get<0>(coeffs_);
		auto const &b = std::get<1>(coeffs_);
		auto const &c = std::get<2>(coeffs_);
		// prepare space variable coefficients:
		auto const &A = [&](T x) {return (lambda*a(x) - gamma * b(x)); };
		auto const &B = [&](T x) {return (lambda*a(x) - delta * c(x)); };
		auto const &D = [&](T x) {return (lambda*a(x) + gamma * b(x)); };
		// wrap up the scheme coefficients:
		auto schemeCoeffs = std::make_tuple(A, B, D);
		// create container to carry mesh in space and then previous solution:
		Container<T, Alloc> initCondition(spaceN_ + 1, T{});
		// populate the container with mesh in space
		discretizeSpace(h, spacer_.lower(), initCondition);
		// use the mesh in space to get values of initial condition
		discretizeInitialCondition(init_, initCondition);
		// get the correct scheme:
		// Here we have only ExplicitEulerScheme available
		ExplicitHeatEulerScheme<T> euler{ spacer_.lower(),terminalT_,
										std::make_pair(k,h),
										coeffs_,schemeCoeffs,
										initCondition,source_,isSourceSet_ };
		euler(left_, right_, solution);
		
	}

}






#endif //_LSS_ONE_DIM_SPACE_VARIABLE_GENERAL_HEAT_EQUATION_SOLVERS