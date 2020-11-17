#pragma once
#if !defined(_LSS_ONE_DIM_HEAT_EQUATION_SOLVERS)
#define _LSS_ONE_DIM_HEAT_EQUATION_SOLVERS

#include<functional>
#include"lss_types.h"
#include"lss_utility.h"
#include"lss_macros.h"
#include"lss_one_dim_pde_utility.h"
#include"lss_one_dim_pde_schemes.h"


namespace lss_one_dim_heat_equation_solvers {

	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_types::ExplicitPDESchemes;
	using lss_one_dim_pde_schemes::ImplicitHeatEquationSchemes;
	using lss_one_dim_pde_schemes::ExplicitHeatEulerScheme;
	using lss_one_dim_pde_schemes::ADEHeatBakaratClarkScheme;
	using lss_one_dim_pde_schemes::ADEHeatSaulyevScheme;
	using lss_utility::Range;
	using lss_one_dim_pde_utility::Discretization;

	namespace implicit_solvers {

		// ==============================================================================================================
		// ===================================== Implicit1DHeatEquation General Template ================================
		// ==============================================================================================================


		template<typename T,
			BoundaryConditionType BType,
			template<typename,
					BoundaryConditionType,
					template<typename, typename> typename Cont,
					typename> typename FDMSolver,
			template<typename, typename> typename Container,
			typename Alloc>
		class Implicit1DHeatEquation{};


		// ==============================================================================================================
		// ======================= Implicit1DHeatEquation Dirichlet Specialisation Template =============================
		// ==============================================================================================================

		template<typename T,
				template<typename,
						BoundaryConditionType,
						template<typename, typename> typename Cont,
						typename> typename FDMSolver,
				template<typename, typename> typename Container,
				typename Alloc>
		class Implicit1DHeatEquation<T,BoundaryConditionType::Dirichlet, FDMSolver,Container,Alloc>
		:public Discretization<T,Container,Alloc>{
		private:
			FDMSolver<T, BoundaryConditionType::Dirichlet, Container, Alloc> fdmSolver_;// finite-difference solver
			Range<T> spacer_;															// space range
			T terminalT_;																// terminal time
			std::size_t timeN_;															// number of time subdivisions
			std::size_t spaceN_;														// number of space subdivisions
			std::function<T(T)> init_;													// initi condition
			std::function<T(T, T)> source_;												// heat source
			std::pair<T, T> boundary_;													// boundaries
			T diffusivity_;																// diffusivity = c^2 in PDE
			bool isSourceSet_;

		public:
			typedef T value_type;
			explicit Implicit1DHeatEquation() = delete;
			explicit Implicit1DHeatEquation(Range<T> const &spaceRange,
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

			~Implicit1DHeatEquation(){}

			Implicit1DHeatEquation(Implicit1DHeatEquation const &) = delete;
			Implicit1DHeatEquation(Implicit1DHeatEquation &&) = delete;
			Implicit1DHeatEquation& operator=(Implicit1DHeatEquation const &) = delete;
			Implicit1DHeatEquation& operator=(Implicit1DHeatEquation &&) = delete;

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
			inline void setThermalDiffusivity(T value) {
				diffusivity_ = value;
			}

			void solve(Container<T,Alloc> &solution,
				ImplicitPDESchemes scheme = ImplicitPDESchemes::CrankNicolson);
		};

		// ==============================================================================================================
		// =========================== Implicit1DHeatEquation Robin Specialisation Template =============================
		// ==============================================================================================================

		template<typename T,
				template<typename,
						BoundaryConditionType,
						template<typename, typename> typename Cont,
						typename> typename FDMSolver,
				template<typename, typename> typename Container,
				typename Alloc>
		class Implicit1DHeatEquation<T, BoundaryConditionType::Robin, FDMSolver, Container, Alloc>:
			public Discretization<T, Container, Alloc> {
		private:
			FDMSolver<T, BoundaryConditionType::Robin, Container, Alloc> fdmSolver_;	// finite-difference solver
			Range<T> spacer_;															// space range
			T terminalT_;																// terminal time
			std::size_t timeN_;															// number of time subdivisions
			std::size_t spaceN_;														// number of space subdivisions
			std::function<T(T)> init_;													// initi condition
			std::pair<T, T> left_;														// left boundary pair
			std::pair<T, T> right_;														// right boundary pair
			T diffusivity_;																// diffusivity = c^2 in PDE

		public:
			typedef T value_type;
			explicit Implicit1DHeatEquation() = delete;
			explicit Implicit1DHeatEquation(Range<T> const &spaceRange,
											T terminalTime,
											std::size_t const &spaceDiscretization,
											std::size_t const &timeDiscretization)
				:fdmSolver_{ spaceDiscretization + 1 },
				spacer_{ spaceRange },
				terminalT_{ terminalTime },
				timeN_{ timeDiscretization },
				spaceN_{ spaceDiscretization } {}

			~Implicit1DHeatEquation() {}

			Implicit1DHeatEquation(Implicit1DHeatEquation const &) = delete;
			Implicit1DHeatEquation(Implicit1DHeatEquation &&) = delete;
			Implicit1DHeatEquation& operator=(Implicit1DHeatEquation const &) = delete;
			Implicit1DHeatEquation& operator=(Implicit1DHeatEquation &&) = delete;

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

			inline void setThermalDiffusivity(T value) {
				diffusivity_ = value;
			}

			void solve(Container<T, Alloc> &solution,
				ImplicitPDESchemes scheme = ImplicitPDESchemes::CrankNicolson);

		};

	}






	namespace explicit_solvers {

		// ==============================================================================================================
		// ===================================== Explicit1DHeatEquation General Template ================================
		// ==============================================================================================================

		template<typename T,
				BoundaryConditionType BType,
				template<typename, typename> typename Container,
				typename Alloc>
		class Explicit1DHeatEquation{};


		// ==============================================================================================================
		// ======================= Explicit1DHeatEquation Dirichlet Specialisation Template =============================
		// ==============================================================================================================

		template<typename T,
			template<typename, typename> typename Container,
			typename Alloc>
		class Explicit1DHeatEquation<T,BoundaryConditionType::Dirichlet,Container,Alloc>:
		public Discretization<T,Container,Alloc>{
		private:
			Range<T> spacer_;											// space range
			T terminalT_;												// terminal time
			std::size_t timeN_;											// number of time subdivisions
			std::size_t spaceN_;										// number of space subdivisions
			std::function<T(T)> init_;									// initi condition
			std::function<T(T, T)> source_;								// heat source
			std::pair<T, T> boundary_;									// boundaries
			T diffusivity_;												// diffusivity = c^2 in PDE
			bool isSourceSet_;

		public:
			typedef T value_type;
			explicit Explicit1DHeatEquation() = delete;
			explicit Explicit1DHeatEquation(Range<T> const &spaceRange,
											T terminalTime,
											std::size_t const &spaceDiscretization,
											std::size_t const &timeDiscretization)
				:spacer_{ spaceRange },
				terminalT_{ terminalTime },
				timeN_{ timeDiscretization },
				spaceN_{ spaceDiscretization },
				source_{nullptr },
				isSourceSet_{ false } {}

			~Explicit1DHeatEquation(){}

			Explicit1DHeatEquation(Explicit1DHeatEquation const &) = delete;
			Explicit1DHeatEquation(Explicit1DHeatEquation &&) = delete;
			Explicit1DHeatEquation& operator=(Explicit1DHeatEquation const &) = delete;
			Explicit1DHeatEquation& operator=(Explicit1DHeatEquation &&) = delete;

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
			inline void setThermalDiffusivity(T value) {
				diffusivity_ = value;
			}

			void solve(Container<T, Alloc> &solution,
				ExplicitPDESchemes scheme = ExplicitPDESchemes::ADEBarakatClark);
		};


		// ==============================================================================================================
		// =========================== Explicit1DHeatEquation Robin Specialisation Template =============================
		// ==============================================================================================================

		template<typename T,
			template<typename, typename> typename Container,
			typename Alloc>
		class Explicit1DHeatEquation<T, BoundaryConditionType::Robin, Container, Alloc>:
			public Discretization<T,Container,Alloc>{
		private:
			Range<T> spacer_;											// space range
			T terminalT_;												// terminal time
			std::size_t timeN_;											// number of time subdivisions
			std::size_t spaceN_;										// number of space subdivisions
			std::function<T(T)> init_;									// initi condition
			std::pair<T, T> left_;										// left boundary pair
			std::pair<T, T> right_;										// right boundary pair
			T diffusivity_;												// diffusivity = c^2 in PDE


		public:
			typedef T value_type;
			explicit Explicit1DHeatEquation() = delete;
			explicit Explicit1DHeatEquation(Range<T> const &spaceRange,
											T terminalTime,
											std::size_t const &spaceDiscretization,
											std::size_t const &timeDiscretization)
				:spacer_{ spaceRange },
				terminalT_{ terminalTime },
				timeN_{ timeDiscretization },
				spaceN_{ spaceDiscretization } {}

			~Explicit1DHeatEquation() {}

			Explicit1DHeatEquation(Explicit1DHeatEquation const &) = delete;
			Explicit1DHeatEquation(Explicit1DHeatEquation &&) = delete;
			Explicit1DHeatEquation& operator=(Explicit1DHeatEquation const &) = delete;
			Explicit1DHeatEquation& operator=(Explicit1DHeatEquation &&) = delete;

			inline T spaceStep()const { return (spacer_.spread() / static_cast<T>(spaceN_)); }
			inline T timeStep()const { return (terminalT_ / static_cast<T>(timeN_)); }

			inline void setBoundaryCondition(std::pair<T, T> const &left,std::pair<T,T> const &right) {
				left_ = left;
				right_ = right;
			}
			inline void setInitialCondition(std::function<T(T)> const &initialCondition) {
				init_ = initialCondition;
			}

			inline void setThermalDiffusivity(T value) {
				diffusivity_ = value;
			}

			void solve(Container<T, Alloc> &solution);

		};


	}


	// ====================================== IMPLEMENTATIONS =======================================================

	// ==============================================================================================================
	// ========================== Implicit1DHeatEquation (Dirichlet) implementation =================================
	// ==============================================================================================================

	template<typename T,
			template<typename,
					BoundaryConditionType,
					template<typename, typename> typename Cont,
					typename> typename FDMSolver,
			template<typename, typename> typename Container,
			typename Alloc>
	void implicit_solvers::Implicit1DHeatEquation<T, BoundaryConditionType::Dirichlet, FDMSolver, Container, Alloc>::
		solve(Container<T,Alloc> &solution, ImplicitPDESchemes scheme) {

		LSS_ASSERT(solution.size() > 0, "The input solution container must be initialized.");
		// get correct theta according to the scheme:
		T const theta = ImplicitHeatEquationSchemes<T>::getTheta(scheme);
		// get space step:
		T const h = spaceStep();
		// get time step:
		T const k = timeStep();
		// calculate lambda:
		T const lambda = (diffusivity_ *  k) / (h*h);
		// create container to carry mesh in space and then previous solution:
		Container<T, Alloc> prevSol(spaceN_ + 1, T{});
		// populate the container with mesh in space
		discretizeSpace(h, spacer_.lower(), prevSol);
		// use the mesh in space to get values of initial condition
		discretizeInitialCondition(init_, prevSol);
		// prepare containers for diagonal vectors for FDMSolver:
		Container<T, Alloc> low(spaceN_ + 1, -1.0*lambda*theta);
		Container<T, Alloc> diag(spaceN_ + 1, (1.0 + 2.0*lambda*theta));
		Container<T, Alloc> up(spaceN_ + 1, -1.0*lambda*theta);
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
			// get the correct scheme:
			auto schemeFun = ImplicitHeatEquationSchemes<T>::getInhomScheme(scheme);
			// create a container to carry discretized source heat
			Container<T, Alloc> sourceCurr(spaceN_ + 1, T{});
			Container<T, Alloc> sourceNext(spaceN_ + 1, T{});
			discretizeInSpace(h, spacer_.lower(), 0.0, source_, sourceCurr);
			discretizeInSpace(h, spacer_.lower(), time, source_, sourceNext);
			// loop for stepping in time:
			while (time <= lastTime) {
				schemeFun(lambda, T{}, k, prevSol, sourceCurr, sourceNext, rhs);
				fdmSolver_.setRhs(rhs);
				fdmSolver_.solve(nextSol);
				prevSol = nextSol;
				discretizeInSpace(h, spacer_.lower(), time, source_, sourceCurr);
				discretizeInSpace(h, spacer_.lower(), 2.0*time, source_, sourceNext);
				time += k;
			}
		}
		else {
			// get the correct scheme:
			auto schemeFun = ImplicitHeatEquationSchemes<T>::getScheme(scheme);
			// loop for stepping in time:
			while (time <= lastTime) {
				schemeFun(lambda, T{}, prevSol, rhs);
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
	// ============================== Implicit1DHeatEquation (Robin) implementation =================================
	// ==============================================================================================================


	template<typename T,
			template<typename,
					BoundaryConditionType,
					template<typename, typename> typename Cont,
					typename> typename FDMSolver,
			template<typename, typename> typename Container,
			typename Alloc>
	void implicit_solvers::Implicit1DHeatEquation<T, BoundaryConditionType::Robin, FDMSolver, Container, Alloc>::
		solve(Container<T, Alloc> &solution, ImplicitPDESchemes scheme) {

		LSS_ASSERT(solution.size() > 0, "The input solution container must be initialized.");
		// get the correct scheme:
		auto schemeFun = ImplicitHeatEquationSchemes<T>::getScheme(scheme);
		// get correct theta according to the scheme:
		T const theta = ImplicitHeatEquationSchemes<T>::getTheta(scheme);
		// get space step:
		T const h = spaceStep();
		// get time step:
		T const k = timeStep();
		// calculate lambda:
		T const lambda = (diffusivity_ *  k) / (h*h);
		// create container to carry mesh in space and then previous solution:
		Container<T, Alloc> prevSol(spaceN_ + 1, T{});
		// populate the container with mesh in space
		discretizeSpace(h,spacer_.lower(), prevSol);
		// use the mesh in space to get values of initial condition
		discretizeInitialCondition(init_, prevSol);
		// prepare containers for diagonal vectors for FDMSolver:
		Container<T, Alloc> low(spaceN_ + 1, -1.0*lambda*theta);
		Container<T, Alloc> diag(spaceN_ + 1, (1.0 + 2.0*lambda*theta));
		Container<T, Alloc> up(spaceN_ + 1, -1.0*lambda*theta);
		Container<T, Alloc> rhs(spaceN_ + 1, T{});
		// create container to carry new solution:
		Container<T, Alloc> nextSol(spaceN_ + 1, T{});
		// create first time point:
		T time = k;
		// store terminal time:
		T const lastTime = terminalT_;
		// set properties of FDMSolver:
		fdmSolver_.setDiagonals(std::move(low), std::move(diag), std::move(up));
		// loop for stepping in time:
		while (time <= lastTime) {
			schemeFun(lambda, T{}, prevSol, rhs);
			fdmSolver_.setRhs(rhs);
			fdmSolver_.solve(nextSol);
			prevSol = nextSol;
			time += k;
		}
		// copy into solution vector
		std::copy(prevSol.begin(), prevSol.end(), solution.begin());
	}


	// ==============================================================================================================
	// ========================== Explicit1DHeatEquation (Dirichlet) implementation =================================
	// ==============================================================================================================

	
	template<typename T,
			template<typename, typename> typename Container,
			typename Alloc>
		void explicit_solvers::Explicit1DHeatEquation<T, BoundaryConditionType::Dirichlet, Container, Alloc>::
		solve(Container<T, Alloc> &solution, ExplicitPDESchemes scheme) {

		LSS_ASSERT(solution.size() > 0, "The input solution container must be initialized.");
		// get space step:
		T const h = spaceStep();
		// get time step:
		T const k = timeStep();
		// create container to carry mesh in space and then previous solution:
		Container<T, Alloc> initCondition(spaceN_ + 1, T{});
		// populate the container with mesh in space
		discretizeSpace(h, boundary_.first, initCondition);
		// use the mesh in space to get values of initial condition
		discretizeInitialCondition(init_, initCondition);
		// get the correct scheme:
		if (scheme == ExplicitPDESchemes::Euler) {
			ExplicitHeatEulerScheme<T> euler{ initCondition,h,k,terminalT_,diffusivity_,isSourceSet_,source_ };
			euler(boundary_, solution);
		}
		else if(scheme == ExplicitPDESchemes::ADEBarakatClark) {
			ADEHeatBakaratClarkScheme<T> adebc{ initCondition,h,k,terminalT_,diffusivity_,isSourceSet_,source_ };
			adebc(boundary_, solution);
		}
		else {
			ADEHeatSaulyevScheme<T> ades{ initCondition,h,k,terminalT_,diffusivity_,isSourceSet_,source_ };
			ades(boundary_, solution);
		}
	}


	// ==============================================================================================================
	// ========================== Explicit1DHeatEquation (Robin) implementation =================================
	// ==============================================================================================================

	template<typename T,
			template<typename, typename> typename Container,
			typename Alloc>
		void explicit_solvers::Explicit1DHeatEquation<T, BoundaryConditionType::Robin, Container, Alloc>::
		solve(Container<T, Alloc> &solution) {

		LSS_ASSERT(solution.size() > 0, "The input solution container must be initialized.");
		// get space step:
		T const h = spaceStep();
		// get time step:
		T const k = timeStep();
		// create container to carry mesh in space and then previous solution:
		Container<T, Alloc> initCondition(spaceN_ + 1, T{});
		// populate the container with mesh in space
		discretizeSpace(h, spacer_.lower(), initCondition);
		// use the mesh in space to get values of initial condition
		discretizeInitialCondition(init_, initCondition);
		// get the correct scheme:
		// Here we have only ExplicitEulerScheme available
		ExplicitHeatEulerScheme<T> euler{ initCondition,h,k,terminalT_,diffusivity_ };
		euler(left_, right_, solution);
		
	}

}






#endif //_LSS_ONE_DIM_HEAT_EQUATION_SOLVERS