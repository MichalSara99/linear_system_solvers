#pragma once
#if !defined(_LSS_ONE_DIM_HEAT_EQUATION_SOLVERS)
#define _LSS_ONE_DIM_HEAT_EQUATION_SOLVERS

#include<functional>
#include"lss_types.h"
#include"lss_utility.h"
#include"lss_macros.h"
#include"lss_one_dim_pde_schemes.h"


namespace lss_one_dim_heat_equation_solvers {

	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
	using lss_types::ExplicitPDESchemes;
	using lss_one_dim_pde_schemes::ImplicitHeatEquationSchemes;
	using lss_one_dim_pde_schemes::ExplicitEulerScheme;
	using lss_one_dim_pde_schemes::ADEBakaratClarkScheme;
	using lss_utility::Range;

	namespace implicit_solvers {


		template<typename T,
			BoundaryConditionType BType,
			template<typename,
					BoundaryConditionType,
					template<typename, typename> typename Cont,
					typename> typename FDMSolver,
			template<typename, typename> typename Container,
			typename Alloc>
		class Implicit1DHeatEquation{};




		template<typename T,
				template<typename,
						BoundaryConditionType,
						template<typename, typename> typename Cont,
						typename> typename FDMSolver,
				template<typename, typename> typename Container,
				typename Alloc>
		class Implicit1DHeatEquation<T,BoundaryConditionType::Dirichlet, FDMSolver,Container,Alloc> {
		private:
			FDMSolver<T, BoundaryConditionType::Dirichlet, Container, Alloc> fdmSolver_;// finite-difference solver
			Range<T> spacer_;															// space range
			T terminalT_;																// terminal time
			std::size_t timeN_;															// number of time subdivisions
			std::size_t spaceN_;														// number of space subdivisions
			std::function<T(T)> init_;													// initi condition
			std::pair<T, T> boundary_;													// boundaries
			T diffusivity_;																// diffusivity = c^2 in PDE
			void createSpaceMesh(Container<T,Alloc> &container)const;
			void discretizeInitialCondition(Container<T, Alloc> &xinput)const;

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
				spaceN_{ spaceDiscretization }{}

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

			inline void setThermalDiffusivity(T value) {
				diffusivity_ = value;
			}

			void solve(Container<T,Alloc> &solution,
				ImplicitPDESchemes scheme = ImplicitPDESchemes::CrankNicolson);
		};


		template<typename T,
				template<typename,
						BoundaryConditionType,
						template<typename, typename> typename Cont,
						typename> typename FDMSolver,
				template<typename, typename> typename Container,
				typename Alloc>
		class Implicit1DHeatEquation<T, BoundaryConditionType::Robin, FDMSolver, Container, Alloc> {
		private:
			FDMSolver<T, BoundaryConditionType::Robin, Container, Alloc> fdmSolver_;	// finite-difference solver
			Range<T> spacer_;															// space range
			T terminalT_;																// terminal time
			std::size_t timeN_;															// number of time subdivisions
			std::size_t spaceN_;														// number of space subdivisions
			std::function<T(T)> init_;													// initi condition
			std::pair<T, T> boundary_;													// boundaries
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

		template<typename T,
				BoundaryConditionType BType,
				template<typename, typename> typename Container,
				typename Alloc>
		class Explicit1DHeatEquation{};


		template<typename T,
			template<typename, typename> typename Container,
			typename Alloc>
		class Explicit1DHeatEquation<T,BoundaryConditionType::Dirichlet,Container,Alloc> {
		private:
			Range<T> spacer_;											// space range
			T terminalT_;												// terminal time
			std::size_t timeN_;											// number of time subdivisions
			std::size_t spaceN_;										// number of space subdivisions
			std::function<T(T)> init_;									// initi condition
			std::pair<T, T> boundary_;									// boundaries
			T diffusivity_;												// diffusivity = c^2 in PDE
			void createSpaceMesh(Container<T, Alloc> &container)const;
			void discretizeInitialCondition(Container<T, Alloc> &xinput)const;

		public:
			explicit Explicit1DHeatEquation() = delete;
			explicit Explicit1DHeatEquation(Range<T> const &spaceRange,
											T terminalTime,
											std::size_t const &spaceDiscretization,
											std::size_t const &timeDiscretization)
				:spacer_{ spaceRange },
				terminalT_{ terminalTime },
				timeN_{ timeDiscretization },
				spaceN_{ spaceDiscretization } {}

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

			inline void setThermalDiffusivity(T value) {
				diffusivity_ = value;
			}

			void solve(Container<T, Alloc> &solution,
				ExplicitPDESchemes scheme = ExplicitPDESchemes::ADE);
		};


		template<typename T,
			template<typename, typename> typename Container,
			typename Alloc>
		class Explicit1DHeatEquation<T, BoundaryConditionType::Robin, Container, Alloc> {



		};


	}


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
		createSpaceMesh(prevSol);
		// use the mesh in space to get values of initial condition
		discretizeInitialCondition(prevSol);
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
			schemeFun(lambda, prevSol, rhs);
			fdmSolver_.setRhs(rhs);
			fdmSolver_.solve(nextSol);
			prevSol = nextSol;
			time += k;
		}
		// copy into solution vector
		std::copy(prevSol.begin(), prevSol.end(), solution.begin());
	}

	template<typename T,
			template<typename,
					BoundaryConditionType,
					template<typename, typename> typename Cont,
					typename> typename FDMSolver,
			template<typename, typename> typename Container,
			typename Alloc>
	void implicit_solvers::Implicit1DHeatEquation<T, BoundaryConditionType::Dirichlet, FDMSolver, Container, Alloc>::
		createSpaceMesh(Container<T, Alloc> &container) const {
		LSS_ASSERT(container.size() > 0, "The input container must be initialized.");
		container[0] = boundary_.first;
		T const h = spaceStep();
		for (std::size_t t = 1; t < container.size(); ++t) {
			container[t] = container[t - 1] + h;
		}
	}

	template<typename T,
			template<typename,
					BoundaryConditionType,
					template<typename, typename> typename Cont,
					typename> typename FDMSolver,
			template<typename, typename> typename Container,
			typename Alloc>
	void implicit_solvers::Implicit1DHeatEquation<T, BoundaryConditionType::Dirichlet, FDMSolver, Container, Alloc>::
		discretizeInitialCondition(Container<T, Alloc> &xinput) const {
		LSS_ASSERT(xinput.size() > 0, "The input container must be initialized.");
		for (std::size_t t = 0; t < xinput.size(); ++t) {
			xinput[t] = init_(xinput[t]);
		}
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
	}



	// ==============================================================================================================
	// ========================== Explicit1DHeatEquation (Dirichlet) implementation =================================
	// ==============================================================================================================
	template<typename T,
			template<typename, typename> typename Container,
			typename Alloc>
	void explicit_solvers::Explicit1DHeatEquation<T, BoundaryConditionType::Dirichlet, Container, Alloc>::
		createSpaceMesh(Container<T, Alloc> &container) const {
		LSS_ASSERT(container.size() > 0, "The input container must be initialized.");
		container[0] = boundary_.first;
		T const h = spaceStep();
		for (std::size_t t = 1; t < container.size(); ++t) {
			container[t] = container[t - 1] + h;
		}
	}

	template<typename T,
			template<typename, typename> typename Container,
			typename Alloc>
	void explicit_solvers::Explicit1DHeatEquation<T, BoundaryConditionType::Dirichlet,Container, Alloc>::
		discretizeInitialCondition(Container<T, Alloc> &xinput) const {
		LSS_ASSERT(xinput.size() > 0, "The input container must be initialized.");
		for (std::size_t t = 0; t < xinput.size(); ++t) {
			xinput[t] = init_(xinput[t]);
		}
	}
	
	
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
		createSpaceMesh(initCondition);
		// use the mesh in space to get values of initial condition
		discretizeInitialCondition(initCondition);
		// get the correct scheme:
		if (scheme == ExplicitPDESchemes::Euler) {
			ExplicitEulerScheme<T> euler{ initCondition,h,k,terminalT_,diffusivity_ };
			euler(boundary_, solution);
		}
		else {
			ADEBakaratClarkScheme<T> ade{ initCondition,h,k,terminalT_,diffusivity_ };
			ade(boundary_, solution);
		}
	}


}






#endif //_LSS_ONE_DIM_HEAT_EQUATION_SOLVERS