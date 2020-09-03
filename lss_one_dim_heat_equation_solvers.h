#pragma once
#if !defined(_LSS_ONE_DIM_HEAT_EQUATION_SOLVERS)
#define _LSS_ONE_DIM_HEAT_EQUATION_SOLVERS

#include<functional>
#include"lss_types.h"
#include"lss_utility.h"


namespace lss_one_dim_heat_equation_solvers {

	using lss_types::BoundaryConditionType;
	using lss_types::ImplicitPDESchemes;
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
			FDMSolver<T, BoundaryConditionType::Dirichlet, Container, Alloc> solver_;	// finite-difference solver
			Range<T> spacer_;															// space range
			Range<T> spacet_;															// time range
			std::size_t timeN_;															// number of time subdivisions
			std::size_t spaceN_;														// number of space subdivisions
			std::function<T(T)> init_;													// initi condition

		public:
			typedef T value_type;
			explicit Implicit1DHeatEquation() = delete;
			explicit Implicit1DHeatEquation(Range<T> const &spaceRange,
											Range<T> const &timeSpace,
											std::size_t const &spaceDiscretization,
											std::size_t const &timeDiscretization)
				:spacer_{ spaceRange }, 
				spacet_{ timeSpace },
				spaceN_{ spaceDiscretization },
				timeN_{ timeDiscretization },
				solver_{ spaceDiscretization } {}

			~Implicit1DHeatEquation(){}

			Implicit1DHeatEquation(Implicit1DHeatEquation const &) = delete;
			Implicit1DHeatEquation(Implicit1DHeatEquation &&) = delete;
			Implicit1DHeatEquation& operator=(Implicit1DHeatEquation const &) = delete;
			Implicit1DHeatEquation& operator=(Implicit1DHeatEquation &&) = delete;

			inline T spaceStep()const { return (spacer_.spread() / static_cast<T>(spaceN_)); }
			inline T timeStep()const { return (spacet_.spread() / static_cast<T>(timeN_)); }

			inline void setBoundaryCondition(std::pair<T, T> const &boundaryPair) { 
				solver_.setBoundaryCondition(boundaryPair); 
			}
			inline void setInitialCondition(std::function<T(T)> const &initialCondition) {
				init_ = initialCondition; 
			}

			template<ImplicitPDESchemes Scheme>
			void solve(Container<T, Alloc> &solution);
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
			FDMSolver<T, BoundaryConditionType::Robin, Container, Alloc> solver_;		// finite-difference solver
			Range<T> spacer_;															// space range
			Range<T> spacet_;															// time range
			std::size_t timeN_;															// number of time subdivisions
			std::size_t spaceN_;														// number of space subdivisions
			std::function<T(T)> init_;													// initi condition

		public:
			typedef T value_type;
			explicit Implicit1DHeatEquation() = delete;
			explicit Implicit1DHeatEquation(Range<T> const &spaceRange,
											Range<T> const &timeSpace,
											std::size_t const &spaceDiscretization,
											std::size_t const &timeDiscretization)
				:spacer_{ spaceRange },
				spacet_{ timeSpace },
				spaceN_{ spaceDiscretization },
				timeN_{ timeDiscretization },
				solver_{ spaceDiscretization } {}

			~Implicit1DHeatEquation() {}

			Implicit1DHeatEquation(Implicit1DHeatEquation const &) = delete;
			Implicit1DHeatEquation(Implicit1DHeatEquation &&) = delete;
			Implicit1DHeatEquation& operator=(Implicit1DHeatEquation const &) = delete;
			Implicit1DHeatEquation& operator=(Implicit1DHeatEquation &&) = delete;

			inline T spaceStep()const { return (spacer_.spread() / static_cast<T>(spaceN_)); }
			inline T timeStep()const { return (spacet_.spread() / static_cast<T>(timeN_)); }

			inline void setBoundaryCondition(std::pair<T, T> const &left,std::pair<T,T> const &right) { 
				solver_.setBoundaryCondition(left, right);
			}
			inline void setInitialCondition(std::function<T(T)> const &initialCondition) { 
				init_ = initialCondition; 
			}

			template<ImplicitPDESchemes Scheme>
			void solve(Container<T, Alloc> &solution);

		};




	}

	namespace explicit_solvers {


	}
}

#endif //_LSS_ONE_DIM_HEAT_EQUATION_SOLVERS