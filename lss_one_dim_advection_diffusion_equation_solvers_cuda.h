#pragma once
#if !defined(_LSS_ONE_DIM_ADVECTION_DIFFUSION_EQUATION_SOLVERS_CUDA)
#define _LSS_ONE_DIM_ADVECTION_DIFFUSION_EQUATION_SOLVERS_CUDA

#include"lss_types.h"
#include"lss_utility.h"
#include"lss_one_dim_pde_utility.h"
#include"lss_one_dim_pde_schemes.h"
#include"lss_one_dim_pde_schemes_cuda.h"
#include"lss_sparse_solvers_cuda.h"


namespace lss_one_dim_advection_diffusion_equation_solvers_cuda {

	using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
	using lss_one_dim_pde_schemes::ImplicitAdvectionDiffusionEquationSchemes;
	using lss_types::BoundaryConditionType;
	using lss_types::MemorySpace;
	using lss_types::ImplicitPDESchemes;
	using lss_utility::Range;
	using lss_utility::FlatMatrix;
	using lss_one_dim_pde_utility::Discretization;
	using lss_one_dim_pde_schemes_cuda::ExplicitEulerHeatEquationScheme;

	namespace implicit_solvers {

		// ==============================================================================================================
		// =========================== Implicit1DAdvectionDiffusionEquationCUDA General Template ========================
		// ==============================================================================================================


		template<typename T,
				BoundaryConditionType BType,
				MemorySpace MemSpace,
				template<MemorySpace, typename> typename RealSparsePolicyCUDA,
				template<typename, typename> typename Container,
				typename Alloc>
		class Implicit1DAdvectionDiffusionEquationCUDA {};

		// ==============================================================================================================
		// =================== Implicit1DHeatEquationCUDA Dirichlet Specialisation Template =============================
		// ==============================================================================================================

		template<typename T,
				MemorySpace MemSpace,
				template<MemorySpace, typename> typename RealSparsePolicyCUDA,
				template<typename, typename> typename Container,
				typename Alloc>
		class Implicit1DAdvectionDiffusionEquationCUDA<T, BoundaryConditionType::Dirichlet, MemSpace, RealSparsePolicyCUDA, Container, Alloc> :
		public Discretization<T, Container, Alloc> {
			private:
				RealSparsePolicyCUDA<MemSpace, T> solver_;									// finite-difference solver
				Range<T> spacer_;															// space range
				T terminalT_;																// terminal time
				std::size_t timeN_;															// number of time subdivisions
				std::size_t spaceN_;														// number of space subdivisions
				std::function<T(T)> init_;													// initi condition
				std::pair<T, T> boundary_;													// boundaries
				T diffusivity_;																// diffusivity = c^2 in PDE
				T convection_;																// convection coefficient in PDE

			public:
				typedef T value_type;
				explicit Implicit1DAdvectionDiffusionEquationCUDA() = delete;
				explicit Implicit1DAdvectionDiffusionEquationCUDA(Range<T> const &spaceRange,
																T terminalTime,
																std::size_t const &spaceDiscretization,
																std::size_t const &timeDiscretization)
					:spacer_{ spaceRange },
					terminalT_{ terminalTime },
					timeN_{ timeDiscretization },
					spaceN_{ spaceDiscretization } {}

				~Implicit1DAdvectionDiffusionEquationCUDA() {}

				Implicit1DAdvectionDiffusionEquationCUDA(Implicit1DAdvectionDiffusionEquationCUDA const &) = delete;
				Implicit1DAdvectionDiffusionEquationCUDA(Implicit1DAdvectionDiffusionEquationCUDA &&) = delete;
				Implicit1DAdvectionDiffusionEquationCUDA& operator=(Implicit1DAdvectionDiffusionEquationCUDA const&) = delete;
				Implicit1DAdvectionDiffusionEquationCUDA& operator=(Implicit1DAdvectionDiffusionEquationCUDA &&) = delete;

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

				inline void setConvection(T value) {
					convection_ = value;
				}

				void solve(Container<T, Alloc> &solution,
					ImplicitPDESchemes scheme = ImplicitPDESchemes::CrankNicolson);
		};


		// ==============================================================================================================
		// =================== Implicit1DAdvectionDiffusionEquationCUDA Robin Specialisation Template ===================
		// ==============================================================================================================

		template<typename T,
				MemorySpace MemSpace,
				template<MemorySpace, typename> typename RealSparsePolicyCUDA,
				template<typename, typename> typename Container,
				typename Alloc>
		class Implicit1DAdvectionDiffusionEquationCUDA<T, BoundaryConditionType::Robin, MemSpace, RealSparsePolicyCUDA, Container, Alloc> :
		public Discretization<T, Container, Alloc> {
			private:
				RealSparsePolicyCUDA<MemSpace, T> solver_;									// finite-difference solver
				Range<T> spacer_;															// space range
				T terminalT_;																// terminal time
				std::size_t timeN_;															// number of time subdivisions
				std::size_t spaceN_;														// number of space subdivisions
				std::function<T(T)> init_;													// initi condition
				std::pair<T, T> leftBoundary_;												// left boundaries
				std::pair<T, T> rightBoundary_;												// right boundaries
				T diffusivity_;																// diffusivity = c^2 in PDE
				T convection_;																// convection coefficient in PDE

			public:
				typedef T value_type;
				explicit Implicit1DAdvectionDiffusionEquationCUDA() = delete;
				explicit Implicit1DAdvectionDiffusionEquationCUDA(Range<T> const &spaceRange,
																T terminalTime,
																std::size_t const &spaceDiscretization,
																std::size_t const &timeDiscretization)
					:spacer_{ spaceRange },
					terminalT_{ terminalTime },
					timeN_{ timeDiscretization },
					spaceN_{ spaceDiscretization } {}

				~Implicit1DAdvectionDiffusionEquationCUDA() {}

				Implicit1DAdvectionDiffusionEquationCUDA(Implicit1DAdvectionDiffusionEquationCUDA const &) = delete;
				Implicit1DAdvectionDiffusionEquationCUDA(Implicit1DAdvectionDiffusionEquationCUDA &&) = delete;
				Implicit1DAdvectionDiffusionEquationCUDA& operator=(Implicit1DAdvectionDiffusionEquationCUDA const&) = delete;
				Implicit1DAdvectionDiffusionEquationCUDA& operator=(Implicit1DAdvectionDiffusionEquationCUDA &&) = delete;

				inline T spaceStep()const { return (spacer_.spread() / static_cast<T>(spaceN_)); }
				inline T timeStep()const { return (terminalT_ / static_cast<T>(timeN_)); }

				inline void setBoundaryCondition(std::pair<T, T> const &left, std::pair<T, T> const &right) {
					leftBoundary_ = left;
					T beta_ = 1.0 / right.first;
					T psi_ = -1.0*right.second / right.first;
					rightBoundary_ = std::make_pair(beta_, psi_);
				}

				inline void setInitialCondition(std::function<T(T)> const &initialCondition) {
					init_ = initialCondition;
				}

				inline void setThermalDiffusivity(T value) {
					diffusivity_ = value;
				}

				inline void setConvection(T value) {
					convection_ = value;
				}

				void solve(Container<T, Alloc> &solution,
					ImplicitPDESchemes scheme = ImplicitPDESchemes::CrankNicolson);
		};


	}

	namespace explicit_solvers {






	}


	// ====================================== IMPLEMENTATIONS =======================================================

	// ==============================================================================================================
	// =================== Implicit1DAdvectionDiffusionEquationCUDA (Dirichlet) implementation ======================
	// ==============================================================================================================

	template<typename T,
			MemorySpace MemSpace,
			template<MemorySpace, typename> typename RealSparsePolicyCUDA,
			template<typename, typename> typename Container,
			typename Alloc>
	void implicit_solvers::Implicit1DAdvectionDiffusionEquationCUDA<T, BoundaryConditionType::Dirichlet, MemSpace, RealSparsePolicyCUDA, Container, Alloc>::
		solve(Container<T, Alloc> &solution, ImplicitPDESchemes scheme) {

		LSS_ASSERT(solution.size() > 0, "The input solution container must be initialized.");
		// get the correct scheme:
		auto schemeFun = ImplicitAdvectionDiffusionEquationSchemes<T>::getSchemeCUDA(BoundaryConditionType::Dirichlet, scheme);
		// get correct theta according to the scheme:
		T const theta = ImplicitAdvectionDiffusionEquationSchemes<T>::getTheta(scheme);
		// get space step:
		T const h = spaceStep();
		// get time step:
		T const k = timeStep();
		// calculate lambda:
		T const lambda = (diffusivity_ *  k) / (h*h);
		// calculate gamma:
		T const gamma = (convection_ * k) / (2.0*h);
		// store size of matrix:
		std::size_t const m = spaceN_ - 1;
		// create container to carry mesh in space and then previous solution:
		Container<T, Alloc> prevSol(m, T{});
		// populate the container with mesh in space
		discretizeSpace(h, (spacer_.lower() + h), prevSol);
		// use the mesh in space to get values of initial condition
		discretizeInitialCondition(init_, prevSol);
		// first create and populate the sparse matrix:
		FlatMatrix<T> fsm;
		fsm.setColumns(m); fsm.setRows(m);
		// populate the matrix:
		fsm.emplace_back(0, 0, (1.0 + 2.0*lambda*theta));
		fsm.emplace_back(0, 1, -1.0*(lambda - gamma)*theta);
		for (std::size_t t = 1; t < m - 1; ++t) {
			fsm.emplace_back(t, t - 1, -1.0*(lambda + gamma)*theta);
			fsm.emplace_back(t, t, (1.0 + 2.0*lambda*theta));
			fsm.emplace_back(t, t + 1, -1.0*(lambda - gamma)*theta);
		}
		fsm.emplace_back(m - 1, m - 2, -1.0*(lambda + gamma)*theta);
		fsm.emplace_back(m - 1, m - 1, (1.0 + 2.0*lambda*theta));
		Container<T, Alloc> rhs(m, T{});
		// create container to carry new solution:
		Container<T, Alloc> nextSol(m, T{});
		// create first time point:
		T time = k;
		// store terminal time:
		T const lastTime = terminalT_;
		// initialize the solver:
		solver_.initialize(m);
		// insert sparse matrix A and vector b:
		solver_.setFlatSparseMatrix(std::move(fsm));
		// loop for stepping in time:
		while (time <= lastTime) {
			schemeFun(lambda, gamma, prevSol, rhs, boundary_, std::pair<T, T>());
			solver_.setRhs(rhs);
			solver_.solve(nextSol);
			prevSol = nextSol;
			time += k;
		}
		// copy into solution vector
		solution[0] = boundary_.first;
		std::copy(prevSol.begin(), prevSol.end(), std::next(solution.begin()));
		solution[solution.size() - 1] = boundary_.second;
	}



	// ==============================================================================================================
	// ===================== Implicit1DAdvectionDiffusionEquationCUDA (Robin BC) implementation =====================
	// ==============================================================================================================

	// not yet modified !!!!
	template<typename T,
			MemorySpace MemSpace,
			template<MemorySpace, typename> typename RealSparsePolicyCUDA,
			template<typename, typename> typename Container,
			typename Alloc>
	void implicit_solvers::Implicit1DAdvectionDiffusionEquationCUDA<T, BoundaryConditionType::Robin, MemSpace, RealSparsePolicyCUDA, Container, Alloc>::
		solve(Container<T, Alloc> &solution, ImplicitPDESchemes scheme) {

		LSS_ASSERT(solution.size() > 0, "The input solution container must be initialized.");
		// get the correct scheme:
		auto schemeFun = ImplicitHeatEquationSchemes<T>::getSchemeCUDA(BoundaryConditionType::Robin, scheme);
		// get correct theta according to the scheme:
		T const theta = ImplicitHeatEquationSchemes<T>::getTheta(scheme);
		// get space step:
		T const h = spaceStep();
		// get time step:
		T const k = timeStep();
		// calculate lambda:
		T const lambda = (diffusivity_ *  k) / (h*h);
		// calculate gamma:
		T const gamma = (convection_ * k) / (2.0*h);
		// store size of matrix:
		std::size_t const m = spaceN_ + 1;
		// create container to carry mesh in space and then previous solution:
		Container<T, Alloc> prevSol(m, T{});
		// populate the container with mesh in space
		discretizeSpace(h, spacer_.lower(), prevSol);
		// use the mesh in space to get values of initial condition
		discretizeInitialCondition(init_, prevSol);
		// first create and populate the sparse matrix:
		FlatMatrix<T> fsm;
		fsm.setColumns(m); fsm.setRows(m);
		// populate the matrix:
		fsm.emplace_back(0, 0, (1.0 + 2.0*lambda*theta));
		fsm.emplace_back(0, 1, -1.0*lambda*theta*(1.0 + leftBoundary_.first));
		for (std::size_t t = 1; t < m - 1; ++t) {
			fsm.emplace_back(t, t - 1, -1.0*lambda*theta);
			fsm.emplace_back(t, t, (1.0 + 2.0*lambda*theta));
			fsm.emplace_back(t, t + 1, -1.0*lambda*theta);
		}
		fsm.emplace_back(m - 1, m - 2, -1.0*lambda*theta*(1.0 + rightBoundary_.first));
		fsm.emplace_back(m - 1, m - 1, (1.0 + 2.0*lambda*theta));
		Container<T, Alloc> rhs(m, T{});
		// create container to carry new solution:
		Container<T, Alloc> nextSol(m, T{});
		// create first time point:
		T time = k;
		// store terminal time:
		T const lastTime = terminalT_;
		// initialize the solver:
		solver_.initialize(m);
		// insert sparse matrix A and vector b:
		solver_.setFlatSparseMatrix(std::move(fsm));
		// loop for stepping in time:
		while (time <= lastTime) {
			schemeFun(lambda, T{}, prevSol, rhs, leftBoundary_, rightBoundary_);
			solver_.setRhs(rhs);
			solver_.solve(nextSol);
			prevSol = nextSol;
			time += k;
		}
		// copy into solution vector
		std::copy(prevSol.begin(), prevSol.end(), solution.begin());
	}



}



















#endif ///_LSS_ONE_DIM_ADVECTION_DIFFUSION_EQUATION_SOLVERS_CUDA