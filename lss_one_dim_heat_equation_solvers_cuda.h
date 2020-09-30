#pragma once
#if !defined(_LSS_ONE_DIM_HEAT_EQUATION_SOLVERS_CUDA)
#define _LSS_ONE_DIM_HEAT_EQUATION_SOLVERS_CUDA


#include"lss_types.h"
#include"lss_utility.h"
#include"lss_one_dim_pde_utility.h"
#include"lss_one_dim_pde_schemes.h"
#include"lss_sparse_solvers_cuda.h"


namespace lss_one_dim_heat_equation_solvers_cuda {

	using lss_sparse_solvers_cuda::RealSparseSolverCUDA;
	using lss_one_dim_pde_schemes::ImplicitHeatEquationSchemes;
	using lss_types::BoundaryConditionType;
	using lss_types::MemorySpace;
	using lss_types::ImplicitPDESchemes;
	using lss_utility::Range;
	using lss_utility::FlatMatrix;
	using lss_one_dim_pde_utility::Discretization;

	namespace implicit_solvers {


		template<typename T,
				BoundaryConditionType BType,
				MemorySpace MemSpace,
				template<MemorySpace,typename> typename RealSparsePolicyCUDA,
				template<typename,typename> typename Container,
				typename Alloc>
		class Implicit1DHeatEquationCUDA{};



		template<typename T,
				MemorySpace MemSpace,
				template<MemorySpace, typename> typename RealSparsePolicyCUDA,
				template<typename, typename> typename Container,
				typename Alloc>
		class Implicit1DHeatEquationCUDA<T,BoundaryConditionType::Dirichlet, MemSpace, RealSparsePolicyCUDA,Container,Alloc>:
		public Discretization<T,Container,Alloc> {
		private:
			RealSparsePolicyCUDA<MemSpace, T> solver_;									// finite-difference solver
			Range<T> spacer_;															// space range
			T terminalT_;																// terminal time
			std::size_t timeN_;															// number of time subdivisions
			std::size_t spaceN_;														// number of space subdivisions
			std::function<T(T)> init_;													// initi condition
			std::pair<T, T> boundary_;													// boundaries
			T diffusivity_;																// diffusivity = c^2 in PDE
			void discretizeSpace(T const &step,
								std::pair<T, T> const &dirichletBC,
								Container<T, Alloc> & container)const override;
			void discretizeInitialCondition(std::function<T(T)> const &init,
											Container<T, Alloc> &container)const override;


		public:
			typedef T value_type;
			explicit Implicit1DHeatEquationCUDA() = delete;
			explicit Implicit1DHeatEquationCUDA(Range<T> const &spaceRange,
												T terminalTime,
												std::size_t const &spaceDiscretization,
												std::size_t const &timeDiscretization)
				:spacer_{ spaceRange },
				terminalT_{ terminalTime },
				timeN_{ timeDiscretization },
				spaceN_{ spaceDiscretization } {}

			~Implicit1DHeatEquationCUDA() {}

			Implicit1DHeatEquationCUDA(Implicit1DHeatEquationCUDA const &) = delete;
			Implicit1DHeatEquationCUDA(Implicit1DHeatEquationCUDA &&) = delete;
			Implicit1DHeatEquationCUDA& operator=(Implicit1DHeatEquationCUDA const&) = delete;
			Implicit1DHeatEquationCUDA& operator=(Implicit1DHeatEquationCUDA &&) = delete;

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
				ImplicitPDESchemes scheme = ImplicitPDESchemes::CrankNicolson);
		};


		// to be done soon:
		template<typename T,
			MemorySpace MemSpace,
			template<MemorySpace, typename> typename RealSparsePolicyCUDA,
			template<typename, typename> typename Container,
			typename Alloc>
			class Implicit1DHeatEquationCUDA<T, BoundaryConditionType::Robin, MemSpace, RealSparsePolicyCUDA, Container, Alloc> {
			private:
				RealSparsePolicyCUDA<MemSpace, T> solver_;									// finite-difference solver
				Range<T> spacer_;															// space range
				T terminalT_;																// terminal time
				std::size_t timeN_;															// number of time subdivisions
				std::size_t spaceN_;														// number of space subdivisions
				std::function<T(T)> init_;													// initi condition
				std::pair<T, T> leftPair_;													// boundaries
				std::pair<T, T> rightPair_;													// boundaries
				T diffusivity_;																// diffusivity = c^2 in PDE
				//void createSpaceMesh(Container<T, Alloc> &container)const;
				//void discretizeInitialCondition(Container<T, Alloc> &xinput)const;


			public:
				typedef T value_type;
				explicit Implicit1DHeatEquationCUDA() = delete;
				explicit Implicit1DHeatEquationCUDA(Range<T> const &spaceRange,
													T terminalTime,
													std::size_t const &spaceDiscretization,
													std::size_t const &timeDiscretization)
					:spacer_{ spaceRange },
					terminalT_{ terminalTime },
					timeN_{ timeDiscretization },
					spaceN_{ spaceDiscretization } {}

				~Implicit1DHeatEquationCUDA() {}

				Implicit1DHeatEquationCUDA(Implicit1DHeatEquationCUDA const &) = delete;
				Implicit1DHeatEquationCUDA(Implicit1DHeatEquationCUDA &&) = delete;
				Implicit1DHeatEquationCUDA& operator=(Implicit1DHeatEquationCUDA const&) = delete;
				Implicit1DHeatEquationCUDA& operator=(Implicit1DHeatEquationCUDA &&) = delete;

				inline T spaceStep()const { return (spacer_.spread() / static_cast<T>(spaceN_)); }
				inline T timeStep()const { return (terminalT_ / static_cast<T>(timeN_)); }

				inline void setBoundaryCondition(std::pair<T, T> const &left, std::pair<T, T> const &right) {
					leftPair_ = left;
					rightPair_ = right;
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
				template<typename,typename> typename Container,
				typename Alloc>
		class Explicit1DHeatEquationCUDA {};


		template<typename T,
				template<typename,typename> typename Container,
				typename Alloc>
		class Explicit1DHeatEquationCUDA<T, BoundaryConditionType::Dirichlet, Container, Alloc>:
		public Discretization<T,Container,Alloc>{
		private:
			Range<T> spacer_;
			T terminalT_;
			std::size_t timeN_;
			std::size_t spaceN_;
			std::function<T(T)> init_;
			std::pair<T, T> boundary_;
			T diffusivity_;
			void discretizeSpace(T const &step,
								std::pair<T, T> const &dirichletBC,
								Container<T, Alloc> & container)const override;
			void discretizeInitialCondition(std::function<T(T)> const &init,
											Container<T, Alloc> &container)const override;

		public:
			typedef T value_type;
			explicit Explicit1DHeatEquationCUDA() = delete;
			explicit Explicit1DHeatEquationCUDA(Range<T> const &spaceRange,
												T terminalTime,
												std::size_t const &spaceDiscretization,
												std::size_t const &timeDiscretization)
				:spacer_{ spaceRange },
				terminalT_{ terminalTime },
				timeN_{ timeDiscretization },
				spaceN_{ spaceDiscretization } {}

			~Explicit1DHeatEquationCUDA() {}

			Explicit1DHeatEquationCUDA(Explicit1DHeatEquationCUDA const &) = delete;
			Explicit1DHeatEquationCUDA(Explicit1DHeatEquationCUDA &&) = delete;
			Explicit1DHeatEquationCUDA& operator=(Explicit1DHeatEquationCUDA const&) = delete;
			Explicit1DHeatEquationCUDA& operator=(Explicit1DHeatEquationCUDA &&) = delete;

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

			// stability check:
			bool inline isStable()const override
			{ return ((2.0*diffusivity_*timeStep() / (spaceStep()*spaceStep())) <= 1.0); }

			void solve(Container<T, Alloc> &solution);
		};



		template<typename T,
			template<typename, typename> typename Container,
			typename Alloc>
		class Explicit1DHeatEquationCUDA<T, BoundaryConditionType::Robin, Container, Alloc> {


		};



	}



	// ==============================================================================================================
	// ======================== Implicit1DHeatEquationCUDA (Dirichlet) implementation ===============================
	// ==============================================================================================================

	template<typename T,
			MemorySpace MemSpace,
			template<MemorySpace, typename> typename RealSparsePolicyCUDA,
			template<typename, typename> typename Container,
			typename Alloc>
	void implicit_solvers::Implicit1DHeatEquationCUDA<T, BoundaryConditionType::Dirichlet, MemSpace, RealSparsePolicyCUDA, Container, Alloc>::
		discretizeSpace(T const &step,
						std::pair<T, T> const &dirichletBC,
						Container<T, Alloc> & container)const {
		LSS_ASSERT(container.size() > 0, "The input container must be initialized.");
		container[0] = dirichletBC.first + step;
		for (std::size_t t = 1; t < container.size(); ++t) {
			container[t] = container[t - 1] + step;
		}
	}

	template<typename T,
			MemorySpace MemSpace,
			template<MemorySpace, typename> typename RealSparsePolicyCUDA,
			template<typename, typename> typename Container,
			typename Alloc>
	void implicit_solvers::Implicit1DHeatEquationCUDA<T, BoundaryConditionType::Dirichlet, MemSpace, RealSparsePolicyCUDA, Container, Alloc>::
		discretizeInitialCondition(std::function<T(T)> const &init,
									Container<T, Alloc> &container) const {
		LSS_ASSERT(container.size() > 0, "The input container must be initialized.");
		for (std::size_t t = 0; t < container.size(); ++t) {
			container[t] = init(container[t]);
		}
	}


	template<typename T,
			MemorySpace MemSpace,
			template<MemorySpace, typename> typename RealSparsePolicyCUDA,
			template<typename, typename> typename Container,
			typename Alloc>
	void implicit_solvers::Implicit1DHeatEquationCUDA<T,BoundaryConditionType::Dirichlet,MemSpace,RealSparsePolicyCUDA,Container,Alloc>::
		solve(Container<T, Alloc> &solution, ImplicitPDESchemes scheme) {

		LSS_ASSERT(solution.size() > 0, "The input solution container must be initialized.");
		// get the correct scheme:
		auto schemeFun = ImplicitHeatEquationSchemes<T>::getSchemeCUDA(scheme);
		// get correct theta according to the scheme:
		T const theta = ImplicitHeatEquationSchemes<T>::getTheta(scheme);
		// get space step:
		T const h = spaceStep();
		// get time step:
		T const k = timeStep();
		// calculate lambda:
		T const lambda = (diffusivity_ *  k) / (h*h);
		// store size of matrix:
		std::size_t const m = spaceN_ - 1;
		// create container to carry mesh in space and then previous solution:
		Container<T, Alloc> prevSol(m, T{});
		// populate the container with mesh in space
		discretizeSpace(h, boundary_, prevSol);
		// use the mesh in space to get values of initial condition
		discretizeInitialCondition(init_, prevSol);
		// first create and populate the sparse matrix:
		FlatMatrix<T> fsm;
		fsm.setColumns(m); fsm.setRows(m);
		// populate the matrix:
		fsm.emplace_back(0, 0, (1.0 + 2.0*lambda*theta)); fsm.emplace_back(0, 1, -1.0*lambda*theta);
		for (std::size_t t = 1; t < m - 1; ++t) {
			fsm.emplace_back(t, t - 1, -1.0*lambda*theta);
			fsm.emplace_back(t, t, (1.0 + 2.0*lambda*theta));
			fsm.emplace_back(t, t + 1, -1.0*lambda*theta);
		}
		fsm.emplace_back(m - 1, m - 2, -1.0*lambda*theta); fsm.emplace_back(m - 1, m - 1, (1.0 + 2.0*lambda*theta));
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
			schemeFun(lambda, prevSol, rhs, boundary_);
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
	// ========================= Implicit1DHeatEquationCUDA (Robin BC) implementation ===============================
	// ==============================================================================================================

	template<typename T,
			MemorySpace MemSpace,
			template<MemorySpace, typename> typename RealSparsePolicyCUDA,
			template<typename, typename> typename Container,
			typename Alloc>
		void implicit_solvers::Implicit1DHeatEquationCUDA<T, BoundaryConditionType::Robin, MemSpace, RealSparsePolicyCUDA, Container, Alloc>::
		solve(Container<T, Alloc> &solution, ImplicitPDESchemes scheme) {

		LSS_ASSERT(solution.size() > 0, "The input solution container must be initialized.");

	}


	// ==============================================================================================================
	// ========================= Explicit1DHeatEquationCUDA (Dirichlet BC) implementation ===========================
	// ==============================================================================================================
	
	template<typename T,
			template<typename, typename> typename Container,
			typename Alloc>
	void explicit_solvers::Explicit1DHeatEquationCUDA<T, BoundaryConditionType::Dirichlet, Container, Alloc>::
		discretizeSpace(T const &step,
						std::pair<T, T> const &dirichletBC,
						Container<T, Alloc> & container)const {
		LSS_ASSERT(container.size() > 0, "The input container must be initialized.");
		container[0] = dirichletBC.first + step;
		for (std::size_t t = 1; t < container.size(); ++t) {
			container[t] = container[t - 1] + step;
		}
	}

	template<typename T,
			template<typename, typename> typename Container,
			typename Alloc>
	void explicit_solvers::Explicit1DHeatEquationCUDA<T, BoundaryConditionType::Dirichlet, Container, Alloc>::
		discretizeInitialCondition(std::function<T(T)> const &init,
			Container<T, Alloc> &container) const {
		LSS_ASSERT(container.size() > 0, "The input container must be initialized.");
		for (std::size_t t = 0; t < container.size(); ++t) {
			container[t] = init(container[t]);
		}
	}

	template<typename T,
			template<typename,typename> typename Container,
			typename Alloc>
	void explicit_solvers::Explicit1DHeatEquationCUDA<T,BoundaryConditionType::Dirichlet,Container,Alloc>::
		solve(Container<T, Alloc> &solution) {
		LSS_ASSERT(isStable() == true,
			"This discretization is not stable.");
		LSS_ASSERT(solution.size() > 0, 
			"The input solution container must be initialized.");
		// get space step:
		T const h = spaceStep();
		// get time step:
		T const k = timeStep();
		// calculate lambda:
		T const lambda = (diffusivity_ *  k) / (h*h);

	}


}















#endif ///_LSS_ONE_DIM_HEAT_EQUATION_SOLVERS_CUDA