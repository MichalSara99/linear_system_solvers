#include<iostream>


#include"lss_sparse_solvers_cuda_t.h"
#include"lss_dense_solvers_cuda_t.h"
#include"lss_fdm_tridiagonal_solvers_t.h"
#include"lss_one_dim_heat_equation_solvers_t.h"
#include"lss_one_dim_heat_equation_solvers_cuda_t.h"
#include"lss_one_dim_advection_diffusion_equation_solvers_t.h"

int main(int argc, char const* argv[]) {

	// =====================================================
    // ========= lss_sparse_solvers_tridiagonal_t.h ========
	// =====================================================

     // testDoubleSweepDirichletBC(); 
     // testThomasLUSolverDirichletBC();
	 // testDoubleSweepDirichletBC1();
	 // testThomasLUSolverDirichletBC1();

	// testDoubleSweepDirichletBC();
	// testThomasLUSolverDirichletBC();
	// testDoubleSweepDirichletBC1();
	// testThomasLUSolverDirichletBC1();
	// testDoubleSweepRobinBC();
	// testThomasLUSolverRobinBC();

	// =====================================================

	// =====================================================
    // ============= lss_sparse_solvers_cuda_t.h ===========
	// =====================================================

    // deviceSparseQRTest();
    // hostSparseQRTest();
    // testDirichletBCBVPOnHost();
	
    // testDirichletBCBVPOnDevice();
	// testRobinBCBVPOnHost();
	// testRobinBCBVPOnDevice();

	// =====================================================


	// =====================================================
    // ============== lss_dense_solvers_cuda_t.h ===========
	// =====================================================

     // deviceDenseQRTest();
     // deviceDenseLUTest();


	 // =====================================================

	// =====================================================
	// ========= lss_one_dim_heat_equation_solvers_t.h =====
	// =====================================================

	// testImplHeatEquationDirichletBCDoubleSweep();
	// testImplHeatEquationRobinBCDoubleSweep();
	// testImplHeatEquationDirichletBCThomasLU();
	// testImplHeatEquationRobinBCThomasLU();
	// testExplHeatEquationDirichletBC();
	// testImplNonHomHeatEquationDirichletBCDoubleSweep();
	// testImplNonHomHeatEquationDirichletBCThomasLU();
	// testExplNonHomHeatEquationDirichletBC();
	// testExplHomHeatEquationRobinBC();

	// =====================================================

	// =====================================================
	// ==== lss_one_dim_heat_equation_solvers_cuda_t.h =====
	// =====================================================

	// testImplHeatEquationDirichletBCDevice();
	// testImplHeatEquationDirichletBCHost();
	// testImplHeatEquationRobinBCDevice();
	// testImplNonHomHeatEquationDirichletBCDevice();
	// testImplNonHomHeatEquationDirichletBCHost();
	// testExplHeatEquationDirichletBCDevice();
	// testExplNonHomHeatEquationDirichletBCDevice();
	// testExplHeatEquationRobinBCDevice();


	// =====================================================

	// ==========================================================
	// == lss_one_dim_advection_diffusion_equation_solvers_t.h ==
	// ==========================================================
	
	// testImplAdvectionDiffEquationDirichletBCDoubleSweep();
	testExplAdvectionDiffEquationDirichletBC();

	// ==========================================================



    std::cout << "\n\n";

    std::cin.get();
    std::cin.get();
    return 0;
}