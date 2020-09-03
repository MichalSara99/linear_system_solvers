#include<iostream>


#include"lss_sparse_solvers_cuda_t.h"
#include"lss_dense_solvers_cuda_t.h"
#include"lss_fdm_tridiagonal_solvers_t.h"

int main(int argc, char const* argv[]) {

    // ==================================================
    // ========= lss_sparse_solvers_tridiagonal_t.h =====
    // ==================================================

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

    // ==================================================

    // ===========================================
    // ========= lss_sparse_solvers_cuda_t.h =====
    // ===========================================

    // deviceSparseQRTest();
    // hostSparseQRTest();
    // testDirichletBCBVPOnHost();
    // testDirichletBCBVPOnDevice();
	// testRobinBCBVPOnHost();
	// testRobinBCBVPOnDevice();

    // ===========================================


    // ===========================================
    // ========= lss_dense_solvers_cuda_t.h =====
    // ===========================================

     // deviceDenseQRTest();
     // deviceDenseLUTest();


    // ===========================================

    std::cout << "\n\n";

    std::cin.get();
    std::cin.get();
    return 0;
}