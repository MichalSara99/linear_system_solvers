#include<iostream>


#include"lss_sparse_solvers_cuda_t.h"
#include"lss_dense_solvers_cuda_t.h"
#include"lss_sparse_solvers_tridiagonal_t.h"

int main(int argc, char const* argv[]) {

    // ==================================================
    // ========= lss_sparse_solvers_tridiagonal_t.h =====
    // ==================================================

    // testDoubleSweep();

    // ==================================================

    // ===========================================
    // ========= lss_sparse_solvers_cuda_t.h =====
    // ===========================================

    // deviceSparseQRTest();
    // hostSparseQRTest();
    // testBVPOnHost();
    // testBVPOnDevice();
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