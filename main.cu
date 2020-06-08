#include<iostream>


#include"lss_sparse_solvers_cuda_t.h"
#include"lss_dense_solvers_cuda_t.h"

int main(int argc, char const* argv[]) {


    // ===========================================
    // ========= lss_sparse_solvers_cuda_t.h =====
    // ===========================================

    // deviceSparseQRtest();


    // ===========================================


    // ===========================================
    // ========= lss_dense_solvers_cuda_t.h =====
    // ===========================================

    // deviceDenseQRTest();
    deviceDenseLUTest();


    // ===========================================

    std::cout << "\n\n";

    std::cin.get();
    std::cin.get();
    return 0;
}