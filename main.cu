#include <iostream>

#include "unit_tests/common/lss_print_t.hpp"
#include "unit_tests/containers/lss_container_2d_t.hpp"
#include "unit_tests/dense_solvers/lss_dense_solvers_cuda_t.hpp"
#include "unit_tests/pde_solvers/one_dimensional/lss_black_scholes_equation_t.hpp"
#include "unit_tests/pde_solvers/one_dimensional/lss_pure_heat_equation_t.hpp"
#include "unit_tests/sparse_solvers/lss_core_cuda_solver_t.hpp"
#include "unit_tests/sparse_solvers/lss_core_sor_solver_cuda_t.hpp"
#include "unit_tests/sparse_solvers/lss_core_sor_solver_t.hpp"
#include "unit_tests/sparse_solvers/lss_cuda_solver_t.hpp"
#include "unit_tests/sparse_solvers/lss_double_sweep_solver_t.hpp"
#include "unit_tests/sparse_solvers/lss_sor_solver_cuda_t.hpp"
#include "unit_tests/sparse_solvers/lss_sor_solver_t.hpp"
#include "unit_tests/sparse_solvers/lss_thomas_lu_solver_t.hpp"

int main(int argc, char const *argv[])
{
    // ====================================================================
    // ====================== lss_container_2d_t.hpp ======================
    // ====================================================================
    // tested:

    // testContainer2d();
    // testCopyContainer2d();
    // testPartialCopyRowContainer2d();

    // ====================================================================

    // ====================================================================
    // ================== lss_sparse_solvers_tridiagonal_t.hpp ============
    // ====================================================================
    // tested:

    // testDoubleSweepDirichletBC();
    // testDoubleSweepRobinBC();
    // testDoubleSweepDirichletNeumannBC();
    // testDoubleSweepNeumannDirichletBC();
    // testDoubleSweepNeumannRobinBC();
    // testDoubleSweepMixBC();
    // testThomasLUDirichletBC();
    // testThomasLURobinBC();
    // testThomasLUNeumannDirichletBC();
    // testThomasLUDirichletNeumannBC();
    // testThomasLUNeumannRobinBC();
    // testThomasLUMixBC();

    // ====================================================================

    // ====================================================================
    // ===================== lss_core_cuda_solver_t.hpp ===================
    // ====================================================================
    // tested:

    // deviceSparseQRTest();
    // hostSparseQRTest();

    // testDirichletBCBVPOnHost();
    // testDirichletBCBVPOnDevice();
    // testRobinBCBVPOnHost();
    // testRobinBCBVPOnDevice();

    // ====================================================================

    // ====================================================================
    // ======================== lss_cuda_solver_t.hpp =====================
    // ====================================================================
    // tested:

    // testCUDADirichletBC();
    // testCUDARobinBC();
    // testCUDADirichletNeumannBC();
    // testCUDANeumannDirichletBC();
    // testCUDANeumannRobinBC();
    // testCUDAMixBC();

    // ====================================================================

    // ====================================================================
    // ====================== lss_core_sor_solver_t.hpp ===================
    // ====================================================================
    // tested:

    // testSOR();
    // testBVPDirichletBCSOR();
    // testBVPRobinBCSOR();

    // ====================================================================

    // ====================================================================
    // ================= lss_core_sor_solver_cuda_t.hpp ===================
    // ====================================================================
    // tested:

    // testSORCUDA();
    // testBVPDirichletBCSORCUDA();

    // ====================================================================

    // ====================================================================
    // ======================= lss_sor_solver_t.hpp =======================
    // ====================================================================
    // tested:

    // testSORDirichletBC();
    // testSORRobinBC();
    // testSORDirichletNeumannBC();
    // testSORNeumannDirichletBC();
    // testSORNeumannRobinBC();
    // testSORMixBC();

    // ====================================================================

    // ====================================================================
    // ==================== lss_sor_solver_cuda_t.hpp =====================
    // ====================================================================
    // tested:

    // testSORCUDADirichletBC();
    // testSORRobinBC();
    // testSORCUDADirichletNeumannBC();
    // testSORCUDANeumannDirichletBC();
    // testSORCUDANeumannRobinBC();
    // testSORMixBC();

    // ====================================================================

    // ====================================================================
    // ==================== lss_dense_solvers_cuda_t.hpp ==================
    // ====================================================================
    // tested:

    // deviceDenseQRTest();
    // deviceDenseLUTest();

    // ====================================================================

    // ====================================================================
    // ============== ONE_DIM: lss_pure_heat_equation_t.h =================
    // ====================================================================

    // testImplPureHeatEquationDirichletBCCUDASolverDeviceQR();
    // testImplPureHeatEquationDirichletBCCUDASolverHostQR();
    // testImplPureHeatEquationDirichletBCSORSolverDevice();
    // testImplPureHeatEquationDirichletBCSORSolverHost();
    // testImplPureHeatEquationDirichletBCDoubleSweepSolver();
    // testImplPureHeatEquationDirichletBCThomasLUSolver();
    // neumann bc:
    // testImplPureHeatEquationNeumannBCCUDASolverDeviceQR();
    // testImplPureHeatEquationNeumannBCThomasLUSolver();
    // testImplPureHeatEquationNeumannBCDoubleSweepSolver();

    // testImplPureHeatEquationDirichletBCCUDASolverDeviceQRStepping();

    // explicit:
    // testExplPureHeatEquationDirichletBCADE();

    // ====================================================================

    // ====================================================================
    // ============== ONE_DIM: lss_black_scholes_equation_t.h =============
    // ====================================================================

    // testImplBlackScholesEquationDirichletBCCUDASolverDeviceQR();
    // testImplBlackScholesEquationDirichletBCSORSolverDevice();
    // testImplBlackScholesEquationDirichletBCSORSolverHost();
    // testImplBlackScholesEquationDirichletBCDoubleSweepSolver();
    // testImplBlackScholesEquationDirichletBCThomasLUSolver();

    // testImplFwdBlackScholesEquationDirichletBCCUDASolverDeviceQR();

    // testImplBlackScholesEquationDirichletBCThomasLUSolverStepping();

    // explicit:
    // testExplBlackScholesEquationDirichletBCADE();

    // ====================================================================

    // ====================================================================
    // ============================= lss_print_t.hpp ======================
    // ====================================================================

    // testImplBlackScholesEquationDirichletBCThomasLUSolverPrint();
    // testImplBlackScholesEquationDirichletBCThomasLUSolverPrintSurf();
    // testImplPureHeatEquationDirichletBCCUDASolverDeviceQRPrintSurface();

    // ====================================================================

    std::cout << "\n\n";

    std::cin.get();
    std::cin.get();
    return 0;
}
