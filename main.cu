#include <iostream>

#include "unit_tests/builders/lss_1d_general_svc_heat_equation_builder_t.hpp"
#include "unit_tests/builders/lss_1d_general_svc_wave_equation_builder_t.hpp"
#include "unit_tests/builders/lss_dirichlet_boundary_builder_t.hpp"
#include "unit_tests/builders/lss_heat_data_config_builder_t.hpp"
#include "unit_tests/builders/lss_heat_solver_config_builder_t.hpp"
#include "unit_tests/builders/lss_neumann_boundary_builder_t.hpp"
#include "unit_tests/builders/lss_pde_discretization_config_builder_t.hpp"
#include "unit_tests/builders/lss_robin_boundary_builder_t.hpp"
#include "unit_tests/builders/lss_wave_data_config_builder_t.hpp"
#include "unit_tests/builders/lss_wave_solver_config_builder_t.hpp"
#include "unit_tests/common/lss_print_t.hpp"
#include "unit_tests/containers/lss_container_2d_t.hpp"
#include "unit_tests/dense_solvers/lss_dense_solvers_cuda_t.hpp"
#include "unit_tests/ode_solvers/second_degree/lss_odes_2_degree_t.hpp"
#include "unit_tests/pde_solvers/one_dimensional/lss_advection_equation_t.hpp"
#include "unit_tests/pde_solvers/one_dimensional/lss_black_scholes_equation_t.hpp"
#include "unit_tests/pde_solvers/one_dimensional/lss_pure_heat_equation_t.hpp"
#include "unit_tests/pde_solvers/one_dimensional/lss_pure_wave_equation_t.hpp"
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
    // with source:
    // testImplPureHeatEquationSourceDirichletBCCUDASolverDeviceQR();
    // testImplPureHeatEquationSourceDirichletBCSORSolverDeviceEuler();
    // neumann bc:
    // testImplPureHeatEquationNeumannBCCUDASolverDeviceQR();
    // testImplPureHeatEquationNeumannBCThomasLUSolver();
    // testImplPureHeatEquationNeumannBCDoubleSweepSolver();

    // testImplPureHeatEquationDirichletBCCUDASolverDeviceQRStepping();

    // explicit:
    // testExplPureHeatEquationDirichletBCADE();
    // testExplPureHeatEquationNeumannBCEuler();
    // testExplPureHeatEquationDirichletBCDevice();

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
    // ================= ONE_DIM: lss_advection_equation_t.h ==============
    // ====================================================================

    // testImplAdvDiffEquationDirichletBCCUDASolverDeviceQR();
    // testImplAdvDiffEquationDirichletBCSORSolverDevice();
    // testImplAdvDiffEquationDirichletBCSORSolverHost();
    // testImplAdvDiffEquationDirichletBCCUDASolverHostQR();
    // testImplAdvDiffEquationDirichletBCDoubleSweepSolver();
    // testImplAdvDiffEquationDirichletBCThomasLUSolver();

    // ====================================================================

    // ====================================================================
    // ===================== ONE_DIM: lss_odes_2_degree_t.h ===============
    // ====================================================================

    // testImplSimpleODEDirichletBCCUDASolverDevice();
    // testImplSimpleODEDirichletNeumannBCCUDASolverDevice();
    // testImplSimpleODEDirichletRobinBCCUDASolverDevice();
    // testImplSimpleODENeumannRobinBCCUDASolverDevice();
    // testImplSimpleODE1NeumannRobinBCCUDASolverDevice();

    // ====================================================================

    // ====================================================================
    // ============================= lss_print_t.hpp ======================
    // ====================================================================
    // testImplSimpleODEThomesLUPrint();

    // testImplBlackScholesEquationDirichletBCThomasLUSolverPrint();
    // testImplBlackScholesEquationDirichletBCThomasLUSolverPrintSurf();
    // testImplPureHeatEquationDirichletBCCUDASolverDeviceQRPrintSurface();
    // testExplPureHeatEquationNeumannBCEulerPrintSurface();
    // testImplAdvDiffEquationDirichletBCThomasLUSolverPrintSurface();

    // testImplPureWaveEquationDirichletBCCUDASolverDeviceQRPrintSurf();
    // testImplWaveEquationDirichletBCSolverHostLUPrintSurf();
    // testImplWaveEquationDirichletBCSolverHostDoubleSweepPrintSurf();
    // testImplPureWaveEquationNeumannBCCUDASolverDeviceQRPrintSurf();

    // testExplPureWaveEquationDirichletBCCUDAHostSolverPrintSurf();

    // ====================================================================

    // ====================================================================
    // ============== ONE_DIM: lss_wave_heat_equation_t.h =================
    // ====================================================================

    // testImplPureWaveEquationDirichletBCCUDASolverDeviceQR();
    // testImplPureWaveEquationDirichletBCCUDASolverHostSOR();
    // testImplPureWaveEquationDirichletBCCUDASolverDeviceSOR();
    // testImplPureWaveEquationDirichletBCSolverDoubleSweep();
    // testImplPureWaveEquationDirichletBCSolverLU();
    // testImplWaveEquationDirichletBCSolverLU();
    // testImplDampedWaveEquationDirichletBCSolverDoubleSweep();

    // testImplPureWaveEquationNeumannBCCUDASolverDeviceQR();

    // explicit:
    // testExplPureWaveEquationDirichletBCCUDAHostSolver();
    // testExplPureWaveEquationDirichletBCCUDADeviceSolver();

    // ====================================================================

    // ====================================================================
    // ============= lss_wave_solver_config_builder_t.hpp =================
    // ====================================================================

    // test_wave_solver_config_implicit_builder();
    // test_wave_solver_config_explicit_builder();

    // ====================================================================

    // ====================================================================
    // ============= lss_heat_solver_config_builder_t.hpp =================
    // ====================================================================

    // test_heat_solver_config_implicit_builder();
    // test_heat_solver_config_explicit_builder();

    // ====================================================================

    // ====================================================================
    // ============= lss_pde_discretization_config_builder_t.hpp ==========
    // ====================================================================

    // test_pde_discretization_config_builder();

    // ====================================================================

    // ====================================================================
    // ================== lss_heat_data_config_builder_t.hpp ==============
    // ====================================================================

    // test_heat_data_config_builder();

    // ====================================================================

    // ====================================================================
    // ================== lss_wave_data_config_builder_t.hpp ==============
    // ====================================================================

    // test_wave_data_config_builder();

    // ====================================================================

    // ====================================================================
    // ================== lss_robin_boundary_builder_t.hpp ================
    // ====================================================================

    // test_robin_boundary_builder();

    // ====================================================================

    // ====================================================================
    // ================== lss_neumann_boundary_builder_t.hpp ==============
    // ====================================================================

    // test_neumann_boundary_builder();

    // ====================================================================

    // ====================================================================
    // ================ lss_dirichlet_boundary_builder_t.hpp ==============
    // ====================================================================

    // test_neumann_boundary_builder();

    // ====================================================================

    // ====================================================================
    // ========= lss_1d_general_svc_wave_equation_builder_t.hpp ===========
    // ====================================================================

    // test_pure_wave_equation_builder();

    // ====================================================================

    // ====================================================================
    // ========= lss_1d_general_svc_heat_equation_builder_t.hpp ===========
    // ====================================================================

    // test_pure_heat_equation_builder();

    // ====================================================================

    std::cout << "\n\n";

    std::cin.get();
    std::cin.get();
    return 0;
}
