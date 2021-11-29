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
#include "unit_tests/builders/lss_2d_general_svc_heston_equation_builder_t.hpp"
#include "unit_tests/builders/lss_general_2nd_ode_equation_builder_t.hpp"
#include "unit_tests/common/lss_print_t.hpp"
#include "unit_tests/containers/lss_container_2d_t.hpp"
#include "unit_tests/dense_solvers/lss_dense_solvers_cuda_t.hpp"
#include "unit_tests/ode_solvers/second_degree/lss_odes_2_degree_t.hpp"
#include "unit_tests/pde_solvers/one_dimensional/lss_advection_equation_t.hpp"
#include "unit_tests/pde_solvers/one_dimensional/lss_black_scholes_equation_t.hpp"
#include "unit_tests/pde_solvers/one_dimensional/lss_pure_heat_equation_t.hpp"
#include "unit_tests/pde_solvers/one_dimensional/lss_pure_wave_equation_t.hpp"
#include "unit_tests/pde_solvers/two_dimensional/lss_heston_equation_t.hpp"
#include "unit_tests/pde_solvers/two_dimensional/lss_sabr_equation_t.hpp"
#include "unit_tests/sparse_solvers/lss_core_cuda_solver_t.hpp"
#include "unit_tests/sparse_solvers/lss_core_sor_solver_cuda_t.hpp"
#include "unit_tests/sparse_solvers/lss_core_sor_solver_t.hpp"
#include "unit_tests/sparse_solvers/lss_cuda_solver_t.hpp"
#include "unit_tests/sparse_solvers/lss_double_sweep_solver_t.hpp"
#include "unit_tests/sparse_solvers/lss_sor_solver_cuda_t.hpp"
#include "unit_tests/sparse_solvers/lss_sor_solver_t.hpp"
#include "unit_tests/sparse_solvers/lss_thomas_lu_solver_t.hpp"
#include "unit_tests/sparse_solvers/lss_karawia_solver_t.hpp"

int main(int argc, char const *argv[])
{
    // ====================================================================
    // ====================== lss_container_2d_t.hpp ======================
    // ====================================================================
    // tested:

    // testContainer2d();                                                           // -- tested
    // testCopyContainer2d();                                                       // -- tested
    // testNotSymetricalContainer2d();                                              // -- tested

    // ====================================================================

    // ====================================================================
    // ================== lss_sparse_solvers_tridiagonal_t.hpp ============
    // ====================================================================
    // tested:

    // testDoubleSweepDirichletBC();                                                // -- tested
    // testKarawiaDirichletBC();                                                    // -- tested
    // testDoubleSweepRobinBC();                                                    // -- tested
    // testDoubleSweepMixBC();                                                      // -- tested
    // testThomasLUDirichletBC();                                                   // -- tested
    // testThomasLURobinBC();                                                       // -- tested
    // testThomasLUMixBC();                                                         // -- tested

    // ====================================================================

    // ====================================================================
    // ===================== lss_core_cuda_solver_t.hpp ===================
    // ====================================================================
    // tested:

    // deviceSparseQRTest();                                                        // -- tested
    // hostSparseQRTest();                                                          // -- tested
    // testDirichletBCBVPOnHost();                                                  // -- tested
    // testDirichletBCBVPOnDevice();                                                // -- tested
    // testRobinBCBVPOnHost();                                                      // -- tested
    // testRobinBCBVPOnDevice();                                                    // -- tested

    // ====================================================================

    // ====================================================================
    // ======================== lss_cuda_solver_t.hpp =====================
    // ====================================================================
    // tested:

    // testCUDADirichletBC();                                                       // -- tested
    // testCUDARobinBC();                                                           // -- tested                            // -- tested
    // testCUDAMixBC();                                                             // -- tested

    // ====================================================================

    // ====================================================================
    // ====================== lss_core_sor_solver_t.hpp ===================
    // ====================================================================
    // tested:

    // testSOR();                                                                   // -- tested
    // testBVPDirichletBCSOR();                                                     // -- tested
    // testBVPRobinBCSOR();                                                         // -- tested

    // ====================================================================

    // ====================================================================
    // ================= lss_core_sor_solver_cuda_t.hpp ===================
    // ====================================================================
    // tested:

    // testSORCUDA();                                                               // -- tested
    // testBVPDirichletBCSORCUDA();                                                 // -- tested

    // ====================================================================

    // ====================================================================
    // ======================= lss_sor_solver_t.hpp =======================
    // ====================================================================
    // tested:

    // testSORDirichletBC();                                                        // -- tested
    // testSORRobinBC();                                                            // -- tested
    // testSORMixBC();                                                              // -- tested

    // ====================================================================

    // ====================================================================
    // ==================== lss_sor_solver_cuda_t.hpp =====================
    // ====================================================================
    // tested:

    // testSORCUDADirichletBC();                                                    // -- tested
    // testSORRobinBC();                                                            // -- tested
    // testSORMixBC();                                                              // -- tested

    // ====================================================================

    // ====================================================================
    // ==================== lss_dense_solvers_cuda_t.hpp ==================
    // ====================================================================
    // tested:

    // deviceDenseQRTest();                                                         // -- tested
    // deviceDenseLUTest();                                                         // -- tested

    // ====================================================================

    // ====================================================================
    // ============== ONE_DIM: lss_pure_heat_equation_t.h =================
    // ====================================================================

    // testImplPureHeatEquationDirichletBCCUDASolverDeviceQR();                     // -- tested
    // testImplPureHeatEquationDirichletBCCUDASolverHostQR();                       // -- tested
    // testImplPureHeatEquationDirichletBCSORSolverDevice();                        // -- tested
    // testImplPureHeatEquationDirichletBCSORSolverHost();                          // -- tested
    // testImplPureHeatEquationDirichletBCDoubleSweepSolver();                      // -- tested
    // testImplPureHeatEquationDirichletBCThomasLUSolver();                         // -- tested
    // with source:
    // testImplPureHeatEquationSourceDirichletBCCUDASolverDeviceQR();               // -- tested
    // testImplPureHeatEquationSourceDirichletBCSORSolverDeviceEuler();             // -- tested
    // neumann bc:
    //testImplPureHeatEquationNeumannBCCUDASolverDeviceQR();                        // -- tested
    // testImplPureHeatEquationNeumannBCThomasLUSolver();                           // -- tested
    // testImplPureHeatEquationNeumannBCDoubleSweepSolver();                        // -- tested

    // testImplPureHeatEquationDirichletBCCUDASolverDeviceQRStepping();             // -- tested 

    // explicit:
    // testExplPureHeatEquationDirichletBCADE();                                    // -- tested
    // testExplPureHeatEquationNeumannBCEuler();                                    // -- tested
    // testExplPureHeatEquationDirichletBCDevice();                                 // -- tested

    // ====================================================================

    // ====================================================================
    // ============== ONE_DIM: lss_black_scholes_equation_t.h =============
    // ====================================================================

    // testImplBlackScholesEquationDirichletBCCUDASolverDeviceQR();                 // -- tested
    // testImplBlackScholesEquationDirichletBCSORSolverDevice();                    // -- tested
    // testImplBlackScholesEquationDirichletBCSORSolverHost();                      // -- tested
    // testImplBlackScholesEquationDirichletBCDoubleSweepSolver();                  // -- tested
    // testImplBlackScholesEquationDirichletBCThomasLUSolver();                     // -- tested
    // testImplFwdBlackScholesEquationDirichletBCCUDASolverDeviceQR();              // -- tested

    // testImplBlackScholesEquationDirichletBCThomasLUSolverStepping();             // -- tested

    // explicit:
    // testExplBlackScholesEquationDirichletBCADE();                                // -- tested

    // ====================================================================

    // ====================================================================
    // ================= ONE_DIM: lss_advection_equation_t.h ==============
    // ====================================================================

    // testImplAdvDiffEquationDirichletBCCUDASolverDeviceQR();                      // -- tested
    // testImplAdvDiffEquationDirichletBCSORSolverDevice();                         // -- tested
    // testImplAdvDiffEquationDirichletBCSORSolverHost();                           // -- tested
    // testImplAdvDiffEquationDirichletBCCUDASolverHostQR();                        // -- tested
    // testImplAdvDiffEquationDirichletBCDoubleSweepSolver();                       // -- tested
    // testImplAdvDiffEquationDirichletBCThomasLUSolver();                          // -- tested

    // ====================================================================

    // ====================================================================
    // ===================== ONE_DIM: lss_odes_2_degree_t.h ===============
    // ====================================================================

    // testImplSimpleODEDirichletBCCUDASolverDevice();                              // -- tested
    // testImplSimpleODEDirichletNeumannBCCUDASolverDevice();                       // -- tested
    // testImplSimpleODEDirichletRobinBCCUDASolverDevice();                         // -- tested
    // testImplSimpleODENeumannRobinBCCUDASolverDevice();                           // -- tested
    // testImplSimpleODE1NeumannRobinBCCUDASolverDevice();                          // -- tested

    // ====================================================================

    // ====================================================================
    // ============================= lss_print_t.hpp ======================
    // ====================================================================
    // testImplSimpleODEThomesLUPrint();                                            // -- tested

    // testImplBlackScholesEquationDirichletBCThomasLUSolverPrint();                // -- tested
    // testImplBlackScholesEquationDirichletBCThomasLUSolverPrintSurf();            // -- tested
    // testImplPureHeatEquationDirichletBCCUDASolverDeviceQRPrintSurface();         // -- tested
    // testExplPureHeatEquationNeumannBCEulerPrintSurface();                        // -- tested
    // testImplAdvDiffEquationDirichletBCThomasLUSolverPrintSurface();              // -- tested

    // testImplPureWaveEquationDirichletBCCUDASolverDeviceQRPrintSurf();            // -- tested
    // testImplWaveEquationDirichletBCSolverHostLUPrintSurf();                      // -- tested
    // testImplWaveEquationDirichletBCSolverHostDoubleSweepPrintSurf();             // -- tested
    // testImplPureWaveEquationNeumannBCCUDASolverDeviceQRPrintSurf();              // -- tested
    // testExplPureWaveEquationDirichletBCCUDAHostSolverPrintSurf();                // -- tested

    // testImplHestonEquationCUDAQRSolverCrankNicolsonPrint();                      // -- tested
    // testImplHestonEquationThomasLUSolverCrankNicolsonPrint();                    // -- tested
    // testImplSABREquationDoubleSweepSolverCrankNicolsonPrint();                   // -- tested
    // testImplHestonEquationThomasLUSolverDouglasRachfordCrankNicolsonPrint();     // -- tested
    // testImplHestonEquationThomasLUSolverCraigSneydCrankNicolsonPrint();          // -- tested
    // testImplHestonEquationThomasLUSolverModCraigSneydCrankNicolsonPrint();       // -- tested
    // testImplHestonEquationThomasLUSolverHundsdorferVerwerCrankNicolsonPrint();   // -- tested

    // ====================================================================

    // ====================================================================
    // ============== ONE_DIM: lss_wave_heat_equation_t.h =================
    // ====================================================================

    // testImplPureWaveEquationDirichletBCCUDASolverDeviceQR();                     // -- tested
    // testImplPureWaveEquationDirichletBCCUDASolverHostSOR();                      // -- tested
    // testImplPureWaveEquationDirichletBCCUDASolverDeviceSOR();                    // -- tested
    // testImplPureWaveEquationDirichletBCSolverDoubleSweep();                      // -- tested
    // testImplPureWaveEquationDirichletBCSolverLU();                               // -- tested
    // testImplWaveEquationDirichletBCSolverLU();                                   // -- tested
    // testImplDampedWaveEquationDirichletBCSolverDoubleSweep();                    // -- tested

    // testImplPureWaveEquationNeumannBCCUDASolverDeviceQR();                       // -- tested

    // explicit:
    // testExplPureWaveEquationDirichletBCCUDAHostSolver();                         // -- tested
    // testExplPureWaveEquationDirichletBCCUDADeviceSolver();                       // -- tested

    // ====================================================================

    // ====================================================================
    // =================== TWO_DIM: lss_heston_equation_t.hpp =============
    // ====================================================================

    // testImplHestonEquationCUDAQRSolver();                                        // -- tested
    // testImplHestonEquationThomasLUSolver();                                      // -- tested

    // ====================================================================

    // ====================================================================
    // =================== TWO_DIM: lss_sabr_equation_t.hpp ===============
    // ====================================================================

    // testImplSABREquationDoubleSweepSolver();                                     // -- tested

    // ====================================================================

    // ====================================================================
    // ============= lss_wave_solver_config_builder_t.hpp =================
    // ====================================================================

    // test_wave_solver_config_implicit_builder();                                  // -- tested
    // test_wave_solver_config_explicit_builder();                                  // -- tested

    // ====================================================================

    // ====================================================================
    // ============= lss_heat_solver_config_builder_t.hpp =================
    // ====================================================================

    // test_heat_solver_config_implicit_builder();                                  // -- tested
    // test_heat_solver_config_explicit_builder();                                  // -- tested

    // ====================================================================

    // ====================================================================
    // ============= lss_pde_discretization_config_builder_t.hpp ==========
    // ====================================================================

    // test_pde_discretization_config_builder();                                    // -- tested

    // ====================================================================

    // ====================================================================
    // ================== lss_heat_data_config_builder_t.hpp ==============
    // ====================================================================

    // test_heat_data_config_builder();                                             // -- tested

    // ====================================================================

    // ====================================================================
    // ================== lss_wave_data_config_builder_t.hpp ==============
    // ====================================================================

    // test_wave_data_config_builder();                                             // -- tested

    // ====================================================================

    // ====================================================================
    // ================== lss_robin_boundary_builder_t.hpp ================
    // ====================================================================

    // test_robin_boundary_builder();                                               // -- tested

    // ====================================================================

    // ====================================================================
    // ================== lss_neumann_boundary_builder_t.hpp ==============
    // ====================================================================

    // test_neumann_boundary_builder();                                             // -- tested

    // ====================================================================

    // ====================================================================
    // ================ lss_dirichlet_boundary_builder_t.hpp ==============
    // ====================================================================

    // test_neumann_boundary_builder();                                             // -- tested

    // ====================================================================

    // ====================================================================
    // ========= lss_1d_general_svc_wave_equation_builder_t.hpp ===========
    // ====================================================================

    // test_pure_wave_equation_builder();                                           // -- tested
    // test_expl_pure_wave_equation_builder();                                      // -- tested

    // ====================================================================

    // ====================================================================
    // ========= lss_1d_general_svc_heat_equation_builder_t.hpp ===========
    // ====================================================================

    // test_pure_heat_equation_builder();                                           // -- tested

    // ====================================================================

    // ====================================================================
    // ======= lss_2d_general_svc_heston_equation_builder_t.hpp ===========
    // ====================================================================

    // test_heston_equation_builder();                                              // -- tested

    // ====================================================================

    // ====================================================================
    // ========== lss_general_2nd_ode_equation_builder_t.hpp ==============
    // ====================================================================

    // test_general_2nd_ode_equation_builder();                                     // -- tested

    // ====================================================================

    std::cout << "\n\n";

    std::cin.get();
    std::cin.get();
    return 0;
}
