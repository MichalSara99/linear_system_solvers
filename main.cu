#include <iostream>

#include "unit_tests/lss_dense_solvers_cuda_t.h"
#include "unit_tests/lss_fdm_tridiagonal_solvers_t.h"
#include "unit_tests/lss_one_dim_advection_diffusion_equation_cuda_t.h"
#include "unit_tests/lss_one_dim_advection_diffusion_equation_t.h"
#include "unit_tests/lss_one_dim_pure_heat_equation_cuda_t.h"
#include "unit_tests/lss_one_dim_pure_heat_equation_t.h"
#include "unit_tests/lss_one_dim_space_variable_advection_diffusion_equation_cuda_t.h"
#include "unit_tests/lss_one_dim_space_variable_advection_diffusion_equation_t.h"
#include "unit_tests/lss_one_dim_space_variable_pure_heat_equation_cuda_t.h"
#include "unit_tests/lss_one_dim_space_variable_pure_heat_equation_t.h"
#include "unit_tests/lss_sparse_solvers_cuda_t.h"

int main(int argc, char const *argv[]) {
  // ====================================================================
  // ================== lss_sparse_solvers_tridiagonal_t.h ==============
  // ====================================================================

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

  // ====================================================================

  // ====================================================================
  // =================== lss_sparse_solvers_cuda_t.h ====================
  // ====================================================================

  // deviceSparseQRTest();
  // hostSparseQRTest();
  // testDirichletBCBVPOnHost();

  // testDirichletBCBVPOnDevice();
  // testRobinBCBVPOnHost();
  // testRobinBCBVPOnDevice();

  // ====================================================================

  // ====================================================================
  // ==================== lss_dense_solvers_cuda_t.h ====================
  // ====================================================================

  // deviceDenseQRTest();
  // deviceDenseLUTest();

  // ====================================================================

  // ====================================================================
  // ================ lss_one_dim_pure_heat_equation_t.h ================
  // ====================================================================

  // testImplPureHeatEquationDirichletBCDoubleSweep();
  // testImplPureHeatEquationDirichletBCThomasLU();
  // testImplPureHeatEquationRobinBCDoubleSweep();
  // testImplPureHeatEquationRobinBCThomasLU();
  // testImplPureHeatEquationSourceDirichletBCDoubleSweep();
  // testImplPureHeatEquationSourceDirichletBCThomasLU();
  // testImplPureHeatEquationSourceRobinBCDoubleSweep();
  // testImplPureHeatEquationSourceRobinBCThomasLU();
  // testImplNonHomPureHeatEquationDirichletBCDoubleSweep();
  // testImplNonHomPureHeatEquationDirichletBCThomasLU();
  // testExplPureHeatEquationDirichletBC();
  // testExplPureHeatEquationSourceDirichletBC();
  // testExplNonHomPureHeatEquationDirichletBC();
  // testExplHomPureHeatEquationRobinBC();
  // testExplHomPureHeatEquationSourceRobinBC();

  // ====================================================================

  // ====================================================================
  // ============ lss_one_dim_pure_heat_equation_cuda_t.h ===============
  // ====================================================================

  // testImplPureHeatEquationDirichletBCDeviceCUDA();
  // testImplPureHeatEquationDirichletBCHostCUDA();
  // testImplPureHeatEquationRobinBCDeviceCUDA();
  // testImplPureHeatEquationSourceDirichletBCCUDA();
  // testImplPureHeatEquationSourceRobinBCCUDA();
  // testImplNonHomPureHeatEquationDirichletBCDeviceCUDA();
  // testImplNonHomPureHeatEquationDirichletBCHostCUDA();
  // testExplPureHeatEquationDirichletBCDeviceCUDA();
  // testExplNonHomPureHeatEquationDirichletBCDeviceCUDA();
  // testExplPureHeatEquationSourceDirichletBCEulerCUDA();
  // testExplPureHeatEquationRobinBCDeviceCUDA();
  // testExplHomPureHeatEquationSourceRobinBCCUDA();

  // ====================================================================

  // ====================================================================
  // ========== lss_one_dim_advection_diffusion_equation_t.h ============
  // ====================================================================

  // testImplAdvDiffEquationDirichletBCDoubleSweep();
  // testImplAdvDiffEquationSourceDirichletBCDoubleSweep();
  // testImplAdvDiffEquationSourceDirichletBCThomasLU();
  // testImplAdvDiffEquationRobinBCDoubleSweep();
  // testImplAdvDiffEquationRobinBCThomasLU();
  // testExplAdvDiffEquationDirichletBC();
  // testExplAdvDiffEquationSourceDirichletBC();
  // testExplAdvDiffEquationRobinBC();

  // ====================================================================

  // ====================================================================
  // ====== lss_one_dim_advection_diffusion_equation_cuda_t.h ===========
  // ====================================================================

  // testImplAdvDiffEquationDirichletBCDeviceCUDA();
  // testImplAdvDiffEquationRobinBCDeviceCUDA();
  // testExplAdvDiffEquationDirichletBCCUDA();
  // testExplAdvDiffEquationRobinBCCUDA();

  // ====================================================================

  // ====================================================================
  // ====== lss_one_dim_space_variable_pure_heat_equation_t.h ===========
  // ====================================================================

  // testImplSpaceVarPureHeatEquationDirichletBCDoubleSweep();
  // testImplSpaceVarPureHeatEquationDirichletBCThomasLU();
  // testImplSpaceVarPureHeatEquationRobinBCDoubleSweep();
  // testImplSpaceVarPureHeatEquationRobinBCThomasLU();
  // testImplSpaceVarPureHeatEquationSourceDirichletBCDoubleSweep();
  // testImplSpaceVarPureHeatEquationSourceDirichletBCThomasLU();
  // testImplSpaceVarPureHeatEquationSourceRobinBCDoubleSweep();
  // testImplSpaceVarPureHeatEquationSourceRobinBCThomasLU();
  // testImplSpaceVarNonHomPureHeatEquationDirichletBCDoubleSweep();
  // testImplSpaceVarNonHomPureHeatEquationDirichletBCThomasLU();
  // testExplSpaceVarPureHeatEquationDirichletBC();
  // testExplSpaceVarPureHeatEquationSourceDirichletBC();
  // testExplSpaceVarNonHomPureHeatEquationDirichletBC();
  // testExplSpaceVarHomPureHeatEquationRobinBC();
  // testExplSpaceVarHomPureHeatEquationSourceRobinBC();

  // testImplSpaceVarHeatEquationDirichletBCDoubleSweep();
  // testImplSpaceVarHeatEquationDirichletBCThomasLU();
  // testExplSpaceVarHeatEquationDirichletBCADE();

  // ====================================================================

  // ====================================================================
  // === lss_one_dim_space_variable_advection_diffusion_equation_t.h ====
  // ====================================================================

  // testImplSpaceVarAdvDiffEquationDirichletBCDoubleSweep();
  // testImplSpaceVarAdvDiffEquationSourceDirichletBCDoubleSweep();
  // testImplSpaceVarAdvDiffEquationSourceDirichletBCThomasLU();
  // testImplSpaceVarAdvDiffEquationRobinBCDoubleSweep();
  // testImplSpaceVarAdvDiffEquationRobinBCThomasLU();
  // testExplSpaceVarAdvDiffEquationDirichletBC();
  // testExplSpaceVarAdvDiffEquationSourceDirichletBC();
  // testExplSpaceVarAdvDiffEquationRobinBC();

  // ====================================================================

  // ====================================================================
  // ==== lss_one_dim_space_variable_pure_heat_equation_cuda_t.h ========
  // ====================================================================

  // testImplSpaceVarPureHeatEquationDirichletBCDeviceCUDA();
  // testImplSpaceVarPureHeatEquationDirichletBCHostCUDA();
  // testImplSpaceVarPureHeatEquationRobinBCDeviceCUDA();
  // testImplSpaceVarPureHeatEquationSourceDirichletBCCUDA();
  // testImplSpaceVarPureHeatEquationSourceRobinBCCUDA();
  // testImplSpaceVarNonHomPureHeatEquationDirichletBCDeviceCUDA();
  // testImplSpaceVarNonHomPureHeatEquationDirichletBCHostCUDA();
  // testExplSpaceVarPureHeatEquationDirichletBCDeviceCUDA();
  // testExplSpaceVarNonHomPureHeatEquationDirichletBCDeviceCUDA();
  // testExplSpaceVarPureHeatEquationSourceDirichletBCEulerCUDA();
  // testExplSpaceVarPureHeatEquationRobinBCDeviceCUDA();
  // testExplSpaceVarHomPureHeatEquationSourceRobinBCCUDA();

  // testImplSpaceVarHeatEquationDirichletBCHostCUDA();
  // testExplSpaceVarHeatEquationDirichletBCHostCUDA();

  // ====================================================================

  // ====================================================================
  // = lss_one_dim_space_variable_advection_diffusion_equation_cuda_t.h =
  // ====================================================================

  // testImplSpaceVarAdvDiffEquationDirichletBCDeviceCUDA();
  // testImplSpaceVarAdvDiffEquationRobinBCDeviceCUDA();
  // testExplSpaceVarAdvDiffEquationDirichletBCCUDA();
  // testExplSpaceVarAdvDiffEquationRobinBCCUDA();

  // ====================================================================

  std::cout << "\n\n";

  std::cin.get();
  std::cin.get();
  return 0;
}
