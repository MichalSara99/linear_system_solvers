#include <iostream>

//#include "unit_tests/lss_dense_solvers_cuda_t.h"
//#include "unit_tests/lss_fdm_tridiagonal_solvers_t.h"
//#include "unit_tests/lss_one_dim_advection_diffusion_equation_cuda_t.h"
//#include "unit_tests/lss_one_dim_advection_diffusion_equation_t.h"
//#include "unit_tests/lss_one_dim_black_scholes_equation_cuda_t.h"
//#include "unit_tests/lss_one_dim_black_scholes_equation_t.h"
//#include "unit_tests/lss_one_dim_pure_heat_equation_cuda_t.h"
//#include "unit_tests/lss_one_dim_pure_heat_equation_t.h"
//#include "unit_tests/lss_one_dim_space_variable_advection_diffusion_equation_cuda_t.h"
//#include "unit_tests/lss_one_dim_space_variable_advection_diffusion_equation_t.h"
//#include "unit_tests/lss_one_dim_space_variable_pure_heat_equation_cuda_t.h"
//#include "unit_tests/lss_one_dim_space_variable_pure_heat_equation_t.h"
//#include "unit_tests/lss_two_dim_pure_heat_equation_cuda_t.h"
//#include "unit_tests/lss_two_dim_pure_heat_equation_t.h"
#include "unit_tests/containers/lss_container_2d_t.hpp"
#include "unit_tests/dense_solvers/lss_dense_solvers_cuda_t.hpp"
#include "unit_tests/pde_solvers/one_dimensional/lss_general_svc_heat_equation_t.hpp"
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
    // testDoubleSweepMixBC();
    // testThomasLUDirichletBC();
    // testThomasLURobinBC();
    // testThomasLUNeumannDirichletBC();
    // testThomasLUDirichletNeumannBC();
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
    // ================ lss_one_dim_pure_heat_equation_t.h ================
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

    // ====================================================================
    // ============== lss_one_dim_black_scholes_equation_t.h ==============
    // ====================================================================

    // testImplEuropeanBlackScholesCallDirichletBCDoubleSweep();
    // testImplEuropeanBlackScholesCallDirichletBCThomasLU();
    // testImplEuropeanBlackScholesPutDirichletBCDoubleSweep();
    // testImplEuropeanBlackScholesPutDirichletBCThomasLU();
    // testExplEuropeanBlackScholesCallOptionDirichletBC();
    // testExplEuropeanBlackScholesPutOptionDirichletBC();

    // ====================================================================

    // ====================================================================
    // ========= lss_one_dim_black_scholes_equation_cuda_t.h ==============
    // ====================================================================

    // testImplEuropeanBlackScholesCallDirichletBCDeviceCUDA();
    // testImplEuropeanBlackScholesPutDirichletBCDeviceCUDA();
    // testExplEuropeanBlackScholesCallDirichletBCDeviceCUDA();
    // testExplEuropeanBlackScholesPutDirichletBCDeviceCUDA();

    // ====================================================================

    // ====================================================================
    // ================ lss_two_dim_pure_heat_equation_t.h ================
    // ====================================================================

    // test2DImplPureHeatEquationDirichletBCDoubleSweep();
    // test2DImplPureHeatEquationNonHomDirichletBCDoubleSweep();
    // test2DImplNonHomPureHeatEquationDirichletBCDoubleSweep();
    // test2DExplPureHeatEquationDirichletBCADE();
    // test2DExplPureHeatEquationNonHomDirichletBCADE();

    // ====================================================================

    // ====================================================================
    // ============= lss_two_dim_pure_heat_equation_cuda_t.h ==============
    // ====================================================================

    // test2DImplPureHeatEquationDirichletBCDeviceCuda();
    // test2DImplPureHeatEquationNonHomDirichletBCDeviceCuda();
    // test2DImplNonHomPureHeatEquationDirichletBCDeviceCuda();
    // test2DExplPureHeatEquationDirichletBCDeviceCuda();
    // test2DExplPureHeatEquationNonHomDirichletBCDeviceCuda();

    // ====================================================================

    std::cout << "\n\n";

    std::cin.get();
    std::cin.get();
    return 0;
}
