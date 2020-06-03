#pragma once
#if !defined(_LSS_TYPES)
#define _LSS_TYPES

#include<vector>
#include<tuple>
#include"lss_macros.h"

namespace lss_types {

	// ==========================================================================
	// ============================= MemorySpace ================================
	// ==========================================================================

	enum class MemorySpace { Host, Device };

	// ==========================================================================
	// ================== SparseSolverFactorizationDevice =======================
	// ==========================================================================

	enum class SparseSolverFactorizationDevice { QR, Cholesky };

	// ==========================================================================
	// ==================== SparseSolverFactorizationHost =======================
	// ==========================================================================

	enum class SparseSolverFactorizationHost { LU, QR, Cholesky };


}






#endif ///_LSS_TYPES