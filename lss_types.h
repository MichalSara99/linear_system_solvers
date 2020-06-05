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
	// ============================= FlatMatrixSort =============================
	// ==========================================================================
	
	enum class FlatMatrixSort { RowMajor, ColumnMajor };


}






#endif ///_LSS_TYPES