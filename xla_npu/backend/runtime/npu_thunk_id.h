#ifndef BACKEND_RUNTIME_NPU_THUNK_ID
#define BACKEND_RUNTIME_NPU_THUNK_ID

#include <cstdint>

#include "xla/tsl/lib/gtl/int_type.h"

namespace xla {

// Unique identifier for a thunk. When creating a thunk graph it's up to the
// emitter to generate unique IDs.
TSL_LIB_GTL_DEFINE_INT_TYPE(NpuThunkId, uint64_t);

// Generates unique IDs for thunks. This class is thread-compatible.
class NpuThunkIdGenerator {
   public:
    NpuThunkId GetNextThunkId() { return NpuThunkId(next_thunk_id_++); }

   private:
    uint64_t next_thunk_id_ = 1;
};

}  // namespace xla

#endif  // BACKEND_RUNTIME_NPU_THUNK_ID
