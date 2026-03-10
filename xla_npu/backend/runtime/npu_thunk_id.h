/**
 * Copyright (c)2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef XLA_XLA_NPU_BACKEND_RUNTIME_NPU_THUNK_ID
#define XLA_XLA_NPU_BACKEND_RUNTIME_NPU_THUNK_ID

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

#endif  // XLA_XLA_NPU_BACKEND_RUNTIME_NPU_THUNK_ID
