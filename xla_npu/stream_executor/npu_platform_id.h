#ifndef XLA_XLA_NPU_STREAM_EXECUTOR_NPU_PLATFORM_ID_H_
#define XLA_XLA_NPU_STREAM_EXECUTOR_NPU_PLATFORM_ID_H_

#include "xla/stream_executor/platform.h"

namespace stream_executor {

// Opaque and unique identifier for the npu platform.
// This is needed so that plugins can refer to/identify this platform without
// instantiating a NpuPlatform object.
// This is broken out here to avoid a circular dependency between NpuPlatform
// and NpuStreamExecutor.
extern const Platform::Id kNpuPlatformId;

}  // namespace stream_executor

#endif  // XLA_XLA_NPU_STREAM_EXECUTOR_NPU_PLATFORM_ID_H_
