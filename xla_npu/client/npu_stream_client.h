#ifndef XLA_XLA_NPU_CLIENT_NPU_STREAM_CLIENT_H_
#define XLA_XLA_NPU_CLIENT_NPU_STREAM_CLIENT_H_

#include "absl/status/statusor.h"
#include "xla/xla_npu/pjrt/npu/npu_pjrt_stream_executor_client.h"

namespace xla {

absl::StatusOr<std::unique_ptr<NpuPjrtStreamExecutorClient>> GetNpuPjrtStreamExecutorClient();

}  // namespace xla

#endif  // XLA_XLA_NPU_CLIENT_NPU_STREAM_CLIENT_H_