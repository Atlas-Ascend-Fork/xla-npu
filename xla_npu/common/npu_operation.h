#ifndef COMMON_NPU_OPERATION_H_
#define COMMON_NPU_OPERATION_H_

#include "absl/status/status.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_npu/dependency/interface/acl_interface.h"

namespace xla {

// npu硬件交互接口封装类
class NpuOperation {
   public:
    enum class MemcpyType { H2D, D2H };

    static absl::Status SetDevice(int device_id);
    static absl::Status CreateStream(aclrtStream* stream);
    static absl::Status Malloc(void** new_device_addr, size_t size_bytes);
    static absl::Status Memcpy(void* dst, void* src, size_t size_bytes, MemcpyType memcpyType);
};

}  // namespace xla

#endif  // COMMON_NPU_OPERATION_H_