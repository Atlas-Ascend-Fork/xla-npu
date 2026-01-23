#ifndef COMMON_NPU_OPERATION_H_
#define COMMON_NPU_OPERATION_H_

#include "absl/status/status.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_npu/third_party/register/aclInterface.h"

namespace xla {

class NpuOperation {
   public:
    enum class MemcpyType { H2D, D2H };

    // 提示: 可以使用"xla/tsl/platform/errors.h"中TF_RETURN_IF_ERROR宏快速对结果进行判断和返回

    static absl::Status SetDevice(int device_id);
    static absl::Status CreateStream(aclrtStream* stream);
    static absl::Status Malloc(void** new_device_addr, size_t size_bytes);
    static absl::Status Memcpy(void* dst, void* src, size_t size_bytes, MemcpyType memcpyType);
};

}  // namespace xla

#endif  // COMMON_NPU_OPERATION_H_