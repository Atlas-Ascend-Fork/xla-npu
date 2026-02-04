#ifndef XLA_XLA_NPU_PJRT_C_PJRT_C_API_NPU_INTERNAL_H_
#define XLA_XLA_NPU_PJRT_C_PJRT_C_API_NPU_INTERNAL_H_
#include "xla/pjrt/c/pjrt_c_api.h"

namespace pjrt {
namespace npu_plugin {

const PJRT_Api* GetNpuPjrtApi();

}  // namespace npu_plugin
}  // namespace pjrt
#endif  // XLA_XLA_NPU_PJRT_C_PJRT_C_API_NPU_INTERNAL_H_
