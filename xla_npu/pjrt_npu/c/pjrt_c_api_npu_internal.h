#ifndef PJRT_C_API_NPU_INTERNAL_H
#define PJRT_C_API_NPU_INTERNAL_H
#include "xla/pjrt/c/pjrt_c_api.h"

namespace pjrt {
namespace npu_plugin {

const PJRT_Api* GetNpuPjrtApi();

}  // namespace npu_plugin
}  // namespace pjrt
#endif  //PJRT_C_API_NPU_INTERNAL_H
