#ifndef PJRT_NPU_PJRT_COMMON_H_
#define PJRT_NPU_PJRT_COMMON_H_

#include "xla/pjrt/pjrt_common.h"

namespace xla {

constexpr char kNpuPlatformName[] = "npu";

inline const char* NpuName() {
    return kNpuPlatformName;
}

}
#endif  //PJRT_NPU_PJRT_COMMON_H_
