#ifndef PJRT_COMMON_H
#define PJRT_COMMON_H
#include "xla/pjrt/pjrt_common.h"

namespace xla {
constexpr char kNpuPlatformName[] = "npu";

inline const char* NpuName()
{
    return kNpuPlatformName;
}
}
#endif  //PJRT_COMMON_H
