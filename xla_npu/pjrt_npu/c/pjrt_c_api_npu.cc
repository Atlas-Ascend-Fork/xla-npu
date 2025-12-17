#include "pjrt_c_api_npu.h"
#include "pjrt_c_api_npu_internal.h"

const PJRT_Api* GetPjrtApi() { return pjrt::npu_plugin::GetNpuPjrtApi(); }