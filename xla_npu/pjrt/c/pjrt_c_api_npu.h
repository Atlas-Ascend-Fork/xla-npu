#ifndef PJRT_C_PJRT_C_API_NPU_H_
#define PJRT_C_PJRT_C_API_NPU_H_

#include "xla/pjrt/c/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Does not pass ownership of returned PJRT_Api* to caller.
const PJRT_Api* GetPjrtApi();

#ifdef __cplusplus
}
#endif

#endif //PJRT_C_PJRT_C_API_NPU_H_
