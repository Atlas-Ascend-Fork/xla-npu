/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef XLA_XLA_NPU_SERVICE_KERNEL_NPU_KERNEL_CONFIG_H_
#define XLA_XLA_NPU_SERVICE_KERNEL_NPU_KERNEL_CONFIG_H_

#include <string>
#include <unordered_map>

#include "xla/xla_npu/service/kernel/npu_compile_result.h"

namespace xla {

class NpuKernelConfig {
   public:
    enum class NpuKernelType {
        FUSION = 0,
    };

    NpuKernelType npu_kernel_type;
    NpuArtifactInfo npu_artifact_info;
    
    // 存储所有相关的 computation HLO 文本，key 是 computation 名称
    std::unordered_map<std::string, std::string> computation_hlo_map;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_KERNEL_NPU_KERNEL_CONFIG_H_