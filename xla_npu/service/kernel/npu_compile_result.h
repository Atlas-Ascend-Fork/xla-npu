/**
 * Copyright (c)2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef XLA_XLA_NPU_SERVICE_KERNEL_NPU_COMPILE_RESULT_H_
#define XLA_XLA_NPU_SERVICE_KERNEL_NPU_COMPILE_RESULT_H_

#include <string>
#include <unordered_map>

namespace xla {

struct NpuArtifactInfo {
    std::string name;  // 编译产物以此命名
    std::string dir;   // 包含所有对应类型kernel编译产物的完整根路径
};

class NpuCompileResult {
   public:
    // key为编译产物名称(afir的编译产物以hlo computation的name命名)
    std::unordered_map<std::string, NpuArtifactInfo> artifact_infos;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_KERNEL_NPU_COMPILE_RESULT_H_