/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "xla/xla_npu/common/npu_config.h"

namespace xla {

void NpuConfig::Init() {
  npu::ConfigLoader::ConfigDescMap desc_map = {
      {static_cast<int>(Key::NPU_AUTO_FUSE_COMPILE_ARTIFACT_ROOT_DIR),
       {"NPU_AUTO_FUSE_COMPILE_ARTIFACT_ROOT_DIR",
        npu::ConfigLoader::DefaultValueType(std::string("")),
        npu::ConfigLoader::ValueType::STRING}},
      {static_cast<int>(Key::USE_OLD_CANN_STYLE),
       {"USE_OLD_CANN_STYLE", npu::ConfigLoader::DefaultValueType(false),
        npu::ConfigLoader::ValueType::BOOL}},
  };
  npu::ConfigLoader::GetInstance().Register(desc_map);
}

}  // namespace xla
