/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef XLA_XLA_NPU_COMMON_NPU_CONFIG_H_
#define XLA_XLA_NPU_COMMON_NPU_CONFIG_H_

#include "xla/xla_npu/dependency/tool/config_loader.h"

namespace xla {

class NpuConfig {
 public:
  enum class Key {
    NPU_AUTO_FUSE_COMPILE_ARTIFACT_ROOT_DIR,
    USE_OLD_CANN_STYLE,
    // 由于存在config_loader和log循环依赖的问题,
    // XLA_NPU_LOG_LEVEL(控制显示日志级别)环境变量直接定义再了log类中
  };

  NpuConfig(const NpuConfig&) = delete;
  NpuConfig& operator=(const NpuConfig&) = delete;

  static NpuConfig& GetInstance() {
    static NpuConfig instance;
    return instance;
  }

  void Init();

 private:
  NpuConfig() = default;
  ~NpuConfig() = default;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_COMMON_NPU_CONFIG_H_
