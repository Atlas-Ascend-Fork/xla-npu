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

#include <cstdlib>
#include <string>
#include "xla/xla_npu/dependency/tool/config_loader.h"

namespace xla {

class NpuConfig {
 public:
  enum class Key {
    NPU_AUTO_FUSE_COMPILE_ARTIFACT_ROOT_DIR, // 融合算子编译产物输出位置
    USE_OLD_CANN_STYLE, // CANN 8.5.0及以下版本在基于afir后端进行算子融合时需要打开此开关
    XLA_NPU_MAX_PRINT_COUNT, // 控制融合算子和ACLNN算子的输入输出打印元素数量
    // 由于存在config_loader和log循环依赖的问题,
    // XLA_NPU_LOG_LEVEL(控制显示日志级别)环境变量直接定义在了log类中
  };

  NpuConfig(const NpuConfig&) = delete;
  NpuConfig& operator=(const NpuConfig&) = delete;

  static NpuConfig& GetInstance() {
    static NpuConfig instance;
    return instance;
  }

  void Init();

  int GetMaxPrintCount() const {
    const char* env = getenv("XLA_NPU_MAX_PRINT_COUNT");
    if (env) {
      try {
        return std::stoi(env);
      } catch (...) {
        return 8;  // 默认值
      }
    }
    return 8;  // 默认值
  }

 private:
  NpuConfig() = default;
  ~NpuConfig() = default;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_COMMON_NPU_CONFIG_H_
