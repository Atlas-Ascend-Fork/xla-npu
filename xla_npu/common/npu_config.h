/* Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_XLA_NPU_COMMON_NPU_CONFIG_H_
#define XLA_XLA_NPU_COMMON_NPU_CONFIG_H_

#include <cstdlib>
#include <string>
#include "xla/xla_npu/dependency/tool/config_loader.h"

namespace xla {

class NpuConfig {
 public:
  enum class Key {
    NPU_AUTO_FUSE_COMPILE_ARTIFACT_ROOT_DIR,  // 融合算子编译产物输出位置
    USE_OLD_CANN_STYLE,  // CANN 8.5.0及以下版本在基于afir后端进行算子融合时需要打开此开关
    XLA_NPU_MAX_PRINT_COUNT,  // 控制融合算子和ACLNN算子的输入输出打印元素数量
    XLA_NPU_DEVICE_ID,          // 指定使用的NPU设备ID
    ENABLE_SYNC_AFTER_EXECUTE,  // 是否在算子执行后同步(用于custom kernel,与ENABLE_PRINT_IO_VALUES独立)
    ENABLE_PRINT_IO_VALUES,  // 是否打印输入输出值(包含d2h操作,影响性能;会自动同步流,无需额外设置ENABLE_SYNC_AFTER_EXECUTE)
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

 private:
  NpuConfig() = default;
  ~NpuConfig() = default;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_COMMON_NPU_CONFIG_H_
