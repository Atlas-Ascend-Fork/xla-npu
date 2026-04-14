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

#ifndef XLA_XLA_NPU_SERVICE_KERNEL_NPU_COMPILE_RESULT_H_
#define XLA_XLA_NPU_SERVICE_KERNEL_NPU_COMPILE_RESULT_H_

#include <string>
#include <unordered_map>

namespace xla {

struct NpuArtifactInfo {
  std::string name;  // 编译产物以此命名
  std::string dir;  // 包含所有对应类型kernel编译产物的完整根路径
};

class NpuCompileResult {
 public:
  // key为编译产物名称(afir的编译产物以hlo computation的name命名)
  std::unordered_map<std::string, NpuArtifactInfo> artifact_infos;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_KERNEL_NPU_COMPILE_RESULT_H_