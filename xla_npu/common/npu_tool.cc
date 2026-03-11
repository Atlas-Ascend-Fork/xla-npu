/**
 * Copyright (c)2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#include "xla/xla_npu/common/npu_tool.h"

#include <string>

#include "absl/random/random.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"

namespace xla {

// 生成不重复文件夹名
std::string NpuTool::generate_unique_folder_name_absl() {
  // 1. 生成秒级时间戳（Abseil 跨平台时间工具）
  absl::Time now = absl::Now();
  // 格式：YYYYMMDDHHMMSSmmm（年-月-日-时-分-秒）
  std::string timestamp =
      absl::FormatTime("%Y%m%d%H%M%S", now, absl::LocalTimeZone());
  // 移除毫秒后的小数点（%3f 会输出 123.456，取前3位）
  timestamp = timestamp.substr(0, timestamp.find('.'));

  // 2. 生成6位随机数（Abseil 正确接口：直接传入 bitgen）
  static absl::BitGen bitgen;  // 全局随机数生成器，仅初始化一次
  // 直接调用 absl::Uniform，传入 bitgen + 范围
  int random_num = absl::Uniform<int>(bitgen, 100000, 999999);  // 6位随机数

  // 3. 拼接（时间戳_随机数），无非法字符
  return absl::StrCat(timestamp, "_", random_num);
}

// 根据 Shape 的元素类型自动选择打印类型
void NpuTool::PrintNpuDeviceMemoryValueByShape(const std::string& title,
                                               const Shape& xla_shape,
                                               const void* device_addr,
                                               int64_t max_print_count) {
  int64_t size = GetXlaShapeSize(xla_shape);
  PrimitiveType element_type = xla_shape.element_type();

  switch (element_type) {
    case F32:
      PrintNpuDeviceMemoryValue<float>(title, size, device_addr,
                                       max_print_count);
      break;
    case F64:
      PrintNpuDeviceMemoryValue<double>(title, size, device_addr,
                                        max_print_count);
      break;
    case S32:
      PrintNpuDeviceMemoryValue<int32_t>(title, size, device_addr,
                                         max_print_count);
      break;
    case S64:
      PrintNpuDeviceMemoryValue<int64_t>(title, size, device_addr,
                                         max_print_count);
      break;
    case U32:
      PrintNpuDeviceMemoryValue<uint32_t>(title, size, device_addr,
                                          max_print_count);
      break;
    case U64:
      PrintNpuDeviceMemoryValue<uint64_t>(title, size, device_addr,
                                          max_print_count);
      break;
    case F16:
      PrintNpuDeviceMemoryValue<uint16_t>(title, size, device_addr,
                                          max_print_count);
      break;
    case BF16:
      PrintNpuDeviceMemoryValue<uint16_t>(title, size, device_addr,
                                          max_print_count);
      break;
    case S16:
      PrintNpuDeviceMemoryValue<int16_t>(title, size, device_addr,
                                         max_print_count);
      break;
    case U16:
      PrintNpuDeviceMemoryValue<uint16_t>(title, size, device_addr,
                                          max_print_count);
      break;
    case S8:
      PrintNpuDeviceMemoryValue<int8_t>(title, size, device_addr,
                                        max_print_count);
      break;
    case U8:
      PrintNpuDeviceMemoryValue<uint8_t>(title, size, device_addr,
                                         max_print_count);
      break;
    default:
      // 默认使用 float 类型打印
      NPU_LOG_FORMAT_PRINT(DEBUG, "%s (unsupported type %d, using float)",
                           title.c_str(), element_type);
      PrintNpuDeviceMemoryValue<float>(title, size, device_addr,
                                       max_print_count);
      break;
  }
}

}  // namespace xla
