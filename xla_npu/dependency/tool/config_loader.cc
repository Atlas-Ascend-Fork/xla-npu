/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "xla/xla_npu/dependency/tool/config_loader.h"

#include "xla/tsl/util/env_var.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace npu {

void ConfigLoader::Register(const ConfigDescMap& desc_map) {
  NPU_LOG_FORMAT_PRINT(INFO, "Register config for ConfigLoader.");

  for (const auto& [key, desc] : desc_map) {
    absl::Status status = absl::OkStatus();

    switch (desc.type) {
      case ValueType::STRING: {
        std::string str_val;
        status = tsl::ReadStringFromEnvVar(
            desc.str_desc, std::get<std::string>(desc.default_value), &str_val);
        if (status.ok()) {
          value_map_[key] = str_val;
          NPU_LOG_FORMAT_PRINT(
              INFO, "Set config key - '%s', value - '%s', type - '%d' success.",
              desc.str_desc, str_val, static_cast<int>(desc.type));
        }
        break;
      }
      case ValueType::BOOL: {
        bool bool_val;
        status = tsl::ReadBoolFromEnvVar(
            desc.str_desc, std::get<bool>(desc.default_value), &bool_val);
        if (status.ok()) {
          value_map_[key] = bool_val;
          NPU_LOG_FORMAT_PRINT(
              INFO, "Set config key - '%s', value - '%v', type - '%d' success.",
              desc.str_desc, bool_val, static_cast<int>(desc.type));
        }
        break;
      }
      default:
        NPU_LOG_FORMAT_PRINT(
            ERROR,
            "Set config key - '%s' failed, config type - '%d' not support.",
            desc.str_desc, static_cast<int>(desc.type));
    }

    if (!status.ok()) {
      NPU_LOG_FORMAT_PRINT(
          WARN, "Set config key - '%s', type - '%d' failed, error status: %s.",
          desc.str_desc, static_cast<int>(desc.type),
          status.ToString().c_str());
    }
  }
}

void ConfigLoader::Register(int key, DefaultValueType value) {
  value_map_[key] = value;
}

ConfigLoader::Desc ConfigLoader::GetDesc(int key) const {
  NPU_LOG_FORMAT_PRINT(ERROR, "Config key - '%d' not found in desc map.", key);
  throw std::out_of_range("Config key not found");
}

ConfigLoader::DefaultValueType ConfigLoader::GetConfigValue(int key) const {
  auto it = value_map_.find(key);
  if (it != value_map_.end()) {
    return it->second;
  }
  NPU_LOG_FORMAT_PRINT(WARN, "Config key - '%d' not found in value map.", key);
  throw std::out_of_range("Config key not found");
}

template std::string ConfigLoader::GetTypedValue<std::string>(int key) const;
template bool ConfigLoader::GetTypedValue<bool>(int key) const;

}  // namespace npu
