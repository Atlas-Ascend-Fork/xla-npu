/**
 * Copyright (c)2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "xla/xla_npu/common/npu_config.h"

#include "xla/tsl/util/env_var.h"
#include "xla/xla_npu/common/npu_constant.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace xla {

NpuConfig::NpuConfig()
    : key_desc_map_{
          {Key::NPU_AUTO_FUSE_COMPILE_ARTIFACT_ROOT_DIR,
           {"NPU_AUTO_FUSE_COMPILE_ARTIFACT_ROOT_DIR", ConfigValue(std::string("")), Type::STRING}},
          {Key::USE_OLD_CANN_STYLE, {"USE_OLD_CANN_STYLE", ConfigValue(false), Type::BOOL}},
      } {}

void NpuConfig::Init() {
    NPU_LOG_FORMAT_PRINT(INFO, "Init user config for %s.", NpuConstant::platformName);

    for (const auto& [key, desc] : key_desc_map_) {
        absl::Status status = absl::OkStatus();

        switch (desc.type) {
            case Type::STRING: {
                std::string str_val;
                status = tsl::ReadStringFromEnvVar(desc.str_desc, std::get<std::string>(desc.default_value), &str_val);
                if (status.ok()) {
                    value_map_[key] = str_val;
                    NPU_LOG_FORMAT_PRINT(INFO, "Set config key - '%s', value - '%s', type - '%d' success.",
                                         desc.str_desc, str_val, static_cast<int>(desc.type));
                }
                break;
            }
            case Type::BOOL: {
                bool bool_val;
                status = tsl::ReadBoolFromEnvVar(desc.str_desc, std::get<bool>(desc.default_value), &bool_val);
                if (status.ok()) {
                    value_map_[key] = bool_val;
                    NPU_LOG_FORMAT_PRINT(INFO, "Set config key - '%s', value - '%v', type - '%d' success.",
                                         desc.str_desc, bool_val, static_cast<int>(desc.type));
                }
                break;
            }
            default:
                NPU_LOG_FORMAT_PRINT(ERROR, "Set config key - '%s' failed, config type - '%d' not support.",
                                     desc.str_desc, static_cast<int>(desc.type));
        }

        if (!status.ok()) {
            NPU_LOG_FORMAT_PRINT(WARN, "Set config key - '%s', type - '%d' failed, error status: %s.", desc.str_desc,
                                 static_cast<int>(desc.type), status.ToString().c_str());
        }
    }
}

NpuConfig::Desc NpuConfig::GetDesc(NpuConfig::Key key) const {
    auto it = key_desc_map_.find(key);
    if (it == key_desc_map_.end()) {
        NPU_LOG_FORMAT_PRINT(ERROR, "Config key - '%d' not found in desc map.", static_cast<int>(key));
        throw std::out_of_range("Config key not found");
    }
    return it->second;
}

NpuConfig::ConfigValue NpuConfig::GetConfigValue(NpuConfig::Key key) const {
    auto it = value_map_.find(key);
    if (it != value_map_.end()) {
        return it->second;
    }
    // 未找到时返回默认值
    NPU_LOG_FORMAT_PRINT(WARN, "Config key - '%d' not found in value map, use default value.", static_cast<int>(key));
    const Desc& desc = GetDesc(key);
    return desc.default_value;
}

template <typename T>
T NpuConfig::GetTypedValue(NpuConfig::Key key) const {
    ConfigValue val = GetConfigValue(key);
    if (auto* typed_val = std::get_if<T>(&val)) {
        return *typed_val;
    }
    // 类型不匹配时尝试从默认值获取
    const Desc& desc = GetDesc(key);
    if (auto* default_val = std::get_if<T>(&desc.default_value)) {
        NPU_LOG_FORMAT_PRINT(WARN, "Config key - '%d' type mismatch, use default value.", static_cast<int>(key));
        return *default_val;
    }
    throw std::bad_variant_access();
}
// 显式实例化仅支持的类型（必须显式实例化，否则链接报错）
template std::string NpuConfig::GetTypedValue<std::string>(Key key) const;
template bool NpuConfig::GetTypedValue<bool>(Key key) const;

}  // namespace xla