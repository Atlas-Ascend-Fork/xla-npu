/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef XLA_XLA_NPU_DEPENDENCY_TOOL_CONFIG_LOADER_H_
#define XLA_XLA_NPU_DEPENDENCY_TOOL_CONFIG_LOADER_H_

#include <string>
#include <unordered_map>
#include <variant>

namespace npu {

class ConfigLoader {
 public:
  enum class ValueType { STRING, BOOL };
  using DefaultValueType = std::variant<std::string, bool>;
  struct Desc {
    std::string str_desc;
    DefaultValueType default_value;
    ValueType type;
  };

  using ConfigDescMap = std::unordered_map<int, Desc>;

  ConfigLoader(const ConfigLoader&) = delete;
  ConfigLoader& operator=(const ConfigLoader&) = delete;

  static ConfigLoader& GetInstance() {
    static ConfigLoader instance;
    return instance;
  }

  void Register(const ConfigDescMap& desc_map);
  void Register(int key, DefaultValueType value);

  Desc GetDesc(int key) const;
  DefaultValueType GetConfigValue(int key) const;
  template <typename T>
  T GetTypedValue(int key) const;

 private:
  ConfigLoader() = default;
  ~ConfigLoader() = default;

  std::unordered_map<int, DefaultValueType> value_map_;
};

template <typename T>
T ConfigLoader::GetTypedValue(int key) const {
  DefaultValueType val = GetConfigValue(key);
  if (auto* typed_val = std::get_if<T>(&val)) {
    return *typed_val;
  }
  const Desc& desc = GetDesc(key);
  if (auto* default_val = std::get_if<T>(&desc.default_value)) {
    return *default_val;
  }
  throw std::bad_variant_access();
}

}  // namespace npu

#define NPU_CONFIG_GET_STRING(key)                             \
  npu::ConfigLoader::GetInstance().GetTypedValue<std::string>( \
      static_cast<int>(key))

#define NPU_CONFIG_GET_BOOL(key) \
  npu::ConfigLoader::GetInstance().GetTypedValue<bool>(static_cast<int>(key))

#endif  // XLA_XLA_NPU_DEPENDENCY_TOOL_CONFIG_LOADER_H_
