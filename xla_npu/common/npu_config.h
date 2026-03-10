/**
 * Copyright (c)2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef XLA_XLA_NPU_COMMON_NPU_CONFIG_H_
#define XLA_XLA_NPU_COMMON_NPU_CONFIG_H_

#include <string>
#include <unordered_map>
#include <variant>

namespace xla {

// npu动态配置读取类
class NpuConfig {
   public:
    enum class Key { NPU_AUTO_FUSE_COMPILE_ARTIFACT_ROOT_DIR, USE_OLD_CANN_STYLE };
    enum class Type { STRING, BOOL };
    using ConfigValue = std::variant<std::string, bool>;
    struct Desc {
        std::string str_desc;
        ConfigValue default_value;
        Type type;
    };

    // 禁止拷贝和赋值
    NpuConfig(const NpuConfig&) = delete;
    NpuConfig& operator=(const NpuConfig&) = delete;

    static NpuConfig& GetInstance() {
        static NpuConfig instance;
        return instance;
    }
    void Init();

    Desc GetDesc(Key key) const;
    ConfigValue GetConfigValue(Key key) const;
    template <typename T>
    T GetTypedValue(Key key) const;

   private:
    // 禁止外部实例化
    NpuConfig();
    ~NpuConfig() = default;

    const std::unordered_map<Key, Desc> key_desc_map_;
    std::unordered_map<Key, ConfigValue> value_map_;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_COMMON_NPU_CONFIG_H_