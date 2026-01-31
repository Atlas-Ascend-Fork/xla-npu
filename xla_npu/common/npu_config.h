#ifndef COMMON_NPU_CONFIG_H_
#define COMMON_NPU_CONFIG_H_

#include <string>
#include <unordered_map>
#include <variant>

namespace xla {

// npu动态配置读取类
class NpuConfig {
   public:
    enum class Key { NPU_AUTO_FUSE_SWITCH, NPU_AUTO_FUSE_COMPILE_ARTIFACT_ROOT_DIR };
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

#endif  // COMMON_NPU_CONFIG_H_