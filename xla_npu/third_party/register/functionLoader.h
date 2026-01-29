#ifndef REGISTER_FUNCTIONLOADER_H_
#define REGISTER_FUNCTIONLOADER_H_

#include <dlfcn.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace npu {

/**
  FunctionLoader is used to store function address in the process.
  */
class FunctionLoader {
   public:
    /**
        ctr
        */
    explicit FunctionLoader(const std::string& filename, int flags = RTLD_LAZY);
    /**
        dectr
        */
    ~FunctionLoader();
    /**
        set function name
        仅注册lib或者func的名称,在get时会进行校验,未注册的不允许调用.
        */
    void Set(const std::string& name);
    /**
        get function address by function name.
        */
    void* Get(const std::string& name);

   private:
    mutable std::mutex mu_;
    std::string fileName;
    int flags;
    void* handle = nullptr;
    mutable std::unordered_map<std::string, void*> registry;
};  // class FunctionLoader

/**
  this class is used to register
  */
class FunctionRegister {
   public:
    /**
        Singleton
        */
    static FunctionRegister* GetInstance();
    /**
        this API is used to store FunctionLoader class
        */
    void Register(const std::string& name, std::unique_ptr<FunctionLoader>& ptr);
    /**
        this API is used to associate library name and function name.
        */
    void Register(const std::string& name, const std::string& funcName);
    /**
        this API is used to get the function address by library and function name.
        */
    void* Get(const std::string& soName, const std::string& funcName);

   private:
    FunctionRegister() = default;
    mutable std::mutex mu_;
    mutable std::unordered_map<std::string, std::unique_ptr<FunctionLoader>> registry;
};  // class FunctionRegister

/**
  FunctionRegisterBuilder is the helper of FunctionRegister.
  */
class FunctionRegisterBuilder {
   public:
    /**
        ctr
        */
    FunctionRegisterBuilder(const std::string& soName, std::unique_ptr<FunctionLoader>& libPtr) noexcept;
    /**
        ctr
        */
    FunctionRegisterBuilder(const std::string& soName, const std::string& funcName) noexcept;
};  // class FunctionRegisterBuilder

#define REGISTER_LIBRARY(soName, ...)                                                                              \
    auto library_##soName = std::unique_ptr<npu::FunctionLoader>(new npu::FunctionLoader(#soName, ##__VA_ARGS__)); \
    static npu::FunctionRegisterBuilder register_library_##soName(#soName, library_##soName);

#define REGISTER_FUNCTION(soName, funcName) \
    static npu::FunctionRegisterBuilder register_function_##funcName(#soName, #funcName);

// 错误值获取模板：默认返回-1，可特化适配不同类型
template <typename T>
constexpr T GetDefaultError() {
    // 对整数类型默认返回-1，其他类型返回默认构造值
    if constexpr (std::is_integral_v<T>) {
        return static_cast<T>(-1);
    } else {
        return T{};
    }
}

// 去除传参列表中参数列表外的额外括号
#define _REMOVE_PARENS(...) __VA_ARGS__
#define STRIP_PARENS(x) _REMOVE_PARENS x

// 声明待注册的函数(为了便于使用,同时自动完成了代码实现)
// 参数说明：
// soName: 库名（如libopapi）
// ret: 返回值类型（如aclnnStatus）
// name: 函数名（如AclnnExpandGetWorkspaceSize）
// args_type: 类型+参数名列表（如const aclTensor* self, ...）
// args_name: 仅参数名列表（如self, size, ...）
#define DECLARE_ACL_FUNC(soName, ret, name, args_type, args_name)                                               \
    inline ret name args_type {                                                                                 \
        using FuncPtr = ret(*) args_type;                                                                       \
        static FuncPtr cached_func = nullptr;                                                                   \
        if (!cached_func) {                                                                                     \
            cached_func = reinterpret_cast<FuncPtr>(npu::FunctionRegister::GetInstance()->Get(#soName, #name)); \
        }                                                                                                       \
        if (cached_func) {                                                                                      \
            return cached_func(STRIP_PARENS(args_name));                                                        \
        } else {                                                                                                \
            return npu::GetDefaultError<ret>();                                                                 \
        }                                                                                                       \
    }

}  // namespace npu

#endif  // REGISTER_FUNCTIONLOADER_H_
