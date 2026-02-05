#ifndef XLA_XLA_NPU_DEPENDENCY_TOOL_LOG_H_
#define XLA_XLA_NPU_DEPENDENCY_TOOL_LOG_H_

#include "absl/strings/str_format.h"

#define NPU_LOG_PRINT(level, log_str) npu::Log::Print(__FILE__, __LINE__, __func__, npu::Log::Level::level, log_str)

#define NPU_LOG_FORMAT_PRINT(level, format, ...) \
    npu::Log::Print(__FILE__, __LINE__, __func__, npu::Log::Level::level, format, ##__VA_ARGS__)

// 封装具有返回absl::Status状态的日志打印宏
#define CHECK_COND_AND_PRINT_NPU_LOG(err_cond_expr, err_format, ...)                \
    do {                                                                            \
        if ((err_cond_expr)) {                                                      \
            const std::string err_msg = absl::StrFormat(err_format, ##__VA_ARGS__); \
            NPU_LOG_PRINT(ERROR, err_msg);                                          \
            return absl::InternalError(err_msg);                                    \
        }                                                                           \
    } while (0)

// 提示: 可以使用"xla/tsl/platform/errors.h"中TF_RETURN_IF_ERROR宏快速对absl::Status结果进行判断和返回

namespace npu {

// npu日志打印控制类
class Log {
   public:
    enum class Level { DEBUG, INFO, WARN, ERROR };

    static void Print(const char* file_path, int line, const char* func, Level log_level, const std::string& log_str);

    template <typename... Args>
    static void Print(const char* file_path, int line, const char* func, Level log_level,
                      absl::FormatSpec<Args...> format, Args&&... args) {
        const std::string& log_str = absl::StrFormat(format, std::forward<Args>(args)...);
        Print(file_path, line, func, log_level, log_str);
    }
};

}  // namespace npu

#endif  // XLA_XLA_NPU_DEPENDENCY_TOOL_LOG_H_