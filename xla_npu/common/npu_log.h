#ifndef COMMON_NPU_LOG_H_
#define COMMON_NPU_LOG_H_

#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include <iostream>

#include "absl/strings/str_format.h"

// 封装具有返回absl::Status状态的日志打印宏
#define CHECK_COND_AND_PRINT_NPU_LOG(err_cond_expr, err_format, ...)                \
    do {                                                                            \
        if ((err_cond_expr)) {                                                      \
            const std::string err_msg = absl::StrFormat(err_format, ##__VA_ARGS__); \
            xla::NpuLog::Print(xla::NpuLog::Level::ERROR, err_msg);                 \
            return absl::InternalError(err_msg);                                    \
        }                                                                           \
    } while (0)

// 提示: 可以使用"xla/tsl/platform/errors.h"中TF_RETURN_IF_ERROR宏快速对absl::Status结果进行判断和返回

namespace xla {

class NpuLog {
   public:
    enum class Level { DEBUG, INFO, WARN, ERROR };

    static void Print(Level logLevel, const std::string& logStr) {
        std::string logLevelString;
        switch (logLevel) {
            case Level::DEBUG:
                logLevelString = "DEBUG";
                break;
            case Level::INFO:
                logLevelString = "INFO";
                break;
            case Level::WARN:
                logLevelString = "WARN";
                break;
            case Level::ERROR:
                logLevelString = "ERROR";
        }

        pid_t pid = getpid();             // 获取进程ID
        pid_t tid = syscall(SYS_gettid);  // 获取线程ID（Linux特有）
        // %6d 表示宽度为6，右对齐，不足补空格；如需左对齐用 %-6d
        std::string pid_str = absl::StrFormat("%6d", pid);
        std::string tid_str = absl::StrFormat("%6d", tid);

        std::cout << "[" << pid_str << ":" << tid_str << "]" << "XLA_NPU." << logLevelString << ": " << logStr
                  << std::endl;
    }

    template <typename... Args>
    static void Print(Level logLevel, absl::FormatSpec<Args...> format, Args&&... args) {
        std::string logStr = absl::StrFormat(format, std::forward<Args>(args)...);
        Print(logLevel, logStr);
    }
};

}  // namespace xla

#endif  // COMMON_NPU_LOG_H_