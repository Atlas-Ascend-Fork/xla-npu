#include "xla/xla_npu/dependency/tool/log.h"

#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include <iostream>

namespace npu {

void Log::Print(const char* file_path, int line, const char* func, Level log_level, const std::string& log_str) {
    // 日志级别转字符串（无修改）
    std::string logLevelString;
    switch (log_level) {
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
            break;
    }

    // 获取进程/线程ID
    pid_t pid = getpid();
    pid_t tid = syscall(SYS_gettid);
    std::string pid_str = absl::StrFormat("%6d", pid);
    std::string tid_str = absl::StrFormat("%6d", tid);

    // 提取纯文件名
    std::string file_path_str = file_path;
    // 从后查找最后一个路径分隔符 '/'
    size_t last_slash_pos = file_path_str.rfind('/');
    std::string pure_filename;
    if (last_slash_pos != std::string::npos) {
        // 截取最后一个'/'之后的部分（纯文件名）
        pure_filename = file_path_str.substr(last_slash_pos + 1);
    } else {
        // 没有找到'/'，说明是纯文件名，直接使用
        pure_filename = file_path_str;
    }

    // 日志打印格式：[PID:TID][XLA_NPU.LEVEL]: 日志内容
    std::cout << "[" << pid_str << ":" << tid_str << "]"
              << "[XLA_NPU." << logLevelString << "]: " << log_str;

    // 在DEBUG日志中追加打印 [@调用日志接口的函数名(文件名:行号)]
    if (log_level == Level::DEBUG) {
        std::cout << " [@" << func << "(" << pure_filename << ":" << line << ")]";
    }

    std::cout << std::endl;
}

}  // namespace npu