#ifndef COMMON_NPU_LOG_H_
#define COMMON_NPU_LOG_H_

#include <iostream>

#include "absl/strings/str_format.h"

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

        std::cout << "NPU." << logLevelString << ": " << logStr << std::endl;
    }

    template <typename... Args>
    static void Print(Level logLevel, absl::FormatSpec<Args...> format, Args&&... args) {
        std::string logStr = absl::StrFormat(format, std::forward<Args>(args)...);
        Print(logLevel, logStr);
    }
};

}  // namespace xla

#endif  // COMMON_NPU_LOG_H_