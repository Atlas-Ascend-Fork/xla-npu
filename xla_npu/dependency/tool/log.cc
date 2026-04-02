/* Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/xla_npu/dependency/tool/log.h"

#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

namespace npu {

// 获取环境变量设置的日志级别
static Log::Level GetEnvLogLevel() {
  const char* env_level = std::getenv("XLA_NPU_LOG_LEVEL");
  if (env_level == nullptr) {
    return Log::Level::INFO;  // 默认级别为 INFO
  }

  std::string level_str = env_level;
  if (level_str == "DEBUG" || level_str == "0") {
    return Log::Level::DEBUG;
  } else if (level_str == "INFO" || level_str == "1") {
    return Log::Level::INFO;
  } else if (level_str == "WARN" || level_str == "2") {
    return Log::Level::WARN;
  } else if (level_str == "ERROR" || level_str == "3") {
    return Log::Level::ERROR;
  }
  return Log::Level::INFO;  // 默认级别为 INFO
}

// 检查是否需要打印该级别的日志
static bool ShouldPrint(Log::Level log_level) {
  static Log::Level min_level = GetEnvLogLevel();
  return static_cast<int>(log_level) >= static_cast<int>(min_level);
}

void Log::Print(const char* file_path, int line, const char* func,
                Level log_level, const std::string& log_str) {
  // 根据环境变量过滤日志级别
  if (!ShouldPrint(log_level)) {
    return;
  }

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

  // 获取当前时间
  auto now = std::chrono::system_clock::now();
  auto time_t_now = std::chrono::system_clock::to_time_t(now);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()) %
            1000;
  struct tm local_time;
  localtime_r(&time_t_now, &local_time);

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

  // 日志打印格式：[YYYY-MM-DD HH:MM:SS.mmm][PID:TID][XLA_NPU.LEVEL]: 日志内容
  std::cout << "[" << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S") << "."
            << std::setfill('0') << std::setw(3) << ms.count() << "]"
            << "[" << pid_str << ":" << tid_str << "]"
            << "[XLA_NPU." << logLevelString << "]: " << log_str;

  std::cout << std::endl;
}

}  // namespace npu