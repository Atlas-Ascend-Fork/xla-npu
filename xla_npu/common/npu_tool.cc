#include "xla/xla_npu/common/npu_tool.h"

#include <string>

#include "absl/random/random.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"

namespace xla {

// 生成不重复文件夹名
std::string NpuTool::generate_unique_folder_name_absl() {
    // 1. 生成秒级时间戳（Abseil 跨平台时间工具）
    absl::Time now = absl::Now();
    // 格式：YYYYMMDDHHMMSSmmm（年-月-日-时-分-秒）
    std::string timestamp = absl::FormatTime("%Y%m%d%H%M%S", now, absl::LocalTimeZone());
    // 移除毫秒后的小数点（%3f 会输出 123.456，取前3位）
    timestamp = timestamp.substr(0, timestamp.find('.'));

    // 2. 生成6位随机数（Abseil 正确接口：直接传入 bitgen）
    static absl::BitGen bitgen;  // 全局随机数生成器，仅初始化一次
    // 直接调用 absl::Uniform，传入 bitgen + 范围
    int random_num = absl::Uniform<int>(bitgen, 100000, 999999);  // 6位随机数

    // 3. 拼接（时间戳_随机数），无非法字符
    return absl::StrCat(timestamp, "_", random_num);
}

}  // namespace xla
