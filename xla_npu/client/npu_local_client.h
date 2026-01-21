#ifndef CLIENT_NPU_LOCAL_CLIENT_H_
#define CLIENT_NPU_LOCAL_CLIENT_H_

#include <set>

#include "xla/client/client_library.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_npu/service/npu_compiler.h"
#include "xla/xla_npu/stream_executor/npu_platform.h"
#include "xla/xla_npu/stream_executor/npu_platform_id.h"

namespace xla {

absl::Status CreateNpuLocalClient() {
    // 在 stream_executor\npu_platform.cc 中实现了自动注册stream_executor::NpuPlatform(在进程启动时),
    // 因此这里可以直接获取
    TF_ASSIGN_OR_RETURN(se::Platform* const platform, PlatformUtil::GetPlatform("npu"));

    xla::Compiler::RegisterCompilerFactory(stream_executor::kNpuPlatformId,
                                           []() { return std::make_unique<NpuCompiler>(); });

    LocalClientOptions client_options;
    client_options.set_platform(platform);
    client_options.set_number_of_replicas(1);
    client_options.set_allowed_devices(std::set<int>{0});
    TF_ASSIGN_OR_RETURN(LocalClient* const client, ClientLibrary::GetOrCreateLocalClient(client_options));
}

}  // namespace xla

#endif  // CLIENT_NPU_LOCAL_CLIENT_H_