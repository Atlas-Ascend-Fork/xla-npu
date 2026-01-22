#ifndef CLIENT_NPU_STREAM_CLIENT_H_
#define CLIENT_NPU_STREAM_CLIENT_H_

#include "absl/status/statusor.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"

namespace xla {

// absl::StatusOr<LocalClient*> CreateNpuLocalClient() {
// // 在 stream_executor\npu_platform.cc 中实现了自动注册stream_executor::NpuPlatform(在进程启动时),
// // 因此这里可以直接获取
// TF_ASSIGN_OR_RETURN(se::Platform* const platform, PlatformUtil::GetPlatform("npu"));

// xla::Compiler::RegisterCompilerFactory(stream_executor::kNpuPlatformId,
//                                        []() { return std::make_unique<NpuCompiler>(); });

// LocalClientOptions client_options;
// client_options.set_platform(platform);
// client_options.set_number_of_replicas(1);
// client_options.set_allowed_devices(std::set<int>{0});
// TF_ASSIGN_OR_RETURN(LocalClient* const client, ClientLibrary::GetOrCreateLocalClient(client_options));
// }

absl::StatusOr<std::unique_ptr<PjRtStreamExecutorClient>> CreateNpuPjrtStreamExecutorClient();

}  // namespace xla

#endif  // CLIENT_NPU_STREAM_CLIENT_H_