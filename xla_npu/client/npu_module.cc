#include "xla/xla_npu/client/npu_module.h"

#include "xla/service/computation_placer.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/xla_npu/common/npu_constant.h"
#include "xla/xla_npu/common/npu_log.h"
#include "xla/xla_npu/service/npu_compiler.h"
#include "xla/xla_npu/service/npu_transfer_manager.h"
#include "xla/xla_npu/stream_executor/npu_platform.h"
#include "xla/xla_npu/stream_executor/npu_platform_id.h"

namespace xla {

static void RegisterNpuPlatform() {
    // 参考cpu
    std::unique_ptr<stream_executor::Platform> platform(new stream_executor::NpuPlatform);
    stream_executor::PlatformManager::RegisterPlatform(std::move(platform));
    NpuLog::Print(NpuLog::Level::INFO, "Resister platform for '%s'.", NpuConstant::platformName);
}

static void RegisterNpuCompiler() {
    // 参考cpu
    Compiler::RegisterCompilerFactory(stream_executor::kNpuPlatformId,
                                      []() { return std::make_unique<NpuCompiler>(); });
    NpuLog::Print(NpuLog::Level::INFO, "Resister compiler for '%s'.", NpuConstant::platformName);
}

static void RegisterNpuTransferManager() {
    // 参考cpu
    TransferManager::RegisterTransferManager(
        stream_executor::kNpuPlatformId,
        []() -> std::unique_ptr<xla::TransferManager> { return std::make_unique<NpuTransferManager>(); });
    NpuLog::Print(NpuLog::Level::INFO, "Resister transfer manager for '%s'.", NpuConstant::platformName);
}

static void RegisterNpuComputationPlacer() {
    // 和cpu/gpu相同,使用公共类
    ComputationPlacer::RegisterComputationPlacer(stream_executor::kNpuPlatformId,
                                                 []() { return std::make_unique<ComputationPlacer>(); });
    NpuLog::Print(NpuLog::Level::INFO, "Resister computation placer for '%s'.", NpuConstant::platformName);
}

void InitNpuSubModule() {
    RegisterNpuPlatform();
    RegisterNpuCompiler();
    RegisterNpuTransferManager();
    RegisterNpuComputationPlacer();
}

}  // namespace xla