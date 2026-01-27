#include "xla/xla_npu/client/npu_module.h"

#include "xla/service/computation_placer.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/xla_npu/common/npu_constant.h"
#include "xla/xla_npu/common/npu_log.h"
// #include "xla/xla_npu/service/gpu_compiler.h"
#include "xla/xla_npu/service/npu_compiler.h"
#include "xla/xla_npu/service/npu_transfer_manager.h"
#include "xla/xla_npu/stream_executor/npu_platform.h"
#include "xla/xla_npu/stream_executor/npu_platform_id.h"

namespace xla {

static void RegisterNpuPlatform() {
    // 用于执行流程.
    // 注册此platform后, 如果调用接口(ClientLibrary::GetOrCreateLocalClient)使用此platform,
    // 则会触发创建并关联LocalService和Backend,其中Backend绑定在LocalService上
    // 实际执行逻辑会通过LocalClient中的LocalService(Backend)这两层中间类执行调用
    // 以典型的执行调用链为例
    // PjRtStreamExecutorLoadedExecutable::EnqueueExecution
    //   'PjRtStreamExecutorClient'->RunAsync
    //     'LocalExecutable'->RunAsync //
    //     这里的LocalExecutable是PjRtStreamExecutorClient的compile后生成的,里面实际封装了NpuExecutable
    //       'NpuExecutable'->ExecuteAsyncOnStreamWrapper
    //         'NpuExecutable'->'ExecuteAsyncOnStream'
    //           'NpuExecutable'->'ExecuteThunks'
    std::unique_ptr<stream_executor::Platform> platform(new stream_executor::NpuPlatform);
    stream_executor::PlatformManager::RegisterPlatform(std::move(platform));
    NpuLog::Print(NpuLog::Level::INFO, "Resister platform for '%s'.", NpuConstant::platformName);
}

static void RegisterNpuCompiler() {
    // 用于编译流程.
    // 在PjRtStreamExecutorClient::CompileInternal中client()->Compile会调用到,
    // 实际编译逻辑会通过LocalClient中的LocalService(Backend)这两层中间类执行调用, LocalClient即上面的client()
    // 即如下的调用链
    // PjRtStreamExecutorClient::CompileInternal
    //   'LocalClient'->Compile
    //     'LocalService'->CompileExecutables
    //       'LocalService'->BuildExecutable
    //         'Backend'->'NpuCompiler'->RunHloPasses
    //         'Backend'->'NpuCompiler'->RunBackend
    Compiler::RegisterCompilerFactory(stream_executor::kNpuPlatformId,
                                      []() { return std::make_unique<NpuCompiler>(); });
    // Compiler::RegisterCompilerFactory(stream_executor::kNpuPlatformId, []() {
    //     return std::make_unique<gpu::GpuCompiler>(stream_executor::kNpuPlatformId, nullptr, nullptr);
    // });
    NpuLog::Print(NpuLog::Level::INFO, "Resister compiler for '%s'.", NpuConstant::platformName);
}

static void RegisterNpuTransferManager() {
    // 用于动态数据输入/输出
    TransferManager::RegisterTransferManager(
        stream_executor::kNpuPlatformId,
        []() -> std::unique_ptr<xla::TransferManager> { return std::make_unique<NpuTransferManager>(); });
    NpuLog::Print(NpuLog::Level::INFO, "Resister transfer manager for '%s'.", NpuConstant::platformName);
}

static void RegisterNpuComputationPlacer() {
    // 和cpu/gpu相同,使用公共类
    // 用于给多卡分配计算任务
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