/**
 * Copyright (c)2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#include "xla/xla_npu/client/npu_module.h"

#include "xla/service/computation_placer.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/xla_npu/common/npu_config.h"
#include "xla/xla_npu/common/npu_constant.h"
#include "xla/xla_npu/dependency/tool/log.h"
#include "xla/xla_npu/service/npu_compiler.h"
#include "xla/xla_npu/service/npu_transfer_manager.h"
#include "xla/xla_npu/stream_executor/npu_platform.h"
#include "xla/xla_npu/stream_executor/npu_platform_id.h"

namespace xla {

static void RegisterNpuPlatform() {
  // 用于执行流程.
  // 注册此platform后,
  // 如果调用接口(ClientLibrary::GetOrCreateLocalClient)使用此platform,
  // 则会触发创建并关联LocalService和Backend,其中Backend绑定在LocalService上
  // 实际执行逻辑会通过LocalClient中的LocalService(Backend)这两层中间类执行调用
  // 以典型的执行调用链为例
  //   'PjRtStreamExecutorClient'->RunAsync
  //     'LocalExecutable'->RunAsync //
  //     这里的LocalExecutable是PjRtStreamExecutorClient的compile后生成的,里面实际封装了NpuExecutable
  //       'NpuExecutable'->ExecuteAsyncOnStreamWrapper
  //         'NpuExecutable'->'ExecuteAsyncOnStream'
  //           'NpuExecutable'->'ExecuteThunks'
  std::unique_ptr<stream_executor::Platform> platform(
      new stream_executor::NpuPlatform);
  stream_executor::PlatformManager::RegisterPlatform(std::move(platform));
  NPU_LOG_FORMAT_PRINT(INFO, "Resister platform for '%s'.",
                       NpuConstant::platformName);
}

static void RegisterNpuCompiler() {
  // 用于编译流程.
  // 在PjRtStreamExecutorClient::CompileInternal中client()->Compile会调用到,
  // 实际编译逻辑会通过LocalClient中的LocalService(Backend)这两层中间类执行调用,
  // LocalClient即上面的client() 即如下的调用链
  //   'LocalClient'->Compile
  //     'LocalService'->CompileExecutables
  //       'LocalService'->BuildExecutable
  //         'Backend'->'NpuCompiler'->RunHloPasses
  //         'Backend'->'NpuCompiler'->RunBackend
  Compiler::RegisterCompilerFactory(stream_executor::kNpuPlatformId, []() {
    return std::make_unique<NpuCompiler>(stream_executor::kNpuPlatformId, "",
                                         "");
  });
  NPU_LOG_FORMAT_PRINT(INFO, "Resister compiler for '%s'.",
                       NpuConstant::platformName);
}

static void RegisterNpuTransferManager() {
  // 用于动态数据输入/输出
  TransferManager::RegisterTransferManager(
      stream_executor::kNpuPlatformId,
      []() -> std::unique_ptr<xla::TransferManager> {
        return std::make_unique<NpuTransferManager>();
      });
  NPU_LOG_FORMAT_PRINT(INFO, "Resister transfer manager for '%s'.",
                       NpuConstant::platformName);
}

static void RegisterNpuComputationPlacer() {
  // 和cpu/gpu相同,使用公共类
  // 用于给多卡分配计算任务
  ComputationPlacer::RegisterComputationPlacer(
      stream_executor::kNpuPlatformId,
      []() { return std::make_unique<ComputationPlacer>(); });
  NPU_LOG_FORMAT_PRINT(INFO, "Resister computation placer for '%s'.",
                       NpuConstant::platformName);
}

void InitNpuSubModule() {
  NpuConfig::GetInstance().Init();  // TODO 判断返回值

  RegisterNpuPlatform();
  RegisterNpuCompiler();
  RegisterNpuTransferManager();
  RegisterNpuComputationPlacer();
}

}  // namespace xla