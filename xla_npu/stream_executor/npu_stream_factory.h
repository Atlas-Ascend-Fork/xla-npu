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

#ifndef XLA_XLA_NPU_STREAM_EXECUTOR_NPU_STREAM_FACTORY_H_
#define XLA_XLA_NPU_STREAM_EXECUTOR_NPU_STREAM_FACTORY_H_

#include "xla/xla_npu/stream_executor/npu_stream.h"

namespace stream_executor {

class NpuStreamFactory {
 public:
  virtual ~NpuStreamFactory() = default;
  virtual absl::StatusOr<std::unique_ptr<NpuStream>> CreateStream(
      StreamExecutor* executor) const = 0;
  static void Register(std::unique_ptr<NpuStreamFactory> factory, int priority);
  static const NpuStreamFactory* GetFactory();
};

class NpuStreamDefaultFactory : public NpuStreamFactory {
 public:
  ~NpuStreamDefaultFactory() override = default;
  absl::StatusOr<std::unique_ptr<NpuStream>> CreateStream(
      StreamExecutor* executor) const override {
    auto npuStream = std::make_unique<NpuStream>(executor);
    TF_RETURN_IF_ERROR(npuStream->Init());
    return std::move(npuStream);
  }
};

template <class Factory>
class NpuStreamFactoryRegistrar {
 public:
  explicit NpuStreamFactoryRegistrar(int priority) {
    NpuStreamFactory::Register(std::make_unique<Factory>(), priority);
  }
};

}  // namespace stream_executor

#define REGISTER_NPU_STREAM_FACTORY(factory, priority) \
  INTERNAL_REGISTER_NPU_STREAM_FACTORY(factory, priority, __COUNTER__)

#define INTERNAL_REGISTER_NPU_STREAM_FACTORY(factory, priority, ctr) \
  static ::stream_executor::NpuStreamFactoryRegistrar<factory>       \
  INTERNAL_REGISTER_LOCAL_NPU_STREAM_FACTORY_NAME(ctr) {             \
    priority                                                         \
  }

// __COUNTER__ must go through another macro to be properly expanded
#define INTERNAL_REGISTER_LOCAL_NPU_STREAM_FACTORY_NAME(ctr) ___##ctr##__object_

#endif  // XLA_XLA_NPU_STREAM_EXECUTOR_NPU_STREAM_FACTORY_H_
