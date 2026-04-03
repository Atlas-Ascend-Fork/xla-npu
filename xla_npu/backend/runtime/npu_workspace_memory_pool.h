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

#ifndef XLA_XLA_NPU_BACKEND_RUNTIME_NPU_WORKSPACE_MEMORY_POOL_H_
#define XLA_XLA_NPU_BACKEND_RUNTIME_NPU_WORKSPACE_MEMORY_POOL_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/xla_npu/dependency/interface/acl_interface.h"

namespace xla {

// Workspace 内存池常量定义
// 所有大小单位均为字节

// 总内存池大小: 1GB
inline constexpr size_t kWorkspaceMemoryPoolTotalSize =
    1ULL * 1024 * 1024 * 1024;

// 每个流分配的内存块大小: 1GB
inline constexpr size_t kWorkspaceMemoryPerStream = 1ULL * 1024 * 1024 * 1024;

// 最大支持的并发流数量 （暂时只支持一个流）
inline constexpr size_t kMaxStreamCount =
    kWorkspaceMemoryPoolTotalSize / kWorkspaceMemoryPerStream;

// 内存池对齐要求: 32字节对齐
inline constexpr size_t kWorkspaceMemoryAlignment = 32;

/**
 * NpuWorkspaceMemoryPool - Workspace 内存池管理器
 *
 * 设计目标:
 * 1. 预分配 1GB 设备内存作为 workspace 专用内存池
 * 2. 按流划分内存块,每个流分配 256MB
 * 3. 同一个流内的算子串行复用内存,不做清零操作
 * 4. 线程安全,支持多流并发访问
 *
 * 使用方式:
 *   auto* pool = NpuWorkspaceMemoryPool::GetInstance();
 *   TF_ASSIGN_OR_RETURN(void* workspace, pool->AllocateForStream(stream, size));
 *   // 使用 workspace 执行算子
 *   pool->ReleaseForStream(stream);  // 可选,流结束时调用
 */
class NpuWorkspaceMemoryPool {
 public:
  // 获取单例实例
  static NpuWorkspaceMemoryPool* GetInstance();

  // 禁止拷贝和移动
  NpuWorkspaceMemoryPool(const NpuWorkspaceMemoryPool&) = delete;
  NpuWorkspaceMemoryPool& operator=(const NpuWorkspaceMemoryPool&) = delete;

  /**
   * 为指定流分配 workspace 内存
   *
   * @param stream ACL 流句柄
   * @param size 需要的内存大小(字节)
   * @return 分配的内存地址,如果超出该流的内存块大小则返回错误
   */
  absl::StatusOr<void*> AllocateForStream(aclrtStream stream, size_t size);

  /**
   * 获取指定流的内存块基地址
   * 用于直接管理内存偏移的场景
   *
   * @param stream ACL 流句柄
   * @return 该流的内存块基地址
   */
  absl::StatusOr<void*> GetStreamMemoryBase(aclrtStream stream);

  /**
   * 获取指定流的内存块大小
   *
   * @param stream ACL 流句柄
   * @return 该流的内存块大小(字节)
   */
  size_t GetStreamMemorySize(aclrtStream stream) const;

  /**
   * 释放指定流占用的内存块
   * 通常在流结束时调用,会将该内存块标记为可复用
   *
   * @param stream ACL 流句柄
   */
  void ReleaseForStream(aclrtStream stream);

  /**
   * 获取内存池总大小
   */
  size_t GetTotalSize() const { return kWorkspaceMemoryPoolTotalSize; }

  /**
   * 获取每个流的内存块大小
   */
  size_t GetPerStreamSize() const { return kWorkspaceMemoryPerStream; }

  /**
   * 检查内存池是否已初始化
   */
  bool IsInitialized() const { return initialized_; }

  /**
   * 销毁内存池,释放所有设备内存
   * 通常在程序退出时调用
   */
  static void DestroyInstance();

 private:
  NpuWorkspaceMemoryPool() = default;
  ~NpuWorkspaceMemoryPool();

  // 初始化内存池,分配设备内存
  absl::Status Initialize();

  // 获取流的唯一标识
  uint64_t GetStreamId(aclrtStream stream) const;

  // 获取或分配流的内存块索引
  absl::StatusOr<size_t> GetOrAssignStreamIndex(aclrtStream stream);

  // 内存池是否已初始化
  bool initialized_ = false;

  // 设备内存基地址
  void* device_memory_base_ = nullptr;

  // 流索引映射: stream id -> memory block index
  // 使用固定大小的数组来避免动态分配
  struct StreamMapping {
    uint64_t stream_id;
    size_t block_index;
    bool active;
  };
  StreamMapping stream_mappings_[kMaxStreamCount];
  size_t stream_mapping_count_ = 0;

  // 保护流索引映射的互斥锁
  mutable std::mutex mutex_;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_BACKEND_RUNTIME_NPU_WORKSPACE_MEMORY_POOL_H_
