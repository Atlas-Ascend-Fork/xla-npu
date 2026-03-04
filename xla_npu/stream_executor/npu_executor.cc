#include "xla/xla_npu/stream_executor/npu_executor.h"

#include <stdint.h>
#include <string.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
// #include "tsl/platform/cpu_info.h"
#include "tsl/platform/mem.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/generic_memory_allocation.h"
#include "xla/stream_executor/generic_memory_allocator.h"
// #include "xla/stream_executor/host/host_event.h"
// #include "xla/stream_executor/host/host_stream.h"
// #include "xla/stream_executor/host/host_stream_factory.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
// #include "xla/tsl/platform/profile_utils/cpu_utils.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla_npu/common/npu_config.h"
#include "xla/xla_npu/common/npu_operation.h"
#include "xla/xla_npu/dependency/interface/acl_interface.h"
#include "xla/xla_npu/dependency/tool/log.h"
#include "xla/xla_npu/service/kernel/npu_kernel_config.h"
#include "xla/xla_npu/stream_executor/npu_auto_fuse_kernel.h"
#include "xla/xla_npu/stream_executor/npu_event.h"
#include "xla/xla_npu/stream_executor/npu_stream.h"
#include "xla/xla_npu/stream_executor/npu_stream_factory.h"

namespace stream_executor {

NpuStream* AsNpuStream(Stream* stream) {
    DCHECK(stream != nullptr);
    return dynamic_cast<NpuStream*>(stream);
}

// 注意jax调用xla接口获取客户端时是用的新的线程(非主线程),因此主线程使用这个device时,需要重新set device
absl::Status NpuExecutor::Init() {
    // thread_pool_ = std::make_shared<tsl::thread::ThreadPool>(tsl::Env::Default(), "npu-executor", 1);

    TF_RETURN_IF_ERROR(xla::NpuOperation::SetDevice(device_ordinal_));

    return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Kernel>> NpuExecutor::LoadKernel(const KernelLoaderSpec& spec) {
    CHECK_COND_AND_PRINT_NPU_LOG(!spec.has_in_process_symbol(), "In process symbol when load kernel is empty");
    auto npu_kernel_config_ptr = static_cast<xla::NpuKernelConfig*>(spec.in_process_symbol().value().symbol);
    switch (npu_kernel_config_ptr->npu_kernel_type) {
        case xla::NpuKernelConfig::NpuKernelType::FUSION:
            auto kernel = std::make_unique<stream_executor::NpuAutoFuseKernel>(
                npu_kernel_config_ptr->npu_artifact_info.dir, npu_kernel_config_ptr->npu_artifact_info.name, this);
            TF_RETURN_IF_ERROR(kernel->Load());
            return std::move(kernel);
    }

    return absl::InternalError(absl::StrFormat("Kernel type %d does not support launch.",
                                               static_cast<int64_t>(npu_kernel_config_ptr->npu_kernel_type)));
}

bool NpuExecutor::DeviceMemoryUsage(int64_t* free, int64_t* total) const {
    tsl::port::MemoryInfo mem_info = tsl::port::GetMemoryInfo();
    *free = (mem_info.free != INT64_MAX) ? mem_info.free : -1;
    *total = (mem_info.total != INT64_MAX) ? mem_info.total : -1;
    return true;
}

DeviceMemoryBase NpuExecutor::Allocate(uint64_t size, int64_t memory_space) {
    CHECK_EQ(memory_space, 0);

    // 保证当前线程执行过set device
    xla::NpuOperation::SetDevice(device_ordinal_);

    void* base = nullptr;
    aclError ret = npu::aclrtMalloc(&base, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        NPU_LOG_FORMAT_PRINT(ERROR, "Allocate %d byte on npu failed, error code: %d", size, ret);
        return DeviceMemoryBase(nullptr, size);
    }

    // 如果该地址之前被释放过,从集合中移除(地址被复用)
    {
        std::lock_guard<std::mutex> lock(freed_addresses_mutex_);
        freed_addresses_.erase(base);
    }

    NPU_LOG_FORMAT_PRINT(DEBUG, "Npu memory allocated, addr %p, size %d", base, size);
    return DeviceMemoryBase(base, size);
}

void NpuExecutor::Deallocate(DeviceMemoryBase* mem) {
    // 保证当前线程执行过set device
    xla::NpuOperation::SetDevice(device_ordinal_);

    // 检查指针是否为空,避免重复释放
    if (mem->opaque() == nullptr) {
        NPU_LOG_FORMAT_PRINT(DEBUG, "Npu memory free skipped, addr is nullptr");
        return;
    }

    // 检查是否已经释放过
    {
        std::lock_guard<std::mutex> lock(freed_addresses_mutex_);
        if (freed_addresses_.find(mem->opaque()) != freed_addresses_.end()) {
            NPU_LOG_FORMAT_PRINT(WARN, "Npu memory free skipped, addr %p already freed", mem->opaque());
            return;
        }
        // 记录已释放的地址
        freed_addresses_.insert(mem->opaque());
    }

    NPU_LOG_FORMAT_PRINT(DEBUG, "Npu memory free, addr: %p, size: %d", mem->opaque(), mem->size());
    npu::aclrtFree(mem->opaque());
}

absl::Status NpuExecutor::SynchronousMemZero(DeviceMemoryBase* location, uint64_t size) {
    // memset(location->opaque(), 0, size);
    // return absl::OkStatus();
    return absl::UnimplementedError("NpuExecutor::SynchronousMemZero is not supported.");
}

absl::Status NpuExecutor::SynchronousMemcpy(DeviceMemoryBase* gpu_dst, const void* host_src, uint64_t size) {
    // memcpy(gpu_dst->opaque(), host_src, size);
    // return absl::OkStatus();
    return absl::UnimplementedError("NpuExecutor::SynchronousMemcpy h2d is not supported.");
}

absl::Status NpuExecutor::SynchronousMemcpy(void* host_dst, const DeviceMemoryBase& gpu_src, uint64_t size) {
    // memcpy(host_dst, gpu_src.opaque(), size);
    // return absl::OkStatus();
    return absl::UnimplementedError("NpuExecutor::SynchronousMemcpy d2h is not supported.");
}

void NpuExecutor::DeallocateStream(Stream* stream) {}

absl::StatusOr<std::unique_ptr<Event>> NpuExecutor::CreateEvent() { return std::make_unique<NpuEvent>(); }

absl::StatusOr<std::unique_ptr<DeviceDescription>> NpuExecutor::CreateDeviceDescription(int device_ordinal) {
    DeviceDescription desc;

    // desc.set_device_address_bits(64);

    // // TODO(rspringer): How to report a value that's based in reality but that
    // // doesn't result in thrashing or other badness? 4GiB chosen arbitrarily.
    // desc.set_device_memory_size(static_cast<uint64_t>(4) * 1024 * 1024 * 1024);

    // float cycle_counter_frequency = static_cast<float>(tsl::profile_utils::CpuUtils::GetCycleCounterFrequency());
    // desc.set_clock_rate_ghz(cycle_counter_frequency / 1e9);

    desc.set_name("npu");
    desc.set_platform_version("Default Version");

    return std::make_unique<DeviceDescription>(std::move(desc));
}

absl::StatusOr<std::unique_ptr<Stream>> NpuExecutor::CreateStream(
    std::optional<std::variant<StreamPriority, int>> priority) {
    const NpuStreamFactory* factory = NpuStreamFactory::GetFactory();
    if (factory != nullptr) {
        TF_ASSIGN_OR_RETURN(auto stream, factory->CreateStream(this));
        return std::move(stream);
    }
    return absl::UnavailableError(absl::StrCat("Npu stream factory not init"));
}

absl::StatusOr<std::unique_ptr<MemoryAllocator>> NpuExecutor::CreateMemoryAllocator(MemoryType type) {
    if (type == MemoryType::kHost) {
        return std::make_unique<GenericMemoryAllocator>(
            [](uint64_t size) -> absl::StatusOr<std::unique_ptr<MemoryAllocation>> {
                void* ptr = new char[size];
                return std::make_unique<GenericMemoryAllocation>(
                    ptr, size, [](void* location, uint64_t size) { delete[] static_cast<char*>(location); });
            });
    }
    return absl::UnimplementedError(absl::StrFormat("Unsupported memory type %d", type));
}

absl::StatusOr<std::shared_ptr<DeviceMemoryBase>> NpuExecutor::CreateOrShareConstant(
    Stream* stream, absl::Span<const uint8_t> content) {
    std::shared_ptr<DeviceMemoryBase> shared_constant;
    auto new_constant = std::make_unique<DeviceMemoryBase>(Allocate(content.size(), /*memory_space=*/0));
    if (new_constant->opaque() == nullptr) {
        return absl::InternalError(absl::StrFormat("Failed to allocate %d bytes for new constant", content.size()));
    }

    TF_RETURN_IF_ERROR(stream->Memcpy(new_constant.get(), content.data(), content.size()));
    absl::Status status = stream->BlockHostUntilDone();
    if (!status.ok()) {
        Deallocate(new_constant.get());
        status.Update(
            absl::InternalError(absl::StrFormat("Memcpy to device address %p failed", new_constant->opaque())));
        return status;
    }

    shared_constant = std::shared_ptr<DeviceMemoryBase>(new_constant.release(), [this](DeviceMemoryBase* p) {
        Deallocate(p);
        delete p;
    });

    // TODO 常量值存入缓存中
    // absl::MutexLock lock{shared_constants_mu_};
    // // We assume all constants are uniquely identified by this hash. In the
    // // (highly unlikely) event of a hash collision, the program will likely crash
    // // (because the cached constant that will be returned by mistake is unlikely
    // // to have the correct size).
    // absl::uint128 fingerprint =
    //     Fingerprint128(absl::string_view(reinterpret_cast<const char*>(content.data()), content.size()));
    // // Must insert nullptr first to get an iterator to the insertion point.
    // auto insert_result = shared_constants_.insert({fingerprint, std::weak_ptr<DeviceMemoryBase>()});
    // auto it = insert_result.first;
    // bool was_already_in_cache = !insert_result.second;
    // std::shared_ptr<DeviceMemoryBase> shared_constant;

    // if (was_already_in_cache) {
    //     shared_constant = it->second.lock();
    // }

    // if (shared_constant == nullptr) {
    //     // Either the constant wasn't found in the cache, or it was but its
    //     // weak_ptr had expired.
    //     auto new_constant = std::make_unique<DeviceMemoryBase>(Allocate(content.size(), /*memory_space=*/0));
    //     if (new_constant->opaque() == nullptr) {
    //         return absl::InternalError(absl::StrFormat("Failed to allocate %d bytes for new constant",
    //         content.size()));
    //     }

    //     TF_RETURN_IF_ERROR(stream->Memcpy(new_constant.get(), content.data(), content.size()));
    //     absl::Status status = stream->BlockHostUntilDone();
    //     if (!status.ok()) {
    //         Deallocate(new_constant.get());
    //         status.Update(
    //             absl::InternalError(absl::StrFormat("Memcpy to device address %p failed", new_constant->opaque())));
    //         return status;
    //     }

    //     // Capturing 'this' in the custom deleter means this executor must
    //     // outlive all shared uses of this constant.
    //     shared_constant = std::shared_ptr<DeviceMemoryBase>(new_constant.release(), [this](DeviceMemoryBase* p) {
    //         Deallocate(p);
    //         delete p;
    //     });
    //     it->second = std::weak_ptr<DeviceMemoryBase>(shared_constant);
    // }

    return shared_constant;
}

}  // namespace stream_executor
