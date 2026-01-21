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
#include "xla/xla_npu/stream_executor/npu_event.h"
#include "xla/xla_npu/stream_executor/npu_stream.h"
#include "xla/xla_npu/stream_executor/npu_stream_factory.h"

namespace stream_executor {

NpuStream* AsNpuStream(Stream* stream) {
    DCHECK(stream != nullptr);
    return dynamic_cast<NpuStream*>(stream);
}

absl::Status NpuExecutor::Init() {
    thread_pool_ =
        std::make_shared<tsl::thread::ThreadPool>(tsl::Env::Default(), "npu-executor", 1);
    return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Kernel>> NpuExecutor::LoadKernel(const KernelLoaderSpec& spec) {
    return absl::InternalError("No method of loading npu kernel provided");
}

bool NpuExecutor::DeviceMemoryUsage(int64_t* free, int64_t* total) const {
    tsl::port::MemoryInfo mem_info = tsl::port::GetMemoryInfo();
    *free = (mem_info.free != INT64_MAX) ? mem_info.free : -1;
    *total = (mem_info.total != INT64_MAX) ? mem_info.total : -1;
    return true;
}

DeviceMemoryBase NpuExecutor::Allocate(uint64_t size, int64_t memory_space) {
    CHECK_EQ(memory_space, 0);
    // Use a minimum alignment of 64 bytes to be friendly to AVX512 code.
    // This should probably be kept in sync with
    // tsl::Allocator::kAllocatorAlignment.
    return DeviceMemoryBase(tsl::port::AlignedMalloc(size, /*minimum_alignment=*/64), size);
}

void NpuExecutor::Deallocate(DeviceMemoryBase* mem) { tsl::port::AlignedFree(mem->opaque()); }

absl::Status NpuExecutor::SynchronousMemZero(DeviceMemoryBase* location, uint64_t size) {
    memset(location->opaque(), 0, size);
    return absl::OkStatus();
}

absl::Status NpuExecutor::SynchronousMemcpy(DeviceMemoryBase* gpu_dst, const void* host_src, uint64_t size) {
    memcpy(gpu_dst->opaque(), host_src, size);
    return absl::OkStatus();
}

absl::Status NpuExecutor::SynchronousMemcpy(void* host_dst, const DeviceMemoryBase& gpu_src, uint64_t size) {
    memcpy(host_dst, gpu_src.opaque(), size);
    return absl::OkStatus();
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
        return factory->CreateStream(this);
    }
    return std::make_unique<NpuStream>(this);
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

}  // namespace stream_executor
