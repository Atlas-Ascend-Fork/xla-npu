#include "xla/xla_npu/stream_executor/npu_stream.h"

#include <string.h>

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/synchronization/notification.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event.h"
// #include "xla/stream_executor/host/host_event.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_common.h"
#include "xla/xla_npu/stream_executor/npu_event.h"

namespace stream_executor {

NpuStream::NpuStream(StreamExecutor* executor) : StreamCommon(executor) {}

NpuStream::~NpuStream() { parent()->DeallocateStream(this); }

absl::Status NpuStream::Memcpy(DeviceMemoryBase* gpu_dst, const DeviceMemoryBase& gpu_src, uint64_t size) {
    // void* dst_mem = gpu_dst->opaque();
    // void* src_mem = const_cast<void*>(gpu_src.opaque());
    // memcpy(dst_mem, src_mem, size);
    // return absl::OkStatus();
    return absl::UnimplementedError("NpuStream::Memcpy d2d is not supported.");
}

absl::Status NpuStream::Memcpy(void* host_dst, const DeviceMemoryBase& gpu_src, uint64_t size) {
    // void* src_mem = const_cast<void*>(gpu_src.opaque());
    // memcpy(host_dst, src_mem, size);
    // return absl::OkStatus();
    return absl::UnimplementedError("NpuStream::Memcpy d2h is not supported.");
}

absl::Status NpuStream::Memcpy(DeviceMemoryBase* gpu_dst, const void* host_src, uint64_t size) {
    // void* dst_mem = gpu_dst->opaque();
    // memcpy(dst_mem, host_src, size);
    // return absl::OkStatus();
    return absl::UnimplementedError("NpuStream::Memcpy h2d is not supported.");
}

absl::Status NpuStream::Memset32(DeviceMemoryBase* location, uint32_t pattern, uint64_t size) {
    // void* gpu_mem = location->opaque();
    // memset(gpu_mem, pattern, size);
    // return absl::OkStatus();
    return absl::UnimplementedError("NpuStream::Memset32 is not supported.");
}

absl::Status NpuStream::MemZero(DeviceMemoryBase* location, uint64_t size) {
    // void* gpu_mem = location->opaque();
    // memset(gpu_mem, 0, size);
    // return absl::OkStatus();
    return absl::UnimplementedError("NpuStream::MemZero is not supported.");
}

absl::Status NpuStream::WaitFor(Stream* other) { 
    // return absl::OkStatus();
    return absl::UnimplementedError("NpuStream::WaitFor other is not supported.");
}

absl::Status NpuStream::WaitFor(Event* event) {
    // std::shared_ptr<absl::Notification> notification = static_cast<NpuEvent*>(event)->notification();
    // notification->WaitForNotification();
    // return absl::OkStatus();
    return absl::UnimplementedError("NpuStream::WaitFor event is not supported.");
}

absl::Status NpuStream::RecordEvent(Event* event) {
    // std::shared_ptr<absl::Notification> notification = static_cast<NpuEvent*>(event)->notification();
    // CHECK(!notification->HasBeenNotified());
    // notification->Notify();
    // return absl::OkStatus();
    return absl::UnimplementedError("NpuStream::RecordEvent is not supported.");
}

absl::Status NpuStream::DoHostCallbackWithStatus(absl::AnyInvocable<absl::Status() &&> callback) {
    // return std::move(callback)();
    return absl::UnimplementedError("NpuStream::DoHostCallbackWithStatus is not supported.");
}

}  // namespace stream_executor
