#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/pjrt/transpose.h"
#include "xla/pjrt/utils.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/traceme.h"

#include "raw_buffer.h"

namespace xla {
constexpr size_t kSmallDataTransferByteSize = 102400;  // 100 KiB
void NpuTrackedDeviceEventPromise::Set(
    tsl::RCReference<PjRtDeviceEvent> event) {
  auto cpu_event =
      tensorflow::down_cast<NpuTrackedDeviceEvent*>(event.get())->event();
  av_->ForwardTo(std::move(cpu_event));
}

void NpuTrackedDeviceEventPromise::SetReady() {
  av_->ForwardTo(tsl::MakeAvailableAsyncValueRef<NpuEvent>());
}

Future<> NpuTrackedDeviceEvent::GetReadyFuture() {
  auto [promise, future] = Future<>::MakePromise();
  event_.AndThen([promise = std::move(promise), event = event_]() mutable {
    if (auto* error = event.GetErrorIfPresent()) {
      promise.Set(*error);
    } else {
      promise.Set();
    }
  });

  return FutureHelpers::WithProfiling(
      std::move(future),
      /*on_block_start=*/
      [callee_method = callee_method_, callee_type = callee_type_]() {
        tsl::profiler::TraceMeProducer traceme(
            [&] { return absl::StrCat(callee_type, "::", callee_method); });
        return FutureHelpers::ProfilingKeys({traceme.GetContextId()});
      },
      /*on_block_end=*/
      [callee_method = callee_method_,
       callee_type = callee_type_](FutureHelpers::ProfilingKeys keys) {
        tsl::profiler::TraceMeConsumer traceme(
            [&] { return absl::StrCat(callee_type, "::", callee_method); },
            keys.traceme_context_id);
      });
}

/*static*/ absl::StatusOr<tsl::RCReference<NpuRawBuffer>>
NpuRawBuffer::Allocate(PjRtMemorySpace* memory_space, size_t size_bytes) {
  std::cout << "NpuRawBuffer::Allocate" << std::endl;
  TF_ASSIGN_OR_RETURN(auto memory,
                      NpuDeviceMemory::Allocate(size_bytes));
  return tsl::MakeRef<NpuRawBuffer>(memory_space, std::move(memory));
}

/*static*/ absl::StatusOr<tsl::RCReference<NpuRawBuffer>>
NpuRawBuffer::ImportForeignMemory(
    void* data, absl::AnyInvocable<void() &&> on_delete_callback,
    size_t on_device_bytes_count, PjRtMemorySpace* memory_space) {
  return absl::UnimplementedError("NPU does not support importing foreign memory");
}

size_t NpuRawBuffer::GetOnDeviceSizeInBytes() const {
  return buffer_->size_bytes();
}

void* NpuRawBuffer::GetHostPointer() const { return buffer_->untyped_data(); }

absl::Status NpuRawBuffer::ValidateSlice(int64_t offset, int64_t slice_size) {
  size_t buffer_size = GetOnDeviceSizeInBytes();
  if (offset < 0 || offset > buffer_size || buffer_size - offset < slice_size) {
    return InvalidArgument(
        "Invalid slicing of buffer size %lld with "
        "invalid offset %lld, slice size %lld",
        buffer_size, offset, slice_size);
  }
  return absl::OkStatus();
}

absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>>
NpuRawBuffer::CopyRawHostToDeviceAndReturnEvent(const void* src, int64_t offset,
                                                int64_t transfer_size) {
  return absl::UnimplementedError("NPU does not support CopyRawHostToDeviceAndReturnEvent");
}

absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>>
NpuRawBuffer::CopyRawDeviceToHostAndReturnEvent(void* dst, int64_t offset,
                                                int64_t transfer_size) {
  return absl::UnimplementedError("NPU does not support CopyRawDeviceToHostAndReturnEvent");
}

absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>> NpuRawBuffer::CopyFromLiteral(
    const LiteralSlice& literal, const xla::Layout& layout,
    AsyncWorkRunner* async_work_runner) {
  return absl::UnimplementedError("NPU does not support CopyFromLiteral");
}

absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>>
NpuRawBuffer::CopyFromHostBuffer(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    PjRtClient::HostBufferSemantics host_buffer_semantics,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer, const Shape& shape,
    AsyncWorkRunner* async_work_runner, absl::Mutex* transpose_mu,
    TransposePlanCache* transpose_cache) {
  tsl::AsyncValueRef<NpuDeviceMemory> device_buffer = buffer_;
  bool has_default_layout =
      !byte_strides || HasMajorToMinorLayout(type, dims, *byte_strides);
  const int bit_width = primitive_util::BitWidth(type);
  // Packed arrays are unpacked on host and packed on device.
  bool is_packed = primitive_util::IsSubByteNonPredType(type);

  size_t byte_size = ShapeUtil::ByteSizeOf(shape);
  size_t dst_byte_size = byte_size;
  if (is_packed) {
    byte_size *= 8 / bit_width;
  }
  auto dst_data_ptr = device_buffer->untyped_data();
  if (!has_default_layout || is_packed) {
    // If the input array does not have a major-to-minor layout, transpose it
    // into major-to-minor layout. Currently we choose to always do this
    // synchronously.
    // TODO(phawkins): consider performing the transpose asynchronously.
    // TODO(phawkins): parallelize the transpose.
    std::shared_ptr<TransposePlan> transpose;
    {
      absl::InlinedVector<int64_t, 4> permutation(dims.size());
      absl::c_iota(permutation, 0);
      TransposePlan::Options options;
      options.elem_size_in_bytes = primitive_util::ByteWidth(type);
      options.dims = dims;
      options.permutation = permutation;
      if (byte_strides) {
        options.input_layout = TransposePlan::Striding{*byte_strides};
      }
      absl::MutexLock lock(*transpose_mu);
      TF_ASSIGN_OR_RETURN(transpose, transpose_cache->GetOrCreate(options));
    }
    if (!is_packed) {
      transpose->Execute(data, dst_data_ptr);
    } else {
      // First transpose the unpacked data into a new temporary buffer, then
      // pack the data.
      // TODO(reedwm): Fuse the transpose and packing by having TransposePlan
      // support packing.
      auto data_transposed = std::make_unique<char[]>(byte_size);
      transpose->Execute(data, data_transposed.get());
      absl::Span<const char> src_data_span(data_transposed.get(), byte_size);
      absl::Span<char> dst_data_span(static_cast<char*>(dst_data_ptr),
                                     dst_byte_size);
      PackIntN(bit_width, src_data_span, dst_data_span);
    }
    if (on_done_with_host_buffer) {
      std::move(on_done_with_host_buffer)();
      on_done_with_host_buffer = nullptr;
    }
  } else {
    bool should_sync_copy =
        host_buffer_semantics ==
            PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall ||
        (byte_size < kSmallDataTransferByteSize);
    if (should_sync_copy) {
      std::memcpy(dst_data_ptr, data, byte_size);
      if (on_done_with_host_buffer) {
        std::move(on_done_with_host_buffer)();
        on_done_with_host_buffer = nullptr;
      }
    } else {
      tsl::AsyncValueRef<NpuEvent> copy_event =
          tsl::MakeConstructedAsyncValueRef<NpuEvent>();
      auto result = tsl::MakeRef<NpuTrackedDeviceEvent>(
          copy_event.CopyRef(), "CpuRawBuffer", "CopyFromHostBuffer");
      async_work_runner->Schedule([device_buffer, dst_data_ptr, data, byte_size,
                                   copy_event = std::move(copy_event),
                                   on_done_with_host_buffer = std::move(
                                       on_done_with_host_buffer)]() mutable {
        tsl::profiler::TraceMe traceme("H2D Dispatch");
        std::memcpy(dst_data_ptr, data, byte_size);
        if (on_done_with_host_buffer) {
          std::move(on_done_with_host_buffer)();
          on_done_with_host_buffer = nullptr;
        }
        // Signal copy is complete.
        copy_event.SetStateConcrete();
      });
      return result;
    }
  }
  return tsl::MakeRef<NpuTrackedDeviceEvent>(
      tsl::MakeAvailableAsyncValueRef<NpuEvent>());
}

void NpuRawBuffer::ReadDynamicShape(tsl::AsyncValueRef<xla::Shape> output_shape,
                                    xla::Shape shape) {
  std::cout << "NPU does not support ReadDynamicShape" << std::endl;
}

absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>>
NpuRawBuffer::MakeAllocationReadyEvent() {
  return absl::UnimplementedError( "NPU does not support MakeAllocationReadyEvent");
}

void NpuRawBuffer::CopyToLiteralAsync(
    Promise<> promise, tsl::RCReference<PjRtDeviceEventPromise> device_promise,
    MutableLiteralBase* literal, xla::Shape shape) {
  std::cout << "NPU does not support CopyToLiteralAsync" << std::endl;
}

void NpuRawBuffer::CopyTo(
    tsl::RCReference<CommonPjRtRawBuffer> dst_raw_buffer,
    tsl::RCReference<PjRtDeviceEventPromise> definition_event_promise,
    tsl::RCReference<PjRtDeviceEventPromise> src_usage_event_promise,
    ::tsl::AsyncValueRef<bool> allocation_event) {
  std::cout << "NPU does not support CopyTo" << std::endl;
}

}
