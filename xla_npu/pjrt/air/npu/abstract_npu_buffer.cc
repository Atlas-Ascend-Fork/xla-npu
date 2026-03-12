/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/alignment.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/pjrt/abstract_tracked_device_buffer.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/common_pjrt_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/transpose.h"
#include "xla/pjrt/utils.h"
#include "xla/primitive_util.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/traceme.h"

#include "xla/xla_npu/pjrt/air/npu/abstract_npu_buffer.h"
#include "xla/xla_npu/pjrt/air/npu/npu_event.h"
#include "xla/xla_npu/pjrt/air/npu/raw_buffer.h"
#include "xla/xla_npu/pjrt/air/npu/tracked_npu_device_buffer.h"
#include "xla/xla_npu/pjrt/air/npu/service/npu_executable.h"

namespace xla {
namespace {

constexpr size_t kSmallDataTransferByteSize = 102400;  // 100 KiB

// Unpacks and copies the packed data at `input` into the literal at the given
// ShapeIndex.
void UnpackIntNToLiteral(PrimitiveType input_element_type,
                         const NpuDeviceMemory& input,
                         MutableLiteralBase* literal,
                         const ShapeIndex& shape_index) {
    absl::Span<const char> input_span{
        static_cast<const char*>(input.untyped_data()), input.size_bytes()};
    size_t output_size = static_cast<size_t>(ShapeUtil::ByteSizeOf(
        ShapeUtil::GetSubshape(literal->shape(), shape_index)));
    absl::Span<char> output_span{
        static_cast<char*>(literal->untyped_data(shape_index)), output_size};
    primitive_util::UnpackIntN(input_element_type, input_span, output_span);
}

}  //  namespace

void PackOrCopy(PrimitiveType element_type, const LiteralSlice& literal,
                void* data, int64_t size) {
    if (primitive_util::IsSubByteNonPredType(element_type)) {
      const int bit_width = primitive_util::BitWidth(element_type);
      absl::Span<const char> src_data_span(
          static_cast<const char*>(literal.untyped_data()), literal.size_bytes());
      absl::Span<char> dst_data_span(static_cast<char*>(data), size);
      PackIntN(bit_width, src_data_span, dst_data_span);
    } else {
      CHECK_EQ(literal.size_bytes(), size);
      std::memcpy(data, literal.untyped_data(), size);
    }
}

/*static*/ bool AbstractNpuBuffer::BufferFromHostBufferSupportsZeroCopy(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides, const Shape& shape) {
    if (byte_strides && !HasMajorToMinorLayout(type, dims, *byte_strides)) {
      return false;
    }
    // Packed arrays are unpacked on host and packed on device.
    if (primitive_util::IsSubByteNonPredType(type)) {
      return false;
    }
    return true;
}

}  // namespace xla
