#ifndef BACKEND_NPU_THUNK_PASS_PIPELINE_H_
#define BACKEND_NPU_THUNK_PASS_PIPELINE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
// #include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_npu/backend/runtime/npu_sequential_thunk.h"

namespace xla {

class NpuThunkPassBufferAllocator {
   public:
    virtual ~NpuThunkPassBufferAllocator() = default;

    // Creates a new allocation of the provided size.
    // The ownership of the returned pointer stays with the
    // NpuThunkPassBufferAllocator object.
    virtual absl::StatusOr<BufferAllocation* absl_nonnull> NewEmptyAllocation(int64_t size) = 0;
};

class NpuThunkPassInterface {
   public:
    virtual ~NpuThunkPassInterface() = default;

    virtual absl::StatusOr<bool> Run(NpuSequentialThunk* root_thunk, const DebugOptions& debug_options,
                                     const HloModule* absl_nullable hlo_module,
                                     const se::DeviceDescription& device_info, NpuThunkPassBufferAllocator& allocator) = 0;

    virtual absl::string_view name() const = 0;
};

class NpuThunkPassPipeline : public NpuThunkPassInterface {
   public:
    explicit NpuThunkPassPipeline(absl::string_view name) : name_(name) {}

    void AddPass(std::unique_ptr<NpuThunkPassInterface> pass) { passes_.push_back(std::move(pass)); }

    absl::string_view name() const override { return name_; }

    // Runs all optimization passes on the given thunk sequence.
    // Returns true if any pass changed the thunk tree.
    absl::StatusOr<bool> Run(NpuSequentialThunk* root_thunk, const DebugOptions& debug_options,
                             const HloModule* absl_nullable hlo_module, const se::DeviceDescription& device_info,
                             NpuThunkPassBufferAllocator& allocator) override;

   private:
    std::string name_;
    std::vector<std::unique_ptr<NpuThunkPassInterface>> passes_;
};

}  // namespace xla

#endif  // BACKEND_NPU_THUNK_PASS_PIPELINE_H_
