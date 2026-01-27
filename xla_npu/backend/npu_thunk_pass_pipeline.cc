#include "xla/xla_npu/backend/npu_thunk_pass_pipeline.h"

#include <memory>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
// #include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_npu/backend/npu_sequential_thunk.h"

namespace xla {
namespace gpu {

absl::StatusOr<bool> NpuThunkPassPipeline::Run(NpuSequentialThunk* root_thunk, const DebugOptions& debug_options,
                                            const HloModule* absl_nullable hlo_module,
                                            const se::DeviceDescription& device_info,
                                            NpuThunkPassBufferAllocator& allocator) {
    bool changed = false;
    for (const auto& pass : passes_) {
        VLOG(1) << "Running ThunkPass: " << pass->name();
        TF_ASSIGN_OR_RETURN(bool pass_changed,
                            pass->Run(root_thunk, debug_options, hlo_module, device_info, allocator));
        changed |= pass_changed;
    }
    return changed;
}

}  // namespace gpu
}  // namespace xla
