#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/c/pjrt_c_api_ffi_extension.h"
#include "xla/pjrt/c/pjrt_c_api_ffi_internal.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_internal.h"

#include "pjrt_c_api_npu_internal.h"
#include "xla/xla_npu/pjrt_npu/npu/npu_client.h"

namespace pjrt {
namespace npu_plugin {
PJRT_Error* PJRT_Client_Create(PJRT_Client_Create_Args* args)
{
    std::cout << "NPU PJRT_CLIENT_CREATE" << std::endl;
    int device_id = 1;
    int process_index = 0;
    std::vector<std::unique_ptr<xla::PjRtNpuDevice>> devices;
    devices.push_back(std::make_unique<xla::PjRtNpuDevice>(process_index, device_id));
    auto client = xla::GetPjRtNpuClient(process_index, std::move(devices));
    args->client = CreateWrapperClient(std::move(client.value()));
    return nullptr;
}

PJRT_Error* PJRT_ExecuteContext_Create(PJRT_ExecuteContext_Create_Args* args)
{
    PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
        "PJRT_ExecuteContext_Create_Args", PJRT_ExecuteContext_Create_Args_STRUCT_SIZE, args->struct_size));
    auto execute_context = std::make_unique<xla::ExecuteContext>();
    args->context = pjrt::CreateWrapperExecuteContext(std::move(execute_context));
    return nullptr;
}
PJRT_Error* PJRT_NpuDeviceTopology_Create(PJRT_TopologyDescription_Create_Args* args)
{
    return new PJRT_Error{absl::UnimplementedError("Topology not supported for CPU compilation.")};
}

const PJRT_Api* GetNpuPjrtApi()
{
    static PJRT_Layouts_Extension layouts_extension = pjrt::CreateLayoutsExtension(nullptr);

    static PJRT_MemoryDescriptions_Extension memory_descriptions_extension =
        pjrt::CreateMemoryDescriptionsExtension(&layouts_extension.base);

    static PJRT_FFI_Extension ffi_extension = pjrt::CreateFfiExtension(&memory_descriptions_extension.base);

    static PJRT_PhaseCompile_Extension phase_compile_extension =
        pjrt::CreatePhaseCompileExtension(&ffi_extension.base,
                                          /*get_compiler=*/nullptr,
                                          /*destroy_compiler=*/nullptr);

    static const PJRT_Api pjrt_api = CreatePjrtApi(PJRT_Client_Create, PJRT_ExecuteContext_Create,
                                                   PJRT_NpuDeviceTopology_Create, PJRT_Plugin_Initialize_NoOp,
                                                   &phase_compile_extension.base, PJRT_Plugin_Attributes_Xla);

    return &pjrt_api;
}
}  // namespace npu_plugin
}  // namespace pjrt
