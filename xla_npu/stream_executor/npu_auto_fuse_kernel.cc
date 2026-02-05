#include "xla/xla_npu/stream_executor/npu_auto_fuse_kernel.h"

#include <string>

#include "absl/status/status.h"
#include "tsl/platform/path.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_npu/dependency/tool/function_loader.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace stream_executor {

static const std::string get_tiling_size_func_name = "GetTilingDataSize";
static const std::string tiling_func_name = "AutoFuseTiling";
static const std::string launch_func_name = "AutofuseLaunchV2";

absl::Status NpuAutoFuseKernel::Load() {
    // 注册lib
    const std::string& tiling_file_path = tsl::io::JoinPath(load_file_base_dir_, GetTilingFileName());
    const std::string& kernel_file_path = tsl::io::JoinPath(load_file_base_dir_, GetKernelFileName());
    registry[GetTilingFileName()] = std::move(std::make_unique<npu::FunctionLoader>(tiling_file_path, ".so"));
    registry[GetKernelFileName()] = std::move(std::make_unique<npu::FunctionLoader>(kernel_file_path, ".bin"));

    // 注册func
    registry[GetTilingFileName()]->Set(get_tiling_size_func_name);
    registry[GetTilingFileName()]->Set(tiling_func_name);
    registry[GetKernelFileName()]->Set(launch_func_name);

    // 获取func
    GetTilingSizeFunc_ =
        reinterpret_cast<GetTilingSizeFunc>(registry[GetTilingFileName()]->Get(get_tiling_size_func_name));
    CHECK_COND_AND_PRINT_NPU_LOG(GetTilingSizeFunc_ == nullptr, "Get func %s from %s failed", get_tiling_size_func_name,
                                 GetTilingFileName());

    TilingFunc_ = reinterpret_cast<TilingFunc>(registry[GetTilingFileName()]->Get(tiling_func_name));
    CHECK_COND_AND_PRINT_NPU_LOG(TilingFunc_ == nullptr, "Get func %s from %s failed", tiling_func_name,
                                 GetTilingFileName());

    LaunchFunc_ = reinterpret_cast<LaunchFunc>(registry[GetKernelFileName()]->Get(launch_func_name));
    CHECK_COND_AND_PRINT_NPU_LOG(LaunchFunc_ == nullptr, "Get func %s from %s failed", launch_func_name,
                                 GetKernelFileName());

    return absl::OkStatus();
}

absl::Status NpuAutoFuseKernel::Launch(const ThreadDim& thread_dims, const BlockDim& block_dims,
                                       const std::optional<ClusterDim>& cluster_dims, Stream* stream,
                                       const KernelArgs& args) {
    // thread_dims, cluster_dims not use

    uint32_t auto_fuse_ret = 0;

    CHECK_COND_AND_PRINT_NPU_LOG(stream == nullptr, "Stream need to be created or init before execute npu thunk.");
    auto stream_handle = stream->platform_specific_handle();
    CHECK_COND_AND_PRINT_NPU_LOG(stream_handle.stream == nullptr,
                                 "Acl stream need to be created or init before execute npu thunk.");
    auto acl_stream = stream_handle.stream;

    size_t tiling_size = GetTilingSizeFunc_();
    std::vector<uint8_t> tiling_data_holder(tiling_size);
    void* tiling_data = static_cast<void*>(tiling_data_holder.data());

    auto workspace_memory = std::make_unique<DeviceMemoryBase>(executor_->Allocate(tiling_size, /*memory_space=*/0));
    void* workspace = workspace_memory->opaque();
    CHECK_COND_AND_PRINT_NPU_LOG(workspace == nullptr, "Failed to allocate %d bytes for workspace.", tiling_size);

    uint32_t workspace_size = 0;
    uint32_t block_dim = block_dims.x;
    auto_fuse_ret = TilingFunc_(tiling_data, &workspace_size, &block_dim, nullptr);
    CHECK_COND_AND_PRINT_NPU_LOG(auto_fuse_ret != 0, "Tiling failed, error code: %d.", auto_fuse_ret);

    // TODO 参考cuda_kernel,使用KernelLoaderSpec中传递进来的pack函数进行参数封装
    if (auto* args_ptr = DynCast<KernelArgsDeviceMemoryArray>(&args)) {
        int32_t input_param_num = args_ptr->number_of_arguments() - 1;
        int32_t output_param_num = 1;  // 当前写死为1
        void* input_param_addr = const_cast<void*>(args_ptr->device_memory_ptr(0));
        void* output_param_addr =
            const_cast<void*>(args_ptr->device_memory_ptr(args_ptr->number_of_arguments() - 1));  // 默认取最后一个

        auto_fuse_ret = LaunchFunc_(block_dim, acl_stream, &input_param_addr, input_param_num, &output_param_addr,
                                    output_param_num, workspace, tiling_data);
        CHECK_COND_AND_PRINT_NPU_LOG(auto_fuse_ret != 0, "Launch fusion kernel failed, error code: %d.", auto_fuse_ret);
    } else {
        const std::string& err_msg = absl::StrFormat("Get args failed.");
        NPU_LOG_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
    }

    TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());  // TODO 所有算子下发完了再同步

    return absl::OkStatus();
}

}  // namespace stream_executor
