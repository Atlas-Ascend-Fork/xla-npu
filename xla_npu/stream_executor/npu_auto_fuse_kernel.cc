#include "xla/xla_npu/stream_executor/npu_auto_fuse_kernel.h"

// #include <cstring>
// #include <fstream>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "tsl/platform/path.h"
#include "tsl/platform/status.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/stream_executor.h"
// #include "xla/xla_npu/dependency/interface/acl_interface.h"
#include "xla/xla_npu/dependency/tool/function_loader.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace stream_executor {

static const std::string get_tiling_size_func_name = "GetTilingDataSize";
static const std::string tiling_func_name = "AutofuseTiling";
static const std::string launch_func_name = "AutofuseLaunchV2";

absl::Status NpuAutoFuseKernel::Load() {
    // 注册lib
    const std::string& tiling_file_path = tsl::io::JoinPath(load_file_base_dir_, GetTilingFileName());
    const std::string& kernel_file_path = tsl::io::JoinPath(load_file_base_dir_, GetKernelFileName());
    if (registry.count(GetTilingFileName()) == 0) {
        registry[GetTilingFileName()] = std::make_unique<npu::FunctionLoader>(tiling_file_path, ".so");
    }
    if (registry.count(GetKernelFileName()) == 0) {
        registry[GetKernelFileName()] = std::make_unique<npu::FunctionLoader>(kernel_file_path, ".so");
    }

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

    /* acl kernel launch 执行方式 - 开始; TODO 当前存在问题,算子可以成功执行,但是执行以后输入结果为零,问题待定位 */
    // aclError acl_ret = ACL_SUCCESS;

    // // 从文件加载二进制数据到内存
    // std::ifstream binary_file(kernel_file_path, std::ios::binary | std::ios::ate);
    // CHECK_COND_AND_PRINT_NPU_LOG(!binary_file.is_open(), "Failed to open binary file: %s", kernel_file_path.c_str());

    // std::streamsize file_size = binary_file.tellg();
    // binary_file.seekg(0, std::ios::beg);

    // std::vector<char> binary_data(file_size);
    // CHECK_COND_AND_PRINT_NPU_LOG(!binary_file.read(binary_data.data(), file_size), "Failed to read binary file: %s",
    //                              kernel_file_path.c_str());
    // binary_file.close();

    // // 设置 binary 加载选项：指定 magic 值以识别二进制文件类型
    // // 注意：aclrtBinaryLoadFromData 支持 MAGIC 选项，从文件加载不支持
    // aclrtBinaryLoadOption load_options[1];
    // load_options[0].type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
    // load_options[0].value.magic = ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE;  // 0x41415246 = "BARF"

    // aclrtBinaryLoadOptions options;
    // options.options = load_options;
    // options.numOpt = 1;

    // // 从内存数据加载 binary
    // aclrtBinHandle bin_handle = nullptr;
    // acl_ret = npu::aclrtBinaryLoadFromData(binary_data.data(), file_size, &options, &bin_handle);
    // CHECK_COND_AND_PRINT_NPU_LOG(acl_ret != ACL_SUCCESS, "Load binary from %s failed.", kernel_file_path.c_str());

    // // 输出函数句柄
    // const std::string kernel_name = "main_1";
    // acl_ret = npu::aclrtBinaryGetFunction(bin_handle, kernel_name.c_str(), &func_handle_);
    // CHECK_COND_AND_PRINT_NPU_LOG(acl_ret != ACL_SUCCESS, "Load function %s from %s failed.", kernel_name,
    //                              kernel_file_path.c_str());
    /* acl kernel launch 执行方式 - 结束 */

    return absl::OkStatus();
}

absl::Status NpuAutoFuseKernel::Launch(const ThreadDim& thread_dims, const BlockDim& block_dims,
                                       const std::optional<ClusterDim>& cluster_dims, Stream* stream,
                                       const KernelArgs& args) {
    // thread_dims, cluster_dims not use

    CHECK_COND_AND_PRINT_NPU_LOG(stream == nullptr, "Stream need to be created or init before execute npu thunk.");
    auto stream_handle = stream->platform_specific_handle();
    CHECK_COND_AND_PRINT_NPU_LOG(stream_handle.stream == nullptr,
                                 "Acl stream need to be created or init before execute npu thunk.");
    auto acl_stream = stream_handle.stream;

    size_t tiling_size = GetTilingSizeFunc_();
    std::vector<uint8_t> tiling_data_holder(tiling_size);
    void* tiling_data = static_cast<void*>(tiling_data_holder.data());

    uint32_t workspace_size = 0;
    uint32_t block_dim;
    uint32_t auto_fuse_ret = TilingFunc_(tiling_data, &workspace_size, &block_dim, nullptr);
    CHECK_COND_AND_PRINT_NPU_LOG(auto_fuse_ret != 0, "Tiling failed, error code: %d.", auto_fuse_ret);

    void* workspace = nullptr;
    if (workspace_size > 0) {
        auto workspace_memory =
            std::make_unique<DeviceMemoryBase>(executor_->Allocate(workspace_size, /*memory_space=*/0));
        workspace = workspace_memory->opaque();
        CHECK_COND_AND_PRINT_NPU_LOG(workspace == nullptr, "Failed to allocate %d bytes for workspace.",
                                     workspace_size);
    }

    // TODO 参考cuda_kernel,使用KernelLoaderSpec中传递进来的pack函数进行参数封装
    if (auto* args_ptr = DynCast<KernelArgsDeviceMemoryArray>(&args)) {
        int32_t input_param_num = args_ptr->number_of_arguments() - 1;
        int32_t output_param_num = 1;  // 当前写死为1
        void* input_param_addr = const_cast<void*>(args_ptr->device_memory_ptr(0));
        void* output_param_addr =
            const_cast<void*>(args_ptr->device_memory_ptr(args_ptr->number_of_arguments() - 1));  // 默认取最后一个

        auto_fuse_ret = LaunchFunc_(block_dim, acl_stream, &input_param_addr, input_param_num, &output_param_addr,
                                 output_param_num, workspace, tiling_data);
        CHECK_COND_AND_PRINT_NPU_LOG(auto_fuse_ret != 0, "Launch fusion kernel failed, error code: %d.",
                                     auto_fuse_ret);

        /* acl kernel launch 执行方式 - 开始 */
        // aclError acl_ret = ACL_SUCCESS;
        // aclrtArgsHandle args_handle = nullptr;
        // acl_ret = npu::aclrtKernelArgsInit(func_handle_, &args_handle);
        // CHECK_COND_AND_PRINT_NPU_LOG(acl_ret != ACL_SUCCESS, "Init fusion kernel args failed, error code: %d.",
        //                              acl_ret);

        // const size_t num_args = args_ptr->number_of_arguments();
        // for (size_t i = 0; i < num_args; ++i) {
        //     const void* ptr = args_ptr->device_memory_ptr(i);
        //     aclrtParamHandle input_output_param_handle = nullptr;  // not use
        //     acl_ret =
        //         npu::aclrtKernelArgsAppend(args_handle, (void**)&ptr, sizeof(uintptr_t), &input_output_param_handle);
        //     CHECK_COND_AND_PRINT_NPU_LOG(acl_ret != ACL_SUCCESS,
        //                                  "Add fusion kernel input/output arg failed, error code: %d.", acl_ret);
        // }

        // aclrtParamHandle workspace_param_handle = nullptr;  // not use
        // acl_ret =
        //     npu::aclrtKernelArgsAppend(args_handle, (void**)&workspace, sizeof(uintptr_t), &workspace_param_handle);
        // CHECK_COND_AND_PRINT_NPU_LOG(acl_ret != ACL_SUCCESS, "Add fusion kernel workspace arg failed, error code:
        // %d.",
        //                              acl_ret);

        // aclrtParamHandle tiling_data_param_handle = nullptr;
        // acl_ret = npu::aclrtKernelArgsAppendPlaceHolder(args_handle, &tiling_data_param_handle);
        // CHECK_COND_AND_PRINT_NPU_LOG(acl_ret != ACL_SUCCESS,
        //                              "Add fusion kernel tiling data arg place holder failed, error code: %d.",
        //                              acl_ret);

        // void* tiling_data_param_handle_buffer = nullptr;
        // acl_ret = npu::aclrtKernelArgsGetPlaceHolderBuffer(args_handle, tiling_data_param_handle, tiling_size,
        //                                                    &tiling_data_param_handle_buffer);
        // CHECK_COND_AND_PRINT_NPU_LOG(acl_ret != ACL_SUCCESS,
        //                              "Get fusion kernel tiling data arg place holder failed, error code: %d.",
        //                              acl_ret);

        // acl_ret = npu::aclrtMemcpy(tiling_data_param_handle_buffer, tiling_size, tiling_data, tiling_size,
        //                            ACL_MEMCPY_HOST_TO_HOST);
        // CHECK_COND_AND_PRINT_NPU_LOG(acl_ret != ACL_SUCCESS,
        //                              "Copy fusion kernel tiling data to arg place holder failed, error code: %d.",
        //                              acl_ret);

        // acl_ret = npu::aclrtKernelArgsFinalize(args_handle);
        // CHECK_COND_AND_PRINT_NPU_LOG(acl_ret != ACL_SUCCESS, "Finalize fusion kernel args failed, error code: %d.",
        //                              acl_ret);

        // acl_ret = npu::aclrtLaunchKernel(func_handle_, block_dim, args_data.data(), args_size, acl_stream);
        // acl_ret = npu::aclrtLaunchKernelWithConfig(func_handle_, block_dim, acl_stream, nullptr, args_handle,
        // nullptr); CHECK_COND_AND_PRINT_NPU_LOG(acl_ret != ACL_SUCCESS, "Launch fusion kernel failed, error code:
        // %d.", acl_ret);
        /* acl kernel launch 执行方式 - 结束 */
    } else {
        const std::string& err_msg = absl::StrFormat("Get args failed.");
        NPU_LOG_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
    }

    TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());  // TODO 所有算子下发完了再同步

    return absl::OkStatus();
}

}  // namespace stream_executor
