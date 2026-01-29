#ifndef KERNEL_NPU_AUTOFUSE_KERNEL_H_
#define KERNEL_NPU_AUTOFUSE_KERNEL_H_

#include "absl/status/status.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_npu/third_party/register/functionLoader.h"

namespace xla {

class NpuAutoFuseKernel : public stream_executor::Kernel {
   public:
    explicit NpuAutoFuseKernel(const std::string& load_file_base_dir, stream_executor::StreamExecutor* executor)
        : load_file_base_dir_(load_file_base_dir), executor_(executor) {}

    void set_arity(unsigned arity) { arity_ = arity; }
    unsigned Arity() const override { return arity_; }

    absl::StatusOr<int32_t> GetMaxOccupiedBlocksPerCore(stream_executor::ThreadDim threads,
                                                        size_t dynamic_shared_memory_bytes) const override {
        return 0;
    }

    absl::Status Load();

    std::string GetTilingFileName() const { return ""; }
    std::string GetKernelFileName() const { return ""; }

   private:
    absl::Status Launch(const stream_executor::ThreadDim& thread_dims, const stream_executor::BlockDim& block_dims,
                        const std::optional<stream_executor::ClusterDim>& cluster_dims, stream_executor::Stream* stream,
                        const stream_executor::KernelArgs& args) override;

    stream_executor::StreamExecutor* executor_ = nullptr;
    unsigned arity_ = 0;  // number of formal parameters the kernel takes

    using graphStatus = uint32_t;
    using GetTilingSizeFunc = size_t (*)();
    using TilingFunc = graphStatus (*)(void*, uint32_t*, uint32_t*, void*);
    using LaunchFunc = uint32_t (*)(uint32_t, void*, void**, int32_t, void**, int32_t, void*, void*);

    GetTilingSizeFunc GetTilingSizeFunc_;
    TilingFunc TilingFunc_;
    LaunchFunc LaunchFunc_;

    std::string load_file_base_dir_;  // 编译生成的.o, .bin, .so后缀文件的根路径
    // 由于这里的库生命周期仅伴随单个融合算子,因此不使用functionLoader中的FunctionRegister类单例
    std::unordered_map<std::string, std::unique_ptr<npu::FunctionLoader>> registry;
};

}  // namespace xla

#endif  // KERNEL_NPU_AUTOFUSE_KERNEL_H_
