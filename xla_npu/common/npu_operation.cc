#include "xla/xla_npu/common/npu_operation.h"

#include "xla/xla_npu/common/npu_constant.h"
#include "xla/xla_npu/common/npu_log.h"

namespace xla {

absl::Status NpuOperation::SetDevice(int device_id) {
    aclError ret = npu::aclrtSetDevice(device_id);
    if (ret != ACL_SUCCESS) {
        NpuLog::Print(NpuLog::Level::ERROR, "Set device %d for '%s' failed, error code: %d", device_id,
                      NpuConstant::platformName, ret);
        return absl::InternalError(absl::StrFormat("Set device %d failed.", device_id));
    }

    NpuLog::Print(NpuLog::Level::DEBUG, "Set device %d for '%s' success.", device_id, NpuConstant::platformName);

    return absl::OkStatus();
}

absl::Status NpuOperation::CreateStream(aclrtStream* stream) {
    aclError ret = npu::aclrtCreateStream(stream);
    if (ret != ACL_SUCCESS) {
        NpuLog::Print(NpuLog::Level::ERROR, "Create stream for '%s' failed. Error code: %d", NpuConstant::platformName,
                      ret);
        return absl::InternalError(absl::StrFormat("Create stream failed."));
    }

    NpuLog::Print(NpuLog::Level::DEBUG, "Create stream for '%s' success.", NpuConstant::platformName);

    return absl::OkStatus();
}

absl::Status NpuOperation::Malloc(void** new_device_addr, size_t size_bytes) {
    aclError ret = npu::aclrtMalloc(new_device_addr, size_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        NpuLog::Print(NpuLog::Level::ERROR, "Allocate %d byte on npu failed, error code: %d", size_bytes, ret);
        return absl::InternalError(absl::StrFormat("Allocate %d byte on npu failed, error code: %d", size_bytes, ret));
    }

    NpuLog::Print(NpuLog::Level::DEBUG, "Npu memory allocated, addr %p, size %d", *new_device_addr, size_bytes);

    return absl::OkStatus();
}

absl::Status NpuOperation::Memcpy(void* dst, void* src, size_t size_bytes, MemcpyType memcpyType) {
    aclrtMemcpyKind memcpyKind = aclrtMemcpyKind::ACL_MEMCPY_DEFAULT;
    std::string descStr = "";

    switch (memcpyType) {
        case MemcpyType::H2D:
            memcpyKind = ACL_MEMCPY_HOST_TO_DEVICE;
            descStr = "Copy date from cpu to npu";
            break;
        case MemcpyType::D2H:
            memcpyKind = ACL_MEMCPY_DEVICE_TO_HOST;
            descStr = "Copy date from npu to cpu";
            break;
    };

    aclError ret = npu::aclrtMemcpy(dst, size_bytes, src, size_bytes, memcpyKind);
    if (ret != ACL_SUCCESS) {
        NpuLog::Print(NpuLog::Level::ERROR, "%s failed, dst addr %p, src addr %p, size bytes %d, error code: %d",
                      descStr, dst, src, size_bytes, ret);
        return absl::InternalError("Acl memory copy failed.");
    }

    NpuLog::Print(NpuLog::Level::DEBUG, "%s success, dst addr %p, src addr %p, size bytes %d", descStr, dst, src,
                  size_bytes);

    return absl::OkStatus();
}

}  // namespace xla