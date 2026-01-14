#include "aclnnopInterface.h"

#include "functionLoader.h"

namespace npu {

#define LOAD_FUNCTION_OPAPI(funcName) REGISTER_FUNCTION(libopapi, funcName)
#define GET_FUNC_OPAPI(funcName) GET_FUNCTION(libopapi, funcName)

REGISTER_LIBRARY(libopapi)
LOAD_FUNCTION_OPAPI(aclnnExpandGetWorkspaceSize)
LOAD_FUNCTION_OPAPI(aclnnExpand)

static const aclnnStatus loadErrorCode = -1;

aclnnStatus AclnnExpandGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* size, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    typedef aclnnStatus (*aclnnExpandGetWorkspaceSizeFunc)(
        const aclTensor* self, const aclIntArray* size, aclTensor* out,
        uint64_t* workspaceSize, aclOpExecutor** executor);
    static aclnnExpandGetWorkspaceSizeFunc func = nullptr;
    if (func == nullptr) {
        func = (aclnnExpandGetWorkspaceSizeFunc)GET_FUNC_OPAPI(aclnnExpandGetWorkspaceSize);
    }
    if (func != nullptr) {
        return func(self, size, out, workspaceSize, executor);
    }
    return loadErrorCode;
}

aclnnStatus AclnnExpand(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    typedef aclnnStatus (*aclnnExpandFunc)(
        void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
    static aclnnExpandFunc func = nullptr;
    if (func == nullptr) {
        func = (aclnnExpandFunc)GET_FUNC_OPAPI(aclnnExpand);
    }
    if (func != nullptr) {
        return func(workspace, workspaceSize, executor, stream);
    }
    return loadErrorCode;
}

} // namespace npu
